"""
Hyperparameter Tuning Script for ResUNet
Trains ResUNet for 20 epochs with different parameter combinations
"""

import os
import sys
import json
import logging
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import paths, train_cfg, model_cfg
from src.dataset import create_datasets, create_loaders
from src.models import create_model
from src.train import DiceBCELoss, DiceLoss, dice_score_per_class, validate


def create_resunet_model(init_filters: int, blocks_down: Tuple[int, ...], blocks_up: Tuple[int, ...]) -> torch.nn.Module:
	"""Create ResUNet (SegResNet) with specified hyperparameters"""
	from monai.networks.nets import SegResNet
	return SegResNet(
		spatial_dims=3,
		in_channels=model_cfg.in_channels,
		out_channels=model_cfg.out_channels,
		init_filters=init_filters,
		blocks_down=blocks_down,
		blocks_up=blocks_up,
	)


def train_single_config(
	config: Dict[str, Any],
	run_id: str,
	max_epochs: int = 20,
	device: torch.device = None
) -> Dict[str, Any]:
	"""Train a single hyperparameter configuration"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Setup logging for this run
	log_dir = os.path.join(paths.logs, "hyperparameter_tuning", "resunet", run_id)
	os.makedirs(log_dir, exist_ok=True)
	log_file = os.path.join(log_dir, f"run_{run_id}.log")
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
	)
	logger = logging.getLogger(__name__)
	
	logger.info(f"Starting hyperparameter tuning run: {run_id}")
	logger.info(f"Configuration: {json.dumps(config, indent=2)}")
	
	# Create datasets
	cache_rate = 0.0 if max_epochs <= 1 else 0.05
	train_ds, val_ds = create_datasets(spatial_dims=3, cache_rate=cache_rate)
	train_loader, val_loader = create_loaders(train_ds, val_ds, spatial_dims=3)
	logger.info(f"Dataset loaded - Train: {len(train_ds)}, Val: {len(val_ds)}")
	
	# Create model with hyperparameters
	model = create_resunet_model(
		init_filters=config["init_filters"],
		blocks_down=config["blocks_down"],
		blocks_up=config["blocks_up"]
	).to(device)
	
	total_params = sum(p.numel() for p in model.parameters())
	logger.info(f"Model params: {total_params:,}")
	
	# Setup optimizer and loss with hyperparameters
	optimizer = AdamW(
		model.parameters(),
		lr=config["lr"],
		weight_decay=config["weight_decay"]
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=max_epochs, eta_min=1e-6
	)
	scaler = GradScaler("cuda", enabled=(train_cfg.amp and torch.cuda.is_available()))
	loss_fn = DiceBCELoss(
		weight_dice=config["weight_dice"],
		weight_bce=config["weight_bce"]
	).to(device)
	
	# Training loop
	best_dice = -1.0
	best_epoch = 0
	training_history = {
		"train_loss": [],
		"val_loss": [],
		"dice": [],
		"dice_per_class": [],
		"dice_wt": [],
		"dice_tc": [],
		"dice_et": [],
		"learning_rate": [],
		"config": config
	}
	class_titles = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
	
	for epoch in range(max_epochs):
		model.train()
		epoch_loss = 0.0
		num_batches = 0
		
		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [{run_id}]", leave=False):
			inputs = torch.cat([batch["t1"], batch["t1ce"], batch["t2"], batch["flair"]], dim=1).to(device, non_blocking=True)
			labels = batch.get("label")
			if labels is None:
				continue
			labels = labels.to(device, non_blocking=True)
			
			optimizer.zero_grad(set_to_none=True)
			with autocast(device_type="cuda", enabled=(train_cfg.amp and torch.cuda.is_available())):
				outputs = model(inputs)
				loss = loss_fn(outputs, labels)
			
			if torch.isnan(loss) or torch.isinf(loss):
				logger.warning(f"Invalid loss at epoch {epoch+1}, batch {num_batches}")
				continue
			
			scaler.scale(loss).backward()
			
			if train_cfg.clip_grad and train_cfg.clip_grad > 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
			
			scaler.step(optimizer)
			scaler.update()
			epoch_loss += loss.item()
			num_batches += 1
		
		if num_batches == 0:
			continue
		
		# Validation
		mean_dice, val_loss, per_class_dice = validate(model, val_loader, loss_fn, device, spatial_dims=3)
		avg_train_loss = epoch_loss / num_batches
		
		training_history["train_loss"].append(avg_train_loss)
		training_history["val_loss"].append(val_loss)
		training_history["dice"].append(mean_dice)
		training_history["dice_per_class"].append(per_class_dice)
		if len(per_class_dice) >= 3:
			training_history["dice_wt"].append(per_class_dice[0] if per_class_dice[0] is not None else 0.0)
			training_history["dice_tc"].append(per_class_dice[1] if per_class_dice[1] is not None else 0.0)
			training_history["dice_et"].append(per_class_dice[2] if per_class_dice[2] is not None else 0.0)
		else:
			training_history["dice_wt"].append(0.0)
			training_history["dice_tc"].append(0.0)
			training_history["dice_et"].append(0.0)
		
		current_lr = optimizer.param_groups[0]['lr']
		training_history["learning_rate"].append(current_lr)
		
		scheduler.step()
		
		# Check for best model
		if mean_dice > best_dice:
			best_dice = mean_dice
			best_epoch = epoch + 1
			# Save best model
			model_path = os.path.join(log_dir, f"best_model_{run_id}.pth")
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"best_dice": best_dice,
				"config": config
			}, model_path)
		
		# Log progress
		dice_str = ", ".join([
			f"{title.split()[0]}:{val:.4f}" if val is not None else f"{title.split()[0]}:None"
			for title, val in zip(class_titles, per_class_dice)
		])
		logger.info(f"Epoch {epoch+1}/{max_epochs} | "
		           f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
		           f"dice={mean_dice:.4f} [{dice_str}]")
	
	# Save training history
	history_path = os.path.join(log_dir, f"history_{run_id}.json")
	with open(history_path, 'w') as f:
		# Convert numpy types to native Python for JSON
		history_for_json = {}
		for key, value in training_history.items():
			if key == "config":
				history_for_json[key] = value
			elif isinstance(value, list):
				history_for_json[key] = [float(v) if isinstance(v, (np.number, float)) else (v if v is not None else None) for v in value]
			else:
				history_for_json[key] = value
		json.dump(history_for_json, f, indent=2)
	
	result = {
		"run_id": run_id,
		"config": config,
		"best_dice": float(best_dice),
		"best_epoch": best_epoch,
		"final_dice": float(training_history["dice"][-1]) if training_history["dice"] else 0.0,
		"final_train_loss": float(training_history["train_loss"][-1]) if training_history["train_loss"] else float('inf'),
		"final_val_loss": float(training_history["val_loss"][-1]) if training_history["val_loss"] else float('inf'),
		"num_epochs_trained": len(training_history["dice"]),
	}
	
	logger.info(f"Run {run_id} completed. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
	
	return result, training_history


def generate_hyperparameter_configs() -> List[Dict[str, Any]]:
	"""Generate all hyperparameter combinations for grid search"""
	configs = []
	
	# Hyperparameter ranges
	lr_options = [1e-4, 5e-4, 1e-3]
	weight_decay_options = [1e-5, 1e-4, 1e-3]
	init_filters_options = [8, 16, 32]
	blocks_down_options = [
		(1, 2, 2, 4),  # Default
		(1, 1, 2, 4),  # Fewer blocks
		(1, 2, 3, 4),  # More blocks
	]
	blocks_up_options = [
		(1, 1, 1),     # Default
		(1, 2, 1),     # More blocks
		(2, 1, 1),     # Different distribution
	]
	loss_weight_options = [
		(0.3, 0.7),   # More BCE
		(0.5, 0.5),   # Balanced
		(0.7, 0.3),   # More Dice
	]
	
	# Reduced grid search (sample key combinations to avoid too many runs)
	# Focus on most impactful hyperparameters
	key_combinations = [
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},  # Baseline
		{"lr": 5e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-4, "init_filters": 16, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 32, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 8, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 1, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 2, 3, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 2, 1), "loss_weight": (0.5, 0.5)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.7, 0.3)},
		{"lr": 1e-4, "weight_decay": 1e-5, "init_filters": 16, "blocks_down": (1, 2, 2, 4), "blocks_up": (1, 1, 1), "loss_weight": (0.3, 0.7)},
	]
	
	for combo in key_combinations:
		config = {
			"lr": combo["lr"],
			"weight_decay": combo["weight_decay"],
			"init_filters": combo["init_filters"],
			"blocks_down": combo["blocks_down"],
			"blocks_up": combo["blocks_up"],
			"weight_dice": combo["loss_weight"][0],
			"weight_bce": combo["loss_weight"][1],
		}
		configs.append(config)
	
	return configs


def create_visualizations(results: List[Dict], all_histories: Dict[str, Any], output_dir: str):
	"""Create comprehensive visualization plots for ResUNet"""
	sns.set_style("whitegrid")
	plt.rcParams['figure.figsize'] = (12, 8)
	
	# 1. Best Dice Comparison
	fig, ax = plt.subplots(figsize=(14, 8))
	run_ids = [r["run_id"] for r in results if "error" not in r]
	best_dices = [r["best_dice"] for r in results if "error" not in r]
	colors = plt.cm.viridis(np.linspace(0, 1, len(best_dices)))
	
	bars = ax.barh(range(len(run_ids)), best_dices, color=colors)
	ax.set_yticks(range(len(run_ids)))
	ax.set_yticklabels([rid.split("_")[-1] for rid in run_ids], fontsize=8)
	ax.set_xlabel("Best Dice Score", fontsize=12)
	ax.set_title("Best Dice Score by Configuration - ResUNet", fontsize=14, fontweight='bold')
	ax.grid(axis='x', alpha=0.3)
	
	# Add value labels on bars
	for i, (bar, dice) in enumerate(zip(bars, best_dices)):
		ax.text(dice + 0.01, i, f"{dice:.4f}", va='center', fontsize=8)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "1_best_dice_comparison.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	# 2. Training Curves - Best 5 configurations
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	fig.suptitle("Training Curves - Top 5 Configurations - ResUNet", fontsize=16, fontweight='bold')
	
	sorted_results = sorted([r for r in results if "error" not in r], key=lambda x: x["best_dice"], reverse=True)[:5]
	
	for idx, result in enumerate(sorted_results):
		run_id = result["run_id"]
		if run_id not in all_histories:
			continue
		
		history = all_histories[run_id]
		epochs = range(1, len(history["dice"]) + 1)
		
		row = idx // 3
		col = idx % 3
		ax = axes[row, col]
		
		ax2 = ax.twinx()
		
		# Loss curves
		line1 = ax.plot(epochs, history["train_loss"], 'b-', label='Train Loss', alpha=0.7)
		line2 = ax.plot(epochs, history["val_loss"], 'r-', label='Val Loss', alpha=0.7)
		
		# Dice curve
		line3 = ax2.plot(epochs, history["dice"], 'g-', label='Dice Score', alpha=0.7, linewidth=2)
		
		ax.set_xlabel("Epoch", fontsize=10)
		ax.set_ylabel("Loss", fontsize=10, color='black')
		ax2.set_ylabel("Dice Score", fontsize=10, color='green')
		ax.tick_params(axis='y', labelcolor='black')
		ax2.tick_params(axis='y', labelcolor='green')
		
		config_str = f"LR:{result['config']['lr']:.0e}, IF:{result['config']['init_filters']}, BD:{result['config']['blocks_down']}"
		ax.set_title(f"Config {idx+1}: {config_str}\nBest Dice: {result['best_dice']:.4f}", fontsize=9)
		ax.grid(True, alpha=0.3)
		
		# Combine legends
		lines = line1 + line2 + line3
		labels = [l.get_label() for l in lines]
		ax.legend(lines, labels, loc='best', fontsize=8)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "2_training_curves_top5.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	# 3. Per-Class Dice Comparison
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle("Per-Class Dice Scores Comparison - ResUNet", fontsize=16, fontweight='bold')
	
	valid_results = [r for r in results if "error" not in r]
	
	for idx, class_name in enumerate(["WT", "TC", "ET"]):
		ax = axes[idx]
		run_labels = [r["run_id"].split("_")[-1] for r in valid_results]
		
		# Get final dice per class for each run
		dice_values = []
		for result in valid_results:
			run_id = result["run_id"]
			if run_id in all_histories:
				history = all_histories[run_id]
				if class_name == "WT" and history["dice_wt"]:
					dice_values.append(history["dice_wt"][-1])
				elif class_name == "TC" and history["dice_tc"]:
					dice_values.append(history["dice_tc"][-1])
				elif class_name == "ET" and history["dice_et"]:
					dice_values.append(history["dice_et"][-1])
				else:
					dice_values.append(0.0)
			else:
				dice_values.append(0.0)
		
		colors = plt.cm.plasma(np.linspace(0, 1, len(dice_values)))
		bars = ax.bar(range(len(run_labels)), dice_values, color=colors)
		ax.set_xticks(range(len(run_labels)))
		ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=7)
		ax.set_ylabel(f"{class_name} Dice Score", fontsize=11)
		ax.set_title(f"Whole Tumor" if class_name == "WT" else ("Tumor Core" if class_name == "TC" else "Enhancing Tumor"), fontsize=12)
		ax.grid(axis='y', alpha=0.3)
		
		# Add value labels
		for bar, val in zip(bars, dice_values):
			ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", 
			        ha='center', va='bottom', fontsize=7, rotation=90)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "3_per_class_dice_comparison.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	# 4. Hyperparameter Impact Analysis
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	fig.suptitle("Hyperparameter Impact on Best Dice Score - ResUNet", fontsize=16, fontweight='bold')
	
	valid_results = [r for r in results if "error" not in r]
	
	# Learning Rate
	ax = axes[0, 0]
	lr_values = [r["config"]["lr"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(lr_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Learning Rate", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.set_xscale('log')
	ax.grid(True, alpha=0.3)
	ax.set_title("Learning Rate Impact", fontsize=12)
	
	# Weight Decay
	ax = axes[0, 1]
	wd_values = [r["config"]["weight_decay"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(wd_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Weight Decay", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.set_xscale('log')
	ax.grid(True, alpha=0.3)
	ax.set_title("Weight Decay Impact", fontsize=12)
	
	# Initial Filters
	ax = axes[0, 2]
	if_values = [r["config"]["init_filters"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(if_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Initial Filters", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Initial Filters Impact", fontsize=12)
	
	# Blocks Down (total)
	ax = axes[1, 0]
	bd_values = [sum(r["config"]["blocks_down"]) for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(bd_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Total Blocks Down", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Blocks Down Impact", fontsize=12)
	
	# Blocks Up (total)
	ax = axes[1, 1]
	bu_values = [sum(r["config"]["blocks_up"]) for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(bu_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Total Blocks Up", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Blocks Up Impact", fontsize=12)
	
	# Dice Weight
	ax = axes[1, 2]
	dice_weight_values = [r["config"]["weight_dice"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(dice_weight_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Dice Loss Weight", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Dice/BCE Weight Ratio", fontsize=12)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "4_hyperparameter_impact.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	# 5. Convergence Speed Comparison
	fig, ax = plt.subplots(figsize=(14, 8))
	
	for result in sorted_results[:10]:
		run_id = result["run_id"]
		if run_id not in all_histories:
			continue
		history = all_histories[run_id]
		epochs = range(1, len(history["dice"]) + 1)
		
		config_label = f"IF:{result['config']['init_filters']}, Dice:{result['best_dice']:.3f}"
		ax.plot(epochs, history["dice"], label=config_label, alpha=0.7, linewidth=2)
	
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Dice Score", fontsize=12)
	ax.set_title("Convergence Speed - Top 10 Configurations - ResUNet", fontsize=14, fontweight='bold')
	ax.legend(loc='lower right', fontsize=8, ncol=2)
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "5_convergence_comparison.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"\nVisualizations saved to {output_dir}")


def main():
	"""Main hyperparameter tuning function"""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	tuning_dir = os.path.join(paths.logs, "hyperparameter_tuning", "resunet")
	os.makedirs(tuning_dir, exist_ok=True)
	
	# Generate configurations
	configs = generate_hyperparameter_configs()
	print(f"Generated {len(configs)} hyperparameter configurations")
	print(f"Each will be trained for 20 epochs")
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	results = []
	all_histories = {}
	
	for idx, config in enumerate(configs, start=1):
		run_id = f"resunet_hp_{idx:02d}_{timestamp}"
		print(f"\n{'='*60}")
		print(f"Run {idx}/{len(configs)}: {run_id}")
		print(f"{'='*60}")
		
		try:
			result, history = train_single_config(
				config=config,
				run_id=run_id,
				max_epochs=20,
				device=device
			)
			results.append(result)
			all_histories[run_id] = history
		except Exception as e:
			print(f"Error in run {run_id}: {e}")
			import traceback
			traceback.print_exc()
			result = {
				"run_id": run_id,
				"config": config,
				"error": str(e)
			}
			results.append(result)
	
	# Save all results
	results_path = os.path.join(tuning_dir, f"all_results_{timestamp}.json")
	with open(results_path, 'w') as f:
		json.dump(results, f, indent=2)
	
	# Create visualizations
	print(f"\n{'='*60}")
	print("Generating visualizations...")
	print(f"{'='*60}")
	create_visualizations(results, all_histories, tuning_dir)
	
	# Find best configuration
	valid_results = [r for r in results if "best_dice" in r and "error" not in r]
	if valid_results:
		best_result = max(valid_results, key=lambda x: x["best_dice"])
		print(f"\n{'='*60}")
		print(f"HYPERPARAMETER TUNING COMPLETE")
		print(f"{'='*60}")
		print(f"Best configuration:")
		print(f"  Run ID: {best_result['run_id']}")
		print(f"  Best Dice: {best_result['best_dice']:.4f}")
		print(f"  Best Epoch: {best_result['best_epoch']}")
		print(f"  Configuration:")
		for key, value in best_result['config'].items():
			print(f"    {key}: {value}")
		print(f"\nResults saved to: {results_path}")
		print(f"Visualizations saved to: {tuning_dir}")
		
		# Save summary
		summary_path = os.path.join(tuning_dir, f"summary_{timestamp}.txt")
		with open(summary_path, 'w') as f:
			f.write("ResUNet Hyperparameter Tuning Summary\n")
			f.write("="*60 + "\n\n")
			f.write(f"Best Configuration:\n")
			f.write(f"  Run ID: {best_result['run_id']}\n")
			f.write(f"  Best Dice: {best_result['best_dice']:.4f}\n")
			f.write(f"  Best Epoch: {best_result['best_epoch']}\n")
			f.write(f"\nConfiguration:\n")
			for key, value in best_result['config'].items():
				f.write(f"  {key}: {value}\n")
			f.write(f"\n\nAll Results (sorted by Dice):\n")
			f.write("-"*60 + "\n")
			sorted_results = sorted(valid_results, key=lambda x: x["best_dice"], reverse=True)
			for r in sorted_results:
				f.write(f"Run: {r['run_id']} | Dice: {r['best_dice']:.4f} | Epoch: {r['best_epoch']}\n")
				f.write(f"  Config: {json.dumps(r['config'], indent=4)}\n\n")


if __name__ == "__main__":
	main()

