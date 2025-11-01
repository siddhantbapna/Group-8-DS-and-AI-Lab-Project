"""
Hyperparameter Tuning Script for AttentionUNet
Trains AttentionUNet for 40 epochs with different parameter combinations
Includes visualization and comparison graphs
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
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config.config import paths, train_cfg, model_cfg
from src.dataset import create_datasets, create_loaders
from src.train import DiceBCELoss, DiceLoss, dice_score_per_class, validate


def create_attenunet_model(channels: Tuple[int, ...]) -> torch.nn.Module:
	"""Create AttentionUNet with specified channel configuration"""
	from monai.networks.nets import AttentionUnet
	return AttentionUnet(
		spatial_dims=3,
		in_channels=model_cfg.in_channels,
		out_channels=model_cfg.out_channels,
		channels=channels,
		strides=(2, 2, 2, 2),
	)


def train_single_config(
	config: Dict[str, Any],
	run_id: str,
	max_epochs: int = 40,
	device: torch.device = None
) -> Dict[str, Any]:
	"""Train a single hyperparameter configuration"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Setup logging for this run
	log_dir = os.path.join(paths.logs, "hyperparameter_tuning", "attenunet", run_id)
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
	
	# Create loaders with config batch size
	batch_size = config.get("batch_size", 2)
	auto_workers = max(4, (os.cpu_count() or 8) // 2)
	num_workers = train_cfg.num_workers if train_cfg.num_workers and train_cfg.num_workers > 0 else auto_workers
	
	train_loader = torch.utils.data.DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=(num_workers > 0),
		prefetch_factor=4,
	)
	val_loader = torch.utils.data.DataLoader(
		val_ds,
		batch_size=1,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=(num_workers > 0),
		prefetch_factor=2,
	)
	
	logger.info(f"Dataset loaded - Train: {len(train_ds)}, Val: {len(val_ds)}")
	
	# Create model with hyperparameters
	model = create_attenunet_model(config["channels"]).to(device)
	
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
	patience_counter = 0
	patience = config.get("patience", 28)
	
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
			
			# Gradient clipping
			if config.get("clip_grad", 0) > 0:
				scaler.unscale_(optimizer)
				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
				if grad_norm > config["clip_grad"] * 2:
					logger.warning(f"Large gradients detected, norm: {grad_norm:.2f}")
			
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
			patience_counter = 0
			# Save best model
			model_path = os.path.join(log_dir, f"best_model_{run_id}.pth")
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"best_dice": best_dice,
				"config": config
			}, model_path)
		else:
			patience_counter += 1
		
		# Early stopping
		if patience_counter >= patience:
			logger.info(f"Early stopping at epoch {epoch+1}")
			break
		
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
				history_for_json[key] = [float(v) if isinstance(v, (np.number, float)) else v for v in value]
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
	
	# Base configuration from user requirements
	base_config = {
		"lr": 1e-4,
		"batch_size": 2,
		"patience": 28,
		"clip_grad": 1.0,
		"weight_decay": 1e-5,
		"channels": (16, 32, 64, 128),
		"weight_dice": 0.5,
		"weight_bce": 0.5,
	}
	
	# Hyperparameter variations to test
	lr_options = [5e-5, 1e-4, 5e-4]
	batch_size_options = [1, 2]
	weight_decay_options = [1e-5, 1e-4]
	channels_options = [
		(16, 32, 64, 128),  # Base
		(8, 16, 32, 64),    # Smaller
		(32, 64, 128, 256), # Larger (may OOM)
	]
	loss_weight_options = [
		(0.3, 0.7),  # More BCE
		(0.5, 0.5),  # Balanced
		(0.7, 0.3),  # More Dice
	]
	clip_grad_options = [0.5, 1.0, 2.0]
	
	# Key combinations focusing on most impactful parameters
	key_combinations = [
		# Baseline
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
		# Learning rate variations
		{**base_config, "lr": 5e-5, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
		{**base_config, "lr": 5e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
		# Weight decay variations
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-4, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
		# Channel size variations
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (8, 16, 32, 64), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
		{**base_config, "lr": 1e-4, "batch_size": 1, "weight_decay": 1e-5, "channels": (32, 64, 128, 256), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
		# Loss weight variations
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.7, "weight_bce": 0.3, "clip_grad": 1.0},
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.3, "weight_bce": 0.7, "clip_grad": 1.0},
		# Gradient clipping variations
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 0.5},
		{**base_config, "lr": 1e-4, "batch_size": 2, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 2.0},
		# Batch size variation
		{**base_config, "lr": 1e-4, "batch_size": 1, "weight_decay": 1e-5, "channels": (16, 32, 64, 128), "weight_dice": 0.5, "weight_bce": 0.5, "clip_grad": 1.0},
	]
	
	for combo in key_combinations:
		config = {
			"lr": combo["lr"],
			"batch_size": combo["batch_size"],
			"weight_decay": combo["weight_decay"],
			"channels": combo["channels"],
			"weight_dice": combo["weight_dice"],
			"weight_bce": combo["weight_bce"],
			"clip_grad": combo["clip_grad"],
			"patience": combo["patience"],
		}
		configs.append(config)
	
	return configs


def create_visualizations(results: List[Dict], all_histories: Dict[str, Any], output_dir: str):
	"""Create comprehensive visualization plots"""
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
	ax.set_title("Best Dice Score by Configuration", fontsize=14, fontweight='bold')
	ax.grid(axis='x', alpha=0.3)
	
	# Add value labels on bars
	for i, (bar, dice) in enumerate(zip(bars, best_dices)):
		ax.text(dice + 0.01, i, f"{dice:.4f}", va='center', fontsize=8)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "1_best_dice_comparison.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	# 2. Training Curves - Best 5 configurations
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	fig.suptitle("Training Curves - Top 5 Configurations", fontsize=16, fontweight='bold')
	
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
		
		config_str = f"LR:{result['config']['lr']:.0e}, BS:{result['config']['batch_size']}, CG:{result['config']['clip_grad']}"
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
	fig.suptitle("Per-Class Dice Scores Comparison", fontsize=16, fontweight='bold')
	
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
	fig.suptitle("Hyperparameter Impact on Best Dice Score", fontsize=16, fontweight='bold')
	
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
	
	# Gradient Clipping
	ax = axes[0, 2]
	clip_values = [r["config"]["clip_grad"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(clip_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Gradient Clipping", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Gradient Clipping Impact", fontsize=12)
	
	# Batch Size
	ax = axes[1, 0]
	bs_values = [r["config"]["batch_size"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(bs_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Batch Size", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Batch Size Impact", fontsize=12)
	
	# Dice Weight
	ax = axes[1, 1]
	dice_weight_values = [r["config"]["weight_dice"] for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(dice_weight_values, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Dice Loss Weight", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Dice/BCE Weight Ratio", fontsize=12)
	
	# Channels (model size)
	ax = axes[1, 2]
	channel_sizes = [sum(r["config"]["channels"]) for r in valid_results]
	dice_values = [r["best_dice"] for r in valid_results]
	ax.scatter(channel_sizes, dice_values, s=100, alpha=0.6, c=dice_values, cmap='viridis')
	ax.set_xlabel("Total Channel Size", fontsize=11)
	ax.set_ylabel("Best Dice Score", fontsize=11)
	ax.grid(True, alpha=0.3)
	ax.set_title("Model Size Impact", fontsize=12)
	
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
		
		config_label = f"LR:{result['config']['lr']:.0e}, Dice:{result['best_dice']:.3f}"
		ax.plot(epochs, history["dice"], label=config_label, alpha=0.7, linewidth=2)
	
	ax.set_xlabel("Epoch", fontsize=12)
	ax.set_ylabel("Dice Score", fontsize=12)
	ax.set_title("Convergence Speed - Top 10 Configurations", fontsize=14, fontweight='bold')
	ax.legend(loc='lower right', fontsize=8, ncol=2)
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "5_convergence_comparison.png"), dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"\nVisualizations saved to {output_dir}")


def main():
	"""Main hyperparameter tuning function"""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	tuning_dir = os.path.join(paths.logs, "hyperparameter_tuning", "attenunet")
	os.makedirs(tuning_dir, exist_ok=True)
	
	# Generate configurations
	configs = generate_hyperparameter_configs()
	print(f"Generated {len(configs)} hyperparameter configurations")
	print(f"Each will be trained for 40 epochs with patience=28")
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	results = []
	all_histories = {}
	
	for idx, config in enumerate(configs, start=1):
		run_id = f"attenunet_hp_{idx:02d}_{timestamp}"
		print(f"\n{'='*60}")
		print(f"Run {idx}/{len(configs)}: {run_id}")
		print(f"{'='*60}")
		
		try:
			result, history = train_single_config(
				config=config,
				run_id=run_id,
				max_epochs=40,
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
			f.write("AttentionUNet Hyperparameter Tuning Summary\n")
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

