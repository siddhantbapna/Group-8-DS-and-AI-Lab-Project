from __future__ import annotations

import os
import json
import logging
from typing import Tuple, Dict, Any
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from monai.data import decollate_batch
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose

from config.config import paths, train_cfg, model_cfg
from src.dataset import create_datasets, create_loaders
from src.models import create_model


def validate(model: torch.nn.Module, val_loader, post_pred, post_label, device, spatial_dims=3) -> Tuple[float, float, list]:
	model.eval()
	dice_metric_mean = DiceMetric(include_background=False, reduction="mean")
	dice_metric_pc = DiceMetric(include_background=False, reduction="none")
	val_loss_total = 0.0
	loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
	
	# Choose appropriate ROI size and overlap based on spatial dimensions
	if spatial_dims == 2:
		roi_size = tuple(train_cfg.slide_infer_roi_2d)
		overlap = train_cfg.overlap_2d
	else:
		roi_size = tuple(train_cfg.slide_infer_roi_3d)
		overlap = train_cfg.overlap_3d
	
	with torch.no_grad():
		for batch in tqdm(val_loader, desc="Validation", unit="batch"):
			inputs = torch.cat([batch["t1"], batch["t1ce"], batch["t2"], batch["flair"]], dim=1).to(device)
			labels = batch.get("label")
			if labels is not None:
				labels = labels.to(device)
				
				# For 2D models, we don't need sliding window inference since inputs are already 2D
				if spatial_dims == 2:
					pred = model(inputs)
				else:
					pred = sliding_window_inference(
						inputs,
						roi_size=roi_size,
						sw_batch_size=1,
						overlap=overlap,
						predictor=model,
					)
				val_loss_total += loss_fn(pred, labels).item()
				pred_list = [post_pred(i) for i in decollate_batch(pred)]
				label_list = [post_label(i) for i in decollate_batch(labels)]
				# update both mean and per-class metrics
				dice_metric_mean(y_pred=pred_list, y=label_list)
				dice_metric_pc(y_pred=pred_list, y=label_list)
			else:
				continue
	mean_dice = dice_metric_mean.aggregate().item()
	per_class_tensor = dice_metric_pc.aggregate()
	# Convert to a 1D Python list and replace NaNs with None for JSON compatibility
	per_class = []
	if hasattr(per_class_tensor, 'detach'):
		pc = per_class_tensor.detach().cpu()
		if pc.ndim > 1:
			pc = pc.mean(dim=0)
		# Replace NaNs (classes absent in GT) with None
		per_class = [float(v.item()) if torch.isfinite(v) else None for v in pc]
	dice_metric_mean.reset()
	dice_metric_pc.reset()
	return mean_dice, val_loss_total / max(1, len(val_loader)), per_class


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   scaler: GradScaler, epoch: int, metrics: Dict[str, float], 
                   filepath: str, is_best: bool = False):
	"""Save training checkpoint with all necessary state"""
	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scaler_state_dict": scaler.state_dict(),
		"metrics": metrics,
		"is_best": is_best
	}
	torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   scaler: GradScaler) -> Tuple[int, Dict[str, float]]:
	"""Load training checkpoint and restore state"""
	if not os.path.exists(filepath):
		return 0, {}
	
	checkpoint = torch.load(filepath, map_location="cpu")
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	scaler.load_state_dict(checkpoint["scaler_state_dict"])
	
	epoch = checkpoint.get("epoch", 0)
	metrics = checkpoint.get("metrics", {})
	return epoch, metrics


def setup_logging(model_name: str) -> Tuple[logging.Logger, SummaryWriter]:
	"""Setup logging and TensorBoard writer"""
	# Create timestamp for unique run
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = f"{model_name}_{timestamp}"
	
	# Setup file logging
	log_file = os.path.join(paths.logs, f"{run_name}.log")
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		]
	)
	logger = logging.getLogger(__name__)
	
	# Setup TensorBoard
	tb_dir = os.path.join(paths.logs, "tensorboard", run_name)
	writer = SummaryWriter(tb_dir)
	
	logger.info(f"Logging setup complete. Run: {run_name}")
	logger.info(f"TensorBoard logs: {tb_dir}")
	logger.info(f"File logs: {log_file}")
	
	return logger, writer


def train(model_name: str = "unet3d", spatial_dims: int = 3, resume_from: str = None, max_epochs: int | None = None, no_resume: bool = False):
	# Setup logging and TensorBoard
	logger, writer = setup_logging(model_name)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(f"Using device: {device}")
	# Optimize cuDNN kernels for variable input shapes
	try:
		torch.backends.cudnn.benchmark = True
	except Exception:
		pass

	# Determine target epochs early and disable caching for quick smoke tests
	target_epochs = max_epochs if isinstance(max_epochs, int) and max_epochs > 0 else train_cfg.max_epochs
	from time import perf_counter
	start_build = perf_counter()
	# Lower cache rate to reduce RAM pressure on large datasets
	cache_rate = 0.0 if target_epochs <= 1 else 0.05
	logger.info(f"Building datasets (cache_rate={cache_rate}) …")
	train_ds, val_ds = create_datasets(spatial_dims=spatial_dims, cache_rate=cache_rate)
	train_loader, val_loader = create_loaders(train_ds, val_ds, spatial_dims)
	logger.info(f"Datasets built in {perf_counter()-start_build:.1f}s")
	logger.info(f"Dataset loaded - Train: {len(train_ds)}, Val: {len(val_ds)}")

	model = create_model(
		name=model_name,
		in_channels=model_cfg.in_channels,
		out_channels=model_cfg.out_channels,
		feature_sizes_2d=model_cfg.feature_sizes_2d,
		feature_sizes_3d=model_cfg.feature_sizes_3d,
	).to(device)
	
	# Log model info
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logger.info(f"Model: {model_name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

	optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, target_epochs), eta_min=1e-6)
	# Use new torch.amp API; enable only for CUDA
	scaler = GradScaler("cuda", enabled=(train_cfg.amp and torch.cuda.is_available()))
	# Class weights for CE to reduce background dominance: [bg, class1, class2]
	ce_weight = torch.tensor([0.2, 1.0, 1.0], device=device)
	loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, weight=ce_weight)

	post_pred = Compose([AsDiscrete(argmax=True, to_onehot=model_cfg.out_channels)])
	post_label = Compose([AsDiscrete(to_onehot=model_cfg.out_channels)])

	# Checkpoint paths
	best_path = os.path.join(paths.models, f"best_{model_name}.pth")
	latest_path = os.path.join(paths.models, f"latest_{model_name}.pth")
	history_path = os.path.join(paths.logs, f"training_history_{model_name}.json")

	# Initialize training state
	start_epoch = 0
	best_dice = -1.0
	patience_counter = 0
	patience = train_cfg.early_stopping_patience
	training_history = {"train_loss": [], "val_loss": [], "dice": [], "dice_per_class": []}
	
	# Log training configuration
	logger.info("Training Configuration:")
	logger.info(f"  - Model: {model_name}")
	logger.info(f"  - Spatial dims: {spatial_dims}")
	logger.info(f"  - Batch size: {train_cfg.batch_size_3d if spatial_dims == 3 else train_cfg.batch_size_2d}")
	logger.info(f"  - Learning rate: {train_cfg.lr}")
	logger.info(f"  - AMP: {train_cfg.amp}")
	logger.info(f"  - Gradient clipping: {train_cfg.clip_grad}")
	logger.info(f"  - Early stopping patience: {patience}")

	# Resume from checkpoint if provided
	if resume_from and os.path.exists(resume_from):
		start_epoch, prev_metrics = load_checkpoint(resume_from, model, optimizer, scaler)
		best_dice = prev_metrics.get("best_dice", -1.0)
		logger.info(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
	elif (not no_resume) and os.path.exists(latest_path):
		start_epoch, prev_metrics = load_checkpoint(latest_path, model, optimizer, scaler)
		best_dice = prev_metrics.get("best_dice", -1.0)
		logger.info(f"Resumed from latest checkpoint, epoch {start_epoch}, best dice: {best_dice:.4f}")

	# Target epochs already computed above
	logger.info(f"Starting training from epoch {start_epoch+1} to {target_epochs}")
	
	for epoch in range(start_epoch, target_epochs):
		model.train()
		epoch_loss = 0.0
		num_batches = 0
		
		for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{target_epochs} [train]", unit="batch"):
			inputs = torch.cat([batch["t1"], batch["t1ce"], batch["t2"], batch["flair"]], dim=1).to(device, non_blocking=True)
			labels = batch.get("label")
			if labels is None:
				continue
			labels = labels.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			with autocast(device_type="cuda", enabled=(train_cfg.amp and torch.cuda.is_available())):
				outputs = model(inputs)
				loss = loss_fn(outputs, labels)
			
			# Check for gradient explosion
			if torch.isnan(loss) or torch.isinf(loss):
				logger.warning(f"Invalid loss detected at epoch {epoch+1}, batch {num_batches}")
				continue
				
			scaler.scale(loss).backward()
			
			# Gradient clipping to prevent explosion
			if train_cfg.clip_grad and train_cfg.clip_grad > 0:
				scaler.unscale_(optimizer)
				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
				if grad_norm > train_cfg.clip_grad * 2:
					logger.warning(f"Large gradients detected, norm: {grad_norm:.2f}")
			
			scaler.step(optimizer)
			scaler.update()
			epoch_loss += loss.item()
			num_batches += 1

		if num_batches == 0:
			logger.warning(f"No valid batches in epoch {epoch+1}, skipping...")
			continue

		# Validation
		mean_dice, val_loss, per_class_dice = validate(model, val_loader, post_pred, post_label, device, spatial_dims)
		avg_train_loss = epoch_loss / num_batches
		
		# Update history
		training_history["train_loss"].append(avg_train_loss)
		training_history["val_loss"].append(val_loss)
		training_history["dice"].append(mean_dice)
		
		# Log to TensorBoard
		writer.add_scalar("Loss/Train", avg_train_loss, epoch)
		writer.add_scalar("Loss/Validation", val_loss, epoch)
		writer.add_scalar("Metrics/Dice", mean_dice, epoch)
		writer.add_scalar("Metrics/Best_Dice", best_dice, epoch)
		# Per-class dice (if available)
		if per_class_dice:
			for cls_idx, cls_dice in enumerate(per_class_dice, start=1):
				if cls_dice is not None:
					writer.add_scalar(f"Metrics/Dice_Class_{cls_idx}", cls_dice, epoch)
		
		# Step scheduler and log learning rate
		scheduler.step()
		current_lr = optimizer.param_groups[0]['lr']
		writer.add_scalar("Learning_Rate", current_lr, epoch)
		
		# Save training history (include per-class dice per epoch)
		if per_class_dice:
			training_history["dice_per_class"].append(per_class_dice)
		else:
			training_history["dice_per_class"].append([])
		with open(history_path, 'w') as f:
			json.dump(training_history, f, indent=2)

		# Check for improvement
		is_best = mean_dice > best_dice
		if is_best:
			best_dice = mean_dice
			patience_counter = 0
			# Save best model
			save_checkpoint(model, optimizer, scaler, epoch, 
			               {"best_dice": best_dice, "val_loss": val_loss}, best_path, is_best=True)
			logger.info(f"New best model saved! Dice: {best_dice:.4f}")
		else:
			patience_counter += 1

		# Save latest checkpoint
		save_checkpoint(model, optimizer, scaler, epoch, 
		               {"best_dice": best_dice, "val_loss": val_loss}, latest_path)

		# Log epoch summary
		logger.info(f"Epoch {epoch+1}/{target_epochs} | "
		           f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
		           f"dice={mean_dice:.4f} | best_dice={best_dice:.4f} | "
		           f"patience={patience_counter}/{patience}")

		# Early stopping
		if patience_counter >= patience:
			logger.info(f"Early stopping triggered after {patience} epochs without improvement")
			break

	logger.info(f"Training completed. Best dice: {best_dice:.4f} | checkpoint: {best_path}")
	logger.info(f"Training history saved to: {history_path}")
	
	# Close TensorBoard writer
	writer.close()
	logger.info("TensorBoard writer closed")


if __name__ == "__main__":
	train()
