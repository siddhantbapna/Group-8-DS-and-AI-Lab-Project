"""
Comprehensive training pipeline with cross-validation support for brain MRI segmentation
"""
import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json
from datetime import datetime

from config.config import Config
from src.preprocessing import BraTS2023Preprocessor
from src.models import create_model
from src.metrics import MetricsComputer, MetricTracker, create_loss_function
from src.checkpoints import CheckpointManager, CheckpointManagerFactory

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

class Trainer:
    """
    Comprehensive trainer for brain MRI segmentation
    """
    
    def __init__(self, config: Config, fold: int = 0):
        self.config = config
        self.fold = fold
        self.device = torch.device(config.system.device)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.preprocessor = BraTS2023Preprocessor(config.data)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_function = None
        self.metrics_computer = None
        self.checkpoint_manager = None
        self.early_stopping = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.training_history = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.config.system.log_dir, f"fold_{self.fold}")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"Trainer_Fold_{self.fold}")
    
    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        # Create model
        self.model = create_model(
            self.config.model.model_name,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            features=self.config.model.features,
            dropout=self.config.model.dropout
        ).to(self.device)
        
        # Create optimizer
        if self.config.training.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Create scheduler
        if self.config.training.scheduler.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.training.scheduler.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10
            )
        elif self.config.training.scheduler.lower() == 'poly':
            # Polynomial learning rate scheduler: (1 - epoch/num_epochs)^0.9
            poly_lambda = lambda epoch: (1 - epoch / self.config.training.num_epochs) ** 0.9
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=poly_lambda)
        else:
            self.scheduler = None
        
        # Create loss function
        self.loss_function = create_loss_function(self.config.training.loss_function)
        
        # Create metrics computer
        self.metrics_computer = MetricsComputer(
            num_classes=self.config.model.out_channels,
            include_background=False
        )
        
        # Create checkpoint manager
        model_name = f"{self.config.model.model_name}_fold_{self.fold}"
        self.checkpoint_manager = CheckpointManagerFactory.create_manager(
            model_name, self.config.system.checkpoint_dir
        )
        
        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta
        )
        
        # Create tensorboard writer
        tb_dir = os.path.join(self.config.system.log_dir, f"tensorboard_fold_{self.fold}")
        self.writer = SummaryWriter(tb_dir)
        
        # Create mixed precision scaler
        if self.config.training.use_amp:
            self.scaler = GradScaler()
        
        self.logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self):
        """Setup data loaders"""
        # Create data dictionaries
        train_data_dicts = self.preprocessor.create_data_dicts(self.config.data.train_data_path)
        
        # Create cross-validation splits
        cv_splits = self.preprocessor.create_cross_validation_splits(train_data_dicts)
        
        if self.fold >= len(cv_splits):
            raise ValueError(f"Fold {self.fold} not available. Total folds: {len(cv_splits)}")
        
        # Get data for current fold
        train_data, val_data = cv_splits[self.fold]
        
        # Create transforms
        train_transforms = self.preprocessor.get_train_transforms(is_training=True)
        val_transforms = self.preprocessor.get_val_transforms()
        
        # Create datasets
        train_dataset, val_dataset = self.preprocessor.create_datasets(
            train_data, val_data, train_transforms, val_transforms
        )
        
        # Create data loaders
        self.train_loader, self.val_loader = self.preprocessor.create_dataloaders(
            train_dataset, val_dataset, 
            self.config.training.batch_size, 
            self.config.system.num_workers
        )
        
        self.logger.info(f"Data setup complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        # Initialize metric tracker
        metric_tracker = MetricTracker(
            num_classes=self.config.model.out_channels,
            include_background=False
        )
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Stack all modalities
            modalities = []
            for modality in self.config.data.modality_keys:
                modalities.append(batch_data[modality])
            inputs = torch.cat(modalities, dim=1).to(self.device)  # Stack along channel dimension
            targets = batch_data[self.config.data.seg_key].to(self.device)
            
            # Targets are already one-hot encoded by ConvertToMultiChannelBasedOnBratsClassesd
            # No need for manual mapping - the preprocessing already handles BraTS label conversion
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.training.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            metric_tracker.update(outputs, targets)
            
            # Calculate metrics for progress bar
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                
                # Targets are already one-hot encoded, convert to class indices
                targets_class = torch.argmax(targets, dim=1)
                
                # Calculate pixel-wise accuracy (dominated by background)
                correct = (predictions == targets_class).float()
                pixel_accuracy = correct.mean().item()
                
                # Calculate class-balanced accuracy (better for imbalanced data)
                class_accuracies = []
                for class_id in range(self.config.model.out_channels):
                    class_mask = (targets_class == class_id)
                    if class_mask.sum() > 0:  # Only if class exists in batch
                        class_correct = (predictions == targets_class) & class_mask
                        class_acc = class_correct.sum().float() / class_mask.sum().float()
                        class_accuracies.append(class_acc.item())
                
                # Average class accuracy (excluding background if it dominates)
                if len(class_accuracies) > 1:
                    balanced_accuracy = np.mean(class_accuracies[1:])  # Exclude background
                else:
                    balanced_accuracy = pixel_accuracy
                
                # Calculate simple Dice score for tumor classes
                dice_scores = []
                for class_id in range(1, self.config.model.out_channels):  # Skip background
                    pred_mask = (predictions == class_id)
                    true_mask = (targets_class == class_id)
                    
                    intersection = (pred_mask & true_mask).sum().float()
                    union = pred_mask.sum().float() + true_mask.sum().float()
                    
                    if union > 0:
                        dice = (2.0 * intersection) / union
                        dice_scores.append(dice.item())
                
                mean_dice = np.mean(dice_scores) if dice_scores else 0.0
            
            # Update progress bar with better metrics
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PixAcc': f'{pixel_accuracy:.3f}',
                'BalAcc': f'{balanced_accuracy:.3f}',
                'Dice': f'{mean_dice:.3f}',
                'AvgLoss': f'{epoch_loss / (batch_idx + 1):.3f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config.system.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Compute epoch metrics
        epoch_metrics = metric_tracker.get_average_metrics()
        epoch_metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        # Initialize metric tracker
        metric_tracker = MetricTracker(
            num_classes=self.config.model.out_channels,
            include_background=False
        )
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch_data in enumerate(progress_bar):
                # Stack all modalities
                modalities = []
                for modality in self.config.data.modality_keys:
                    modalities.append(batch_data[modality])
                inputs = torch.cat(modalities, dim=1).to(self.device)  # Stack along channel dimension
                targets = batch_data[self.config.data.seg_key].to(self.device)
                
                # Targets are already one-hot encoded by ConvertToMultiChannelBasedOnBratsClassesd
                # No need for manual mapping - the preprocessing already handles BraTS label conversion
                
                # Forward pass
                if self.config.training.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                
                # Update metrics
                epoch_loss += loss.item()
                metric_tracker.update(outputs, targets)
                
                # Calculate metrics for progress bar
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1)
                    
                    # Targets are already one-hot encoded, convert to class indices
                    targets_class = torch.argmax(targets, dim=1)
                    
                    # Calculate pixel-wise accuracy (dominated by background)
                    correct = (predictions == targets_class).float()
                    pixel_accuracy = correct.mean().item()
                    
                    # Calculate class-balanced accuracy (better for imbalanced data)
                    class_accuracies = []
                    for class_id in range(self.config.model.out_channels):
                        class_mask = (targets_class == class_id)
                        if class_mask.sum() > 0:  # Only if class exists in batch
                            class_correct = (predictions == targets_class) & class_mask
                            class_acc = class_correct.sum().float() / class_mask.sum().float()
                            class_accuracies.append(class_acc.item())
                    
                    # Average class accuracy (excluding background if it dominates)
                    if len(class_accuracies) > 1:
                        balanced_accuracy = np.mean(class_accuracies[1:])  # Exclude background
                    else:
                        balanced_accuracy = pixel_accuracy
                    
                    # Calculate simple Dice score for tumor classes
                    dice_scores = []
                    for class_id in range(1, self.config.model.out_channels):  # Skip background
                        pred_mask = (predictions == class_id)
                        true_mask = (targets_class == class_id)
                        
                        intersection = (pred_mask & true_mask).sum().float()
                        union = pred_mask.sum().float() + true_mask.sum().float()
                        
                        if union > 0:
                            dice = (2.0 * intersection) / union
                            dice_scores.append(dice.item())
                    
                    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
                
                # Update progress bar with better metrics
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PixAcc': f'{pixel_accuracy:.3f}',
                    'BalAcc': f'{balanced_accuracy:.3f}',
                    'Dice': f'{mean_dice:.3f}',
                    'AvgLoss': f'{epoch_loss / (batch_idx + 1):.3f}'
                })
        
        # Compute epoch metrics
        epoch_metrics = metric_tracker.get_average_metrics()
        epoch_metrics['loss'] = epoch_loss / len(self.val_loader)
        
        return epoch_metrics
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        
        # Save model summary
        self.checkpoint_manager.save_model_summary(
            self.model, self.config.to_dict()
        )
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint, self.model, self.optimizer, self.scheduler, self.device
            )
            self.current_epoch = checkpoint_data['epoch'] + 1
            self.best_metric = checkpoint_data['best_metric']
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('dice_class_1', 0))
                else:
                    self.scheduler.step()
            
            # Check if this is the best model
            current_metric = val_metrics.get('dice_class_1', 0)  # Use WT dice as main metric
            is_best = self.checkpoint_manager.update_best_metric(current_metric, epoch)
            
            # Save checkpoint
            if epoch % self.config.system.save_interval == 0 or is_best:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    epoch, self.model, self.optimizer, self.scheduler,
                    val_metrics, val_metrics['loss'], is_best
                )
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            train_acc = train_metrics.get('accuracy', 0.0)
            val_acc = val_metrics.get('accuracy', 0.0)
            val_dice = val_metrics.get('mean_dice', 0.0)
            self.logger.info(
                f"Epoch {epoch}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val Dice: {val_dice:.4f}, Time: {epoch_time:.2f}s"
            )
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Dice', val_dice, epoch)
            
            # Log additional metrics if available
            if 'mean_dice' in train_metrics:
                self.writer.add_scalar('Epoch/Train_Dice', train_metrics['mean_dice'], epoch)
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'epoch_time': epoch_time
            })
            
            # Early stopping
            if self.early_stopping(current_metric):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final results
        self.save_training_results()
        
        self.logger.info("Training completed!")
    
    def save_training_results(self):
        """Save training results and history"""
        results_dir = os.path.join(self.config.system.output_dir, f"results_fold_{self.fold}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save training history
        history_path = os.path.join(results_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save final metrics
        if self.training_history:
            final_metrics = self.training_history[-1]['val_metrics']
            metrics_path = os.path.join(results_dir, "final_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2)
        
        # Export best model
        best_model_path = os.path.join(results_dir, "best_model.pth")
        self.checkpoint_manager.export_model(
            self.model, best_model_path, "pytorch"
        )
        
        self.logger.info(f"Training results saved to: {results_dir}")

class CrossValidationTrainer:
    """
    Cross-validation trainer for multiple folds
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.fold_results = []
    
    def train_all_folds(self, resume_fold: Optional[int] = None):
        """Train all folds"""
        self.logger = logging.getLogger("CrossValidationTrainer")
        
        for fold in range(self.config.data.n_folds):
            if resume_fold is not None and fold < resume_fold:
                continue
            
            self.logger.info(f"Starting training for fold {fold}")
            
            # Create trainer for this fold
            trainer = Trainer(self.config, fold)
            
            try:
                # Train
                trainer.train()
                
                # Store results
                self.fold_results.append({
                    'fold': fold,
                    'best_metric': trainer.best_metric,
                    'training_history': trainer.training_history
                })
                
            except Exception as e:
                self.logger.error(f"Error training fold {fold}: {e}")
                continue
        
        # Save cross-validation results
        self.save_cv_results()
    
    def save_cv_results(self):
        """Save cross-validation results"""
        cv_results_dir = os.path.join(self.config.system.output_dir, "cv_results")
        os.makedirs(cv_results_dir, exist_ok=True)
        
        # Compute average metrics
        if self.fold_results:
            avg_metrics = self.compute_average_metrics()
            
            # Save results
            results_path = os.path.join(cv_results_dir, "cv_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'fold_results': self.fold_results,
                    'average_metrics': avg_metrics
                }, f, indent=2)
            
            self.logger.info(f"Cross-validation results saved to: {cv_results_dir}")
            self.logger.info(f"Average Dice Score: {avg_metrics.get('dice_class_1', 0):.4f}")
    
    def compute_average_metrics(self) -> Dict[str, float]:
        """Compute average metrics across all folds"""
        if not self.fold_results:
            return {}
        
        # Get all metrics from the last epoch of each fold
        all_metrics = []
        for fold_result in self.fold_results:
            if fold_result['training_history']:
                last_epoch_metrics = fold_result['training_history'][-1]['val_metrics']
                all_metrics.append(last_epoch_metrics)
        
        if not all_metrics:
            return {}
        
        # Compute averages
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)
        
        return avg_metrics

# Example usage
if __name__ == "__main__":
    from config.config import get_config
    
    # Get configuration
    config = get_config('unet3d')
    
    # Create trainer
    trainer = Trainer(config, fold=0)
    
    # Train
    trainer.train()
    
    # Or run cross-validation
    cv_trainer = CrossValidationTrainer(config)
    cv_trainer.train_all_folds()
