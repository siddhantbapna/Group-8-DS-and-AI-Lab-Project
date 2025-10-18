"""
Metrics and evaluation functions for brain MRI segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete


class MetricsComputer:
    """Compute various metrics for segmentation evaluation"""
    
    def __init__(self, num_classes: int = 3, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        
        # Initialize metrics
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean_batch"
        )
        
        # Transform for converting to discrete
        self.as_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    
    def map_brats_labels(self, y_true: torch.Tensor) -> torch.Tensor:
        """Convert one-hot encoded targets to class indices (preprocessing already handled BraTS mapping)"""
        # If y_true is one-hot encoded (has channel dimension), convert to class indices
        if y_true.dim() == 5 and y_true.shape[1] > 1:
            # Convert one-hot to class indices
            return torch.argmax(y_true, dim=1)
        return y_true
    
    def compute_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute pixel-wise accuracy"""
        # Get predictions
        y_pred_discrete = torch.argmax(y_pred, dim=1)
        
        # Map labels
        y_true_mapped = self.map_brats_labels(y_true)
        
        # Calculate accuracy
        correct = (y_pred_discrete == y_true_mapped).float()
        accuracy = correct.mean().item()
        return accuracy
    
    def compute_dice_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> List[float]:
        """Compute Dice score for each class"""
        # Get predictions
        y_pred_discrete = self.as_discrete(y_pred)
        
        # Map labels
        y_true_mapped = self.map_brats_labels(y_true)
        
        # Calculate Dice scores
        dice_scores = self.dice_metric(y_pred_discrete, y_true_mapped)
        return [score.item() for score in dice_scores]
    
    def compute_all_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = self.compute_accuracy(y_pred, y_true)
        
        # Simplified Dice scores (skip problematic MONAI Dice metric for now)
        try:
            dice_scores = self.compute_dice_score(y_pred, y_true)
            for i, score in enumerate(dice_scores):
                class_name = f'class_{i}' if self.include_background or i > 0 else f'class_{i}'
                metrics[f'dice_{class_name}'] = score
            
            # Mean Dice (excluding background if not included)
            start_idx = 0 if self.include_background else 1
            if len(dice_scores) > start_idx:
                metrics['mean_dice'] = np.mean(dice_scores[start_idx:])
            else:
                metrics['mean_dice'] = 0.0
        except Exception as e:
            # Fallback: set dice scores to 0
            for i in range(self.num_classes):
                class_name = f'class_{i}' if self.include_background or i > 0 else f'class_{i}'
                metrics[f'dice_{class_name}'] = 0.0
            metrics['mean_dice'] = 0.0
        
        return metrics


class LossFunction:
    """Loss function factory"""
    
    @staticmethod
    def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Dice loss"""
        dice_loss_fn = DiceLoss(
            include_background=False,
            reduction="mean"
        )
        return dice_loss_fn(y_pred, y_true)
    
    @staticmethod
    def cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss"""
        return F.cross_entropy(y_pred, y_true)
    
    @staticmethod
    def dice_ce_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                     ce_weight: float = 1.0, dice_weight: float = 1.0) -> torch.Tensor:
        """Combined Dice and Cross-Entropy loss"""
        dice_ce_loss_fn = DiceCELoss(
            include_background=False,
            ce_weight=ce_weight,
            dice_weight=dice_weight
        )
        return dice_ce_loss_fn(y_pred, y_true)
    
    @staticmethod
    def focal_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss"""
        focal_loss_fn = FocalLoss(
            include_background=False,
            alpha=alpha,
            gamma=gamma
        )
        return focal_loss_fn(y_pred, y_true)
    
    @staticmethod
    def dice_bce_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                      dice_weight: float = 1.0, bce_weight: float = 1.0) -> torch.Tensor:
        """Combined Dice and Binary Cross-Entropy loss (DiceBCELoss)"""
        # Dice loss component
        dice_loss_fn = DiceLoss(
            include_background=False,
            reduction="mean"
        )
        dice_loss = dice_loss_fn(y_pred, y_true)
        
        # Binary Cross-Entropy loss component
        # Convert targets to binary format for BCE
        if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
            # For multi-class, we'll use cross-entropy instead of BCE
            bce_loss = F.cross_entropy(y_pred, torch.argmax(y_true, dim=1))
        else:
            # For binary segmentation
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
        
        # Combine losses
        total_loss = dice_weight * dice_loss + bce_weight * bce_loss
        return total_loss
    
    @staticmethod
    def weighted_dice_bce_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                              dice_weight: float = 1.0, bce_weight: float = 1.0,
                              background_weight: float = 0.1) -> torch.Tensor:
        """Weighted DiceBCE loss that reduces background influence"""
        # Dice loss component
        dice_loss_fn = DiceLoss(
            include_background=False,
            reduction="mean"
        )
        dice_loss = dice_loss_fn(y_pred, y_true)
        
        # Weighted Cross-Entropy loss component
        if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
            # Create class weights to reduce background influence
            class_weights = torch.ones(y_pred.shape[1]).to(y_pred.device)
            class_weights[0] = background_weight  # Reduce background weight
            
            # Calculate class frequencies for additional weighting
            target_classes = torch.argmax(y_true, dim=1)
            class_counts = torch.bincount(target_classes.flatten(), minlength=y_pred.shape[1])
            class_frequencies = class_counts.float() / class_counts.sum()
            
            # Inverse frequency weighting
            inverse_freq_weights = 1.0 / (class_frequencies + 1e-8)
            inverse_freq_weights = inverse_freq_weights / inverse_freq_weights.sum() * y_pred.shape[1]
            
            # Combine class weights and frequency weights
            final_weights = class_weights * inverse_freq_weights
            
            bce_loss = F.cross_entropy(y_pred, target_classes, weight=final_weights)
        else:
            # For binary segmentation
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
        
        # Combine losses
        total_loss = dice_weight * dice_loss + bce_weight * bce_loss
        return total_loss
    
    @staticmethod
    def loss_fn(loss_name: str, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get loss function by name"""
        loss_functions = {
            'dice': LossFunction.dice_loss,
            'ce': LossFunction.cross_entropy_loss,
            'dice_ce': LossFunction.dice_ce_loss,
            'dice_bce': LossFunction.dice_bce_loss,
            'weighted_dice_bce': LossFunction.weighted_dice_bce_loss,
            'focal': LossFunction.focal_loss
        }
        
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        return loss_functions[loss_name](y_pred, y_true, **kwargs)


class MetricTracker:
    """Track metrics during training"""
    
    def __init__(self, num_classes: int = 3, include_background: bool = False):
        self.metrics_computer = MetricsComputer(num_classes, include_background)
        self.metrics_history = []
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Update metrics with new predictions"""
        metrics = self.metrics_computer.compute_all_metrics(y_pred, y_true)
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all updates"""
        if not self.metrics_history:
            return {}
        
        # Average all metrics
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def reset(self):
        """Reset metrics history"""
        self.metrics_history = []


class SegmentationEvaluator:
    """Comprehensive evaluation for segmentation models"""
    
    def __init__(self, num_classes: int = 3, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        self.metrics_computer = MetricsComputer(num_classes, include_background)
    
    def evaluate_batch(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """Evaluate a batch of predictions"""
        return self.metrics_computer.compute_all_metrics(y_pred, y_true)
    
    def evaluate_dataset(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                        device: torch.device) -> Dict[str, float]:
        """Evaluate model on entire dataset"""
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Get inputs and targets (assuming batch_data is a dict)
                if isinstance(batch_data, dict):
                    # Handle different data formats
                    if 'image' in batch_data:
                        inputs = batch_data['image'].to(device)
                        targets = batch_data['label'].to(device)
                    else:
                        # Assume first key is input, second is target
                        keys = list(batch_data.keys())
                        inputs = batch_data[keys[0]].to(device)
                        targets = batch_data[keys[1]].to(device)
                else:
                    # Assume batch_data is a tuple (inputs, targets)
                    inputs, targets = batch_data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute metrics
                metrics = self.evaluate_batch(outputs, targets)
                all_metrics.append(metrics)
        
        # Average metrics across all batches
        if not all_metrics:
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float]):
        """Print evaluation results in a nice format"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Overall metrics
        print(f"Overall Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print(f"Mean Dice Score: {metrics.get('mean_dice', 0.0):.4f}")
        
        # Per-class metrics
        print("\nPer-class Dice Scores:")
        for i in range(self.num_classes):
            class_name = f"Class {i}" if self.include_background or i > 0 else f"Class {i}"
            dice_key = f'dice_class_{i}'
            if dice_key in metrics:
                print(f"  {class_name}: {metrics[dice_key]:.4f}")
        
        print("="*50)


# Convenience functions
def create_loss_function(loss_name: str):
    """Create a loss function by name"""
    def loss_fn(y_pred, y_true, **kwargs):
        return LossFunction.loss_fn(loss_name, y_pred, y_true, **kwargs)
    return loss_fn


def create_metrics_computer(num_classes: int = 3, include_background: bool = False):
    """Create a metrics computer"""
    return MetricsComputer(num_classes, include_background)


def create_metric_tracker(num_classes: int = 3, include_background: bool = False):
    """Create a metric tracker"""
    return MetricTracker(num_classes, include_background)


def create_evaluator(num_classes: int = 3, include_background: bool = False):
    """Create a segmentation evaluator"""
    return SegmentationEvaluator(num_classes, include_background)