"""
Utility functions for brain MRI segmentation project
"""
import os
import torch
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nibabel as nib

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__
    }
    
    return device_info

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            json.dump(config, f, indent=2)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config

def create_directory_structure(base_dir: str, subdirs: List[str]):
    """Create directory structure"""
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)

def plot_training_history(history: List[Dict[str, Any]], save_path: str):
    """Plot training history"""
    if not history:
        return
    
    # Extract metrics
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_metrics']['loss'] for h in history]
    val_losses = [h['val_metrics']['loss'] for h in history]
    
    # Get dice scores
    val_dice_scores = []
    for h in history:
        dice_score = h['val_metrics'].get('dice_class_1', 0)
        val_dice_scores.append(dice_score)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axes[0].plot(epochs, val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice score plot
    axes[1].plot(epochs, val_dice_scores, label='Validation Dice', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_segmentation(volume: np.ndarray, segmentation: np.ndarray, 
                          slice_idx: int, save_path: str):
    """Visualize segmentation on a slice"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original volume
    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title('Original Volume')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(segmentation[slice_idx], cmap='viridis')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(volume[slice_idx], cmap='gray', alpha=0.7)
    axes[2].imshow(segmentation[slice_idx], cmap='viridis', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_volume_statistics(volume: np.ndarray) -> Dict[str, float]:
    """Compute volume statistics"""
    return {
        'mean': np.mean(volume),
        'std': np.std(volume),
        'min': np.min(volume),
        'max': np.max(volume),
        'median': np.median(volume),
        'percentile_25': np.percentile(volume, 25),
        'percentile_75': np.percentile(volume, 75)
    }

def compute_segmentation_statistics(segmentation: np.ndarray, 
                                   class_names: List[str]) -> Dict[str, Any]:
    """Compute segmentation statistics"""
    stats = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = (segmentation == i)
        class_volume = np.sum(class_mask)
        class_percentage = (class_volume / segmentation.size) * 100
        
        stats[class_name] = {
            'volume': class_volume,
            'percentage': class_percentage
        }
    
    return stats

def load_nifti_volume(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI volume and return data and affine"""
    volume = nib.load(file_path)
    return volume.get_fdata(), volume.affine

def save_nifti_volume(data: np.ndarray, affine: np.ndarray, file_path: str):
    """Save NIfTI volume"""
    volume = nib.Nifti1Image(data, affine)
    nib.save(volume, file_path)

def resize_volume(volume: np.ndarray, target_shape: Tuple[int, ...], 
                 order: int = 1) -> np.ndarray:
    """Resize volume to target shape"""
    from scipy.ndimage import zoom
    
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(len(target_shape))]
    resized_volume = zoom(volume, zoom_factors, order=order)
    
    return resized_volume

def normalize_volume(volume: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """Normalize volume"""
    if method == 'zscore':
        return (volume - np.mean(volume)) / np.std(volume)
    elif method == 'minmax':
        return (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    elif method == 'robust':
        median = np.median(volume)
        mad = np.median(np.abs(volume - median))
        return (volume - median) / (1.4826 * mad)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_ensemble_prediction(predictions: List[np.ndarray], 
                              method: str = 'average') -> np.ndarray:
    """Create ensemble prediction from multiple predictions"""
    if method == 'average':
        return np.mean(predictions, axis=0)
    elif method == 'majority_vote':
        return np.argmax(np.sum(predictions, axis=0), axis=0)
    elif method == 'weighted_average':
        # Simple weighted average (can be customized)
        weights = np.ones(len(predictions)) / len(predictions)
        return np.average(predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

def compute_ensemble_metrics(predictions: List[np.ndarray], 
                            ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute metrics for ensemble predictions"""
    from src.metrics import SegmentationMetrics
    
    metrics_computer = SegmentationMetrics(num_classes=3, include_background=False)
    
    ensemble_metrics = {}
    
    for i, prediction in enumerate(predictions):
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0)
        
        metrics = metrics_computer.compute_all_metrics(pred_tensor, gt_tensor)
        
        for key, value in metrics.items():
            ensemble_metrics[f'model_{i}_{key}'] = value
    
    # Compute ensemble prediction
    ensemble_pred = create_ensemble_prediction(predictions)
    ensemble_tensor = torch.from_numpy(ensemble_pred).unsqueeze(0).unsqueeze(0)
    
    ensemble_metrics_ensemble = metrics_computer.compute_all_metrics(ensemble_tensor, gt_tensor)
    
    for key, value in ensemble_metrics_ensemble.items():
        ensemble_metrics[f'ensemble_{key}'] = value
    
    return ensemble_metrics

def create_model_comparison_report(model_results: Dict[str, Dict[str, float]], 
                                  save_path: str):
    """Create model comparison report"""
    # Create comparison table
    models = list(model_results.keys())
    metrics = list(model_results[models[0]].keys())
    
    # Create DataFrame
    import pandas as pd
    
    df = pd.DataFrame(model_results).T
    
    # Save as CSV
    csv_path = save_path.replace('.html', '.csv')
    df.to_csv(csv_path)
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .best {{ background-color: #90EE90; }}
        </style>
    </head>
    <body>
        <h1>Model Comparison Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Results Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                {''.join([f'<th>{metric}</th>' for metric in metrics])}
            </tr>
            {''.join([f'<tr><td>{model}</td>' + 
                     ''.join([f'<td>{results[metric]:.4f}</td>' for metric in metrics]) + 
                     '</tr>' for model, results in model_results.items()])}
        </table>
        
        <h2>Best Models by Metric</h2>
        <ul>
            {''.join([f'<li><strong>{metric}:</strong> {max(model_results.items(), key=lambda x: x[1][metric])[0]} ({max(model_results.items(), key=lambda x: x[1][metric])[1][metric]:.4f})</li>' for metric in metrics])}
        </ul>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)

# Example usage
if __name__ == "__main__":
    # Test utility functions
    set_seed(42)
    
    # Test device info
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Test parameter counting
    import torch.nn as nn
    model = nn.Conv3d(4, 3, 3)
    params = count_parameters(model)
    print(f"\nModel Parameters: {params}")
    
    # Test volume statistics
    volume = np.random.randn(64, 64, 64)
    stats = compute_volume_statistics(volume)
    print(f"\nVolume Statistics: {stats}")
    
    # Test segmentation statistics
    segmentation = np.random.randint(0, 3, (64, 64, 64))
    seg_stats = compute_segmentation_statistics(segmentation, ['Background', 'WT', 'TC', 'ET'])
    print(f"\nSegmentation Statistics: {seg_stats}")
