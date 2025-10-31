from __future__ import annotations

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from typing import List, Dict, Tuple, Optional
import torch
from monai.transforms import AsDiscrete, Compose
from monai.inferers import sliding_window_inference

from config.config import paths, train_cfg, model_cfg
from src.models import create_model


def visualize_data_sample(case_dir: str, save_path: Optional[str] = None):
    """Visualize a single case with all modalities and segmentation"""
    # Map visualization names to actual file patterns
    modality_mapping = {
        't1': ['*t1n.nii.gz', '*t1n.nii', '*t1*.nii.gz', '*t1*.nii'],
        't1ce': ['*t1c.nii.gz', '*t1c.nii', '*t1ce*.nii.gz', '*t1ce*.nii'],
        't2': ['*t2w.nii.gz', '*t2w.nii', '*t2*.nii.gz', '*t2*.nii'],
        'flair': ['*t2f.nii.gz', '*t2f.nii', '*flair*.nii.gz', '*flair*.nii']
    }
    modalities = ['t1', 't1ce', 't2', 'flair']
    modality_paths = {}
    
    # Load modalities
    for mod in modalities:
        patterns = modality_mapping[mod]
        for pattern in patterns:
            matches = glob.glob(os.path.join(case_dir, pattern))
            if matches:
                modality_paths[mod] = os.path.join(case_dir, matches[0])
                break
    
    # Load segmentation
    seg_patterns = ["*seg.nii.gz", "*seg.nii", "*seg*.nii.gz", "*seg*.nii"]
    seg_path = None
    for pattern in seg_patterns:
        matches = glob.glob(os.path.join(case_dir, pattern))
        if matches:
            seg_path = os.path.join(case_dir, matches[0])
            break
    
    # Load data
    data = {}
    for mod, path in modality_paths.items():
        img = nib.load(path)
        data[mod] = img.get_fdata().astype(np.float32)
    
    if seg_path:
        seg_img = nib.load(seg_path)
        data['seg'] = seg_img.get_fdata().astype(np.uint8)
    
    # Get middle slices
    shape = data[modalities[0]].shape
    mid_slices = [shape[i] // 2 for i in range(3)]
    
    # Create visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'Data Sample: {os.path.basename(case_dir)}', fontsize=16)
    
    # Plot each modality in 3 orthogonal views
    for i, mod in enumerate(modalities):
        img = data[mod]
        
        # Normalize for display
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Axial (Z slice)
        axes[0, i].imshow(img_norm[mid_slices[0], :, :], cmap='gray', origin='lower')
        axes[0, i].set_title(f'{mod.upper()} - Axial')
        axes[0, i].axis('off')
        
        # Coronal (Y slice)
        axes[1, i].imshow(img_norm[:, mid_slices[1], :], cmap='gray', origin='lower')
        axes[1, i].set_title(f'{mod.upper()} - Coronal')
        axes[1, i].axis('off')
        
        # Sagittal (X slice)
        axes[2, i].imshow(img_norm[:, :, mid_slices[2]], cmap='gray', origin='lower')
        axes[2, i].set_title(f'{mod.upper()} - Sagittal')
        axes[2, i].axis('off')
    
    # Segmentation overlay
    if 'seg' in data:
        seg = data['seg']
        
        # Create colored segmentation
        seg_colored = np.zeros((*seg.shape, 3))
        seg_colored[seg == 1] = [1, 0, 0]  # Red for NCR/NET
        seg_colored[seg == 2] = [0, 1, 0]  # Green for ED
        seg_colored[seg == 3] = [0, 0, 1]  # Blue for ET
        
        # Overlay on FLAIR
        flair = data['flair']
        flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
        
        # Axial overlay
        axes[0, 4].imshow(flair_norm[mid_slices[0], :, :], cmap='gray', origin='lower')
        axes[0, 4].imshow(seg_colored[mid_slices[0], :, :], alpha=0.5, origin='lower')
        axes[0, 4].set_title('Segmentation - Axial')
        axes[0, 4].axis('off')
        
        # Coronal overlay
        axes[1, 4].imshow(flair_norm[:, mid_slices[1], :], cmap='gray', origin='lower')
        axes[1, 4].imshow(seg_colored[:, mid_slices[1], :], alpha=0.5, origin='lower')
        axes[1, 4].set_title('Segmentation - Coronal')
        axes[1, 4].axis('off')
        
        # Sagittal overlay
        axes[2, 4].imshow(flair_norm[:, :, mid_slices[2]], cmap='gray', origin='lower')
        axes[2, 4].imshow(seg_colored[:, :, mid_slices[2]], alpha=0.5, origin='lower')
        axes[2, 4].set_title('Segmentation - Sagittal')
        axes[2, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Data visualization saved to: {save_path}")
    
    plt.show()


def visualize_training_history(history_path: str, save_path: Optional[str] = None):
    """Visualize training history from JSON file"""
    import json
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice score
    axes[0, 1].plot(epochs, history['dice'], label='Validation Dice', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss comparison (log scale)
    axes[1, 0].semilogy(epochs, history['train_loss'], label='Train Loss', color='blue')
    axes[1, 0].semilogy(epochs, history['val_loss'], label='Validation Loss', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Loss Curves (Log Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Best performance summary
    best_dice = max(history['dice'])
    best_epoch = history['dice'].index(best_dice) + 1
    final_dice = history['dice'][-1]
    
    summary_text = f"""Training Summary:
Best Dice: {best_dice:.4f} (Epoch {best_epoch})
Final Dice: {final_dice:.4f}
Total Epochs: {len(epochs)}
Final Train Loss: {history['train_loss'][-1]:.4f}
Final Val Loss: {history['val_loss'][-1]:.4f}"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history visualization saved to: {save_path}")
    
    plt.show()


def visualize_predictions(model_name: str, case_dir: str, ckpt_path: str, 
                         save_path: Optional[str] = None):
    """Visualize model predictions on validation data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_model(
        name=model_name,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        feature_sizes_3d=model_cfg.feature_sizes_3d,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()
    
    # Load data
    modality_mapping = {
        't1': ['*t1n.nii.gz', '*t1n.nii', '*t1*.nii.gz', '*t1*.nii'],
        't1ce': ['*t1c.nii.gz', '*t1c.nii', '*t1ce*.nii.gz', '*t1ce*.nii'],
        't2': ['*t2w.nii.gz', '*t2w.nii', '*t2*.nii.gz', '*t2*.nii'],
        'flair': ['*t2f.nii.gz', '*t2f.nii', '*flair*.nii.gz', '*flair*.nii']
    }
    modalities = ['t1', 't1ce', 't2', 'flair']
    modality_paths = {}
    
    for mod in modalities:
        patterns = modality_mapping[mod]
        for pattern in patterns:
            matches = glob.glob(os.path.join(case_dir, pattern))
            if matches:
                modality_paths[mod] = os.path.join(case_dir, matches[0])
                break
    
    # Load segmentation
    seg_patterns = ["*seg.nii.gz", "*seg.nii", "*seg*.nii.gz", "*seg*.nii"]
    seg_path = None
    for pattern in seg_patterns:
        matches = glob.glob(os.path.join(case_dir, pattern))
        if matches:
            seg_path = os.path.join(case_dir, matches[0])
            break
    
    # Load and preprocess data
    data = {}
    for mod, path in modality_paths.items():
        img = nib.load(path)
        data[mod] = img.get_fdata().astype(np.float32)
        # Simple normalization
        data[mod] = (data[mod] - data[mod].min()) / (data[mod].max() - data[mod].min() + 1e-8)
    
    if seg_path:
        seg_img = nib.load(seg_path)
        data['seg'] = seg_img.get_fdata().astype(np.uint8)
    
    # Prepare input tensor
    input_tensor = np.stack([data[mod] for mod in modalities], axis=0)
    input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dimension
    input_tensor = torch.from_numpy(input_tensor).to(device)
    
    # Run inference
    with torch.no_grad():
        pred = sliding_window_inference(
            input_tensor,
            size=tuple(train_cfg.slide_infer_roi_3d),
            overlap=train_cfg.overlap_3d,
            predictor=model,
        )
        
        # Post-process prediction
        post_pred = Compose([AsDiscrete(argmax=True)])
        pred = post_pred(pred)
        pred_np = pred[0].cpu().numpy().astype(np.uint8)
    
    # Get middle slices
    shape = data[modalities[0]].shape
    mid_slices = [shape[i] // 2 for i in range(3)]
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Model Predictions: {model_name} on {os.path.basename(case_dir)}', fontsize=16)
    
    # Ground truth and prediction
    if 'seg' in data:
        gt_seg = data['seg']
        pred_seg = pred_np
        
        # Create colored segmentations
        gt_colored = np.zeros((*gt_seg.shape, 3))
        gt_colored[gt_seg == 1] = [1, 0, 0]  # Red for NCR/NET
        gt_colored[gt_seg == 2] = [0, 1, 0]  # Green for ED
        gt_colored[gt_seg == 3] = [0, 0, 1]  # Blue for ET
        
        pred_colored = np.zeros((*pred_seg.shape, 3))
        pred_colored[pred_seg == 1] = [1, 0, 0]  # Red for NCR/NET
        pred_colored[pred_seg == 2] = [0, 1, 0]  # Green for ED
        pred_colored[pred_seg == 3] = [0, 0, 1]  # Blue for ET
        
        # Use FLAIR as background
        flair = data['flair']
        
        views = ['Axial', 'Coronal', 'Sagittal']
        slices = [mid_slices[0], mid_slices[1], mid_slices[2]]
        
        for i, (view, slice_idx) in enumerate(zip(views, slices)):
            if i == 0:  # Axial
                bg = flair[slice_idx, :, :].T
                gt = gt_colored[slice_idx, :, :].T
                pred = pred_colored[slice_idx, :, :].T
            elif i == 1:  # Coronal
                bg = flair[:, slice_idx, :].T
                gt = gt_colored[:, slice_idx, :].T
                pred = pred_colored[:, slice_idx, :].T
            else:  # Sagittal
                bg = flair[:, :, slice_idx].T
                gt = gt_colored[:, :, slice_idx].T
                pred = pred_colored[:, :, slice_idx].T
            
            # Ground truth
            axes[i, 0].imshow(bg, cmap='gray', origin='lower')
            axes[i, 0].imshow(gt, alpha=0.7, origin='lower')
            axes[i, 0].set_title(f'Ground Truth - {view}')
            axes[i, 0].axis('off')
            
            # Prediction
            axes[i, 1].imshow(bg, cmap='gray', origin='lower')
            axes[i, 1].imshow(pred, alpha=0.7, origin='lower')
            axes[i, 1].set_title(f'Prediction - {view}')
            axes[i, 1].axis('off')
            
            # Difference
            diff = np.abs(gt_seg - pred_seg)
            if i == 0:
                diff_slice = diff[slice_idx, :, :].T
            elif i == 1:
                diff_slice = diff[:, slice_idx, :].T
            else:
                diff_slice = diff[:, :, slice_idx].T
            
            axes[i, 2].imshow(bg, cmap='gray', origin='lower')
            axes[i, 2].imshow(diff_slice, alpha=0.7, cmap='Reds', origin='lower')
            axes[i, 2].set_title(f'Difference - {view}')
            axes[i, 2].axis('off')
            
            # Overlay comparison
            axes[i, 3].imshow(bg, cmap='gray', origin='lower')
            axes[i, 3].imshow(gt, alpha=0.5, origin='lower')
            axes[i, 3].imshow(pred, alpha=0.5, origin='lower')
            axes[i, 3].set_title(f'Overlay - {view}')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    plt.show()


def compare_models_performance(model_names: List[str], save_path: Optional[str] = None):
    """Compare training performance across multiple models"""
    import json
    import glob
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, model_name in enumerate(model_names):
        # Find history file
        history_pattern = os.path.join(paths.logs, f"training_history_{model_name}.json")
        history_files = glob.glob(history_pattern)
        
        if not history_files:
            print(f"No history file found for {model_name}")
            continue
        
        history_path = history_files[0]
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors[i % len(colors)]
        
        # Plot loss curves
        axes[0, 0].plot(epochs, history['train_loss'], label=f'{model_name} (Train)', 
                       color=color, linestyle='-', alpha=0.7)
        axes[0, 0].plot(epochs, history['val_loss'], label=f'{model_name} (Val)', 
                       color=color, linestyle='--', alpha=0.7)
        
        # Plot dice scores
        axes[0, 1].plot(epochs, history['dice'], label=model_name, color=color)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Comparison')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score Comparison')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True)
    
    # Performance summary table
    summary_data = []
    for model_name in model_names:
        history_pattern = os.path.join(paths.logs, f"training_history_{model_name}.json")
        history_files = glob.glob(history_pattern)
        
        if history_files:
            with open(history_files[0], 'r') as f:
                history = json.load(f)
            
            best_dice = max(history['dice'])
            final_dice = history['dice'][-1]
            best_epoch = history['dice'].index(best_dice) + 1
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            
            summary_data.append([
                model_name, f"{best_dice:.4f}", f"{final_dice:.4f}", 
                str(best_epoch), f"{final_train_loss:.4f}", f"{final_val_loss:.4f}"
            ])
    
    # Create summary table
    if summary_data:
        table_data = [['Model', 'Best Dice', 'Final Dice', 'Best Epoch', 'Final Train Loss', 'Final Val Loss']]
        table_data.extend(summary_data)
        
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=table_data[1:], colLabels=table_data[0], 
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 0].set_title('Performance Summary')
    
    # Best model comparison
    if summary_data:
        models = [row[0] for row in summary_data]
        best_dices = [float(row[1]) for row in summary_data]
        
        bars = axes[1, 1].bar(models, best_dices, color=colors[:len(models)])
        axes[1, 1].set_ylabel('Best Dice Score')
        axes[1, 1].set_title('Best Dice Score Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, best_dices):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization tools ready!")
    print("Use visualize_data_sample() to inspect data before training")
    print("Use visualize_training_history() to monitor training progress")
    print("Use visualize_predictions() to see model predictions")
    print("Use compare_models_performance() to compare multiple models")
