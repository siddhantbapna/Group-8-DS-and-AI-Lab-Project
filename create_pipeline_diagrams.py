#!/usr/bin/env python3
"""
Create visual pipeline diagrams with images for BraTS2023 brain MRI segmentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path
import os

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_preprocessing_flow_diagram():
    """Create preprocessing pipeline flow diagram"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Brain MRI Preprocessing Pipeline Flow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input data
    input_box = FancyBboxPatch((0.5, 9.5), 1.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 10.25, 'Raw Data\n• T1N.nii.gz\n• T1C.nii.gz\n• T2W.nii.gz\n• T2F.nii.gz\n• SEG.nii.gz', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Preprocessing steps
    steps = [
        (3, 9.5, 'LoadImaged\n• Load NIfTI\n• 240×240×155'),
        (5, 9.5, 'EnsureChannelFirstd\n• Add channel dim\n• 1×240×240×155'),
        (7, 9.5, 'Spacingd\n• Resample 1mm³\n• 1×240×240×155'),
        (9, 9.5, 'ScaleIntensityRanged\n• Normalize [0,1]\n• 1×240×240×155'),
        (1, 7, 'CropForegroundd\n• Remove background\n• 1×200×200×140'),
        (3, 7, 'Resized\n• Resize to 128³\n• 1×128×128×128'),
        (5, 7, 'ConvertToMultiChannel\n• BraTS labels\n• 3×128×128×128'),
        (7, 7, 'EnsureTyped\n• float32\n• Ready for training'),
    ]
    
    colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 
              'lightcyan', 'lightgray', 'lightsteelblue', 'lightseagreen']
    
    for i, (x, y, text) in enumerate(steps):
        box = FancyBboxPatch((x-0.7, y-0.7), 1.4, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    arrows = [
        (1.5, 10.25, 2.3, 10.25),  # Input to LoadImaged
        (3.7, 10.25, 4.3, 10.25),  # LoadImaged to EnsureChannelFirstd
        (5.7, 10.25, 6.3, 10.25),  # EnsureChannelFirstd to Spacingd
        (7.7, 10.25, 8.3, 10.25),  # Spacingd to ScaleIntensityRanged
        (1.25, 8.8, 1.25, 8.2),    # ScaleIntensityRanged to CropForegroundd
        (1.7, 7, 2.3, 7),          # CropForegroundd to Resized
        (3.7, 7, 4.3, 7),          # Resized to ConvertToMultiChannel
        (5.7, 7, 6.3, 7),          # ConvertToMultiChannel to EnsureTyped
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Final output
    output_box = FancyBboxPatch((8.5, 6.5), 1.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(9.25, 7.25, 'Final Output\n• 4×128×128×128\n• 3×128×128×128\n• Ready for Training', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow to output
    ax.annotate('', xy=(8.5, 7), xytext=(7.7, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    plt.tight_layout()
    plt.savefig('docs/preprocessing_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_architecture_diagram():
    """Create model architecture comparison diagram"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
    
    # UNet (2D)
    ax1.set_title('UNet (2D)', fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    # UNet architecture
    unet_boxes = [
        (1, 6, 'Input\n4×128×128'),
        (1, 4, 'Conv 4→32\n128×128'),
        (1, 2, 'Conv 32→64\n64×64'),
        (5, 2, 'Conv 64→128\n32×32'),
        (5, 4, 'Conv 128→64\n64×64'),
        (5, 6, 'Conv 64→3\n128×128'),
    ]
    
    for x, y, text in unet_boxes:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightblue', edgecolor='black', linewidth=1)
        ax1.add_patch(box)
        ax1.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 3D UNet
    ax2.set_title('3D UNet', fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    # 3D UNet architecture
    unet3d_boxes = [
        (1, 6, 'Input\n4×128×128×128'),
        (1, 4, 'Conv3D 4→32\n128×128×128'),
        (1, 2, 'Conv3D 32→64\n64×64×64'),
        (5, 2, 'Conv3D 64→128\n32×32×32'),
        (5, 4, 'Conv3D 128→64\n64×64×64'),
        (5, 6, 'Conv3D 64→3\n128×128×128'),
    ]
    
    for x, y, text in unet3d_boxes:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightgreen', edgecolor='black', linewidth=1)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # ResUNet
    ax3.set_title('ResUNet', fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    
    # ResUNet architecture
    resunet_boxes = [
        (1, 6, 'Input\n4×128×128×128'),
        (1, 4, 'Residual Block\n4→32'),
        (1, 2, 'Residual Block\n32→64'),
        (5, 2, 'Residual Block\n64→128'),
        (5, 4, 'Residual Block\n128→64'),
        (5, 6, 'Residual Block\n64→3'),
    ]
    
    for x, y, text in resunet_boxes:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightcoral', edgecolor='black', linewidth=1)
        ax3.add_patch(box)
        ax3.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Attention UNet
    ax4.set_title('Attention UNet', fontweight='bold')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    
    # Attention UNet architecture
    attention_boxes = [
        (1, 6, 'Input\n4×128×128×128'),
        (1, 4, 'Conv3D 4→32\n128×128×128'),
        (1, 2, 'Conv3D 32→64\n64×64×64'),
        (5, 2, 'Attention Gate\nFocus Features'),
        (5, 4, 'Conv3D 64→32\n64×64×64'),
        (5, 6, 'Conv3D 32→3\n128×128×128'),
    ]
    
    for x, y, text in attention_boxes:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightyellow', edgecolor='black', linewidth=1)
        ax4.add_patch(box)
        ax4.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/model_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_pipeline_diagram():
    """Create training pipeline flow diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Training Pipeline Flow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Training components
    components = [
        (1, 8, 'DataLoader\n• Batch Size: 2\n• Shuffle: True\n• Workers: 4'),
        (3, 8, 'Model\n• Architecture\n• Parameters\n• Device'),
        (5, 8, 'Optimizer\n• AdamW\n• LR: 1e-4\n• Weight Decay'),
        (7, 8, 'Scheduler\n• Polynomial\n• Decay Rate'),
        (9, 8, 'Loss Function\n• Weighted Dice\n• BCE'),
        (2, 6, 'Training Loop\n• Forward Pass\n• Loss Compute\n• Backward Pass\n• Optimize'),
        (6, 6, 'Validation\n• Metrics\n• Logging\n• Checkpoint'),
        (4, 4, 'Trained Model\n• Checkpoints\n• Metrics\n• Logs'),
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
              'lightpink', 'lightcyan', 'lightgray', 'gold']
    
    for i, (x, y, text) in enumerate(components):
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrows = [
        (1.8, 8, 2.2, 8),  # DataLoader to Model
        (3.8, 8, 4.2, 8),  # Model to Optimizer
        (5.8, 8, 6.2, 8),  # Optimizer to Scheduler
        (7.8, 8, 8.2, 8),  # Scheduler to Loss Function
        (2, 7.2, 2, 6.8),  # Loss Function to Training Loop
        (6, 7.2, 6, 6.8),  # Model to Validation
        (2.8, 6, 3.2, 4.8), # Training Loop to Trained Model
        (5.2, 6, 4.8, 4.8), # Validation to Trained Model
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    plt.tight_layout()
    plt.savefig('docs/training_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_cross_validation_diagram():
    """Create cross-validation flow diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, '5-Fold Cross-Validation Flow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Dataset
    dataset_box = FancyBboxPatch((0.5, 8), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(dataset_box)
    ax.text(1.5, 8.5, 'Dataset\n1000 Patients', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # K-Fold Split
    split_box = FancyBboxPatch((3, 8), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(split_box)
    ax.text(4, 8.5, 'K-Fold Split\n5 Folds', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Folds
    folds = [
        (1, 6, 'Fold 0\nTrain: 800\nVal: 200'),
        (3, 6, 'Fold 1\nTrain: 800\nVal: 200'),
        (5, 6, 'Fold 2\nTrain: 800\nVal: 200'),
        (7, 6, 'Fold 3\nTrain: 800\nVal: 200'),
        (9, 6, 'Fold 4\nTrain: 800\nVal: 200'),
    ]
    
    colors = ['lightcoral', 'lightyellow', 'lightpink', 'lightcyan', 'lightgray']
    
    for i, (x, y, text) in enumerate(folds):
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Results
    results_box = FancyBboxPatch((4, 3), 4, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(results_box)
    ax.text(6, 3.5, 'Aggregated Results\n• Mean Dice Score\n• Standard Deviation\n• Best Model', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        (2.5, 8.5, 3, 8.5),  # Dataset to K-Fold Split
        (4, 8, 1, 6.8),       # K-Fold Split to Fold 0
        (4, 8, 3, 6.8),       # K-Fold Split to Fold 1
        (4, 8, 5, 6.8),       # K-Fold Split to Fold 2
        (4, 8, 7, 6.8),       # K-Fold Split to Fold 3
        (4, 8, 9, 6.8),       # K-Fold Split to Fold 4
        (1, 5.2, 4, 4),       # Fold 0 to Results
        (3, 5.2, 4, 4),       # Fold 1 to Results
        (5, 5.2, 6, 4),       # Fold 2 to Results
        (7, 5.2, 6, 4),       # Fold 3 to Results
        (9, 5.2, 8, 4),       # Fold 4 to Results
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    plt.tight_layout()
    plt.savefig('docs/cross_validation_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_memory_usage_diagram():
    """Create memory usage visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'GPU Memory Usage for Different Models (RTX 4070 8GB)', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Models and their memory usage
    models = [
        (1, 8, 'UNet\n2GB VRAM\nBatch Size: 4'),
        (3, 8, '3D UNet\n6GB VRAM\nBatch Size: 1'),
        (5, 8, 'ResUNet\n8GB VRAM\nBatch Size: 1'),
        (7, 8, 'Attention UNet\n12GB VRAM\nBatch Size: 1'),
        (9, 8, 'nnUNet\n16GB VRAM\nBatch Size: 1'),
        (11, 8, 'VNet\n10GB VRAM\nBatch Size: 1'),
    ]
    
    colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightpink', 'lightcyan']
    status = ['Optimal', 'Good', 'Max VRAM', 'OOM', 'OOM', 'OOM']
    
    for i, (x, y, text) in enumerate(models):
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Status
        ax.text(x, y-1.2, f'Status: {status[i]}', 
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='red' if status[i] == 'OOM' else 'green')
    
    # GPU limit line
    ax.axhline(y=6, xmin=0.1, xmax=0.9, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax.text(6, 6.2, 'RTX 4070 8GB Limit', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
    
    # Memory usage bars
    memory_values = [2, 6, 8, 12, 16, 10]
    for i, (x, y, text) in enumerate(models):
        # Memory bar
        bar_height = memory_values[i] / 16 * 3  # Scale to fit
        bar = FancyBboxPatch((x-0.6, 2), 1.2, bar_height, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors[i], edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(bar)
        ax.text(x, 2 + bar_height/2, f'{memory_values[i]}GB', 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Legend
    ax.text(6, 1, 'Memory Usage (GB)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/memory_usage_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_summary_diagram():
    """Create complete data flow summary diagram"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Complete Brain MRI Segmentation Data Flow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Main pipeline stages
    stages = [
        (2, 9, 'Raw Data\n• T1N, T1C, T2W, T2F\n• SEG\n• 240×240×155'),
        (5, 9, 'Preprocessing\n• Load & Resize\n• Normalize\n• Augment\n• Brain Mask'),
        (8, 9, 'Training\n• 6 Models\n• Cross-Validation\n• Metrics\n• Checkpoints'),
        (11, 9, 'Inference\n• Prediction\n• Post-processing\n• Analysis'),
        (14, 9, 'Results\n• Metrics\n• Visualizations\n• Reports'),
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'gold']
    
    for i, (x, y, text) in enumerate(stages):
        box = FancyBboxPatch((x-1, y-1), 2, 2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows between stages
    arrows = [
        (3, 9, 4, 9),   # Raw Data to Preprocessing
        (6, 9, 7, 9),   # Preprocessing to Training
        (9, 9, 10, 9),  # Training to Inference
        (12, 9, 13, 9), # Inference to Results
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    
    # Detailed preprocessing steps
    ax.text(5, 7, 'Preprocessing Details:', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    prep_steps = [
        (1, 6, 'LoadImaged'),
        (3, 6, 'Spacingd'),
        (5, 6, 'ScaleIntensity'),
        (7, 6, 'CropForeground'),
        (9, 6, 'Resized'),
    ]
    
    for x, y, text in prep_steps:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor='lightcyan', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Model details
    ax.text(8, 7, 'Model Details:', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    model_details = [
        (6, 5, 'UNet\n2GB VRAM'),
        (8, 5, '3D UNet\n6GB VRAM'),
        (10, 5, 'ResUNet\n8GB VRAM'),
    ]
    
    for x, y, text in model_details:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor='lightsteelblue', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Data transformations
    ax.text(8, 3, 'Data Transformations:', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    transformations = [
        (2, 2, '240×240×155\n(int16)'),
        (5, 2, '1×240×240×155\n(float32)'),
        (8, 2, '1×128×128×128\n(float32)'),
        (11, 2, '4×128×128×128\n(float32)'),
        (14, 2, '3×128×128×128\n(float32)'),
    ]
    
    for x, y, text in transformations:
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor='lightpink', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows for transformations
    trans_arrows = [
        (2.8, 2, 4.2, 2),  # 240×240×155 to 1×240×240×155
        (5.8, 2, 7.2, 2),  # 1×240×240×155 to 1×128×128×128
        (8.8, 2, 10.2, 2), # 1×128×128×128 to 4×128×128×128
        (11.8, 2, 13.2, 2), # 4×128×128×128 to 3×128×128×128
    ]
    
    for x1, y1, x2, y2 in trans_arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    plt.tight_layout()
    plt.savefig('docs/data_flow_summary_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all pipeline diagrams"""
    print("Creating visual pipeline diagrams...")
    
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    # Generate diagrams
    print("Creating preprocessing flow diagram...")
    create_preprocessing_flow_diagram()
    
    print("Creating model architecture diagram...")
    create_model_architecture_diagram()
    
    print("Creating training pipeline diagram...")
    create_training_pipeline_diagram()
    
    print("Creating cross-validation diagram...")
    create_cross_validation_diagram()
    
    print("Creating memory usage diagram...")
    create_memory_usage_diagram()
    
    print("Creating data flow summary diagram...")
    create_data_flow_summary_diagram()
    
    print("All pipeline diagrams generated successfully!")
    print("Diagrams saved in 'docs/' directory:")
    print("   - preprocessing_flow_diagram.png")
    print("   - model_architecture_diagram.png")
    print("   - training_pipeline_diagram.png")
    print("   - cross_validation_diagram.png")
    print("   - memory_usage_diagram.png")
    print("   - data_flow_summary_diagram.png")

if __name__ == "__main__":
    main()
