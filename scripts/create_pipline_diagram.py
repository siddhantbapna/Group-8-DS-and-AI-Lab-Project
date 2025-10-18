#!/usr/bin/env python3
"""
Create detailed pipeline diagrams with data flow visualization
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
plt.rcParams['font.weight'] = 'bold'

def create_data_flow_pipeline():
    """Create comprehensive data flow pipeline diagram"""
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(10, 11.5, 'BraTS2023 Brain MRI Segmentation - Complete Data Flow Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Define colors
    colors = {
        'data': '#E8F4FD',
        'preprocessing': '#FFF2CC',
        'training': '#D5E8D4',
        'model': '#F8CECC',
        'output': '#E1D5E7'
    }
    
    # Stage 1: Raw Data
    stage1_box = FancyBboxPatch((0.5, 9), 3, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['data'], edgecolor='black', linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(2, 10, 'Raw Data\n\n• T1N: 240³\n• T1C: 240³\n• T2W: 240³\n• T2F: 240³\n• SEG: 240³', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stage 2: Data Loading
    stage2_box = FancyBboxPatch((4.5, 9), 3, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(6, 10, 'Data Loading\n\n• LoadImaged\n• EnsureChannelFirstd\n• File I/O', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stage 3: Preprocessing
    stage3_box = FancyBboxPatch((8.5, 9), 3, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(10, 10, 'Preprocessing\n\n• Spacingd (1mm³)\n• ScaleIntensityRanged\n• CropForegroundd', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stage 4: Resizing & Normalization
    stage4_box = FancyBboxPatch((12.5, 9), 3, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(14, 10, 'Resizing &\nNormalization\n\n• Resized (128³)\n• ConvertToMultiChannel\n• EnsureTyped', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stage 5: Brain Masking (Optional)
    stage5_box = FancyBboxPatch((16.5, 9), 3, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(stage5_box)
    ax.text(18, 10, 'Brain Masking\n(Optional)\n\n• Otsu Thresholding\n• Morphological Ops\n• Mask Application', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows between stages
    arrows = [(3.5, 10, 4.5, 10), (7.5, 10, 8.5, 10), (11.5, 10, 12.5, 10), (15.5, 10, 16.5, 10)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Training Pipeline
    ax.text(10, 8, 'TRAINING PIPELINE', fontsize=16, fontweight='bold', ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # Data Augmentation
    aug_box = FancyBboxPatch((1, 6), 4, 1.5, boxstyle="round,pad=0.1", 
                            facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(aug_box)
    ax.text(3, 6.75, 'Data Augmentation\n• RandFlipd, RandRotate90d\n• RandGaussianNoised, RandShiftIntensityd', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Model Selection
    model_box = FancyBboxPatch((6, 6), 4, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['model'], edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(8, 6.75, 'Model Selection\n• UNet, 3D UNet, ResUNet\n• Attention UNet, nnUNet, VNet', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Training Loop
    train_box = FancyBboxPatch((11, 6), 4, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['training'], edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(13, 6.75, 'Training Loop\n• Forward Pass, Loss Compute\n• Backward Pass, Optimize', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Validation
    val_box = FancyBboxPatch((16, 6), 3, 1.5, boxstyle="round,pad=0.1", 
                            facecolor=colors['training'], edgecolor='black', linewidth=2)
    ax.add_patch(val_box)
    ax.text(17.5, 6.75, 'Validation\n• Metrics Compute\n• Checkpoint Save', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows for training pipeline
    train_arrows = [(5, 6.75, 6, 6.75), (10, 6.75, 11, 6.75), (15, 6.75, 16, 6.75)]
    for x1, y1, x2, y2 in train_arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Data Flow Details
    ax.text(10, 5, 'DATA FLOW DETAILS', fontsize=16, fontweight='bold', ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Input dimensions
    input_box = FancyBboxPatch((1, 3), 3, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(2.5, 3.75, 'Input:\n4×240×240×240\n(4 modalities)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Processed dimensions
    processed_box = FancyBboxPatch((5, 3), 3, 1.5, boxstyle="round,pad=0.1", 
                                  facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(processed_box)
    ax.text(6.5, 3.75, 'Processed:\n4×128×128×128\n(4 modalities)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Segmentation dimensions
    seg_box = FancyBboxPatch((9, 3), 3, 1.5, boxstyle="round,pad=0.1", 
                            facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(seg_box)
    ax.text(10.5, 3.75, 'Segmentation:\n3×128×128×128\n(3 classes)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Memory usage
    memory_box = FancyBboxPatch((13, 3), 3, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='black', linewidth=1)
    ax.add_patch(memory_box)
    ax.text(14.5, 3.75, 'Memory Usage:\n2-16GB VRAM\n(depends on model)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((17, 3), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='black', linewidth=1)
    ax.add_patch(output_box)
    ax.text(18, 3.75, 'Output:\nPredictions\nMetrics', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows for data flow
    data_arrows = [(4, 3.75, 5, 3.75), (8, 3.75, 9, 3.75), (12, 3.75, 13, 3.75), (16, 3.75, 17, 3.7