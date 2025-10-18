#!/usr/bin/env python3
"""
Generate visualizations and charts for the BraTS2023 brain MRI segmentation project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_model_comparison_chart():
    """Create model comparison chart"""
    models = ['UNet', '3D UNet', 'ResUNet', 'Attention UNet', 'nnUNet', 'VNet']
    parameters = [31, 19, 39, 42, 30, 65]  # in millions
    memory_gb = [2, 6, 8, 12, 16, 10]
    training_speed = [5, 4, 3, 2, 1, 2]  # 1-5 scale
    accuracy = [3, 4, 4, 5, 5, 4]  # 1-5 scale
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Charts', fontsize=16, fontweight='bold')
    
    # Parameters comparison
    bars1 = ax1.bar(models, parameters, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax1.set_title('Model Parameters (Millions)', fontweight='bold')
    ax1.set_ylabel('Parameters (M)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, param in zip(bars1, parameters):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{param}M', ha='center', va='bottom', fontweight='bold')
    
    # Memory usage
    bars2 = ax2.bar(models, memory_gb, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('GPU Memory Usage (GB)', fontweight='bold')
    ax2.set_ylabel('Memory (GB)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='RTX 4070 Limit')
    ax2.legend()
    for bar, mem in zip(bars2, memory_gb):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{mem}GB', ha='center', va='bottom', fontweight='bold')
    
    # Training speed
    bars3 = ax3.bar(models, training_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax3.set_title('Training Speed (1-5 Scale)', fontweight='bold')
    ax3.set_ylabel('Speed Rating')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 6)
    for bar, speed in zip(bars3, training_speed):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{speed}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy
    bars4 = ax4.bar(models, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax4.set_title('Expected Accuracy (1-5 Scale)', fontweight='bold')
    ax4.set_ylabel('Accuracy Rating')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 6)
    for bar, acc in zip(bars4, accuracy):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/model_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_time_chart():
    """Create training time comparison chart"""
    models = ['UNet', '3D UNet', 'ResUNet', 'Attention UNet', 'nnUNet', 'VNet']
    single_epoch = [2, 5, 7, 12, 20, 15]  # minutes
    hundred_epochs = [3.3, 8.3, 11.7, 20, 33.3, 25]  # hours
    cv_training = [16.5, 41.5, 58.5, 100, 166.5, 125]  # hours
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Training Time Comparison', fontsize=16, fontweight='bold')
    
    # Single epoch
    bars1 = ax1.bar(models, single_epoch, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax1.set_title('Single Epoch Time', fontweight='bold')
    ax1.set_ylabel('Time (minutes)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, time in zip(bars1, single_epoch):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{time}m', ha='center', va='bottom', fontweight='bold')
    
    # 100 epochs
    bars2 = ax2.bar(models, hundred_epochs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('100 Epochs Training Time', fontweight='bold')
    ax2.set_ylabel('Time (hours)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, time in zip(bars2, hundred_epochs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{time}h', ha='center', va='bottom', fontweight='bold')
    
    # Cross-validation
    bars3 = ax3.bar(models, cv_training, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax3.set_title('5-Fold Cross-Validation Time', fontweight='bold')
    ax3.set_ylabel('Time (hours)')
    ax3.tick_params(axis='x', rotation=45)
    for bar, time in zip(bars3, cv_training):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{time}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/training_time_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_functions_chart():
    """Create loss functions comparison chart"""
    loss_functions = ['Dice', 'Cross-Entropy', 'Dice+CE', 'Dice+BCE', 'Focal', 'Weighted Dice+BCE']
    dice_score = [5, 3, 4, 4, 3, 5]  # 1-5 scale
    stability = [3, 5, 4, 4, 3, 4]  # 1-5 scale
    imbalance_handling = [4, 2, 3, 4, 5, 5]  # 1-5 scale
    computational_cost = [4, 5, 3, 3, 3, 2]  # 1-5 scale (higher = more expensive)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Loss Functions Comparison', fontsize=16, fontweight='bold')
    
    # Dice score
    bars1 = ax1.bar(loss_functions, dice_score, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax1.set_title('Dice Score Performance', fontweight='bold')
    ax1.set_ylabel('Performance (1-5 Scale)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 6)
    for bar, score in zip(bars1, dice_score):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # Training stability
    bars2 = ax2.bar(loss_functions, stability, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('Training Stability', fontweight='bold')
    ax2.set_ylabel('Stability (1-5 Scale)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 6)
    for bar, stab in zip(bars2, stability):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{stab}', ha='center', va='bottom', fontweight='bold')
    
    # Class imbalance handling
    bars3 = ax3.bar(loss_functions, imbalance_handling, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax3.set_title('Class Imbalance Handling', fontweight='bold')
    ax3.set_ylabel('Imbalance Handling (1-5 Scale)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 6)
    for bar, imb in zip(bars3, imbalance_handling):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{imb}', ha='center', va='bottom', fontweight='bold')
    
    # Computational cost
    bars4 = ax4.bar(loss_functions, computational_cost, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax4.set_title('Computational Cost', fontweight='bold')
    ax4.set_ylabel('Cost (1-5 Scale, Higher = More Expensive)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 6)
    for bar, cost in zip(bars4, computational_cost):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{cost}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/loss_functions_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pipeline_comparison_chart():
    """Create pipeline comparison chart"""
    pipelines = ['Standard', 'Brain-Only\n(Intensity)', 'Brain-Only\n(Otsu)', 'Brain-Only\n(Adaptive)']
    processing_speed = [5, 4, 3, 2]  # 1-5 scale
    memory_usage = [4, 3, 3, 2]  # 1-5 scale (higher = more memory)
    accuracy = [3, 4, 5, 5]  # 1-5 scale
    robustness = [3, 2, 4, 5]  # 1-5 scale
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pipeline Comparison', fontsize=16, fontweight='bold')
    
    # Processing speed
    bars1 = ax1.bar(pipelines, processing_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Processing Speed', fontweight='bold')
    ax1.set_ylabel('Speed (1-5 Scale)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 6)
    for bar, speed in zip(bars1, processing_speed):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{speed}', ha='center', va='bottom', fontweight='bold')
    
    # Memory usage
    bars2 = ax2.bar(pipelines, memory_usage, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('Memory Usage', fontweight='bold')
    ax2.set_ylabel('Memory (1-5 Scale, Higher = More Memory)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 6)
    for bar, mem in zip(bars2, memory_usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mem}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy
    bars3 = ax3.bar(pipelines, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('Expected Accuracy', fontweight='bold')
    ax3.set_ylabel('Accuracy (1-5 Scale)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 6)
    for bar, acc in zip(bars3, accuracy):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc}', ha='center', va='bottom', fontweight='bold')
    
    # Robustness
    bars4 = ax4.bar(pipelines, robustness, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_title('Robustness', fontweight='bold')
    ax4.set_ylabel('Robustness (1-5 Scale)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 6)
    for bar, rob in zip(bars4, robustness):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{rob}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/pipeline_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_metrics_radar_chart():
    """Create radar chart for metrics comparison"""
    categories = ['Dice Score', 'Training\nStability', 'Class Imbalance\nHandling', 
                  'Boundary\nAccuracy', 'Computational\nEfficiency', 'Medical\nRelevance']
    
    # Model scores (1-5 scale)
    unet_scores = [5, 3, 4, 3, 5, 3]
    unet3d_scores = [4, 4, 4, 4, 4, 4]
    resunet_scores = [4, 5, 4, 4, 3, 4]
    attention_scores = [5, 4, 5, 5, 2, 5]
    nnunet_scores = [5, 4, 5, 5, 1, 5]
    vnet_scores = [4, 4, 4, 4, 2, 4]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Add scores for each model
    unet_scores += unet_scores[:1]
    unet3d_scores += unet3d_scores[:1]
    resunet_scores += resunet_scores[:1]
    attention_scores += attention_scores[:1]
    nnunet_scores += nnunet_scores[:1]
    vnet_scores += vnet_scores[:1]
    
    # Plot
    ax.plot(angles, unet_scores, 'o-', linewidth=2, label='UNet', color='#FF6B6B')
    ax.fill(angles, unet_scores, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, unet3d_scores, 'o-', linewidth=2, label='3D UNet', color='#4ECDC4')
    ax.fill(angles, unet3d_scores, alpha=0.25, color='#4ECDC4')
    
    ax.plot(angles, resunet_scores, 'o-', linewidth=2, label='ResUNet', color='#45B7D1')
    ax.fill(angles, resunet_scores, alpha=0.25, color='#45B7D1')
    
    ax.plot(angles, attention_scores, 'o-', linewidth=2, label='Attention UNet', color='#96CEB4')
    ax.fill(angles, attention_scores, alpha=0.25, color='#96CEB4')
    
    ax.plot(angles, nnunet_scores, 'o-', linewidth=2, label='nnUNet', color='#FFEAA7')
    ax.fill(angles, nnunet_scores, alpha=0.25, color='#FFEAA7')
    
    ax.plot(angles, vnet_scores, 'o-', linewidth=2, label='VNet', color='#DDA0DD')
    ax.fill(angles, vnet_scores, alpha=0.25, color='#DDA0DD')
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)
    
    # Add title and legend
    plt.title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('docs/metrics_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """Create data flow diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'BraTS2023 Brain MRI Segmentation Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Data flow boxes
    boxes = [
        (1, 6, 'Raw Data\n• T1N, T1C, T2W, T2F\n• Segmentation'),
        (3, 6, 'Preprocessing\n• Load & Resize\n• Normalize\n• Augment'),
        (5, 6, 'Training\n• 6 Models\n• Cross-Validation\n• Metrics'),
        (7, 6, 'Inference\n• Prediction\n• Post-processing\n• Analysis'),
        (9, 6, 'Results\n• Metrics\n• Visualizations\n• Reports')
    ]
    
    # Draw boxes
    for x, y, text in boxes:
        rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Draw arrows
    arrows = [(1.4, 6, 2.6, 6), (3.4, 6, 4.6, 6), (5.4, 6, 6.6, 6), (7.4, 6, 8.6, 6)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Add details
    ax.text(2, 4, '• LoadImaged\n• EnsureChannelFirstd\n• Spacingd\n• ScaleIntensityRanged\n• CropForegroundd\n• Resized\n• ConvertToMultiChannelBasedOnBratsClassesd', 
            fontsize=9, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax.text(5, 4, '• UNet\n• 3D UNet\n• ResUNet\n• Attention UNet\n• nnUNet\n• VNet', 
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax.text(8, 4, '• Dice Score\n• Accuracy\n• Hausdorff Distance\n• Sensitivity\n• Specificity', 
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    plt.tight_layout()
    plt.savefig('docs/data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_memory_usage_chart():
    """Create memory usage visualization"""
    models = ['UNet', '3D UNet', 'ResUNet', 'Attention UNet', 'nnUNet', 'VNet']
    memory_usage = [2, 6, 8, 12, 16, 10]  # GB
    gpu_limit = 8  # RTX 4070 8GB limit
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars
    bars = ax.bar(models, memory_usage, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    
    # Add GPU limit line
    ax.axhline(y=gpu_limit, color='red', linestyle='--', linewidth=3, 
               label=f'RTX 4070 8GB Limit ({gpu_limit}GB)', alpha=0.8)
    
    # Add text on bars
    for bar, mem in zip(bars, memory_usage):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
                f'{mem}GB', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add warning for models exceeding limit
        if mem > gpu_limit:
            ax.text(bar.get_x() + bar.get_width()/2, height/2, 
                    'OOM!', ha='center', va='center', fontweight='bold', 
                    fontsize=14, color='red')
    
    ax.set_title('GPU Memory Usage by Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Memory Usage (GB)', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/memory_usage_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all visualizations"""
    print("Generating visualizations for BraTS2023 project...")
    
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    # Generate charts
    print("Creating model comparison chart...")
    create_model_comparison_chart()
    
    print("Creating training time chart...")
    create_training_time_chart()
    
    print("Creating loss functions chart...")
    create_loss_functions_chart()
    
    print("Creating pipeline comparison chart...")
    create_pipeline_comparison_chart()
    
    print("Creating metrics radar chart...")
    create_metrics_radar_chart()
    
    print("Creating data flow diagram...")
    create_data_flow_diagram()
    
    print("Creating memory usage chart...")
    create_memory_usage_chart()
    
    print("All visualizations generated successfully!")
    print("Charts saved in 'docs/' directory:")
    print("   - model_comparison_chart.png")
    print("   - training_time_chart.png")
    print("   - loss_functions_chart.png")
    print("   - pipeline_comparison_chart.png")
    print("   - metrics_radar_chart.png")
    print("   - data_flow_diagram.png")
    print("   - memory_usage_chart.png")

if __name__ == "__main__":
    main()
