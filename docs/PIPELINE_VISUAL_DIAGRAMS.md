# Pipeline Visual Diagrams and Data Flow

## Overview
This document provides visual representations of data flow through our BraTS2023 brain MRI segmentation pipeline, including preprocessing, training, and model architectures.

## 1. Overall Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BRAIN MRI SEGMENTATION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw Data  │───▶│Preprocessing│───▶│   Training  │───▶│  Inference  │───▶│  Results    │
│             │    │             │    │             │    │             │    │             │
│ • T1N       │    │ • Loading   │    │ • 6 Models  │    │ • Prediction│    │ • Metrics   │
│ • T1C       │    │ • Resizing  │    │ • CV        │    │ • Post-proc │    │ • Visuals   │
│ • T2W       │    │ • Normalize │    │ • Checkpoint│    │ • Analysis  │    │ • Reports   │
│ • T2F       │    │ • Augment   │    │ • Logging   │    │ • Export    │    │             │
│ • SEG       │    │ • Brain Mask│    │ • Metrics   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 2. Data Preprocessing Pipeline

### 2.1 Standard Preprocessing Flow
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PREPROCESSING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Input Data (NIfTI files)
         │
         ▼
┌─────────────────┐
│   LoadImaged    │ ◄── Load T1N, T1C, T2W, T2F, SEG files
│                 │
│ • T1N: 240³     │
│ • T1C: 240³     │
│ • T2W: 240³     │
│ • T2F: 240³     │
│ • SEG: 240³     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│EnsureChannelFirst│ ◄── Add channel dimension
│                 │
│ • T1N: 1×240³   │
│ • T1C: 1×240³   │
│ • T2W: 1×240³   │
│ • T2F: 1×240³   │
│ • SEG: 1×240³   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│    Spacingd     │ ◄── Resample to 1mm³ voxels
│                 │
│ • All: 1×240³   │
│ • Spacing: 1mm³ │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ScaleIntensityRng│ ◄── Normalize to [0,1]
│                 │
│ • Range: [0,1]  │
│ • Clip: True    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ CropForegroundd │ ◄── Remove background
│                 │
│ • Margin: 10px  │
│ • Source: T1N   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│     Resized     │ ◄── Resize to 128³
│                 │
│ • Size: 128³    │
│ • Mode: area    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ConvertToMultiCh │ ◄── Convert BraTS labels
│                 │
│ • 0→0 (BG)      │
│ • 1→1 (NCR/NET) │
│ • 2→1 (ED)      │
│ • 3→2 (ET)      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   EnsureTyped   │ ◄── Convert to float32
│                 │
│ • dtype: float32│
└─────────────────┘
         │
         ▼
    Final Output
    • 4×128³ (modalities)
    • 3×128³ (segmentation)
```

### 2.2 Brain-Only Preprocessing Flow
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BRAIN-ONLY PREPROCESSING                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Standard Preprocessing
         │
         ▼
┌─────────────────┐
│ BrainOnlyTransform│ ◄── Apply brain mask
│                 │
│ • Method: Otsu  │
│ • BG Weight: 0.05│
│ • Foreground: T │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Brain Mask     │ ◄── Generated mask
│                 │
│ • T1N > threshold│
│ • Morphological │
│ • Binary mask   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Apply Mask     │ ◄── Mask all modalities
│                 │
│ • Mask T1N      │
│ • Mask T1C      │
│ • Mask T2W      │
│ • Mask T2F      │
│ • Mask SEG      │
└─────────────────┘
         │
         ▼
    Masked Output
    • Focus on brain tissue
    • Reduced background
    • Better training
```

## 3. Data Augmentation Pipeline

### 3.1 Training Augmentation
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA AUGMENTATION                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Preprocessed Data
         │
         ▼
┌─────────────────┐
│   RandFlipd     │ ◄── Random flipping
│                 │
│ • Prob: 0.5     │
│ • Axis: 0       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  RandRotate90d  │ ◄── Random rotation
│                 │
│ • Prob: 0.5     │
│ • Max k: 3      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│RandShiftIntensity│ ◄── Intensity shift
│                 │
│ • Offset: 0.1   │
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ RandGaussianNoise│ ◄── Add noise
│                 │
│ • Std: 0.01     │
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│RandGaussianSmooth│ ◄── Gaussian blur
│                 │
│ • Sigma: 0.5-1.0│
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│RandAdjustContrast│ ◄── Contrast adjustment
│                 │
│ • Gamma: 0.8-1.2│
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
    Augmented Data
    • Increased diversity
    • Better generalization
    • Robust training
```

## 4. Model Architecture Diagrams

### 4.1 UNet Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  UNET (2D)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128 (4 modalities)
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Conv 4→32       │
│ Conv 32→64      │
│ Conv 64→128     │
│ Conv 128→256    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Conv 256→128    │
│ Conv 128→64     │
│ Conv 64→32      │
│ Conv 32→3       │
└─────────────────┘
         │
         ▼
Output: 3×128×128 (3 classes)
```

### 4.2 3D UNet Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                3D UNET                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128×128 (4 modalities, 3D)
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Conv3D 4→32     │
│ Conv3D 32→64    │
│ Conv3D 64→128   │
│ Conv3D 128→256  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Conv3D 256→128  │
│ Conv3D 128→64   │
│ Conv3D 64→32    │
│ Conv3D 32→3     │
└─────────────────┘
         │
         ▼
Output: 3×128×128×128 (3 classes, 3D)
```

### 4.3 ResUNet Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 RESUNET                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128×128
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Residual Block  │
│ • Conv3D 4→32   │
│ • Conv3D 32→32  │
│ • Skip Connection│
│                 │
│ Residual Block  │
│ • Conv3D 32→64  │
│ • Conv3D 64→64  │
│ • Skip Connection│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Residual Block  │
│ • Conv3D 64→32  │
│ • Conv3D 32→32  │
│ • Skip Connection│
│                 │
│ Residual Block  │
│ • Conv3D 32→3   │
│ • Conv3D 3→3    │
│ • Skip Connection│
└─────────────────┘
         │
         ▼
Output: 3×128×128×128
```

### 4.4 Attention UNet Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ATTENTION UNET                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128×128
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Conv3D 4→32     │
│ Conv3D 32→64    │
│ Conv3D 64→128   │
│ Conv3D 128→256  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Attention Gates │
│                 │
│ • Gate 1        │
│ • Gate 2        │
│ • Gate 3        │
│ • Gate 4        │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Conv3D 256→128  │
│ Conv3D 128→64   │
│ Conv3D 64→32    │
│ Conv3D 32→3     │
└─────────────────┘
         │
         ▼
Output: 3×128×128×128
```

## 5. Training Pipeline Flow

### 5.1 Single Model Training
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Data Loading
         │
         ▼
┌─────────────────┐
│   DataLoader    │ ◄── Load batches
│                 │
│ • Batch Size: 2 │
│ • Shuffle: True │
│ • Workers: 4    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Model         │ ◄── Initialize model
│                 │
│ • Architecture  │
│ • Parameters    │
│ • Device        │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Optimizer     │ ◄── Setup optimizer
│                 │
│ • AdamW         │
│ • LR: 1e-4      │
│ • Weight Decay  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Scheduler     │ ◄── Setup scheduler
│                 │
│ • Polynomial    │
│ • Decay Rate    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Loss Function │ ◄── Setup loss
│                 │
│ • Weighted Dice │
│ • BCE           │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Training Loop │ ◄── Train model
│                 │
│ • Forward Pass  │
│ • Loss Compute  │
│ • Backward Pass │
│ • Optimize      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Validation    │ ◄── Validate model
│                 │
│ • Metrics       │
│ • Logging       │
│ • Checkpoint    │
└─────────────────┘
         │
         ▼
    Trained Model
    • Checkpoints
    • Metrics
    • Logs
```

### 5.2 Cross-Validation Training
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CROSS-VALIDATION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────┘

Dataset
         │
         ▼
┌─────────────────┐
│   K-Fold Split  │ ◄── Split data
│                 │
│ • 5 Folds       │
│ • Stratified    │
│ • Random Seed   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 0        │ ◄── Train on folds 1,2,3,4
│                 │    Validate on fold 0
│ • Train: 80%    │
│ • Val: 20%      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 1        │ ◄── Train on folds 0,2,3,4
│                 │    Validate on fold 1
│ • Train: 80%    │
│ • Val: 20%      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 2        │ ◄── Train on folds 0,1,3,4
│                 │    Validate on fold 2
│ • Train: 80%    │
│ • Val: 20%      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 3        │ ◄── Train on folds 0,1,2,4
│                 │    Validate on fold 3
│ • Train: 80%    │
│ • Val: 20%      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 4        │ ◄── Train on folds 0,1,2,3
│                 │    Validate on fold 4
│ • Train: 80%    │
│ • Val: 20%      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Results       │ ◄── Aggregate results
│                 │
│ • Mean Dice     │
│ • Std Dice      │
│ • Best Model    │
└─────────────────┘
```

## 6. Model Comparison Chart

### 6.1 Model Characteristics
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL COMPARISON                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

Model           │ Parameters │ Memory │ Speed │ Accuracy │ Best For
────────────────┼────────────┼────────┼───────┼──────────┼─────────────────────
UNet (2D)       │    31M     │  2GB   │ ⭐⭐⭐⭐⭐│   ⭐⭐⭐   │ Quick prototyping
3D UNet         │    19M     │  6GB   │ ⭐⭐⭐⭐ │  ⭐⭐⭐⭐  │ Standard 3D
ResUNet         │    39M     │  8GB   │ ⭐⭐⭐  │  ⭐⭐⭐⭐  │ Stable training
Attention UNet  │    42M     │ 12GB   │  ⭐⭐  │ ⭐⭐⭐⭐⭐ │ High accuracy
nnUNet          │    30M     │ 16GB   │   ⭐  │ ⭐⭐⭐⭐⭐ │ State-of-the-art
VNet            │    65M     │ 10GB   │  ⭐⭐  │  ⭐⭐⭐⭐  │ Volumetric data
```

### 6.2 Training Time Comparison
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING TIME COMPARISON                             │
└─────────────────────────────────────────────────────────────────────────────────┘

Model           │ Single Epoch │ 100 Epochs │ 5-Fold CV │ Memory Usage
────────────────┼──────────────┼────────────┼───────────┼─────────────
UNet (2D)       │    2 min     │  3.3 hours │ 16.5 hours│    2GB
3D UNet         │    5 min     │  8.3 hours │ 41.5 hours│    6GB
ResUNet         │    7 min     │ 11.7 hours │ 58.5 hours│    8GB
Attention UNet  │   12 min     │  20 hours  │ 100 hours │   12GB
nnUNet          │   20 min     │ 33.3 hours │166.5 hours│   16GB
VNet            │   15 min     │  25 hours  │ 125 hours │   10GB
```

## 7. Data Flow Visualization

### 7.1 Complete Data Flow
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Raw NIfTI Files
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │
│ 240×240×240 │    │ 240×240×240 │    │ 240×240×240 │    │ 240×240×240 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    │                   │                   │                   │
    └───────────────────┼───────────────────┼───────────────────┘
                        │                   │
                        ▼                   ▼
                ┌─────────────┐    ┌─────────────┐
                │ Preprocessing│    │ Segmentation│
                │             │    │             │
                │ • Load      │    │ • Load      │
                │ • Resize    │    │ • Resize    │
                │ • Normalize │    │ • Convert   │
                │ • Augment   │    │ • Mask      │
                └─────────────┘    └─────────────┘
                        │                   │
                        └───────────────────┘
                                │
                                ▼
                        ┌─────────────┐
                        │   Training  │
                        │             │
                        │ • 6 Models  │
                        │ • CV        │
                        │ • Metrics   │
                        │ • Logging   │
                        └─────────────┘
                                │
                                ▼
                        ┌─────────────┐
                        │   Results   │
                        │             │
                        │ • Metrics   │
                        │ • Checkpoints│
                        │ • Logs      │
                        │ • Reports   │
                        └─────────────┘
```

## 8. Memory Usage Visualization

### 8.1 GPU Memory Usage
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GPU MEMORY USAGE                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

RTX 4070 (12GB VRAM)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  UNet (2GB)     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  3D UNet (6GB)  ████████████████████████████████████████████████████████████████ │
│  ResUNet (8GB)  ████████████████████████████████████████████████████████████████ │
│  Attention (12GB)███████████████████████████████████████████████████████████████ │
│  nnUNet (16GB)  ████████████████████████████████████████████████████████████████ │
│  VNet (10GB)    ████████████████████████████████████████████████████████████████ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 9. Performance Metrics Visualization

### 9.1 Expected Performance
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXPECTED PERFORMANCE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

Model           │ Dice Score │ Training Time │ Memory │ Best Use Case
────────────────┼────────────┼───────────────┼────────┼─────────────────────
UNet (2D)       │   0.75-0.80│    3.3 hours  │  2GB   │ Quick prototyping
3D UNet         │   0.85-0.90│    8.3 hours  │  6GB   │ Standard 3D
ResUNet         │   0.86-0.91│   11.7 hours  │  8GB   │ Stable training
Attention UNet  │   0.88-0.92│    20 hours   │ 12GB   │ High accuracy
nnUNet          │   0.89-0.93│   33.3 hours  │ 16GB   │ State-of-the-art
VNet            │   0.87-0.91│    25 hours   │ 10GB   │ Volumetric data
```

## 10. Implementation Guide

### 10.1 Quick Start
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                QUICK START                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

1. Quick Test All Models
   ┌─────────────────┐
   │ python quick_train_all.py │
   └─────────────────┘
           │
           ▼
2. Full Training
   ┌─────────────────┐
   │ python train_all_models.py │
   └─────────────────┘
           │
           ▼
3. Best Model Selection
   ┌─────────────────┐
   │ python main.py --model unet3d --mode cv │
   └─────────────────┘
           │
           ▼
4. Production Deployment
   ┌─────────────────┐
   │ python main.py --mode inference │
   └─────────────────┘
```

### 10.2 Configuration Selection
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CONFIGURATION SELECTION                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Priority          │ Model      │ Loss Function    │ Pipeline
──────────────────┼────────────┼──────────────────┼─────────────────────────────
Best Balance      │ 3D UNet    │ Weighted Dice+BCE│ Brain-Only (Otsu)
Maximum Accuracy  │ Attention  │ Weighted Dice+BCE│ Brain-Only (Otsu)
Fastest Training  │ UNet       │ Dice Loss        │ Standard
State-of-the-Art  │ nnUNet     │ Weighted Dice+BCE│ Brain-Only (Otsu)
```

This comprehensive visual documentation provides a complete understanding of data flow, preprocessing, training, and model architectures in our BraTS2023 brain MRI segmentation project.
