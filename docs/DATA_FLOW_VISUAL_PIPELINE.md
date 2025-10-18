# Data Flow Visual Pipeline - Complete Diagrammatic Representation

## Overview
This document provides comprehensive visual representations of data flow through preprocessing, training, and inference pipelines for different models in our BraTS2023 brain MRI segmentation project.

## 1. Complete Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    BRAIN MRI SEGMENTATION PIPELINE                              │
│                                         DATA FLOW DIAGRAM                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw Data  │───▶│Preprocessing│───▶│   Training  │───▶│  Inference  │───▶│  Results    │
│             │    │             │    │             │    │             │    │             │
│ • T1N.nii.gz│    │ • Loading   │    │ • 6 Models  │    │ • Prediction│    │ • Metrics   │
│ • T1C.nii.gz│    │ • Resizing  │    │ • CV        │    │ • Post-proc │    │ • Visuals   │
│ • T2W.nii.gz│    │ • Normalize │    │ • Checkpoint│    │ • Analysis  │    │ • Reports   │
│ • T2F.nii.gz│    │ • Augment   │    │ • Logging   │    │ • Export    │    │             │
│ • SEG.nii.gz│    │ • Brain Mask│    │ • Metrics   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 2. Detailed Preprocessing Pipeline Flow

### 2.1 Input Data Structure
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    INPUT DATA STRUCTURE                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Patient Directory: BraTS-GLI-00000-000/
├── BraTS-GLI-00000-000-t1n.nii.gz    (T1-weighted native)
├── BraTS-GLI-00000-000-t1c.nii.gz    (T1-weighted contrast-enhanced)
├── BraTS-GLI-00000-000-t2w.nii.gz    (T2-weighted)
├── BraTS-GLI-00000-000-t2f.nii.gz    (T2-FLAIR)
└── BraTS-GLI-00000-000-seg.nii.gz    (Segmentation mask)

Original Dimensions: 240×240×155 voxels
Original Spacing: ~1.0×1.0×1.0 mm³
Data Type: 16-bit integers
```

### 2.2 Preprocessing Step-by-Step Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PREPROCESSING PIPELINE FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Step 1: LoadImaged
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 240×240×155 │    │ 240×240×155 │    │ 240×240×155 │    │ 240×240×155 │    │ 240×240×155 │
│ (int16)     │    │ (int16)     │    │ (int16)     │    │ (int16)     │    │ (int16)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Step 2: EnsureChannelFirstd
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│
│ (int16)     │    │ (int16)     │    │ (int16)     │    │ (int16)     │    │ (int16)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Step 3: Spacingd (Resample to 1mm³)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│
│ (float32)   │    │ (float32)   │    │ (float32)   │    │ (float32)   │    │ (float32)   │
│ Spacing: 1mm³│   │ Spacing: 1mm³│   │ Spacing: 1mm³│   │ Spacing: 1mm³│   │ Spacing: 1mm³│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Step 4: ScaleIntensityRanged (Normalize to [0,1])
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│   │ 1×240×240×155│
│ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Step 5: CropForegroundd (Remove background)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 1×200×200×140│   │ 1×200×200×140│   │ 1×200×200×140│   │ 1×200×200×140│   │ 1×200×200×140│
│ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Step 6: Resized (Resize to 128³)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 1×128×128×128│   │ 1×128×128×128│   │ 1×128×128×128│   │ 1×128×128×128│   │ 1×128×128×128│
│ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Step 7: ConvertToMultiChannelBasedOnBratsClassesd
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │    │   SEG       │
│ 1×128×128×128│   │ 1×128×128×128│   │ 1×128×128×128│   │ 1×128×128×128│   │ 3×128×128×128│
│ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0.0, 1.0]  │    │ [0, 1]      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┼───────────────────┘
                             │                   │                   │
                             ▼                   ▼                   ▼
Final Output: Stack Modalities
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FINAL PREPROCESSED DATA                                      │
│                                                                                                 │
│ Input: 4×128×128×128 (4 modalities stacked)                                                    │
│ Target: 3×128×128×128 (3 segmentation classes)                                                 │
│                                                                                                 │
│ Modalities: [T1N, T1C, T2W, T2F]                                                               │
│ Classes: [Background, NCR/NET/ED, ET]                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Brain-Only Preprocessing Flow

### 3.1 Brain Mask Generation
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              BRAIN-ONLY PREPROCESSING FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

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
│  Generation     │
│                 │
│ 1. T1N > threshold│
│ 2. Morphological │
│ 3. Binary mask   │
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

### 3.2 Brain Mask Methods Comparison
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              BRAIN MASK GENERATION METHODS                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Method 1: Intensity-Based
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │───▶│ Threshold   │───▶│ Brain Mask  │
│ Image       │    │ > 5th %ile  │    │ (Binary)    │
└─────────────┘    └─────────────┘    └─────────────┘
   240×240×155        240×240×155        240×240×155

Method 2: Otsu Thresholding
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │───▶│ Otsu        │───▶│ Brain Mask  │
│ Image       │    │ Threshold   │    │ (Binary)    │
└─────────────┘    └─────────────┘    └─────────────┘
   240×240×155        240×240×155        240×240×155

Method 3: Adaptive Thresholding
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │───▶│ Adaptive    │───▶│ Brain Mask  │
│ Image       │    │ Threshold   │    │ (Binary)    │
└─────────────┘    └─────────────┘    └─────────────┘
   240×240×155        240×240×155        240×240×155
```

## 4. Data Augmentation Pipeline Flow

### 4.1 Training Augmentation Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA AUGMENTATION PIPELINE                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Preprocessed Data (4×128×128×128)
         │
         ▼
┌─────────────────┐
│   RandFlipd     │ ◄── Random flipping (50% prob)
│                 │
│ • Prob: 0.5     │
│ • Axis: 0       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  RandRotate90d  │ ◄── Random rotation (50% prob)
│                 │
│ • Prob: 0.5     │
│ • Max k: 3      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│RandShiftIntensity│ ◄── Intensity shift (50% prob)
│                 │
│ • Offset: 0.1   │
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ RandGaussianNoise│ ◄── Add noise (50% prob)
│                 │
│ • Std: 0.01     │
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│RandGaussianSmooth│ ◄── Gaussian blur (50% prob)
│                 │
│ • Sigma: 0.5-1.0│
│ • Prob: 0.5     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│RandAdjustContrast│ ◄── Contrast adjustment (50% prob)
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

## 5. Model Architecture Data Flow

### 5.1 UNet (2D) Data Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    UNET (2D) DATA FLOW                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128 (4 modalities, 2D slices)
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Conv 4→32       │ ◄── 128×128
│ Conv 32→64      │ ◄── 64×64
│ Conv 64→128     │ ◄── 32×32
│ Conv 128→256    │ ◄── 16×16
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Conv 256→128    │ ◄── 32×32
│ Conv 128→64     │ ◄── 64×64
│ Conv 64→32      │ ◄── 128×128
│ Conv 32→3       │ ◄── 128×128
└─────────────────┘
         │
         ▼
Output: 3×128×128 (3 classes, 2D)
```

### 5.2 3D UNet Data Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   3D UNET DATA FLOW                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128×128 (4 modalities, 3D volume)
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Conv3D 4→32     │ ◄── 128×128×128
│ Conv3D 32→64    │ ◄── 64×64×64
│ Conv3D 64→128   │ ◄── 32×32×32
│ Conv3D 128→256  │ ◄── 16×16×16
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Conv3D 256→128  │ ◄── 32×32×32
│ Conv3D 128→64   │ ◄── 64×64×64
│ Conv3D 64→32    │ ◄── 128×128×128
│ Conv3D 32→3     │ ◄── 128×128×128
└─────────────────┘
         │
         ▼
Output: 3×128×128×128 (3 classes, 3D)
```

### 5.3 ResUNet Data Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESUNET DATA FLOW                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128×128
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Residual Block  │ ◄── Conv3D + Skip Connection
│ • Conv3D 4→32   │
│ • Conv3D 32→32  │
│ • Skip Connection│
│                 │
│ Residual Block  │ ◄── Conv3D + Skip Connection
│ • Conv3D 32→64  │
│ • Conv3D 64→64  │
│ • Skip Connection│
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Residual Block  │ ◄── Conv3D + Skip Connection
│ • Conv3D 64→32  │
│ • Conv3D 32→32  │
│ • Skip Connection│
│                 │
│ Residual Block  │ ◄── Conv3D + Skip Connection
│ • Conv3D 32→3   │
│ • Conv3D 3→3    │
│ • Skip Connection│
└─────────────────┘
         │
         ▼
Output: 3×128×128×128
```

### 5.4 Attention UNet Data Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ATTENTION UNET DATA FLOW                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Input: 4×128×128×128
         │
         ▼
┌─────────────────┐
│   Encoder       │
│                 │
│ Conv3D 4→32     │ ◄── 128×128×128
│ Conv3D 32→64    │ ◄── 64×64×64
│ Conv3D 64→128   │ ◄── 32×32×32
│ Conv3D 128→256  │ ◄── 16×16×16
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Attention Gates │ ◄── Focus on relevant features
│                 │
│ • Gate 1        │ ◄── 32×32×32
│ • Gate 2        │ ◄── 64×64×64
│ • Gate 3        │ ◄── 128×128×128
│ • Gate 4        │ ◄── 128×128×128
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Decoder       │
│                 │
│ Conv3D 256→128  │ ◄── 32×32×32
│ Conv3D 128→64   │ ◄── 64×64×64
│ Conv3D 64→32    │ ◄── 128×128×128
│ Conv3D 32→3     │ ◄── 128×128×128
└─────────────────┘
         │
         ▼
Output: 3×128×128×128
```

## 6. Training Pipeline Data Flow

### 6.1 Single Model Training Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE DATA FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

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

### 6.2 Cross-Validation Training Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           CROSS-VALIDATION TRAINING FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Dataset (1000 patients)
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
│ • Train: 800    │
│ • Val: 200      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 1        │ ◄── Train on folds 0,2,3,4
│                 │    Validate on fold 1
│ • Train: 800    │
│ • Val: 200      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 2        │ ◄── Train on folds 0,1,3,4
│                 │    Validate on fold 2
│ • Train: 800    │
│ • Val: 200      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 3        │ ◄── Train on folds 0,1,2,4
│                 │    Validate on fold 3
│ • Train: 800    │
│ • Val: 200      │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Fold 4        │ ◄── Train on folds 0,1,2,3
│                 │    Validate on fold 4
│ • Train: 800    │
│ • Val: 200      │
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

## 7. Memory Usage Flow for Different Models

### 7.1 Memory Usage During Training
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY USAGE FLOW (8GB VRAM)                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Model: UNet (2D)
┌─────────────────┐
│ Input Data      │ ◄── 4×128×128×2 = 131KB
│ • Batch Size: 4 │
│ • Modalities: 4 │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Model Weights   │ ◄── 31M parameters = 124MB
│ • Parameters    │
│ • Gradients     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Total Memory    │ ◄── ~2GB VRAM
│ • Data: 131KB   │
│ • Model: 124MB  │
│ • Overhead: 1.8GB│
└─────────────────┘

Model: 3D UNet
┌─────────────────┐
│ Input Data      │ ◄── 4×128×128×128×1 = 8.4MB
│ • Batch Size: 1 │
│ • Modalities: 4 │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Model Weights   │ ◄── 19M parameters = 76MB
│ • Parameters    │
│ • Gradients     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Total Memory    │ ◄── ~6GB VRAM
│ • Data: 8.4MB   │
│ • Model: 76MB   │
│ • Overhead: 5.9GB│
└─────────────────┘

Model: ResUNet
┌─────────────────┐
│ Input Data      │ ◄── 4×128×128×128×1 = 8.4MB
│ • Batch Size: 1 │
│ • Modalities: 4 │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Model Weights   │ ◄── 39M parameters = 156MB
│ • Parameters    │
│ • Gradients     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Total Memory    │ ◄── ~8GB VRAM (MAX)
│ • Data: 8.4MB   │
│ • Model: 156MB  │
│ • Overhead: 7.8GB│
└─────────────────┘
```

## 8. Inference Pipeline Data Flow

### 8.1 Inference Flow
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              INFERENCE PIPELINE DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Input: New Patient Data
         │
         ▼
┌─────────────────┐
│ Preprocessing   │ ◄── Same as training
│                 │
│ • Load          │
│ • Resize        │
│ • Normalize     │
│ • Stack         │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Load Model      │ ◄── Load trained model
│                 │
│ • Checkpoint    │
│ • Weights       │
│ • Architecture  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Forward Pass    │ ◄── Run inference
│                 │
│ • No Gradients  │
│ • Batch Size: 1 │
│ • Fast Mode     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │ ◄── Process output
│                 │
│ • Softmax       │
│ • Argmax        │
│ • Resize        │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Output          │ ◄── Final segmentation
│                 │
│ • 3D Volume     │
│ • NIfTI Format  │
│ • Metrics       │
└─────────────────┘
```

## 9. Complete End-to-End Data Flow

### 9.1 Complete Pipeline Visualization
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE END-TO-END DATA FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Raw NIfTI Files (240×240×155)
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   T1N       │    │   T1C       │    │   T2W       │    │   T2F       │
│ 240×240×155 │    │ 240×240×155 │    │ 240×240×155 │    │ 240×240×155 │
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

## 10. Data Flow Summary

### 10.1 Key Data Transformations
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA TRANSFORMATION SUMMARY                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

Input: Raw NIfTI Files
• Format: 240×240×155 voxels
• Data Type: 16-bit integers
• Spacing: ~1.0×1.0×1.0 mm³
• Files: 5 per patient (T1N, T1C, T2W, T2F, SEG)

    │
    ▼

Preprocessing Pipeline
• LoadImaged: Load NIfTI files
• EnsureChannelFirstd: Add channel dimension
• Spacingd: Resample to 1mm³
• ScaleIntensityRanged: Normalize to [0,1]
• CropForegroundd: Remove background
• Resized: Resize to 128³
• ConvertToMultiChannelBasedOnBratsClassesd: Convert labels

    │
    ▼

Training Data
• Input: 4×128×128×128 (4 modalities)
• Target: 3×128×128×128 (3 classes)
• Batch Size: 1-4 (depending on model)
• Data Type: float32

    │
    ▼

Model Processing
• UNet: 2D processing (4×128×128)
• 3D UNet: 3D processing (4×128×128×128)
• ResUNet: 3D with residuals (4×128×128×128)
• Attention UNet: 3D with attention (4×128×128×128)
• nnUNet: Advanced 3D (4×128×128×128)
• VNet: Volumetric 3D (4×128×128×128)

    │
    ▼

Output: Segmentation
• Format: 3×128×128×128 (3 classes)
• Classes: [Background, NCR/NET/ED, ET]
• Data Type: float32
• Post-processing: Softmax, Argmax, Resize
```

This comprehensive diagrammatic representation shows the complete data flow through preprocessing, training, and inference pipelines for different models in our BraTS2023 brain MRI segmentation project.
