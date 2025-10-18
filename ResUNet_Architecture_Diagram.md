# ResUNet3D Architecture Diagram

## Your Trained Model Configuration:
- **Input**: 4 modalities (T1, T1c, T2w, T2f) → 4 channels
- **Output**: 3 classes (Background, Tumor Core, Enhancing Tumor) → 3 channels
- **Input Shape**: 128×128×128 voxels
- **Feature Maps**: [32, 64, 128, 256]
- **Residual Units**: 2 per block
- **Parameters**: 9,806,723 (9.8M parameters)

## Architecture Flow:

```
Input: [Batch, 4, 128, 128, 128]
    ↓
Initial Conv3D: 4 → 32 channels
    ↓
ENCODER PATH:
    ↓
MaxPool3D(2) + ResUnit: 32 → 64 channels
    ↓ [64, 64, 64, 64]
MaxPool3D(2) + ResUnit: 64 → 128 channels  
    ↓ [128, 32, 32, 32]
MaxPool3D(2) + ResUnit: 128 → 256 channels
    ↓ [256, 16, 16, 16]
Bottleneck ResUnit: 256 → 256 channels
    ↓ [256, 16, 16, 16]
DECODER PATH:
    ↓
ConvTranspose3D(2) + Skip Connection + ResUnit: 256+128 → 128 channels
    ↓ [128, 32, 32, 32]
ConvTranspose3D(2) + Skip Connection + ResUnit: 128+64 → 64 channels
    ↓ [64, 64, 64, 64]
ConvTranspose3D(2) + Skip Connection + ResUnit: 64+32 → 32 channels
    ↓ [32, 128, 128, 128]
Final Conv3D(1): 32 → 3 channels
    ↓
Output: [Batch, 3, 128, 128, 128]
```

## Detailed Block Structure:

### 1. Initial Convolution
```
Input: [B, 4, 128, 128, 128]
Conv3D(3×3×3) + BatchNorm + ReLU
Output: [B, 32, 128, 128, 128]
```

### 2. Encoder Blocks
```
Block 1: [B, 32, 128, 128, 128]
    ↓ MaxPool3D(2)
    [B, 32, 64, 64, 64]
    ↓ ResUnit(32→64, 2 subunits)
    [B, 64, 64, 64, 64]

Block 2: [B, 64, 64, 64, 64]
    ↓ MaxPool3D(2)
    [B, 64, 32, 32, 32]
    ↓ ResUnit(64→128, 2 subunits)
    [B, 128, 32, 32, 32]

Block 3: [B, 128, 32, 32, 32]
    ↓ MaxPool3D(2)
    [B, 128, 16, 16, 16]
    ↓ ResUnit(128→256, 2 subunits)
    [B, 256, 16, 16, 16]
```

### 3. Bottleneck
```
[B, 256, 16, 16, 16]
    ↓ ResUnit(256→256, 2 subunits)
[B, 256, 16, 16, 16]
```

### 4. Decoder Blocks
```
Block 1: [B, 256, 16, 16, 16]
    ↓ ConvTranspose3D(2) + Interpolate
    [B, 128, 32, 32, 32]
    ↓ Skip Connection (from encoder)
    [B, 128+128=256, 32, 32, 32]
    ↓ ResUnit(256→128, 2 subunits)
    [B, 128, 32, 32, 32]

Block 2: [B, 128, 32, 32, 32]
    ↓ ConvTranspose3D(2) + Interpolate
    [B, 64, 64, 64, 64]
    ↓ Skip Connection (from encoder)
    [B, 64+64=128, 64, 64, 64]
    ↓ ResUnit(128→64, 2 subunits)
    [B, 64, 64, 64, 64]

Block 3: [B, 64, 64, 64, 64]
    ↓ ConvTranspose3D(2) + Interpolate
    [B, 32, 128, 128, 128]
    ↓ Skip Connection (from encoder)
    [B, 32+32=64, 128, 128, 128]
    ↓ ResUnit(64→32, 2 subunits)
    [B, 32, 128, 128, 128]
```

### 5. Final Output
```
[B, 32, 128, 128, 128]
    ↓ Conv3D(1×1×1)
[B, 3, 128, 128, 128]
```

## Residual Unit Structure:
```
Input: [B, C_in, H, W, D]
    ↓
Conv3D(3×3×3) + BatchNorm + ReLU
    ↓
Conv3D(3×3×3) + BatchNorm
    ↓
+ (Residual Connection)
    ↓
ReLU
    ↓
Output: [B, C_out, H, W, D]
```

## Skip Connections:
- **Purpose**: Preserve fine-grained details from encoder to decoder
- **Implementation**: Concatenation of upsampled features with encoder features
- **Channel Handling**: Input channels = upsampled_channels + skip_channels

## Key Features:
1. **3D Convolutions**: Processes entire 3D volume
2. **Residual Connections**: Helps with gradient flow and training stability
3. **Skip Connections**: Preserves spatial details
4. **Batch Normalization**: Stabilizes training
5. **ReLU Activation**: Non-linear feature learning
6. **Dropout (0.1)**: Prevents overfitting

## Memory Usage:
- **Input**: 4 × 128³ × 4 bytes = ~33.5 MB per sample
- **Model Parameters**: 9.8M × 4 bytes = ~39.2 MB
- **Peak GPU Memory**: ~6-8 GB (with batch_size=2)

## Training Configuration:
- **Optimizer**: AdamW (lr=0.0001, weight_decay=1e-5)
- **Scheduler**: Cosine Annealing
- **Loss**: Dice Loss
- **Batch Size**: 2 (optimized for RTX 4070)
- **Epochs**: 3 (as configured)
