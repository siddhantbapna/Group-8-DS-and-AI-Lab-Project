# Brain MRI Segmentation Project Documentation

## Overview
This project implements a comprehensive brain MRI segmentation system using the BraTS2023 dataset. It provides multiple deep learning models, preprocessing pipelines, training frameworks, and evaluation tools for brain tumor segmentation.

## Project Structure

```
Project/
├── config/
│   └── config.py                 # Configuration management
├── src/
│   ├── models/                   # Model architectures
│   │   ├── __init__.py          # Model factory
│   │   ├── Unet.py              # 2D UNet implementation
│   │   ├── Unet3D.py            # 3D UNet implementation
│   │   ├── resUnet.py           # ResUNet implementation
│   │   ├── nnUnet.py            # nnUNet implementation
│   │   ├── attentionUnet.py     # Attention UNet implementation
│   │   └── vnet.py              # VNet implementation
│   ├── preprocessing.py         # Data preprocessing and augmentation
│   ├── metrics.py               # Evaluation metrics and loss functions
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # Inference pipeline
│   ├── checkpoints.py           # Checkpoint management
│   └── utils.py                 # Utility functions
├── outputs/                     # Training outputs and logs
├── checkpoints/                 # Model checkpoints
├── data/                        # Dataset directory
├── main.py                      # Main training script
├── train_with_tensorboard.py    # Training with TensorBoard logging
├── debug_tensors.py             # Tensor debugging utility
├── requirements.txt             # Python dependencies
├── requirements_frozen.txt      # Exact package versions
└── README.md                    # Basic project information
```

## Core Components

### 1. Configuration System (`config/config.py`)

**Purpose**: Centralized configuration management for all training parameters.

**Key Features**:
- **Model Configuration**: Architecture parameters, input/output channels, feature maps
- **Training Configuration**: Epochs, batch size, learning rate, optimizer settings
- **Data Configuration**: Paths, output shapes, modality keys, augmentation settings
- **System Configuration**: Device settings, logging, checkpoint management

**Example Usage**:
```python
from config.config import get_config
config = get_config('unet3d')
print(f"Model: {config.model.model_name}")
print(f"Batch size: {config.training.batch_size}")
```

### 2. Model Architectures (`src/models/`)

**Available Models**:
1. **UNet** (`Unet.py`): Classic 2D U-Net architecture
2. **UNet3D** (`Unet3D.py`): 3D U-Net for volumetric data
3. **ResUNet** (`resUnet.py`): U-Net with residual connections
4. **nnUNet** (`nnUnet.py`): Self-configuring U-Net variant
5. **Attention UNet** (`attentionUnet.py`): U-Net with attention mechanisms
6. **VNet** (`vnet.py`): V-shaped network for 3D segmentation

**Model Factory** (`__init__.py`):
```python
from src.models import create_model
model = create_model('unet3d', in_channels=4, out_channels=3)
```

### 3. Data Preprocessing (`src/preprocessing.py`)

**BraTS2023Preprocessor Class**:
- **Data Loading**: Handles BraTS2023 dataset structure
- **Preprocessing Pipeline**: MONAI-based transforms for normalization and resizing
- **Data Augmentation**: Spatial and intensity augmentations
- **Cross-Validation**: K-fold cross-validation splits
- **Data Loaders**: PyTorch DataLoader creation

**Key Transforms**:
- `LoadImaged`: Load NIfTI files
- `EnsureChannelFirstd`: Add channel dimension
- `Spacingd`: Resample to target voxel spacing
- `ScaleIntensityRanged`: Normalize intensity values
- `CropForegroundd`: Remove background
- `Resized`: Resize to target shape
- `ConvertToMultiChannelBasedOnBratsClassesd`: Convert segmentation to multi-channel

**Augmentation Transforms**:
- Spatial: Random flips, rotations, affine transforms, zoom
- Intensity: Noise, smoothing, contrast adjustment, bias field
- Advanced: Gibbs noise, Rician noise, low resolution simulation

### 4. Metrics and Loss Functions (`src/metrics.py`)

**MetricsComputer Class**:
- **Accuracy**: Pixel-wise classification accuracy
- **Dice Score**: Overlap-based segmentation metric
- **Hausdorff Distance**: Boundary-based metric
- **Surface Distance**: Surface-based evaluation

**Loss Functions**:
- **Dice Loss**: Segmentation-focused loss
- **Cross-Entropy Loss**: Standard classification loss
- **Dice-CE Loss**: Combined loss function
- **Focal Loss**: Handles class imbalance

**MetricTracker Class**:
- Tracks metrics during training
- Computes average metrics across batches
- Provides training history

### 5. Training Pipeline (`src/train.py`)

**Trainer Class**:
- **Model Setup**: Model, optimizer, scheduler initialization
- **Data Setup**: Dataset and DataLoader creation
- **Training Loop**: Forward/backward pass, gradient clipping
- **Validation**: Model evaluation on validation set
- **Checkpointing**: Automatic model saving
- **Early Stopping**: Prevents overfitting
- **TensorBoard Logging**: Real-time training monitoring

**CrossValidationTrainer Class**:
- **K-Fold CV**: Implements k-fold cross-validation
- **Aggregated Results**: Combines results across folds
- **Statistical Analysis**: Mean and standard deviation

**Key Features**:
- Mixed Precision Training (AMP)
- Gradient Clipping
- Learning Rate Scheduling
- Comprehensive Logging
- Progress Bars with Real-time Metrics

### 6. Inference Pipeline (`src/inference.py`)

**InferenceEngine Class**:
- **Model Loading**: Load trained checkpoints
- **Preprocessing**: Apply same transforms as training
- **Prediction**: Generate segmentation masks
- **Post-processing**: Convert predictions to final format
- **Evaluation**: Compute metrics on test data

**Batch Processing**:
- Process multiple volumes efficiently
- Memory-optimized inference
- Support for different input formats

### 7. Checkpoint Management (`src/checkpoints.py`)

**CheckpointManager Class**:
- **Model Saving**: Save model state, optimizer, scheduler
- **Model Loading**: Resume training from checkpoints
- **Best Model Tracking**: Keep track of best performing model
- **Metadata Storage**: Save training configuration and metrics

**Features**:
- Automatic checkpoint naming
- Model summary generation
- Training history tracking
- Resume functionality

### 8. Utility Functions (`src/utils.py`)

**Key Functions**:
- **Device Setup**: GPU/CPU detection and configuration
- **Logging Setup**: Configure logging system
- **Seed Setting**: Reproducible training
- **Directory Creation**: Automatic output directory setup

## Usage Examples

### 1. Basic Training
```bash
# Train UNet3D for 100 epochs
python main.py --model unet3d --epochs 100 --batch_size 2

# Train with specific output shape
python main.py --model unet3d --epochs 50 --output_shape 128 128 128

# Train with cross-validation
python main.py --model unet3d --n_folds 5 --epochs 30
```

### 2. Training with TensorBoard
```bash
# Run training with TensorBoard logging
python train_with_tensorboard.py

# Start TensorBoard
tensorboard --logdir outputs/logs/tensorboard_fold_0
```

### 3. Model Inference
```python
from src.inference import InferenceEngine
from config.config import get_config

config = get_config('unet3d')
engine = InferenceEngine(config)
results = engine.infer_single_volume(volume_paths)
```

### 4. Custom Configuration
```python
from config.config import get_config

config = get_config('unet3d')
config.training.batch_size = 4
config.training.learning_rate = 0.001
config.model.features = [32, 64, 128, 256]
```

## Data Format

### BraTS2023 Dataset Structure
```
data/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│   ├── BraTS-GLI-00000-000/
│   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   │   └── BraTS-GLI-00000-000-seg.nii.gz
│   └── ...
└── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
    └── ...
```

### Modalities
- **T1**: T1-weighted MRI
- **T1c**: T1-weighted MRI with contrast
- **T2w**: T2-weighted MRI
- **T2f**: T2-FLAIR MRI
- **Seg**: Segmentation mask

### Segmentation Labels
- **0**: Background
- **1**: Necrotic and non-enhancing tumor core (NCR/NET)
- **2**: Peritumoral edema (ED)
- **3**: GD-enhancing tumor (ET)

## Training Configuration

### Hardware Requirements
- **GPU**: NVIDIA RTX 4070 (8GB VRAM) or better
- **RAM**: 32GB recommended
- **Storage**: SSD recommended for faster data loading

### Optimized Settings
```python
# For RTX 4070 with 32GB RAM
config.training.batch_size = 2
config.data.output_shape = [128, 128, 128]
config.model.features = [32, 64, 128, 256]
config.training.use_amp = True
config.system.num_workers = 4
```

### Performance Tips
1. **Use Mixed Precision**: Enable AMP for faster training
2. **Optimize Data Loading**: Use multiple workers for data loading
3. **Batch Size**: Adjust based on GPU memory
4. **Output Shape**: Smaller shapes for faster training, larger for better quality

## Evaluation Metrics

### Primary Metrics
- **Dice Score**: Measures overlap between predicted and ground truth
- **Accuracy**: Pixel-wise classification accuracy
- **Hausdorff Distance**: Measures maximum boundary distance
- **Surface Distance**: Average surface distance

### Interpretation
- **Dice Score**: 0.0 (no overlap) to 1.0 (perfect overlap)
- **Accuracy**: 0.0 (0% correct) to 1.0 (100% correct)
- **Hausdorff Distance**: Lower is better (measured in mm)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Reduce output shape
   - Use gradient accumulation

2. **Slow Training**:
   - Increase num_workers
   - Use SSD storage
   - Enable mixed precision

3. **Poor Performance**:
   - Increase model size
   - Adjust learning rate
   - Use data augmentation

4. **Tensor Shape Errors**:
   - Check input data format
   - Verify preprocessing pipeline
   - Ensure consistent tensor shapes

### Debug Tools
- `debug_tensors.py`: Debug tensor shapes and data flow
- TensorBoard: Monitor training progress
- Logging: Detailed training logs

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **MONAI**: Medical imaging toolkit
- **nibabel**: NIfTI file handling
- **numpy**: Numerical computing
- **tqdm**: Progress bars
- **tensorboard**: Training visualization

### Installation
```bash
pip install -r requirements.txt
```

## File Descriptions

### Main Scripts
- **`main.py`**: Primary training script with command-line interface
- **`train_with_tensorboard.py`**: Training with TensorBoard integration
- **`debug_tensors.py`**: Debugging utility for tensor operations

### Configuration
- **`config/config.py`**: Centralized configuration management
- **`requirements.txt`**: Python package dependencies
- **`requirements_frozen.txt`**: Exact package versions

### Source Code
- **`src/models/`**: Model architecture implementations
- **`src/preprocessing.py`**: Data preprocessing and augmentation
- **`src/metrics.py`**: Evaluation metrics and loss functions
- **`src/train.py`**: Training pipeline implementation
- **`src/inference.py`**: Inference pipeline implementation
- **`src/checkpoints.py`**: Checkpoint management system
- **`src/utils.py`**: Utility functions

### Output Directories
- **`outputs/`**: Training logs, results, and visualizations
- **`checkpoints/`**: Saved model checkpoints
- **`data/`**: Dataset storage directory

## Best Practices

### Training
1. **Start Small**: Begin with smaller models and datasets
2. **Monitor Progress**: Use TensorBoard for real-time monitoring
3. **Save Checkpoints**: Regular checkpointing prevents data loss
4. **Validate Early**: Use validation set to prevent overfitting

### Data Handling
1. **Preprocessing**: Consistent preprocessing across train/val/test
2. **Augmentation**: Use appropriate augmentations for medical data
3. **Memory Management**: Monitor GPU memory usage
4. **Data Loading**: Optimize data loading for performance

### Model Development
1. **Architecture**: Choose appropriate model for 3D medical data
2. **Loss Function**: Use segmentation-appropriate loss functions
3. **Hyperparameters**: Tune learning rate, batch size, and model size
4. **Regularization**: Use dropout and weight decay appropriately

## Future Enhancements

### Potential Improvements
1. **Advanced Models**: Implement newer architectures (TransUNet, Swin-UNet)
2. **Multi-Scale Training**: Progressive training with different resolutions
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Active Learning**: Intelligent sample selection for annotation
5. **Federated Learning**: Distributed training across institutions

### Additional Features
1. **Web Interface**: User-friendly web application
2. **API Endpoints**: RESTful API for model inference
3. **Docker Support**: Containerized deployment
4. **Cloud Integration**: AWS/Azure deployment support

## Conclusion

This project provides a comprehensive framework for brain MRI segmentation using state-of-the-art deep learning techniques. It includes multiple model architectures, robust preprocessing pipelines, comprehensive evaluation metrics, and extensive training utilities. The modular design allows for easy customization and extension for specific research needs.

The system is optimized for the BraTS2023 dataset but can be adapted for other medical imaging segmentation tasks with minimal modifications. The extensive documentation and examples make it suitable for both research and educational purposes.
