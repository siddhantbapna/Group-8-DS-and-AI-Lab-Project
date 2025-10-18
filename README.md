# Brain MRI Segmentation Project

A comprehensive deep learning pipeline for brain MRI segmentation using the BraTS2023 dataset. This project implements multiple state-of-the-art architectures and provides a complete training and inference pipeline.

## Features

- **Multiple Model Architectures**: UNet, 3D UNet, ResUNet, nnUNet, Attention UNet, and V-Net
- **Cross-Validation Support**: Built-in k-fold cross-validation for robust evaluation
- **Comprehensive Metrics**: Dice score, Hausdorff distance, surface distance, and more
- **Advanced Preprocessing**: MONAI-based preprocessing with extensive data augmentation
- **Checkpoint Management**: Automatic checkpoint saving and model versioning
- **Inference Pipeline**: Complete inference system with test-time augmentation
- **Mixed Precision Training**: Support for automatic mixed precision for faster training

## Hardware Requirements

- **GPU**: RTX 4070 or better (8GB+ VRAM recommended)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for dataset and outputs

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd brain-mri-segmentation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the BraTS2023 dataset from the official website
2. Extract the dataset to the `data/` directory:
```
data/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
└── BraTS2023_2017_GLI_Mapping.xlsx
```

## Quick Start

### Single Model Training

Train a single model (e.g., 3D UNet):
```bash
python main.py --model unet3d --epochs 100 --batch_size 2 --learning_rate 1e-4
```

### Cross-Validation Training

Train with 5-fold cross-validation:
```bash
python main.py --mode cv --model unet3d --n_folds 5 --epochs 100
```

### Inference

Run inference on validation data:
```bash
python main.py --mode inference --model_path checkpoints/unet3d_fold_0/best_model_epoch_50.pth --inference_data_path data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData
```

## Model Architectures

### 1. UNet
- 2D UNet for slice-wise processing
- Good for memory-constrained environments
- Fast training and inference

### 2. 3D UNet
- Full 3D processing
- Better spatial context understanding
- Higher memory requirements

### 3. ResUNet
- Residual connections for better gradient flow
- Improved training stability
- Better performance on complex cases

### 4. nnUNet
- Based on the nnU-Net framework
- Adaptive architecture
- State-of-the-art performance

### 5. Attention UNet
- Attention mechanisms for better feature focus
- Improved boundary segmentation
- Better handling of small lesions

### 6. V-Net
- Volumetric processing
- Good for 3D medical images
- Efficient memory usage

## Configuration

The project uses a comprehensive configuration system. Key parameters can be modified in `config/config.py`:

```python
# Model configuration
model_config = ModelConfig(
    model_name="unet3d",
    in_channels=4,  # 4 modalities
    out_channels=3,  # 3 classes (WT, TC, ET)
    features=[32, 64, 128, 256]
)

# Training configuration
training_config = TrainingConfig(
    batch_size=2,
    num_epochs=100,
    learning_rate=1e-4,
    loss_function="dice_ce"
)
```

## Training Options

### Command Line Arguments

- `--model`: Model architecture to use
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--loss_function`: Loss function to use
- `--mode`: Training mode (train, cv, inference)
- `--fold`: Fold number for cross-validation
- `--n_folds`: Number of folds for cross-validation
- `--use_amp`: Enable automatic mixed precision
- `--resume`: Resume from checkpoint

### Loss Functions

- `dice`: Dice loss
- `dice_ce`: Combined Dice and Cross-Entropy loss
- `dice_focal`: Combined Dice and Focal loss
- `focal`: Focal loss
- `tversky`: Tversky loss
- `boundary`: Boundary loss

## Evaluation Metrics

The project computes comprehensive evaluation metrics:

- **Dice Score**: Overlap between prediction and ground truth
- **Hausdorff Distance**: Maximum distance between boundaries
- **Surface Distance**: Average distance between surfaces
- **Volume Metrics**: Volume difference and ratio
- **Confusion Matrix Metrics**: Precision, recall, F1-score, IoU

## Output Structure

```
outputs/
├── logs/                    # Training logs
├── models/                  # Saved models
├── predictions/             # Inference predictions
├── cv_results/             # Cross-validation results
└── tensorboard/            # TensorBoard logs
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs/logs/tensorboard
```

### Logs
Training logs are saved in `outputs/logs/` with detailed information about:
- Training and validation losses
- Metrics evolution
- Learning rate changes
- Model performance

## Advanced Features

### Test-Time Augmentation (TTA)
Enable TTA during inference for better performance:
```python
pipeline = InferencePipeline(config, model_path, output_dir)
results = pipeline.process_dataset(data_path, use_tta=True)
```

### Ensemble Methods
Combine predictions from multiple models:
```python
from src.utils import create_ensemble_prediction
ensemble_pred = create_ensemble_prediction(predictions, method='average')
```

### Custom Preprocessing
Modify preprocessing pipeline in `src/preprocessing.py`:
```python
# Add custom transforms
custom_transforms = [
    # Your custom transforms here
]
```

## Performance Optimization

### For RTX 4070 (8GB VRAM)
- Use batch size of 1-2 for 3D models
- Enable mixed precision training (`--use_amp`)
- Use gradient accumulation for larger effective batch sizes
- Consider using 2D models for faster training

### Memory Optimization
- Reduce input resolution in config
- Use gradient checkpointing
- Enable data loading with multiple workers
- Use CPU offloading for large models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**
   - Increase number of workers
   - Use SSD storage for data
   - Enable mixed precision training

3. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Try different loss functions
   - Use data augmentation

### Debug Mode
Enable debug logging:
```bash
python main.py --log_level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{brain_mri_segmentation,
  title={Brain MRI Segmentation with Deep Learning},
  author=Ravineel Singhi,
  year=2025,

}
```

## Acknowledgments

- BraTS2023 dataset organizers
- MONAI team for the excellent medical imaging framework
- PyTorch team for the deep learning framework
- All contributors to the open-source medical imaging community
