# Brain Tumor Segmentation (MONAI + PyTorch)

This project provides a complete training and inference pipeline for BraTS 2023 GLI data using MONAI and PyTorch. Models supported: 2D UNet, 3D UNet, AttentionUNet, DynUNet (nnUNet-like), SegResNet (ResUNet-like), VNet.

## Environment

- Python 3.13 (pika venv present at `pika/`)
- GPU: RTX 4070 8GB (laptop) recommended; AMP enabled by default

### Setup

```bash
# (Windows PowerShell)
# From project root
./pika/Scripts/python.exe -m pip install --upgrade pip
./pika/Scripts/pip.exe install -r requirements.txt
```

## Data

Place BraTS GLI data under `data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` and testing under `data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TestingData`.

## Quickstart

### 🚀 Complete Pipeline (Recommended)
```bash
# Run complete pipeline: data viz → test training → full training → prediction viz
./pika/Scripts/python.exe scripts/run_complete_pipeline.py --test-epochs 2 --full-epochs 50
```

### 📊 Individual Steps
```bash
# 1. Visualize data before training
./pika/Scripts/python.exe scripts/visualize_data.py --num-samples 3

# 2. Train all models for testing (2 epochs)
./pika/Scripts/python.exe scripts/train_all_models.py --epochs 2

# 3. Train all models for full training (50 epochs)
./pika/Scripts/python.exe scripts/train_all_models.py --epochs 50

# 4. Visualize predictions after training
./pika/Scripts/python.exe scripts/visualize_predictions.py --num-samples 2

# 5. View training progress with TensorBoard
./pika/Scripts/tensorboard.exe --logdir outputs/logs/tensorboard
```

### 🔧 Manual Training
```bash
# Train single model
./pika/Scripts/python.exe main.py --model unet2d

# Resume training from checkpoint
./pika/Scripts/python.exe main.py --model unet3d --resume outputs/models/latest_unet3d.pth

# Inference on a case
./pika/Scripts/python.exe main.py --mode infer --model unet3d --input path/to/case --output outputs/predictions/case_pred.nii.gz
```

## Features

### 🧠 Models Supported
- **2D UNet**: Memory-efficient 2D segmentation
- **3D UNet**: Full 3D brain tumor segmentation
- **AttentionUNet**: 3D UNet with attention mechanisms
- **DynUNet**: nnUNet-like dynamic architecture
- **SegResNet**: ResNet-based 3D segmentation
- **VNet**: V-shaped 3D CNN architecture

### 🚀 Training Features
- **Comprehensive Logging**: File logs + TensorBoard visualization
- **Checkpoint Management**: Auto-save best/latest models, resume training
- **Early Stopping**: Prevents overfitting with configurable patience
- **Gradient Clipping**: Prevents gradient explosion
- **AMP Training**: Automatic Mixed Precision for 8GB GPUs
- **Sliding Window Inference**: Memory-efficient validation

### 📊 Visualization & Analysis
- **Pre-training Data Inspection**: Multi-modal visualization with segmentation overlays
- **Training Progress Monitoring**: Real-time loss/metrics tracking
- **Post-training Predictions**: Ground truth vs prediction comparisons
- **Model Performance Comparison**: Side-by-side model evaluation
- **Automated Report Generation**: Comprehensive training summaries

## 8GB VRAM Tips

- Use `--model unet2d` to start; 3D models are heavier.
- Defaults set for 8GB: `roi=96^3`, `batch_size_3d=1`, AMP on.
- If OOM occurs:
  - Reduce `TrainConfig.slide_infer_roi_3d` to [80, 80, 80] or [64, 64, 64]
  - Ensure background apps are closed; set `num_workers=2`
  - Prefer `SegResNet` over `DynUNet` for lower memory

## 📁 Output Structure

```
outputs/
├── logs/
│   ├── tensorboard/           # TensorBoard logs for each run
│   ├── data_visualizations/   # Pre-training data samples
│   ├── prediction_visualizations/  # Post-training predictions
│   ├── training_history_*.json     # Training metrics
│   ├── training_report_*.md        # Automated reports
│   └── *.log                       # Training logs
├── models/
│   ├── best_*.pth             # Best model checkpoints
│   └── latest_*.pth           # Latest model checkpoints
└── predictions/               # Inference outputs
```

## 📊 Monitoring Training

```bash
# View real-time training progress
./pika/Scripts/tensorboard.exe --logdir outputs/logs/tensorboard

# Check logs
tail -f outputs/logs/*.log

# View training reports
cat outputs/logs/training_report_*.md
```

## 🎯 Pipeline Options

```bash
# Quick test (2 epochs for all models)
./pika/Scripts/python.exe scripts/run_complete_pipeline.py --test-epochs 2 --skip-full-training

# Full training only (skip test)
./pika/Scripts/python.exe scripts/run_complete_pipeline.py --skip-test-training --full-epochs 50

# Specific models only
./pika/Scripts/python.exe scripts/run_complete_pipeline.py --models unet2d unet3d resunet

# Skip visualization steps
./pika/Scripts/python.exe scripts/run_complete_pipeline.py --skip-data-viz --skip-prediction-viz
```

See `docs/` for pipeline and optimization notes.
