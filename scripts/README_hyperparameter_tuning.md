# Hyperparameter Tuning Scripts

## ResUNet Hyperparameter Tuning

This script performs comprehensive hyperparameter tuning for the ResUNet (SegResNet) model with visualizations.

### Usage
```bash
python scripts/hyperparameter_tuning_resunet.py
```

### What it does
- Trains ResUNet with 10 different hyperparameter configurations
- Each configuration is trained for 20 epochs
- Tests different combinations of:
  - Learning rate: [1e-4, 5e-4, 1e-3]
  - Weight decay: [1e-5, 1e-4, 1e-3]
  - Initial filters: [8, 16, 32]
  - Blocks down: [(1,2,2,4), (1,1,2,4), (1,2,3,4)]
  - Blocks up: [(1,1,1), (1,2,1)]
  - Loss weights: [(0.3,0.7), (0.5,0.5), (0.7,0.3)]

### Output
Results are saved to `outputs/logs/hyperparameter_tuning/resunet/`:
- **5 Visualization PNG files:**
  1. `1_best_dice_comparison.png` - Bar chart comparing all configurations
  2. `2_training_curves_top5.png` - Training/validation curves for top 5 configs
  3. `3_per_class_dice_comparison.png` - Per-class (WT, TC, ET) Dice scores
  4. `4_hyperparameter_impact.png` - Scatter plots showing impact of each hyperparameter
  5. `5_convergence_comparison.png` - Convergence speed comparison
- Each run directory with training history, checkpoints, and logs
- Summary file with best configuration
- All results JSON file

---

## AttentionUNet Hyperparameter Tuning

This script performs comprehensive hyperparameter tuning for AttentionUNet with visualizations.

### Usage
```bash
python scripts/hyperparameter_tuning_attenunet.py
```

### What it does
- Trains AttentionUNet with 11 different hyperparameter configurations
- Each configuration is trained for **40 epochs** with **patience=28**
- Base parameters:
  - Learning Rate: 1e-4 (varied: 5e-5, 1e-4, 5e-4)
  - Batch Size: 2 (varied: 1, 2)
  - Gradient Clipping: Enabled (varied: 0.5, 1.0, 2.0)
- Additional hyperparameters tested:
  - Weight Decay: [1e-5, 1e-4]
  - Channel Sizes: [(8,16,32,64), (16,32,64,128), (32,64,128,256)]
  - Loss Weights: Dice/BCE ratios [(0.3,0.7), (0.5,0.5), (0.7,0.3)]

### Output
Results are saved to `outputs/logs/hyperparameter_tuning/attenunet/`:
- **5 Visualization PNG files** (same as ResUNet)
- Each run directory with training history, checkpoints, and logs
- Summary file with best configuration
- All results JSON file

## Finding the Best Model

Both scripts automatically:
- Identify and report the best configuration
- Generate comprehensive visualizations
- Save all training histories for further analysis

