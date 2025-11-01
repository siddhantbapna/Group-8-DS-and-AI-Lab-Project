# Hyperparameter Tuning for AttentionUNet

This script performs comprehensive hyperparameter tuning for the AttentionUNet model with visualizations.

## Usage

Run the hyperparameter tuning script:
```bash
python scripts/hyperparameter_tuning_attenunet.py
```

## What it does

- Trains AttentionUNet with 11 different hyperparameter configurations
- Each configuration is trained for **40 epochs** with **patience=28** (early stopping)
- Base parameters:
  - Learning Rate: 1e-4 (varied: 5e-5, 1e-4, 5e-4)
  - Batch Size: 2 (varied: 1, 2)
  - Patience: 28
  - Gradient Clipping: Enabled (varied: 0.5, 1.0, 2.0)
  
## Hyperparameters Tested

1. **Learning Rate**: [5e-5, 1e-4, 5e-4]
2. **Batch Size**: [1, 2]
3. **Weight Decay**: [1e-5, 1e-4]
4. **Channel Sizes**: [(8,16,32,64), (16,32,64,128), (32,64,128,256)]
5. **Loss Weights**: Dice/BCE ratios [(0.3,0.7), (0.5,0.5), (0.7,0.3)]
6. **Gradient Clipping**: [0.5, 1.0, 2.0]

## Output

Results are saved to `outputs/logs/hyperparameter_tuning/attenunet/`:

### Files Generated:
- `all_results_{timestamp}.json` - All results in JSON format
- `summary_{timestamp}.txt` - Text summary of best configuration
- Individual run directories with:
  - Training history JSON
  - Best model checkpoint
  - Log file

### Visualizations Generated:
1. **1_best_dice_comparison.png** - Bar chart comparing best Dice scores
2. **2_training_curves_top5.png** - Training/validation curves for top 5 configs
3. **3_per_class_dice_comparison.png** - Per-class (WT, TC, ET) Dice comparisons
4. **4_hyperparameter_impact.png** - Scatter plots showing impact of each hyperparameter
5. **5_convergence_comparison.png** - Convergence speed comparison for top 10 configs

## Finding the Best Model

The script automatically:
- Identifies and reports the best configuration
- Generates comprehensive visualizations
- Saves all training histories for further analysis

