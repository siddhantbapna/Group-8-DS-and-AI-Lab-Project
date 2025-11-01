# Training Report - 20 Epochs

Generated on: 2025-11-01 15:16:34

## Model Status

| Model | Status | Best Model | History |
|-------|--------|------------|----------|
| unet3d | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| resunet | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| attenunet | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| nnunet | [FAILED] Failed/Incomplete | [FAILED] | [FAILED] |
| vnet | [FAILED] Failed/Incomplete | [SUCCESS] | [FAILED] |

## Performance Summary

| Rank | Model | Best Dice | Final Dice | Best Epoch | Final Train Loss | Final Val Loss |
|------|-------|-----------|------------|------------|------------------|----------------|
| 1 | resunet | 0.7587 | 0.7587 | 1 | 0.1170 | 0.0912 |
| 2 | unet3d | 0.5282 | 0.5282 | 2 | 0.3745 | 0.3743 |
| 3 | attenunet | 0.2589 | 0.2334 | 1 | 0.5725 | 0.6371 |

## Best Performing Model

**resunet** achieved the highest Dice score of **0.7587**

## Files Generated

- Model checkpoints: `outputs/models/best_*.pth`
- Training histories: `outputs/logs/training_history_*.json`
- TensorBoard logs: `outputs/logs/tensorboard/`
- Training logs: `outputs/logs/*.log`
- Comparison plots: `outputs/logs/model_comparison_*.png`
