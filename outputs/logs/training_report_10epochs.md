# Training Report - 10 Epochs

Generated on: 2025-11-01 08:09:36

## Model Status

| Model | Status | Best Model | History |
|-------|--------|------------|----------|
| unet2d | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| unet3d | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| attenunet | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| nnunet | [FAILED] Failed/Incomplete | [FAILED] | [FAILED] |
| resunet | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |
| vnet | [SUCCESS] Complete | [SUCCESS] | [SUCCESS] |

## Performance Summary

| Rank | Model | Best Dice | Final Dice | Best Epoch | Final Train Loss | Final Val Loss |
|------|-------|-----------|------------|------------|------------------|----------------|
| 1 | resunet | 0.7795 | 0.7795 | 10 | 0.1645 | 0.1540 |
| 2 | unet3d | 0.5382 | 0.5382 | 10 | 0.1808 | 0.1514 |
| 3 | attenunet | 0.5097 | 0.4259 | 2 | 0.2402 | 0.4748 |
| 4 | vnet | 0.2670 | 0.2592 | 3 | 0.5765 | 0.6224 |
| 5 | unet2d | 0.0002 | 0.0001 | 6 | 0.7011 | 0.7200 |

## Best Performing Model

**resunet** achieved the highest Dice score of **0.7795**

## Files Generated

- Model checkpoints: `outputs/models/best_*.pth`
- Training histories: `outputs/logs/training_history_*.json`
- TensorBoard logs: `outputs/logs/tensorboard/`
- Training logs: `outputs/logs/*.log`
- Comparison plots: `outputs/logs/model_comparison_*.png`
