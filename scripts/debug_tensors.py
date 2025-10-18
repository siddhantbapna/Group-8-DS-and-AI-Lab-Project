"""
Debug tensor shapes in the training pipeline
"""
import sys
import os
sys.path.append('src')

import torch
from config.config import get_config
from src.preprocessing import BraTS2023Preprocessor

def debug_tensor_shapes():
    """Debug tensor shapes"""
    print("Debugging tensor shapes...")
    
    # Get configuration
    config = get_config('unet3d')
    config.training.batch_size = 2
    config.data.output_shape = (64, 64, 64)  # Smaller for debugging
    
    # Setup data
    preprocessor = BraTS2023Preprocessor(config.data)
    
    # Create data dictionaries
    train_data_dicts = preprocessor.create_data_dicts(config.data.train_data_path)
    print(f"Found {len(train_data_dicts)} training samples")
    
    # Create cross-validation splits
    cv_splits = preprocessor.create_cross_validation_splits(train_data_dicts)
    train_data, val_data = cv_splits[0]  # Use first fold
    
    # Create transforms
    train_transforms = preprocessor.get_train_transforms(is_training=True)
    
    # Create datasets
    train_dataset, _ = preprocessor.create_datasets(
        train_data[:2], val_data[:2], train_transforms, train_transforms  # Just 2 samples
    )
    
    # Create data loader
    train_loader, _ = preprocessor.create_dataloaders(
        train_dataset, train_dataset, config.training.batch_size, 0  # No multiprocessing
    )
    
    # Get a batch
    batch_data = next(iter(train_loader))
    
    print("\nBatch data keys:", list(batch_data.keys()))
    
    # Check each modality
    for modality in config.data.modality_keys:
        tensor = batch_data[modality]
        print(f"{modality}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Check segmentation
    seg_tensor = batch_data[config.data.seg_key]
    print(f"seg: shape={seg_tensor.shape}, dtype={seg_tensor.dtype}")
    print(f"seg unique values: {torch.unique(seg_tensor)}")
    
    # Stack modalities
    modalities = []
    for modality in config.data.modality_keys:
        modalities.append(batch_data[modality])
    inputs = torch.cat(modalities, dim=1)
    print(f"\nStacked inputs: shape={inputs.shape}")
    
    # Map labels
    targets = seg_tensor
    targets_mapped = torch.zeros_like(targets)
    targets_mapped[targets == 0] = 0  # Background
    targets_mapped[targets == 1] = 1  # NCR/NET -> class 1
    targets_mapped[targets == 2] = 1  # ED -> class 1 (combine with NCR/NET)
    targets_mapped[targets == 3] = 2  # ET -> class 2
    print(f"Mapped targets: shape={targets_mapped.shape}")
    print(f"Mapped targets unique values: {torch.unique(targets_mapped)}")
    
    # Test model output shape
    from src.models import create_model
    model = create_model(
        config.model.model_name,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        features=[16, 32, 64],  # Smaller model
        dropout=config.model.dropout
    )
    
    with torch.no_grad():
        outputs = model(inputs)
        print(f"\nModel outputs: shape={outputs.shape}")
        
        # Test argmax
        predictions = torch.argmax(outputs, dim=1)
        print(f"Predictions: shape={predictions.shape}")
        print(f"Predictions unique values: {torch.unique(predictions)}")
        
        # Test comparison
        print(f"Can compare predictions and targets: {predictions.shape == targets_mapped.shape}")

if __name__ == "__main__":
    debug_tensor_shapes()
