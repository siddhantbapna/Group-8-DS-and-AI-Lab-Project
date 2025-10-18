"""
Example usage script for brain MRI segmentation project
"""
import sys
import os
sys.path.append('src')

from config.config import get_config
from src.models import create_model, get_available_models
from src.preprocessing import BraTS2023Preprocessor
from src.metrics import SegmentationMetrics, create_loss_function
from src.utils import set_seed, get_device_info, count_parameters
import torch

def example_model_creation():
    """Example of creating different models"""
    print("=== Model Creation Examples ===")
    
    # Get available models
    available_models = get_available_models()
    print(f"Available models: {available_models}")
    
    # Create different models
    for model_name in ['unet3d', 'resunet', 'attentionunet']:
        try:
            model = create_model(
                model_name,
                in_channels=4,
                out_channels=3,
                features=[32, 64, 128, 256]
            )
            
            params = count_parameters(model)
            print(f"\n{model_name.upper()}:")
            print(f"  Total parameters: {params['total_parameters']:,}")
            print(f"  Trainable parameters: {params['trainable_parameters']:,}")
            
            # Test forward pass
            x = torch.randn(1, 4, 64, 64, 64)
            with torch.no_grad():
                output = model(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            
        except Exception as e:
            print(f"Error creating {model_name}: {e}")

def example_preprocessing():
    """Example of preprocessing pipeline"""
    print("\n=== Preprocessing Examples ===")
    
    # Get configuration
    config = get_config('unet3d')
    
    # Create preprocessor
    preprocessor = BraTS2023Preprocessor(config.data)
    
    # Get transforms
    train_transforms = preprocessor.get_train_transforms(is_training=True)
    val_transforms = preprocessor.get_val_transforms()
    
    print(f"Training transforms: {len(train_transforms.transforms)} transforms")
    print(f"Validation transforms: {len(val_transforms.transforms)} transforms")
    
    # Example data dictionary (you would load real data)
    example_data = {
        't1n': 'path/to/t1n.nii.gz',
        't1c': 'path/to/t1c.nii.gz',
        't2w': 'path/to/t2w.nii.gz',
        't2f': 'path/to/t2f.nii.gz',
        'seg': 'path/to/seg.nii.gz'
    }
    
    print("Example data dictionary structure:")
    for key, value in example_data.items():
        print(f"  {key}: {value}")

def example_metrics():
    """Example of metrics computation"""
    print("\n=== Metrics Examples ===")
    
    # Create metrics computer
    metrics_computer = SegmentationMetrics(num_classes=3, include_background=False)
    
    # Create dummy data
    batch_size, num_classes, depth, height, width = 2, 3, 64, 64, 64
    y_pred = torch.randn(batch_size, num_classes, depth, height, width)
    y_true = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Compute metrics
    metrics = metrics_computer.compute_all_metrics(y_pred, y_true)
    
    print("Computed metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

def example_loss_functions():
    """Example of loss functions"""
    print("\n=== Loss Functions Examples ===")
    
    # Create dummy data
    batch_size, num_classes, depth, height, width = 2, 3, 64, 64, 64
    y_pred = torch.randn(batch_size, num_classes, depth, height, width)
    y_true = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Test different loss functions
    loss_functions = ['dice', 'dice_ce', 'dice_focal', 'focal', 'tversky']
    
    for loss_name in loss_functions:
        try:
            loss_fn = create_loss_function(loss_name)
            loss = loss_fn(y_pred, y_true)
            print(f"{loss_name}: {loss.item():.4f}")
        except Exception as e:
            print(f"Error with {loss_name}: {e}")

def example_configuration():
    """Example of configuration system"""
    print("\n=== Configuration Examples ===")
    
    # Get different model configurations
    models = ['unet', 'unet3d', 'resunet', 'nnunet', 'attentionunet', 'vnet']
    
    for model_name in models:
        try:
            config = get_config(model_name)
            print(f"\n{model_name.upper()} Configuration:")
            print(f"  Model: {config.model.model_name}")
            print(f"  Features: {config.model.features}")
            print(f"  Batch size: {config.training.batch_size}")
            print(f"  Learning rate: {config.training.learning_rate}")
            print(f"  Loss function: {config.training.loss_function}")
        except Exception as e:
            print(f"Error getting config for {model_name}: {e}")

def example_device_info():
    """Example of device information"""
    print("\n=== Device Information ===")
    
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")

def main():
    """Main example function"""
    print("Brain MRI Segmentation Project - Example Usage")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run examples
    example_device_info()
    example_configuration()
    example_model_creation()
    example_preprocessing()
    example_metrics()
    example_loss_functions()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nTo start training, run:")
    print("  python main.py --model unet3d --epochs 10 --batch_size 2")
    print("\nTo run cross-validation:")
    print("  python main.py --mode cv --model unet3d --n_folds 3 --epochs 10")
    print("\nTo run inference:")
    print("  python main.py --mode inference --model_path path/to/model.pth --inference_data_path path/to/data")

if __name__ == "__main__":
    main()
