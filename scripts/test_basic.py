"""
Basic test script for the brain MRI segmentation project
"""
import sys
sys.path.append('src')

from config.config import get_config
from src.models import create_model
from src.utils import set_seed, get_device_info, count_parameters
import torch

def test_basic_functionality():
    """Test basic functionality without problematic models"""
    print("Brain MRI Segmentation Project - Basic Test")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    
    # Print device information
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Test working models
    working_models = ['unet3d', 'nnunet', 'vnet']
    
    for model_name in working_models:
        try:
            print(f"\nTesting {model_name.upper()}:")
            
            # Get configuration
            config = get_config(model_name)
            print(f"  Config loaded successfully")
            
            # Create model
            model = create_model(
                model_name,
                in_channels=4,
                out_channels=3,
                features=[16, 32, 64, 128]  # Smaller model for testing
            )
            
            # Count parameters
            params = count_parameters(model)
            print(f"  Total parameters: {params['total_parameters']:,}")
            print(f"  Trainable parameters: {params['trainable_parameters']:,}")
            
            # Test forward pass
            x = torch.randn(1, 4, 32, 32, 32)  # Smaller input for testing
            with torch.no_grad():
                output = model(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  ✓ {model_name} working correctly")
            
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")
    
    print("\n" + "=" * 50)
    print("Basic test completed!")

if __name__ == "__main__":
    test_basic_functionality()
