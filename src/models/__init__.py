"""
Model factory for brain MRI segmentation
"""
from .Unet import create_unet
from .Unet3D import create_unet3d
from .resUnet import create_resunet
from .attentionUnet import create_attentionunet
from .nnUnet import create_nnunet
from .vnet import create_vnet
from typing import Dict, Any
import torch.nn as nn

MODEL_FACTORIES = {
    'unet': create_unet,
    'unet3d': create_unet3d,
    'resunet': create_resunet,
    'attentionunet': create_attentionunet,
    'nnunet': create_nnunet,
    'vnet': create_vnet,
}

def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model by name
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments for model creation
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_FACTORIES:
        available_models = list(MODEL_FACTORIES.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    return MODEL_FACTORIES[model_name](**kwargs)

def get_available_models() -> list:
    """Get list of available models"""
    return list(MODEL_FACTORIES.keys())

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create a test model to get information
    test_model = create_model(model_name, in_channels=4, out_channels=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in test_model.parameters())
    trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    
    return {
        'name': model_name,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_class': test_model.__class__.__name__,
        'factory_function': MODEL_FACTORIES[model_name].__name__
    }

# Example usage
if __name__ == "__main__":
    # Test model creation
    models_to_test = ['unet', 'unet3d', 'resunet', 'attentionunet', 'nnunet', 'vnet']
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name, in_channels=4, out_channels=3)
            info = get_model_info(model_name)
            print(f"Model: {model_name}")
            print(f"  Class: {info['model_class']}")
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Trainable: {info['trainable_parameters']:,}")
            print("-" * 50)
        except Exception as e:
            print(f"Error creating {model_name}: {e}")
            print("-" * 50)
