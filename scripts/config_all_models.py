#!/usr/bin/env python3
"""
Configuration for training all models with optimized settings
"""

from config.config import Config, DataConfig, ModelConfig, TrainingConfig, SystemConfig

def get_optimized_config_for_model(model_name: str) -> Config:
    """Get optimized configuration for specific model"""
    
    # Base configuration with EDA-optimized settings
    base_config = Config()
    
    # EDA-optimized data settings
    base_config.data.brain_only_training = True
    base_config.data.brain_mask_method = "otsu"
    base_config.data.background_weight = 0.05
    base_config.data.foreground_sampling = True
    
    # EDA-optimized training settings
    base_config.training.loss_function = "weighted_dice_bce"
    base_config.training.scheduler = "poly"
    base_config.training.optimizer = "adamw"
    base_config.training.learning_rate = 1e-4
    base_config.training.weight_decay = 1e-5
    base_config.training.use_amp = True
    base_config.training.max_grad_norm = 1.0
    base_config.training.patience = 15
    
    # Model-specific optimizations for RTX 4070 8GB VRAM
    if model_name == "unet":
        base_config.training.batch_size = 4  # 2D model, can use larger batch
        base_config.training.num_epochs = 100
        base_config.model.features = [32, 64, 128, 256]
        
    elif model_name == "unet3d":
        base_config.training.batch_size = 1  # 3D model, limited by 8GB VRAM
        base_config.training.num_epochs = 100
        base_config.model.features = [32, 64, 128, 256]
        
    elif model_name == "resunet":
        base_config.training.batch_size = 1  # ResUNet, limited by 8GB VRAM
        base_config.training.num_epochs = 100
        base_config.model.features = [32, 64, 128, 256]
        base_config.model.num_res_units = 2
        
    elif model_name == "attentionunet":
        base_config.training.batch_size = 1  # Large model, may need gradient accumulation
        base_config.training.num_epochs = 90  # Fewer epochs for large model
        base_config.model.features = [32, 64, 128, 256]
        base_config.training.learning_rate = 5e-5  # Lower LR for stability
        
    elif model_name == "nnunet":
        base_config.training.batch_size = 1  # Very large model, needs gradient accumulation
        base_config.training.num_epochs = 80  # Fewer epochs
        base_config.model.features = [32, 64, 128, 256, 512]
        base_config.training.learning_rate = 5e-5  # Lower LR
        base_config.training.patience = 20  # More patience for complex model
        
    elif model_name == "vnet":
        base_config.training.batch_size = 1  # Large volumetric model, may need gradient accumulation
        base_config.training.num_epochs = 90
        base_config.model.features = [32, 64, 128, 256]
        base_config.training.learning_rate = 5e-5  # Lower LR for stability
    
    # Set model name
    base_config.model.model_name = model_name
    
    return base_config

def get_quick_test_config(model_name: str) -> Config:
    """Get configuration for quick testing"""
    config = get_optimized_config_for_model(model_name)
    
    # Quick test settings
    config.training.num_epochs = 5
    config.data.n_folds = 2  # Use 2 folds for quick testing
    config.training.patience = 3
    
    return config

def get_all_models_configs() -> dict:
    """Get configurations for all models"""
    from src.models import get_available_models
    
    configs = {}
    for model_name in get_available_models():
        configs[model_name] = get_optimized_config_for_model(model_name)
    
    return configs

def print_model_configs():
    """Print configuration summary for all models"""
    configs = get_all_models_configs()
    
    print("Model Configurations Summary")
    print("=" * 80)
    print(f"{'Model':<15} {'Batch Size':<10} {'Epochs':<8} {'LR':<10} {'Features':<20}")
    print("-" * 80)
    
    for model_name, config in configs.items():
        features_str = str(config.model.features)[:18] + "..." if len(str(config.model.features)) > 18 else str(config.model.features)
        print(f"{model_name:<15} {config.training.batch_size:<10} {config.training.num_epochs:<8} {config.training.learning_rate:<10} {features_str:<20}")
    
    print("\nCommon Settings:")
    print(f"- Brain-only training: {configs['unet3d'].data.brain_only_training}")
    print(f"- Brain mask method: {configs['unet3d'].data.brain_mask_method}")
    print(f"- Background weight: {configs['unet3d'].data.background_weight}")
    print(f"- Foreground sampling: {configs['unet3d'].data.foreground_sampling}")
    print(f"- Loss function: {configs['unet3d'].training.loss_function}")
    print(f"- Optimizer: {configs['unet3d'].training.optimizer}")
    print(f"- Scheduler: {configs['unet3d'].training.scheduler}")
    print(f"- Mixed precision: {configs['unet3d'].training.use_amp}")

if __name__ == "__main__":
    print_model_configs()
