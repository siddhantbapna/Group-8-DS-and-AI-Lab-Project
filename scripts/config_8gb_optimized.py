#!/usr/bin/env python3
"""
Optimized configuration for RTX 4070 8GB VRAM
"""

from config.config import Config, DataConfig, ModelConfig, TrainingConfig, SystemConfig

def get_8gb_optimized_config_for_model(model_name: str) -> Config:
    """Get optimized configuration for specific model on 8GB VRAM"""
    
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
    base_config.training.use_amp = True  # Essential for 8GB VRAM
    base_config.training.max_grad_norm = 1.0
    base_config.training.patience = 15
    
    # Model-specific optimizations for 8GB VRAM
    if model_name == "unet":
        # 2D model - can use larger batch size
        base_config.training.batch_size = 4
        base_config.training.num_epochs = 100
        base_config.model.features = [32, 64, 128, 256]
        
    elif model_name == "unet3d":
        # 3D model - limited by 8GB VRAM
        base_config.training.batch_size = 1
        base_config.training.num_epochs = 100
        base_config.model.features = [32, 64, 128, 256]
        
    elif model_name == "resunet":
        # ResUNet - maximum utilization of 8GB VRAM
        base_config.training.batch_size = 1
        base_config.training.num_epochs = 100
        base_config.model.features = [32, 64, 128, 256]
        base_config.model.num_res_units = 2
        
    elif model_name == "attentionunet":
        # Attention UNet - may need gradient accumulation
        base_config.training.batch_size = 1
        base_config.training.num_epochs = 90
        base_config.model.features = [32, 64, 128, 256]
        base_config.training.learning_rate = 5e-5  # Lower LR for stability
        
    elif model_name == "nnunet":
        # nnUNet - needs significant optimization
        base_config.training.batch_size = 1
        base_config.training.num_epochs = 80
        base_config.model.features = [32, 64, 128, 256, 512]
        base_config.training.learning_rate = 5e-5
        base_config.training.patience = 20
        
    elif model_name == "vnet":
        # VNet - may need gradient accumulation
        base_config.training.batch_size = 1
        base_config.training.num_epochs = 90
        base_config.model.features = [32, 64, 128, 256]
        base_config.training.learning_rate = 5e-5
    
    # Set model name
    base_config.model.model_name = model_name
    
    return base_config

def get_8gb_quick_test_config(model_name: str) -> Config:
    """Get configuration for quick testing on 8GB VRAM"""
    config = get_8gb_optimized_config_for_model(model_name)
    
    # Quick test settings
    config.training.num_epochs = 5
    config.data.n_folds = 2  # Use 2 folds for quick testing
    config.training.patience = 3
    
    return config

def get_8gb_all_models_configs() -> dict:
    """Get configurations for all models optimized for 8GB VRAM"""
    from src.models import get_available_models
    
    configs = {}
    for model_name in get_available_models():
        configs[model_name] = get_8gb_optimized_config_for_model(model_name)
    
    return configs

def print_8gb_model_configs():
    """Print configuration summary for all models on 8GB VRAM"""
    configs = get_8gb_all_models_configs()
    
    print("Model Configurations for RTX 4070 8GB VRAM")
    print("=" * 80)
    print(f"{'Model':<15} {'Batch Size':<10} {'Epochs':<8} {'LR':<10} {'Memory':<10} {'Status':<15}")
    print("-" * 80)
    
    memory_usage = {
        'unet': '2GB',
        'unet3d': '6GB', 
        'resunet': '8GB',
        'attentionunet': '12GB',
        'nnunet': '16GB',
        'vnet': '10GB'
    }
    
    status = {
        'unet': 'Optimal',
        'unet3d': 'Good',
        'resunet': 'Max VRAM',
        'attentionunet': 'OOM',
        'nnunet': 'OOM',
        'vnet': 'OOM'
    }
    
    for model_name, config in configs.items():
        features_str = str(config.model.features)[:18] + "..." if len(str(config.model.features)) > 18 else str(config.model.features)
        print(f"{model_name:<15} {config.training.batch_size:<10} {config.training.num_epochs:<8} {config.training.learning_rate:<10} {memory_usage.get(model_name, 'Unknown'):<10} {status.get(model_name, 'Unknown'):<15}")
    
    print("\nCommon Settings:")
    print(f"- Brain-only training: {configs['unet3d'].data.brain_only_training}")
    print(f"- Brain mask method: {configs['unet3d'].data.brain_mask_method}")
    print(f"- Background weight: {configs['unet3d'].data.background_weight}")
    print(f"- Foreground sampling: {configs['unet3d'].data.foreground_sampling}")
    print(f"- Loss function: {configs['unet3d'].training.loss_function}")
    print(f"- Optimizer: {configs['unet3d'].training.optimizer}")
    print(f"- Scheduler: {configs['unet3d'].training.scheduler}")
    print(f"- Mixed precision: {configs['unet3d'].training.use_amp}")
    
    print("\nRecommendations for 8GB VRAM:")
    print("1. Best Choice: UNet (2GB VRAM, batch size 4)")
    print("2. Good Choice: 3D UNet (6GB VRAM, batch size 1)")
    print("3. Maximum: ResUNet (8GB VRAM, batch size 1)")
    print("4. Advanced: Attention UNet, VNet, nnUNet (need gradient accumulation)")

def get_gradient_accumulation_config(model_name: str, target_batch_size: int = 4) -> Config:
    """Get configuration with gradient accumulation for large models"""
    config = get_8gb_optimized_config_for_model(model_name)
    
    # Calculate gradient accumulation steps
    current_batch_size = config.training.batch_size
    accumulation_steps = target_batch_size // current_batch_size
    
    if accumulation_steps > 1:
        config.training.gradient_accumulation_steps = accumulation_steps
        config.training.effective_batch_size = target_batch_size
        print(f"Gradient accumulation: {accumulation_steps} steps for effective batch size {target_batch_size}")
    
    return config

if __name__ == "__main__":
    print_8gb_model_configs()
    
    print("\n" + "="*80)
    print("Gradient Accumulation Examples:")
    print("="*80)
    
    # Example for Attention UNet
    attention_config = get_gradient_accumulation_config("attentionunet", target_batch_size=4)
    print(f"Attention UNet: batch_size={attention_config.training.batch_size}, "
          f"accumulation_steps={getattr(attention_config.training, 'gradient_accumulation_steps', 1)}")
    
    # Example for VNet
    vnet_config = get_gradient_accumulation_config("vnet", target_batch_size=4)
    print(f"VNet: batch_size={vnet_config.training.batch_size}, "
          f"accumulation_steps={getattr(vnet_config.training, 'gradient_accumulation_steps', 1)}")
    
    # Example for nnUNet
    nnunet_config = get_gradient_accumulation_config("nnunet", target_batch_size=4)
    print(f"nnUNet: batch_size={nnunet_config.training.batch_size}, "
          f"accumulation_steps={getattr(nnunet_config.training, 'gradient_accumulation_steps', 1)}")
