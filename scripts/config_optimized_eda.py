#!/usr/bin/env python3
"""
Optimized Configuration Based on EDA Analysis
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class OptimizedDataConfig(BaseModel):
    """Optimized data configuration based on EDA findings"""
    
    # Core paths
    data_path: str = "data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    val_data_path: str = "data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    
    # EDA-validated settings
    output_shape: List[int] = [128, 128, 128]  # ✅ Validated: Good balance
    target_voxel_spacing: List[float] = [1.0, 1.0, 1.0]  # ✅ Perfect match
    
    # EDA-optimized intensity ranges
    intensity_range: tuple = (0.0, 1400.0)  # ✅ Validated: Perfect for most modalities
    modality_specific_ranges: Dict[str, tuple] = {
        't1n': (0.0, 1400.0),  # ✅ Perfect
        't1c': (0.0, 2000.0),  # 🔧 Enhanced: Higher range for contrast
        't2w': (0.0, 1400.0),  # ✅ Perfect
        't2f': (0.0, 1400.0)   # ✅ Perfect
    }
    
    # Keys
    all_keys: List[str] = ['t1n', 't1c', 't2w', 't2f', 'seg']
    modality_keys: List[str] = ['t1n', 't1c', 't2w', 't2f']
    seg_key: str = 'seg'
    
    # Cross-validation
    n_folds: int = 5
    random_seed: int = 42
    
    # EDA-optimized brain-only training
    brain_only_training: bool = True  # ✅ Essential for 98.6% background
    brain_mask_method: str = "otsu"  # 🔧 Enhanced: More robust than intensity
    background_weight: float = 0.05  # 🔧 Enhanced: Even less background influence
    foreground_sampling: bool = True  # 🔧 Enhanced: Sample more tumor pixels

class OptimizedTrainingConfig(BaseModel):
    """Optimized training configuration based on EDA findings"""
    
    # EDA-validated settings
    epochs: int = 100
    batch_size: int = 2  # ✅ Good for memory efficiency
    learning_rate: float = 1e-4  # ✅ Conservative for medical data
    
    # Optimizer
    optimizer: str = "adamw"  # ✅ Good choice
    weight_decay: float = 1e-5
    
    # EDA-optimized scheduler
    scheduler: str = "poly"  # ✅ Good for medical data
    
    # EDA-optimized loss function
    loss_function: str = "weighted_dice_bce"  # ✅ Essential for class imbalance
    
    # Training options
    use_amp: bool = True  # ✅ Memory efficient
    gradient_clipping: Optional[float] = 1.0  # 🔧 Added: Prevent gradient explosion
    early_stopping_patience: int = 20

class OptimizedModelConfig(BaseModel):
    """Optimized model configuration"""
    
    # EDA-validated settings
    in_channels: int = 4  # ✅ T1N, T1C, T2W, T2F
    out_channels: int = 3  # ✅ After BraTS label mapping
    spatial_dims: int = 3  # ✅ 3D medical imaging
    
    # Model architecture
    model_name: str = "resunet"  # ✅ Good choice for medical imaging
    features: List[int] = [32, 64, 128, 256]  # ✅ Standard configuration

class OptimizedConfig:
    """Complete optimized configuration based on EDA analysis"""
    
    def __init__(self):
        self.data = OptimizedDataConfig()
        self.training = OptimizedTrainingConfig()
        self.model = OptimizedModelConfig()
    
    def get_eda_insights(self) -> Dict[str, Any]:
        """Return key EDA insights that informed these optimizations"""
        return {
            "class_imbalance": {
                "background_percentage": 98.6,
                "tumor_percentage": 1.4,
                "recommendation": "Brain-only training + weighted loss essential"
            },
            "intensity_analysis": {
                "t1c_highest_variability": True,
                "t1c_max_intensity": 14753,
                "recommendation": "Higher intensity range for T1C"
            },
            "image_properties": {
                "consistent_dimensions": True,
                "isotropic_voxels": True,
                "recommendation": "Current preprocessing optimal"
            },
            "tumor_distribution": {
                "ed_most_common": 0.87,
                "ncr_net_rarest": 0.20,
                "recommendation": "Foreground sampling + class-specific augmentation"
            }
        }

# Usage example
if __name__ == "__main__":
    config = OptimizedConfig()
    
    print("EDA-Optimized Configuration")
    print("=" * 50)
    print(f"Brain-only training: {config.data.brain_only_training}")
    print(f"Brain mask method: {config.data.brain_mask_method}")
    print(f"Background weight: {config.data.background_weight}")
    print(f"Foreground sampling: {config.data.foreground_sampling}")
    print(f"Loss function: {config.training.loss_function}")
    print(f"T1C intensity range: {config.data.modality_specific_ranges['t1c']}")
    
    print("\nEDA Insights:")
    insights = config.get_eda_insights()
    for category, details in insights.items():
        print(f"\n{category.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
