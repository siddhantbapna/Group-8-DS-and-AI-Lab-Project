"""
Configuration file for BraTS2023 Brain MRI Segmentation Project
"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch

@dataclass
class DataConfig:
    """Data configuration parameters"""
    # Dataset paths
    train_data_path: str = "./data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    val_data_path: str = "./data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    processed_data_path: str = "./data/processed_data"
    
    # Data keys
    modality_keys: List[str] = None
    all_keys: List[str] = None
    seg_key: str = "seg"
    
    # Preprocessing parameters
    target_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    output_shape: Tuple[int, int, int] = (128, 128, 128)
    intensity_range: Tuple[float, float] = (0.0, 1400.0)
    normalized_range: Tuple[float, float] = (0.0, 1.0)
    
    # Cross-validation
    n_folds: int = 5
    random_seed: int = 42
    
    # Brain-only training options
    brain_only_training: bool = True  # Focus training only on brain tissue
    brain_mask_method: str = "otsu"  # intensity, otsu, adaptive
    background_weight: float = 0.1  # Weight for background pixels in loss (0.0 = ignore completely)
    foreground_sampling: bool = True  # Sample more foreground pixels during training
    
    def __post_init__(self):
        if self.modality_keys is None:
            self.modality_keys = ["t1n", "t1c", "t2w", "t2f"]
        if self.all_keys is None:
            self.all_keys = self.modality_keys + [self.seg_key]

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Model selection
    model_name: str = "unet"  # unet, unet3d, resunet, nnunet, attentionunet, vnet
    
    # Model architecture parameters
    in_channels: int = 4  # 4 modalities
    out_channels: int = 3  # 3 classes (WT, TC, ET)
    features: List[int] = None
    dropout: float = 0.1
    
    # 3D specific parameters
    spatial_dims: int = 3
    
    # ResUNet parameters
    num_res_units: int = 2
    
    # Attention UNet parameters
    attention_dropout: float = 0.1
    
    # VNet parameters
    act: str = "relu"
    norm: str = "batch"
    
    def __post_init__(self):
        if self.features is None:
            self.features = [32, 64, 128, 256, 512]

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Training parameters
    batch_size: int = 2  # Reduced for RTX 4070 with 3D data
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer
    optimizer: str = "adam"  # adam, adamw, sgd
    scheduler: str = "poly"  # cosine, step, plateau, poly
    
    # Loss function
    loss_function: str = "dice"  # dice, ce, dice_ce, dice_bce, focal,weighted_dice_bce
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "outputs/logs"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.system.output_dir,
            self.system.checkpoint_dir,
            self.system.log_dir,
            self.data.processed_data_path,
            "outputs/models",
            "outputs/predictions"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__
        }

# Default configurations for different models
DEFAULT_CONFIGS = {
    'unet': Config(
        model=ModelConfig(model_name='unet', features=[32, 64, 128, 256]),
        training=TrainingConfig(batch_size=4, learning_rate=1e-4)
    ),
    'unet3d': Config(
        model=ModelConfig(model_name='unet3d', features=[32, 64, 128, 256]),
        training=TrainingConfig(batch_size=2, learning_rate=1e-4)
    ),
    'resunet': Config(
        model=ModelConfig(model_name='resunet', features=[32, 64, 128, 256], num_res_units=2),
        training=TrainingConfig(batch_size=2, learning_rate=1e-4)
    ),
    'nnunet': Config(
        model=ModelConfig(model_name='nnunet', features=[32, 64, 128, 256, 320, 320]),
        training=TrainingConfig(batch_size=1, learning_rate=1e-3)
    ),
    'attentionunet': Config(
        model=ModelConfig(model_name='attentionunet', features=[32, 64, 128, 256]),
        training=TrainingConfig(batch_size=2, learning_rate=1e-4)
    ),
    'vnet': Config(
        model=ModelConfig(model_name='vnet', features=[16, 32, 64, 128, 256]),
        training=TrainingConfig(batch_size=2, learning_rate=1e-4)
    )
}

def get_config(model_name: str = 'unet') -> Config:
    """Get configuration for a specific model"""
    if model_name.lower() in DEFAULT_CONFIGS:
        config = DEFAULT_CONFIGS[model_name.lower()]
        config.create_directories()
        return config
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEFAULT_CONFIGS.keys())}")
