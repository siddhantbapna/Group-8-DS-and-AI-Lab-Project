"""
Quick training script for testing the pipeline
"""
import sys
import os
sys.path.append('src')

from config.config import get_config
from src.train import Trainer
from src.utils import set_seed, setup_logging, get_device_info

def quick_train():
    """Quick training for testing"""
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging("outputs/logs", "INFO")
    logger.info("Starting quick training test")
    
    # Print device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Get configuration for a lightweight model
    config = get_config('unet3d')
    
    # Modify config for quick training
    config.training.num_epochs = 5  # Just 5 epochs for testing
    config.training.batch_size = 1  # Small batch size
    config.data.output_shape = (64, 64, 64)  # Smaller input size
    config.model.features = [16, 32, 64, 128]  # Smaller model
    
    # Create directories
    config.create_directories()
    
    # Create trainer
    trainer = Trainer(config, fold=0)
    
    try:
        # Train
        trainer.train()
        logger.info("Quick training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    quick_train()
