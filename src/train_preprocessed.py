"""
Training script that uses preprocessed data for faster training
"""

import os
import sys
import logging
from typing import Optional

# Add project root to path
sys.path.append('.')

from config.config import get_config, Config
from src.train import Trainer
from src.preprocessed_dataset import PreprocessedBraTSPreprocessor
from src.preprocessing import BraTS2023Preprocessor
from src.utils import set_seed, setup_logging, get_device_info

class PreprocessedTrainer(Trainer):
    """
    Trainer that uses preprocessed data when available
    """
    
    def __init__(self, config: Config, fold: int = 0, use_preprocessed: bool = True):
        super().__init__(config, fold)
        self.use_preprocessed = use_preprocessed
        
        # Check if preprocessed data is available
        if use_preprocessed:
            self.preprocessed_train_path = os.path.join('./processed_data', 'train')
            self.preprocessed_val_path = os.path.join('./processed_data', 'val')
            
            if not (os.path.exists(self.preprocessed_train_path) and os.path.exists(self.preprocessed_val_path)):
                self.logger.warning("Preprocessed data not found, falling back to real-time preprocessing")
                self.use_preprocessed = False
    
    def setup_data(self):
        """Setup data loaders - use preprocessed if available"""
        if self.use_preprocessed:
            self.logger.info("Using preprocessed data for faster training")
            self.setup_preprocessed_data()
        else:
            self.logger.info("Using real-time preprocessing")
            self.setup_realtime_data()
    
    def setup_preprocessed_data(self):
        """Setup data loaders using preprocessed data"""
        # Create preprocessor for preprocessed data
        self.preprocessor = PreprocessedBraTSPreprocessor(self.config.data)
        
        # Create data dictionaries
        train_data_dicts = self.preprocessor.create_data_dicts(self.preprocessed_train_path)
        
        # Create cross-validation splits
        cv_splits = self.preprocessor.create_cross_validation_splits(train_data_dicts)
        
        if self.fold >= len(cv_splits):
            raise ValueError(f"Fold {self.fold} not available. Total folds: {len(cv_splits)}")
        
        # Get data for current fold
        train_data, val_data = cv_splits[self.fold]
        
        # Create transforms
        train_transforms = self.preprocessor.get_train_transforms(is_training=True)
        val_transforms = self.preprocessor.get_val_transforms()
        
        # Create datasets
        train_dataset, val_dataset = self.preprocessor.create_datasets(
            train_data, val_data, train_transforms, val_transforms
        )
        
        # Create data loaders
        self.train_loader, self.val_loader = self.preprocessor.create_dataloaders(
            train_dataset, val_dataset, 
            self.config.training.batch_size, 
            self.config.system.num_workers
        )
        
        self.logger.info(f"Preprocessed data setup complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def setup_realtime_data(self):
        """Setup data loaders using real-time preprocessing"""
        # Use the original preprocessor
        self.preprocessor = BraTS2023Preprocessor(self.config.data)
        
        # Create data dictionaries
        train_data_dicts = self.preprocessor.create_data_dicts(self.config.data.train_data_path)
        
        # Create cross-validation splits
        cv_splits = self.preprocessor.create_cross_validation_splits(train_data_dicts)
        
        if self.fold >= len(cv_splits):
            raise ValueError(f"Fold {self.fold} not available. Total folds: {len(cv_splits)}")
        
        # Get data for current fold
        train_data, val_data = cv_splits[self.fold]
        
        # Create transforms
        train_transforms = self.preprocessor.get_train_transforms(is_training=True)
        val_transforms = self.preprocessor.get_val_transforms()
        
        # Create datasets
        train_dataset, val_dataset = self.preprocessor.create_datasets(
            train_data, val_data, train_transforms, val_transforms
        )
        
        # Create data loaders
        self.train_loader, self.val_loader = self.preprocessor.create_dataloaders(
            train_dataset, val_dataset, 
            self.config.training.batch_size, 
            self.config.system.num_workers
        )
        
        self.logger.info(f"Real-time data setup complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

def main():
    """Main function for training with preprocessed data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with preprocessed data')
    parser.add_argument('--model', type=str, default='unet3d',
                       choices=['unet', 'unet3d', 'resunet', 'nnunet', 'attentionunet', 'vnet'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--loss_function', type=str, default='weighted_dice_bce',
                       choices=['dice', 'dice_ce', 'dice_focal', 'focal', 'tversky', 'boundary', 'dice_bce', 'weighted_dice_bce'],
                       help='Loss function to use')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number for cross-validation')
    parser.add_argument('--use_preprocessed', action='store_true', default=True,
                       help='Use preprocessed data (default: True)')
    parser.add_argument('--use_realtime', action='store_true',
                       help='Force use of real-time preprocessing')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging('outputs/logs', 'INFO')
    logger.info("Starting training with preprocessed data")
    logger.info(f"Arguments: {args}")
    
    # Print device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Get configuration
    config = get_config(args.model)
    
    # Update configuration with command line arguments
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.loss_function = args.loss_function
    
    # Create directories
    config.create_directories()
    
    # Determine whether to use preprocessed data
    use_preprocessed = args.use_preprocessed and not args.use_realtime
    
    # Create trainer
    trainer = PreprocessedTrainer(config, args.fold, use_preprocessed=use_preprocessed)
    
    # Train
    trainer.train(resume_from_checkpoint=args.resume)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
