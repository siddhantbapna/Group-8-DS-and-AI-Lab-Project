"""
Main training script for brain MRI segmentation project
"""
import argparse
import os
import sys
import logging
from typing import Optional

# Add src to path
sys.path.append('src')

from config.config import get_config, Config
from src.train import Trainer, CrossValidationTrainer
from src.inference import InferencePipeline
from src.utils import set_seed, setup_logging, get_device_info

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Brain MRI Segmentation Training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='unet3d',
                       choices=['unet', 'unet3d', 'resunet', 'nnunet', 'attentionunet', 'vnet'],
                       help='Model architecture to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--loss_function', type=str, default='dice',
                       choices=['dice', 'dice_ce', 'dice_focal', 'focal', 'tversky', 'boundary', 'dice_bce', 'weighted_dice_bce'],
                       help='Loss function to use')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData',
                       help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default='data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData',
                       help='Path to validation data')
    parser.add_argument('--output_shape', type=int, nargs=3, default=[128, 128, 128],
                       help='Output shape for preprocessing')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'cv', 'inference'],
                       help='Training mode')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number for cross-validation')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    # Checkpoint and resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    # Inference parameters
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for inference')
    parser.add_argument('--inference_data_path', type=str, default=None,
                       help='Path to data for inference')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_dir', type=str, default='outputs/logs',
                       help='Directory for logs')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.log_level)
    logger.info("Starting brain MRI segmentation training")
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
    config.training.use_amp = args.use_amp
    
    config.data.train_data_path = args.data_path
    config.data.val_data_path = args.val_data_path
    config.data.output_shape = tuple(args.output_shape)
    config.data.n_folds = args.n_folds
    
    config.system.device = args.device
    config.system.num_workers = args.num_workers
    config.system.checkpoint_dir = args.checkpoint_dir
    config.system.output_dir = args.output_dir
    config.system.log_dir = args.log_dir
    
    # Create directories
    config.create_directories()
    
    if args.mode == 'train':
        # Single fold training
        logger.info(f"Starting training for fold {args.fold}")
        trainer = Trainer(config, args.fold)
        trainer.train(resume_from_checkpoint=args.resume)
        
    elif args.mode == 'cv':
        # Cross-validation training
        logger.info(f"Starting cross-validation training with {args.n_folds} folds")
        cv_trainer = CrossValidationTrainer(config)
        cv_trainer.train_all_folds(resume_fold=args.fold if args.resume else None)
        
    elif args.mode == 'inference':
        # Inference
        if args.model_path is None:
            logger.error("Model path is required for inference")
            return
        
        if args.inference_data_path is None:
            logger.error("Inference data path is required for inference")
            return
        
        logger.info(f"Starting inference with model: {args.model_path}")
        pipeline = InferencePipeline(config, args.model_path, args.output_dir)
        results = pipeline.process_dataset(args.inference_data_path, use_tta=True, save_predictions=True)
        
        logger.info("Inference completed")
        logger.info(f"Results: {results}")
    
    logger.info("Training/inference completed successfully")

if __name__ == "__main__":
    main()
