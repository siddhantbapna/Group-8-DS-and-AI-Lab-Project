"""
Training script with TensorBoard logging and proper BraTS label mapping
"""
import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.config import get_config
from src.models import create_model
from src.preprocessing import BraTS2023Preprocessor
from src.utils import set_seed, setup_logging, get_device_info
from src.checkpoints import CheckpointManager
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

def simple_loss_function(outputs, targets):
    """Simple cross-entropy loss"""
    return F.cross_entropy(outputs, targets)

def calculate_accuracy(outputs, targets):
    """Calculate accuracy"""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).float()
    accuracy = correct.mean()
    return accuracy.item()

def train_with_tensorboard():
    """Train with real BraTS2023 data and TensorBoard logging"""
    print("Training with Real BraTS2023 Data + TensorBoard")
    print("=" * 60)
    
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging("outputs/logs", "INFO")
    logger.info("Starting training with BraTS2023 data and TensorBoard logging")
    
    # Print device information
    device_info = get_device_info()
    logger.info(f"Device information: {device_info}")
    
    # Get configuration
    config = get_config('unet3d')
    config.training.batch_size = 2
    config.training.num_epochs = 10  # More epochs for better results
    config.data.output_shape = (128, 128, 128)  # Larger for better quality
    config.model.features = [32, 64, 128, 256]  # Larger model
    
    # Create directories
    config.create_directories()
    
    # Setup device
    device = torch.device(config.system.device)
    
    # Create model
    model = create_model(
        config.model.model_name,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        features=config.model.features,
        dropout=config.model.dropout
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager("checkpoints", "unet3d_tensorboard")
    checkpoint_manager.save_model_summary(model, config.to_dict())
    
    # Setup TensorBoard
    tensorboard_dir = "outputs/logs/tensorboard_fold_0"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    # Setup data
    preprocessor = BraTS2023Preprocessor(config.data)
    
    # Create data dictionaries
    train_data_dicts = preprocessor.create_data_dicts(config.data.train_data_path)
    logger.info(f"Found {len(train_data_dicts)} training samples")
    
    # Create cross-validation splits
    cv_splits = preprocessor.create_cross_validation_splits(train_data_dicts)
    train_data, val_data = cv_splits[0]  # Use first fold
    
    logger.info(f"Using fold 0: {len(train_data)} train, {len(val_data)} val samples")
    
    # Create transforms
    train_transforms = preprocessor.get_train_transforms(is_training=True)
    val_transforms = preprocessor.get_val_transforms()
    
    # Create datasets
    train_dataset, val_dataset = preprocessor.create_datasets(
        train_data, val_data, train_transforms, val_transforms
    )
    
    # Create data loaders
    train_loader, val_loader = preprocessor.create_dataloaders(
        train_dataset, val_dataset, config.training.batch_size, config.system.num_workers
    )
    
    logger.info(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Training loop
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Stack all modalities
            modalities = []
            for modality in config.data.modality_keys:
                modalities.append(batch_data[modality])
            inputs = torch.cat(modalities, dim=1).to(device)
            targets = batch_data[config.data.seg_key].to(device)
            
            # Map BraTS labels to 3-class output: 0->0, 1->1, 2->1, 3->2
            targets_mapped = torch.zeros_like(targets)
            targets_mapped[targets == 0] = 0  # Background
            targets_mapped[targets == 1] = 1  # NCR/NET -> class 1
            targets_mapped[targets == 2] = 1  # ED -> class 1 (combine with NCR/NET)
            targets_mapped[targets == 3] = 2  # ET -> class 2
            targets = targets_mapped
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = simple_loss_function(outputs, targets)
            accuracy = calculate_accuracy(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_accuracy += accuracy
            train_batches += 1
            global_step += 1
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Accuracy/Train', accuracy, global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
            
            # Log every 50 batches
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
        
        avg_train_loss = train_loss / train_batches
        avg_train_accuracy = train_accuracy / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                # Stack all modalities
                modalities = []
                for modality in config.data.modality_keys:
                    modalities.append(batch_data[modality])
                inputs = torch.cat(modalities, dim=1).to(device)
                targets = batch_data[config.data.seg_key].to(device)
                
                # Map BraTS labels to 3-class output: 0->0, 1->1, 2->1, 3->2
                targets_mapped = torch.zeros_like(targets)
                targets_mapped[targets == 0] = 0  # Background
                targets_mapped[targets == 1] = 1  # NCR/NET -> class 1
                targets_mapped[targets == 2] = 1  # ED -> class 1 (combine with NCR/NET)
                targets_mapped[targets == 3] = 2  # ET -> class 2
                targets = targets_mapped
                
                outputs = model(inputs)
                loss = simple_loss_function(outputs, targets)
                accuracy = calculate_accuracy(outputs, targets)
                
                val_loss += loss.item()
                val_accuracy += accuracy
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_accuracy / val_batches
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Loss/Train_Avg', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/Validation', avg_val_accuracy, epoch)
        writer.add_scalar('Accuracy/Train_Avg', avg_train_accuracy, epoch)
        
        # Log results
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
        
        # Save checkpoint if best
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_loss:.4f}")
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            epoch, model, optimizer, scheduler,
            {'val_loss': avg_val_loss, 'train_loss': avg_train_loss},
            avg_val_loss, is_best
        )
        
        # Update scheduler
        scheduler.step()
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info(f"Training completed! Best validation loss: {best_loss:.4f}")
    logger.info(f"TensorBoard logs saved to: {tensorboard_dir}")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        # Get a sample from validation set
        sample_batch = next(iter(val_loader))
        modalities = []
        for modality in config.data.modality_keys:
            modalities.append(sample_batch[modality])
        x_test = torch.cat(modalities, dim=1).to(device)
        
        outputs = model(x_test)
        predictions = torch.argmax(outputs, dim=1)
        
        logger.info(f"Test inference successful!")
        logger.info(f"Input shape: {x_test.shape}")
        logger.info(f"Output shape: {outputs.shape}")
        logger.info(f"Prediction shape: {predictions.shape}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"TensorBoard logs available at: {tensorboard_dir}")
    print("Run: tensorboard --logdir outputs/logs/tensorboard_fold_0")
    print("Your BraTS2023 training pipeline is working!")

if __name__ == "__main__":
    train_with_tensorboard()
