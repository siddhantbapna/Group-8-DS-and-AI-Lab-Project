#!/usr/bin/env python3
"""
Preprocess BraTS2023 data and save as .npy files for faster training
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# Add src to path
sys.path.append('src')

from config.config import get_config
from src.preprocessing import BraTS2023Preprocessor
from monai.data import Dataset

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def preprocess_dataset(config, data_path: str, output_path: str, split_name: str = "train"):
    """Preprocess and save dataset"""
    logger = logging.getLogger(__name__)
    
    # Create preprocessor
    preprocessor = BraTS2023Preprocessor(config.data)
    
    # Create data dictionaries
    logger.info(f"Creating data dictionaries for {split_name}...")
    # For validation data, don't require segmentation files
    require_segmentation = (split_name == "train")
    data_dicts = preprocessor.create_data_dicts(data_path, require_segmentation=require_segmentation)
    logger.info(f"Found {len(data_dicts)} samples")
    
    if len(data_dicts) == 0:
        logger.warning(f"No data found in {data_path}")
        return
    
    # Create validation transforms (no augmentation for preprocessing)
    if split_name == "train":
        transforms = preprocessor.get_val_transforms()
    else:
        # For validation data without segmentation, create custom transforms
        from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped
        
        # Only include modalities, not segmentation
        modality_keys = config.data.modality_keys
        transforms = Compose([
            LoadImaged(keys=modality_keys, image_only=True),
            EnsureChannelFirstd(keys=modality_keys),
            Spacingd(keys=modality_keys, pixdim=config.data.target_voxel_spacing, mode="bilinear"),
            ScaleIntensityRanged(keys=modality_keys, a_min=config.data.intensity_range[0], a_max=config.data.intensity_range[1], b_min=config.data.normalized_range[0], b_max=config.data.normalized_range[1], clip=True),
            CropForegroundd(keys=modality_keys, source_key=modality_keys[0], margin=10),
            Resized(keys=modality_keys, spatial_size=config.data.output_shape, mode="area"),
            EnsureTyped(keys=modality_keys, dtype=torch.float32)
        ])
    
    # Create dataset
    dataset = Dataset(data=data_dicts, transform=transforms)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each sample
    logger.info(f"Preprocessing {len(dataset)} samples...")
    
    for i, data in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        try:
            # Get patient ID from original data
            patient_id = data_dicts[i]['patient_id'] if 'patient_id' in data_dicts[i] else f"{split_name}_{i:04d}"
            
            # Create patient directory
            patient_dir = os.path.join(output_path, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Save modalities
            for modality in config.data.modality_keys:
                modality_data = data[modality].numpy()
                np.save(os.path.join(patient_dir, f"{modality}.npy"), modality_data)
            
            # Save segmentation (if available)
            if config.data.seg_key in data:
                seg_data = data[config.data.seg_key].numpy()
                np.save(os.path.join(patient_dir, f"{config.data.seg_key}.npy"), seg_data)
            else:
                # Create dummy segmentation for validation data
                dummy_seg = np.zeros_like(data[config.data.modality_keys[0]].numpy())
                np.save(os.path.join(patient_dir, f"{config.data.seg_key}.npy"), dummy_seg)
            
            # Save metadata
            metadata = {
                'patient_id': patient_id,
                'original_shape': data_dicts[i].get('original_shape', 'unknown'),
                'preprocessed_shape': list(data[config.data.modality_keys[0]].shape),
                'modalities': config.data.modality_keys,
                'seg_key': config.data.seg_key
            }
            
            np.save(os.path.join(patient_dir, "metadata.npy"), metadata)
            
            # Log progress
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(dataset)} samples")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue
    
    logger.info(f"Preprocessing completed for {split_name}. Saved to: {output_path}")
    
    # Save dataset info
    dataset_info = {
        'total_samples': len(dataset),
        'modalities': config.data.modality_keys,
        'seg_key': config.data.seg_key,
        'output_shape': config.data.output_shape,
        'target_voxel_spacing': config.data.target_voxel_spacing,
        'intensity_range': config.data.intensity_range,
        'normalized_range': config.data.normalized_range
    }
    
    np.save(os.path.join(output_path, "dataset_info.npy"), dataset_info)
    logger.info(f"Dataset info saved to: {os.path.join(output_path, 'dataset_info.npy')}")

def create_preprocessed_data_dicts(preprocessed_path: str):
    """Create data dictionaries for preprocessed data"""
    data_dicts = []
    
    for patient_dir in os.listdir(preprocessed_path):
        patient_path = os.path.join(preprocessed_path, patient_dir)
        
        if not os.path.isdir(patient_path):
            continue
            
        # Check if all required files exist
        required_files = ['t1n.npy', 't1c.npy', 't2w.npy', 't2f.npy', 'seg.npy', 'metadata.npy']
        if not all(os.path.exists(os.path.join(patient_path, f)) for f in required_files):
            continue
        
        # Create data dictionary
        data_dict = {
            'patient_id': patient_dir,
            't1n': os.path.join(patient_path, 't1n.npy'),
            't1c': os.path.join(patient_path, 't1c.npy'),
            't2w': os.path.join(patient_path, 't2w.npy'),
            't2f': os.path.join(patient_path, 't2f.npy'),
            'seg': os.path.join(patient_path, 'seg.npy')
        }
        
        data_dicts.append(data_dict)
    
    return data_dicts

def estimate_disk_usage(config, data_path: str):
    """Estimate disk usage for preprocessed data"""
    preprocessor = BraTS2023Preprocessor(config.data)
    data_dicts = preprocessor.create_data_dicts(data_path)
    
    if len(data_dicts) == 0:
        return 0, 0
    
    # Estimate size per sample (4 modalities + 1 segmentation + metadata)
    # Each is 128x128x128 float32 = 128^3 * 4 bytes = ~8.4MB
    size_per_modality = np.prod(config.data.output_shape) * 4  # 4 bytes for float32
    size_per_sample = size_per_modality * 5  # 4 modalities + 1 segmentation
    size_per_sample += 1024  # metadata overhead
    
    total_size = size_per_sample * len(data_dicts)
    
    return total_size, len(data_dicts)

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Preprocess BraTS2023 data')
    parser.add_argument('--model', type=str, default='unet3d', 
                       choices=['unet', 'unet3d', 'resunet', 'nnunet', 'attentionunet', 'vnet'],
                       help='Model to use for configuration')
    parser.add_argument('--train_data_path', type=str, 
                       default='data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData',
                       help='Path to training data')
    parser.add_argument('--val_data_path', type=str,
                       default='data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData', 
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory for preprocessed data')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training data preprocessing')
    parser.add_argument('--skip_val', action='store_true',
                       help='Skip validation data preprocessing')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Get configuration
    config = get_config(args.model)
    
    logger.info("Starting BraTS2023 data preprocessing")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output shape: {config.data.output_shape}")
    logger.info(f"Target voxel spacing: {config.data.target_voxel_spacing}")
    
    # Estimate disk usage
    if not args.skip_train:
        train_size, train_samples = estimate_disk_usage(config, args.train_data_path)
        logger.info(f"Training data: {train_samples} samples, estimated size: {train_size / (1024**3):.2f} GB")
    
    if not args.skip_val:
        val_size, val_samples = estimate_disk_usage(config, args.val_data_path)
        logger.info(f"Validation data: {val_samples} samples, estimated size: {val_size / (1024**3):.2f} GB")
    
    # Create output directories
    train_output = os.path.join(args.output_dir, 'train')
    val_output = os.path.join(args.output_dir, 'val')
    
    # Preprocess training data
    if not args.skip_train:
        logger.info("Preprocessing training data...")
        preprocess_dataset(config, args.train_data_path, train_output, "train")
    
    # Preprocess validation data
    if not args.skip_val:
        logger.info("Preprocessing validation data...")
        preprocess_dataset(config, args.val_data_path, val_output, "val")
    
    logger.info("Preprocessing completed successfully!")
    
    # Print summary
    if not args.skip_train and os.path.exists(train_output):
        train_samples = len([d for d in os.listdir(train_output) if os.path.isdir(os.path.join(train_output, d))])
        logger.info(f"Training data: {train_samples} samples preprocessed")
    
    if not args.skip_val and os.path.exists(val_output):
        val_samples = len([d for d in os.listdir(val_output) if os.path.isdir(os.path.join(val_output, d))])
        logger.info(f"Validation data: {val_samples} samples preprocessed")

if __name__ == "__main__":
    main()
