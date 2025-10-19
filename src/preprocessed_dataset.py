"""
Preprocessed dataset loader for faster training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
from monai.transforms import Compose, LoadImaged, EnsureTyped
import logging

class PreprocessedBraTSDataset(Dataset):
    """
    Dataset for loading preprocessed BraTS data from .npy files
    """
    
    def __init__(self, data_dicts: List[Dict[str, str]], transforms: Optional[Compose] = None):
        """
        Initialize preprocessed dataset
        
        Args:
            data_dicts: List of dictionaries with paths to .npy files
            transforms: Optional transforms to apply
        """
        self.data_dicts = data_dicts
        self.transforms = transforms
        
        # Load dataset info if available
        self.dataset_info = self._load_dataset_info()
        
    def _load_dataset_info(self) -> Optional[Dict]:
        """Load dataset info from the first sample's directory"""
        if not self.data_dicts:
            return None
            
        # Get the parent directory of the first sample
        first_sample_path = self.data_dicts[0]['t1n']
        dataset_dir = os.path.dirname(os.path.dirname(first_sample_path))
        info_path = os.path.join(dataset_dir, 'dataset_info.npy')
        
        if os.path.exists(info_path):
            return np.load(info_path, allow_pickle=True).item()
        return None
    
    def __len__(self) -> int:
        return len(self.data_dicts)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Load preprocessed data"""
        data_dict = self.data_dicts[index]
        
        # Load all modalities
        loaded_data = {}
        for key, file_path in data_dict.items():
            if key == 'patient_id':
                loaded_data[key] = file_path
                continue
                
            # Load .npy file
            data = np.load(file_path)
            
            # Convert to torch tensor
            if data.ndim == 3:
                # Add channel dimension
                data = torch.from_numpy(data).unsqueeze(0).float()
            else:
                data = torch.from_numpy(data).float()
            
            loaded_data[key] = data
        
        # Apply transforms if provided
        if self.transforms:
            loaded_data = self.transforms(loaded_data)
        
        return loaded_data

class PreprocessedBraTSPreprocessor:
    """
    Preprocessor for loading preprocessed BraTS data
    """
    
    def __init__(self, config):
        self.config = config
        self.modality_keys = config.modality_keys
        self.seg_key = config.seg_key
        self.all_keys = config.all_keys
        
    def create_data_dicts(self, preprocessed_path: str) -> List[Dict[str, str]]:
        """Create data dictionaries for preprocessed data"""
        data_dicts = []
        
        if not os.path.exists(preprocessed_path):
            raise ValueError(f"Preprocessed data path does not exist: {preprocessed_path}")
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(preprocessed_path) 
                       if os.path.isdir(os.path.join(preprocessed_path, d))]
        
        for patient_dir in sorted(patient_dirs):
            patient_path = os.path.join(preprocessed_path, patient_dir)
            
            # Check if all required files exist
            required_files = [f"{mod}.npy" for mod in self.modality_keys] + [f"{self.seg_key}.npy"]
            if not all(os.path.exists(os.path.join(patient_path, f)) for f in required_files):
                logging.warning(f"Missing files for patient {patient_dir}, skipping")
                continue
            
            # Create data dictionary
            data_dict = {'patient_id': patient_dir}
            for modality in self.modality_keys:
                data_dict[modality] = os.path.join(patient_path, f"{modality}.npy")
            data_dict[self.seg_key] = os.path.join(patient_path, f"{self.seg_key}.npy")
            
            data_dicts.append(data_dict)
        
        logging.info(f"Found {len(data_dicts)} preprocessed samples in {preprocessed_path}")
        return data_dicts
    
    def get_train_transforms(self, is_training: bool = True) -> Compose:
        """Get training transforms (minimal since data is already preprocessed)"""
        transforms = [
            # Data is already preprocessed, just ensure correct type
            EnsureTyped(keys=self.all_keys, dtype=torch.float32)
        ]
        
        if is_training:
            # Add augmentation transforms
            from monai.transforms import (
                RandFlipd, RandRotate90d, RandShiftIntensityd,
                RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
                RandHistogramShiftd, RandBiasFieldd, RandCoarseDropoutd,
                RandGibbsNoised, RandKSpaceSpikeNoised, RandRicianNoised,
                RandSimulateLowResolutiond, RandAffined, RandZoomd
            )
            
            # Add augmentation transforms
            augmentation_transforms = [
                # Spatial augmentations
                RandFlipd(keys=self.all_keys, prob=0.5, spatial_axis=0),
                RandRotate90d(keys=self.all_keys, prob=0.5, max_k=3),
                RandAffined(
                    keys=self.all_keys,
                    prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                    translate_range=(10, 10, 10),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")
                ),
                RandZoomd(
                    keys=self.all_keys,
                    prob=0.5,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest")
                ),
                
                # Intensity augmentations
                RandShiftIntensityd(keys=self.modality_keys, prob=0.5, offsets=0.1),
                RandGaussianNoised(keys=self.modality_keys, prob=0.5, std=0.01),
                RandGaussianSmoothd(keys=self.modality_keys, prob=0.5, sigma_x=(0.5, 1.0)),
                RandAdjustContrastd(keys=self.modality_keys, prob=0.5, gamma=(0.8, 1.2)),
                RandHistogramShiftd(keys=self.modality_keys, prob=0.5, num_control_points=10),
                RandBiasFieldd(keys=self.modality_keys, prob=0.5, degree=3),
                RandCoarseDropoutd(keys=self.modality_keys, prob=0.5, holes=10, spatial_size=8),
                RandGibbsNoised(keys=self.modality_keys, prob=0.5, alpha=(0.5, 1.0)),
                RandKSpaceSpikeNoised(keys=self.modality_keys, prob=0.5, intensity_range=(10, 20)),
                RandRicianNoised(keys=self.modality_keys, prob=0.5, std=0.01),
                RandSimulateLowResolutiond(keys=self.modality_keys, prob=0.5, zoom_range=(0.5, 1.0))
            ]
            
            transforms.extend(augmentation_transforms)
        
        return Compose(transforms)
    
    def get_val_transforms(self) -> Compose:
        """Get validation transforms (minimal since data is already preprocessed)"""
        return Compose([
            EnsureTyped(keys=self.all_keys, dtype=torch.float32)
        ])
    
    def create_cross_validation_splits(self, data_dicts: List[Dict[str, str]]) -> List[tuple]:
        """Create cross-validation splits"""
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_seed)
        
        splits = []
        for train_indices, val_indices in kfold.split(data_dicts):
            train_data = [data_dicts[i] for i in train_indices]
            val_data = [data_dicts[i] for i in val_indices]
            splits.append((train_data, val_data))
        
        return splits
    
    def create_datasets(self, train_data: List[Dict], val_data: List[Dict], 
                       train_transforms: Compose, val_transforms: Compose):
        """Create datasets"""
        train_dataset = PreprocessedBraTSDataset(train_data, train_transforms)
        val_dataset = PreprocessedBraTSDataset(val_data, val_transforms)
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, batch_size: int, num_workers: int):
        """Create data loaders"""
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    from config.config import get_config
    
    config = get_config('unet3d')
    preprocessor = PreprocessedBraTSPreprocessor(config.data)
    
    # Load preprocessed data
    train_data_dicts = preprocessor.create_data_dicts('processed_data/train')
    print(f"Found {len(train_data_dicts)} training samples")
    
    # Create datasets
    train_transforms = preprocessor.get_train_transforms(is_training=True)
    val_transforms = preprocessor.get_val_transforms()
    
    train_dataset = PreprocessedBraTSDataset(train_data_dicts, train_transforms)
    print(f"Created dataset with {len(train_dataset)} samples")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"T1N shape: {sample['t1n'].shape}")
    print(f"Segmentation shape: {sample['seg'].shape}")
