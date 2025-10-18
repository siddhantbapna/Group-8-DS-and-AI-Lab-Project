"""
Preprocessing pipeline for BraTS2023 dataset using MONAI transforms
"""
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, Resized, ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureTyped, RandFlipd, RandRotate90d, RandShiftIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandSpatialCropd, RandZoomd, RandAffined, RandBiasFieldd,
    RandCoarseDropoutd, RandCoarseShuffled, RandHistogramShiftd,
    RandAdjustContrastd, RandGibbsNoised,
    RandKSpaceSpikeNoised, RandRicianNoised, RandSimulateLowResolutiond
)
from monai.data import Dataset, DataLoader, decollate_batch
# from monai.utils import set_deterministic  # Not available in this MONAI version
from sklearn.model_selection import KFold
import nibabel as nib
from config.config import DataConfig

# Fallback for deterministic setting
def set_deterministic(seed=42):
    """Fallback deterministic setting function"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_brain_mask(data_dict: Dict[str, Any]) -> torch.Tensor:
    """
    Create a brain mask to exclude background from training.
    Uses T1 modality to identify brain tissue.
    """
    # Get T1 modality (usually the most reliable for brain tissue)
    t1_data = data_dict.get('t1n', None)
    if t1_data is None:
        # Fallback to first available modality
        for key in ['t1c', 't2w', 't2f']:
            if key in data_dict:
                t1_data = data_dict[key]
                break
    
    if t1_data is None:
        raise ValueError("No modality data found for brain mask creation")
    
    # Convert to numpy if it's a tensor
    if isinstance(t1_data, torch.Tensor):
        t1_np = t1_data.numpy()
    else:
        t1_np = t1_data
    
    # Create brain mask using intensity thresholding
    # Brain tissue typically has intensity > 0
    brain_mask = t1_np > 0
    
    # Apply morphological operations to clean up the mask
    from scipy import ndimage
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((3,3,3)))
    
    return torch.from_numpy(brain_mask.astype(np.float32))

def apply_brain_mask(data_dict: Dict[str, Any], brain_mask: torch.Tensor) -> Dict[str, Any]:
    """
    Apply brain mask to all modalities and segmentation.
    Sets background pixels to 0.
    """
    masked_data = {}
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            # Apply mask to each channel
            if value.dim() == 4:  # [C, H, W, D]
                masked_value = value * brain_mask.unsqueeze(0)
            elif value.dim() == 3:  # [H, W, D]
                masked_value = value * brain_mask
            else:
                masked_value = value
            masked_data[key] = masked_value
        else:
            masked_data[key] = value
    
    return masked_data

class BrainOnlyTransform:
    """
    Custom transform to focus training only on brain tissue.
    Applies brain mask and optionally samples foreground pixels.
    """
    
    def __init__(self, brain_mask_method: str = "intensity", 
                 background_weight: float = 0.1,
                 foreground_sampling: bool = False):
        self.brain_mask_method = brain_mask_method
        self.background_weight = background_weight
        self.foreground_sampling = foreground_sampling
    
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply brain-only processing to the data"""
        # Create brain mask
        brain_mask = self._create_brain_mask(data_dict)
        
        # Apply mask to all modalities and segmentation
        masked_data = apply_brain_mask(data_dict, brain_mask)
        
        # Add brain mask to the data for potential use in loss calculation
        masked_data['brain_mask'] = brain_mask
        
        return masked_data
    
    def _create_brain_mask(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Create brain mask using specified method"""
        if self.brain_mask_method == "intensity":
            return self._intensity_based_mask(data_dict)
        elif self.brain_mask_method == "otsu":
            return self._otsu_based_mask(data_dict)
        elif self.brain_mask_method == "adaptive":
            return self._adaptive_based_mask(data_dict)
        else:
            raise ValueError(f"Unknown brain mask method: {self.brain_mask_method}")
    
    def _intensity_based_mask(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Create mask based on intensity thresholding"""
        t1_data = data_dict.get('t1n', None)
        if t1_data is None:
            for key in ['t1c', 't2w', 't2f']:
                if key in data_dict:
                    t1_data = data_dict[key]
                    break
        
        if isinstance(t1_data, torch.Tensor):
            t1_np = t1_data.numpy()
        else:
            t1_np = t1_data
        
        # Simple intensity thresholding
        brain_mask = t1_np > 0
        
        # Clean up the mask
        from scipy import ndimage
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((3,3,3)))
        
        return torch.from_numpy(brain_mask.astype(np.float32))
    
    def _otsu_based_mask(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Create mask using Otsu thresholding"""
        from skimage.filters import threshold_otsu
        
        t1_data = data_dict.get('t1n', None)
        if t1_data is None:
            for key in ['t1c', 't2w', 't2f']:
                if key in data_dict:
                    t1_data = data_dict[key]
                    break
        
        if isinstance(t1_data, torch.Tensor):
            t1_np = t1_data.numpy()
        else:
            t1_np = t1_data
        
        # Apply Otsu thresholding
        threshold = threshold_otsu(t1_np)
        brain_mask = t1_np > threshold
        
        # Clean up the mask
        from scipy import ndimage
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((3,3,3)))
        
        return torch.from_numpy(brain_mask.astype(np.float32))
    
    def _adaptive_based_mask(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Create mask using adaptive thresholding"""
        from skimage.filters import threshold_local
        
        t1_data = data_dict.get('t1n', None)
        if t1_data is None:
            for key in ['t1c', 't2w', 't2f']:
                if key in data_dict:
                    t1_data = data_dict[key]
                    break
        
        if isinstance(t1_data, torch.Tensor):
            t1_np = t1_data.numpy()
        else:
            t1_np = t1_data
        
        # Apply adaptive thresholding
        threshold = threshold_local(t1_np, block_size=35, offset=0.1)
        brain_mask = t1_np > threshold
        
        # Clean up the mask
        from scipy import ndimage
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_opening(brain_mask, structure=np.ones((3,3,3)))
        
        return torch.from_numpy(brain_mask.astype(np.float32))

class BraTS2023Preprocessor:
    """Preprocessor for BraTS2023 dataset"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.modality_keys = config.modality_keys
        self.all_keys = config.all_keys
        self.seg_key = config.seg_key
        
    def get_train_transforms(self, is_training: bool = True) -> Compose:
        """Get training transforms with augmentation"""
        transforms = [
            # Load images
            LoadImaged(keys=self.all_keys, image_only=True),
            EnsureChannelFirstd(keys=self.all_keys),
            
            # Spatial normalization
            Spacingd(keys=self.modality_keys, pixdim=self.config.target_voxel_spacing, mode="bilinear"),
            Spacingd(keys=[self.seg_key], pixdim=self.config.target_voxel_spacing, mode="nearest"),
            
            # Intensity normalization
            ScaleIntensityRanged(
                keys=self.modality_keys,
                a_min=self.config.intensity_range[0],
                a_max=self.config.intensity_range[1],
                b_min=self.config.normalized_range[0],
                b_max=self.config.normalized_range[1],
                clip=True
            ),
            
            # Crop foreground
            CropForegroundd(keys=self.all_keys, source_key='t1n', margin=10),
            
            # Resize to target shape
            Resized(keys=self.modality_keys, spatial_size=self.config.output_shape, mode="area"),
            Resized(keys=[self.seg_key], spatial_size=self.config.output_shape, mode="nearest"),
            
            # Convert segmentation to multi-channel
            ConvertToMultiChannelBasedOnBratsClassesd(keys=self.seg_key),
            
            # Ensure correct data type
            EnsureTyped(keys=self.all_keys, dtype=torch.float32)
        ]
        
        # Add brain-only training transform if enabled
        if is_training and self.config.brain_only_training:
            transforms.append(
                BrainOnlyTransform(
                    brain_mask_method=self.config.brain_mask_method,
                    background_weight=self.config.background_weight,
                    foreground_sampling=self.config.foreground_sampling
                )
            )
        
        if is_training:
            # Add augmentation transforms
            augmentation_transforms = [
                # Spatial augmentations
                RandFlipd(keys=self.all_keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=self.all_keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=self.all_keys, prob=0.5, spatial_axis=2),
                RandRotate90d(keys=self.all_keys, prob=0.5, max_k=3),
                RandAffined(
                    keys=self.all_keys,
                    prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                    translate_range=(10, 10, 10),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")
                ),
                RandZoomd(keys=self.all_keys, prob=0.5, min_zoom=0.8, max_zoom=1.2, mode=("area", "area", "area", "area", "nearest")),
                
                # Intensity augmentations
                RandShiftIntensityd(keys=self.modality_keys, prob=0.5, offsets=0.1),
                RandScaleIntensityd(keys=self.modality_keys, prob=0.5, factors=0.1),
                RandGaussianNoised(keys=self.modality_keys, prob=0.5, std=0.01),
                RandGaussianSmoothd(keys=self.modality_keys, prob=0.5, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
                RandAdjustContrastd(keys=self.modality_keys, prob=0.5, gamma=(0.8, 1.2)),
                RandHistogramShiftd(keys=self.modality_keys, prob=0.5, num_control_points=10),
                
                # Advanced augmentations
                RandBiasFieldd(keys=self.modality_keys, prob=0.3, coeff_range=(0.0, 0.1)),
                RandCoarseDropoutd(keys=self.modality_keys, prob=0.3, holes=6, spatial_size=5, fill_value=0),
                RandGibbsNoised(keys=self.modality_keys, prob=0.3, alpha=(0.5, 1.0)),
                RandKSpaceSpikeNoised(keys=self.modality_keys, prob=0.3, intensity_range=(10, 13)),
                RandRicianNoised(keys=self.modality_keys, prob=0.3, std=0.1),
                RandSimulateLowResolutiond(keys=self.modality_keys, prob=0.3, zoom_range=(0.5, 1.0)),
            ]
            
            transforms.extend(augmentation_transforms)
        
        return Compose(transforms)
    
    def get_val_transforms(self) -> Compose:
        """Get validation transforms (no augmentation)"""
        return self.get_train_transforms(is_training=False)
    
    def create_data_dicts(self, data_path: str) -> List[Dict[str, str]]:
        """Create data dictionaries for MONAI dataset"""
        data_dicts = []
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        for patient_id in patient_dirs:
            patient_path = os.path.join(data_path, patient_id)
            
            # Find modality files
            data_dict = {}
            found_all_modalities = True
            
            for modality in self.modality_keys:
                # Try both naming conventions: underscore and hyphen
                modality_file_underscore = os.path.join(patient_path, f"{patient_id}_{modality}.nii.gz")
                modality_file_hyphen = os.path.join(patient_path, f"{patient_id}-{modality}.nii.gz")
                
                if os.path.exists(modality_file_underscore):
                    data_dict[modality] = modality_file_underscore
                elif os.path.exists(modality_file_hyphen):
                    data_dict[modality] = modality_file_hyphen
                else:
                    found_all_modalities = False
                    break
            
            # Find segmentation file
            seg_file_underscore = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
            seg_file_hyphen = os.path.join(patient_path, f"{patient_id}-seg.nii.gz")
            
            if os.path.exists(seg_file_underscore):
                data_dict[self.seg_key] = seg_file_underscore
            elif os.path.exists(seg_file_hyphen):
                data_dict[self.seg_key] = seg_file_hyphen
            else:
                found_all_modalities = False
            
            if found_all_modalities:
                data_dicts.append(data_dict)
        
        return data_dicts
    
    def create_cross_validation_splits(self, data_dicts: List[Dict[str, str]]) -> List[Tuple[List[Dict], List[Dict]]]:
        """Create cross-validation splits"""
        kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_seed)
        splits = []
        
        indices = np.arange(len(data_dicts))
        for train_idx, val_idx in kfold.split(indices):
            train_data = [data_dicts[i] for i in train_idx]
            val_data = [data_dicts[i] for i in val_idx]
            splits.append((train_data, val_data))
        
        return splits
    
    def create_datasets(self, train_data: List[Dict], val_data: List[Dict], 
                       train_transforms: Compose, val_transforms: Compose) -> Tuple[Dataset, Dataset]:
        """Create MONAI datasets"""
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=val_transforms)
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, 
                          batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders"""
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
    
    def preprocess_and_save(self, data_path: str, output_path: str, 
                           transforms: Compose, split_name: str = "train"):
        """Preprocess data and save to disk"""
        data_dicts = self.create_data_dicts(data_path)
        dataset = Dataset(data=data_dicts, transform=transforms)
        
        os.makedirs(output_path, exist_ok=True)
        
        for i, data in enumerate(dataset):
            # Save preprocessed data
            patient_id = f"{split_name}_{i:04d}"
            save_path = os.path.join(output_path, patient_id)
            os.makedirs(save_path, exist_ok=True)
            
            # Save modalities
            for modality in self.modality_keys:
                modality_data = data[modality].numpy()
                np.save(os.path.join(save_path, f"{modality}.npy"), modality_data)
            
            # Save segmentation
            seg_data = data[self.seg_key].numpy()
            np.save(os.path.join(save_path, f"{self.seg_key}.npy"), seg_data)
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(dataset)} samples for {split_name}")

def create_preprocessing_pipeline(config: DataConfig) -> BraTS2023Preprocessor:
    """Create preprocessing pipeline"""
    return BraTS2023Preprocessor(config)

# Example usage
if __name__ == "__main__":
    from config.config import get_config
    
    config = get_config('unet')
    preprocessor = create_preprocessing_pipeline(config.data)
    
    # Create data dictionaries
    train_data_dicts = preprocessor.create_data_dicts(config.data.train_data_path)
    print(f"Found {len(train_data_dicts)} training samples")
    
    # Create cross-validation splits
    cv_splits = preprocessor.create_cross_validation_splits(train_data_dicts)
    print(f"Created {len(cv_splits)} cross-validation splits")
    
    # Example of creating datasets and dataloaders for first fold
    train_data, val_data = cv_splits[0]
    train_transforms = preprocessor.get_train_transforms(is_training=True)
    val_transforms = preprocessor.get_val_transforms()
    
    train_dataset, val_dataset = preprocessor.create_datasets(
        train_data, val_data, train_transforms, val_transforms
    )
    
    train_loader, val_loader = preprocessor.create_dataloaders(
        train_dataset, val_dataset, config.training.batch_size, config.system.num_workers
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
