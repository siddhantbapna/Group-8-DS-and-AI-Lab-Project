from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
from monai.data import Dataset, CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
	Compose,
	EnsureChannelFirstd,
	LoadImaged,
	Spacingd,
	Orientationd,
	ScaleIntensityRanged,
	CropForegroundd,
	RandSpatialCropd,
	RandFlipd,
	RandAffined,
	ToTensord,
	EnsureTyped,
	MapLabelValueD,
	Resized,
	SqueezeDimd,
)
from monai.utils import set_determinism

from config.config import paths, train_cfg


def _gather_brats_cases(root: str) -> List[Dict[str, str]]:
	cases = []
	for case_dir in sorted(os.listdir(root)):
		full = os.path.join(root, case_dir)
		if not os.path.isdir(full):
			continue
		# BraTS GLI modalities - try different naming patterns
		modalities = [
			("t1", ["*t1n.nii.gz", "*t1n.nii", "*t1*.nii.gz", "*t1*.nii"]),
			("t1ce", ["*t1c.nii.gz", "*t1c.nii", "*t1ce*.nii.gz", "*t1ce*.nii"]),
			("t2", ["*t2w.nii.gz", "*t2w.nii", "*t2*.nii.gz", "*t2*.nii"]),
			("flair", ["*t2f.nii.gz", "*t2f.nii", "*flair*.nii.gz", "*flair*.nii"]),
		]
		entry = {}
		for key, patterns in modalities:
			found = False
			for pattern in patterns:
				matches = glob.glob(os.path.join(full, pattern))
				if matches:
					entry[key] = matches[0]
					found = True
					break
			if not found:
				entry = {}
				break
		# label optional for test
		label_patterns = ["*seg.nii.gz", "*seg.nii", "*seg*.nii.gz", "*seg*.nii"]
		label = None
		for pattern in label_patterns:
			matches = glob.glob(os.path.join(full, pattern))
			if matches:
				label = matches[0]
				break
		if entry:
			if label:
				entry["label"] = label
			cases.append(entry)
	return cases


def get_transforms_3d(train: bool) -> Compose:
	common = [
		LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"], image_only=False),
		EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
		Orientationd(keys=["t1", "t1ce", "t2", "flair", "label"], axcodes="RAS"),
		Spacingd(keys=["t1", "t1ce", "t2", "flair", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear",) * 4 + ("nearest",)),
		ScaleIntensityRanged(keys=["t1", "t1ce", "t2", "flair"], a_min=0, a_max=1400, b_min=0.0, b_max=1.0, clip=True),
		CropForegroundd(keys=["t1", "t1ce", "t2", "flair", "label"], source_key="flair"),
		# Map BraTS labels (0,1,2,3) to (0,1,2) for 3-class segmentation
		MapLabelValueD(keys=["label"], orig_labels=[0, 1, 2, 3], target_labels=[0, 1, 2, 4]),
	]
	if train:
		aug = [
			RandSpatialCropd(keys=["t1", "t1ce", "t2", "flair", "label"], roi_size=train_cfg.slide_infer_roi_3d, random_size=False),
			RandFlipd(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.5, spatial_axis=0),
			RandFlipd(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.5, spatial_axis=1),
			RandFlipd(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.5, spatial_axis=2),
			RandAffined(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.2, rotate_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1), mode=("bilinear",) * 4 + ("nearest",)),
		]
	else:
		aug = []
	return Compose(common + aug + [EnsureTyped(keys=["t1", "t1ce", "t2", "flair", "label"])])


def get_transforms_2d(train: bool, spatial_dims: int = 2) -> Compose:
	common = [
		LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"], image_only=False),
		EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
		Orientationd(keys=["t1", "t1ce", "t2", "flair", "label"], axcodes="RAS"),
		Spacingd(keys=["t1", "t1ce", "t2", "flair", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear",) * 4 + ("nearest",)),
		ScaleIntensityRanged(keys=["t1", "t1ce", "t2", "flair"], a_min=0, a_max=1400, b_min=0.0, b_max=1.0, clip=True),
		CropForegroundd(keys=["t1", "t1ce", "t2", "flair", "label"], source_key="flair"),
		# Map BraTS labels (0,1,2,3) to (0,1,2) for 3-class segmentation
		MapLabelValueD(keys=["label"], orig_labels=[0, 1, 2, 3], target_labels=[0, 1, 2, 2]),
	]
	if train:
		aug = [
			# Crop to fixed 2D size and resize to ensure consistent dimensions
			RandSpatialCropd(keys=["t1", "t1ce", "t2", "flair", "label"], roi_size=train_cfg.slide_infer_roi_2d + [1], random_size=False),
			Resized(keys=["t1", "t1ce", "t2", "flair", "label"], spatial_size=train_cfg.slide_infer_roi_2d + [1], mode=("bilinear",) * 4 + ("nearest",)),
			SqueezeDimd(keys=["t1", "t1ce", "t2", "flair", "label"], dim=-1),  # Remove the last dimension to make it 2D
			RandFlipd(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.5, spatial_axis=0),
			RandFlipd(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.5, spatial_axis=1),
			RandAffined(keys=["t1", "t1ce", "t2", "flair", "label"], prob=0.2, rotate_range=(0.05, 0.05), scale_range=(0.1, 0.1), mode=("bilinear",) * 4 + ("nearest",)),
		]
	else:
		aug = []
	
	# For 2D training, we need to resize and squeeze even for validation
	if spatial_dims == 2:
		aug.append(Resized(keys=["t1", "t1ce", "t2", "flair", "label"], spatial_size=train_cfg.slide_infer_roi_2d + [1], mode=("bilinear",) * 4 + ("nearest",)))
		aug.append(SqueezeDimd(keys=["t1", "t1ce", "t2", "flair", "label"], dim=-1))  # Remove the last dimension to make it 2D
	
	return Compose(common + aug + [EnsureTyped(keys=["t1", "t1ce", "t2", "flair", "label"])])


def create_datasets(spatial_dims: int = 3, cache_rate: float = 0.2) -> Tuple[Dataset, Dataset]:
	set_determinism(seed=train_cfg.seed)
	train_list = _gather_brats_cases(paths.data_train)
	# simple split
	num_total = len(train_list)
	num_val = max(1, int(num_total * train_cfg.val_ratio))
	val_list = train_list[:num_val]
	tr_list = train_list[num_val:]

	transforms = get_transforms_3d if spatial_dims == 3 else lambda train: get_transforms_2d(train, spatial_dims)
	train_ds = CacheDataset(tr_list, transform=transforms(train=True), cache_rate=cache_rate, num_workers=train_cfg.num_workers)
	val_ds = CacheDataset(val_list, transform=transforms(train=False), cache_rate=0.0, num_workers=train_cfg.num_workers)
	return train_ds, val_ds


def create_loaders(train_ds: Dataset, val_ds: Dataset, spatial_dims: int) -> Tuple[DataLoader, DataLoader]:
	batch = train_cfg.batch_size_3d if spatial_dims == 3 else train_cfg.batch_size_2d
	train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=train_cfg.num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=train_cfg.num_workers, pin_memory=True)
	return train_loader, val_loader
