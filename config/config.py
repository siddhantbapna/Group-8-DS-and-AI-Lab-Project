import os
from dataclasses import dataclass
from typing import List


@dataclass
class Paths:
	project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	data_train: str = os.path.join(project_root, "data", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
	data_test: str = os.path.join(project_root, "data", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TestingData")
	processed: str = os.path.join(project_root, "data", "processed_data")
	outputs: str = os.path.join(project_root, "outputs")
	logs: str = os.path.join(outputs, "logs")
	models: str = os.path.join(outputs, "models")
	predictions: str = os.path.join(outputs, "predictions")


@dataclass
class TrainConfig:
	seed: int = 42
	folds: int = 5
	selected_fold: int = 0
	val_ratio: float = 0.2
	batch_size_2d: int = 8
	batch_size_3d: int = 1
	num_workers: int = 4
	max_epochs: int = 500
	lr: float = 1e-4
	weight_decay: float = 1e-5
	amp: bool = True
	clip_grad: float = 1.0  # Gradient clipping to prevent explosion
	early_stopping_patience: int = 20  # Stop if no improvement for N epochs
	slide_infer_roi_2d: List[int] = None
	slide_infer_roi_3d: List[int] = None
	overlap_2d: float = 0.25
	overlap_3d: float = 0.5

	def __post_init__(self):
		if self.slide_infer_roi_2d is None:
			self.slide_infer_roi_2d = [256, 256]
		if self.slide_infer_roi_3d is None:
			# Suitable for 8GB VRAM; adjust if needed
			self.slide_infer_roi_3d = [96, 96, 96]


@dataclass
class ModelConfig:
	in_channels: int = 4  # BraTS: T1, T1CE, T2, FLAIR
	out_channels: int = 3  # WT, TC, ET (multi-channel binary masks)
	spatial_dims: int = 3
	feature_sizes_2d: List[int] = None
	feature_sizes_3d: List[int] = None

	def __post_init__(self):
		if self.feature_sizes_2d is None:
			self.feature_sizes_2d = [32, 64, 128, 256]
		if self.feature_sizes_3d is None:
			self.feature_sizes_3d = [16, 32, 64, 128]


paths = Paths()
train_cfg = TrainConfig()
model_cfg = ModelConfig()

# Ensure output directories exist
os.makedirs(paths.logs, exist_ok=True)
_os_dirs = [paths.models, paths.predictions, paths.processed]
for d in _os_dirs:
	os.makedirs(d, exist_ok=True)
