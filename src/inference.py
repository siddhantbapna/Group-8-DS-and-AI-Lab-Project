from __future__ import annotations

import os
import glob
from typing import Optional

import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, AsDiscrete

from config.config import train_cfg, model_cfg
from src.models import create_model


def _load_modalities(case_dir: str):
	def _f(pattern: str):
		m = glob.glob(os.path.join(case_dir, pattern))
		return m[0] if m else None
	t1 = _f("*t1.nii.gz")
	t1ce = _f("*t1ce.nii.gz")
	t2 = _f("*t2.nii.gz")
	flair = _f("*flair.nii.gz")
	imgs = [nib.load(p).get_fdata().astype(np.float32) for p in [t1, t1ce, t2, flair] if p]
	# simple normalization
	imgs = [(x - x.min()) / (x.max() - x.min() + 1e-8) for x in imgs]
	arr = np.stack(imgs, axis=0)  # C, H, W, D
	arr = np.expand_dims(arr, 0)  # B,C,H,W,D
	return torch.from_numpy(arr)


def run_inference(case_dir: str, ckpt_path: str, model_name: str, output_path: str):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = create_model(
		name=model_name,
		in_channels=model_cfg.in_channels,
		out_channels=model_cfg.out_channels,
		feature_sizes_3d=model_cfg.feature_sizes_3d,
	).to(device)
	state = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
	model.eval()

	input_tensor = _load_modalities(case_dir).to(device)
	with torch.no_grad():
		pred = sliding_window_inference(
			input_tensor,
			size=tuple(train_cfg.slide_infer_roi_3d),
			overlap=train_cfg.overlap_3d,
			predictor=model,
		)
		post = Compose([AsDiscrete(argmax=True)])
		pred = post(pred)

	pred_np = pred[0].cpu().numpy().astype(np.uint8)
	# Save as NIfTI using flair affine as reference if available
	flair = glob.glob(os.path.join(case_dir, "*flair.nii.gz"))
	affine = nib.load(flair[0]).affine if flair else np.eye(4)
	nib.save(nib.Nifti1Image(pred_np, affine), output_path)
