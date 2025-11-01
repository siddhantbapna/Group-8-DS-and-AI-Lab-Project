"""
Script to fix BraTS segmentation labels
Changes label 3 to 4 in all segmentation files
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import nibabel as nib
import numpy as np
from tqdm import tqdm

from config.config import paths

# Path to the training data directory
RAW_DATA_DIR = paths.data_train

print(f"Processing segmentation files in: {RAW_DATA_DIR}")

if not os.path.exists(RAW_DATA_DIR):
	print(f"Error: Data directory not found at {RAW_DATA_DIR}")
	sys.exit(1)

# List all patient folders
patient_ids = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

if not patient_ids:
	print(f"No patient directories found in {RAW_DATA_DIR}")
	sys.exit(1)

print(f"Found {len(patient_ids)} patient directories")

# Loop through every patient folder with a progress bar
updated_count = 0
skipped_count = 0

for patient_id in tqdm(patient_ids, desc="Processing patients"):
	patient_path = os.path.join(RAW_DATA_DIR, patient_id)
	
	# Try different naming patterns for segmentation files
	seg_paths = [
		os.path.join(patient_path, f'{patient_id}-seg.nii.gz'),
		os.path.join(patient_path, 'seg.nii.gz'),
		os.path.join(patient_path, 'segmentation.nii.gz'),
	]
	
	seg_path = None
	for sp in seg_paths:
		if os.path.exists(sp):
			seg_path = sp
			break
	
	if not seg_path:
		# Try glob pattern
		import glob
		seg_patterns = [
			os.path.join(patient_path, '*seg*.nii.gz'),
			os.path.join(patient_path, '*seg*.nii'),
		]
		for pattern in seg_patterns:
			matches = glob.glob(pattern)
			if matches:
				seg_path = matches[0]
				break
	
	if not seg_path or not os.path.exists(seg_path):
		# Only print first few warnings to avoid spam
		if skipped_count < 5:
			print(f"\n[WARNING] No segmentation file found for {patient_id}")
		skipped_count += 1
		continue
	
	try:
		# Load segmentation image
		img = nib.load(seg_path)
		data = img.get_fdata().astype(int)
		
		# Check if label 3 exists
		unique_labels = np.unique(data)
		if 3 not in unique_labels:
			# No label 3 to change, skip
			skipped_count += 1
			continue
		
		# Replace label 3 with 4
		data[data == 3] = 4
		
		# Create a new NIfTI image with the modified data
		new_img = nib.Nifti1Image(data, img.affine, img.header)
		
		# Overwrite the file
		nib.save(new_img, seg_path)
		updated_count += 1
		
	except Exception as e:
		print(f"\n[ERROR] Error processing {patient_id}: {e}")
		skipped_count += 1
		continue

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"[SUCCESS] Updated {updated_count} segmentation files")
print(f"[SKIPPED] Skipped {skipped_count} files (no label 3 or not found)")
print(f"{'='*60}")

