import os
import numpy as np
import torch
from tqdm import tqdm
import concurrent.futures
from functools import partial
import nibabel as nib

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    Resized, ScaleIntensityRanged, ConvertToMultiChannelBasedOnBratsClassesd, EnsureTyped
)

# --- Configuration --
BASE_DIR = './BRATS'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'train')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

TARGET_VOXEL_SPACING = (1.0, 1.0, 1.0)
OUTPUT_SHAPE = (128, 128, 128)
MODALITY_KEYS = ['t1c', 't1n', 't2f', 't2w']
ALL_KEYS = MODALITY_KEYS + ['seg']




def find_patient_files(data_dir):
    patient_files = []
    for patient_id in sorted(os.listdir(data_dir)):
        patient_folder = os.path.join(data_dir, patient_id)
        if os.path.isdir(patient_folder):
            files = {
                "t1c": os.path.join(patient_folder, f"{patient_id}-t1c.nii.gz"),
                "t1n": os.path.join(patient_folder, f"{patient_id}-t1n.nii.gz"),
                "t2f": os.path.join(patient_folder, f"{patient_id}-t2f.nii.gz"),
                "t2w": os.path.join(patient_folder, f"{patient_id}-t2w.nii.gz"),
                "seg": os.path.join(patient_folder, f"{patient_id}-seg.nii.gz"),
                "id": patient_id
            }
            if all(os.path.exists(f) for k, f in files.items() if k != "id"):
                patient_files.append(files)
    return patient_files

monai_preprocess_pipeline = Compose([
    LoadImaged(keys=ALL_KEYS, image_only=True, ensure_channel_first=True),
    ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
    Spacingd(keys=ALL_KEYS, pixdim=TARGET_VOXEL_SPACING, mode=["bilinear"] * 4 + ["nearest"]),
    ScaleIntensityRanged(keys=MODALITY_KEYS, a_min=0.0, a_max=1400.0, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=ALL_KEYS, source_key='t1c', margin=10),
    Resized(keys=ALL_KEYS, spatial_size=OUTPUT_SHAPE, mode=["area"] * 4 + ["nearest"]),
    EnsureTyped(keys=ALL_KEYS, dtype=torch.float16)
])

def preprocess_and_save(patient_data, output_dir):
    try:
        processed_data = monai_preprocess_pipeline(patient_data)
        final_image = torch.cat([processed_data[key] for key in MODALITY_KEYS], dim=0)
        final_mask = processed_data['seg']
        output_filepath = os.path.join(output_dir, f"{patient_data['id']}.npz")
        np.savez_compressed(output_filepath, image=final_image.numpy(), mask=final_mask.numpy().astype(np.uint8))
    except Exception as e:
        return f"Failed {patient_data['id']}: {e}"
    return None

def main():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Path to the training data directory
    patient_ids = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    #patient_ids = ["BraTS-GLI-00005-000"]

    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        
        patient_path = os.path.join(RAW_DATA_DIR, patient_id)
        seg_path = os.path.join(patient_path, f'{patient_id}-seg.nii.gz')
        if not os.path.exists(seg_path):
            print(f"⚠️ No segmentation file found for {patient_id}")
            continue

        # Load segmentation image
        img = nib.load(seg_path)
        data = img.get_fdata().astype(int)

        # Replace label 3 with 4
        data[data == 3] = 4

        # Create a new NIfTI image with the modified data
        new_img = nib.Nifti1Image(data, img.affine, img.header)

        # Overwrite the file
        nib.save(new_img, seg_path)
        #print(f"✅ Updated {patient_id} - saved to {seg_path}")

    patient_list = find_patient_files(RAW_DATA_DIR)

    if not os.listdir(PROCESSED_DIR):
        print(f"Found {len(patient_list)} patients to preprocess.")
        process_func = partial(preprocess_and_save, output_dir=PROCESSED_DIR)
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(process_func, patient_list), total=len(patient_list), desc="Preprocessing"))
        print("\nPreprocessing complete!")
    else:
        print("Processed data already exists. Skipping preprocessing.")

if __name__ == "__main__":
    main()