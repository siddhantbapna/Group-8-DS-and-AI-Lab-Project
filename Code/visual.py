import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    CropForegroundd,
    Resized,
    ScaleIntensityRanged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureTyped
)
from skimage.measure import label, marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROCESSED_DIR = "./BRATS/processed"
BEST_MODEL_PATH = "./models/best_model_fold_0_newsba4.pth"
testId = "BraTS-GLI-00106-000"

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Definition ---
model = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

# --- Load Pre-trained Weights ---
map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(BEST_MODEL_PATH, map_location=map_location)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

print("Model loaded successfully.")



# --- Post-Processing Function ---
def remove_small_lesions(pred_mask_np, min_size_map):
    processed_mask = np.zeros_like(pred_mask_np)
    for c in range(pred_mask_np.shape[0]):
        channel_mask = pred_mask_np[c]
        if np.sum(channel_mask) > 0:
            labeled_mask = label(channel_mask)
            min_size = min_size_map.get(c, 50)
            for region_label in range(1, np.max(labeled_mask) + 1):
                if np.sum(labeled_mask == region_label) > min_size:
                    processed_mask[c][labeled_mask == region_label] = 1
    return processed_mask


def calculate_and_print_volumes(true_mask, pred_mask):
    """
    Calculates and prints tumor volumes and Dice scores (accuracy) for each class.
    """
    voxel_volume = 1.0  # Assuming 1x1x1mm spacing
    smooth = 1e-6
    labels = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]

    print("\n--- Volumetric and Accuracy Analysis ---")
    print("-" * 85)
    print(f"{'Tumor Component':<25} | {'Ground Truth Volume':<20} | {'Predicted Volume':<20} | {'Dice Score (Accuracy)':<20}")
    print("-" * 85)

    for i, label_name in enumerate(labels):
        # --- Volume Calculation ---
        true_volume = np.sum(true_mask[i]) * voxel_volume
        pred_volume = np.sum(pred_mask[i]) * voxel_volume

        # --- Dice Score Calculation (based on your provided logic) ---
        true_flat = true_mask[i].flatten()
        pred_flat = pred_mask[i].flatten()
        
        intersection = np.sum(true_flat * pred_flat)
        sum_of_sets = np.sum(true_flat) + np.sum(pred_flat)
        
        dice_score = (2. * intersection + smooth) / (sum_of_sets + smooth)

        # --- Print the combined results for the current class ---
        print(f"{label_name:<25} | {true_volume:<20.2f} | {pred_volume:<20.2f} | {dice_score:<20.4f}")
    
    print("-" * 85)


# --- Plots ---
def perform_volumetric_analysis(patient_id):
    """
    Loads a patient's data, runs inference, and performs 2D/3D analysis.
    Args:
        patient_id (str): The ID of the patient to validate.
    """
    patient_file_path = os.path.join(PROCESSED_DIR, f"{patient_id}.npz")

    if not os.path.exists(patient_file_path):
        print(f"Processed file for patient {patient_id} not found at {patient_file_path}")
        return

    # Load and preprocess the data
    with np.load(patient_file_path) as data:
        image = torch.from_numpy(data['image'].astype(np.float32))
        true_mask = torch.from_numpy(data['mask'].astype(np.float32))

    input_tensor = image.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction_logits = model(input_tensor)

    # Post-process the prediction
    pred_mask_raw = (torch.sigmoid(prediction_logits).cpu().squeeze(0) > 0.5).numpy()
    min_lesion_sizes = {0: 100, 1: 75, 2: 50}  # WT, TC, ET
    pred_mask_postprocessed = remove_small_lesions(pred_mask_raw, min_lesion_sizes)
    
    true_mask_np = true_mask.numpy()

    # --- 1. Perform Volumetric Analysis ---
    calculate_and_print_volumes(true_mask_np, pred_mask_postprocessed)

    # --- 2. Visualize 2D Slice Comparison ---
    slice_idx = image.shape[-1] // 2
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'2D Slice Comparison for Patient: {patient_id} (Slice {slice_idx})', fontsize=16)

    titles = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
    for i, title in enumerate(titles):
        axes[i, 0].imshow(image[0, :, :, slice_idx], cmap='gray')
        axes[i, 0].set_title(f'Input T1c\n({title})')
        axes[i, 1].imshow(true_mask_np[i, :, :, slice_idx], cmap='viridis')
        axes[i, 1].set_title(f'Ground Truth\n({title})')
        axes[i, 2].imshow(pred_mask_postprocessed[i, :, :, slice_idx], cmap='viridis')
        axes[i, 2].set_title(f'Prediction\n({title})')
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return pred_mask_postprocessed



if os.path.exists(PROCESSED_DIR):
    validation_patient_ids = [f.split('.')[0] for f in os.listdir(PROCESSED_DIR) if f.endswith('.npz')]
    if validation_patient_ids:
        # --- Run the full analysis for a specific patient ID ---
        # You can change this ID to any other from your processed data
        patient_id_to_test = testId
        print(f"\n=== Running Full Analysis for Patient: {patient_id_to_test} ===")
        perform_volumetric_analysis(patient_id_to_test)
    else:
        print("No processed patient data found.")
else:
    print("The 'processed' directory does not exist.")