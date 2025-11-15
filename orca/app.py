# --- Standard library imports ---
import os
import io
import base64
import re
import time
import shutil

# --- Third-party library imports ---
from flask import Flask, request, jsonify, send_file, render_template
import nibabel as nib
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import torch
from skimage.measure import label, marching_cubes

# --- MONAI (Medical Open Network for AI) imports ---
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    Resized, ScaleIntensityRanged, EnsureTyped
)

# --- FLASK APP AND FOLDER SETUP ---
app = Flask(__name__) # Initializes the Flask application.

# --- HELPER FUNCTION FOR SECURE FILE PATHS ---
def get_safe_session_path(session_id_from_client):
    """Creates a secure, sanitized file path for a user session to prevent security risks."""
    if not session_id_from_client:
        raise ValueError("Session ID is missing or empty.")
    
    # Sanitize the ID to prevent path traversal attacks (e.g., '../../etc/passwd').
    safe_folder_name = secure_filename(session_id_from_client)
    
    # Ensure the sanitized ID is not empty.
    if not safe_folder_name:
         raise ValueError("Invalid Session ID provided.")

    # Return the full, safe path for the session's data.
    return os.path.join(app.config['UPLOAD_FOLDER'], safe_folder_name)


# --- CONFIGURATION FOR FILE STORAGE ---
EXAMPLES_FOLDER = 'examples' # Folder containing pre-loaded example cases.
UPLOAD_FOLDER = 'data/uploads' # Main folder for storing user-uploaded data.
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Creates the upload folder if it doesn't exist.
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Registers the upload folder with Flask.


# Part 1: GLOBAL MODEL AND PIPELINE SETUP

# --- Configuration for the ML model ---
BEST_MODEL_PATH = 'model/best_model.pth' # Path to the pre-trained PyTorch model file.
DEVICE = torch.device('cpu') # Sets the computation device to CPU.

print("--- Initializing Brain Segmentation Server ---")
# --- Model Definition ---
# Defines the Attention U-Net model architecture.
model = AttentionUnet(
    spatial_dims=3, in_channels=4, out_channels=3,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
).to(DEVICE)

# --- Load Pre-trained Model Weights ---
try:
    if os.path.exists(BEST_MODEL_PATH):
        # Loads the model's state from the checkpoint file.
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Sets the model to evaluation mode (disables dropout, etc.).
        print(f"--> PyTorch Model loaded successfully from {BEST_MODEL_PATH}")
    else:
        print(f"--> WARNING: Model file not found at {BEST_MODEL_PATH}. Prediction will fail.")
except Exception as e:
    print(f"--> ERROR: Failed to load PyTorch model: {e}")

# --- MONAI Preprocessing Pipeline Definition ---
MODALITY_KEYS = ['t1c', 't1n', 't2f', 't2w'] # Dictionary keys for the different MRI modalities.
# Defines a sequence of transformations to apply to the input images before prediction.
monai_preprocess_pipeline = Compose([
    LoadImaged(keys=MODALITY_KEYS, image_only=False, ensure_channel_first=True), # Loads NIfTI files and ensures they have a channel dimension.
    Spacingd(keys=MODALITY_KEYS, pixdim=(1.0, 1.0, 1.0), mode=["bilinear"] * 4), # Resamples images to a standard 1x1x1mm spacing.
    ScaleIntensityRanged(keys=MODALITY_KEYS, a_min=0.0, a_max=1400.0, b_min=0.0, b_max=1.0, clip=True), # Normalizes pixel intensity values to a [0, 1] range.
    CropForegroundd(keys=MODALITY_KEYS, source_key='t1c', margin=10), # Crops away empty background space.
    Resized(keys=MODALITY_KEYS, spatial_size=(128, 128, 128), mode=["area"] * 4), # Resizes images to the model's expected input size.
    EnsureTyped(keys=MODALITY_KEYS) # Ensures the data is in the correct tensor format.
])

# --- Post-Processing and Volumetry Functions ---
def remove_small_lesions(pred_mask_np, min_size_map):
    # (Same as before)
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

def calculate_and_log_volumes(label_map):
    """
    --- NEW ---
    Calculates the volume of each tumor component from the prediction and logs it.
    """
    print("\n--- Predicted Tumor Volumetry ---")
    print("-" * 35)
    # BraTS labels: WT is 1,2,4; TC is 1,4; ET is 4
    vol_wt = np.sum(np.isin(label_map, [1, 2, 4]))
    vol_tc = np.sum(np.isin(label_map, [1, 4]))
    vol_et = np.sum(label_map == 4)
    
    print(f"{'Whole Tumor (WT)':<25}: {vol_wt} voxels")
    print(f"{'Tumor Core (TC)':<25}: {vol_tc} voxels")
    print(f"{'Enhancing Tumor (ET)':<25}: {vol_et} voxels")
    print("-" * 35, "\n")
    return {"WT": vol_wt, "TC": vol_tc, "ET": vol_et}


# Part 2: FLASK ENDPOINTS


@app.route('/')
def index():
    """Serves the main HTML page of the application."""
    return render_template('index.html')


@app.route('/list_examples', methods=['GET'])
def list_examples():
    """Lists available example cases by scanning the examples folder."""
    if not os.path.exists(EXAMPLES_FOLDER):
        return jsonify([]) # Return empty list if folder doesn't exist
    
    try:
        # List subdirectories in the examples folder
        example_ids = [d for d in os.listdir(EXAMPLES_FOLDER) if os.path.isdir(os.path.join(EXAMPLES_FOLDER, d))]
        return jsonify(sorted(example_ids))
    except Exception as e:
        print(f"Error listing examples: {e}")
        return jsonify({"error": "Could not read examples directory"}), 500

@app.route('/load_example', methods=['POST'])
def load_example():
    """Copies files from an example case into a user's session folder."""
    data = request.get_json()
    example_id = data.get('id')
    session_id = data.get('session_id')
    found = []


    try:
        # --- MODIFIED SECTION: FROM HERE ---

        session_folder = get_safe_session_path(session_id)
        os.makedirs(session_folder, exist_ok=True)

        # 1. Define the mapping from our internal modality names to the file suffixes.
        # This is the key to making the system flexible.
        modality_suffix_map = {
            'flair': '-t2f',
            't1': '-t1n',
            't1ce': '-t1c',
            't2': '-t2w',
            'gt': '-seg'  # Also look for the optional ground truth file
        }
        
        # 2. Get a list of all files in the example directory.
        example_dir = os.path.join(EXAMPLES_FOLDER, example_id)
        all_files_in_dir = os.listdir(example_dir)
        print(example_id, all_files_in_dir)


        # 3. Loop through each modality we need and find its corresponding file.
        modalities_to_find = ['flair', 't1', 't1ce', 't2', 'gt']
        for mod in modalities_to_find:
            suffix = modality_suffix_map[mod]
            # Create a regex pattern: .*                 -t1c            \.nii         (\.gz)?     $
            #                       (any prefix) (the literal suffix) (literal .nii) (optional .gz) (end of string)
            pattern = re.compile(f".*?{suffix}\.nii(\.gz)?$")
            
            found_file = None
            for filename in all_files_in_dir:
                if pattern.match(filename):
                    found_file = filename
                    break # Found the file, move to the next modality

            if found_file:
                source_path = os.path.join(example_dir, found_file)
                # Standardize the destination filename
                destination_filename = f"{mod}.nii"
                destination_path = os.path.join(session_folder, destination_filename)
                found.append(mod)
                import shutil
                shutil.copy(source_path, destination_path) # Copy the file
                print(f"  -> Copied '{found_file}' to '{destination_filename}' for session '{session_id}'")
            # Only fail if a required input file is missing. 'gt' is optional.
            elif mod != 'gt':
                error_msg = f"Missing file for modality '{mod}' (pattern: *{suffix}.nii[.gz]) in example '{example_id}'"
                print(f"  -> ERROR: {error_msg}")
                return jsonify({"error": error_msg}), 500

 
        # The rest of the function remains the same.
        t1ce_path = os.path.join(session_folder, 't1ce.nii')
        img = nib.load(t1ce_path)

        # loaded_modalities = list(session['files'].keys())

        print(f"--- Example {example_id} loaded successfully into session. ---")
        
        return jsonify({
            "message": f"Example '{example_id}' loaded",
            "num_slices": img.shape[2],
            "slice_index": img.shape[2] // 2,
            "loaded_modalities": found
        })
    except Exception as e:
        print(f"Error loading example: {e}")
        return jsonify({"error": str(e)}), 500



LABEL_COLORS = {
    "wt": [255, 70, 70, 255], "tc": [70, 255, 70, 255], 
    "et": [255, 255, 70, 255], "bg": [255, 255, 255, 255],
    "net" : [70, 255, 70, 255], "edema" : [255, 70, 70, 255]
}

def nifti_slice_to_base64(nifti_path, slice_index, view_type="modality", labels=None, color=[255,255,255,255]):
    """Extracts a 2D slice from a NIfTI file and converts it to a base64-encoded PNG image."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    slice_index = max(0, min(slice_index, data.shape[2] - 1))
    slice_data = np.rot90(data[:, :, slice_index])
    if view_type == "prediction":
        rgba_slice = np.zeros((*slice_data.shape, 4), dtype=np.uint8)
        mask = np.isin(slice_data, labels)
        rgba_slice[mask] = color
        pil_img = Image.fromarray(rgba_slice, 'RGBA')


    elif view_type == "segmentation":
        rgba_slice = np.zeros((*slice_data.shape, 4), dtype=np.uint8)
        net_mask = (slice_data == 1)
        edema_mask = (slice_data == 2)
        et_mask = (slice_data == 4)

        label_set = set(labels if labels is not None else [])

        if label_set == {1, 2, 4}:  # Signal to derive Whole Tumor (WT)
            mask = np.isin(slice_data, [1, 2, 4])
        elif label_set == {1, 4}:    # Signal to derive Tumor Core (TC)
            mask = np.isin(slice_data, [1, 4])
        elif label_set == {4}:       # Signal to derive Enhancing Tumor (ET)
            mask = (slice_data == 4)
        elif label_set == {0}:       # Signal to derive Background (BG)
            mask = (slice_data == 0)
        elif label_set == {1}:       # Signal to derive Background (BG)
            mask = (slice_data == 1)
        elif label_set == {2}:       # Signal to derive Background (BG)
            mask = (slice_data == 2)
        
        if mask is not None:
            rgba_slice[mask] = color
            
        pil_img = Image.fromarray(rgba_slice, 'RGBA')

    else:
        if np.max(slice_data) > 0: slice_data = (slice_data / np.max(slice_data)) * 255.0
        pil_img = Image.fromarray(slice_data.astype(np.uint8), 'L')
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads from the user."""
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    modality = request.form.get('modality', 'unknown')
    session_id = request.form.get('session_id') # <-- Get ID from form data

    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    try:
        # Use the helper to get the safe path
        session_folder = get_safe_session_path(session_id)
        os.makedirs(session_folder, exist_ok=True)

        # --- KEY CHANGE: Standardize the filename ---
        # This makes finding files later trivial.
        standardized_filename = f"{modality}.nii"
        filepath = os.path.join(session_folder, standardized_filename)
        
        file.save(filepath)
        
        img = nib.load(filepath)
        return jsonify({ "message": f"{modality} uploaded", "num_slices": img.shape[2], "slice_index": img.shape[2] // 2 })

    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/get_all_slices', methods=['GET'])
def get_all_slices():
    """Fetches all 2D slices for a given modality and returns them as a JSON object."""
    modality = request.args.get('modality')
    session_id = request.args.get('session_id')

    try:
        session_folder = get_safe_session_path(session_id)
        filepath = os.path.join(session_folder, f"{modality}.nii")
        
        if not os.path.exists(filepath):
            return jsonify({"error": f"File for modality '{modality}' not found."}), 404
        
        img = nib.load(filepath)
        num_slices = img.shape[2]
        all_slices = {}
        if modality == 'pred':
            all_slices['wt'] = [nifti_slice_to_base64(filepath, i, "prediction", [1, 2, 4], LABEL_COLORS['wt']) for i in range(num_slices)]
            all_slices['tc'] = [nifti_slice_to_base64(filepath, i, "prediction", [1, 4], LABEL_COLORS['tc']) for i in range(num_slices)]
            all_slices['et'] = [nifti_slice_to_base64(filepath, i, "prediction", [4], LABEL_COLORS['et']) for i in range(num_slices)]
            all_slices['bg'] = [nifti_slice_to_base64(filepath, i, "prediction", [0], LABEL_COLORS['bg']) for i in range(num_slices)]
        elif modality == 'gt':
            all_slices['gt_net'] = [nifti_slice_to_base64(filepath, i, "segmentation", [1], LABEL_COLORS['net']) for i in range(num_slices)]
            all_slices['gt_edema'] = [nifti_slice_to_base64(filepath, i, "segmentation", [2], LABEL_COLORS['edema']) for i in range(num_slices)]
            
            all_slices['gt_bg'] = [nifti_slice_to_base64(filepath, i, "segmentation", [0], LABEL_COLORS['bg']) for i in range(num_slices)]
            all_slices['gt_wt'] = [nifti_slice_to_base64(filepath, i, "segmentation", [1, 2, 4], LABEL_COLORS['wt']) for i in range(num_slices)]
            all_slices['gt_tc'] = [nifti_slice_to_base64(filepath, i, "segmentation", [1, 4], LABEL_COLORS['tc']) for i in range(num_slices)]
            all_slices['gt_et'] = [nifti_slice_to_base64(filepath, i, "segmentation", [4], LABEL_COLORS['et']) for i in range(num_slices)]
        else:
            all_slices[modality] = [nifti_slice_to_base64(filepath, i) for i in range(num_slices)]
        return jsonify(all_slices)

    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400



    


@app.route('/predict', methods=['POST'])
def predict():
    """Runs the full brain tumor segmentation pipeline on the uploaded data."""

    start_time = time.time()
    session_id = request.json.get('session_id')
    print("\n\n--- Received Prediction Request ---")
    

    try:

        session_folder = get_safe_session_path(session_id)
        # Step 1: Preprocessing - Apply the MONAI pipeline to the input images.
        print("Step 1/5: Preprocessing input images...")
        input_data = {
            "t1c": os.path.join(session_folder, 't1ce.nii'),
            "t1n": os.path.join(session_folder, 't1.nii'),
            "t2f": os.path.join(session_folder, 'flair.nii'),
            "t2w": os.path.join(session_folder, 't2.nii'),
        }
        processed_data = monai_preprocess_pipeline(input_data)
        print("--> Preprocessing DONE.")

        # Step 2: Inference - Run the model on the preprocessed data.
        print("Step 2/5: Running model inference...")
        input_tensor = torch.cat([processed_data[key] for key in MODALITY_KEYS], dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prediction_logits = model(input_tensor)
        print("--> Model inference DONE.")

        # Step 3: Post-processing - Clean up the raw model output.
        print("Step 3/5: Post-processing prediction mask...")
        pred_mask_raw = (torch.sigmoid(prediction_logits).cpu().squeeze(0) > 0.5).numpy()
        min_lesion_sizes = {0: 100, 1: 75, 2: 50}  # WT, TC, ET (channel indices)
        pred_mask_postprocessed = remove_small_lesions(pred_mask_raw, min_lesion_sizes)
        
        # Converts the 3-channel model output (WT,TC,ET) to a single-channel BraTS label map (0,1,2,4).
        label_map = np.zeros(pred_mask_postprocessed[0].shape, dtype=np.uint8)
        label_map[pred_mask_postprocessed[1] > 0] = 2  # Whole Tumor
        label_map[pred_mask_postprocessed[0] > 0] = 1  # Tumor Core
        label_map[pred_mask_postprocessed[2] > 0] = 4  # Enhancing Tumor
        print("--> Post-processing DONE.")

        # Step 4: Calculate Volumes - Compute the size of the predicted tumor regions.
        print("Step 4/5: Calculating predicted volumes...")
        calculate_and_log_volumes(label_map)
        print("--> Volume calculation DONE.")
        
        # Step 5: Save Prediction - Save the final label map as a NIfTI file.
        print("Step 5/5: Saving prediction as NIfTI file...")
        output_affine = processed_data['t1c_meta_dict']['affine']
        nifti_image = nib.Nifti1Image(label_map, output_affine)
        pred_filepath = os.path.join(session_folder, 'pred.nii')
        nib.save(nifti_image, pred_filepath)


        print(f"--> Prediction saved to {pred_filepath}")
        end_time = time.time()
        print(f"--- Prediction pipeline finished in {end_time - start_time:.2f} seconds ---\n")

        return jsonify({
            "message": "Prediction successful",
            "num_slices": label_map.shape[2],
            "modality": "pred"
        })

    except Exception as e:
        print(f"!!! An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.route('/get_3d_mesh_data', methods=['GET'])
def get_3d_mesh_data():
    """Generates 3D surface mesh data (vertices and faces) from a NIfTI volume."""

    modality = request.args.get('modality')
    session_id = request.args.get('session_id')

    try:
        # Locates the correct NIfTI file in the user's session folder.
        session_folder = get_safe_session_path(session_id)
        filepath = os.path.join(session_folder, f"{modality}.nii")

        # 4. Check if the required file actually exists before proceeding
        if not os.path.exists(filepath):
            return jsonify({"error": f"File '{modality}.nii' not found in session '{session_id}'"}), 404

        print(f"\n--- Generating 3D mesh data for {modality} from {filepath} ---")
        img = nib.load(filepath)
        data = img.get_fdata()
        mesh_data_for_json = {}

        # Generates a brain mesh from a T1ce scan.
        if modality == 't1ce':
            print(" -> Generating Brain mesh...")
            brain_mask = data > (data.min() + 0.01)
            verts, faces, _, _ = marching_cubes(brain_mask, level=0.5)
            mesh_data_for_json['Brain'] = {
                "vertices": verts.flatten().tolist(), "faces": faces.flatten().tolist(),
                "color": 0xcccccc, "opacity": 0.1
            }

        # Generates meshes for each tumor component from the prediction file.
        elif modality == 'pred':
            wt_mask = data > 0
            if np.sum(wt_mask) > 0:
                print(" -> Generating Whole Tumor (WT) mesh...")
                verts, faces, _, _ = marching_cubes(wt_mask, level=0.5)
                mesh_data_for_json['WT'] = {
                    "vertices": verts.flatten().tolist(), "faces": faces.flatten().tolist(),
                    "color": 0xff0000, "opacity": 0.4
                }

            tc_mask = np.isin(data, [1, 4])
            if np.sum(tc_mask) > 0:
                print(" -> Generating Tumor Core (TC) mesh...")
                verts, faces, _, _ = marching_cubes(tc_mask, level=0.5)
                mesh_data_for_json['TC'] = {
                    "vertices": verts.flatten().tolist(), "faces": faces.flatten().tolist(),
                    "color": 0x00ff00, "opacity": 0.6
                }

            et_mask = (data == 4)
            if np.sum(et_mask) > 0:
                print(" -> Generating Enhancing Tumor (ET) mesh...")
                verts, faces, _, _ = marching_cubes(et_mask, level=0.5)
                mesh_data_for_json['ET'] = {
                    "vertices": verts.flatten().tolist(), "faces": faces.flatten().tolist(),
                    "color": 0x0000ff, "opacity": 0.8
                }

        print(f"--- Mesh generation complete for {modality} ---\n")
        return jsonify(mesh_data_for_json) # Returns mesh data ready for 3D rendering.

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"!!! An error occurred during 3D mesh generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Mesh generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # HF default port
    app.run(host="0.0.0.0", port=port, debug=False)
