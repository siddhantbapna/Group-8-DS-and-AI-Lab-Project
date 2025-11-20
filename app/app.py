# --- Standard library imports ---
import os
import io
import base64
import re
import time
import shutil
import traceback
import json

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
    Resized, ScaleIntensityRanged, EnsureTyped, ConvertToMultiChannelBasedOnBratsClassesd
)

# --- FLASK APP AND FOLDER SETUP ---
app = Flask(__name__)
UPLOAD_FOLDER = 'data/uploads'
EXAMPLES_FOLDER = 'examples'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Part 1: GLOBAL MODEL AND PIPELINE SETUP ---
print("--- Initializing Brain Segmentation Server ---")

# FIX: Set device to CPU as requested
DEVICE = torch.device("cpu")
print(f"--> Using device: {DEVICE}")

BEST_MODEL_PATH = 'model/best_model.pth'

# --- Model Definition ---
model = AttentionUnet(
    spatial_dims=3, in_channels=4, out_channels=3,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
).to(DEVICE)

# --- Load Pre-trained Model Weights ---
try:
    if os.path.exists(BEST_MODEL_PATH):
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # No .half() conversion for CPU
        print(f"--> PyTorch Model loaded successfully from {BEST_MODEL_PATH}")
    else:
        print(f"--> WARNING: Model file not found at {BEST_MODEL_PATH}.")
except Exception as e:
    print(f"--> ERROR: Failed to load PyTorch model: {e}")

# --- REPLICATED MONAI PREPROCESSING PIPELINE ---
MODALITY_KEYS = ['t1c', 't1n', 't2f', 't2w']
SEG_KEY = 'seg'
ALL_KEYS = MODALITY_KEYS + [SEG_KEY]
SPATIAL_SHAPE = (128, 128, 128)

monai_preprocess_pipeline = Compose([
    LoadImaged(keys=ALL_KEYS, image_only=True, ensure_channel_first=True, allow_missing_keys=True),
    ConvertToMultiChannelBasedOnBratsClassesd(keys=SEG_KEY, allow_missing_keys=True),
    Spacingd(keys=ALL_KEYS, pixdim=(1.0, 1.0, 1.0), mode=["bilinear"] * 4 + ["nearest"], allow_missing_keys=True),
    ScaleIntensityRanged(keys=MODALITY_KEYS, a_min=0.0, a_max=1400.0, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True),
    CropForegroundd(keys=ALL_KEYS, source_key='t1c', margin=10, allow_missing_keys=True),
    Resized(keys=ALL_KEYS, spatial_size=SPATIAL_SHAPE, mode=["area"] * 4 + ["nearest"], allow_missing_keys=True),
    # FIX: Ensure input is float32 for CPU model
    EnsureTyped(keys=ALL_KEYS, dtype=torch.float32, allow_missing_keys=True),
])


def get_safe_session_path(session_id_from_client):
    if not session_id_from_client: raise ValueError("Session ID is missing.")
    safe_folder_name = secure_filename(session_id_from_client)
    if not safe_folder_name: raise ValueError("Invalid Session ID.")
    return os.path.join(app.config['UPLOAD_FOLDER'], safe_folder_name)

# --- Post-Processing, Analysis, and Mesh Generation ---

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

# FIX: VOLUMETRIC ANALYSIS WITH JSON SERIALIZABLE TYPES
def perform_volumetric_analysis(true_mask, pred_mask):
    results = {"volumetry": [], "dice_scores": []}
    smooth = 1e-6
    labels_map = {1: "Whole Tumor (WT)", 0: "Tumor Core (TC)", 2: "Enhancing Tumor (ET)"}

    if pred_mask is not None:
        results["volumetry"].append({
            "type": "Predicted",
            "wt": float(np.sum(pred_mask[1])), "tc": float(np.sum(pred_mask[0])), "et": float(np.sum(pred_mask[2]))
        })
    if true_mask is not None and np.sum(true_mask) > 0:
        results["volumetry"].append({
            "type": "Ground Truth",
            "wt": float(np.sum(true_mask[1])), "tc": float(np.sum(true_mask[0])), "et": float(np.sum(true_mask[2]))
        })

    if true_mask is not None and pred_mask is not None and np.sum(true_mask) > 0:
        for i in [1, 0, 2]:
            true_flat, pred_flat = true_mask[i].flatten(), pred_mask[i].flatten()
            intersection = np.sum(true_flat * pred_flat)
            sum_of_sets = np.sum(true_flat) + np.sum(pred_flat)
            dice_score = (2. * intersection + smooth) / (sum_of_sets + smooth)
            results["dice_scores"].append({"label": labels_map[i], "score": float(dice_score)})
    
    return results

def generate_and_save_meshes(session_folder):
    """
    Generates 3D surface meshes from processed numpy arrays and NIfTI files.
    It creates meshes for the brain, ground truth segmentations, and predicted segmentations,
    then saves them all into a single JSON file for the frontend renderer.
    """
    print("--> Generating and saving 3D mesh data...")
    mesh_data_for_json = {}
    npz_path = os.path.join(session_folder, 'processed.npz')
    pred_path = os.path.join(session_folder, 'pred.nii')
    json_output_path = os.path.join(session_folder, 'meshes.json')

    def create_mesh_dict(volume_data_3d, level, color, opacity):
        """Helper function to run marching cubes and format the output."""
        try:
            # Marching cubes algorithm to extract surface mesh from a 3D volume
            verts, faces, _, _ = marching_cubes(volume_data_3d, level=level)
            if len(verts) == 0 or len(faces) == 0:
                return None # No surface found at this level
            # The frontend expects flattened lists for vertices and faces
            return {
                "v": verts.flatten().tolist(),
                "f": faces.flatten().tolist(),
                "c": color,
                "o": opacity
            }
        except Exception as e:
            print(f"    - Could not generate mesh: {e}")
            return None

    # 1. Process the main data file (processed.npz)
    if os.path.exists(npz_path):
        print(f"    - Loading data from {npz_path}")
        with np.load(npz_path) as data:
            # --- Generate Brain Mesh from T1c channel ---
            if 'image' in data:
                print("    - Generating brain mesh from T1c...")
                t1c_volume = data['image'][0] # T1c is the first channel
                # We choose a level slightly above the background noise
                brain_mesh = create_mesh_dict(t1c_volume, level=0.2, color="#CCCCCC", opacity=0.3)
                if brain_mesh:
                    mesh_data_for_json['Brain'] = brain_mesh
                    print("      ... Brain mesh created.")

            # --- Generate Ground Truth Meshes from 'mask' ---
            if 'mask' in data and np.any(data['mask']):
                print("    - Generating ground truth meshes...")
                gt_mask = data['mask']
                # The level for a binary mask should be between 0 and 1
                level = 0.5
                
                # Channel 1: Whole Tumor (WT)
                gt_wt_mesh = create_mesh_dict(gt_mask[1], level, "#ff4646", 1.0) # Red
                if gt_wt_mesh: mesh_data_for_json['GT_WT'] = gt_wt_mesh
                
                # Channel 0: Tumor Core (TC)
                gt_tc_mesh = create_mesh_dict(gt_mask[0], level, "#46ff46", 1.0) # Green
                if gt_tc_mesh: mesh_data_for_json['GT_TC'] = gt_tc_mesh
                
                # Channel 2: Enhancing Tumor (ET)
                gt_et_mesh = create_mesh_dict(gt_mask[2], level, "#ffff46", 1.0) # Yellow
                if gt_et_mesh: mesh_data_for_json['GT_ET'] = gt_et_mesh
                print("      ... Ground truth meshes created.")

    # 2. Process the prediction file (pred.nii) if it exists
    if os.path.exists(pred_path):
        print(f"    - Loading prediction from {pred_path}")
        pred_img = nib.load(pred_path)
        pred_data = pred_img.get_fdata()
        
        if np.any(pred_data):
            print("    - Generating prediction meshes...")
            level = 0.5 # Level for binary masks
            
            # Create binary masks from the multi-class label map
            # WT = ET (4) + TC (1) + non-enhancing core (2) -> everything non-zero
            pred_wt_mask = (pred_data > 0).astype(np.float32)
            # TC = ET (4) + TC (1)
            pred_tc_mask = np.isin(pred_data, [1, 4]).astype(np.float32)
            # ET = ET (4)
            pred_et_mask = (pred_data == 4).astype(np.float32)

            # --- Generate Prediction Meshes ---
            pred_wt_mesh = create_mesh_dict(pred_wt_mask, level, "#ff4646", 0.8) # Red
            if pred_wt_mesh: mesh_data_for_json['Pred_WT'] = pred_wt_mesh
            
            pred_tc_mesh = create_mesh_dict(pred_tc_mask, level, "#46ff46", 0.9) # Green
            if pred_tc_mesh: mesh_data_for_json['Pred_TC'] = pred_tc_mesh
            
            pred_et_mesh = create_mesh_dict(pred_et_mask, level, "#4646ff", 1.0) # Blue
            if pred_et_mesh: mesh_data_for_json['Pred_ET'] = pred_et_mesh
            print("      ... Prediction meshes created.")

    # 3. Save the final dictionary to a JSON file
    if mesh_data_for_json:
        try:
            with open(json_output_path, 'w') as f:
                json.dump(mesh_data_for_json, f)
            print(f"--> Successfully saved mesh data to {json_output_path}")
        except Exception as e:
            print(f"--> ERROR: Failed to save meshes.json: {e}")
    else:
        print("--> No mesh data was generated, skipping save.")

# --- Part 2: FLASK ENDPOINTS ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/list_examples', methods=['GET'])
def list_examples():
    try: return jsonify(sorted([d for d in os.listdir(EXAMPLES_FOLDER) if os.path.isdir(os.path.join(EXAMPLES_FOLDER, d))]))
    except FileNotFoundError: return jsonify([])

@app.route('/reset_session', methods=['POST'])
def reset_session():
    try:
        session_folder = get_safe_session_path(request.json.get('session_id'))
        if os.path.exists(session_folder): shutil.rmtree(session_folder)
        return jsonify({"message": "Session reset."})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    try:
        data = request.get_json()
        session_folder = get_safe_session_path(data.get('session_id'))
        modality = secure_filename(data.get('modality'))
        file_path = os.path.join(session_folder, f"{modality}.nii")
        if os.path.exists(file_path): os.remove(file_path)
        return jsonify({"message": f"File {modality} deleted."})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        session_folder = get_safe_session_path(request.form.get('session_id'))
        os.makedirs(session_folder, exist_ok=True)
        file, modality = request.files['file'], request.form.get('modality')
        file.save(os.path.join(session_folder, f"{modality}.nii"))
        loaded_modalities = [f.replace('.nii', '') for f in os.listdir(session_folder) if f.endswith('.nii')]
        return jsonify({"message": f"{modality} uploaded", "loaded_modalities": loaded_modalities})
    except Exception as e: return jsonify({"error": f"Upload failed: {e}"}), 500

# FIX: CORRECTED REGEX LOGIC FOR EXAMPLE LOADING
@app.route('/load_example', methods=['POST'])
def load_example():
    data = request.get_json()
    try:
        session_folder = get_safe_session_path(data.get('session_id'))
        if os.path.exists(session_folder): shutil.rmtree(session_folder)
        os.makedirs(session_folder, exist_ok=True)

        example_id = secure_filename(data.get('id'))
        example_dir = os.path.join(EXAMPLES_FOLDER, example_id)
        
        modality_suffix_map = {'t1c': '-t1c', 't1n': '-t1n', 't2f': '-t2f', 't2w': '-t2w', 'seg': '-seg'}
        
        loaded_modalities = []
        for modality, suffix in modality_suffix_map.items():
            pattern = re.compile(f".*?{suffix}\.nii(\.gz)?$")
            found_file = next((f for f in os.listdir(example_dir) if pattern.match(f)), None)
            
            if found_file:
                source_path = os.path.join(example_dir, found_file)
                dest_path = os.path.join(session_folder, f"{modality}.nii")
                
                # Load with nibabel (handles .nii and .nii.gz) and save as .nii
                img = nib.load(source_path)
                nib.save(img, dest_path)
                
                loaded_modalities.append(modality)

        return jsonify({"message": "Example files loaded and decompressed", "loaded_modalities": loaded_modalities})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_files():
    try:
        session_folder = get_safe_session_path(request.json.get('session_id'))
        print(f"--- Processing files for session: {os.path.basename(session_folder)} ---")
        for m in MODALITY_KEYS:
            if not os.path.exists(os.path.join(session_folder, f"{m}.nii")):
                return jsonify({"error": f"Missing required file: {m.upper()}"}), 400
        
        input_dict = {k: os.path.join(session_folder, f"{k}.nii") for k in ALL_KEYS if os.path.exists(os.path.join(session_folder, f"{k}.nii"))}
        processed_dict = monai_preprocess_pipeline(input_dict)
        stacked_image = torch.cat([processed_dict[key] for key in MODALITY_KEYS], dim=0)
        stacked_mask = processed_dict.get(SEG_KEY, torch.zeros((3, *SPATIAL_SHAPE), dtype=torch.float32))
        np.savez_compressed(os.path.join(session_folder, "processed.npz"), image=stacked_image.numpy(), mask=stacked_mask.numpy())
        print("--> Saved 'processed.npz'")
        generate_and_save_meshes(session_folder)
        return jsonify({"message": "Files processed", "num_slices": SPATIAL_SHAPE[2], "slice_index": SPATIAL_SHAPE[2] // 2})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/get_all_slices', methods=['GET'])
def get_all_slices():
    modality, session_id = request.args.get('modality'), request.args.get('session_id')
    try:
        session_folder = get_safe_session_path(session_id)
        npz_path, pred_path = os.path.join(session_folder, 'processed.npz'), os.path.join(session_folder, 'pred.nii')
        all_slices = {}
        if not os.path.exists(npz_path) and modality not in ['pred']: return jsonify({"error": "processed.npz not found"}), 404
        
        if modality in MODALITY_KEYS:
            with np.load(npz_path) as data:
                img_data, idx = data['image'], MODALITY_KEYS.index(modality)
                all_slices[modality] = [array_slice_to_base64(img_data[idx, :, :, i]) for i in range(img_data.shape[3])]
        elif modality == 'seg':
            with np.load(npz_path) as data:
                if 'mask' not in data or not np.any(data['mask']): return jsonify({}), 200
                mask = data['mask']
                all_slices['gt_wt'] = [array_slice_to_base64(mask[1,:,:,i], is_mask=True, color=[255, 70, 70, 255]) for i in range(mask.shape[3])]
                all_slices['gt_tc'] = [array_slice_to_base64(mask[0,:,:,i], is_mask=True, color=[70, 255, 70, 255]) for i in range(mask.shape[3])]
                all_slices['gt_et'] = [array_slice_to_base64(mask[2,:,:,i], is_mask=True, color=[255, 255, 70, 255]) for i in range(mask.shape[3])]
        elif modality == 'pred':
            if not os.path.exists(pred_path): return jsonify({"error": "Prediction not found"}), 404
            pred_data = nib.load(pred_path).get_fdata()
            all_slices['wt'] = [array_slice_to_base64(np.isin(pred_data[:,:,i], [1,2,4]), is_mask=True, color=[255, 0, 0, 255]) for i in range(pred_data.shape[2])]
            all_slices['tc'] = [array_slice_to_base64(np.isin(pred_data[:,:,i], [1,4]), is_mask=True, color=[0, 255, 0, 255]) for i in range(pred_data.shape[2])]
            all_slices['et'] = [array_slice_to_base64(pred_data[:,:,i] == 4, is_mask=True, color=[0, 0, 255, 255]) for i in range(pred_data.shape[2])]
        return jsonify(all_slices)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to get slices: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    session_id = request.json.get('session_id')
    print("\n--- Received Prediction Request ---")
    try:
        session_folder = get_safe_session_path(session_id)
        npz_path = os.path.join(session_folder, 'processed.npz')
        if not os.path.exists(npz_path): return jsonify({"error": "processed.npz not found"}), 404
        
        with np.load(npz_path) as data: image_data, gt_mask_data = data['image'], data.get('mask')

        input_tensor = torch.from_numpy(image_data).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): prediction_logits = model(input_tensor)

        pred_mask_raw = (torch.sigmoid(prediction_logits).cpu().squeeze(0).float().numpy() > 0.5)
        pred_mask_postprocessed = remove_small_lesions(pred_mask_raw, {0: 100, 1: 75, 2: 50})
        analysis_results = perform_volumetric_analysis(gt_mask_data, pred_mask_postprocessed)

        label_map = np.zeros(pred_mask_postprocessed[0].shape, dtype=np.uint8)
        label_map[pred_mask_postprocessed[1] > 0] = 2; label_map[pred_mask_postprocessed[0] > 0] = 1; label_map[pred_mask_postprocessed[2] > 0] = 4
        nib.save(nib.Nifti1Image(label_map, np.eye(4)), os.path.join(session_folder, 'pred.nii'))
        
        generate_and_save_meshes(session_folder)

        print(f"--- Prediction finished in {time.time() - start_time:.2f} seconds ---\n")
        return jsonify({"message": "Prediction successful", "analysis": analysis_results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route('/get_mesh_json', methods=['GET'])
def get_mesh_json():
    try:
        session_folder = get_safe_session_path(request.args.get('session_id'))
        json_path = os.path.join(session_folder, 'meshes.json')
        if not os.path.exists(json_path): return jsonify({"error": "meshes.json not found."}), 404
        return send_file(json_path, mimetype='application/json')
    except Exception as e: return jsonify({"error": f"Failed to get mesh data: {str(e)}"}), 500

def array_slice_to_base64(slice_data_2d, is_mask=False, color=[255, 255, 255, 255]):
    slice_data = np.rot90(slice_data_2d)
    if is_mask:
        rgba = np.zeros((*slice_data.shape, 4), dtype=np.uint8)
        rgba[slice_data > 0] = color
        pil_img = Image.fromarray(rgba, 'RGBA')
    else:
        norm_data = (slice_data / np.max(slice_data) * 255.0) if np.max(slice_data) > 0 else slice_data
        pil_img = Image.fromarray(norm_data.astype(np.uint8), 'L')
    buf = io.BytesIO(); pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)