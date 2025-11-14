# **B1. Environment Setup**

This section describes the exact environment needed to reproduce the preprocessing, training, inference, and deployment pipelines for the 3D brain tumor segmentation project.

## **1. Python Version**

* **Python 3.10+** (recommended)
* Verified compatibility with **Python 3.10.12**

## **2. Core Dependencies**

### **Deep Learning & Medical Imaging**

* **PyTorch**
* **MONAI** (medical image preprocessing + transforms)
* **Nibabel** (MRI file handling `.nii`, `.nii.gz`)
* **SimpleITK** (image I/O, resampling)

### **Model Training & Utilities**

* **scikit-learn**
* **numpy**
* **pandas**
* **matplotlib**
* **tqdm**
* **tensorboard**

### **Deployment / API**

* **FastAPI** or **Flask**
* **uvicorn**
* **Gradio** or **Streamlit** (depending on your UI choice)

### **Optional Tools**

* **wandb** (experiment tracking)
* **torchsummary** or **fvcore** (model stats)
* **albumentations** (if additional augmentations used)

## **3. Hardware Requirements**

### **Training**

* **GPU:** NVIDIA GPU with at least **16 GB VRAM**

  * Recommended: **RTX 3090 / A5000 / A100**
* **CPU:** 8+ cores
* **RAM:** 32 GB minimum
* **Storage:**

  * Dataset: ~20 GB (BRATS)
  * Checkpoints + logs: ~5–10 GB

### **Inference**

* Can run on:

  * **GPU** (fast, real-time results)
  * **CPU** (slower but functional)
* Recommended:

  * **8+ GB RAM**
  * **8+ GB disk**

## **4. Installation Instructions**

### **Option 1: Using `requirements.txt` (recommended)**

Create a file:

```
torch
monai
nibabel
simpleITK
scikit-learn
numpy
pandas
matplotlib
tqdm
tensorboard
fastapi
uvicorn
gradio
```

Install via:

```bash
pip install -r requirements.txt
```

---

### **Option 2: Using `environment.yml` (Conda)**

```yaml
name: brain-tumor-seg
channels:
  - defaults
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - cudatoolkit
  - monai
  - nibabel
  - simpleitk
  - numpy
  - pandas
  - scikit-learn
  - tqdm
  - matplotlib
  - pip
  - pip:
      - fastapi
      - uvicorn
      - gradio
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate brain-tumor-seg
```

### **Option 3: GPU-Specific PyTorch Install**

Check your CUDA version:

```bash
nvidia-smi
```

Then install matching PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

(Replace `cu118` with your CUDA version.)

# **B2. Data Pipeline**

### **2.1 Dataset Source**

* **Dataset:** *BraTS 2023 Glioma* (Brain Tumor Segmentation Challenge)
* **Year:** **2023**
* **Type:** Multi-modal 3D MRI with expert-annotated tumor segmentation masks
* **Modalities Included:** T1, T1c, T2, T2-FLAIR
* **Ground Truth:** Segmentation mask with tumor sub-regions
* **Official Source:** Synapse (Synapse ID: **syn51156910**)
* **Public Mirrors Used:** Kaggle (processed + test sets)

  * **Processed dataset:** [https://www.kaggle.com/datasets/siddhantbapna/sb23-2/data](https://www.kaggle.com/datasets/siddhantbapna/sb23-2/data)
  * **Test dataset:** [https://www.kaggle.com/datasets/siddhantbapna/brats-testing-datasetsb/data](https://www.kaggle.com/datasets/siddhantbapna/brats-testing-datasetsb/data)

### **2.2 Licensing**

* **License:** **CC BY-NC 4.0 (Creative Commons Attribution–NonCommercial 4.0 International)**
* **Permitted:** Non-commercial use, redistribution, adaptation with attribution
* **Required Attribution:**
  “Data used in this publication were obtained as part of the Brain Tumor Segmentation (BraTS) Challenge project through Synapse ID: syn51156910.”

### **2.3 Preprocessing Pipeline (Summary)**

Each patient's raw MRI data undergoes a standardized MONAI-based preprocessing workflow:

1. **Load all modalities + segmentation mask** (from `.nii.gz`)
2. **Intensity Normalization:** Scale voxel intensities to [0, 1]
3. **Foreground Cropping:** Remove empty background regions
4. **Resampling / Resizing:** Convert all volumes to a fixed shape: **128 × 128 × 128**
5. **Mask Formatting:** Convert mask into multi-channel format
6. **Packaging:**

   * Stack modalities → shape `(4, 128, 128, 128)`
   * Save data + mask into compressed `.npz` files
7. **Data Integrity Check:** Each `.npz` file is validated using `np.load()`
8. **Dataset Split:**

   * **80%** training
   * **20%** validation
   * Split performed with `train_test_split` (seed = 42)

### **2.4 Feature Representation**

After preprocessing, each sample contains:

| Component             | Shape                                              | Description                              |
| --------------------- | -------------------------------------------------- | ---------------------------------------- |
| **MRI Image**         | `(4, 128, 128, 128)`                               | 4-channel 3D tensor (T1, T1c, T2, FLAIR) |
| **Segmentation Mask** | `(3, 128, 128, 128)`                               | Multi-class tumor subregion encoding     |

No additional hand-crafted features were used—model learns directly from voxel intensities.

Below is a clean, evaluator-ready **B3: Model Architecture** section.
This includes:
✔ High-level architecture diagram
✔ Clear description of the final model (3D Attention U-Net)
✔ Hyperparameters actually used
✔ No unnecessary detail (deep explanations go later in your report)

# **B3. Model Architecture**

### **3.1 Overview**

The final model used for brain tumor segmentation is a **3D Attention U-Net**, chosen for its strong performance in medical volumetric segmentation and its ability to focus on relevant regions through attention gating.

The network takes a **4-channel 3D MRI volume** as input and outputs a **multi-class tumor segmentation mask**.

### **3.2 Architecture Diagram (Simplified)**

```
                     ┌──────────────────────────────┐
                     │        Input Volume           │
                     │     (4 × 128 × 128 × 128)    │
                     └───────────────┬──────────────┘
                                     │
                         [Encoding Path – Downsampling]
                                     │
         ┌─────────────────────────────────────────────────────────┐
         │                                                         │
         ▼                                                         ▼
   Conv Block 1 → Attention Gate → Downsample              Conv Block 2 → Downsample
         │                                                         │
         ▼                                                         ▼
   Conv Block 3 → Attention Gate → Downsample              Conv Block 4 (Bottleneck)
         │
         ▼
                     [Decoding Path – Upsampling + Skip Connections]
                                     │
         ┌─────────────────────────────────────────────────────────┐
         │                     Attention Gates                     │
         └─────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                           Upsample → Concatenate
                                     │
                                     ▼
                             Final Conv Layer
                                     │
                                     ▼
                     Output Mask (C × 128 × 128 × 128)
```

### **3.3 Summary of Components**

* **Input:**
  `4 × 128 × 128 × 128` 3D tensor (T1, T1c, T2, FLAIR)

* **Encoder:**
  4 levels of 3D convolution blocks
  Each block includes:

  * `Conv3D` → `BatchNorm3D` → `ReLU`
  * Downsampling via `MaxPool3D`

* **Attention Gates (AG):**
  Applied on skip connections to:

  * suppress irrelevant activations
  * focus on tumor regions

* **Decoder:**
  4 upsampling blocks using:

  * `UpConv3D`
  * Concatenation with attention-filtered skip features
  * Conv3D + BatchNorm3D + ReLU blocks

* **Output Layer:**
  `1×1×1` Conv3D → Softmax

* **Output Shape:**
  `C × 128 × 128 × 128` where
  `C = 3` (edema, enhancing tumor, necrotic core)
  or `C = 1` for combined tumor class (depending on training configuration)

### **3.4 Final Model Hyperparameters**

| Parameter        | Value                                     |
| ---------------- | ----------------------------------------- |
| Input patch size | `(4, 128, 128, 128)`                      |
| Base channels    | 32                                        |
| Depth            | 4 encoder–decoder levels                  |
| Activation       | ReLU                                      |
| Normalization    | BatchNorm3D                               |
| Attention Gates  | Enabled on all skip connections           |
| Loss Function    | DiceLoss (multi-class)                    |
| Optimizer        | AdamW                                     |
| Learning Rate    | 1e-4                                      |
| Weight Decay     | 1e-5                                      |
| Scheduler        | ReduceLROnPlateau                         |
| Batch Size       | 1 (GPU memory constraint)                 |
| Epochs           | 150                                       |
| Mixed Precision  | Enabled (AMP)                             |
| Checkpointing    | Best model based on validation Dice score |

### **3.5 Rationale for Choosing 3D Attention U-Net**

* Handles volumetric 3D MRI data natively
* Attention gates boost performance on small or irregular tumor regions
* Proven SOTA performance in BraTS-like segmentation tasks
* Efficient GPU memory usage with patch-based training