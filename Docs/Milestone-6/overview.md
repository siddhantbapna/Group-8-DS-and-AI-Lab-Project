### **Comprehensive Overview Report: Project Orca**
**Automated Brain Tumor Segmentation Using Deep Learning**

#### **1. Introduction and Project Charter**

Project Orca is an end-to-end, AI-driven system developed by Team 8 of the IITM BS Degree Program to address the critical clinical challenge of brain tumor segmentation. The project's primary motivation stems from the limitations of manual tumor delineation in clinical practice, which is time-intensive (30-90 minutes per case), prone to inter-observer variability, and operationally inefficient.

The project's proposed solution is a fully automated pipeline that leverages a **3D Attention U-Net** deep learning model to segment gliomas from multi-modal MRI scans. The system is designed to be **automated**, **accurate**, **robust**, and **accessible**, providing quantitative volumetric outputs and a user-friendly web interface to support radiologists and neuro-oncologists.

**Project Objectives:**
*   **Performance:** Achieve competitive Dice scores against the BraTS challenge benchmarks.
*   **Automation:** Create a reproducible, end-to-end workflow from raw MRI input to final segmentation results.
*   **Clinical Integration:** Provide actionable volumetric data and intuitive visualizations through a web-based interface.
*   **Reliability:** Improve generalization across different cases and minimize false predictions.

---

#### **2. Foundation: Datasets and State-of-the-Art Context**

The project is built upon the globally recognized gold standard for this task: **The Brain Tumor Segmentation (BraTS) Challenge dataset**. Specifically, the project utilized:
*   **Training Data:** 1,251 multi-modal MRI samples from the **BraTS-GLI 2023 Challenge**.
*   **Testing Data:** 150 completely unseen samples from the **BraTS-GLI 2024 Challenge** to ensure an unbiased evaluation of the model's generalization capabilities.

The model is trained to identify three clinically relevant, nested sub-regions:
1.  **Whole Tumor (WT):** The entire tumor mass, including the core, edema (swelling), and enhancing parts.
2.  **Tumor Core (TC):** The solid part of the tumor, excluding the surrounding edema.
3.  **Enhancing Tumor (ET):** The most active part of the tumor, which enhances with a contrast agent and is a key indicator of malignancy.

---

#### **3. The Development Pipeline: A Multi-Stage Methodology**

Project Orca's development followed a structured, multi-stage process from data preparation to model optimization.

**Stage 1: Exploratory Data Analysis (EDA) and Preprocessing**
Initial analysis of the BraTS data confirmed consistent scan dimensions but revealed significant variability in intensity values, noise patterns, and tumor shapes. A robust, fully automated preprocessing pipeline was developed using the **MONAI framework** to standardize the data. This critical pipeline involved:
1.  **Loading** the four MRI modalities (T1, T1ce, T2, FLAIR) and the segmentation mask.
2.  **Converting** the single-channel label mask into a three-channel binary mask for WT, TC, and ET.
3.  **Resampling** all volumes to a uniform isotropic voxel spacing (1.0x1.0x1.0 mm).
4.  **Normalizing** voxel intensity values to a standard range of [0.0, 1.0].
5.  **Cropping** the foreground to remove empty background space and focus on the brain.
6.  **Resizing** all volumes to a fixed spatial dimension of **(128, 128, 128)** for uniform model input.

Finally, an **early fusion** strategy was employed, stacking the four modalities to create a single 4-channel, 3D tensor input for the model.

**Stage 2: Model Architecture and Experimental Trials**
The core of the system is a **3D Attention U-Net**. This architecture was chosen for its ability to process full 3D volumes, combine high-level and low-level features via skip connections, and focus on relevant tumor regions using attention gates.

Initial experiments explored K-Fold Cross-Validation (deemed too computationally expensive) and 3D Patch-Based Training (found to be misleading as the model learned to predict background). Consequently, the team proceeded with full-volume training.

**Stage 3: Hyperparameter Sweep and Final Training Configuration**
A systematic hyperparameter sweep was conducted to find the optimal model configuration. After extensive testing, the best-performing and most stable setup was finalized:
*   **Model Channels:** (16, 32, 64, 128, 256) for the encoder-decoder path.
*   **Optimizer:** AdamW.
*   **Learning Rate:** 0.0001 with a Polynomial Decay scheduler.
*   **Batch Size:** 2.
*   **Loss Function:** A combined loss of **0.5 x Dice Loss** (to handle class imbalance) and **0.5 x BCE Loss** (to stabilize training).
*   **Regularization:** Weight Decay (1e-05) and extensive data augmentation (random flips, rotations, intensity scaling).
The model was trained for **93 epochs** on a Kaggle NVIDIA Tesla P100 GPU, with early stopping based on validation Dice score improvement.

---

#### **4. Evaluation and Results: A Detailed Performance Analysis**

The final model was rigorously evaluated on the 150-case unseen test set.

**4.1. Quantitative Performance**
The Dice Similarity Coefficient (DSC) was the primary metric:

| Tumor Label           | Mean DSC | Median DSC | Standard Deviation |
| --------------------- | :------: | :--------: | :----------------: |
| **Whole Tumor (WT)**    |  0.6892  |   0.8393   |       0.3140       |
| **Tumor Core (TC)**     |  0.5947  |   0.7515   |       0.3489       |
| **Enhancing Tumor (ET)** |  0.5079  |   0.6373   |       0.3267       |

**Key Findings:**
*   The model performed best on **WT**, followed by TC, and was weakest on ET.
*   Performance on WT was competitive with the **BraTS 2023 leaderboard average (0.87 vs 0.86 on the validation set)**.
*   Linear regression analysis revealed a **systematic under-prediction of tumor volumes** across all classes (regression slopes < 1).
*   There were **22 complete misses** for the Enhancing Tumor, highlighting the difficulty in detecting small, sparse regions.

**4.2. Qualitative and Error Analysis**
*   **Successful Cases:** Large, well-defined tumors consistently produced high Dice scores (>0.9), with predicted masks nearly identical to the ground truth.
*   **Challenging Cases:** Small, low-contrast, or diffusely infiltrating tumors resulted in fragmented predictions, false positives, or complete misses.

The analysis categorized failures into three main types:
1.  **Complete Misses:** Primarily affecting ET when regions were small (<2 ml).
2.  **Fragmented/Noisy Predictions:** Common in tumors with ambiguous boundaries.
3.  **Under-Segmentation of Large Tumors:** The model captured the central bulk but often missed irregular peripheral extensions.

---

#### **5. System Deployment and User Workflow**

The project culminated in a fully functional, end-to-end web application with a clear system architecture:
*   **Backend:** A lightweight **Flask** server in Python handles all core logic.
*   **Frontend:** A single-page application built with **HTML, CSS, and JavaScript** provides the user interface.
*   **Deployment:** The application is containerized with **Docker** and deployed on Hugging Face Spaces for public demonstration.

The user workflow is streamlined into three simple steps:
1.  **Load Data:** Upload a set of four NIfTI files or select a pre-loaded example.
2.  **Run Prediction:** A single button triggers the entire backend inference pipeline.
3.  **Visualize Results:** Inspect the output using an interactive **2D slice viewer** and a **3D volume viewer** powered by `three.js`.

---

#### **6. Conclusion and Future Directions**

Project Orca successfully achieved its objective of creating a complete, deployable system for automated brain tumor segmentation. The 3D Attention U-Net demonstrates strong performance for delineating the Whole Tumor and Tumor Core, showcasing the potential of deep learning to enhance clinical research workflows.

However, the evaluation also transparently highlighted the persistent challenge of accurately segmenting small and variable sub-regions like the Enhancing Tumor. Based on these findings, the report proposes several avenues for future work:
*   **Explore advanced model architectures** (e.g., ResU-Net, Vision Transformers).
*   **Fine-tune the model on real-world clinical data** from diverse institutions to improve generalization.
*   **Integrate a Generative AI model (LLM)** to translate the quantitative outputs into a human-readable report summary.