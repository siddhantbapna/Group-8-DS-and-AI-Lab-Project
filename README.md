# Project ORCA  
### Deep Learning for Automated Brain Tumor Segmentation  
**Group 8 — B.Sc. Data Science (DS & AI Lab Project)**  

---

## Abstract

**Project ORCA** presents a deep learning–based approach for **automated brain tumor segmentation** using **multi-modal MRI data**.  
Manual delineation of tumors is time-consuming and prone to subjective variation among radiologists.  
This project builds an **Attention U-Net–based model** that automatically segments three tumor subregions — enhancing tumor, tumor core, and whole tumor — with strong generalization capability.

Our approach combines **3D patch-based preprocessing**, **attention-driven feature learning**, and **cross-validation training** to achieve accurate and reproducible segmentation results.  

---

## Problem Statement (Milestone 1)

Brain tumors vary in size, shape, and location, making manual segmentation from MRI scans a **complex and inconsistent** process.  
The **need for automation** arises from:

- The **sheer volume** of MRI data radiologists must analyze.
- The **inherent subjectivity** of manual tumor boundary delineation.
- The **clinical importance** of tumor subregion identification for treatment planning.

**Goal:**  
> Develop an automated deep learning model capable of accurately segmenting brain tumors across multiple MRI modalities, while maintaining high Dice and IoU scores comparable to expert annotations.

**Challenges Identified:**
- Data imbalance between tumor subregions.
- Need for 3D spatial context in segmentation.
- Large dataset size and preprocessing complexity.

---

## Dataset & Preprocessing (Milestone 2)

### Dataset
- **Dataset:** BraTS 2023 (RSNA-ASNR-MICCAI Challenge)
- **Modalities Used:**
  - T1-weighted
  - T1Gd (post-contrast)
  - T2-weighted
  - FLAIR
- **Ground Truth Labels:**  
  - **ET (Enhancing Tumor)**
  - **TC (Tumor Core)**
  - **WT (Whole Tumor)**

### Preprocessing Pipeline
1. **NIfTI File Handling** — Loaded `.nii.gz` MRI volumes.
2. **Normalization:**
   - Applied z-score normalization to each modality.
   - Ensured consistent intensity distributions.
3. **Resampling:**
   - Resized all volumes to `(128, 128, 128)` for uniformity.
4. **3D Patch Extraction:**
   - Split volumes into overlapping **3D patches (64×64×64)**.
   - Reduced memory load for training.
5. **Augmentation:**
   - Random flips, rotations, and brightness scaling to improve robustness.

### Exploratory Data Analysis
- Visualized intensity distributions per modality.
- Compared tumor volume ratios across subregions.
- Confirmed class balance and dataset integrity.

> *For detailed visualizations and code, refer to:*  
> `Milestone-2/eda-brats2023_with_3d.ipynb`

---

## Model Architecture (Milestone 3)

### Architecture: Attention U-Net
Project ORCA builds upon **U-Net**, enhanced with **Attention Gates (AGs)** to focus on relevant spatial regions.

#### Key Components:
| Layer | Function |
|--------|-----------|
| **Encoder** | 3D convolutions + batch normalization + ReLU |
| **Attention Gates** | Dynamically suppress irrelevant background features |
| **Decoder** | Skip connections + upsampling for spatial reconstruction |
| **Output Layer** | 1×1 convolution → sigmoid activation |

<p align="center">
  <img src="./Milestone-3/images/model.png" width="550"/>
</p>

#### Implementation:
- Framework: **PyTorch**
- Input size: **(4, 128, 128, 128)** (4 MRI modalities)
- Loss Function: **Dice Loss + Cross Entropy**
- Optimizer: **AdamW**
- Learning Rate: **1e-4**
- Scheduler: **ReduceLROnPlateau**

> **Why Attention U-Net?**  
> Unlike vanilla U-Net, the Attention U-Net dynamically highlights tumor-relevant regions while suppressing background noise — improving segmentation precision on heterogeneous MRI volumes.

---

## Model Training (Milestone 4)

### Training Setup
| Parameter | Value |
|------------|--------|
| **Epochs** | 150 |
| **Batch Size** | 4 |
| **Optimizer** | AdamW |
| **Loss Function** | Dice + CrossEntropy |
| **Validation Split** | 20% |
| **Cross-Validation** | 5-Fold (K-Fold SB configuration) |

### Training Procedure
- Monitored loss and Dice score per epoch.
- Saved best checkpoints (`val_loss`-based).
- Used early stopping to prevent overfitting.
- Employed **GPU acceleration** (Colab T4 GPU).

### Sample Outputs
| Visualization | Description |
|----------------|--------------|
| ![](./Milestone-3/images/0foldsb/SBAttentionUnet_BestTraining.png) | Loss vs. Dice per epoch |
| ![](./Milestone-4/images/trainingGraph.jpg) | Final Training Curve |
| ![](./Milestone-3/images/0foldsb/SBAttentionUnet_ResultOfBest_1.png) | Sample Segmentation Results |

---

## Evaluation Metrics

| Metric | Description | Formula |
|---------|--------------|----------|
| **Dice Score** | Overlap between prediction and ground truth | 2TP / (2TP + FP + FN) |
| **IoU** | Intersection over Union | TP / (TP + FP + FN) |
| **Sensitivity** | True positive rate | TP / (TP + FN) |
| **Specificity** | True negative rate | TN / (TN + FP) |

### Quantitative Results

| Region | Dice | IoU |
|---------|------|-----|
| Whole Tumor | 0.89 | 0.80 |
| Tumor Core | 0.84 | 0.76 |
| Enhancing Tumor | 0.81 | 0.73 |

> *The Attention U-Net demonstrated strong generalization across K-Folds, achieving mean Dice > 0.85.*

---

## Summary of Findings

- The Attention U-Net achieved consistent segmentation performance with strong localization of tumor regions.  
- Incorporating 3D patch extraction improved training stability and reduced GPU memory usage.  
- Dice scores indicated reliable overlap with expert annotations.

---

## Future Work

- Extend to **multi-task learning** with survival prediction.  
- Implement **Transformer-based U-Net variants (TransUNet)**.  
- Optimize inference time for deployment in clinical settings.

---

## Key Takeaways

| Aspect | Approach | Outcome |
|---------|-----------|----------|
| **Data** | BraTS 2023 with 4 MRI modalities | Rich volumetric representation |
| **Model** | Attention U-Net | Focused segmentation |
| **Training** | K-Fold Cross-Validation | Stable generalization |
| **Performance** | Dice ≈ 0.85–0.89 | High accuracy |
| **Goal** | Automated clinical segmentation | Achieved |

---

## References

1. Baid, U. et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark*, arXiv:2107.02314 (2021)  
2. Menze, B.H. et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*, IEEE TMI, 2015  
3. Oktay, O. et al., *Attention U-Net: Learning Where to Look for the Pancreas*, arXiv:1804.03999 (2018)

---

## Quick Links

| Section | Folder |
|----------|---------|
| Milestone 1: Problem Definition | [Milestone-1/](./Milestone-1/) |
| Milestone 2: EDA & Preprocessing | [Milestone-2/](./Milestone-2/) |
| Milestone 3: Model Architecture | [Milestone-3/](./Milestone-3/) |
| Milestone 4: Training & Evaluation | [Milestone-4/](./Milestone-4/) |

---

> **Note:**  
> This README contains all major project information — problem context, methods, architecture, results, and references — eliminating the need to open milestone files unless detailed analysis or code inspection is required.

---

*Last Updated: October 2025*  
*Maintained by Group 8 — Project ORCA*