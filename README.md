# Project ORCA

### Deep Learning for Automated Brain Tumor Segmentation

**Group-8 – DS and AI Lab Project**

━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. Project Overview

**Project ORCA** presents a deep learning based approach for **automated brain tumor segmentation** using **multi-modal MRI data**.
Manual delineation of tumors is time-consuming and prone to subjective variation among radiologists.
This project builds an **Attention U-Net–based model** that automatically segments three tumor subregions *enhancing tumor, tumor core, and whole tumor* with strong generalization capability.

Our approach combines **3D patch-based preprocessing**, **attention-driven feature learning**, and **cross-validation training** to achieve accurate and reproducible segmentation results.

## 2. Problem Statement

Brain tumor segmentation from MRI scans is crucial for diagnosis and treatment planning but is still performed manually a process that’s **slow and subjective**.
**Project ORCA** aims to develop a **deep learning–based automated segmentation model** using the **BraTS dataset**, focusing on accurately identifying tumor sub-regions (enhancing tumor, core, edema) from multi-modal MRI (T1, T1ce, T2, FLAIR).

### Objectives

* Automate brain tumor segmentation with deep learning.
* Leverage multi-modal MRI for robust predictions.
* Generate quantitative tumor volume outputs.
* Prototype a report generator for clinical interpretation.

<p align="center">
  <img src="./Milestone-1/mriModalities.jpg" width="450"/><br>
  <em>MRI modalities in BraTS dataset</em>
</p>

**Chosen Approach:** After reviewing U-Net, DeepMedic, V-Net, and Transformer-based methods, we selected **Attention U-Net** for its ability to focus on relevant tumor regions through attention gating.
**Evaluation Metric:** Dice Similarity Coefficient (DSC).

## 3. Milestone Documents

| Milestone       | Description                           | Folder Link                    |
| --------------- | ------------------------------------- | ------------------------------ |
| **Milestone 1** | Problem Statement & Literature Review | [Milestone-1/](./Milestone-1/) |
| **Milestone 2** | Dataset Analysis & Preprocessing      | [Milestone-2/](./Milestone-2/) |
| **Milestone 3** | Model Architecture & Implementation   | [Milestone-3/](./Milestone-3/) |
| **Milestone 4** | Model Training, Evaluation & Results  | [Milestone-4/](./Milestone-4/) |

## 4. Repository Structure

```bash
│   README.md
│
├───Milestone-1
│       milestone_1.md
│       mriModalities.jpg
│
├───Milestone-2
│       eda-brats2023.ipynb
│       eda-brats2023_3D-patches.ipynb
│       eda-brats2023_with_3d.ipynb
│       milestone_2.md
│
├───Milestone-3
│   │   attention-btsb.ipynb
│   │   milestone_3.md
│   │
│   └───images/
│       ├── model.png
│       ├── 0foldsb/
│       └── kfoldsb/
│
└───Milestone-4
    │   milestone_4.md
    └───images/
            torchSummary.png
            trainingGraph.png
```

## 5. Reproducibility Setup

To ensure reproducible experiments:

| Component                | Details                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| **Dataset Access**       | [BraTS 2023 Dataset – Synapse ID: syn51156910](https://www.synapse.org/#!Synapse:syn51156910) |
| **Environment File**     | `requirements.txt`                                                                            |
| **Random Seed**          | `random_state = 42`                                                                           |
| **Data Split**           | 80% training / 20% validation                                                                 |
| **Sequential Notebooks** | Run EDA → Preprocessing → Model Training → Evaluation                                         |
| **Hardware**             | NVIDIA GPU (tested on Kaggle P100)                                                            |

## 6. Code Files and Usage

1. Downloading Raw Data and Preprocessing : 
2. Initial Model Training : 
3. Resume Model Training : 
4. Making Predictions and Visualizing : 



## 7. Team Members

- **Siddhant Bapna**
- **Ajsal**
- **Saurabh**
- **Ravineel Singhi**
- **Hardik Bapna**


## 8. License

### a) Project License

This project itself is under no licensed.

### b) Dataset Credits and License

This project uses the **BraTS 2023 Dataset**, licensed under **CC BY-NC 4.0** for academic use.

## 9. Acknowledgements

This work builds upon the **BraTS** benchmark dataset and the **MONAI** medical imaging framework.

Special thanks to:

* **synapse** for open dataset access.
* **MONAI team** for robust medical imaging tools
* **Kaggle community** for accessible GPU infrastructure

### Dataset Citations

> A. Karargyris et al., “Federated benchmarking of medical artificial intelligence with MedPerf”, *Nature Machine Intelligence*, 2023. [DOI:10.1038/s42256-023-00652-2](https://doi.org/10.1038/s42256-023-00652-2)
> U. Baid et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark”, *arXiv:2107.02314*, 2021.
> B. H. Menze et al., “The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)”, *IEEE TMI*, 2015.
> S. Bakas et al., “Segmentation Labels and Radiomic Features for the TCGA Collections”, *The Cancer Imaging Archive*, 2017.

━━━━━━━━━━━━━━━━━━━━━━━━━━━

*Last Updated: November 2025*
*Maintained by Group 8 – Project ORCA*