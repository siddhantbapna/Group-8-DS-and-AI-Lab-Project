# Project ORCA

### Deep Learning for Automated Brain Tumor Segmentation

**Group-8 – DS and AI Lab Project**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/siddhantbapna/orca)

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
  <img src="./Docs/Milestone-1/mriModalities.jpg" width="450"/><br>
  <em>MRI modalities in BraTS dataset</em>
</p>

**Chosen Approach:** After reviewing U-Net, DeepMedic, V-Net, and Transformer-based methods, we selected **Attention U-Net** for its ability to focus on relevant tumor regions through attention gating.
**Evaluation Metric:** Dice Similarity Coefficient (DSC).

## 3. Milestone Documents

| Milestone       | Description                           | Folder Link                         |
| --------------- | ------------------------------------- | ------------------------------      |
| **Milestone 1** | Problem Statement & Literature Review | [Milestone-1/](./Docs/Milestone-1/) |
| **Milestone 2** | Dataset Analysis & Preprocessing      | [Milestone-2/](./Docs/Milestone-2/) |
| **Milestone 3** | Model Architecture & Implementation   | [Milestone-3/](./Docs/Milestone-3/) |
| **Milestone 4** | Model Training, Validation            | [Milestone-4/](./Docs/Milestone-4/) |
| **Milestone 5** | Evaluation & Results                  | [Milestone-5/](./Docs/Milestone-5/) |
| **Milestone 6** | Model Deplyment                       | [Milestone-6/](./Docs/Milestone-6/) |

## 4. Repository Structure

```bash
├── app
│   ├── data
│   │   └── uploads
│   ├── examples
│   │   └── 1
│   ├── model
│   └── templates
│
├── Code
│   ├── BRATS
│   │   └── processed
│   └── models
│
├── Docs
│   ├── Milestone-1
│   ├── Milestone-2
│   ├── Milestone-3
│   │   └── images
│   │       ├── 0foldsb
│   │       └── kfoldsb
│   ├── Milestone-4
│   │   └── images
│   │       ├── epochPerformace
│   │       ├── hyperparameterTuning
│   │       └── ValidationPerformance
│   │           ├── BraTS-GLI-00132-000
│   │           ├── BraTS-GLI-00426-000
│   │           └── BraTS-GLI-01265-000
│   ├── Milestone-5
│   │   ├── checkpointResults
│   │   │   ├── 55
│   │   │   ├── 67
│   │   │   └── 93_latest
│   │   └── images
│   │       ├── errorAnalysis
│   │       └── testPerformance
│   │           ├── BraTS-GLI-02405-100
│   │           ├── BraTS-GLI-02426-100
│   │           └── BraTS-GLI-02506-101
│   └── Milestone-6
│       └── visuals
│           ├── errorAnalysis
│           ├── testPerformance
│           │   ├── BraTS-GLI-02405-100
│           │   ├── BraTS-GLI-02426-100
│           │   └── BraTS-GLI-02506-101
│           └── validationPerformance
│               ├── BraTS-GLI-00132-000
│               ├── BraTS-GLI-00426-000
│               └── BraTS-GLI-01265-000
│
└── Worklog
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

Here’s an improved, professional, and more readable version of your **GitHub README section**, with consistent formatting, better grammar, and clearer step-by-step instructions:

---

## 6. Code Files and Usage

All source code is available in the [**Code Folder**](./Code).



### Setup Instructions for Training



**Step 1:** Clone the repository

```bash
git clone https://github.com/siddhantbapna/Group-8-DS-and-AI-Lab-Project.git
```

**Step 2:** Navigate to the code directory

```bash
cd Group-8-DS-and-AI-Lab-Project/Code
```

**Step 3:** Install the required dependencies

```bash
pip install -r requirements.txt
```


### Downloading the Dataset

To access the full raw dataset, visit [**Synapse.org**](https://www.synapse.org/#!Synapse:syn51156910) and **register** to obtain your **access token**.

Once you have the token, run:

```bash
python download_data.py
```

You’ll be prompted to enter the token, and the script will automatically start downloading the dataset.



### Data Preprocessing

After downloading the raw data, preprocess it by running:

```bash
python preprocess.py
```

This will clean and prepare the dataset for training using the preprocessing pipeline.



### Model Training

To train or resume training of the model, execute:

```bash
python training.py
```

You can modify the **model file name or path** directly in the script to specify where to save or resume from.
If resuming, simply update the model checkpoint path. The script will handle the rest automatically.



### Visualization and Testing

To visualize model performance or run testing on saved checkpoints, run:

```bash
python visual.py
```

and for testing you can run the `testing.ipunb` notebook

Provide the model path and patient data as inputs.
The script will generate volumetric analysis results and relevant visualizations.

```bash
python visual.py
```

and for testing you can run the `testing.ipunb` notebook

Provide the model path and patient data as inputs.
The script will generate volumetric analysis results and relevant visualizations.

### Inference

You can run the inference of the model locally or on the deplyed hugging face space.

- For Local : [Application Guide](./app/userguide.md)
- For Remote : [Hugging Face Spcae](https://huggingface.co/spaces/siddhantbapna/orca)


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

**A. Karargyris et al.**, *Federated benchmarking of medical artificial intelligence with MedPerf*, **Nature Machine Intelligence**, 2023.  
DOI: [10.1038/s42256-023-00652-2](https://doi.org/10.1038/s42256-023-00652-2)

**U. Baid et al.**, *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark*, **arXiv:2107.02314**, 2021.

**B. H. Menze et al.**, *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*, **IEEE Transactions on Medical Imaging**, 2015.

**S. Bakas et al.**, *Segmentation Labels and Radiomic Features for the TCGA Collections*, **The Cancer Imaging Archive (TCIA)**, 2017.

━━━━━━━━━━━━━━━━━━━━━━━━━━━

*Last Updated: November 2025*
*Maintained by Group 8 – Project ORCA*