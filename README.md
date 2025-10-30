# 🧠 Project ORCA  
### Deep Learning for Automated Brain Tumor Segmentation  
**Group 8 — DS & AI Lab Project**

---

## 📘 Overview
Accurate delineation of brain tumors from multi-modal MRI scans is crucial for diagnosis, surgical planning, and treatment monitoring.  
Manual segmentation by radiologists, however, is **time-consuming, labor-intensive**, and subject to **inter-observer variability**.

**Project ORCA** aims to develop a **deep learning–based automatic segmentation model** that can detect and classify tumor sub-regions across different MRI modalities.  
This tool seeks to provide a **quantitative, reproducible, and objective** alternative for tumor assessment and clinical decision support.

---

## 🎯 Objectives
- Develop a **multi-class segmentation model** for brain tumors.  
- Leverage **multi-modal MRI data** (T1, T1Gd, T2, FLAIR).  
- Incorporate **attention-based architectures (e.g., Attention U-Net)** for improved precision.  
- Evaluate model performance using metrics such as **Dice, IoU, and Sensitivity**.  
- Build a **reproducible training pipeline** for consistent results.

---

## 🗂️ Repository Structure

```bash
Project-ORCA/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Dependencies
├── LICENSE
│
├── internal/                        ← Model weights & internal data
│   └── sb/
│       └── sb_brats_attention_model.pth
│
├── utils/                           ← Helper scripts (metrics, data utils, viz)
│
├── references/                      ← Dataset and paper citations
│   ├── dataset_citations.md
│   ├── literature_refs.md
│   └── references.bib
│
├── M1_Problem_Definition_LitReview/
│   ├── 01_problem_definition.md
│   ├── 02_literature_review.md
│   ├── 03_project_scope_objectives.md
│   └── images/
│
├── M2_Data_Preparation_EDA/
│   ├── eda_brats2023.ipynb
│   ├── data_preprocessing.py
│   ├── 04_data_preparation.md
│   └── images/
│
├── M3_Model_Architecture/
│   ├── model.py
│   ├── attention_unet.py
│   ├── 05_model_architecture.md
│   ├── model_summary.txt
│   └── results/
│
├── M4_Model_Training/
│   ├── train_model.ipynb
│   ├── evaluate_model.ipynb
│   ├── inference_demo.ipynb
│   ├── 06_model_training_v1.md
│   ├── models/
│   └── results/
│
└── reports/
    ├── Project_ORCA_Final_Report.pdf
    ├── presentation_slides.pptx
    └── milestone_summary_table.md
````

---

## 📅 Milestone Progress

| Milestone | Deliverable                                                                                          | Description                                                         | Status         |
| --------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------- |
| **M1**    | [Problem Definition & Literature Review](./M1_Problem_Definition_LitReview/01_problem_definition.md) | Defined project goals, reviewed prior work, identified research gap | ✅ Completed    |
| **M2**    | [Dataset Preparation & EDA](./M2_Data_Preparation_EDA/04_data_preparation.md)                        | Dataset acquisition, preprocessing, and EDA                         | ✅ Completed    |
| **M3**    | [Model Architecture](./M3_Model_Architecture/05_model_architecture.md)                               | Designed and justified model architectures                          | ✅ Completed    |
| **M4**    | [Model Training](./M4_Model_Training/06_model_training_v1.md)                                        | Training, tuning, and performance evaluation                        | ✅ Completed    |
| **M5**    | Model Evaluation & Report                                                                            | Cross-validation, visualization, and report submission              | 🔜 In Progress |

---

## ⚙️ How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Project-ORCA.git
cd Project-ORCA
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Run preprocessing

```bash
python M2_Data_Preparation_EDA/data_preprocessing.py
```

### 4. Train the model

Open and run:

```
M4_Model_Training/train_model.ipynb
```

### 5. Evaluate or infer

```
M4_Model_Training/evaluate_model.ipynb
M4_Model_Training/inference_demo.ipynb
```

---

## 🧠 Model Summary

| Component             | Description                         |
| --------------------- | ----------------------------------- |
| **Base Architecture** | Attention U-Net                     |
| **Framework**         | PyTorch                             |
| **Input Modalities**  | T1, T1Gd, T2, FLAIR                 |
| **Loss Function**     | Dice + Cross-Entropy                |
| **Optimizer**         | AdamW                               |
| **Metrics**           | Dice, IoU, Sensitivity, Specificity |

---

## 📈 Preliminary Results

| Metric                     | Value |
| -------------------------- | ----- |
| **Dice (Whole Tumor)**     | 0.89  |
| **Dice (Tumor Core)**      | 0.84  |
| **Dice (Enhancing Tumor)** | 0.81  |
| **IoU (Mean)**             | 0.77  |

*Detailed results and visualizations are available in [M4_Model_Training/results/](./M4_Model_Training/results/).*

---

## 📚 Dataset Citations

1. **Baid, U. et al.**
   “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification.”
   *arXiv:2107.02314 (2021)*. [Link](https://arxiv.org/abs/2107.02314)

2. **Karargyris, A. et al.**
   “Federated benchmarking of medical artificial intelligence with MedPerf.”
   *Nature Machine Intelligence*, 5:799–810 (2023).
   [DOI: 10.1038/s42256-023-00652-2](https://doi.org/10.1038/s42256-023-00652-2)

3. **Menze, B.H. et al.**
   “The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).”
   *IEEE Trans. on Medical Imaging*, 34(10):1993–2024 (2015).
   [DOI: 10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694)

*(Full list in [`references/dataset_citations.md`](./references/dataset_citations.md))*

---

## 👥 Team ORCA (Group 8)

| Member   | Role                  | Contributions                                     |
| -------- | --------------------- | ------------------------------------------------- |
| [Name 1] | Data & Preprocessing  | Dataset handling, normalization, patch extraction |
| [Name 2] | Modeling              | Architecture design, model development            |
| [Name 3] | Training & Evaluation | Hyperparameter tuning, results analysis           |
| [Name 4] | Documentation         | Reports, visualization, and repo organization     |

---

## 🧾 License & Acknowledgments

* **Dataset:** RSNA-ASNR-MICCAI BraTS Challenge 2021, TCGA Collections
* **References:** Listed in `references/`
* **License:** MIT (or specify alternative)

---

## 🔗 Quick Navigation

| Section             | File/Folder                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| Problem Definition  | [M1_Problem_Definition_LitReview/](./M1_Problem_Definition_LitReview/) |
| Dataset Preparation | [M2_Data_Preparation_EDA/](./M2_Data_Preparation_EDA/)                 |
| Model Architecture  | [M3_Model_Architecture/](./M3_Model_Architecture/)                     |
| Model Training      | [M4_Model_Training/](./M4_Model_Training/)                             |
| References          | [references/](./references/)                                           |

---

> 🧩 **Note for Evaluators:**
> Each milestone folder contains a `README.md` with internal navigation, code notebooks, and artifacts relevant to that phase.
> All notebooks are commented, versioned, and aligned with the DS & AI Lab project evaluation milestones.

---

*Last updated: October 2025*
*Maintained by Group-8 (DS & AI Lab)*