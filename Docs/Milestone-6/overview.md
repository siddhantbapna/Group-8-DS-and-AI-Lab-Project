# **A. Overview**

## **1. Purpose**

The goal of this project is to build a **3D brain tumor segmentation system** that automatically identifies and segments tumor subregions from **multi-modal MRI scans**.
This assists clinicians and researchers by providing:

* Faster and more consistent tumor localization
* Automated volumetric measurements
* A fully reproducible medical-imaging pipeline
* Deployment-ready inference suitable for web applications

The system is designed to be **accurate, modular, and production-ready**, covering preprocessing → model inference → visualization in both 2D and 3D.

## **2. Architecture Summary**

Below is a simplified high-level view of how data flows through the system:

```
Input MRI (FLAIR, T1, T1ce, T2)
              ↓
        Preprocessing (MONAI)
              ↓
      3D Attention U-Net Model
              ↓
 Post-processing + Mesh Generation
              ↓
2D Slice Viewer + 3D Tumor Visualization
```

<p align="center">
  <img src="./visuals/model.png">
  <br>
  <em>Figure: 3D Attention U-Net</em>
</p>

## **3. Deployed Components**

### **Frontend (HTML + CSS + JavaScript)**

* Simple **single-page application (SPA)**
* Provides:
  * File upload UI
  * Buttons to trigger inference
  * Embedded **2D slice viewer**
  * Interactive **3D tumor mesh viewer (Three.js)**
* Communicates with backend via **Fetch API**

**Live URL:**
`https://huggingface.co/spaces/siddhantbapna/orca`

### **Backend API (Flask + Python)**

The backend is implemented using **Flask**, responsible for:
* Handling MRI uploads
* Running **MONAI preprocessing**
* Performing inference using the **trained 3D Attention U-Net**
* Post-processing outputs
* Generating:
  * 2D PNG slice overlays
  * 3D mesh objects for visualization

**Endpoints:**
* `POST /predict`
  * Accepts the four MRI modalities
  * Returns predicted segmentation + 2D + 3D data
* `GET /example`
  * Provides a sample patient case

**Deployment Target:**
Deployed as a **Docker container** on **Hugging Face Spaces**.
(Also runnable locally using Docker or Python.)

## **4. Visual Overview of the System**

[Insert 1–2 screenshots here]

