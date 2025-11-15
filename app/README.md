---
title: Brain Tumor Segmentation
sdk: docker
app_port: 7860
hardware: cpu-basic # This specifies the free tier CPU hardware
---

# **Orca: Interactive 3D Brain Tumor Segmentation**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/siddhantbapna/orca)

Project Orca is an end-to-end deep learning application for the automated segmentation of brain tumors from multi-modal MRI scans. This tool leverages a **3D Attention U-Net** model, trained on the BraTS 2023 dataset, to identify and delineate three key tumor sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

The interactive web interface allows users to upload their own data or explore pre-loaded examples, visualizing the results in both 2D and 3D.

## **Live Demo**

You can access the live, interactive demo of the application hosted on Hugging Face Spaces:

**[https://huggingface.co/spaces/siddhantbapna/orca](https://huggingface.co/spaces/siddhantbapna/orca)**

## **Features**

*   **NIfTI File Upload**: Upload 4 MRI modalities (FLAIR, T1, T1ce, T2) in `.nii` format.
*   **Pre-loaded Examples**: Explore sample patient cases from the BraTS dataset without needing your own files.
*   **Automated Segmentation**: Run inference with the trained 3D Attention U-Net model with a single click.
*   **Interactive 2D Slice Viewer**: Inspect the input scans and the predicted segmentation masks slice-by-slice using a simple slider.
*   **Interactive 3D Viewer**: Visualize the predicted tumor components as 3D meshes rendered within the brain volume. Rotate, pan, and zoom for detailed inspection.
*   **CPU-Powered Inference**: The application is optimized to run on standard CPU hardware, making it accessible without requiring a dedicated GPU.

## **Technology Stack**

*   **Backend**: Flask, PyTorch, MONAI
*   **Frontend**: HTML, CSS, JavaScript
*   **3D Visualization**: Three.js
*   **Medical Imaging**: Nibabel

## **Running Locally**

Follow these steps to set up and run the application on your local machine.

### **1. Prerequisites**
*   Python 3.8 or newer
*   `git` for cloning the repository

### **2. Setup Instructions**

1.  **Clone the Repository**
    ```bash
    git clone https://huggingface.co/spaces/siddhantbapna/orca
    cd orca
    ```

2.  **Create a Virtual Environment (Recommended)**
    This isolates the project's dependencies from your system's Python environment.
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    Start the Flask server using the following command:
    ```bash
    python app.py
    ```   

5.  **Access the UI**
    Once the server is running, open your web browser and navigate to:
    [**http://127.0.0.1:7860**](http://127.0.0.1:7860)

You should now see the Orca application running in your browser.

## **Dataset and Citation**

This project was trained on the **BraTS 2023 dataset**. If you use this work in your research, please ensure you cite the original dataset providers as specified in their official documentation.

## **License**

This project is under the no License. All Rights Reserved.