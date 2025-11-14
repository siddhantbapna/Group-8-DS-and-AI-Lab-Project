# **Project Orca: Code and Training Guide**

This document provides a guide to the source code for **Project Orca**. It details the repository structure, explains the purpose of each script, and provides step-by-step instructions for running the complete data processing and model training pipeline.

## **Overview**

This directory contains all the necessary scripts and resources to reproduce the training and evaluation of the 3D Attention U-Net model for brain tumor segmentation. It also includes the source code for the deployable inference application.

## **Folder Structure**

Here is a breakdown of the key files and directories within this `Code/` folder:

```
./
├── BRATS/
│   └── ... (This folder will contain the downloaded and processed BraTS dataset)
├── models/
│   ├── best_model_fold_0_newsba4.pth
│   └── ... (Contains trained model checkpoints)
├── requirements.txt
├── download_data.py
├── preprocess.py
├── training.py
└── visual.py

../inference_app/
   ├── app.py
   └── ... (Contains the Flask application for the interactive UI)


```

*   `BRATS/`: The root directory for all patient data. The `download_data.py` script will download the raw data here, and `preprocess.py` will create a `processed/` subfolder within it.
*   `models/`: This directory stores the trained model weights (`.pth` files). The filenames indicate that these are checkpoints from a K-Fold cross-validation run (e.g., `fold_0`).
*   `Docs/`: Contains all comprehensive project documentation, including milestone reports and technical summaries.
*   `../inference_app/`: Contains the complete, self-contained Flask application for deployment. This is the code that powers the interactive web demo.
*   `requirements.txt`: A list of all Python dependencies required to run the project.
*   `download_data.py`: A script to automatically download the BraTS dataset from its source.
*   `preprocess.py`: A script that runs the full MONAI preprocessing pipeline on the raw BraTS data to prepare it for training.
*   `training.py`: The main script used to train the 3D Attention U-Net model. It handles data loading, model initialization, the training loop, validation, and saving checkpoints.
*   `visual.py`: A utility script for generating visual outputs, such as plotting segmentation masks over MRI slices, which is useful for debugging and evaluation.

## **How to Run the Full Pipeline**

Follow these steps to download the data, preprocess it, and train the model from scratch.

### **Step 1: [Setup](./requirements.txt) the Environment**

First, ensure you have Python 3.8+ installed. Then, set up a virtual environment and install the required dependencies.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```

### **Step 2: [Download the Data](./download_data.py) (CPU)**

To access the full raw dataset, visit [**Synapse.org**](https://www.synapse.org/#!Synapse:syn51156910) and **register** to obtain your **access token**.

Once you have the token, :

Run the `download_data.py` script to fetch the BraTS dataset. This may take a significant amount of time and disk space.

```bash
python download_data.py
```
This will create a `BRATS/` directory and populate it with the raw patient data.

### **Step 3: [Preprocess](./preprocess.py) the Data (CPU)**

Next, run the preprocessing script. This will apply the MONAI pipeline to normalize, resize, and crop the raw NIfTI files, saving the output in a `BRATS/processed/` directory.

```bash
python preprocess.py
```

### **Step 4: [Train](./training.py) the Model (GPU Needed)**

With the data prepared, you can now start the training process. The `training.py` script will load the preprocessed data, build the model, and begin training.

```bash
python training.py
```
During training, the script will periodically evaluate the model on the validation set and save the best-performing model checkpoints to the `models/` directory.

You can modify the **model file name or path** directly in the script to specify where to save or resume from.
If resuming, simply update the model name / checkpoint path. The script will handle the rest automatically.

## **[Inference Application](../inference_app/) (CPU)**

The code for the final, deployable web application is located in the `../inference_app/` directory. This is a standalone Flask application that serves the interactive UI.

To run the inference application locally:

1.  **Navigate to the [app directory](../inference_app/):**
    ```bash
    cd ../inference_app
    ```

2.  **Ensure the model file is in place:**
    Make sure the desired trained model (e.g., `best_model.pth`) is placed in the correct location as expected by `app.py` (e.g., `inference_app/model/`).

3.  **Run the Flask server:**
    ```bash
    python app.py
    ```

4.  **Open your browser** and go to `http://127.0.0.1:7860` to use the application.

To use it online, you can navigate to our deployed Hugging face space [Orca Segmentation](https://huggingface.co/spaces/siddhantbapna/orca)