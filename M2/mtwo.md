### **Milestone 2: Dataset Preparation [October 10]**

**Official Documentation: Full Pipeline for Dataset Collection, Preprocessing, and Structuring**

This document provides a comprehensive overview of the procedures executed to fulfill Milestone 2. The objectives for this milestone were to: identify and collect a suitable dataset, prepare and preprocess the data for model consumption, verify the integrity of the processed data, and finally, split the data into distinct sets for training and validation.

---

### **1. The Dataset: BraTS (Brain Tumor Segmentation)**

A robust machine learning model is built upon a high-quality, relevant dataset. For this project, we have selected the internationally recognized **BraTS (Brain Tumor Segmentation) dataset**.

*   **What is the BraTS Dataset?**
    The BraTS dataset is a collection of 3D MRI (Magnetic Resonance Imaging) scans from patients with brain tumors. It is the gold standard for developing and benchmarking algorithms for brain tumor segmentation. Each patient's record in the dataset includes:
    *   **Multi-modal MRI Scans:** Four different types of MRI scans, each highlighting different tissue properties:
        1.  **T1-weighted (T1):** Provides good contrast for anatomical structures.
        2.  **T1-weighted with contrast enhancement (T1c):** A T1 scan where a contrast agent makes the active tumor regions more visible.
        3.  **T2-weighted (T2w):** Particularly effective at showing areas with high water content, like edema (swelling) around the tumor.
        4.  **T2-weighted FLAIR (T2f):** Similar to T2, but suppresses the signal from cerebrospinal fluid, making lesions near fluid-filled spaces easier to see.
    *   **Ground Truth Segmentation Mask (seg):** A precise, manually-labeled 3D map created by medical experts. This mask identifies the exact locations of different tumor sub-regions, serving as the "answer key" for our model during training.

*   **Why Was the BraTS Dataset Chosen?**
    This dataset was chosen for several critical reasons:
    *   **Clinical Relevance:** It directly addresses the project's goal of brain tumor analysis.
    *   **Benchmark Standard:** It is used globally for the annual BraTS Challenge, meaning any model developed on it can be fairly compared to state-of-the-art solutions.
    *   **Large Scale and High Quality:** The dataset is large, multi-institutional, and has been carefully curated and annotated by experts, ensuring its reliability.
    *   **Publicly Available:** Its availability allows for reproducible and transparent research.

*   **Data Source and Acquisition (Wherefrom and How)**
    The data was sourced from the official BraTS Challenge repositories, which are made available through platforms like Synapse.org, The Cancer Imaging Archive (TCIA) and Kaggle. It represents a collaborative effort from numerous institutions to provide a standardized resource for the medical imaging community. For this project, the dataset was downloaded and stored in a designated working directory for preprocessing.


### **2. Exploratory Data Analysis (EDA)**

Before preprocessing, we conducted an Exploratory Data Analysis to understand the dataset's intrinsic properties, uncover potential challenges, and make data-driven decisions for the pipeline design.

*   **Purpose of EDA:** The goal was to answer key questions: Are the scans uniform? What do the different MRI modalities show? What is the distribution of tumor sizes and classes? The insights from this analysis directly guided the choice of our preprocessing steps.

*   **Methodology and Key Findings:**

    1.  **Visual Inspection of Scans and Masks:**
        *   We visualized random samples of the four MRI modalities and their corresponding segmentation masks. This confirmed that each modality provides unique information: T1c clearly highlights the active tumor core, while T2/FLAIR are excellent for visualizing edema (swelling).
        *   Overlaying the masks on the scans verified that the annotations were correctly aligned with the anatomical structures.

    2.  **Analysis of Data Consistency and Distribution:**
        *   **Image Dimensions:** We found that the 3D dimensions (height, width, depth) of the scans were **highly inconsistent** across different patients. Some scans were `240x240x155`, while others had different dimensions.
        *   **Voxel Spacing:** The physical size represented by each voxel was also **not uniform**. This is a critical issue, as it means a tumor might appear larger in one scan than another simply due to different image resolutions.
        *   **Intensity Values:** The range of voxel intensity values varied significantly from one scan to another, even within the same modality. This could negatively bias a machine learning model.
        *   **Tumor Presence:** We confirmed that not all tumor sub-classes (e.g., enhancing tumor, necrotic core) were present in every single patient, a factor the model must be robust enough to handle.

    3.  **Conclusion from EDA:**
        The EDA made it clear that the raw data was not suitable for direct use. The inconsistencies in size, spacing, and intensity **mandated** a strong, standardized preprocessing pipeline to bring all data into a uniform and comparable format.

### **3. The Preprocessing Pipeline**

Raw medical data cannot be fed directly into a neural network. It must be rigorously cleaned and standardized. Our automated script was designed to perform this crucial task efficiently.

*   **Overview and Objectives:**
    The primary goal of the preprocessing pipeline is to convert the raw, heterogeneous BraTS data into a uniform, model-ready format. This involves standardizing properties like image size, voxel spacing, and intensity, and packaging the data for efficient loading during training.

*   **Key Technologies:**
    *   **MONAI:** A specialized, high-performance PyTorch-based library for deep learning in medical imaging.
    *   **Parallel Processing:** The script leverages all available CPU cores to process multiple patient files simultaneously, dramatically reducing execution time.

*   **Step-by-Step Workflow:**
    For every patient, a series of automated transformations are applied using a MONAI pipeline:
    1.  **Load Images:** The four MRI scans and the segmentation mask are loaded from their `.nii.gz` files.
    2.  **Standardize Voxel Spacing:** All scans are resampled to a uniform voxel spacing of `(1.0, 1.0, 1.0) mm` to ensure spatial consistency.
    3.  **Normalize Intensity:** Image brightness values are scaled to a standard `[0.0, 1.0]` range.
    4.  **Crop Foreground:** The script automatically removes large empty background areas, focusing computational resources on the brain itself.
    5.  **Resize:** All 3D scans are resized to a fixed shape of `(128, 128, 128)`, ensuring uniform input size for the model.
    6.  **Format Mask:** The segmentation mask is converted to a multi-channel format required for the model's loss function.
    7.  **Save and Cleanup:** The four processed scans are stacked into a single 4-channel 3D image, which is then saved along with its processed mask into a single, compressed NumPy file (`.npz`). To conserve disk space, the original raw data folder for the patient is then automatically deleted.

### **4. Data Integrity Verification**

After preprocessing hundreds of files, it is possible for some to become corrupted due to interruptions or other issues. A corrupted file can halt the entire training process. Therefore, a verification step was implemented.

*   **Purpose of Verification:**
    This step ensures that every processed `.npz` file is valid and can be loaded without error. It acts as a quality control check before proceeding to the model training phase.

*   **The Verification Process:**
    The script iterates through every processed `.npz` file in the output directory. For each file, it performs a simple but effective test: it attempts to load the file using `np.load()`. If the file is corrupted, this operation will fail and raise an error (such as `zipfile.BadZipFile`). The script catches these specific errors, identifies the problematic file, and adds it to a list of corrupted files.

### **5. Data Splitting for Training and Validation**

To properly train a machine learning model and evaluate its true performance, we must split our dataset into at least two independent sets.

*   **The Importance of Splitting Data:**
    *   **Training Set:** This is the majority of the data (in our case, 80%) that the model learns from. The model adjusts its internal parameters by looking at the images and corresponding masks in this set.
    *   **Validation Set:** This is a smaller, unseen portion of the data (20%) that the model does *not* train on. It is used to periodically evaluate the model's performance during the training process. This helps us understand how well the model is **generalizing** to new data and prevents a common issue called **overfitting**, where the model simply memorizes the training data but fails on new examples.

*   **Implementation and Configuration:**
    Using the list of verified, non-corrupted file paths, we employed the `train_test_split` function from the `scikit-learn` library to perform the split.
    *   **Split Ratio:** We designated **20%** of the data for the validation set (`VALIDATION_SPLIT_SIZE = 0.20`).
    *   **Reproducibility:** We set a `RANDOM_STATE = 42`. This ensures that the random shuffle and split of the data is **identical** every time the script is run. This is critical for scientific reproducibility, as it guarantees that experiments can be repeated under the exact same conditions.

### **6. Final Output**

The successful completion of this milestone yields two crucial outputs:
1.  `train_files`: A Python list containing the file paths to the 80% of processed `.npz` files designated for training.
2.  `val_files`: A Python list containing the file paths to the 20% of processed `.npz` files designated for validation.

With the dataset now collected, preprocessed, verified, and strategically split, the project is fully prepared for the next major phase: Model Development and Training.


Processed Dataset link : https://www.kaggle.com/datasets/siddhantbapna/trainsb23/