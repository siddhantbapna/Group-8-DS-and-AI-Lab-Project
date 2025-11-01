## **Pre-M1: Project Ideation and Topic Finalization**

### **Activities**

* **All Members:** Conducted research on potential project topics.
* **Topic Suggestions:**

  * *Siddhant:* Cloth try-on, Music Generation
  * *Ravineel:* Brain Tumor Detection and Segmentation
  * *Hardik & Saurabh:* RAG-based tool for simplifying research paper understanding

**Final Topic Selected:** *3D Brain Tumor Segmentation using Deep Learning Architectures*

---

## **Milestone 1 (M1): Problem Understanding and Literature Review**

### **Work Division**

* **Hardik & Saurabh:** Identified gaps and opportunities in existing brain tumor segmentation studies.
* **Siddhant, Ajsal & Ravineel:** Conducted a review of existing solutions, baselines, and benchmarks.

### **Updates on Changes Made**

* Based on feedback, the team **added details on industry-existing solutions** and benchmark models.

---

## **Milestone 2 (M2): Dataset Research, EDA, and Preprocessing**

### **Work Division**

* **Ajsal:** Performed dataset research and exploratory data analysis (EDA).
* **Siddhant:** Led data preprocessing, including data normalization, augmentation, and modality alignment.

### **Updates on Changes Made**

* Dataset quality and structure were refined based on insights from EDA.
* Data preprocessing pipeline improved to ensure consistency across 3D volumes.
* Documentation structured for clarity.

---

## **Milestone 3 (M3): Model Design, Patch Extraction, and Documentation**

### **Work Division**

* **Hardik & Saurabh:**

  * Designed a **3D tumor-focused patch extraction system** tailored for 3D CNN architectures.
  * Addressed **class imbalance (<1% tumor voxels)** through intelligent patch sampling.
  * Conducted statistical validation and 3D visualization for patch quality.
  * Researched and designed the full pipeline for **3D Attention U-Net/V-Net** using 4-channel 3D patches.

* **Ajsal:**

  * Finalized **M3 Documentation (v2)**.
  * Researched **EMCAD**, a low-compute efficient segmentation model.
  * Visualized the **Attention U-Net** architecture diagram.

* **Ravineel:**

  * Built a 3D visualization of the brain dataset.
  * Designed pipelines for **3D U-Net, Attention U-Net, ResU-Net, and V-Net**.
  * Contributed to documentation with detailed data flow representations.

* **Siddhant:**

  * Implemented **Attention U-Net** (with and without K-Fold), **3D U-Net**, and **Attention U-Net** with different modality combinations.
  * Drafted **M3 Documentation (v1)**.

### **Updates on Changes Made**

Feedback: “Add model architecture diagram and other Recommendations ”

* **Siddhant** added detailed architecture visualizations.
* All feedbacks mentioned for M3 updates in documentation were integrated by **Ajsal**.


---

## **Milestone 4 (M4): Model Training, Comparison, and Repository Organization**

### **Work Division**

* **Ajsal & Siddhant:** Trained the **primary Attention U-Net model**.
* **Ravineel:** Conducted comparative analysis between different models (3D U-Net, ResU-Net, V-Net, Attention U-Net) and prepared **M4 v1 documentation**.
* **Saurabh:** Performed **code commenting** based on feedback for better readability.
* **Siddhant:** Organized the **GitHub repository** structure for consistency and clarity.

### **Updates on Changes Made**

* Integrated all feedback from M3, including documentation clarity.
* Repository cleaned, commented, and standardized for better maintainability.
* Model comparison results were compiled.

---