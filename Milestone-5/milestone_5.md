## **Evaluation Report: 3D Brain Tumor Segmentation using an Attention U-Net Architecture**

### **1. Overview & Objective**

This report details the evaluation of a deep learning pipeline designed for 3D segmentation of brain tumors from multi-modal MRI scans. The implemented solution uses an **Attention U-Net**, a convolutional neural network architecture that leverages attention mechanisms to enhance segmentation accuracy by focusing on the most relevant image features.

The primary objective of this evaluation is to analyze the performance of the trained model, identify its strengths and weaknesses, and outline a clear path for future improvements. The end-to-end process, from data acquisition and preprocessing to model training and inference, is built upon the PyTorch and MONAI frameworks, ensuring the use of industry-standard tools for medical imaging analysis.

### **2. Evaluation Setup**

To ensure the reproducibility and integrity of the results, a well-defined evaluation protocol was established.

*   **Dataset:** The evaluation was conducted on the publicly available BraTS (Brain Tumor Segmentation) dataset after preprocessing and data integrity checks.
    The dataset was partitioned using Scikit-learn's KFold with N_SPLITS=5, producing five 80/20 train-validation splits. For this experiment, a single fold (FOLD_TO_RUN = 0) was used, yielding 1000 files for training and 251 for validation. The split was shuffled with a random state of 42 for reproducibility.

*   **Validation Strategy:** The results presented in this report are from the complete training and validation cycle, where 80% of the data was used for training and the remaining 20% for validation.

*   **Preprocessing Pipeline:** All data, for both training and validation, was subjected to a standardized preprocessing pipeline to ensure consistency. The key steps included:
    The NIfTI files were preprocessed into .npz arrays using a MONAI pipeline. The following sequential transforms were applied:
    1. **LoadImaged**: Loaded all 5 NIfTI files per patient.
    2. **Label Remapping**: Before preprocessing, the segmentation label '3' was remapped to '4' to align with the BraTS labeling convention. (1: Necrotic Core, 2: Peritumoral Edema, 4: Enhancing Tumor).
    3. **ConvertToMultiChannelBasedOnBratsClassesd**: Transformed the single-channel segmentation mask (labels 1, 2, 4) into a three-channel binary mask for the three tumor subregions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
    4. **Spacingd**: Resampled all volumes to a uniform voxel spacing of (1.0, 1.0, 1.0) mm.
    5. **ScaleIntensityRanged**: Normalized the intensity of the MRI modalities from a range of [0, 1400] to [0, 1].
    6. **CropForegroundd**: Cropped the volumes to the non-zero foreground region, determined from the T1c image, with a 10-voxel margin.
    7. **Resized**: Resized the cropped volumes to a fixed shape of (128, 128, 128).
    8. **EnsureTyped**: Converted tensors to torch.float16 to reduce storage requirements.

### **3. Performance Metrics**

The model's performance was assessed using the following metrics:

*   **Primary Metric: Dice Similarity Coefficient (DSC):** The Dice score was selected as the primary metric for evaluating segmentation accuracy. It measures the spatial overlap between the predicted segmentation and the ground truth, making it highly suitable for tasks where class imbalance (e.g., small tumors in large brain volumes) is a significant factor. The DSC was calculated independently for each of the three tumor sub-regions:
    *   **Whole Tumor (WT)**
    *   **Tumor Core (TC)**
    *   **Enhancing Tumor (ET)**
    The average of these three scores was used to track overall model performance and for early stopping decisions.
*   **Loss Function:** The model was optimized using a **DiceBCE Loss**, a composite function that combines the stability of Binary Cross-Entropy with the direct optimization of the Dice score.

### 4. Quantitative Results

#### 4.1 Dice score statistics

| Label             |   Mean  | Std (population) | Median  |   Min   |   Max   |
|-------------------|--------:|-----------------:|--------:|--------:|--------:|
| TC (Tumor Core)   | 0.5947  |          0.3489  | 0.7515  | 0.0000  | 0.9537  |
| WT (Whole Tumor)  | 0.6892  |          0.3140  | 0.8393  | 0.0000  | 0.9544  |
| ET (Enhancing Tumor) | 0.5079 |         0.3267  | 0.6413  | 0.0000  | 1.0000  |
| Mean Dice per patient | 0.5973 |        0.3057  | 0.7183  | 0.0000  | 0.9243  |

**Note:** Minimum = 0.0 indicates some patient-label pairs are full misses (predicted 0 when GT > 0).

#### 4.2 Ground-truth vs Predicted volume correlation (Pearson)

Computed on per-patient totals or per-label volumes:

| Metric       | Pearson correlation (GT vs Pred) |
| ------------ | -------------------------------: |
| `TC_corr`    |                     **0.907687** |
| `WT_corr`    |                     **0.878649** |
| `ET_corr`    |                     **0.803350** |
| `total_corr` |                     **0.892378** |

#### 4.3 Linear regression: predicted = slope * GT + intercept (GT → Pred)

(Computed with least-squares fit across the 150 patients.)

| Label |        Slope |        Intercept |           R² |
| ----- | -----------: | ---------------: | -----------: |
| TC    | **0.931400** |  **-812.336265** | **0.823896** |
| WT    | **0.867768** | **-1883.732017** | **0.772024** |
| ET    | **0.667674** |  **-522.340795** | **0.645371** |
| Total | **0.889222** | **-5724.065155** | **0.796338** |

**Interpretation:** slopes < 1 indicate a tendency to *under-predict volumes* across labels, especially for ET (slope ≈ 0.6677). High R² for TC & total show reasonable linear relationship but still systematic bias.

#### 4.4 Missed detections (Predicted volume = 0 while GT > 0)

| Label | Missed count (patients) |
| ----- | ----------------------: |
| TC    |                  **12** |
| WT    |                   **3** |
| ET    |                  **22** |

**Note:** ET has the most full misses - consistent with low ET Dice and low slope/R².

#### 4.5 Volume error (relative absolute error = |pred_total - gt_total| / gt_total)

Key statistics (per-patient relative absolute error):

* **Mean relative absolute error:** **0.296379**
* **Median:** **0.164180**
* **75th percentile:** **0.461332**
* **Max:** **1.000000**

**Interpretation:** while many patients have low relative error (median ≈ 16%), a sizable tail exists - 25% of patients have relative errors ≥ ~46%, and the worst case is a 100% relative error.


### 5. Test Set Evaluation on Unseen Data

To further assess the model’s generalization capability, it was evaluated on three samples.

Before presenting the results, here are the common terms used:
1. Ground truth visuals show the MRI modalities and the corresponding segmentation of the brain tumor. 
2. The raw ground truth shows the Non-Enhancing Tumor (NET), Edema, and Enhancing Tumor (ET).
3. The derived masks visualize the three standard BraTS sub-regions: Whole Tumor (WT = Edema + NET + ET), Tumor Core (TC = NET + ET), and Enhancing Tumor (ET).
4. Prediction plots show the segmentation outputs produced by the trained 3D Attention U-Net model.
5. The raw model output sometimes contains small disconnected regions, known as lesions, which represent prediction noise. Therefore, we present two versions: one with these lesions and one after post-processing to remove them. After post-processing, the output mask is smoother than the raw prediction, with small spurious lesions removed.

#### 5.1 Test Sample "BraTS-GLI-02506-101"


##### Volumetric and Accuracy Analysis
-------------------------------------------------------------------------------------
Tumor Component           | Ground Truth Volume  | Predicted Volume     | Dice Score (Accuracy)
|-----------------|--------------------:|-----------------:|----------------------:|
Tumor Core (TC)           | 26805.00             | 27970.00             | 0.9317              
Whole Tumor (WT)          | 49522.00             | 51834.00             | 0.9465              
Enhancing Tumor (ET)      | 22927.00             | 22288.00             | 0.8947              
-------------------------------------------------------------------------------------
Slice index where ET mask is biggest: 78
Number of ET voxels in that slice: 888

<details open>
  <summary>GROUND TRUTH PLOT</summary>
  <img src="./images/testPerformance/BraTS-GLI-02506-101/GroundTruth.png" width="500">
</details>

<details open>
  <summary>Pred PLOT (With Lesions)</summary>
  <img src="./images/testPerformance/BraTS-GLI-02506-101/PredWithLegions.png" width="500">
</details>

<details open>
  <summary>Pred PLOT (Without Lesions)</summary>
  <img src="./images/testPerformance/BraTS-GLI-02506-101/PredWithoutLegions.png" width="500">
</details>

#### 5.2 Test Sample "BraTS-GLI-02405-100


##### Volumetric and Accuracy Analysis
-------------------------------------------------------------------------------------
Tumor Component           | Ground Truth Volume  | Predicted Volume     | Dice Score (Accuracy)
|-----------------|--------------------:|-----------------:|----------------------:|
Tumor Core (TC)           | 8419.00              | 8817.00              | 0.9117              
Whole Tumor (WT)          | 12529.00             | 14535.00             | 0.8841              
Enhancing Tumor (ET)      | 6093.00              | 6636.00              | 0.8060              
-------------------------------------------------------------------------------------
Slice index where ET mask is biggest: 54
Number of ET voxels in that slice: 338


<details open>
  <summary>GROUND TRUTH PLOT</summary>
  <img src="./images/testPerformance/BraTS-GLI-02405-100/GroundTruth.png" width="500">
</details>

<details open>
  <summary>Pred PLOT (With Lesions)</summary>
  <img src="./images/testPerformance/BraTS-GLI-02405-100/PredWithLegions.png" width="500">
</details>

<details open>
  <summary>Pred PLOT (Without Lesions)</summary>
  <img src="./images/testPerformance/BraTS-GLI-02405-100/PredWithoutLegions.png" width="500">
</details>

#### 5.3 Test Sample "BraTS-GLI-02426-100"


##### Volumetric and Accuracy Analysis
-------------------------------------------------------------------------------------
Tumor Component           | Ground Truth Volume  | Predicted Volume     | Dice Score (Accuracy)
|-----------------|--------------------:|-----------------:|----------------------:|
Tumor Core (TC)           | 7011.00              | 6168.00              | 0.5899              
Whole Tumor (WT)          | 37846.00             | 26524.00             | 0.7535              
Enhancing Tumor (ET)      | 5151.00              | 2907.00              | 0.5905              
-------------------------------------------------------------------------------------
Slice index where ET mask is biggest: 67
Number of ET voxels in that slice: 353


<details open>
  <summary>GROUND TRUTH PLOT</summary>
  <img src="./images/testPerformance/BraTS-GLI-02426-100/GroundTruth.png" width="500">
</details>

<details open>
  <summary>Pred PLOT (With Lesions)</summary>
  <img src="./images/testPerformance/BraTS-GLI-02426-100/PredWithLegions.png" width="500">
</details>

<details open>
  <summary>Pred PLOT (Without Lesions)</summary>
  <img src="./images/testPerformance/BraTS-GLI-02426-100/PredWithoutLegions.png" width="500">
</details>

### 📊 Performance Comparison with BraTS 2023 Leaderboard (Validation Set)

| Metric | Our Model (Validation) | BraTS 2023 Average | Top Method (nnUNet) | Gap  |
|--------|------------------------|--------------------|---------------------|--------|
| **Whole Tumor (WT) Dice** | 0.8725 | 0.860 | 0.929 | **+1.5%** |
| **Tumor Core (TC) Dice** | 0.8033 | 0.810 | 0.881 | **-0.8%** |
| **Enhancing Tumor (ET) Dice** | 0.6679 | 0.780 | 0.859 | **-14.4%**  |
| **Overall Mean Dice** | 0.7812 | 0.817 | 0.890 | **-4.4%** |

### 6. Error Analysis

#### 6.1 Trends by label

* **Best performing label (aggregate):** WT (mean Dice = **0.689214**).
* **Intermediate:** TC (mean Dice = **0.594692**).
* **Worst / highest variance:** ET (mean Dice = **0.507910**, std ≈ 0.3267) and the most full misses (22 patients).

#### 6.2 Systematic errors & patterns

* **Under-prediction bias:** slope < 1 across labels (TC 0.9314, WT 0.8678, ET 0.6677, total 0.8892) → model tends to predict smaller volumes than GT.
* **High GT–Pred correlations (0.80–0.91):** model preserves relative ordering (bigger tumors produce bigger predictions) but not scale.
* **ET is inconsistent:** lowest slope and R² among labels; more misses and higher variance - likely due to fewer/smaller ET regions in training and/or more subtle contrast.
* **Outlier tail:** relative absolute error max = 1.0 (100%) for at least one patient, and 75th percentile ≈ 0.4613 indicates substantial errors in the upper quartile.

#### 6.3 Anomalies observed

* **Full misses** (pred=0 while GT>0) - especially for ET: 22 patients.
* **Some patients with perfect/near-perfect Dice** indicate “easy” cases - useful for reverse analysis (what makes them easy?).
* **Cases where pred_total >> gt_total** (over-segmentation) do exist - inspect individually in the CSV.

#### 6.4 Root-cause hypotheses

* **Class imbalance** (ET small/rare) → undertraining and more false negatives for ET.
* **Loss function bias** (e.g., Dice loss focusing on large regions) → model rewards WT more than small ET areas.
* **Aggressive post-processing thresholds** removing small islands → converts small ET predictions to zero.
* **Architectural or receptive-field limits** → inability to capture small, fine-detail ET regions.
* **Training / domain mismatch** (contrast, scanners) → calibration & scaling issues (negative intercepts suggest systematic underestimation).

#### 7. Limitations

The current evaluation, while thorough, has several limitations:

*   **Computational Constraints:** The model's network capacity (i.e., the number of channels in convolutional layers) was intentionally limited to operate within the memory constraints of the evaluation environment. A model with higher capacity, trained on more powerful hardware, could potentially yield superior results.
*   **Dataset Generalization:** The model was trained and evaluated exclusively on the BraTS dataset. Its performance on clinical data from different scanners, institutions, or with different acquisition parameters is yet to be determined.

### 8. Proposed Improvements & Next Steps

Based on the current evaluation, the following enhancements are proposed:

**Model Enhancement - Attention ResU-Net Ensemble**

At present, we use an Attention U-Net model for prediction, which performs well in 3D segmentation and effectively delineates tumor regions by focusing on salient features.

We plan to combine it with the ResU-Net model (currently under testing – [ResUNet](./resUnet.md)) to form an ensemble architecture, tentatively referred to as Attention ResU-Net.

This hybrid model is expected to further improve segmentation accuracy by:
- Capturing rich semantic information through residual connections (ResU-Net component).
- Preserving fine-grained tumor details through attention gates (Attention U-Net component).

**Deployment & Visualization Pipeline**
The next step in this project is model deployment, where we aim to develop an user interface (UI) that allows clinicians or researchers to:

- Upload raw patient MRI data (BRATS compatible format).
- Automatically preprocess and feed it into the trained segmentation model.
- Visualize the resulting tumor masks in an interactive 3D and 2D viewer for detailed analysis.

This stage will make the model more accessible for practical use in clinical and research settings, enabling seamless end-to-end brain tumor segmentation and visualization.
