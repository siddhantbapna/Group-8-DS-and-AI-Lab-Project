## The Complete Guide to the Orca Project : Brain Tumor Segmentation Tool

This document provides a complete overview of the Brain Tumor Segmentation application, designed for both end-users (researchers, students, clinicians) and developers.

### **Part 1: User Guide (For Everyone)**

This section is written for users of all technical levels. It explains how to use the application and interpret its results, assuming no prior knowledge of medical imaging or AI.

#### **Section 1: Introduction**

**1.1 What is this tool?**

Welcome! This application is a web-based tool that uses a powerful Artificial Intelligence (AI) model to automatically find and map out different parts of a brain tumor from a set of MRI scans. It allows you to:

*   **Upload** standard clinical MRI scans.
*   **Visualize** the scans in both 2D (slice-by-slice) and interactive 3D.
*   **Run an AI model** to predict the location and extent of the tumor.
*   **Analyze** the size of the predicted tumor regions.
*   **Compare** the AI's prediction to a manual segmentation from an expert (if you have one).

**1.2 Who is this for?**

This tool is designed for medical researchers, students, and radiologists who wish to visualize, analyze, and get an automated, second-opinion segmentation of brain tumor data.

**1.3 ★ Important Disclaimer ★**

This application is a **research tool only**. It is **NOT** a certified medical device and should **NEVER** be used for clinical diagnosis, patient treatment decisions, or any primary clinical purpose. The AI's predictions are not a substitute for the judgment of a qualified medical professional.

---

#### **Section 2: Before You Begin - Understanding the Data**

To get the most out of this tool, it's helpful to understand the data you're working with.

**2.1 What are the different MRI scans?**

The AI needs four different types of MRI scans to work accurately. Each scan is like a different "filter" that highlights specific features of the brain tissue.

*   **T1ce (Post-contrast T1-weighted):** This is a key scan. The patient is given a special dye (contrast agent) before the scan. This dye makes the most active, aggressive parts of the tumor "light up" brightly. This scan is crucial for finding the **Enhancing Tumor**.
*   **T1 (Pre-contrast T1-weighted):** This is a basic anatomical scan that provides a clear map of the brain's structure, showing sharp boundaries between different tissue types.
*   **FLAIR (Fluid-Attenuated Inversion Recovery):** This scan is an expert at finding swelling. It cleverly cancels out the signal from normal brain fluid, which makes the fluid buildup (called *edema*) around the tumor stand out very clearly.
*   **T2 (T2-weighted):** This scan is also very sensitive to water content and helps to highlight the edema as well as other tumor structures.

**2.2 What do the tumor labels mean?**

The AI identifies three distinct, nested regions of the tumor, much like the layers of an onion.

*   <span style="color:red;">**Whole Tumor (WT)**</span>: This is the largest region, encompassing the entire abnormality. It includes the solid tumor mass plus all the surrounding swelling (edema) visible on the FLAIR scan.
*   <span style="color:green;">**Tumor Core (TC)**</span>: This is the solid part of the tumor, located *inside* the Whole Tumor. It excludes the surrounding edema.
*   <span style="color:blue;">**Enhancing Tumor (ET)**</span>: This is the most active part of the tumor, identified by the "lit-up" areas on the T1ce scan. It is located *inside* the Tumor Core.



---

#### **Section 3: A Step-by-Step Walkthrough**

Using the application involves a simple 4-step workflow.

**Step 1: Load Your Data**

You have two options to start:

*   **Option A: Use Examples (Recommended for first-time users)**
    1.  Click the **"Use Examples"** tab.
    2.  Click on one of the pre-loaded case buttons (e.g., "BraTS-GLI-00000-000").
    3.  The application will automatically load all the necessary files for that case and begin processing them.

*   **Option B: Upload Your Own Files**
    1.  Make sure you are on the **"Upload Files"** tab.
    2.  For each of the four required modalities (T1ce, T1, FLAIR, T2), click "Choose File" and select the correct `.nii` or `.nii.gz` file from your computer.
    3.  **(Optional)** If you have a segmentation file created by a human expert (a "Ground Truth"), you can upload it under the "Ground Truth" input. This will allow you to quantitatively compare the AI's performance.

**Step 2: Process the Files**

1.  Once all four required scans are uploaded, the **"Process Files"** button will become clickable.
2.  Click it. The application will now perform crucial preprocessing steps: it aligns all the scans, standardizes their brightness, and crops them to the brain region. This ensures the AI receives clean, consistent data.
3.  Once complete, the viewer section will appear, and the "Run Prediction" button will become active.

**Step 3: Run the Prediction**

1.  Click the **"Run Prediction"** button.
2.  The AI model will now analyze the processed scans. This may take a few moments.
3.  When finished, the results will automatically load in the 2D and 3D viewers, and the "Analysis" tab will become available if you provided a ground truth file.

**Step 4: Explore and Analyze the Results**

Now you can investigate the outcome using the three viewer tabs.

*   **The 2D Slices Tab:**
    This view shows you the brain as a series of cross-sectional images.
    *   **Use the Slider:** Drag the slider at the bottom to move through the different slices of the brain.
    *   **Sub-Tabs (Modalities, Prediction, Ground Truth):** Switch between these tabs to see the original input scans, the AI's prediction overlaid on the T1ce scan, and the expert's segmentation (if provided).

*   **The 3D Volume Tab:**
    This view provides a powerful, interactive 3D reconstruction of the brain and tumor.
    *   **Interact with the Model:**
        *   **Rotate:** Left-click and drag.
        *   **Zoom:** Use the mouse wheel.
        *   **Pan (Move):** Right-click and drag.
    *   **Use the Control Panels:** On the side(s) of the viewer, you can use checkboxes to toggle the visibility of the brain surface and each of the tumor components (WT, TC, ET).
    *   **Side-by-Side Comparison:** If you loaded a ground truth file, you will see a split-screen view, allowing for a direct 3D comparison between the expert's segmentation and the AI's prediction.

*   **The Analysis Tab:**
    This view provides the quantitative numbers behind the visuals.
    *   **Volume (in voxels):** A "voxel" is a 3D pixel. This number tells you the size of each tumor component. This is useful for tracking tumor size over time.
    *   **Dice Score:** This is a standard accuracy metric in medical imaging. It measures the overlap between the AI's prediction and the ground truth, on a scale from 0 (no overlap at all) to 1 (a perfect match). A higher Dice Score means a more accurate prediction.

---

#### **Section 4: Troubleshooting & FAQ**

*   **Q: The "Process Files" button is gray and I can't click it.**
    *   **A:** This means you have not yet uploaded all four required MRI scans (T1ce, T1, FLAIR, and T2). Make sure a file has been selected for each of these inputs.

*   **Q: What do the different colors mean in the prediction?**
    *   **A:** Red = Whole Tumor, Green = Tumor Core, Blue = Enhancing Tumor. If you upload a ground truth, the Enhancing Tumor is shown in Yellow for easier comparison.

*   **Q: What does the "Reset" button in the header do?**
    *   **A:** This will completely clear all uploaded data and predictions from your session, allowing you to start fresh with a new case.

*   **Q: The 3D view is slow or choppy.**
    *   **A:** 3D rendering can be computationally intensive. Performance will depend on your computer's hardware. The experience is generally better on modern computers with dedicated graphics cards.

---

### **Part 2: Developer's Guide**

This section is for developers who wish to understand, modify, or extend the application.

#### **Section 1: Architecture Overview**

*   **Backend:** A Python server using the **Flask** web framework.
    *   **AI/ML:** Utilizes **PyTorch** and the **MONAI** (Medical Open Network for AI) framework for the deep learning model (`AttentionUnet`) and data preprocessing pipelines.
    *   **Image Processing:** Uses **Nibabel** to handle NIfTI file I/O and **scikit-image** for post-processing (removing small lesions) and 3D mesh generation (`marching_cubes`).
*   **Frontend:** A single-page application built with vanilla **HTML, CSS, and JavaScript**. No external frameworks are used.
    *   **3D Rendering:** Uses the **three.js** library to create and display the interactive 3D visualizations.
*   **Communication:** The frontend communicates with the backend via a RESTful API, sending data and receiving results as JSON objects or files. Each user is assigned a unique session ID (`UUID`) to keep their data isolated on the server.

#### **Section 2: Backend API Endpoints**

The following endpoints are defined in `app.py`.

| Method | Endpoint             | Description                                                                                             | Request Body (JSON)                                    | Success Response (JSON)                                                                                                                                                                             |
| :----- | :------------------- | :------------------------------------------------------------------------------------------------------ | :----------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/`                  | Serves the main `index.html` file.                                                                      | N/A                                                    | `200 OK`: HTML content.                                                                                                                                                                             |
| `GET`  | `/list_examples`     | Returns a list of available example case IDs found in the `/examples` directory.                        | N/A                                                    | `200 OK`: `["id1", "id2", ...]`                                                                                                                                                                      |
| `POST` | `/reset_session`     | Deletes the session folder for a given `session_id`.                                                    | `{"session_id": "..."}`                                | `200 OK`: `{"message": "Session reset."}`                                                                                                                                                           |
| `POST` | `/delete_file`       | Deletes a specific modality file from a session folder.                                                 | `{"session_id": "...", "modality": "t1c"}`             | `200 OK`: `{"message": "File t1c deleted."}`                                                                                                                                                        |
| `POST` | `/upload`            | Uploads a `.nii`/`.nii.gz` file. This is a `multipart/form-data` request.                                 | Form Data: `session_id`, `modality`, `file`            | `200 OK`: `{"message": "t1c uploaded", "loaded_modalities": ["t1c", ... ]}`                                                                                                                          |
| `POST` | `/load_example`      | Copies files from an example case into a new session folder.                                            | `{"session_id": "...", "id": "example_id"}`            | `200 OK`: `{"message": "...", "loaded_modalities": [...]}`                                                                                                                                          |
| `POST` | `/process`           | Runs the MONAI preprocessing pipeline on all uploaded files and saves the result as `processed.npz`.    | `{"session_id": "..."}`                                | `200 OK`: `{"message": "Files processed", "num_slices": 128, "slice_index": 64}`                                                                                                                    |
| `GET`  | `/get_all_slices`    | Retrieves all 2D slices for a given modality as base64-encoded PNGs.                                    | Query Params: `modality`, `session_id`                 | `200 OK`: `{"t1c": ["data:image/png;base64,..."]}` or `{"wt": [...], "tc": [...], "et": [...]}`                                                                                                     |
| `POST` | `/predict`           | Runs the model inference, performs post-processing, saves the prediction, and generates analysis data.  | `{"session_id": "..."}`                                | `200 OK`: `{"message": "Prediction successful", "analysis": {...}}` (See previous documentation for analysis structure).                                                                              |
| `GET`  | `/get_mesh_json`     | Serves the `meshes.json` file containing 3D model data for the frontend.                                | Query Params: `session_id`                             | `200 OK`: JSON file content.                                                                                                                                                                        |

#### **Section 3: Frontend Code Structure**

The frontend logic is contained within the `<script>` tag in `index.html`.

*   **State Management:**
    *   `sessionId`: A UUID generated on the client's first visit or retrieved from `localStorage` to maintain the session across reloads.
    *   `sliceCache`: A global object that stores the base64-encoded slices for each modality once fetched, preventing redundant API calls.
    *   `viewerState`: A small object that tracks the number of slices and the set of currently uploaded modalities to control UI element states (e.g., enabling/disabling buttons).
    *   `threeD`: An object containing the entire state for the `three.js` viewer, including the scene, camera, renderer, controls, and mesh groups.

*   **Workflow Functions:**
    *   The primary workflow is managed by a chain of `async` functions: `handleFileUpload` -> `processFiles` -> `runPrediction`.
    *   Each function updates the UI with the current status (`loadingStatus`) and calls the next function upon successful completion.

*   **Rendering:**
    *   **2D:** The `drawImageOnCanvas` function decodes base64 strings and draws them onto the appropriate `<canvas>` element. `updateAllSlices` is the central function that redraws all visible canvases when the slider's value changes.
    *   **3D:** The `init3DViewer` function sets up the entire `three.js` scene once. `fetchAndRender3D` is called to fetch the `meshes.json` file, clean the scene with `cleanup3DScene`, create new meshes with `createMesh`, and add them to the scene. The animation loop handles rendering and orbit controls updates. A key feature is the use of `renderer.setScissorTest(true)` to render two separate viewports (for GT and Prediction) within the same canvas.

#### **Section 4: Setup and Deployment**

1.  **Prerequisites:**
    *   Python 3.8+
    *   A pre-trained model file named `best_model.pth` placed in a `/model` directory.
    *   Example cases placed in subdirectories within an `/examples` directory.

2.  **Installation:**
    *   Clone the repository.
    *   Create a Python virtual environment: `python -m venv venv` and activate it.
    *   Install the required packages. You would create a `requirements.txt` file from the imports:
        ```
        flask
        torch
        monai
        nibabel
        numpy
        Pillow
        scikit-image
        werkzeug
        ```
    *   Run `pip install -r requirements.txt`. Note that `torch` might require a specific installation command depending on your system's CUDA capabilities. The provided code is set to run on the CPU (`DEVICE = torch.device("cpu")`), so a standard PyTorch installation will suffice.

3.  **Running the Application:**
    *   Execute the main Python script from your terminal: `python app.py`
    *   The application will be accessible at `http://127.0.0.1:7860` by default. The `debug=True` flag enables hot-reloading for development.