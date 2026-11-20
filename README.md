# Automated Newspaper Advertisement Detection & Cropping

An end-to-end computer vision system that automatically detects, refines, and extracts display advertisements from high-resolution scanned newspaper pages. Built with a fine-tuned **YOLO11s** object detector and a multi-stage **OpenCV** boundary-snapping pipeline, deployed as an interactive web application on **Streamlit Community Cloud**.

- **Live Demo:** https://newspaper-ad-detector-cropper-8holoaild2o7tqzwgxkdel.streamlit.app/
- **GitHub Repository:** https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper

---

## 1. Key Capabilities

- **High-Resolution Upload Support:** Accepts individual newspaper pages (`JPG`, `JPEG`, `PNG`, `WEBP`) as well as multi-file batches and `.zip` archives.
- **Multi-Ad Detection & Isolation:** Locates multiple advertisements of varying sizes, aspect ratios, and page positions while filtering out editorial articles, headlines, mastheads, and statistical tables.
- **Complete Boundary Preservation:** Combines deep-learning localization with pixel-level border and background-color snapping so headlines, product imagery, brand logos, QR codes, and fine-print footers are captured without clipping.
- **Lossless Full-Resolution Export:** Performs geometry refinement on fast proxy scales while cropping from untouched original pixels, exporting individual `PNG` crops alongside a `summary.csv` coordinate manifest.

---

## 2. Model Selection & Architecture Rationale

We evaluated two primary paradigms for newspaper layout extraction:

1. **Promptable / Zero-Shot Segmentation (e.g., SAM 2):** Produces sharp masks on visually distinct photo blocks, but lacks semantic awareness of what constitutes a commercial advertisement versus an editorial photo, multi-column article box, or sports table—often splitting a single multi-element ad (headline + photo + footer) into disconnected segments.
2. **Fine-Tuned Single-Stage Object Detector + Vision Post-Processing (`YOLO11s` + OpenCV — Selected):**
   - **Semantic Discrimination:** Fine-tuning `YOLO11s` directly on annotated newspaper pages teaches the network to group multi-element advertisements (banner text, visuals, logos, borders) into a single unified bounding box while ignoring editorial content.
   - **CPU Deployment Efficiency:** `YOLO11s` delivers strong localization accuracy while running in sub-second latency on Streamlit Community Cloud's CPU environment without requiring a dedicated GPU server.
   - **Hybrid Boundary Refinement:** Pairing `YOLO11s` with deterministic OpenCV border snapping eliminates minor box clipping (~2–4% edge margin loss) without sacrificing inference speed.

---

## 3. Dataset Preparation, Annotation & Fine-Tuning

- **Dataset & Splitting:** Curated an annotated corpus of full-resolution newspaper pages split into leak-free **Training**, **Validation (107 pages)**, and **Held-Out Test (108 pages, 266 advertisement instances)** sets.
- **Annotation Guidelines:** Every commercial and house display advertisement was labeled with tight bounding boxes enclosing all internal components (top headlines, hero imagery, brand marks, and bottom contact/legal text) while excluding surrounding news columns and section headers.
- **Data Augmentation & Preprocessing:** Applied standard spatial and photometric augmentations during training. At inference time, a custom reflection-seam detector strips symmetric mirror padding when evaluating augmented datasets while leaving natural newsprint margins untouched.
- **Training & Hyperparameter Selection:**
  - **Model:** `YOLO11s` initialized from pretrained weights (`imgsz=1024`, `iou=0.45`, class-agnostic NMS).
  - **Environment:** Trained with Ultralytics and PyTorch on an NVIDIA Tesla T4 GPU.
  - **Threshold Freezing:** Swept confidence thresholds on the **107-page validation split** and froze `conf=0.60` (achieving **0.882 F1**) before running a single final evaluation on the held-out test split.

---

## 4. Quantitative Evaluation

| Split             | Pages | Ad Instances | Confidence |  mAP@50   | mAP@50–95 | Precision |  Recall   | F1 Score  |
| :---------------- | :---: | :----------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Validation**    |  107  |      —       |    0.60    |     —     |     —     |     —     |     —     | **0.882** |
| **Held-Out Test** |  108  |     266      |    0.60    | **0.862** | **0.791** | **0.818** | **0.831** | **0.824** |

---

## 5. End-to-End Processing Pipeline

1. **Mirror-Padding Check:** Scans outer left and right margins on a `480 px` grayscale proxy to strip symmetric reflection padding on augmented images without altering standard blank page margins.
2. **YOLO11s Inference:** Predicts candidate advertisement bounding boxes at `1024 px` resolution (`conf=0.60`) and filters out degenerate edge slivers.
3. **Nearest-Rule Boundary Snapping:** Estimates background newsprint color from page margins, combines Otsu and adaptive thresholding on a `1600 px` working scale, and expands each box outward to the nearest enclosing border rule or tinted ad frame without crossing column gutters.
4. **Horizontal Fragment Merging & Cropping:** Merges overlapping detections sharing the same vertical band (`>80%` vertical overlap), maps coordinates back to the original image resolution, and packages lossless `PNG` crops + `summary.csv` into a `.zip` archive.

---

## 6. Project Structure

    ├── app.py               # Streamlit web interface, YOLO11s detector, and OpenCV post-processing
    ├── models/
    │   └── best.pt          # Fine-tuned YOLO11s model weights
    ├── samples/             # Sample newspaper page previews and extracted ad crops
    ├── requirements.txt     # Python dependencies (ultralytics, streamlit, opencv-python-headless)
    ├── packages.txt         # Linux system dependencies (libgl1, libglib2.0-0) for Streamlit Cloud
    ├── LICENSE              # MIT License
    └── README.md            # Project documentation and setup guide

---

## 7. Local Setup & Execution

    # 1. Clone the repository
    git clone https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper.git
    cd Newspaper-Ad-Detector-Cropper

    # 2. Create and activate a virtual environment (recommended)
    python -m venv .venv
    source .venv/bin/activate        # On Windows: .venv\Scripts\activate

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Launch the Streamlit application
    streamlit run app.py

---

## 8. Sample Outputs & Known Limitations

- **Sample Outputs:** See the `samples/` directory (and the live web app) for side-by-side comparisons of full newspaper pages with detected bounding boxes alongside individually cropped advertisement PNGs and `summary.csv` coordinate logs.
- **Known Limitations:** The current weights are optimized for English/Latin-script broadsheet layouts. Regional newspapers with dense non-Latin scripts (such as Devanagari mastheads or shaded editorial callout boxes) require domain-specific fine-tuning examples to suppress editorial false positives.
