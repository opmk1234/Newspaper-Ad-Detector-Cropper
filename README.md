# Newspaper Advertisement Detector & Cropper

An end-to-end computer vision web application that automatically detects, refines, and extracts display advertisements from full-resolution newspaper pages. Built with a fine-tuned **YOLO11s** detector and a multi-stage OpenCV post-processing pipeline, deployed on **Streamlit Community Cloud**.

- **Live App:** [https://newspaper-ad-detector-cropper-8holoaild2o7tqzwgxkdel.streamlit.app/](https://newspaper-ad-detector-cropper-8holoaild2o7tqzwgxkdel.streamlit.app/)
- **Repository:** [https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper](https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper)

---

## Key Features

- **Single & Bulk Processing:** Upload individual pages (`PNG`, `JPG`, `JPEG`, `WEBP`) or batch archives (`.zip` or multi-file selection) with optional per-page folder grouping.
- **Full-Resolution Lossless Exports:** Runs fast proxy-scale geometry analysis while cropping from the untouched original image pixels to preserve print detail and fine print.
- **Automated Coordinate Manifest:** Bundles all extracted ad crops alongside a `summary.csv` file containing source filenames, confidence scores, bounding-box coordinates (`x1, y1, x2, y2`), and pixel dimensions.
- **Vision-Based Border Fitting:** Snaps raw detector boxes to outer border rules and tinted ad frames while merging horizontally split ad fragments.

---

## Model Performance

Trained on custom-annotated newspaper pages using **YOLO11s** (`imgsz=1024`). The operating confidence threshold (`conf=0.60`) was selected on the held-out validation split before evaluating once on the test split:

| Split             | Pages | Ad Instances | Confidence |  mAP@50   | mAP@50–95 | Precision |  Recall   | F1 Score  |
| :---------------- | :---: | :----------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Validation**    |  107  |      —       |    0.60    |     —     |     —     |     —     |     —     | **0.882** |
| **Held-Out Test** |  108  |     266      |    0.60    | **0.862** | **0.791** | **0.818** | **0.831** | **0.824** |

---

## Processing Pipeline

1. **Reflection Padding Removal:** Detects and strips symmetric border-reflection padding on augmented scans using a `480 px` grayscale proxy while leaving natural blank margins intact.
2. **Object Detection:** Runs `YOLO11s` inference (`imgsz=1024`, `iou=0.45`, agnostic NMS) and filters out ultra-narrow edge slivers.
3. **Nearest-Rule Boundary Snapping:** Estimates background newsprint color from outer page margins, combines Otsu and adaptive thresholding on a `1600 px` working scale, and expands each box to the nearest enclosing border rule without crossing column gutters.
4. **Fragment Merging & Export:** Merges horizontally overlapping detections sharing the same vertical band (`>80%` vertical overlap), renders a lightweight `1400 px` preview JPEG, and packages full-resolution PNG crops + `summary.csv` into a downloadable `.zip`.

---

## Project Structure

```text
├── app.py               # Streamlit UI, YOLO inference, and OpenCV post-processing pipeline
├── models/
│   └── best.pt          # Fine-tuned YOLO11s weights
├── requirements.txt     # Python dependencies (ultralytics, streamlit, opencv-python-headless)
├── packages.txt         # Linux system libraries (libgl1, libglib2.0-0) for OpenCV on Streamlit Cloud
├── LICENSE              # MIT License
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper.git
cd Newspaper-Ad-Detector-Cropper
pip install -r requirements.txt
streamlit run app.py
```

---

## Known Limitations

- **Script & Layout Domain Shift:** The detector was trained primarily on English/Latin-script broadsheets. Regional newspapers with dense non-Latin typography (such as Devanagari mastheads or shaded editorial callout boxes) may produce false positives unless fine-tuned on domain-specific examples.
