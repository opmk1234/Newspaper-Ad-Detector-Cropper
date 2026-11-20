---
title: Newspaper Ad Detector & Cropper
emoji: 📰
colorFrom: green
colorTo: slate
sdk: streamlit
app_file: app.py
pinned: false
---

# Newspaper Ad Detector & Cropper

Detects and crops display advertisements from full-resolution newspaper pages using a fine-tuned **YOLO11s** model paired with full-resolution border/background boundary snapping.

## Model & Held-Out Test Results

- **Architecture:** YOLO11s (`9.41M` params, `imgsz=1024`, single class: `ad`)
- **Operating Threshold:** `conf = 0.60` (selected on the 107-page validation split at F1 `0.882`, then frozen prior to test evaluation)
- **Held-Out Test Split:** 108 unseen pages (69 ad-bearing pages, 39 ad-free pages, 266 ground-truth ads)

| Metric    | Validation (`conf=0.60`) | Held-Out Test (`conf=0.60`) |
| --------- | ------------------------ | --------------------------- |
| Precision | 0.902                    | **0.818**                   |
| Recall    | 0.864                    | **0.831**                   |
| F1 Score  | 0.882                    | **0.824**                   |
| mAP@50    | 0.929                    | **0.862**                   |
| mAP@50–95 | 0.826                    | **0.791**                   |

## Post-Processing Pipeline

1. **Roboflow Mirror-Padding Cleanup:** Detects and trims vertical reflection seams on augmented dataset margins so mirrored duplicates are excluded.
2. **Boundary & Rule Snapping:** Expands raw YOLO boxes proportionally and scans outward against newsprint background color and 1 px rule lines to enclose full ad frames and detached footers without crossing column gutters.
3. **Fragment Merging & Sliver Filtering:** Merges horizontally split ad boxes and drops narrow edge-margin ribbons.

## Run Locally

**Using `uv`:**

```bash
uv sync
uv run streamlit run app.py
```
