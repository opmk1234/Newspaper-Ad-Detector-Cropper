# Newspaper Advertisement Detector & Cropper

An end-to-end computer vision web application built with **Ultralytics YOLO11s**, **OpenCV**, and **Streamlit** to detect, refine, and extract advertisements from high-resolution newspaper broadsheets (English and Hindi layouts).

**Live App:** [newspaper-ad-detector-cropper.streamlit.app](https://newspaper-ad-detector-cropper-8holoaild2o7tqzwgxkdel.streamlit.app/)

---

## Key Features

- **Bilingual Broadsheet Detection:** Fine-tuned `YOLO11s` (`imgsz=1024`) trained on a curated dataset combining _Times of India_ broadsheets and mirror-padding-cleaned newspaper pages.
- **Precision OpenCV Border Snapping:** Automatically aligns predicted bounding boxes onto physical column rules, strips outer blank newsprint margins, and captures attached white footer/disclaimer bars without bleeding into adjacent editorial text.
- **Compact Solid-Ad Recovery:** Recovers smooth, uniform-background sidebar product banners that often fall below standard object-detection thresholds while filtering out textured news photographs and editorial callout boxes.
- **Single & Bulk Processing:** Upload individual pages or batch `.zip` archives, inspect annotated previews side-by-side, and export cropped PNGs alongside a structured `summary.csv` coordinate manifest.

---

## Model Benchmark (_Times of India_ Test Split)

Evaluated on **24 held-out _Times of India_ broadsheets** (`118` ground-truth ad instances and `8` ad-free editorial background pages) at `imgsz=1024` and `conf=0.50`:

| Metric                       | Baseline `v1` | Fine-Tuned `bilingual_v2` |      Improvement       |
| :--------------------------- | :-----------: | :-----------------------: | :--------------------: |
| **Box Precision (P)**        |     0.605     |         **0.818**         |       **+0.213**       |
| **Box Recall (R)**           |     0.441     |         **0.763**         |       **+0.322**       |
| **mAP@50**                   |    0.3649     |        **0.7215**         |  **+0.3566 (1.98x)**   |
| **mAP@50–95**                |    0.3201     |        **0.7027**         |  **+0.3826 (2.20x)**   |
| **Inference Speed (T4 GPU)** |    19.6 ms    |        **21.2 ms**        | ~47 FPS per broadsheet |

---

## Hybrid Detection & Post-Processing Pipeline

1. **Reflection Artifact Cleanup (`trim_mirror_padding`):** Detects and strips synthetic edge-reflection padding using grayscale symmetry seam matching before inference.
2. **Primary Detection (`YOLOAdDetector`):** Runs single-pass `YOLO11s` inference (`imgsz=1024`, `iou=0.45`, `conf=0.70` default) and filters narrow edge ribbons.
3. **Uniform-Background Ad Recovery (`is_uniform_solid_ad_block`):** Identifies smooth, single-hue display blocks via local smoothness (`local_diff <= 6.0`) and dominant color-bin consistency, rejecting high-texture news photos and shallow quote boxes.
4. **Outward Rule Snapping & Gutter Trimming (`snap_outward_to_border`, `trim_exact_ad_edges`):** Uses Otsu + adaptive thresholding and median newsprint color estimation (`estimate_paper_color`) to snap boxes onto enclosing border rules and remove overshot editorial lines across white separator gutters.
5. **Footer Rule Extension (`extend_to_footer_rule`):** Scans beneath colored ad graphics for horizontal enclosing rules (both dark and light-gray) so white disclaimer strips are included flush to the bottom border.

---

## Project Structure

```text
NewPaper-ad-detection/
├── app.py                  # Streamlit UI, YOLO inference, and OpenCV post-processing pipeline
├── models/
│   └── best.pt             # Fine-tuned YOLO11s weights (bilingual_v2)
├── requirements.txt        # Python dependencies (ultralytics, opencv-python-headless, streamlit)
└── README.md
```

---

## Local Setup

```bash
git clone [https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper.git](https://github.com/opmk1234/Newspaper-Ad-Detector-Cropper.git)
cd Newspaper-Ad-Detector-Cropper
pip install -r requirements.txt
streamlit run app.py
```
