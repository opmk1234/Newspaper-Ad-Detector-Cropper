import csv
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_weights_path() -> Path | None:
    """Locate trained YOLO weights in models/best.pt (or fallback model/best.pt)."""
    for candidate in (Path("models/best.pt"), Path("model/best.pt")):
        if candidate.exists():
            return candidate
    return None


def trim_mirror_padding(img_bgr: np.ndarray, mae_thresh: float = 9.5) -> np.ndarray:
    """Strip Roboflow reflection padding from both left and right edges without touching blank margins."""
    h, w = img_bgr.shape[:2]
    proxy_w = min(w, 480)
    scale_x = w / proxy_w

    if proxy_w < w:
        proxy_h = max(1, int(round(h / scale_x)))
        small = cv2.resize(img_bgr, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    min_span = max(8, int(0.015 * proxy_w))
    max_margin = int(0.48 * proxy_w)

    # 1. Left-side reflection seam (real content lies to the right of left_seam)
    left_seam = 0
    best_left_span = 0
    for s in range(min_span, max_margin):
        span = min(s, proxy_w - 1 - s)
        if span < min_span:
            continue
        reflected = gray[:, s - span : s][:, ::-1]
        # Skip blank white newspaper margins
        if np.std(reflected) < 12.0:
            continue
        diff_101 = np.mean(np.abs(reflected - gray[:, s + 1 : s + 1 + span]))
        diff_ref = np.mean(np.abs(reflected - gray[:, s : s + span]))
        if min(diff_101, diff_ref) < mae_thresh and span > best_left_span:
            best_left_span = span
            left_seam = s

    # 2. Right-side reflection seam (real content lies to the left of right_seam)
    right_seam = proxy_w
    best_right_span = 0
    for s in range(proxy_w - max_margin, proxy_w - min_span):
        span = min(s, proxy_w - 1 - s)
        if span < min_span:
            continue
        reflected = gray[:, s - span : s][:, ::-1]
        if np.std(reflected) < 12.0:
            continue
        diff_101 = np.mean(np.abs(reflected - gray[:, s + 1 : s + 1 + span]))
        diff_ref = np.mean(np.abs(reflected - gray[:, s : s + span]))
        if min(diff_101, diff_ref) < mae_thresh and span > best_right_span:
            best_right_span = span
            right_seam = s

    x1 = int(round(left_seam * scale_x))
    x2 = int(round(right_seam * scale_x))
    if 0 <= x1 < x2 <= w and (x1 > 0 or x2 < w):
        return img_bgr[:, x1:x2]
    return img_bgr


def estimate_paper_color(img_bgr: np.ndarray) -> np.ndarray:
    """Estimate the blank newsprint background color from outer page margins."""
    h, w = img_bgr.shape[:2]
    my = max(4, int(0.015 * h))
    mx = max(4, int(0.015 * w))
    border_pixels = np.concatenate(
        [
            img_bgr[:my, ::2, :].reshape(-1, 3),
            img_bgr[-my:, ::2, :].reshape(-1, 3),
            img_bgr[::2, :mx, :].reshape(-1, 3),
            img_bgr[::2, -mx:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0).astype(np.float32)


def snap_box_to_ad_boundary(
    img_bgr: np.ndarray,
    gray: np.ndarray,
    paper_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    expand_pct: float = 0.0,
    max_scan_ratio: float = 0.02,
    safety_pad_px: int = 2,
) -> tuple[int, int, int, int]:
    """Snap a YOLO box to the nearest outer ad border rule without crossing into editorial text."""
    h, w = gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    grow_x = int(round(bw * expand_pct))
    grow_y = int(round(bh * expand_pct))
    bx1 = max(0, x1 - grow_x)
    by1 = max(0, y1 - grow_y)
    bx2 = min(w, x2 + grow_x)
    by2 = min(h, y2 + grow_y)

    scan_x = max(10, int(w * max_scan_ratio))
    scan_y = max(10, int(h * max_scan_ratio))
    sx1, sy1 = max(0, bx1 - scan_x), max(0, by1 - scan_y)
    sx2, sy2 = min(w, bx2 + scan_x), min(h, by2 + scan_y)

    roi_gray = gray[sy1:sy2, sx1:sx2]
    roi_bgr = img_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    if roi_gray.size == 0:
        return bx1, by1, bx2, by2

    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    ink_mask = cv2.bitwise_or(otsu, adapt) > 0

    color_diff = np.linalg.norm(roi_bgr - paper_bgr, axis=2)
    fg_mask = np.logical_or(ink_mask, color_diff > 22.0)

    rx1, ry1 = bx1 - sx1, by1 - sy1
    rx2, ry2 = bx2 - sx1, by2 - sy1
    gutter_stop_px = max(4, int(0.004 * min(w, h)))

    def scan_side(strip_ink: np.ndarray, strip_fg: np.ndarray, reverse: bool) -> int:
        length = strip_ink.shape[0]
        if length == 0:
            return 0

        ink_density = strip_ink.mean(axis=1)
        fg_density = strip_fg.mean(axis=1)
        if reverse:
            ink_density = ink_density[::-1]
            fg_density = fg_density[::-1]

        border_hits = np.where((ink_density > 0.60) | (fg_density > 0.70))[0]
        if len(border_hits) > 0:
            return int(border_hits.min()) + 1

        best_offset = 0
        empty_run = 0
        for idx, d in enumerate(fg_density):
            if d > 0.025:
                best_offset = idx + 1
                empty_run = 0
            else:
                empty_run += 1
                if empty_run >= gutter_stop_px:
                    break
        return best_offset

    left_ext = scan_side(ink_mask[ry1:ry2, :rx1].T, fg_mask[ry1:ry2, :rx1].T, reverse=True)
    right_ext = scan_side(ink_mask[ry1:ry2, rx2:].T, fg_mask[ry1:ry2, rx2:].T, reverse=False)
    top_ext = scan_side(ink_mask[:ry1, rx1:rx2], fg_mask[:ry1, rx1:rx2], reverse=True)
    bottom_ext = scan_side(ink_mask[ry2:, rx1:rx2], fg_mask[ry2:, rx1:rx2], reverse=False)

    fx1 = max(0, bx1 - left_ext - safety_pad_px)
    fy1 = max(0, by1 - top_ext - safety_pad_px)
    fx2 = min(w, bx2 + right_ext + safety_pad_px)
    fy2 = min(h, by2 + bottom_ext + safety_pad_px)

    return (fx1, fy1, fx2, fy2) if (fx2 > fx1 and fy2 > fy1) else (bx1, by1, bx2, by2)


class YOLOAdDetector:
    def __init__(self, weights_path: str = "models/best.pt", conf: float = 0.60, imgsz: int = 1024):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        h, w = image_bgr.shape[:2]

        res = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return []

        snap_scale = min(1.0, 1600.0 / max(h, w))
        if snap_scale < 1.0:
            sw = max(1, int(round(w * snap_scale)))
            sh = max(1, int(round(h * snap_scale)))
            work_bgr = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            work_bgr = image_bgr

        work_gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        paper_bgr = estimate_paper_color(work_bgr)
        scaled_pad = max(1, int(round(2 * snap_scale)))

        raw_boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        order = np.lexsort((raw_boxes[:, 0], raw_boxes[:, 1]))

        detections = []
        for idx in order:
            x1, y1, x2, y2 = map(int, np.round(raw_boxes[idx]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw, bh = x2 - x1, y2 - y1
            if bw < int(0.03 * w) or bh < int(0.02 * h):
                continue

            # Drop narrow vertical ribbons hugging the outer left/right edges
            edge_margin = max(6, int(0.006 * w))
            touches_lr_edge = (x1 <= edge_margin) or (x2 >= w - edge_margin)
            if touches_lr_edge and bw < int(0.24 * w) and (bh / max(1, bw)) > 3.5:
                continue

            wx1 = int(round(x1 * snap_scale))
            wy1 = int(round(y1 * snap_scale))
            wx2 = int(round(x2 * snap_scale))
            wy2 = int(round(y2 * snap_scale))

            sx1, sy1, sx2, sy2 = snap_box_to_ad_boundary(
                img_bgr=work_bgr,
                gray=work_gray,
                paper_bgr=paper_bgr,
                box=(wx1, wy1, wx2, wy2),
                expand_pct=0.0,
                safety_pad_px=scaled_pad,
            )

            fx1 = max(0, int(round(sx1 / snap_scale)))
            fy1 = max(0, int(round(sy1 / snap_scale)))
            fx2 = min(w, int(round(sx2 / snap_scale)))
            fy2 = min(h, int(round(sy2 / snap_scale)))

            detections.append({
                "bbox": (fx1, fy1, fx2, fy2),
                "confidence": round(float(confs[idx]), 4),
            })

        # Merge horizontally overlapping boxes that share the same vertical band
        merged: list[dict] = []
        for det in sorted(detections, key=lambda d: d["bbox"][0]):
            cx1, cy1, cx2, cy2 = det["bbox"]
            merged_into_existing = False
            for idx_m, prev in enumerate(merged):
                px1, py1, px2, py2 = prev["bbox"]
                inter_y = max(0, min(py2, cy2) - max(py1, cy1))
                v_overlap = inter_y / max(1, min(py2 - py1, cy2 - cy1))
                if v_overlap > 0.80 and cx1 <= px2 and cx2 >= px1:
                    merged[idx_m] = {
                        "bbox": (min(px1, cx1), min(py1, cy1), max(px2, cx2), max(py2, cy2)),
                        "confidence": max(prev["confidence"], det["confidence"]),
                    }
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                merged.append(det)

        merged.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return merged


def build_preview_jpeg(img_bgr: np.ndarray, detections: list[dict], max_dim: int = 1400) -> bytes:
    """Create a lightweight JPEG preview of the page with green bounding boxes."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        preview = cv2.resize(
            img_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = img_bgr.copy()

    for ad_idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(c * scale)) for c in det["bbox"]]
        conf = det["confidence"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            preview,
            f"ad #{ad_idx} ({conf:.2f})",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
        )

    _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def unpack_uploaded_images(uploaded_files) -> list[tuple[str, str, bytes]]:
    """Extract individual images from direct uploads and .zip archives."""
    items: list[tuple[str, str, bytes]] = []
    seen_stems: dict[str, int] = {}

    def unique_stem(raw_name: str) -> str:
        base = Path(raw_name).stem
        count = seen_stems.get(base, 0) + 1
        seen_stems[base] = count
        return base if count == 1 else f"{base}_{count}"

    for uf in uploaded_files:
        raw = uf.getvalue()
        suffix = Path(uf.name).suffix.lower()

        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                for member in sorted(zf.namelist()):
                    p = Path(member)
                    if member.endswith("/") or "__MACOSX" in p.parts or p.name.startswith("."):
                        continue
                    if p.suffix.lower() in VALID_EXTS:
                        items.append((p.name, unique_stem(p.name), zf.read(member)))
        elif suffix in VALID_EXTS:
            items.append((uf.name, unique_stem(uf.name), raw))

    return items


def run_extraction_pipeline(
    image_items: list[tuple[str, str, bytes]],
    detector: YOLOAdDetector,
    organize_by_folder: bool = False,
    show_progress: bool = False,
) -> dict:
    progress = st.progress(0, text="Starting ad detection...") if show_progress else None
    pages_data = []
    manifest_rows = []
    zip_buffer = io.BytesIO()
    total_ads = 0
    pages_with_ads = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as master_zip:
        for idx, (display_name, stem, img_bytes) in enumerate(image_items):
            if progress is not None:
                progress.progress(
                    idx / len(image_items),
                    text=f"Processing {idx + 1}/{len(image_items)}: {display_name}",
                )

            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            img_bgr = trim_mirror_padding(img_bgr)
            detections = detector.detect(img_bgr)
            page_crops = []

            if detections:
                pages_with_ads += 1

            for ad_idx, det in enumerate(detections, start=1):
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                total_ads += 1

                crop_bgr = img_bgr[y1:y2, x1:x2]
                ok, png_buf = cv2.imencode(".png", crop_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if ok:
                    crop_filename = f"{stem}_ad_{ad_idx:02d}.png"
                    zip_member_path = f"{stem}/{crop_filename}" if organize_by_folder else crop_filename
                    png_bytes = png_buf.tobytes()
                    master_zip.writestr(zip_member_path, png_bytes)

                    page_crops.append({
                        "id": ad_idx,
                        "confidence": conf,
                        "filename": crop_filename,
                        "png_bytes": png_bytes,
                        "bbox": (x1, y1, x2, y2),
                    })
                    manifest_rows.append([
                        display_name,
                        zip_member_path,
                        f"{conf:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                        x2 - x1,
                        y2 - y1,
                    ])

            pages_data.append({
                "display_name": display_name,
                "stem": stem,
                "preview_bytes": build_preview_jpeg(img_bgr, detections),
                "crops": page_crops,
            })

        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["source_page", "crop_file", "confidence", "x1", "y1", "x2", "y2", "width", "height"])
        writer.writerows(manifest_rows)
        master_zip.writestr("summary.csv", csv_buf.getvalue())

    if progress is not None:
        progress.empty()

    return {
        "pages": pages_data,
        "zip_bytes": zip_buffer.getvalue(),
        "total_pages": len(pages_data),
        "pages_with_ads": pages_with_ads,
        "total_ads": total_ads,
    }


@st.cache_resource
def load_detector(weights_path: str) -> YOLOAdDetector:
    return YOLOAdDetector(weights_path=weights_path, conf=0.60, imgsz=1024)


st.set_page_config(page_title="Newspaper Ad Cropper", layout="wide")
st.title("Newspaper Ad Detector & Cropper")
st.caption("Upload a single newspaper page or a batch of pages to detect and extract advertisements.")

weights_file = resolve_weights_path()
if weights_file is None:
    st.error("Model weights not found at `models/best.pt` (or `model/best.pt`).")
    st.stop()

detector = load_detector(str(weights_file))

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Minimum confidence required to keep a detected ad (0.60 is the validated default).",
    )
    st.caption("Adjust the threshold, then click Submit on your upload.")

detector.conf = conf_threshold

# --- Two Separate Upload & Submit Sections ---
col_single, col_bulk = st.columns(2)

with col_single:
    with st.container(border=True):
        st.subheader("1. Single Image Upload")
        single_file = st.file_uploader(
            "Choose one newspaper page",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="single_uploader",
        )
        submit_single = st.button(
            "Submit Single Image",
            type="primary",
            use_container_width=True,
            disabled=(single_file is None),
            key="submit_single",
        )

with col_bulk:
    with st.container(border=True):
        st.subheader("2. Bulk File Upload")
        bulk_files = st.file_uploader(
            "Choose multiple newspaper pages or a .zip archive",
            type=["png", "jpg", "jpeg", "webp", "zip"],
            accept_multiple_files=True,
            key="bulk_uploader",
        )
        organize_by_folder = st.checkbox(
            "Group crops into separate folders by page inside ZIP",
            value=False,
        )
        submit_bulk = st.button(
            "Submit Bulk Files",
            type="primary",
            use_container_width=True,
            disabled=(not bulk_files),
            key="submit_bulk",
        )

if submit_single and single_file is not None:
    items = unpack_uploaded_images([single_file])
    if not items:
        st.error("Could not read the uploaded image.")
    else:
        with st.spinner(f"Detecting ads on {single_file.name}..."):
            res = run_extraction_pipeline(items, detector, organize_by_folder=False, show_progress=False)
        res["mode"] = "single"
        st.session_state["last_run"] = res

if submit_bulk and bulk_files:
    items = unpack_uploaded_images(bulk_files)
    if not items:
        st.error("No valid image files found in the bulk upload.")
    else:
        res = run_extraction_pipeline(
            items,
            detector,
            organize_by_folder=organize_by_folder,
            show_progress=True,
        )
        res["mode"] = "bulk"
        st.session_state["last_run"] = res

if "last_run" not in st.session_state:
    st.info("Upload a file in either **Single Image Upload** or **Bulk File Upload**, then click its **Submit** button.")
    st.stop()

results = st.session_state["last_run"]
pages = results["pages"]

if not pages:
    st.error("No readable pages were processed.")
    st.stop()

st.divider()

# --- Results Display ---
if results["mode"] == "single":
    active_page = pages[0]
    ad_count = len(active_page["crops"])

    if ad_count > 0:
        st.success(f"Detected {ad_count} ad(s) on **{active_page['display_name']}**.")
        st.download_button(
            label=f"Download All {ad_count} Crops (.zip)",
            data=results["zip_bytes"],
            file_name=f"{active_page['stem']}_ads.zip",
            mime="application/zip",
            type="primary",
        )
    else:
        st.warning(f"No ads detected on **{active_page['display_name']}**.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages Processed", results["total_pages"])
    c2.metric("Pages with Ads", results["pages_with_ads"])
    c3.metric("Ad-Free Pages", results["total_pages"] - results["pages_with_ads"])
    c4.metric("Total Ads Extracted", results["total_ads"])

    if results["total_ads"] > 0:
        st.download_button(
            label=f"Download All {results['total_ads']} Crops + summary.csv (.zip)",
            data=results["zip_bytes"],
            file_name="extracted_newspaper_ads.zip",
            mime="application/zip",
            type="primary",
        )
    else:
        st.warning("No ads were detected across the uploaded batch.")

    if len(pages) == 1:
        active_page = pages[0]
    else:
        filter_ads_only = st.checkbox("Show only pages with detected ads", value=False)
        visible_indices = [
            i for i, p in enumerate(pages)
            if (not filter_ads_only or len(p["crops"]) > 0)
        ]

        if not visible_indices:
            st.info("No pages matched the filter. Uncheck 'Show only pages with detected ads' to inspect all pages.")
            st.stop()

        selected_idx = st.selectbox(
            "Inspect page",
            options=visible_indices,
            format_func=lambda i: f"{pages[i]['display_name']} — {len(pages[i]['crops'])} ad(s)",
        )
        active_page = pages[selected_idx]

col_page, col_crops = st.columns([1.2, 1])

with col_page:
    st.subheader(f"Page: {active_page['display_name']}")
    st.image(active_page["preview_bytes"], use_container_width=True)

with col_crops:
    st.subheader(f"Extracted Ads ({len(active_page['crops'])})")
    if not active_page["crops"]:
        st.info("No ads detected on this page.")
    else:
        for crop in active_page["crops"]:
            w_px = crop["bbox"][2] - crop["bbox"][0]
            h_px = crop["bbox"][3] - crop["bbox"][1]
            st.image(
                crop["png_bytes"],
                caption=f"Ad #{crop['id']} — conf {crop['confidence']:.2f} ({w_px}×{h_px} px)",
                use_container_width=True,
            )
            st.download_button(
                label=f"Download {crop['filename']}",
                data=crop["png_bytes"],
                file_name=crop["filename"],
                mime="image/png",
                key=f"dl_{results['mode']}_{active_page['stem']}_{crop['id']}",
            )