import csv
import io
import itertools
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


def overlap_ratio(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection over smaller box area to prevent duplicate detections."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / min(area_a, area_b)


def snap_outward_to_border(
    work_bgr: np.ndarray,
    work_gray: np.ndarray,
    paper_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    max_scan_ratio: float = 0.02,
) -> tuple[int, int, int, int]:
    """Expand a slightly tight YOLO box outward to the nearest enclosing border rule (0 px extra padding)."""
    h, w = work_gray.shape
    bx1, by1, bx2, by2 = box

    scan_x = max(8, int(w * max_scan_ratio))
    scan_y = max(8, int(h * max_scan_ratio))
    sx1, sy1 = max(0, bx1 - scan_x), max(0, by1 - scan_y)
    sx2, sy2 = min(w, bx2 + scan_x), min(h, by2 + scan_y)

    roi_gray = work_gray[sy1:sy2, sx1:sx2]
    roi_bgr = work_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
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

    fx1 = max(0, bx1 - left_ext)
    fy1 = max(0, by1 - top_ext)
    fx2 = min(w, bx2 + right_ext)
    fy2 = min(h, by2 + bottom_ext)

    return (fx1, fy1, fx2, fy2) if (fx2 > fx1 and fy2 > fy1) else (bx1, by1, bx2, by2)


def trim_exact_ad_edges(
    img_bgr: np.ndarray,
    paper_bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Trim any overshot editorial text across a clean white separator gutter and strip
    outer blank newsprint margins on all 4 sides so zero extra padding remains.
    """
    fx1, fy1, fx2, fy2 = box
    crop_bgr = img_bgr[fy1:fy2, fx1:fx2].astype(np.float32)
    bh, bw = crop_bgr.shape[:2]
    if bh < 30 or bw < 30:
        return box

    # 1. Trim overshot top editorial text across a white separator gutter (within top 10%)
    fg = np.linalg.norm(crop_bgr - paper_bgr, axis=2) > 18.0
    row_fg = fg.mean(axis=1)
    max_trim_y = int(0.10 * bh)
    blank_top = np.where(row_fg[:max_trim_y] < 0.02)[0]
    if len(blank_top) >= 3:
        last_blank = int(blank_top[-1])
        if last_blank + 3 < bh and row_fg[last_blank + 1 : last_blank + 4].mean() > 0.30:
            above = row_fg[:last_blank]
            has_editorial = np.any((above > 0.04) & (above < 0.78))
            all_blank = np.all(above < 0.02) if len(above) > 0 else True
            if all_blank or has_editorial:
                fy1 += last_blank + 1

    # 2. Trim overshot bottom editorial text across a white separator gutter (within bottom 10%)
    crop_bgr = img_bgr[fy1:fy2, fx1:fx2].astype(np.float32)
    bh, bw = crop_bgr.shape[:2]
    fg = np.linalg.norm(crop_bgr - paper_bgr, axis=2) > 18.0
    row_fg = fg.mean(axis=1)
    max_trim_y = int(0.10 * bh)
    blank_bot = np.where(row_fg[bh - max_trim_y :] < 0.02)[0]
    if len(blank_bot) >= 3:
        first_blank = (bh - max_trim_y) + int(blank_bot[0])
        if first_blank - 3 >= 0 and row_fg[first_blank - 3 : first_blank].mean() > 0.30:
            below = row_fg[first_blank + 1 :]
            has_editorial = np.any((below > 0.04) & (below < 0.78))
            all_blank = np.all(below < 0.02) if len(below) > 0 else True
            if all_blank or has_editorial:
                fy2 = fy1 + first_blank

    # 3. Strip outer blank paper rows/columns on all 4 edges (0 px extra padding)
    crop_bgr = img_bgr[fy1:fy2, fx1:fx2].astype(np.float32)
    fg = np.linalg.norm(crop_bgr - paper_bgr, axis=2) > 18.0
    row_fg = fg.mean(axis=1)
    col_fg = fg.mean(axis=0)

    while fy1 < fy2 - 20 and row_fg[0] < 0.015:
        fy1 += 1
        row_fg = row_fg[1:]
    while fy2 > fy1 + 20 and row_fg[-1] < 0.015:
        fy2 -= 1
        row_fg = row_fg[:-1]
    while fx1 < fx2 - 20 and col_fg[0] < 0.015:
        fx1 += 1
        col_fg = col_fg[1:]
    while fx2 > fx1 + 20 and col_fg[-1] < 0.015:
        fx2 -= 1
        col_fg = col_fg[:-1]

    return (fx1, fy1, fx2, fy2) if (fx2 > fx1 and fy2 > fy1) else box


def extend_to_footer_rule(
    full_gray: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    If a white footer bar (e.g. 'Only COPPER MOTOR...' or disclaimer text) sits
    directly beneath a colored graphic and ends at an enclosing horizontal rule,
    extend y2 down flush to that rule without bleeding into content below it.
    """
    h, _ = full_gray.shape
    fx1, fy1, fx2, fy2 = box
    bw = fx2 - fx1
    if bw < 20:
        return box

    max_scan = max(16, int(0.048 * h))
    sy2 = min(h, fy2 + max_scan)
    strip = full_gray[fy2:sy2, fx1:fx2]
    if strip.size == 0:
        return box

    # Detect thin dark OR light-gray horizontal rules via horizontal line continuity
    line_mask = (strip < 228).astype(np.uint8)
    closed_h = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=np.uint8))

    rule_rows = []
    min_run = max(15, int(0.42 * bw))
    for r in range(closed_h.shape[0]):
        row = closed_h[r]
        if row.mean() < 0.35:
            continue
        longest_run = max((len(list(g)) for val, g in itertools.groupby(row) if val == 1), default=0)
        if longest_run >= min_run:
            rule_rows.append(r)

    if not rule_rows:
        return box

    first_rule = rule_rows[0]
    # Include the full thickness (1-3 px) of that rule, then stop immediately
    rule_end = first_rule
    while rule_end + 1 in rule_rows and (rule_end - first_rule) < 4:
        rule_end += 1

    if first_rule < 4:
        return (fx1, fy1, fx2, min(h, fy2 + rule_end + 1))

    above_rule = (strip[:first_rule] < 225).mean(axis=1)
    has_footer_text = np.any(above_rule > 0.04)
    blank_gap = max(
        (len(list(g)) for is_blank, g in itertools.groupby(above_rule <= 0.025) if is_blank),
        default=0,
    )
    if has_footer_text and blank_gap <= max(10, int(0.015 * h)):
        return (fx1, fy1, fx2, min(h, fy2 + rule_end + 1))
    return box


class YOLOAdDetector:
    def __init__(self, weights_path: str = "models/best.pt", conf: float = 0.50, imgsz: int = 1024):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        h, w = image_bgr.shape[:2]

        # Single forward pass down to floor=0.15 (zero extra PyTorch RAM)
        floor_conf = min(0.15, self.conf)
        res = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=floor_conf,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )[0]

        snap_scale = min(1.0, 1600.0 / max(h, w))
        if snap_scale < 1.0:
            sw = max(1, int(round(w * snap_scale)))
            sh = max(1, int(round(h * snap_scale)))
            work_bgr = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            work_bgr = image_bgr
            sh, sw = h, w

        work_gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        full_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        paper_work = estimate_paper_color(work_bgr)
        paper_full = estimate_paper_color(image_bgr)

        candidates: list[tuple[int, int, int, int, float]] = []

        if res.boxes is not None and len(res.boxes) > 0:
            raw_boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            order = np.lexsort((raw_boxes[:, 0], raw_boxes[:, 1]))

            for idx in order:
                x1, y1, x2, y2 = map(int, np.round(raw_boxes[idx]))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                conf = float(confs[idx])

                bw, bh = x2 - x1, y2 - y1
                if bw < int(0.03 * w) or bh < int(0.02 * h):
                    continue

                # Drop narrow vertical ribbons hugging the outer left/right edges
                edge_margin = max(6, int(0.006 * w))
                touches_lr_edge = (x1 <= edge_margin) or (x2 >= w - edge_margin)
                if touches_lr_edge and bw < int(0.24 * w) and (bh / max(1, bw)) > 3.5:
                    continue

                if conf >= self.conf:
                    candidates.append((x1, y1, x2, y2, conf))
                else:
                    # Rescue lower-confidence boxes only if they sit on a tinted/colored ad block
                    if bw < int(0.06 * w) or bh < int(0.03 * h):
                        continue
                    if y1 < int(0.08 * h) and (bw / max(1, bh)) > 3.2:
                        continue
                    crop_bgr = image_bgr[y1:y2, x1:x2]
                    tinted_ratio = float(
                        (np.linalg.norm(crop_bgr.astype(np.float32) - paper_full, axis=2) > 16.0).mean()
                    )
                    mean_sat = float(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                    if tinted_ratio >= 0.35 and mean_sat >= 18.0:
                        candidates.append((x1, y1, x2, y2, conf))

        # Pastel/colored rectangular display-ad recovery (catches solid-tint sidebar ads like USHA)
        color_dist = np.linalg.norm(work_bgr.astype(np.float32) - paper_work, axis=2)
        sat = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        color_mask = ((color_dist > 16.0) & (sat > 16)).astype(np.uint8) * 255
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        )
        num_lbls, _, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)

        for lbl in range(1, num_lbls):
            cx, cy, cw, ch, area = stats[lbl]
            if cw < int(0.08 * sw) or ch < int(0.035 * sh) or cw > int(0.95 * sw) or ch > int(0.85 * sh):
                continue
            if (area / float(max(1, cw * ch))) < 0.74:
                continue

            rx1 = max(0, int(round(cx / snap_scale)))
            ry1 = max(0, int(round(cy / snap_scale)))
            rx2 = min(w, int(round((cx + cw) / snap_scale)))
            ry2 = min(h, int(round((cy + ch) / snap_scale)))

            if ry1 < int(0.08 * h) and (rx2 - rx1) / max(1, ry2 - ry1) > 3.2:
                continue
            if any(overlap_ratio((rx1, ry1, rx2, ry2), c[:4]) > 0.20 for c in candidates):
                continue

            candidates.append((rx1, ry1, rx2, ry2, 0.72))

        detections = []
        for x1, y1, x2, y2, conf in candidates:
            wx1 = int(round(x1 * snap_scale))
            wy1 = int(round(y1 * snap_scale))
            wx2 = int(round(x2 * snap_scale))
            wy2 = int(round(y2 * snap_scale))

            sx1, sy1, sx2, sy2 = snap_outward_to_border(
                work_bgr=work_bgr,
                work_gray=work_gray,
                paper_bgr=paper_work,
                box=(wx1, wy1, wx2, wy2),
            )

            fx1 = max(0, int(round(sx1 / snap_scale)))
            fy1 = max(0, int(round(sy1 / snap_scale)))
            fx2 = min(w, int(round(sx2 / snap_scale)))
            fy2 = min(h, int(round(sy2 / snap_scale)))

            fx1, fy1, fx2, fy2 = trim_exact_ad_edges(image_bgr, paper_full, (fx1, fy1, fx2, fy2))
            fx1, fy1, fx2, fy2 = extend_to_footer_rule(full_gray, (fx1, fy1, fx2, fy2))

            detections.append({
                "bbox": (fx1, fy1, fx2, fy2),
                "confidence": round(conf, 4),
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
    return YOLOAdDetector(weights_path=weights_path, conf=0.50, imgsz=1024)


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
        value=0.50,
        step=0.05,
        help="Minimum confidence required to keep a detected ad (0.50 is tuned for bilingual_v2).",
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