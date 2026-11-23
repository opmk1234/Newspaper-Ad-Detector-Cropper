import itertools
import cv2
import numpy as np


def _estimate_paper_bgr(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    my, mx = max(4, int(0.015 * h)), max(4, int(0.015 * w))
    borders = np.concatenate(
        [
            img_bgr[:my, ::2].reshape(-1, 3),
            img_bgr[-my:, ::2].reshape(-1, 3),
            img_bgr[::2, :mx].reshape(-1, 3),
            img_bgr[::2, -mx:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(borders, axis=0).astype(np.float32)


def _iou_or_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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


def recover_missed_display_ads(
    img_bgr: np.ndarray,
    model,
    existing_boxes: list[tuple[int, int, int, int, float]],
    imgsz: int = 1024,
) -> list[tuple[int, int, int, int, float]]:
    """
    1. Recovers low-confidence YOLO detections (0.28..conf_thresh) that sit on a
       colored/non-paper background or inside a crisp rectangular frame.
    2. Recovers solid colored rectangular display ads (like the sky-blue USHA fan ad)
       and snaps bottom borders down across white footer strips up to the enclosing rule.
    """
    h, w = img_bgr.shape[:2]
    paper_bgr = _estimate_paper_bgr(img_bgr)
    boxes = list(existing_boxes)

    # Pass A: TTA rescue pass for medium-confidence YOLO candidates (0.28+)
    tta_res = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=0.28,
        iou=0.45,
        augment=True,
        agnostic_nms=True,
        verbose=False,
    )[0]

    if tta_res.boxes is not None and len(tta_res.boxes) > 0:
        xyxy = tta_res.boxes.xyxy.cpu().numpy()
        confs = tta_res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, np.round(xyxy[i]))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            conf = float(confs[i])

            if bw < int(0.05 * w) or bh < int(0.03 * h):
                continue
            # Skip top masthead nameplate strip
            if y1 < int(0.08 * h) and (bw / max(1, bh)) > 3.2:
                continue
            if any(_iou_or_overlap((x1, y1, x2, y2), b[:4]) > 0.25 for b in boxes):
                continue

            # Accept if TTA pushed confidence >= 0.48 OR box has strong non-paper color fill
            crop = img_bgr[y1:y2, x1:x2].astype(np.float32)
            non_paper_ratio = float((np.linalg.norm(crop - paper_bgr, axis=2) > 24.0).mean())
            if conf >= 0.48 or non_paper_ratio >= 0.35:
                boxes.append((x1, y1, x2, y2, conf))

    # Pass B: OpenCV colored rectangular block detector (catches solid-background product ads)
    scale = min(1.0, 1200.0 / max(h, w))
    work = (
        cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img_bgr
    )
    sh, sw = work.shape[:2]
    color_dist = np.linalg.norm(work.astype(np.float32) - paper_bgr, axis=2)
    sat = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[:, :, 1]

    mask = ((color_dist > 26.0) & (sat > 30)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    num_lbls, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for lbl in range(1, num_lbls):
        cx, cy, cw, ch, area = stats[lbl]
        if cw < int(0.10 * sw) or ch < int(0.04 * sh) or cw > int(0.95 * sw) or ch > int(0.85 * sh):
            continue
        # Rectangular display ads fill most of their bounding box (editorial photos have text holes)
        if (area / float(max(1, cw * ch))) < 0.78:
            continue

        x1 = max(0, int(round(cx / scale)))
        y1 = max(0, int(round(cy / scale)))
        x2 = min(w, int(round((cx + cw) / scale)))
        y2 = min(h, int(round((cy + ch) / scale)))

        if y1 < int(0.08 * h) and (x2 - x1) / max(1, y2 - y1) > 3.2:
            continue
        if any(_iou_or_overlap((x1, y1, x2, y2), b[:4]) > 0.20 for b in boxes):
            continue

        boxes.append((x1, y1, x2, y2, 0.75))

    # Pass C: Snap bottom boundary down across white footer bars (e.g., "Only COPPER MOTOR...")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    snapped: list[tuple[int, int, int, int, float]] = []
    max_footer_scan = max(14, int(0.045 * h))

    for x1, y1, x2, y2, conf in boxes:
        sy2 = min(h, y2 + max_footer_scan)
        strip = gray[y2:sy2, x1:x2]
        if strip.size > 0:
            row_ink = (strip < 200).mean(axis=1)
            rule_rows = np.where(row_ink > 0.45)[0]
            if len(rule_rows) > 0:
                first_rule = int(rule_rows[0])
                above_rule = (strip[:first_rule] < 225).mean(axis=1)
                blank_gap = max(
                    (len(list(g)) for is_blank, g in itertools.groupby(above_rule <= 0.03) if is_blank),
                    default=0,
                )
                # Only extend if there isn't a wide empty paragraph gap before the rule
                if blank_gap <= max(10, int(0.015 * h)):
                    y2 = min(h, y2 + first_rule + 2)
        snapped.append((x1, y1, x2, y2, conf))

    return snapped