"""Pure-NumPy pre/post-processing: runs identically on a laptop and a Raspberry Pi."""
import math

import cv2
import numpy as np

INPUT_SIZE = 320
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # torchvision SSD box-coder weights
DW_CLAMP = math.log(1000.0 / 16)

# COCO class ids (torchvision 91-index convention) that SafeCross cares about
PEDESTRIAN_CLASSES = {1: "person"}
VEHICLE_CLASSES = {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
TARGET_CLASSES = {**PEDESTRIAN_CLASSES, **VEHICLE_CLASSES}


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 [1,3,320,320] RGB in [0,1]."""
    img = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def decode_boxes(rel: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """SSD box decoding (matches torchvision BoxCoder.decode_single)."""
    wx, wy, ww, wh = BOX_WEIGHTS
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah
    dx, dy = rel[:, 0] / wx, rel[:, 1] / wy
    dw = np.minimum(rel[:, 2] / ww, DW_CLAMP)
    dh = np.minimum(rel[:, 3] / wh, DW_CLAMP)
    cx, cy = dx * aw + acx, dy * ah + acy
    w, h = np.exp(dw) * aw, np.exp(dh) * ah
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(rel, scores, anchors, orig_w, orig_h, conf=0.4, iou=0.55,
                classes=TARGET_CLASSES, merge_vehicles=True, merge_iou=0.7):
    """Returns list of (x1, y1, x2, y2, score, class_id) in original image pixels."""
    rel, scores = rel[0], scores[0]
    boxes = decode_boxes(rel, anchors)
    sx, sy = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
    dets = []
    for cid in classes:
        s = scores[:, cid]
        m = s > conf
        if not m.any():
            continue
        b, sc = boxes[m], s[m]
        for k in nms(b, sc, iou):
            x1, y1, x2, y2 = b[k]
            x1, x2 = np.clip([x1 * sx, x2 * sx], 0, orig_w)
            y1, y2 = np.clip([y1 * sy, y2 * sy], 0, orig_h)
            dets.append((float(x1), float(y1), float(x2), float(y2), float(sc[k]), cid))
    if merge_vehicles:
        dets = merge_vehicle_duplicates(dets, merge_iou)
    return dets


def merge_vehicle_duplicates(dets, iou_thr=0.7):
    """Per-class NMS can label one van as both 'car' and 'truck'. For counting we keep
    only the highest-scoring vehicle label when vehicle boxes overlap heavily."""
    veh = [d for d in dets if d[5] in VEHICLE_CLASSES]
    other = [d for d in dets if d[5] not in VEHICLE_CLASSES]
    if len(veh) < 2:
        return dets
    b = np.array([d[:4] for d in veh], dtype=np.float32)
    s = np.array([d[4] for d in veh], dtype=np.float32)
    return other + [veh[k] for k in nms(b, s, iou_thr)]


# --------------------------------------------------------------------------- #
# Safety analytics
# --------------------------------------------------------------------------- #
def _expand(box, factor):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * factor, (y2 - y1) * factor
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def proximity_pairs(dets, expand=1.5):
    """Pedestrian-vehicle pairs whose (pedestrian box x expand) overlaps the vehicle box.
    A cheap per-frame proxy for 'pedestrian is close to a vehicle'."""
    peds = [d[:4] for d in dets if d[5] in PEDESTRIAN_CLASSES]
    vehs = [d[:4] for d in dets if d[5] in VEHICLE_CLASSES]
    return sum(1 for p in peds for v in vehs if _overlaps(_expand(p, expand), v))


def risk_level(n_ped, n_veh, n_pairs, ped_min=3, veh_min=2):
    if n_pairs >= 2 or (n_ped >= ped_min and n_veh >= veh_min and n_pairs >= 1):
        return "HIGH"
    if n_pairs >= 1 or (n_ped >= ped_min and n_veh >= veh_min):
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
COLORS = {"person": (0, 200, 255), "bicycle": (255, 160, 0), "car": (60, 220, 60),
          "motorcycle": (255, 0, 200), "bus": (0, 120, 255), "truck": (200, 80, 80)}
RISK_COLORS = {"LOW": (60, 180, 60), "MEDIUM": (0, 190, 255), "HIGH": (0, 0, 230)}


def annotate(bgr, dets, counts, risk, infer_ms):
    img = bgr.copy()
    th = max(1, round(min(img.shape[:2]) / 300))
    for x1, y1, x2, y2, sc, cid in dets:
        name = TARGET_CLASSES[cid]
        c = COLORS[name]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, th + 1)
        cv2.putText(img, f"{name} {sc:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * th, c, th)
    lines = [f"Pedestrians: {counts['pedestrians']}   Vehicles: {counts['vehicles']}",
             f"Risk: {risk}   ({infer_ms:.1f} ms inference)"]
    pad, lh = 8, int(24 * th * 0.8)
    cv2.rectangle(img, (0, 0), (int(330 * th * 0.8), pad * 2 + lh * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        col = RISK_COLORS[risk] if i == 1 else (255, 255, 255)
        cv2.putText(img, t, (pad, pad + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * th * 0.8, col, th)
    return img
