"""SafeCross: offline pedestrian & vehicle counting with ExecuTorch on Raspberry Pi.

    python run_safecross.py --input clip.mp4
    python run_safecross.py --input images_dir/ --model models/ssdlite_int8.pte
    python run_safecross.py --input clip.mp4 --every 3       # process every 3rd frame

Outputs (in --out):
    counts.csv          per-frame counts per class, risk level, inference ms
    annotated/          annotated frames / images
    summary.json        aggregate counts + performance report
"""
import argparse
import csv
import glob
import json
import os
import platform
import statistics
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime

from safecross.postprocess import (PEDESTRIAN_CLASSES, TARGET_CLASSES, VEHICLE_CLASSES,
                                   annotate, postprocess, preprocess, proximity_pairs,
                                   risk_level)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def iter_frames(path, every):
    """Yields (name, bgr_frame)."""
    if os.path.isdir(path):
        files = sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(IMG_EXT))
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield os.path.splitext(os.path.basename(f))[0], img
        return
    if path.lower().endswith(IMG_EXT):
        yield os.path.splitext(os.path.basename(path))[0], cv2.imread(path)
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            yield f"frame_{idx:05d}", frame
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="video file, image file, or folder of images")
    ap.add_argument("--model", default="models/ssdlite_int8.pte")
    ap.add_argument("--anchors", default="models/anchors.npy")
    ap.add_argument("--out", default="output")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--every", type=int, default=1, help="process every Nth video frame")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-save", action="store_true", help="don't write annotated images")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "annotated"), exist_ok=True)
    anchors = np.load(args.anchors)

    t0 = time.perf_counter()
    method = Runtime.get().load_program(args.model).load_method("forward")
    load_ms = (time.perf_counter() - t0) * 1000

    dummy = torch.rand(1, 3, 320, 320)
    for _ in range(args.warmup):
        method.execute([dummy])

    rows, infer_times, total_times = [], [], []
    names = list(TARGET_CLASSES.values())
    print(f"Model: {args.model}  (loaded in {load_ms:.0f} ms)")
    print(f"{'frame':<22}{'peds':>5}{'vehs':>5}{'pairs':>6}  {'risk':<7}{'infer ms':>9}  detail")
    print("-" * 90)

    for name, frame in iter_frames(args.input, args.every):
        t_start = time.perf_counter()
        x = torch.from_numpy(preprocess(frame))
        t_inf = time.perf_counter()
        rel, scores = method.execute([x])
        infer_ms = (time.perf_counter() - t_inf) * 1000
        h, w = frame.shape[:2]
        dets = postprocess(rel.numpy(), scores.numpy(), anchors, w, h, conf=args.conf)
        total_ms = (time.perf_counter() - t_start) * 1000

        per_class = {n: 0 for n in names}
        for d in dets:
            per_class[TARGET_CLASSES[d[5]]] += 1
        n_ped = sum(per_class[n] for n in PEDESTRIAN_CLASSES.values())
        n_veh = sum(per_class[n] for n in VEHICLE_CLASSES.values())
        pairs = proximity_pairs(dets)
        risk = risk_level(n_ped, n_veh, pairs)

        infer_times.append(infer_ms)
        total_times.append(total_ms)
        rows.append({"frame": name, **per_class, "pedestrians": n_ped, "vehicles": n_veh,
                     "proximity_pairs": pairs, "risk": risk,
                     "infer_ms": round(infer_ms, 2), "total_ms": round(total_ms, 2)})
        detail = ", ".join(f"{k}={v}" for k, v in per_class.items() if v)
        print(f"{name:<22}{n_ped:>5}{n_veh:>5}{pairs:>6}  {risk:<7}{infer_ms:>9.1f}  {detail}")

        if not args.no_save:
            img = annotate(frame, dets, {"pedestrians": n_ped, "vehicles": n_veh}, risk, infer_ms)
            cv2.imwrite(os.path.join(args.out, "annotated", f"{name}.jpg"), img)

    if not rows:
        raise SystemExit("No frames processed.")

    with open(os.path.join(args.out, "counts.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    avg_inf, avg_tot = statistics.mean(infer_times), statistics.mean(total_times)
    summary = {
        "model": os.path.basename(args.model),
        "model_size_mb": round(os.path.getsize(args.model) / 1e6, 2),
        "device": f"{platform.machine()} / {platform.platform()}",
        "frames_processed": len(rows),
        "conf_threshold": args.conf,
        "totals_across_frames": {n: sum(r[n] for r in rows) for n in names},
        "avg_pedestrians_per_frame": round(statistics.mean(r["pedestrians"] for r in rows), 2),
        "avg_vehicles_per_frame": round(statistics.mean(r["vehicles"] for r in rows), 2),
        "peak_pedestrians": max(r["pedestrians"] for r in rows),
        "peak_vehicles": max(r["vehicles"] for r in rows),
        "risk_frames": {lvl: sum(r["risk"] == lvl for r in rows) for lvl in ("LOW", "MEDIUM", "HIGH")},
        "avg_inference_ms": round(avg_inf, 2),
        "p95_inference_ms": round(float(np.percentile(infer_times, 95)), 2),
        "avg_end_to_end_ms": round(avg_tot, 2),
        "inference_fps": round(1000 / avg_inf, 2),
        "end_to_end_fps": round(1000 / avg_tot, 2),
        "model_load_ms": round(load_ms, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 90)
    print(f"Frames: {summary['frames_processed']}   Peak peds: {summary['peak_pedestrians']}   "
          f"Peak vehicles: {summary['peak_vehicles']}   Risk frames: {summary['risk_frames']}")
    print(f"Avg inference: {avg_inf:.1f} ms ({summary['inference_fps']} FPS)   "
          f"End-to-end incl. pre/post: {avg_tot:.1f} ms ({summary['end_to_end_fps']} FPS)")
    print(f"Results -> {args.out}/counts.csv, {args.out}/summary.json, {args.out}/annotated/")


if __name__ == "__main__":
    main()
