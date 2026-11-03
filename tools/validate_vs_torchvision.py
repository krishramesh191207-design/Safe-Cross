"""Sanity check: compare SafeCross (ExecuTorch + NumPy post-proc) counts against the
reference torchvision eager model on the same images. Run on a laptop.

    python tools/validate_vs_torchvision.py --input samples/ --model models/ssdlite_fp32.pte
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime

from safecross.model_def import load_pretrained
from safecross.postprocess import TARGET_CLASSES, postprocess, preprocess


def counts(dets):
    c = {n: 0 for n in TARGET_CLASSES.values()}
    for d in dets:
        c[TARGET_CLASSES[d[-1]]] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="models/ssdlite_fp32.pte")
    ap.add_argument("--anchors", default="models/anchors.npy")
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    det, _ = load_pretrained()
    det.score_thresh = args.conf
    anchors = np.load(args.anchors)
    method = Runtime.get().load_program(args.model).load_method("forward")

    agree = total = 0
    for f in sorted(glob.glob(os.path.join(args.input, "*.jpg"))):
        bgr = cv2.imread(f)
        h, w = bgr.shape[:2]
        rel, sc = method.execute([torch.from_numpy(preprocess(bgr))])
        # merge_vehicles=False: compare like-for-like with torchvision's per-class NMS
        ours = counts(postprocess(rel.numpy(), sc.numpy(), anchors, w, h, conf=args.conf,
                                  merge_vehicles=False))

        x = torch.from_numpy(preprocess(bgr))[0]  # same 320x320 resize for a fair comparison
        with torch.no_grad():
            ref = det([x])[0]
        ref_dets = [(*b.tolist(), s.item(), int(l)) for b, s, l in
                    zip(ref["boxes"], ref["scores"], ref["labels"]) if int(l) in TARGET_CLASSES]
        theirs = counts(ref_dets)
        same = ours == theirs
        agree += same
        total += 1
        diff = {k: (ours[k], theirs[k]) for k in ours if ours[k] != theirs[k]}
        print(f"{os.path.basename(f):<28} {'MATCH' if same else 'DIFF ' + str(diff)}")
    print(f"\n{agree}/{total} images have identical per-class counts (ours vs torchvision)")


if __name__ == "__main__":
    main()
