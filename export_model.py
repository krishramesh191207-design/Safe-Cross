"""Export SSDLite320-MobileNetV3 to ExecuTorch (.pte) with the XNNPACK backend.

Run on a laptop/desktop (x86 or ARM); copy the resulting files in models/ to the Pi.

    python export_model.py                       # FP32 + INT8
    python export_model.py --calib-dir samples/  # use real images for INT8 calibration
"""
import argparse
import glob
import json
import os
import warnings

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from safecross.model_def import INPUT_SIZE, SSDLiteCore, compute_anchors, load_pretrained
from safecross.postprocess import preprocess


def lower_and_save(exported, path):
    prog = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(path, "wb") as f:
        f.write(prog.buffer)
    print(f"  saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def calibration_batches(calib_dir, n=32):
    files = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")) +
                   glob.glob(os.path.join(calib_dir, "*.png"))) if calib_dir else []
    if files:
        print(f"  calibrating INT8 on {len(files[:n])} real images from {calib_dir}")
        for f in files[:n]:
            yield torch.from_numpy(preprocess(cv2.imread(f)))
    else:
        print("  WARNING: no calibration images; using random noise (lower INT8 accuracy)")
        for _ in range(8):
            yield torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--calib-dir", default=None, help="folder of jpg/png for INT8 calibration")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    det, categories = load_pretrained()
    core = SSDLiteCore(det).eval()
    example = (torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE),)

    np.save(os.path.join(args.out, "anchors.npy"), compute_anchors(det))
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(categories, f)

    print("[1/2] FP32 export")
    with torch.no_grad():
        ep = torch.export.export(core, example)
    lower_and_save(ep, os.path.join(args.out, "ssdlite_fp32.pte"))

    if args.skip_int8:
        return
    print("[2/2] INT8 (PT2E static quantization) export")
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    with torch.no_grad():
        graph = torch.export.export(core, example).module()
        prepared = prepare_pt2e(graph, quantizer)
        for x in calibration_batches(args.calib_dir):
            prepared(x)
        quantized = convert_pt2e(prepared)
        ep_q = torch.export.export(quantized, example)
    lower_and_save(ep_q, os.path.join(args.out, "ssdlite_int8.pte"))


if __name__ == "__main__":
    main()
