"""Benchmark FP32 vs INT8 ExecuTorch models on this device (run it ON the Raspberry Pi).

    python benchmark.py                 # 50 timed runs per model
    python benchmark.py --threads 4     # (XNNPACK uses all cores by default)
"""
import argparse
import json
import os
import platform
import statistics
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from executorch.runtime import Runtime


def bench(path, runs, warmup):
    method = Runtime.get().load_program(path).load_method("forward")
    x = torch.rand(1, 3, 320, 320)
    for _ in range(warmup):
        method.execute([x])
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        method.execute([x])
        t.append((time.perf_counter() - s) * 1000)
    return {"model": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "mean_ms": round(statistics.mean(t), 2), "median_ms": round(statistics.median(t), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2), "fps": round(1000 / statistics.mean(t), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["models/ssdlite_fp32.pte", "models/ssdlite_int8.pte"])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    device = f"{platform.machine()} | {platform.platform()}"
    try:
        with open("/proc/device-tree/model") as f:
            device = f.read().strip("\x00") + " | " + device
    except OSError:
        pass

    res = [bench(m, args.runs, args.warmup) for m in args.models if os.path.exists(m)]
    print(f"Device: {device}\n")
    print(f"{'model':<22}{'size MB':>9}{'mean ms':>10}{'median':>9}{'p95':>9}{'FPS':>8}")
    for r in res:
        print(f"{r['model']:<22}{r['size_mb']:>9}{r['mean_ms']:>10}{r['median_ms']:>9}{r['p95_ms']:>9}{r['fps']:>8}")
    if len(res) == 2:
        print(f"\nINT8 speed-up: {res[0]['mean_ms'] / res[1]['mean_ms']:.2f}x   "
              f"size reduction: {res[0]['size_mb'] / res[1]['size_mb']:.1f}x")
    with open(args.out, "w") as f:
        json.dump({"device": device, "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
