"""Build a ~15 s test video from a folder of images (slow pan/zoom over each image),
so the video path can be tested before the organisers' clip is available.

    python tools/make_test_clip.py --images samples/ --out test_clip.mp4
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="test_clip.mp4")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    W, H = 960, 540
    per = int(args.seconds * args.fps / len(files))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in files:
        img = cv2.resize(cv2.imread(f), (W, H))
        for i in range(per):
            z = 1 + 0.15 * i / per  # gentle zoom-in
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
            vw.write(cv2.warpAffine(img, M, (W, H)))
    vw.release()
    print(f"wrote {args.out}: {per * len(files)} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
