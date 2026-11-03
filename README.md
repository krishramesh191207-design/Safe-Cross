# SafeCross — Offline Pedestrian & Vehicle Counting on Raspberry Pi with ExecuTorch

Namma AI Hackathon · *AI for Safer Communities – Edge-Based Object Detection and Counting*

SafeCross runs a COCO-pretrained **SSDLite320-MobileNetV3-Large** detector through
**ExecuTorch (XNNPACK backend)** on a Raspberry Pi. It counts **pedestrians** and
**vehicles** (car, motorcycle, bicycle, bus, truck) in each frame of a video or image set,
flags frames where pedestrians and vehicles are close together, and reports latency and FPS.
It needs no cloud and no GPU.

## Pipeline

```
frame (BGR) ─► resize 320x320, RGB, [0,1] ─► ExecuTorch .pte (XNNPACK, FP32 or INT8)
          ─► box regression + class scores (3234 anchors × 91 classes)
          ─► NumPy: SSD box decode ─► per-class NMS ─► vehicle de-dup
          ─► counts per class ─► proximity pairs ─► risk level (LOW/MEDIUM/HIGH)
          ─► counts.csv · annotated/*.jpg · summary.json
```

Anchor decoding and NMS run in NumPy outside the model. They depend on the data, so they
don't lower cleanly to XNNPACK. Keeping them out also keeps the `.pte` small and fully delegated.

## Layout

| Path | Purpose |
|---|---|
| `export_model.py` | **Laptop.** Exports FP32 and INT8 (PT2E static quantisation) `.pte` files, plus `anchors.npy` |
| `run_safecross.py` | **Pi.** Main runner: detect → count → risk → CSV / annotated images / summary |
| `benchmark.py` | **Pi.** FP32 vs INT8 latency / FPS / size table |
| `safecross/model_def.py` | Wraps torchvision SSDLite into a tensor-only module for export |
| `safecross/postprocess.py` | Pre-processing, box decoding, NMS, risk logic, drawing (NumPy + OpenCV only) |
| `tools/validate_vs_torchvision.py` | Checks that ExecuTorch + NumPy counts match the torchvision reference |
| `tools/make_test_clip.py` | Builds a test video from images, for use before the official clip arrives |
| `build_docs.py` | Regenerates `docs/SafeCross_Report.docx` and `docs/SafeCross_Presentation.pptx` from results |

## 1. Export (on a laptop)

```bash
pip install -r requirements-dev.txt
python export_model.py --calib-dir path/to/sample_images     # -> models/
python tools/validate_vs_torchvision.py --input path/to/sample_images
```

Copy the `models/` folder, `safecross/`, `run_safecross.py` and `benchmark.py` to the Pi.

## 2. Run (on the Raspberry Pi)

```bash
./setup_pi.sh && source .venv/bin/activate
python run_safecross.py --input organiser_clip.mp4 --out output_pi      # INT8 by default
python run_safecross.py --input organiser_images/ --out output_pi_imgs
python run_safecross.py --input organiser_clip.mp4 --model models/ssdlite_fp32.pte --out output_pi_fp32
python benchmark.py --out benchmark_pi.json
```

Useful flags: `--conf 0.4` (detection threshold), `--every 3` (process every 3rd video frame),
`--no-save` (skip writing annotated images).

## 3. Regenerate report and slides with Pi numbers

```bash
python build_docs.py --summary output_pi/summary.json --benchmark benchmark_pi.json \
                     --annotated output_pi/annotated --device "Raspberry Pi 5 (8 GB)"
```

## Risk logic (per frame)

- **Proximity pair**: a pedestrian box (enlarged 1.5×) overlaps a vehicle box.
- **HIGH**: 2 or more proximity pairs, or (3+ pedestrians, 2+ vehicles and at least 1 pair).
- **MEDIUM**: 1 pair, or a crowded mix (3+ pedestrians and 2+ vehicles).
- **LOW**: everything else.

This is a 2-D heuristic, not real distance or speed. It shows which moments are worth a
human looking at.

## Validated results (development machine)

- ExecuTorch + NumPy post-processing gives **the same per-class counts as torchvision's
  reference model on 11/11 test images**, for both FP32 and INT8.
- INT8 model: **3.9 MB** vs 13.8 MB FP32 (3.6× smaller).
- Latency on the dev machine (Apple M-series, arm64): ~5.5 ms per frame. **The Pi will be
  slower.** Run `benchmark.py` on the Pi and report those numbers in the submission.

## Limitations

- SSDLite at 320×320 misses small or distant objects. Dense Indian traffic with heavy
  occlusion will undercount.
- Counts are per frame, not unique objects (no tracking, as the brief allows).
- The risk score is a 2-D overlap heuristic with no depth or speed information.
