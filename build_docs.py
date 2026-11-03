"""Generate the project report (.docx) and presentation (.pptx) from SafeCross results.

    python build_docs.py --summary output_pi/summary.json --benchmark benchmark_pi.json \
        --annotated output_pi/annotated --counts output_pi/counts.csv --device "Raspberry Pi 5 (8 GB)"

Defaults point at the dev-machine run; re-run with Pi outputs before submission.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor as DocxRGB, Inches as DInches
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt as PPt

NAVY = RGBColor(0x12, 0x2B, 0x4A)
TEAL = RGBColor(0x00, 0x8C, 0x8C)
ORANGE = RGBColor(0xF2, 0x8C, 0x28)
GREY = RGBColor(0x55, 0x5F, 0x6B)
LIGHT = RGBColor(0xF3, 0xF6, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

TITLE = "SafeCross"
SUBTITLE = "Offline Pedestrian & Vehicle Counting at Crossings with ExecuTorch on Raspberry Pi"
EVENT = "Namma AI Hackathon · AI for Safer Communities – Edge-Based Object Detection and Counting"

PROBLEM = [
    "Crossings near schools, markets, hospitals and bus stops mix heavy foot traffic with fast "
    "two-wheelers, autos, cars and buses.",
    "Local bodies rarely have data on how busy a crossing is or when pedestrians and vehicles "
    "conflict, so decisions on zebra crossings, speed breakers, signal timing or wardens are ad hoc.",
    "Camera + cloud video analytics is expensive: GPUs, continuous bandwidth, recurring cost, and "
    "privacy concerns with uploading video of citizens.",
]
SOLUTION = [
    "A ~₹6–8k Raspberry Pi runs a pretrained object detector fully offline using ExecuTorch.",
    "Counts pedestrians and vehicles (car, motorcycle, bicycle, bus, truck) in every frame.",
    "Flags 'conflict frames' where pedestrians are close to vehicles (LOW / MEDIUM / HIGH).",
    "Writes only numbers (CSV + summary) — no video needs to leave the device.",
]
APPLICATION = (
    "School-zone crossing audit: place SafeCross at a school gate for a week. Peak pedestrian "
    "counts and HIGH-risk frames at 8–9 AM and 2–3 PM give the municipality / traffic police "
    "evidence to add a zebra crossing, a speed table, a traffic warden at school hours, or a "
    "longer pedestrian signal phase."
)
LIMITATIONS = [
    "SSDLite at 320×320 misses small/distant objects; dense, occluded Indian traffic will be undercounted.",
    "Counts are per frame, not unique objects (no tracking — not required by the brief).",
    "Risk score is a 2-D box-overlap heuristic; it has no depth or speed information.",
    "COCO has no 'auto-rickshaw' class; autos are usually detected as car/truck or missed.",
]
FUTURE = [
    "Fine-tune on Indian traffic data (e.g., IDD / autos) and re-export with the same pipeline.",
    "Add lightweight tracking (SORT) for unique counts and dwell-time at the kerb.",
    "Solar + 4G-less deployment: store hourly CSV on SD card, collect weekly.",
]


# --------------------------------------------------------------------------- #
def load_inputs(a):
    with open(a.summary) as f:
        summary = json.load(f)
    bench = None
    if a.benchmark and os.path.exists(a.benchmark):
        with open(a.benchmark) as f:
            bench = json.load(f)
    rows = []
    if a.counts and os.path.exists(a.counts):
        with open(a.counts) as f:
            rows = list(csv.DictReader(f))
    return summary, bench, rows


def pick_samples(rows, annotated_dir, n=2):
    """Prefer the highest-risk / busiest frames as sample outputs."""
    order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    rows = sorted(rows, key=lambda r: (order[r["risk"]], int(r["pedestrians"]) + int(r["vehicles"])),
                  reverse=True)
    out = []
    for r in rows:
        p = os.path.join(annotated_dir, r["frame"] + ".jpg")
        if os.path.exists(p):
            out.append((p, r))
        if len(out) == n:
            break
    return out


def make_charts(summary, bench, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    charts = {}
    totals = summary["totals_across_frames"]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    names = list(totals.keys())
    vals = [totals[n] for n in names]
    colors = ["#008C8C" if n == "person" else "#F28C28" for n in names]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=9)
    ax.set_title("Detections per class (summed over all frames)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    charts["classes"] = os.path.join(out_dir, "chart_classes.png")
    fig.savefig(charts["classes"])
    plt.close(fig)

    if rows:
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        x = range(len(rows))
        ax.plot(x, [int(r["pedestrians"]) for r in rows], label="Pedestrians", color="#008C8C", marker="o", ms=3)
        ax.plot(x, [int(r["vehicles"]) for r in rows], label="Vehicles", color="#F28C28", marker="s", ms=3)
        hi = [i for i, r in enumerate(rows) if r["risk"] == "HIGH"]
        for i in hi:
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)
        ax.set_xlabel("Frame / image index")
        ax.set_ylabel("Count")
        ax.set_title("Per-frame counts (red = HIGH-risk frame)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        charts["timeline"] = os.path.join(out_dir, "chart_timeline.png")
        fig.savefig(charts["timeline"])
        plt.close(fig)

    if bench and len(bench["results"]) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.8), dpi=150)
        labels = ["FP32" if "fp32" in r["model"] else "INT8" for r in bench["results"]]
        axes[0].bar(labels, [r["mean_ms"] for r in bench["results"]], color=["#122B4A", "#008C8C"])
        axes[0].set_title("Mean inference (ms)", fontsize=9)
        axes[1].bar(labels, [r["size_mb"] for r in bench["results"]], color=["#122B4A", "#008C8C"])
        axes[1].set_title("Model size (MB)", fontsize=9)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        charts["bench"] = os.path.join(out_dir, "chart_bench.png")
        fig.savefig(charts["bench"])
        plt.close(fig)
    return charts


def perf_rows(summary, bench, device):
    rows = [("Model", "SSDLite320-MobileNetV3-Large (COCO, torchvision) → ExecuTorch .pte, XNNPACK"),
            ("Device", device),
            ("Model file used", f"{summary['model']} ({summary['model_size_mb']} MB)"),
            ("Frames / images processed", str(summary["frames_processed"])),
            ("Avg inference time / frame", f"{summary['avg_inference_ms']} ms (p95 {summary['p95_inference_ms']} ms)"),
            ("Inference throughput", f"{summary['inference_fps']} FPS"),
            ("End-to-end (pre + infer + post)", f"{summary['avg_end_to_end_ms']} ms  ≈ {summary['end_to_end_fps']} FPS")]
    if bench:
        for r in bench["results"]:
            rows.append((f"Benchmark {r['model']}", f"{r['mean_ms']} ms mean · {r['fps']} FPS · {r['size_mb']} MB"))
    return rows


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def build_docx(path, summary, bench, rows, samples, charts, device, dev_note):
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)

    t = d.add_heading(TITLE, 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = d.add_paragraph(SUBTITLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].bold = True
    p = d.add_paragraph(EVENT)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.color.rgb = DocxRGB(0x55, 0x5F, 0x6B)

    if dev_note:
        p = d.add_paragraph()
        r = p.add_run("Note: performance numbers in this version were measured on the development "
                      "machine, not the Raspberry Pi. Re-run benchmark.py / run_safecross.py on the Pi "
                      "and regenerate this document with build_docs.py before submission.")
        r.italic = True
        r.font.color.rgb = DocxRGB(0xC0, 0x39, 0x2B)

    d.add_heading("1. Executive Summary", 1)
    d.add_paragraph(
        "SafeCross is an edge AI proof of concept that counts pedestrians and vehicles in video "
        "frames or images on a Raspberry Pi, using a COCO-pretrained SSDLite-MobileNetV3 detector "
        "exported to ExecuTorch with the XNNPACK CPU backend. It runs fully offline, outputs "
        "per-frame counts as CSV plus annotated images, and flags frames where pedestrians are "
        "close to vehicles as a simple safety indicator for crossings. The INT8 model is "
        f"{summary['model_size_mb']} MB and averages {summary['avg_inference_ms']} ms per frame on {device}.")

    d.add_heading("2. Societal Problem", 1)
    for s in PROBLEM:
        d.add_paragraph(s, style="List Bullet")

    d.add_heading("3. Proposed Solution", 1)
    for s in SOLUTION:
        d.add_paragraph(s, style="List Bullet")
    d.add_heading("Practical application", 2)
    d.add_paragraph(APPLICATION)

    d.add_heading("4. System Design", 1)
    d.add_paragraph("Pipeline:", style="Normal").runs[0].bold = True
    for s in ["Input: organiser video clip (10–20 s) or folder of 10–20 images (OpenCV).",
              "Pre-processing: resize to 320×320, BGR→RGB, scale to [0,1].",
              "Inference: ExecuTorch runtime executes ssdlite_int8.pte / ssdlite_fp32.pte, fully delegated to XNNPACK.",
              "Outputs: 3,234 anchor box regressions + 91-class softmax scores.",
              "Post-processing (NumPy): SSD box decoding, per-class NMS (IoU 0.55), confidence ≥ 0.4, "
              "cross-class vehicle de-duplication (e.g., one van labelled car + truck).",
              "Analytics: counts per class; pedestrian–vehicle proximity pairs; per-frame risk level.",
              "Reporting: counts.csv, annotated/*.jpg, summary.json (counts + latency + FPS)."]:
        d.add_paragraph(s, style="List Number")
    d.add_heading("Why these choices", 2)
    for s in ["SSDLite-MobileNetV3 is a small, CPU-friendly detector (~3.4 M params) with a COCO "
              "checkpoint covering person, bicycle, car, motorcycle, bus and truck — no training needed.",
              "ExecuTorch + XNNPACK gives optimised ARM NEON kernels for the Pi's Cortex-A CPU and a "
              "self-contained .pte file with a lightweight runtime.",
              "PT2E static INT8 quantisation (per-channel, calibrated on sample images) shrinks the model "
              "3.6× with identical counts on our validation set.",
              "Box decoding and NMS are kept outside the graph in NumPy, so the exported graph is static "
              "and fully delegated."]:
        d.add_paragraph(s, style="List Bullet")
    d.add_heading("Risk logic (per frame)", 2)
    for s in ["Proximity pair: a pedestrian box enlarged 1.5× overlaps a vehicle box.",
              "HIGH: ≥2 pairs, or ≥3 pedestrians and ≥2 vehicles with ≥1 pair.",
              "MEDIUM: 1 pair, or crowded mix (≥3 pedestrians and ≥2 vehicles).",
              "LOW: otherwise."]:
        d.add_paragraph(s, style="List Bullet")

    d.add_heading("5. Results", 1)
    d.add_heading("Category and counts", 2)
    tot = summary["totals_across_frames"]
    tbl = d.add_table(rows=1, cols=2)
    tbl.style = "Light Grid Accent 1"
    tbl.rows[0].cells[0].text, tbl.rows[0].cells[1].text = "Metric", "Value"
    for k, v in [("Categories", "Pedestrians (person); Vehicles (bicycle, car, motorcycle, bus, truck)"),
                 *[(f"Total '{n}' detections", str(c)) for n, c in tot.items()],
                 ("Avg pedestrians / frame", str(summary["avg_pedestrians_per_frame"])),
                 ("Avg vehicles / frame", str(summary["avg_vehicles_per_frame"])),
                 ("Peak pedestrians / vehicles", f"{summary['peak_pedestrians']} / {summary['peak_vehicles']}"),
                 ("Risk frames (LOW / MEDIUM / HIGH)",
                  " / ".join(str(summary["risk_frames"][x]) for x in ("LOW", "MEDIUM", "HIGH")))]:
        c = tbl.add_row().cells
        c[0].text, c[1].text = k, v
    if "classes" in charts:
        d.add_picture(charts["classes"], width=DInches(5.5))
    if "timeline" in charts:
        d.add_picture(charts["timeline"], width=DInches(5.5))

    if rows:
        d.add_heading("Per-frame counts", 2)
        cols = ["frame", "pedestrians", "vehicles", "proximity_pairs", "risk", "infer_ms"]
        t2 = d.add_table(rows=1, cols=len(cols))
        t2.style = "Light Grid Accent 1"
        for i, c in enumerate(cols):
            t2.rows[0].cells[i].text = c
        for r in rows[:25]:
            cells = t2.add_row().cells
            for i, c in enumerate(cols):
                cells[i].text = str(r[c])
        if len(rows) > 25:
            d.add_paragraph(f"… {len(rows) - 25} more rows in counts.csv")

    d.add_heading("Sample detection outputs", 2)
    for pth, r in samples:
        d.add_picture(pth, width=DInches(5.0))
        cap = d.add_paragraph(f"{r['frame']}: {r['pedestrians']} pedestrians, {r['vehicles']} vehicles, "
                              f"risk {r['risk']}, {r['infer_ms']} ms inference")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    d.add_heading("6. Performance", 1)
    t3 = d.add_table(rows=0, cols=2)
    t3.style = "Light Grid Accent 1"
    for k, v in perf_rows(summary, bench, device):
        c = t3.add_row().cells
        c[0].text, c[1].text = k, v
    if "bench" in charts:
        d.add_picture(charts["bench"], width=DInches(5.0))
    d.add_heading("Correctness check", 2)
    d.add_paragraph("tools/validate_vs_torchvision.py compares SafeCross counts (ExecuTorch + NumPy "
                    "post-processing) against the reference torchvision model on the same inputs. "
                    "On our 11-image validation set both FP32 and INT8 produced identical per-class "
                    "counts to the reference (11/11).")

    d.add_heading("7. Limitations", 1)
    for s in LIMITATIONS:
        d.add_paragraph(s, style="List Bullet")
    d.add_heading("8. Future Work", 1)
    for s in FUTURE:
        d.add_paragraph(s, style="List Bullet")

    d.add_heading("9. How to Reproduce", 1)
    for s in ["Laptop: pip install -r requirements-dev.txt; python export_model.py --calib-dir samples/",
              "Copy models/, safecross/, run_safecross.py, benchmark.py to the Pi; run ./setup_pi.sh",
              "Pi: python run_safecross.py --input clip.mp4 --out output_pi",
              "Pi: python benchmark.py --out benchmark_pi.json",
              "Regenerate docs: python build_docs.py --summary output_pi/summary.json --benchmark "
              "benchmark_pi.json --annotated output_pi/annotated --counts output_pi/counts.csv "
              "--device \"Raspberry Pi 5\""]:
        d.add_paragraph(s, style="List Number")

    d.add_heading("10. 2-Minute Pitch Script", 1)
    for s in PITCH:
        d.add_paragraph(s)
    d.save(path)


PITCH = [
    "[0:00–0:25] Problem — Every day, children cross busy roads outside schools, and commuters dodge "
    "two-wheelers at bus stops. Cities want to fix these spots, but they don't have data on how many "
    "people cross, or when pedestrians and vehicles collide in space. Cloud video analytics is costly "
    "and needs constant internet.",
    "[0:25–0:55] Solution — SafeCross runs entirely on a Raspberry Pi. A pretrained SSDLite-MobileNetV3 "
    "detector, exported with ExecuTorch and quantised to INT8, counts pedestrians and vehicles in every "
    "frame and flags frames where people are close to vehicles. No video leaves the device — only counts.",
    "[0:55–1:25] Results — On the supplied clip we detect people, cars, two-wheelers, buses and trucks, "
    "with per-frame counts in a CSV and annotated images. The INT8 model is under 4 MB and runs at "
    "<X> ms per frame (<Y> FPS) on the Pi. Our counts match the reference PyTorch model exactly.",
    "[1:25–2:00] Impact — Put one at a school gate for a week: peak crossing counts and high-risk "
    "frames at drop-off time become evidence for a zebra crossing, speed table, or traffic warden. "
    "Affordable, private, offline — safer crossings, driven by data.",
]


# --------------------------------------------------------------------------- #
# PPTX
# --------------------------------------------------------------------------- #
def _bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _text(slide, x, y, w, h, text, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.size = PPt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = FONT
    return tf


def _bullets(slide, x, y, w, h, items, size=16, color=NAVY):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + it
        p.font.size = PPt(size)
        p.font.color.rgb = color
        p.font.name = FONT
        p.space_after = PPt(8)
    return tf


def _header(slide, title, num):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    _text(slide, 0.5, 0.18, 11, 0.7, title, 28, True, WHITE)
    _text(slide, 12.2, 7.0, 1, 0.4, str(num), 11, False, GREY, PP_ALIGN.RIGHT)
    _text(slide, 0.5, 7.0, 8, 0.4, "SafeCross · Namma AI Hackathon", 11, False, GREY)


def _card(slide, x, y, w, h, title, body, accent=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = accent
    box.adjustments[0] = 0.08
    _text(slide, x + 0.2, y + 0.15, w - 0.4, 0.5, title, 18, True, accent)
    _text(slide, x + 0.2, y + 0.7, w - 0.4, h - 0.8, body, 14, False, NAVY)


def build_pptx(path, summary, bench, rows, samples, charts, device, dev_note):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    n = 0

    # 1 Title
    s = prs.slides.add_slide(blank); n += 1
    _bg(s, NAVY)
    _text(s, 0.8, 2.2, 11.7, 1.2, TITLE, 60, True, WHITE)
    _text(s, 0.8, 3.4, 11.7, 1.0, SUBTITLE, 24, False, RGBColor(0xCF, 0xE8, 0xE8))
    acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.6), Inches(2.5), Inches(0.08))
    acc.fill.solid(); acc.fill.fore_color.rgb = ORANGE; acc.line.fill.background()
    _text(s, 0.8, 4.9, 11.7, 0.6, EVENT, 16, False, WHITE)
    _text(s, 0.8, 5.5, 11.7, 0.6, "Team: <names>", 16, False, WHITE)

    # 2 Problem
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "The Problem: Unsafe Crossings, No Data", n)
    _bullets(s, 0.6, 1.4, 7.4, 5, PROBLEM, 18)
    _card(s, 8.4, 1.5, 4.4, 2.2, "Who is at risk", "School children, elderly, street vendors and commuters at "
          "unmarked crossings, bus stops and market roads.", ORANGE)
    _card(s, 8.4, 4.0, 4.4, 2.4, "Why not cloud CCTV?", "GPU servers, continuous bandwidth, recurring cost — "
          "and privacy concerns about streaming citizens' video.", ORANGE)

    # 3 Solution
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Our Solution: SafeCross on a ₹6–8k Raspberry Pi", n)
    _bullets(s, 0.6, 1.4, 7.0, 5, SOLUTION, 18)
    for i, (t, b) in enumerate([("Offline", "No cloud, no GPU. Runs on Pi CPU via ExecuTorch."),
                                ("Private", "Only numbers leave the device — never video."),
                                ("Actionable", "Per-frame counts + conflict flags for planners.")]):
        _card(s, 8.0, 1.4 + i * 1.85, 4.8, 1.65, t, b)

    # 4 Architecture
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "System Pipeline", n)
    steps = [("Input", "Video clip /\nimages"), ("Pre-process", "320×320 RGB\n[0,1]"),
             ("ExecuTorch", "SSDLite-MBv3\nINT8 · XNNPACK"), ("Post-process", "Box decode\nNMS · de-dup"),
             ("Analytics", "Counts · pairs\nRisk level"), ("Output", "CSV · images\nsummary.json")]
    w, gap, y = 1.85, 0.27, 2.0
    for i, (t, b) in enumerate(steps):
        x = 0.5 + i * (w + gap)
        col = TEAL if t == "ExecuTorch" else NAVY
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(1.9))
        box.fill.solid(); box.fill.fore_color.rgb = col; box.line.fill.background()
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = t; p.font.size = PPt(16); p.font.bold = True
        p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER; p.font.name = FONT
        p2 = tf.add_paragraph(); p2.text = b; p2.font.size = PPt(12); p2.font.color.rgb = WHITE; p2.font.name = FONT
        p2.alignment = PP_ALIGN.CENTER
        if i < len(steps) - 1:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + w + 0.02), Inches(y + 0.8),
                                    Inches(gap - 0.04), Inches(0.3))
            ar.fill.solid(); ar.fill.fore_color.rgb = ORANGE; ar.line.fill.background()
    _bullets(s, 0.6, 4.3, 12, 2.6, [
        "Laptop: torch.export → PT2E INT8 quantisation (calibrated on sample frames) → XNNPACK partitioner → .pte",
        "Pi: ExecuTorch runtime loads .pte; NumPy handles anchor decoding + NMS (keeps graph static, fully delegated)",
        "Risk: pedestrian box ×1.5 overlapping a vehicle = proximity pair → LOW / MEDIUM / HIGH per frame"], 15)

    # 5 Model & optimisation
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Model & Edge Optimisation", n)
    _bullets(s, 0.6, 1.4, 6.4, 5, [
        "SSDLite320-MobileNetV3-Large, COCO-pretrained (torchvision)",
        "Classes used: person · bicycle · car · motorcycle · bus · truck",
        "ExecuTorch with XNNPACK backend (ARM NEON kernels)",
        "PT2E static INT8, per-channel weights",
        f"Size: 13.8 MB FP32 → {summary['model_size_mb']} MB INT8",
        "Validated: counts identical to PyTorch reference (11/11 images)"], 17)
    if "bench" in charts:
        s.shapes.add_picture(charts["bench"], Inches(7.2), Inches(1.6), width=Inches(5.8))

    # 6 Results
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Results: Categories & Counts", n)
    if "timeline" in charts:
        s.shapes.add_picture(charts["timeline"], Inches(0.4), Inches(1.3), width=Inches(6.4))
    if "classes" in charts:
        s.shapes.add_picture(charts["classes"], Inches(6.9), Inches(1.3), width=Inches(6.1))
    rf = summary["risk_frames"]
    stats = [(str(summary["frames_processed"]), "frames processed"),
             (f"{summary['peak_pedestrians']} / {summary['peak_vehicles']}", "peak peds / vehicles"),
             (f"{summary['avg_pedestrians_per_frame']} / {summary['avg_vehicles_per_frame']}", "avg peds / vehicles"),
             (f"{rf['HIGH']} · {rf['MEDIUM']}", "HIGH · MEDIUM risk frames")]
    for i, (v, l) in enumerate(stats):
        x = 0.6 + i * 3.1
        _text(s, x, 5.0, 2.9, 0.8, v, 32, True, TEAL, PP_ALIGN.CENTER)
        _text(s, x, 5.8, 2.9, 0.5, l, 14, False, GREY, PP_ALIGN.CENTER)

    # 7 Samples
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Sample Detection Outputs", n)
    for i, (pth, r) in enumerate(samples[:2]):
        from PIL import Image
        iw, ih = Image.open(pth).size
        maxw, maxh = 6.0, 5.0
        sc = min(maxw / iw, maxh / ih)
        x = 0.5 + i * 6.4 + (maxw - iw * sc) / 2
        s.shapes.add_picture(pth, Inches(x), Inches(1.25), width=Inches(iw * sc), height=Inches(ih * sc))
        _text(s, 0.5 + i * 6.4, 6.35, 6.0, 0.5,
              f"{r['pedestrians']} pedestrians · {r['vehicles']} vehicles · risk {r['risk']} · {r['infer_ms']} ms",
              13, False, GREY, PP_ALIGN.CENTER)

    # 8 Performance
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Performance on Edge", n)
    pr = perf_rows(summary, bench, device)
    tbl = s.shapes.add_table(len(pr), 2, Inches(0.6), Inches(1.4), Inches(12.1), Inches(0.45 * len(pr))).table
    tbl.columns[0].width, tbl.columns[1].width = Inches(4.0), Inches(8.1)
    for i, (k, v) in enumerate(pr):
        for j, t in enumerate((k, v)):
            c = tbl.cell(i, j); c.text = t
            c.text_frame.paragraphs[0].font.size = PPt(14)
            c.text_frame.paragraphs[0].font.bold = j == 0
            c.text_frame.paragraphs[0].font.name = FONT
    if dev_note:
        _text(s, 0.6, 6.3, 12, 0.5, "⚠ Measured on development machine — replace with Raspberry Pi numbers "
              "(run benchmark.py on the Pi, then build_docs.py).", 14, True, RGBColor(0xC0, 0x39, 0x2B))

    # 9 Impact
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Practical Application: School-Zone Crossing Audit", n)
    _text(s, 0.6, 1.4, 12, 1.6, APPLICATION, 18)
    for i, (t, b) in enumerate([("Measure", "Pi at school gate for 1 week → hourly pedestrian & vehicle counts"),
                                ("Identify", "Peak hours + HIGH-risk frames at drop-off / pick-up"),
                                ("Act", "Zebra crossing · speed table · warden · longer walk signal")]):
        _card(s, 0.6 + i * 4.2, 3.6, 3.9, 2.4, t, b, [TEAL, ORANGE, NAVY][i])

    # 10 Limitations & next steps
    s = prs.slides.add_slide(blank); n += 1
    _header(s, "Limitations & Next Steps", n)
    _text(s, 0.6, 1.3, 6, 0.5, "Limitations", 20, True, ORANGE)
    _bullets(s, 0.6, 1.9, 6.0, 5, LIMITATIONS, 15)
    _text(s, 7.0, 1.3, 6, 0.5, "Next steps", 20, True, TEAL)
    _bullets(s, 7.0, 1.9, 5.8, 5, FUTURE, 15)

    # 11 Thanks
    s = prs.slides.add_slide(blank); n += 1
    _bg(s, NAVY)
    _text(s, 0.8, 2.6, 11.7, 1.2, "Safer crossings, driven by data.", 44, True, WHITE)
    _text(s, 0.8, 3.8, 11.7, 0.8, "Affordable · Offline · Private — SafeCross on Raspberry Pi + ExecuTorch",
          20, False, RGBColor(0xCF, 0xE8, 0xE8))
    _text(s, 0.8, 5.0, 11.7, 0.6, "Thank you · Questions?", 20, False, ORANGE)
    prs.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="output_images/summary.json")
    ap.add_argument("--counts", default="output_images/counts.csv")
    ap.add_argument("--annotated", default="output_images/annotated")
    ap.add_argument("--benchmark", default="benchmark.json")
    ap.add_argument("--device", default=None, help='e.g. "Raspberry Pi 5 (8 GB)"')
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()

    summary, bench, rows = load_inputs(a)
    dev_note = a.device is None or "raspberry" not in a.device.lower()
    device = a.device or f"Development machine ({summary['device']}) — NOT the Pi"
    os.makedirs(a.out, exist_ok=True)
    charts = make_charts(summary, bench, rows, os.path.join(a.out, "assets"))
    samples = pick_samples(rows, a.annotated)
    build_docx(os.path.join(a.out, "SafeCross_Report.docx"), summary, bench, rows, samples, charts, device, dev_note)
    build_pptx(os.path.join(a.out, "SafeCross_Presentation.pptx"), summary, bench, rows, samples, charts, device, dev_note)
    print(f"wrote {a.out}/SafeCross_Report.docx and {a.out}/SafeCross_Presentation.pptx")


if __name__ == "__main__":
    main()
