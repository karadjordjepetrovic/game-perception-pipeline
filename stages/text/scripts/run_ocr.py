"""
Step 1 of the HUD-mask project: run OCR (text detection + recognition)
on a folder of game frames and save what it finds.

Usage (from the project root, with the conda env active):
    python scripts/run_ocr.py
    python scripts/run_ocr.py --device gpu
    python scripts/run_ocr.py --fast --no-images

Input:  data/frames/*.png (or .jpg)
Output: outputs/ocr/images/     -> frames with detected text boxes drawn on them
        outputs/ocr/json/       -> raw results per frame (boxes, text, confidence)
        outputs/ocr/summary.csv -> one row per detected text region

At the end it prints a timing breakdown: model loading, inference, and saving.
"""

import argparse
import csv
import os
import time
from pathlib import Path

# Skip the "checking connectivity to model hosters" step on every run.
# Must be set before paddleocr is imported.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

# --- Paths -------------------------------------------------------------------
# Path(__file__) is this script; .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_DIR = PROJECT_ROOT / "outputs" / "ocr"
IMG_OUT = OUT_DIR / "images"
JSON_OUT = OUT_DIR / "json"

# Only keep text the model is reasonably sure about (0.0 - 1.0).
MIN_CONFIDENCE = 0.5

# Smaller, faster models used with --fast.
FAST_DET_MODEL = "PP-OCRv5_mobile_det"
FAST_REC_MODEL = "PP-OCRv5_mobile_rec"


def parse_args():
    p = argparse.ArgumentParser(description="Run OCR over game frames.")
    p.add_argument("--device", default="cpu", help="'cpu' or 'gpu' (gpu needs paddlepaddle-gpu installed)")
    p.add_argument("--fast", action="store_true",
                   help=f"Use the small mobile models ({FAST_DET_MODEL} / {FAST_REC_MODEL})")
    p.add_argument("--det-size", type=int, default=0,
                   help="Resize the long side of the frame to this many pixels before text detection "
                        "(e.g. 960). Smaller = faster but misses tiny text. 0 = model default")
    p.add_argument("--mkldnn", action="store_true",
                   help="Turn oneDNN CPU acceleration back on (only works on paddlepaddle versions "
                        "without the Windows crash, e.g. 3.2.0)")
    p.add_argument("--no-images", action="store_true", help="Don't save the boxed preview images")
    p.add_argument("--no-json", action="store_true", help="Don't save per-frame JSON files")
    return p.parse_args()


def main():
    args = parse_args()

    IMG_OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    frames = sorted(
        p for p in FRAMES_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frames:
        print(f"No frames found in {FRAMES_DIR}. Run scripts/extract_frames.py first.")
        return
    print(f"Found {len(frames)} frames. Device: {args.device}, fast models: {args.fast}")

    # --- Build the OCR pipeline (timed) ---------------------------------------
    ocr_kwargs = dict(
        lang="en",
        device=args.device,
        # Document-specific extras: game frames are not rotated/warped scans.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # oneDNN (Intel CPU acceleration) is off by default because recent
        # PaddlePaddle CPU builds on Windows crash with it on.
        enable_mkldnn=args.mkldnn,
    )
    if args.fast:
        ocr_kwargs["text_detection_model_name"] = FAST_DET_MODEL
        ocr_kwargs["text_recognition_model_name"] = FAST_REC_MODEL
    if args.det_size > 0:
        ocr_kwargs["text_det_limit_side_len"] = args.det_size
        ocr_kwargs["text_det_limit_type"] = "max"

    t0 = time.perf_counter()
    ocr = PaddleOCR(**ocr_kwargs)
    load_time = time.perf_counter() - t0
    print(f"Model loading: {load_time:.1f} s\n")

    rows = []
    infer_times = []  # seconds per frame spent inside the model
    save_times = []   # seconds per frame spent writing images/json

    for i, frame_path in enumerate(frames, start=1):
        # Inference: the model actually looking at the image.
        t0 = time.perf_counter()
        results = ocr.predict(str(frame_path))
        t_infer = time.perf_counter() - t0
        infer_times.append(t_infer)

        # Saving + collecting rows: everything that isn't the model.
        t0 = time.perf_counter()
        for res in results:
            if not args.no_images:
                res.save_to_img(str(IMG_OUT))
            if not args.no_json:
                res.save_to_json(str(JSON_OUT))

            data = res.json.get("res", res.json)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys = data.get("rec_polys", [])

            for text, score, poly in zip(texts, scores, polys):
                if score < MIN_CONFIDENCE:
                    continue
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                rows.append({
                    "frame": frame_path.name,
                    "text": text,
                    "confidence": round(float(score), 3),
                    "x_min": int(min(xs)), "y_min": int(min(ys)),
                    "x_max": int(max(xs)), "y_max": int(max(ys)),
                })
        t_save = time.perf_counter() - t0
        save_times.append(t_save)

        print(f"[{i}/{len(frames)}] {frame_path.name}: "
              f"inference {t_infer * 1000:.0f} ms, saving {t_save * 1000:.0f} ms")

    summary_path = OUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame", "text", "confidence", "x_min", "y_min", "x_max", "y_max"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # --- Timing report ---------------------------------------------------------
    # The first frame is usually slower (warm-up: memory allocation, kernel
    # selection), so we report the average without it as well.
    def avg_ms(values):
        return sum(values) / len(values) * 1000 if values else 0.0

    steady = infer_times[1:] if len(infer_times) > 1 else infer_times
    total_infer = sum(infer_times)
    total_save = sum(save_times)
    total_loop = total_infer + total_save

    print()
    print(f"Done. {len(rows)} text regions above confidence {MIN_CONFIDENCE}.")
    print("Timing breakdown")
    print(f"  Model loading (once):      {load_time:8.1f} s")
    print(f"  Inference, first frame:    {infer_times[0] * 1000:8.0f} ms  (warm-up)")
    print(f"  Inference, avg after that: {avg_ms(steady):8.0f} ms per frame")
    print(f"  Saving, avg:               {avg_ms(save_times):8.0f} ms per frame")
    print(f"  Share of per-frame time:   inference {total_infer / total_loop * 100:.0f}%, "
          f"saving {total_save / total_loop * 100:.0f}%")
    print(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
