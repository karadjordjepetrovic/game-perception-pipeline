"""
Copy a few OCR results into examples/ so they can be committed and shown in the README.

Boxed preview images are converted to JPG and downscaled (much smaller than the
full-size PNGs), and outputs/ocr/summary.csv is copied as-is.

Usage (from the project root, with the conda env active):
    python scripts/make_examples.py frame_0035 frame_0042

The first frame becomes examples/example_1.jpg, the second example_2.jpg, and so on.
"""

import argparse
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "outputs" / "ocr" / "images"
SUMMARY = PROJECT_ROOT / "outputs" / "ocr" / "summary.csv"
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def main():
    parser = argparse.ArgumentParser(description="Copy sample results into examples/.")
    parser.add_argument("frames", nargs="+", help="Frame names without extension, e.g. frame_0035")
    parser.add_argument("--width", type=int, default=1280, help="Width of the saved JPGs")
    args = parser.parse_args()

    EXAMPLES_DIR.mkdir(exist_ok=True)

    for n, name in enumerate(args.frames, start=1):
        # PaddleOCR names its preview images after the frame, plus a suffix.
        matches = sorted(IMAGES_DIR.glob(f"{name}*"))
        if not matches:
            print(f"No preview image found for {name} in {IMAGES_DIR}")
            continue

        img = cv2.imread(str(matches[0]))
        h, w = img.shape[:2]
        if w > args.width:
            img = cv2.resize(img, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)

        out = EXAMPLES_DIR / f"example_{n}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{matches[0].name} -> {out.name} ({out.stat().st_size // 1024} KB)")

    if SUMMARY.exists():
        shutil.copy(SUMMARY, EXAMPLES_DIR / "summary.csv")
        print("summary.csv -> examples/summary.csv")


if __name__ == "__main__":
    main()
