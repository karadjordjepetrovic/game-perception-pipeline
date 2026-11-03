"""Text stage: HUD vs in-world text from stages/text, joined in by frame number.

The text stage runs PaddleOCR in its own environment (stages/text/README), separately from
the PyTorch stages, so it is not loaded into this process. It also runs at a lower rate:
OCR takes ~120-170 ms per 1080p frame, and on-screen text changes slowly, so a few
results per second are enough. That makes this a multi-rate pipeline: depth and objects
every frame, text every Nth frame, its last result shown in between.

Input: outputs/hud/classified.csv from stages/text/scripts/classify_hud.py, produced from
frames extracted with extract_frames.py. Those frames are named frame_0001.png,
frame_0002.png, ... and frame k is video frame (k - 1) * step, step = round(video fps / extract fps).
"""
import csv
import re
from collections import defaultdict

from ..stage import Stage

_FRAME_RE = re.compile(r"frame_(\d+)")


def video_frame(name, step):
    """'frame_0003.png' with step 12 -> video frame 24."""
    match = _FRAME_RE.search(name)
    if not match:
        raise ValueError(f"unexpected frame name: {name}")
    return (int(match.group(1)) - 1) * step


def load_classified(path, step, min_conf=0.0):
    """classified.csv -> {video frame index: [ {box, text, label, confidence}, ... ]}"""
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conf = float(row.get("confidence") or 0)
            if conf < min_conf:
                continue
            by_frame[video_frame(row["frame"], step)].append({
                "box": (float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])),
                "text": row.get("text", ""),
                "label": row.get("label") or "unknown",
                "confidence": conf,
            })
    return dict(by_frame)


class TextStage(Stage):
    name = "text"
    timed = False  # computed in its own process; its timing is in stages/text/README

    def __init__(self, classified_csv, step, min_conf=0.8):
        self.by_frame = load_classified(classified_csv, step, min_conf)
        self.every_n = step

    def process(self, index, frame):
        return self.by_frame.get(index, [])
