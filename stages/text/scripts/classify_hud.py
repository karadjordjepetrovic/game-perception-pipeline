"""
Step 2 of the HUD-mask project: decide which detected text is HUD and which is
part of the game world, using how each text box behaves across frames.

Idea:
  - HUD text is glued to the SCREEN: it stays at the same pixel position while
    the camera moves (objective, subtitles, ammo counter).
  - In-world text is glued to the WORLD: when the camera moves, the same text
    shows up at a different screen position (signs, posters).

Rules (applied to outputs/ocr/summary.csv):
  1. Group detections at the same screen position across frames into "tracks"
     (boxes that overlap strongly). The text inside may change, e.g. an ammo
     counter 30 -> 29, so we match on position only.
  2. HUD: a track that stays in place across at least MIN_HUD_FRAMES frames
     while the rest of the picture changes (the camera moved). If the camera
     did not move, everything looks static, so that frame pair proves nothing.
  3. WORLD: the same text (fuzzy match) appears in nearby frames at a clearly
     different position, i.e. it moved with the scene.
  4. UNKNOWN: seen only once, or never seen with camera motion. We cannot tell.

Usage (from the project root, with the conda env active):
    python scripts/classify_hud.py
    python scripts/classify_hud.py --csv examples/summary.csv

Output:
  outputs/hud/classified.csv  -> every detection with a label (hud/world/unknown) and the reason
  outputs/hud/images/         -> frames with boxes: green = HUD, red = world, grey = unknown
  outputs/hud/masks/          -> black/white HUD mask per frame (white = HUD text)
"""

import argparse
import csv
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_DIR = PROJECT_ROOT / "outputs" / "hud"

# --- Tunable thresholds ---------------------------------------------------------
MIN_CONFIDENCE = 0.8      # ignore weak detections (mostly textures misread as text)
SAME_POS_IOU = 0.5        # boxes overlapping at least this much = "same screen position"
MIN_HUD_FRAMES = 2        # a HUD track must be seen in place in at least this many frames
SCENE_CHANGE = 8.0        # mean pixel difference (0-255) above which we say "the camera moved"
MOVED_PX = 20             # same text whose centre moved more than this = it moved with the world
TEXT_SIMILARITY = 0.8     # how alike two strings must be to count as "the same text" (0-1)
MIN_TEXT_LEN = 4          # short strings ("N", "3") are too ambiguous for text matching
NEARBY_FRAMES = 2         # look for the same text up to this many frames apart
MASK_PADDING = 4          # grow each HUD box by this many pixels in the mask


# --- Small helpers --------------------------------------------------------------
def iou(a, b):
    """Intersection over Union: how much two boxes overlap, 0 (none) to 1 (identical)."""
    x1, y1 = max(a["x_min"], b["x_min"]), max(a["y_min"], b["y_min"])
    x2, y2 = min(a["x_max"], b["x_max"]), min(a["y_max"], b["y_max"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a["x_max"] - a["x_min"]) * (a["y_max"] - a["y_min"])
    area_b = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centre(d):
    return ((d["x_min"] + d["x_max"]) / 2, (d["y_min"] + d["y_max"]) / 2)


def distance(a, b):
    (ax, ay), (bx, by) = centre(a), centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def rigid_vertical_shift(a, b):
    """
    True if box b is box a moved straight up/down with the same left/right edges
    and the same size. That is what a HUD layout shift looks like (a new subtitle
    line pushes the old one up). Text in the 3D world almost never moves like this
    as the camera turns, because its size and horizontal position change too.
    """
    same_x = abs(a["x_min"] - b["x_min"]) <= 4 and abs(a["x_max"] - b["x_max"]) <= 4
    ha, hb = a["y_max"] - a["y_min"], b["y_max"] - b["y_min"]
    # OCR box heights wobble a few pixels between frames, so allow 25%.
    same_height = abs(ha - hb) <= max(3, 0.25 * max(ha, hb))
    return same_x and same_height


def similar_text(a, b):
    a, b = a.strip().upper(), b.strip().upper()
    if len(a) < MIN_TEXT_LEN or len(b) < MIN_TEXT_LEN:
        return False
    return SequenceMatcher(None, a, b).ratio() >= TEXT_SIMILARITY


def load_small_gray(path):
    """Load a frame as a small greyscale image; enough to measure camera motion cheaply."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    return cv2.resize(img, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32)


# --- Main steps -----------------------------------------------------------------
def load_detections(csv_path):
    detections = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["confidence"]) < MIN_CONFIDENCE:
                continue
            for key in ("x_min", "y_min", "x_max", "y_max"):
                row[key] = int(row[key])
            row["confidence"] = float(row["confidence"])
            detections.append(row)
    return detections


def measure_scene_change(frame_names):
    """For each pair of consecutive frames, how much did the picture change?"""
    change = {}
    if not FRAMES_DIR.exists():
        return change
    prev_name, prev_img = None, None
    for name in frame_names:
        img = load_small_gray(FRAMES_DIR / name)
        if img is not None and prev_img is not None and img.shape == prev_img.shape:
            change[(prev_name, name)] = float(np.mean(np.abs(img - prev_img)))
        prev_name, prev_img = name, img
    return change


def build_tracks(detections, frame_index):
    """Link detections in consecutive frames that sit at the same screen position."""
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d["frame"]].append(d)

    next_id = 0
    frames = sorted(by_frame, key=lambda n: frame_index[n])
    for i, name in enumerate(frames):
        prev = by_frame[frames[i - 1]] if i > 0 else []
        # Only link to the directly previous frame that had detections, and only if
        # it is the neighbouring frame (a gap means we lost the text in between).
        prev_is_neighbour = i > 0 and frame_index[name] - frame_index[frames[i - 1]] == 1
        for d in by_frame[name]:
            best = None
            if prev_is_neighbour:
                candidates = [(iou(d, p), p) for p in prev]
                candidates = [c for c in candidates if c[0] >= SAME_POS_IOU]
                if candidates:
                    best = max(candidates, key=lambda c: c[0])[1]
            if best is not None:
                d["track"] = best["track"]
            else:
                d["track"] = next_id
                next_id += 1
    return by_frame


def classify(detections, by_frame, frame_index, scene_change):
    have_motion_data = bool(scene_change)

    tracks = defaultdict(list)
    for d in detections:
        tracks[d["track"]].append(d)

    for track in tracks.values():
        track.sort(key=lambda d: frame_index[d["frame"]])
        # Count the frame-to-frame steps where the box stayed put AND the camera moved.
        proven_steps = 0
        for a, b in zip(track, track[1:]):
            moved = scene_change.get((a["frame"], b["frame"]), SCENE_CHANGE + 1 if not have_motion_data else 0)
            if moved >= SCENE_CHANGE:
                proven_steps += 1

        if len(track) >= MIN_HUD_FRAMES and proven_steps >= MIN_HUD_FRAMES - 1:
            label = "hud"
            reason = f"same position in {len(track)} frames while the scene changed"
        elif len(track) >= MIN_HUD_FRAMES:
            label = "unknown"
            reason = f"same position in {len(track)} frames, but the camera did not move"
        else:
            label = None  # decided below
            reason = ""
        for d in track:
            d["label"], d["reason"] = label, reason

    # Rule 3: same text, nearby frame, different position -> moved with the world.
    for d in detections:
        if d["label"] is not None:
            continue
        i = frame_index[d["frame"]]
        for other in detections:
            j = frame_index[other["frame"]]
            if other is d or not (0 < abs(i - j) <= NEARBY_FRAMES):
                continue
            if similar_text(d["text"], other["text"]) and distance(d, other) > MOVED_PX:
                if rigid_vertical_shift(d, other):
                    # Looks like HUD layout moving (e.g. subtitles), not camera motion.
                    d["label"] = "unknown"
                    d["reason"] = (f'"{other["text"]}" in {other["frame"]} moved only vertically '
                                   "with identical size: possibly HUD layout shifting")
                else:
                    d["label"] = "world"
                    d["reason"] = f'"{other["text"]}" in {other["frame"]} at a different position'
                break
        if d["label"] is None:
            d["label"] = "unknown"
            d["reason"] = "seen once; no evidence either way"


def save_outputs(detections, frame_names):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_DIR / "classified.csv"
    fields = ["frame", "text", "confidence", "x_min", "y_min", "x_max", "y_max", "track", "label", "reason"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detections)

    if not FRAMES_DIR.exists():
        print(f"No frames in {FRAMES_DIR}; skipping preview images and masks.")
        return

    colours = {"hud": (0, 200, 0), "world": (0, 0, 230), "unknown": (160, 160, 160)}  # BGR
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d["frame"]].append(d)

    (OUT_DIR / "images").mkdir(exist_ok=True)
    (OUT_DIR / "masks").mkdir(exist_ok=True)
    for name in frame_names:
        img = cv2.imread(str(FRAMES_DIR / name))
        if img is None:
            continue
        mask = np.zeros(img.shape[:2], np.uint8)
        for d in by_frame.get(name, []):
            p1, p2 = (d["x_min"], d["y_min"]), (d["x_max"], d["y_max"])
            cv2.rectangle(img, p1, p2, colours[d["label"]], 3)
            cv2.putText(img, d["label"].upper(), (d["x_min"], max(15, d["y_min"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[d["label"]], 2)
            if d["label"] == "hud":
                cv2.rectangle(mask,
                              (d["x_min"] - MASK_PADDING, d["y_min"] - MASK_PADDING),
                              (d["x_max"] + MASK_PADDING, d["y_max"] + MASK_PADDING),
                              255, thickness=-1)
        stem = Path(name).stem
        cv2.imwrite(str(OUT_DIR / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(OUT_DIR / "masks" / f"{stem}.png"), mask)


def main():
    parser = argparse.ArgumentParser(description="Classify detected text as HUD or in-world.")
    parser.add_argument("--csv", default="outputs/ocr/summary.csv", help="OCR results to classify")
    args = parser.parse_args()

    detections = load_detections(PROJECT_ROOT / args.csv)
    if not detections:
        print("No detections above the confidence threshold.")
        return

    # Frame order comes from the frames folder if present, else from the CSV names.
    if FRAMES_DIR.exists():
        frame_names = sorted(p.name for p in FRAMES_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    else:
        frame_names = []
    for d in detections:
        if d["frame"] not in frame_names:
            frame_names.append(d["frame"])
    frame_names.sort()

    # Frame position in time. Use the number in the file name (frame_0037 -> 37) so
    # that frames with no detections still count as gaps; fall back to list order.
    def frame_number(name, fallback):
        digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
        return int(digits) if digits else fallback
    frame_index = {name: frame_number(name, i) for i, name in enumerate(frame_names)}

    scene_change = measure_scene_change(frame_names)
    if not scene_change:
        print("Frames not found: assuming the camera moved between every pair of frames.")
    else:
        values = sorted(scene_change.values())
        moving = sum(v >= SCENE_CHANGE for v in values)
        print(f"Camera motion between frames: min {values[0]:.1f}, median {values[len(values) // 2]:.1f}, "
              f"max {values[-1]:.1f} (threshold {SCENE_CHANGE}); "
              f"{moving} of {len(values)} frame pairs count as 'camera moved'")

    by_frame = build_tracks(detections, frame_index)
    classify(detections, by_frame, frame_index, scene_change)
    save_outputs(detections, frame_names)

    counts = defaultdict(int)
    for d in detections:
        counts[d["label"]] += 1
    print(f"Classified {len(detections)} detections (confidence >= {MIN_CONFIDENCE}):")
    for label in ("hud", "world", "unknown"):
        print(f"  {label:8s} {counts[label]}")
    print(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
