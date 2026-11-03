"""Draws every stage's latest result on the frame and writes the overlay video."""
import cv2
import numpy as np

TEXT_COLOURS = {"hud": (80, 200, 80), "world": (60, 60, 230), "unknown": (160, 160, 160)}  # BGR


def _id_colour(track_id):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def depth_to_colour(depth, size):
    """Relative depth map (any float array) -> BGR colour image of the given (w, h) size."""
    d = np.asarray(depth, dtype=np.float32)
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    img = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def draw(frame, results, inset_scale=0.3):
    """Return a copy of frame with depth (inset, top right), text and object boxes drawn."""
    out = frame.copy()
    h, w = out.shape[:2]

    depth = results.get("depth")
    if depth is not None:
        iw, ih = int(w * inset_scale), int(h * inset_scale)
        out[8:8 + ih, w - iw - 8:w - 8] = depth_to_colour(depth, (iw, ih))
        cv2.rectangle(out, (w - iw - 8, 8), (w - 8, 8 + ih), (255, 255, 255), 2)
        cv2.putText(out, "depth", (w - iw, 8 + ih - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for det in results.get("text") or []:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        colour = TEXT_COLOURS.get(det.get("label", "unknown"), TEXT_COLOURS["unknown"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, det.get("label", ""), (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    for obj in results.get("objects") or []:
        x1, y1, x2, y2 = (int(v) for v in obj["box"])
        colour = _id_colour(obj["id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, f"#{obj['id']}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    return out


class OverlayWriter:
    """Writes overlay frames to an .mp4 at the source frame rate."""

    def __init__(self, path, info):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, info.fps, (info.width, info.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot write video: {path}")

    def __call__(self, index, frame, results):
        self.writer.write(draw(frame, results))

    def close(self):
        self.writer.release()
