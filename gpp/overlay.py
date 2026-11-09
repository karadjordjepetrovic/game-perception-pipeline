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
    """Return a copy of frame with text and object boxes, and the depth map as an inset (top right)."""
    out = frame.copy()
    h, w = out.shape[:2]

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

    agent = results.get("agent")
    if agent:
        draw_agent(out, agent["decision"], agent["world"])

    # depth inset last, so boxes never draw over it
    depth = results.get("depth")
    if depth is not None:
        iw, ih = int(w * inset_scale), int(h * inset_scale)
        out[8:8 + ih, w - iw - 8:w - 8] = depth_to_colour(depth, (iw, ih))
        cv2.rectangle(out, (w - iw - 8, 8), (w - 8, 8 + ih), (255, 255, 255), 2)
        cv2.putText(out, "depth", (w - iw, 8 + ih - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


STATE_COLOURS = {"ENGAGE": (40, 40, 230), "SEARCH": (0, 190, 255), "RELOAD": (230, 160, 40),
                 "RECOVER": (200, 60, 200), "EXPLORE": (90, 200, 90)}  # BGR


def draw_agent(out, decision, world):
    """Decision agent: state and reason (top left), selected target, intended look as an arrow from the centre."""
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    colour = STATE_COLOURS.get(decision.state.split(".")[0], (255, 255, 255))

    for t in world.targets:
        if not t.visible:  # remembered: a small ring where it was last seen
            cv2.circle(out, (int(t.aim[0]), int(t.aim[1])), 10, (0, 190, 255), 2)
    for t in world.targets:
        if t.visible and t.id == decision.target:
            x1, y1, x2, y2 = (int(v) for v in t.box)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 4)
            cv2.line(out, (cx, cy), (int(t.aim[0]), int(t.aim[1])), colour, 1, cv2.LINE_AA)
            cv2.circle(out, (int(t.aim[0]), int(t.aim[1])), 5, colour, -1)

    a = decision.action
    scale = 8.0  # pixels per degree of turn this tick
    tip = (int(cx + a.yaw * scale), int(cy - a.pitch * scale))
    if tip != (cx, cy):
        cv2.arrowedLine(out, (cx, cy), tip, colour, 3, cv2.LINE_AA, tipLength=0.3)
    if a.fire:
        cv2.circle(out, (cx, cy), 18, (0, 0, 255), 3)

    flags = [n for n in ("fire", "ads", "reload", "crouch", "jump") if getattr(a, n)]
    lines = [decision.state, decision.reason,
             f"move {a.forward:+.1f} {a.strafe:+.1f}  look {a.yaw:+.1f} {a.pitch:+.1f}  {' '.join(flags)}"]
    if world.me.ammo is not None:
        lines[-1] += f"  ammo {world.me.ammo}"
    y = 12
    for i, line in enumerate(lines):
        size = 0.9 if i == 0 else 0.6
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, size, 2)
        cv2.rectangle(out, (8, y), (16 + tw, y + th + 12), (0, 0, 0), -1)
        cv2.putText(out, line, (12, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, size, colour if i == 0 else (255, 255, 255), 2)
        y += th + 16


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
