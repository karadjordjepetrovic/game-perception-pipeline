"""Save one video frame next to its depth map, for slides.

  python tools/depth_snapshot.py footage/clip01.mp4 --seconds 12 --out outputs/frame_vs_depth.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpp.adapters.depth import DepthStage  # noqa: E402
from gpp.overlay import depth_to_colour  # noqa: E402
from gpp.video import probe  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--out", default="outputs/frame_vs_depth.png")
args = p.parse_args()

info = probe(args.video)
cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.seconds * info.fps))
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("could not read that frame")

stage = DepthStage(cuda_graph=False)
stage.setup(info)
depth = stage.process(0, frame)
side = np.hstack([frame, depth_to_colour(depth, (info.width, info.height))])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(args.out, side)
print(f"saved {args.out}")
