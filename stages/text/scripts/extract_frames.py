"""
Extract frames from a gameplay video using OpenCV (no ffmpeg needed).

Usage (from the project root, with the conda env active):
    python scripts/extract_frames.py
    python scripts/extract_frames.py --video data/other.mp4 --fps 2 --max-frames 100 --width 1920

Defaults: reads data/gameplay.mp4, takes 1 frame per second, stops at 50 frames,
and saves PNGs to data/frames/.
"""

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # argparse lets you change settings from the command line without editing code.
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", default="data/gameplay.mp4", help="Path to the video file")
    parser.add_argument("--out", default="data/frames", help="Folder to save frames into")
    parser.add_argument("--fps", type=float, default=1.0, help="How many frames to keep per second of video")
    parser.add_argument("--max-frames", type=int, default=50, help="Stop after this many frames")
    parser.add_argument("--width", type=int, default=0,
                        help="Resize frames to this width (keeps aspect ratio). 0 = original size")
    args = parser.parse_args()

    video_path = PROJECT_ROOT / args.video
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open the video.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}, {video_fps:.1f} fps, {total} frames total")

    # Keep every Nth frame. E.g. 60 fps video, 1 fps wanted -> keep every 60th frame.
    step = max(1, round(video_fps / args.fps))

    frame_index = 0
    saved = 0
    while saved < args.max_frames:
        ok, frame = cap.read()  # read the next frame; ok is False at the end of the video
        if not ok:
            break

        if frame_index % step == 0:
            if args.width > 0 and frame.shape[1] != args.width:
                h, w = frame.shape[:2]
                new_h = round(h * args.width / w)
                frame = cv2.resize(frame, (args.width, new_h), interpolation=cv2.INTER_AREA)

            saved += 1
            out_path = out_dir / f"frame_{saved:04d}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path.name} (video frame {frame_index})")

        frame_index += 1

    cap.release()
    print(f"\nDone. Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
