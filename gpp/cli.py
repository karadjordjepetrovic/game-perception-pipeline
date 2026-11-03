"""Command line.

  python -m gpp bench footage/mw2.mp4 --stages depth,objects --tag baseline
  python -m gpp run   footage/mw2.mp4 --stages depth,objects,text --text-csv stages/text/outputs/hud/classified.csv --text-step 12

bench: times each stage, no video written; prints a table and appends to outputs/bench.csv.
run:   the same, plus an overlay video in outputs/<clip>_<tag>.mp4.
"""
import argparse
import platform
from pathlib import Path

from . import pipeline, report
from .video import VideoReader, probe


def build_stages(args):
    stages = []
    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if name == "depth":
            from .adapters.depth import DepthStage
            stages.append(DepthStage(size=args.depth_size, dtype=args.depth_dtype,
                                     channels_last=not args.no_channels_last,
                                     cuda_graph=not args.no_cuda_graph))
        elif name == "objects":
            from .adapters.objects import ObjectStage
            stages.append(ObjectStage(model=args.objects_model, imgsz=args.objects_imgsz))
        elif name == "text":
            if not args.text_csv:
                raise SystemExit("--stages text needs --text-csv (classified.csv from stages/text)")
            from .adapters.text import TextStage
            stages.append(TextStage(args.text_csv, step=args.text_step))
        else:
            raise SystemExit(f"unknown stage: {name} (choose from depth, objects, text)")
    return stages


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return platform.processor() or "cpu"


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["bench", "run"])
    p.add_argument("video")
    p.add_argument("--stages", default="depth,objects", help="comma-separated: depth, objects, text")
    p.add_argument("--tag", default="default", help="name for this run in outputs/")
    p.add_argument("--warmup", type=int, default=20, help="first N frames left out of the timing")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-threaded-decode", action="store_true")
    p.add_argument("--no-sync", action="store_true", help="don't wait for the GPU after each stage")
    p.add_argument("--out-dir", default="outputs")
    # depth
    p.add_argument("--depth-size", type=int, default=518)
    p.add_argument("--depth-dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--no-channels-last", action="store_true")
    p.add_argument("--no-cuda-graph", action="store_true")
    # objects
    p.add_argument("--objects-model", default="stages/objects/yolo11m_fp16.engine")
    p.add_argument("--objects-imgsz", type=int, default=640)
    # text
    p.add_argument("--text-csv", default=None)
    p.add_argument("--text-step", type=int, default=12, help="video frames between OCR frames")
    args = p.parse_args(argv)

    info = probe(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = build_stages(args)
    reader = VideoReader(args.video, threaded=not args.no_threaded_decode, max_frames=args.max_frames)

    writer = None
    if args.command == "run":
        from .overlay import OverlayWriter
        video_out = out_dir / f"{Path(args.video).stem}_{args.tag}.mp4"
        writer = OverlayWriter(video_out, info)
    try:
        result = pipeline.run(reader, stages, info, warmup=args.warmup, sync=not args.no_sync, on_frame=writer)
    finally:
        if writer:
            writer.close()

    rows = report.summarize(result)
    title = f"{Path(args.video).name} · {args.tag} · {info.width}x{info.height} · {gpu_name()}"
    print(report.to_markdown(rows, result, title=title))
    report.write_csv(rows, out_dir / "bench.csv", extra={
        "clip": Path(args.video).name, "tag": args.tag, "command": args.command,
        "stages": args.stages, "gpu": gpu_name(), "wall_fps": f"{result.wall_fps:.2f}",
    })
    if writer:
        print(f"Overlay video: {video_out}")
