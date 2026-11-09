"""Command line.

  python -m gpp bench footage/mw2.mp4 --stages depth,objects --tag baseline
  python -m gpp run   footage/mw2.mp4 --stages depth,objects,text --text-csv stages/text/outputs/hud/classified.csv --text-step 12

  python -m gpp run   footage/mw2.mp4 --stages depth,objects --agent --tag agent

bench: times each stage, no video written; prints a table and appends to outputs/bench.csv.
run:   the same, plus an overlay video in outputs/<clip>_<tag>.mp4.
--agent: adds the decision agent (gpp/agent): one FPS input decision per frame, logged to
         outputs/<clip>_<tag>_actions.jsonl and drawn on the overlay video.
"""
from collections import Counter
import argparse
import platform
from pathlib import Path

from . import pipeline, report
from .video import VideoReader, probe


def build_stages(args):
    stages = []
    names = [s.strip() for s in args.stages.split(",") if s.strip()]
    if getattr(args, "agent", False) and "motion" not in names:
        names.append("motion")  # the agent uses camera motion to keep lost targets in place
    for name in names:
        if name == "depth":
            from .adapters.depth import DepthStage
            stages.append(DepthStage(size=args.depth_size, dtype=args.depth_dtype,
                                     channels_last=not args.no_channels_last,
                                     cuda_graph=not args.no_cuda_graph))
        elif name == "objects":
            from .adapters.objects import ObjectStage
            stages.append(ObjectStage(model=args.objects_model, imgsz=args.objects_imgsz,
                                      **({"tracker": args.objects_tracker} if args.objects_tracker else {})))
        elif name == "motion":
            from .adapters.motion import MotionStage
            stages.append(MotionStage(hfov_deg=args.hfov))
        elif name == "text":
            if not args.text_csv:
                raise SystemExit("--stages text needs --text-csv (classified.csv from stages/text)")
            from .adapters.text import TextStage
            stages.append(TextStage(args.text_csv, step=args.text_step))
        else:
            raise SystemExit(f"unknown stage: {name} (choose from depth, objects, text, motion)")
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
    p.add_argument("--stages", default="depth,objects", help="comma-separated: depth, objects, text, motion (--agent adds motion)")
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
    p.add_argument("--objects-tracker", default=None,
                   help="tracker .yaml (default gpp/adapters/bytetrack_gpp.yaml; 'bytetrack.yaml' = Ultralytics default)")
    # decision agent
    p.add_argument("--agent", action="store_true", help="run the decision agent on the stage results")
    p.add_argument("--agent-log", default=None, help="JSONL path (default outputs/<clip>_<tag>_actions.jsonl)")
    p.add_argument("--hfov", type=float, default=80.0, help="in-game horizontal FOV in degrees")
    p.add_argument("--aim", choices=("smooth", "snap"), default="smooth",
                   help="smooth: turn gradually, fire when on target; snap: turn and shoot at the target in one tick")
    p.add_argument("--target-policy", choices=("closest", "score"), default="closest")
    p.add_argument("--no-depth-distance", action="store_true", help="distance from box height only")
    # text
    p.add_argument("--text-csv", default=None)
    p.add_argument("--text-step", type=int, default=12, help="video frames between OCR frames")
    args = p.parse_args(argv)

    info = probe(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = build_stages(args)
    reader = VideoReader(args.video, threaded=not args.no_threaded_decode, max_frames=args.max_frames)
    agents, states = [], Counter()
    if args.agent:
        from .agent import AgentConfig, DecisionAgent, WorldConfig
        log_path = Path(args.agent_log) if args.agent_log else out_dir / f"{Path(args.video).stem}_{args.tag}_actions.jsonl"
        agents.append(DecisionAgent(
            log_path=log_path,
            world_cfg=WorldConfig(hfov_deg=args.hfov, use_depth_distance=not args.no_depth_distance),
            agent_cfg=AgentConfig(aim_mode=args.aim, target_policy=args.target_policy)))

    writer = None
    if args.command == "run":
        from .overlay import OverlayWriter
        video_out = out_dir / f"{Path(args.video).stem}_{args.tag}.mp4"
        writer = OverlayWriter(video_out, info)
    try:
        def on_frame(index, frame, latest):
            if agents and latest.get("agent"):
                states[latest["agent"]["decision"].state] += 1
            if writer:
                writer(index, frame, latest)

        result = pipeline.run(reader, stages, info, warmup=args.warmup, sync=not args.no_sync,
                              on_frame=on_frame if (writer or agents) else None, agents=agents)
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
    if agents:
        total = sum(states.values()) or 1
        print("Agent states: " + ", ".join(f"{k} {v / total:.0%}" for k, v in states.most_common()))
        print(f"Action log: {agents[0].log_path}")
