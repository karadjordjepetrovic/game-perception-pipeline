"""Runs the stages over a video and times each one."""
import time
from dataclasses import dataclass, field


@dataclass
class FrameTiming:
    index: int
    stages: dict  # stage name -> milliseconds (only stages that ran on this frame)
    total_ms: float


@dataclass
class RunResult:
    timings: list = field(default_factory=list)
    wall_seconds: float = 0.0
    warmup: int = 0

    @property
    def timed_frames(self):
        return len(self.timings)

    @property
    def wall_fps(self):
        return self.timed_frames / self.wall_seconds if self.wall_seconds > 0 else 0.0


def gpu_sync():
    """Return a function that waits for the GPU, or a no-op without CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return lambda: None


def run(reader, stages, info, warmup=20, sync=True, on_frame=None):
    """Run every stage on every frame (respecting each stage's every_n).

    warmup:   the first N frames are left out of the timing (CUDA start-up, engine caches).
    sync:     wait for the GPU after each stage, so each stage's time is its own.
              Turn off to measure the pipeline with stages overlapping.
    on_frame: called as on_frame(index, frame, latest_results) after each frame, e.g. to
              draw and write the overlay video. Its time is not in the stage timings,
              but it is in wall-clock FPS, so benchmark without it.
    """
    for stage in stages:
        stage.setup(info)
    sync_fn = gpu_sync() if sync else (lambda: None)
    latest = {stage.name: None for stage in stages}
    result = RunResult(warmup=warmup)
    wall_start = None
    try:
        for index, frame in reader:
            if index == warmup:
                sync_fn()
                wall_start = time.perf_counter()
            frame_start = time.perf_counter()
            stage_ms = {}
            for stage in stages:
                if index % stage.every_n:
                    continue
                t0 = time.perf_counter()
                latest[stage.name] = stage.process(index, frame)
                sync_fn()
                if stage.timed:
                    stage_ms[stage.name] = (time.perf_counter() - t0) * 1000
            total_ms = (time.perf_counter() - frame_start) * 1000
            if index >= warmup:
                result.timings.append(FrameTiming(index, stage_ms, total_ms))
            if on_frame is not None:
                on_frame(index, frame, dict(latest))
        if wall_start is not None:
            sync_fn()
            result.wall_seconds = time.perf_counter() - wall_start
    finally:
        for stage in stages:
            stage.close()
    return result
