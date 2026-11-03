"""Track objects in a clip and time each stage of the pipeline.

Writes, into outputs/<clip name>/:
  <tag>.txt          tracks in MOT format (one line per box per frame, in original video pixels)
  <tag>_timing.json  average milliseconds per stage, wall-clock FPS
  <tag>.mp4          video with boxes drawn (only with --video)

Examples:
  python src/track.py data/clips/clip01.mp4 --model yolo11x.pt --imgsz 1280 --tag reference --video
  python src/track.py data/clips/clip01.mp4 --model yolo11m_fp16.engine --tag trt_threaded --threaded-decode
  python src/track.py data/clips/clip01.mp4 --model yolo11m_fp16.engine --tag trt_gpuprep --threaded-decode --gpu-preprocess
  nsys profile -t cuda,nvtx -o outputs/profile python src/track.py ... --nvtx

Options that change the pipeline:
  --threaded-decode  decode frames in a background thread, overlapping with inference.
  --gpu-preprocess   letterbox the frame on the GPU (upload, resize, pad, normalise) and hand
                     the model a ready tensor, instead of Ultralytics doing it on the CPU.
                     With --threaded-decode, this runs in the background thread on its own
                     CUDA stream, so it also overlaps with inference.

Timing:
  - The first --warmup frames are left out (CUDA start-up).
  - "fps" is wall-clock: timed frames divided by the real time they took. It is the honest
    number once stages overlap.
  - The per-stage numbers say where time goes on the main thread. Work done in the
    background thread is reported separately as "bg thread" (it runs in parallel).
"""
import argparse
import json
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('clip', help='path to the video clip')
parser.add_argument('--model', default='yolo11n.pt', help='model weights (.pt) or TensorRT engine (.engine)')
parser.add_argument('--tracker', default='bytetrack.yaml', help='tracker config (.yaml)')
parser.add_argument('--tag', help='name for the output files (default: the model file name)')
parser.add_argument('--classes', type=int, nargs='*', default=[0],
                    help='class ids to keep (0 = person). Use --classes with no numbers to keep all')
parser.add_argument('--imgsz', type=int, default=640, help='model input size in pixels (must match an engine)')
parser.add_argument('--warmup', type=int, default=20, help='first N frames are left out of the timing')
parser.add_argument('--video', action='store_true', help='also write a video with boxes drawn (slow)')
parser.add_argument('--threaded-decode', action='store_true',
                    help='decode frames in a background thread, overlapping with GPU work')
parser.add_argument('--gpu-preprocess', action='store_true',
                    help='resize/pad/normalise frames on the GPU instead of the CPU')
parser.add_argument('--nvtx', action='store_true',
                    help='mark stages with NVTX ranges so they show up in Nsight Systems')
args = parser.parse_args()

if args.gpu_preprocess or args.nvtx:
    import torch
    import torch.nn.functional as F

# --- NVTX markers: named bars on the Nsight Systems timeline -------------------------------
if args.nvtx:
    @contextmanager
    def nvtx(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
else:
    @contextmanager
    def nvtx(name):
        yield


# --- GPU preprocessing --------------------------------------------------------------------------
class GpuLetterbox:
    """Turns a BGR video frame into the model's input tensor, on the GPU.

    Does what Ultralytics' CPU letterbox does: scale the frame to fit imgsz x imgsz keeping
    its aspect ratio, pad the rest with grey (114), BGR -> RGB, HWC -> CHW, 0-255 -> 0-1.
    The frame is copied to the GPU once as raw bytes; everything after that runs on the GPU.
    """

    def __init__(self, frame_h, frame_w, imgsz, device):
        self.imgsz = imgsz
        self.device = device
        self.scale = min(imgsz / frame_h, imgsz / frame_w)
        self.new_w = round(frame_w * self.scale)
        self.new_h = round(frame_h * self.scale)
        # Centre the image, rounding the same way as Ultralytics' letterbox.
        self.left = round((imgsz - self.new_w) / 2 - 0.1)
        self.top = round((imgsz - self.new_h) / 2 - 0.1)
        cuda = device.type == 'cuda'
        # Page-locked ("pinned") host memory lets the CPU->GPU copy run asynchronously and faster.
        self.host = torch.empty((frame_h, frame_w, 3), dtype=torch.uint8, pin_memory=cuda)
        self.copied = torch.cuda.Event() if cuda else None

    def __call__(self, frame):
        if self.copied is not None:
            self.copied.synchronize()  # the previous upload must finish before we overwrite the buffer
        self.host.numpy()[:] = frame
        gpu = self.host.to(self.device, non_blocking=True)
        if self.copied is not None:
            self.copied.record()
        x = gpu.permute(2, 0, 1).flip(0).unsqueeze(0).float().div_(255.0)  # BGR HWC -> RGB 1CHW, 0-1
        x = F.interpolate(x, size=(self.new_h, self.new_w), mode='bilinear', align_corners=False)
        out = torch.full((1, 3, self.imgsz, self.imgsz), 114 / 255.0, device=self.device)
        out[:, :, self.top:self.top + self.new_h, self.left:self.left + self.new_w] = x
        return out

    def to_frame_coords(self, xyxy):
        """Map boxes from the model's imgsz x imgsz input back to original frame pixels."""
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - self.left) / self.scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - self.top) / self.scale
        return xyxy


# --- Frame sources ----------------------------------------------------------------------------
class DirectReader:
    """Decodes (and optionally preprocesses) a frame only when asked, on the calling thread."""

    def __init__(self, cap, prep=None):
        self.cap = cap
        self.prep = prep
        self.bg_ms = 0.0
        self.prep_ms = 0.0

    def read(self):
        with nvtx('decode'):
            ok, frame = self.cap.read()
        tensor = None
        if ok and self.prep:
            t0 = time.perf_counter()
            with nvtx('gpu preprocess'):
                tensor = self.prep(frame)
            self.prep_ms += (time.perf_counter() - t0) * 1000
        return ok, frame, tensor

    def stop(self):
        pass


class ThreadedReader:
    """Decodes (and optionally preprocesses) frames ahead of time in a background thread.

    OpenCV releases Python's GIL while decoding, so decoding really runs in parallel with
    inference on the main thread. GPU preprocessing runs on its own CUDA stream, so the GPU
    can resize the next frame while it runs inference on the current one.
    """

    def __init__(self, cap, prep=None, depth=8):
        self.cap = cap
        self.prep = prep
        self.frames = queue.Queue(maxsize=depth)
        self.bg_ms = 0.0
        self.prep_ms = 0.0
        self.running = True
        if prep is not None and prep.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
            self.main_stream = torch.cuda.current_stream()
        else:
            self.stream = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            t0 = time.perf_counter()
            with nvtx('decode (thread)'):
                ok, frame = self.cap.read()
            tensor = None
            if ok and self.prep:
                with nvtx('gpu preprocess (thread)'):
                    if self.stream is not None:
                        with torch.cuda.stream(self.stream):
                            tensor = self.prep(frame)
                            # Tell PyTorch's memory allocator the main stream will use this tensor.
                            tensor.record_stream(self.main_stream)
                        self.stream.synchronize()  # tensor is complete before the main thread sees it
                    else:
                        tensor = self.prep(frame)
            self.bg_ms += (time.perf_counter() - t0) * 1000
            self.frames.put((ok, frame, tensor))
            if not ok:
                break

    def read(self):
        return self.frames.get()

    def stop(self):
        self.running = False
        while not self.frames.empty():  # unblock the thread if it is waiting on a full queue
            self.frames.get_nowait()


clip = Path(args.clip)
tag = args.tag or Path(args.model).stem
out_dir = Path('outputs') / clip.stem
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(clip))
if not cap.isOpened():
    raise SystemExit(f'Could not open {clip}')
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = YOLO(args.model)

prep = None
if args.gpu_preprocess:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: CUDA not available, --gpu-preprocess will run on the CPU')
    prep = GpuLetterbox(frame_h, frame_w, args.imgsz, device)

writer = None
if args.video:
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(out_dir / f'{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps_in,
                             (frame_w, frame_h))

reader = ThreadedReader(cap, prep) if args.threaded_decode else DirectReader(cap, prep)

stages = ['decode', 'gpu_prep', 'preprocess', 'inference', 'postprocess', 'tracker']
totals = dict.fromkeys(stages, 0.0)
timed_frames = 0
frame_idx = 0
wall_start = None
bg_ms_at_start = 0.0

with open(out_dir / f'{tag}.txt', 'w') as track_file:
    while True:
        if frame_idx == args.warmup:  # start the wall clock once warm-up is over
            wall_start = time.perf_counter()
            bg_ms_at_start = reader.bg_ms

        prep_before = reader.prep_ms
        t0 = time.perf_counter()
        with nvtx('get frame'):
            ok, frame, tensor = reader.read()
        t1 = time.perf_counter()
        if not ok:
            break
        frame_idx += 1  # MOT format counts frames from 1
        prep_ms = reader.prep_ms - prep_before  # GPU preprocessing done on this thread (if any)

        with nvtx('track'):
            source = tensor if tensor is not None else frame
            result = model.track(source, persist=True, tracker=args.tracker, classes=args.classes or None,
                                 imgsz=args.imgsz, verbose=False)[0]
        t2 = time.perf_counter()

        with nvtx('write'):
            boxes = result.boxes
            xyxy = boxes.xyxy.tolist()
            if prep is not None:
                xyxy = prep.to_frame_coords(xyxy).tolist()  # back to original video pixels
            if boxes.id is not None:
                for (x1, y1, x2, y2), tid, conf in zip(xyxy, boxes.id.int().tolist(), boxes.conf.tolist()):
                    # frame, id, left, top, width, height, confidence, -1, -1, -1
                    track_file.write(f'{frame_idx},{tid},{x1:.1f},{y1:.1f},'
                                     f'{x2 - x1:.1f},{y2 - y1:.1f},{conf:.3f},-1,-1,-1\n')

        if writer:
            if prep is None:
                writer.write(result.plot())
            else:  # result.plot() would draw on the small letterboxed image, so draw on the frame
                ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(xyxy)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                    if tid is not None:
                        cv2.putText(frame, str(tid), (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 200, 255), 2)
                writer.write(frame)

        if frame_idx > args.warmup:
            speed = result.speed  # milliseconds, measured inside Ultralytics
            model_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
            totals['decode'] += (t1 - t0) * 1000 - prep_ms
            totals['gpu_prep'] += prep_ms
            totals['preprocess'] += speed['preprocess']
            totals['inference'] += speed['inference']
            totals['postprocess'] += speed['postprocess']
            totals['tracker'] += max((t2 - t1) * 1000 - model_ms, 0.0)  # tracker + small overheads
            timed_frames += 1

wall_end = time.perf_counter()
reader.stop()
cap.release()
if writer:
    writer.release()

if frame_idx == 0:
    raise SystemExit('No frames were read - check the clip with ffprobe')
if timed_frames == 0:
    raise SystemExit(f'Clip has only {frame_idx} frames, fewer than --warmup {args.warmup}')

avg = {stage: totals[stage] / timed_frames for stage in stages}
stage_sum_ms = sum(avg.values())
wall_ms = (wall_end - wall_start) * 1000 / timed_frames
fps = 1000 / wall_ms

mode = []
mode.append('threaded decode' if args.threaded_decode else 'sequential')
if args.gpu_preprocess:
    mode.append('GPU preprocess')
print(f'\n{tag} ({", ".join(mode)}): {frame_idx} frames ({timed_frames} timed, first {args.warmup} skipped)')
labels = {'decode': 'decode wait' if args.threaded_decode else 'decode', 'gpu_prep': 'gpu prep'}
for stage in stages:
    if stage == 'gpu_prep' and not (args.gpu_preprocess and not args.threaded_decode):
        continue  # only shown when GPU preprocessing runs on the main thread
    print(f'  {labels.get(stage, stage):>13}: {avg[stage]:7.2f} ms/frame')
print(f'  {"stage sum":>13}: {stage_sum_ms:7.2f} ms/frame')
if args.threaded_decode:
    bg_ms = (reader.bg_ms - bg_ms_at_start) / timed_frames
    what = 'decode + gpu prep' if args.gpu_preprocess else 'decode'
    print(f'  {"bg thread":>13}: {bg_ms:7.2f} ms/frame  ({what}, runs in parallel)')
print(f'  {"wall clock":>13}: {wall_ms:7.2f} ms/frame = {fps:.1f} FPS')

timing = {
    'clip': clip.stem, 'tag': tag, 'model': args.model, 'tracker': args.tracker,
    'imgsz': args.imgsz, 'threaded_decode': args.threaded_decode, 'gpu_preprocess': args.gpu_preprocess,
    'frames': frame_idx, 'timed_frames': timed_frames,
    'ms_per_frame': {k: round(v, 3) for k, v in avg.items()},
    'stage_sum_ms': round(stage_sum_ms, 3), 'wall_ms': round(wall_ms, 3), 'fps': round(fps, 2),
}
(out_dir / f'{tag}_timing.json').write_text(json.dumps(timing, indent=2))
print(f'\nSaved {out_dir / (tag + ".txt")} and {tag}_timing.json')
