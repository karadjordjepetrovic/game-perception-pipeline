"""Video input: every frame is decoded once and shared by all stages."""
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


def probe(path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


_END = object()


class VideoReader:
    """Iterates (index, frame) over a video, frames as BGR uint8 arrays, index from 0.

    With threaded=True a background thread decodes ahead into a small queue, so decoding
    overlaps with GPU work (OpenCV releases Python's GIL while decoding). Same idea as
    --threaded-decode in stages/objects, which gave +37% there.
    """

    def __init__(self, path, threaded=True, queue_size=8, max_frames=None):
        self.path = Path(path)
        self.threaded = threaded
        self.queue_size = queue_size
        self.max_frames = max_frames

    def _decode(self):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        try:
            index = 0
            while self.max_frames is None or index < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield index, frame
                index += 1
        finally:
            cap.release()

    def __iter__(self):
        if not self.threaded:
            yield from self._decode()
            return

        q = queue.Queue(maxsize=self.queue_size)
        stop = threading.Event()

        def worker():
            try:
                for item in self._decode():
                    if stop.is_set():
                        break
                    q.put(item)
            except Exception as exc:  # hand decode errors to the consumer
                q.put(exc)
            q.put(_END)

        thread = threading.Thread(target=worker, name="decode", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            while thread.is_alive():  # unblock a worker waiting on a full queue
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                thread.join(timeout=0.05)
