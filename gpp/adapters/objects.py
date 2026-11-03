"""Object stage: YOLO11 + ByteTrack from stages/objects.

Best configuration there: YOLO11m as a TensorRT FP16 engine at 640 px, 126.5 FPS with
threaded decode (stages/objects/README). The engine file is built on your machine and
is not in git; pass a .pt file to run PyTorch instead.
"""
from ..stage import Stage


class ObjectStage(Stage):
    name = "objects"

    def __init__(self, model="yolo11m_fp16.engine", imgsz=640, classes=(0,), tracker="bytetrack.yaml", every_n=1):
        self.model_path = model
        self.imgsz = imgsz
        self.classes = list(classes) if classes else None
        self.tracker = tracker
        self.every_n = every_n

    def setup(self, info):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path, task="detect")

    def process(self, index, frame):
        result = self.model.track(frame, persist=True, tracker=self.tracker, classes=self.classes,
                                  imgsz=self.imgsz, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        return [{"box": tuple(b), "id": int(i), "conf": float(c)}
                for b, i, c in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.conf.tolist())]
