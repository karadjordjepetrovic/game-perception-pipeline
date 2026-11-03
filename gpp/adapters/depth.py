"""Depth stage: Depth Anything V2 Small with the optimizations from stages/depth.

FP16 + channels_last + CUDA Graph: 6.86 ms median for the model alone on an RTX 3070
(stages/depth/README). Here the stage also includes preprocessing on the GPU and copying
the depth map back to the CPU, so it is measured end to end.

Input is resized to size x size (default 518, a multiple of 14), as in the benchmark.
The Hugging Face processor keeps the aspect ratio instead; the square input keeps the
CUDA Graph's input shape fixed.
"""
from ..stage import Stage

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DepthStage(Stage):
    name = "depth"

    def __init__(self, size=518, dtype="fp16", channels_last=True, cuda_graph=True,
                 model_id=MODEL_ID, device="cuda", every_n=1):
        self.size = size
        self.dtype_name = dtype
        self.channels_last = channels_last
        self.cuda_graph = cuda_graph
        self.model_id = model_id
        self.device = device
        self.every_n = every_n

    def setup(self, info):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForDepthEstimation

        self.torch, self.F = torch, F
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.dtype_name]
        model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device, dtype=self.dtype).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.static_in = torch.zeros(1, 3, self.size, self.size, device=self.device, dtype=self.dtype)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
            self.static_in = self.static_in.contiguous(memory_format=torch.channels_last)
        self.model = model

        def forward():
            return self.model(pixel_values=self.static_in).predicted_depth

        self.forward = forward
        self.run = forward
        if self.cuda_graph:
            with torch.inference_mode():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):  # warm up on a side stream before capture
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(side)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_out = forward()

            def replay():
                self.graph.replay()
                return self.static_out

            self.run = replay

    def preprocess(self, frame):
        """BGR uint8 HxWx3 (numpy) -> normalized 1x3xSxS tensor on the GPU."""
        torch, F = self.torch, self.F
        x = torch.from_numpy(frame).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).flip(0).unsqueeze(0).float().div_(255.0)  # BGR HWC -> RGB 1CHW
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return ((x - self.mean) / self.std).to(self.dtype)

    def process(self, index, frame):
        with self.torch.inference_mode():
            self.static_in.copy_(self.preprocess(frame))
            depth = self.run()
            return depth[0].float().cpu().numpy()  # SxS relative inverse depth (bigger = closer)
