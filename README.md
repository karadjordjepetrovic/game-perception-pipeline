# Game Perception Pipeline

Real-time perception on gameplay footage: **depth**, **HUD vs in-world text** and **tracked objects** from the same video, with every stage measured on its own.

It brings together three earlier projects, each added with `git subtree` so their full history is kept:

| Stage | Project | Best result there (RTX 3070) |
| --- | --- | --- |
| Depth | [realtime-depth-optimization](https://github.com/karadjordjepetrovic/realtime-depth-optimization) → `stages/depth` | 19.13 → 6.86 ms per frame (2.79×), 0.07% mean error vs FP32 |
| Text | [game-text-recognition](https://github.com/karadjordjepetrovic/game-text-recognition) → `stages/text` | 15.3 s → 172 ms per 1080p frame on GPU (~90×), HUD mask per frame |
| Objects | [game-object-tracking](https://github.com/karadjordjepetrovic/game-object-tracking) → `stages/objects` | 15.8 → 126.5 FPS (8×) at matching accuracy, TensorRT FP16 |

## How it works

```
video ──decode once (background thread)──► frame
                                            ├─► depth    every frame   (PyTorch FP16, channels_last, CUDA Graph)
                                            ├─► objects  every frame   (YOLO11m TensorRT FP16 + ByteTrack)
                                            └─► text     every Nth     (PaddleOCR, own environment, joined by frame number)
                                                    │
                                    timing per stage + overlay video
```

- **One decode, shared by all stages.** Decoding runs in a background thread, overlapping with GPU work.
- **Multi-rate.** Depth and objects run on every frame. OCR costs ~120–170 ms per frame and on-screen text changes slowly, so it runs every Nth frame and its last result is held in between.
- **Text in its own process.** PaddleOCR and PyTorch need different CUDA library builds, so the text stage runs in its own environment and its results are joined in by frame number.
- **Measured per stage.** The GPU is synchronised after each stage, so each number is that stage's own cost. The first frames (CUDA warm-up) are left out. `--no-sync` measures the whole pipeline with stages overlapping.

## Results

_Coming next: per-stage and whole-pipeline timing on Modern Warfare 2 and Modern Warfare 3 footage._

## Setup (Windows, NVIDIA GPU)

```powershell
cd game-perception-pipeline
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest            # pipeline tests, no GPU needed
```

The text stage has its own environment: see `stages/text/README.md`.

## Usage

Footage is recorded from the retail games and is not committed. Put clips in `footage/`.

```powershell
# time depth and objects
python -m gpp bench footage/mw2.mp4 --stages depth,objects --tag baseline

# overlay video with all three stages (text from a classify_hud.py run at 5 fps on a 60 fps clip: step 12)
python -m gpp run footage/mw2.mp4 --stages depth,objects,text --text-csv stages/text/outputs/hud/classified.csv --text-step 12 --tag all
```

Results are appended to `outputs/bench.csv`; overlay videos go to `outputs/`.

## Layout

```
gpp/               pipeline: video input, Stage interface, timing, report, overlay, CLI
gpp/adapters/      one wrapper per stage
stages/            the three projects (git subtree)
tests/             pipeline tests with dummy stages, no GPU needed
```
