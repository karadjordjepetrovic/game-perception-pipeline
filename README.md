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

First clip: 41 s of Call of Duty: Modern Warfare 2 campaign, 1920×1080, 60 fps, 2,458 frames (recorded from the retail game, not committed). RTX 3070, first 20 frames left out.

| Stage | What runs | Median | p95 | Runs on |
| --- | --- | --- | --- | --- |
| Depth | Depth Anything V2 Small, FP16 + channels_last + CUDA Graph, incl. GPU preprocessing and copy back | 8.37 ms | 8.70 ms | every frame |
| Objects | YOLO11m TensorRT FP16 + ByteTrack | 8.88 ms | 10.85 ms | every frame |
| **Depth + objects per frame** | | **17.25 ms** | 19.32 ms | **57.7 FPS** wall clock |
| Text | PaddleOCR on GPU + HUD/in-world classification, own process | 216 ms (mean) | | every 12th frame (5 per second) |

Text on this clip: 1,231 regions found; of 970 above 0.8 confidence, 438 HUD, 276 in-world, 256 unknown.

Second clip: 73 s of Call of Duty: Modern Warfare 3, 1920×1080, recorded at a variable frame rate and converted to a constant 30 fps (2,192 frames), so text runs every 6th frame.

| Stage | Median | p95 | Runs on |
| --- | --- | --- | --- |
| Depth | 8.34 ms | 8.50 ms | every frame |
| Objects | 7.17 ms | 9.26 ms | every frame |
| **Depth + objects per frame** | **15.54 ms** | 17.80 ms | **62.6 FPS** wall clock |

Text on this clip: 1,372 regions found; of 1,134 above 0.8 confidence, 448 HUD, 322 in-world, 364 unknown.

**What the measurements showed**

- **Depth costs 1.5 ms more in the pipeline than alone** (8.37 vs 6.86 ms): preprocessing the 1080p frame on the GPU and copying the depth map back to the CPU.
- **Waiting for the GPU between stages costs nothing measurable:** without it (`--no-sync`) the pipeline ran at 56.3 FPS, within run-to-run noise. Both stages already wait for the GPU themselves (the depth copy-back, the tracker), so they run one after another, not overlapped.
- **Drawing and encoding the overlay video costs more than perception:** with it (`run`), 23 FPS. Benchmarks use `bench`, which writes no video.
- **OCR is the slowest stage by far** (216 ms here, 172 ms on a clip with less text), so it runs at a lower rate and its last result is held in between.

**Next steps**

1. Overlap depth and objects: separate CUDA streams or threads, and keep the depth map on the GPU.
2. Run OCR in a background worker in real time instead of ahead of time.
3. Move the overlay drawing off the main thread.
4. INT8 and a mobile chip's NPU, for running on a device.

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

## Decision agent

`--agent` adds a decision agent after the stages: one first-person-shooter input decision per frame (move, look, fire, aim down sights, reload), from the stages' results.

```
stage results ──► WorldBuilder (gpp/world.py) ──► state machine (gpp/agent/fsm.py) ──► action + reason ──► JSONL log, overlay
```

- **WorldBuilder** turns raw detections into stable facts: each confirmed track's aim error in degrees (pinhole camera with the in-game FOV, `--hfov`), distance from box height (Depth Anything gives relative depth, not metres), a near / mid / far class with hysteresis, short memory of lost targets, ammo read from HUD text, and how blocked the path ahead is from the depth map.
- **State machine**, highest priority first: `RECOVER` (low health) · `RELOAD` · `ENGAGE` (`APPROACH` far targets, `ACQUIRE` = turn onto the target, `FIRE` in bursts) · `SEARCH` (turn toward where a target was lost) · `EXPLORE`. Higher-priority states take over at once; lower ones need a minimum dwell time, and FIRE has an entry/exit hysteresis, so noisy per-frame perception doesn't make it flicker.
- **Look control** is a proportional controller with a time constant and a turn-rate cap, so it behaves the same at 30 and 60 fps.
- **Distance** fuses two estimates: box height (metric, but a crouching or half-hidden target reads too far) and depth (Depth Anything's relative inverse depth, which pose and cover don't change). The depth scale is calibrated every frame from targets that look like whole standing people, then applied to all of them. `--no-depth-distance` uses box height only.
- **Target choice** (`--target-policy`): `closest` (default) picks the nearest foe and only switches to one at least 20% closer; `score` mixes closeness, distance from the crosshair and stickiness.
- **Aim** (`--aim`): `smooth` (default) turns gradually and fires once on target, for driving a real game; `snap` turns the whole error and shoots in the same tick ("shoot #126 at (1012, 488)"), which suits recorded footage where the crosshair never follows the agent. Every ENGAGE action logs the target's screen point and angle (`"aim": {"px": [...], "deg": [...]}`). During short detection gaps the agent keeps aiming at the last position.
- **Camera motion** (`motion` stage, added automatically with `--agent`): phase correlation between consecutive frames, on a band that skips the HUD and the weapon, gives how far the view turned (~1.5 ms, CPU). Remembered targets are moved by that turn each frame, so a lost NPC's position (the ring on the overlay) stays where the NPC was instead of freezing on the screen, and SEARCH turns toward it.

```powershell
python -m gpp run footage/mw2.mp4 --stages depth,objects,text --text-csv stages/text/outputs/hud/classified.csv --text-step 12 --agent --tag agent
```

Each frame is one line in `outputs/<clip>_<tag>_actions.jsonl`:

```json
{"frame": 1240, "t": 20.67, "state": "ENGAGE.ACQUIRE", "action": {"move": [0.0, 0.0], "look": [3.1, -0.4], "fire": false, "ads": true, "reload": false, "crouch": false, "jump": false}, "target": 7, "reason": "#7 mid ~14m, 4.2° right: aligning (err 4.3° > 1.2°)", "world": {...}}
```

Limits for now: there is no friend-or-foe stage, so every person counts as a foe; there is no health stage, so `RECOVER` never triggers; on recorded footage the loop is open (the agent's turns don't change the next frame), so each decision is "what to do given this frame".

## Layout

```
gpp/               pipeline: video input, Stage interface, timing, report, overlay, CLI
gpp/adapters/      one wrapper per stage
gpp/world.py       WorldState: targets, memory, ammo, free space, from the stage results
gpp/adapters/motion.py  camera motion (yaw/pitch per frame)
gpp/agent/         decision agent: state machine, actions, JSONL log
stages/            the three projects (git subtree)
tests/             pipeline tests with dummy stages, no GPU needed
```
