import sys, time, cv2
from ultralytics import YOLO

clip = sys.argv[1]                      # e.g. data/clips/clip01.mp4
name = clip.replace('\\', '/').split('/')[-1].rsplit('.', 1)[0]

model = YOLO('yolo11n.pt')              # smallest model; downloads on first run
cap = cv2.VideoCapture(clip)
if not cap.isOpened():
    sys.exit(f'Could not open {clip}')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_out = cv2.VideoWriter(f'outputs/{name}_tracked.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
track_file = open(f'outputs/{name}.txt', 'w')

frame_idx = 0
times = {'decode': 0.0, 'track': 0.0, 'draw': 0.0}
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1                      # MOT format counts frames from 1
    t1 = time.perf_counter()

    result = model.track(frame, persist=True, tracker='bytetrack.yaml',
                         classes=[0], verbose=False)[0]   # class 0 = person
    t2 = time.perf_counter()

    boxes = result.boxes
    if boxes.id is not None:
        for (x1, y1, x2, y2), tid, conf in zip(boxes.xyxy.tolist(),
                                               boxes.id.int().tolist(),
                                               boxes.conf.tolist()):
            # MOT line: frame, id, left, top, width, height, conf, -1, -1, -1
            track_file.write(f'{frame_idx},{tid},{x1:.1f},{y1:.1f},'
                             f'{x2 - x1:.1f},{y2 - y1:.1f},{conf:.3f},-1,-1,-1\n')
    video_out.write(result.plot())
    t3 = time.perf_counter()

    times['decode'] += t1 - t0
    times['track'] += t2 - t1
    times['draw'] += t3 - t2

if frame_idx == 0:
    sys.exit('No frames were read - check the clip with ffprobe')
for stage, total in times.items():
    print(f'{stage:>7}: {1000 * total / frame_idx:6.1f} ms/frame')
print(f'  total: {frame_idx / sum(times.values()):6.1f} FPS')