"""Compare a tracker run against a reference run of the same clip.

The reference is the slow, careful run (big model, every frame). This script measures
how closely a faster run agrees with it:

  Precision  share of the run's boxes that match a reference box
  Recall     share of the reference boxes that the run also found
  F1         one number combining precision and recall
  IDF1       share of boxes that carry a consistent ID (1.0 = IDs agree perfectly)
  ID switch  times a reference object's matched ID changed in the run

Each result is also appended to outputs/<clip>/results.csv, with the FPS from the
run's timing file, so you can build the README table from it.

Example:
  python src/compare.py outputs/clip01/reference.txt outputs/clip01/nano.txt
  python src/compare.py results/clip01/reference.txt results/clip01/trt_threaded.txt --no-csv
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MIN = 0.5  # boxes overlapping at least this much count as the same object


def load_tracks(path):
    """Read a MOT file into {frame: (ids array, boxes array as x1, y1, x2, y2)}."""
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame, tid = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            rows[frame].append((tid, x, y, x + w, y + h))
    return {fr: (np.array([r[0] for r in rs]), np.array([r[1:] for r in rs])) for fr, rs in rows.items()}


def iou_matrix(a, b):
    """IoU of every box in a against every box in b (boxes as x1, y1, x2, y2)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def compare(ref, run):
    n_ref = sum(len(ids) for ids, _ in ref.values())
    n_run = sum(len(ids) for ids, _ in run.values())
    matched = 0
    switches = 0
    last_match = {}                 # reference id -> run id it was last matched to
    overlap_frames = defaultdict(int)  # (reference id, run id) -> frames they overlap

    for frame in sorted(set(ref) | set(run)):
        if frame not in ref or frame not in run:
            continue
        ref_ids, ref_boxes = ref[frame]
        run_ids, run_boxes = run[frame]
        iou = iou_matrix(ref_boxes, run_boxes)

        # For IDF1: every pair that overlaps enough is a possible same-object pair.
        for i, j in zip(*np.nonzero(iou >= IOU_MIN)):
            overlap_frames[(ref_ids[i], run_ids[j])] += 1

        # For precision/recall and ID switches: best one-to-one match in this frame.
        rows, cols = linear_sum_assignment(-iou)
        for i, j in zip(rows, cols):
            if iou[i, j] < IOU_MIN:
                continue
            matched += 1
            r, t = ref_ids[i], run_ids[j]
            if r in last_match and last_match[r] != t:
                switches += 1
            last_match[r] = t

    # IDF1: pair each reference ID with at most one run ID for the whole clip,
    # choosing the pairing that maximises frames in agreement.
    id_tp = 0
    if overlap_frames:
        ref_list = sorted({r for r, _ in overlap_frames})
        run_list = sorted({t for _, t in overlap_frames})
        ri = {r: k for k, r in enumerate(ref_list)}
        ti = {t: k for k, t in enumerate(run_list)}
        counts = np.zeros((len(ref_list), len(run_list)))
        for (r, t), c in overlap_frames.items():
            counts[ri[r], ti[t]] = c
        rows, cols = linear_sum_assignment(-counts)
        id_tp = counts[rows, cols].sum()

    precision = matched / n_run if n_run else 0.0
    recall = matched / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    idf1 = 2 * id_tp / (n_ref + n_run) if n_ref + n_run else 0.0
    return {'ref_boxes': n_ref, 'run_boxes': n_run, 'precision': precision, 'recall': recall,
            'f1': f1, 'idf1': idf1, 'id_switches': switches}


if __name__ == '__main__':
    argv = [a for a in sys.argv[1:] if a != '--no-csv']
    save_csv = '--no-csv' not in sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit('Usage: python src/compare.py <reference.txt> <run.txt> [--no-csv]')
    ref_path, run_path = Path(argv[0]), Path(argv[1])
    m = compare(load_tracks(ref_path), load_tracks(run_path))

    timing_path = run_path.with_name(run_path.stem + '_timing.json')
    fps = json.loads(timing_path.read_text())['fps'] if timing_path.exists() else ''

    print(f'\n{run_path.stem} vs {ref_path.stem}')
    print(f'  boxes        reference {m["ref_boxes"]}, run {m["run_boxes"]}')
    print(f'  precision    {m["precision"]:.3f}')
    print(f'  recall       {m["recall"]:.3f}')
    print(f'  F1           {m["f1"]:.3f}')
    print(f'  IDF1         {m["idf1"]:.3f}')
    print(f'  ID switches  {m["id_switches"]}')
    if fps != '':
        print(f'  FPS          {fps}')

    if not save_csv:
        raise SystemExit(0)
    results = run_path.parent / 'results.csv'
    new_file = not results.exists()
    with open(results, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['run', 'reference', 'fps', 'precision', 'recall', 'f1', 'idf1', 'id_switches'])
        w.writerow([run_path.stem, ref_path.stem, fps, f'{m["precision"]:.3f}', f'{m["recall"]:.3f}',
                    f'{m["f1"]:.3f}', f'{m["idf1"]:.3f}', m['id_switches']])
    print(f'\nAppended to {results}')
