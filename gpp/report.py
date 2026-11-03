"""Turns a RunResult into a per-stage table: median, p95 and mean milliseconds."""
import csv
import statistics


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def summarize(result):
    """One row per stage plus a 'frame' row for the whole frame, in run order."""
    per_stage = {}
    for t in result.timings:
        for name, ms in t.stages.items():
            per_stage.setdefault(name, []).append(ms)
    per_stage["frame"] = [t.total_ms for t in result.timings]

    rows = []
    for name, values in per_stage.items():
        values = sorted(values)
        rows.append({
            "stage": name,
            "runs": len(values),
            "median_ms": statistics.median(values) if values else 0.0,
            "p95_ms": _percentile(values, 0.95),
            "mean_ms": statistics.fmean(values) if values else 0.0,
        })
    return rows


def to_markdown(rows, result=None, title=None):
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += ["| Stage | Runs | Median | p95 | Mean |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['runs']} | {r['median_ms']:.2f} ms | "
                     f"{r['p95_ms']:.2f} ms | {r['mean_ms']:.2f} ms |")
    if result is not None:
        lines += ["", f"Wall clock: {result.timed_frames} frames in {result.wall_seconds:.2f} s "
                      f"= {result.wall_fps:.1f} FPS (first {result.warmup} frames left out)."]
    return "\n".join(lines) + "\n"


def write_csv(rows, path, extra=None):
    """Append rows to a CSV, writing the header if the file is new. extra = columns added to each row."""
    extra = extra or {}
    fields = list(extra) + ["stage", "runs", "median_ms", "p95_ms", "mean_ms"]
    try:
        new = not open(path).readline()
    except FileNotFoundError:
        new = True
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        for r in rows:
            writer.writerow({**extra, **{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}})
