"""Pipeline tests with dummy stages and a small generated video: no GPU or models needed."""
import cv2
import numpy as np
import pytest

from gpp import pipeline, report
from gpp.adapters.text import load_classified, video_frame
from gpp.overlay import draw
from gpp.stage import Stage
from gpp.video import VideoReader, probe


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(30):
        frame = np.full((48, 64, 3), i * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class Recorder(Stage):
    def __init__(self, name, every_n=1):
        self.name, self.every_n, self.seen, self.closed = name, every_n, [], False

    def process(self, index, frame):
        self.seen.append(index)
        return index

    def close(self):
        self.closed = True


@pytest.mark.parametrize("threaded", [False, True])
def test_reader_yields_every_frame_in_order(clip, threaded):
    indices = [i for i, _ in VideoReader(clip, threaded=threaded)]
    assert indices == list(range(30))


def test_reader_max_frames_and_early_stop(clip):
    assert len(list(VideoReader(clip, max_frames=5))) == 5
    for i, _ in VideoReader(clip, threaded=True, queue_size=2):
        if i == 3:
            break  # must not hang the decode thread


def test_probe(clip):
    info = probe(clip)
    assert (info.width, info.height, info.frame_count) == (64, 48, 30)


def test_multi_rate_stages_and_held_results(clip):
    fast, slow = Recorder("fast"), Recorder("slow", every_n=10)
    held = []
    pipeline.run(VideoReader(clip), [fast, slow], probe(clip), warmup=0, sync=False,
                 on_frame=lambda i, f, latest: held.append(latest["slow"]))
    assert fast.seen == list(range(30))
    assert slow.seen == [0, 10, 20]
    assert held[:12] == [0] * 10 + [10, 10]  # the slow result is held between its runs
    assert fast.closed and slow.closed


def test_warmup_frames_left_out_of_timing(clip):
    result = pipeline.run(VideoReader(clip), [Recorder("a")], probe(clip), warmup=5, sync=False)
    assert result.timed_frames == 25
    assert result.timings[0].index == 5
    rows = {r["stage"]: r for r in report.summarize(result)}
    assert rows["a"]["runs"] == 25 and rows["frame"]["runs"] == 25


def test_untimed_stage_not_in_report(clip):
    untimed = Recorder("text")
    untimed.timed = False
    result = pipeline.run(VideoReader(clip), [Recorder("a"), untimed], probe(clip), warmup=0, sync=False)
    assert {r["stage"] for r in report.summarize(result)} == {"a", "frame"}


def test_markdown_and_csv(tmp_path, clip):
    result = pipeline.run(VideoReader(clip), [Recorder("a")], probe(clip), warmup=0, sync=False)
    rows = report.summarize(result)
    assert "| a |" in report.to_markdown(rows, result)
    out = tmp_path / "bench.csv"
    report.write_csv(rows, out, extra={"tag": "x"})
    report.write_csv(rows, out, extra={"tag": "y"})
    lines = out.read_text().splitlines()
    assert lines[0].startswith("tag,stage") and len(lines) == 1 + 2 * len(rows)


def test_text_frame_mapping(tmp_path):
    assert video_frame("frame_0001.png", 12) == 0
    assert video_frame("frame_0003.png", 12) == 24
    csv_path = tmp_path / "classified.csv"
    csv_path.write_text(
        "frame,text,confidence,x_min,y_min,x_max,y_max,track,label,reason\n"
        "frame_0002.png,AMMO 30,0.95,10,10,50,20,1,hud,static\n"
        "frame_0002.png,blur,0.40,0,0,5,5,2,unknown,low\n", encoding="utf-8")
    by_frame = load_classified(csv_path, step=12, min_conf=0.8)
    assert list(by_frame) == [12] and by_frame[12][0]["label"] == "hud"


def test_overlay_draws_without_changing_input():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    results = {"depth": np.random.rand(16, 16), "objects": [{"box": (10, 10, 50, 60), "id": 3, "conf": 0.9}],
               "text": [{"box": (60, 70, 120, 90), "label": "hud"}]}
    out = draw(frame, results)
    assert out.shape == frame.shape and out.any() and not frame.any()
