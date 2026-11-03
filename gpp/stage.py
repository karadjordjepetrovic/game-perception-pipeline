"""The interface every perception stage implements."""
from typing import Any

from .video import VideoInfo


class Stage:
    """One perception stage: load the model in setup(), do per-frame work in process().

    every_n: run on every Nth frame only; the pipeline keeps showing the last result in
             between. Lets slow stages (OCR) run at a lower rate than fast ones (depth).
    timed:   False for stages whose results are computed elsewhere (e.g. in another
             process); they are left out of the timing table.
    """

    name = "stage"
    every_n = 1
    timed = True

    def setup(self, info: VideoInfo) -> None:
        pass

    def process(self, index: int, frame) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass
