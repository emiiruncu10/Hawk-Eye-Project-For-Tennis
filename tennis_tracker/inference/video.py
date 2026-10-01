"""Frame extraction from an MP4 with ``cv2.VideoCapture``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int          # as reported by the container (may be approximate)
    fourcc: str

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else float("nan")


def probe_video(path: str | Path) -> VideoInfo:
    """Open the video only long enough to read its metadata."""
    path = Path(path)
    if not path.exists():
        raise VideoError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Could not open video: {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
        if w <= 0 or h <= 0:
            # some containers report nothing until a frame is decoded
            ok, frame = cap.read()
            if not ok:
                raise VideoError(f"Could not decode any frame from {path}")
            h, w = frame.shape[:2]
    finally:
        cap.release()
    return VideoInfo(str(path), w, h, fps, n, fourcc)


class VideoReader:
    """Iterate over (frame_index, frame_bgr, timestamp_s) of a video.

    Usage::

        with VideoReader("match.mp4", start=0, end=None) as vr:
            for idx, frame, t in vr:
                ...
    """

    def __init__(self, path: str | Path, start: int = 0, end: int | None = None, step: int = 1):
        if step < 1:
            raise ValueError("step must be >= 1")
        self.path = Path(path)
        self.start = max(0, start)
        self.end = end
        self.step = step
        self.info = probe_video(self.path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise VideoError(f"Could not open video: {self.path}")
        if self.start > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __len__(self) -> int:
        last = self.info.frame_count if self.end is None else min(self.end, self.info.frame_count)
        return max(0, (last - self.start + self.step - 1) // self.step)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, float]]:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        idx = self.start
        fps = self.info.fps if self.info.fps > 0 else 30.0
        while True:
            if self.end is not None and idx >= self.end:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield idx, frame, idx / fps
            # skip step-1 frames cheaply with grab()
            for _ in range(self.step - 1):
                if not self._cap.grab():
                    return
            idx += self.step
