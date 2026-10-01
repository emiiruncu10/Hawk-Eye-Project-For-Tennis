"""Rolling buffer that turns a stream of frames into (t-2, t-1, t) sequences."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameSequence:
    """Three consecutive preprocessed frames plus bookkeeping for frame t."""

    frame_indices: tuple[int, ...]          # chronological, e.g. (10, 11, 12)
    frames: list[np.ndarray]                # chronological, each (3, H, W) float32
    timestamp_s: float | None = None        # of the newest frame
    original_frame: np.ndarray | None = None  # BGR original of the newest frame (optional)

    @property
    def current_index(self) -> int:
        return self.frame_indices[-1]

    @property
    def length(self) -> int:
        return len(self.frames)


class FrameSequenceBuffer:
    """Keeps the last ``length`` frames and emits a sequence for every new frame.

    warmup
        ``"skip"``   – no output until ``length`` real frames have been seen
                       (frames 0 and 1 get no prediction; matches the
                       reference repositories).
        ``"repeat"`` – the first frame is replicated so that a sequence is
                       emitted from the very first frame.
    """

    def __init__(self, length: int = 3, warmup: str = "skip", keep_original: bool = False):
        if length < 1:
            raise ValueError("length must be >= 1")
        if warmup not in ("skip", "repeat"):
            raise ValueError("warmup must be 'skip' or 'repeat'")
        self.length = length
        self.warmup = warmup
        self.keep_original = keep_original
        self._frames: deque[np.ndarray] = deque(maxlen=length)
        self._indices: deque[int] = deque(maxlen=length)
        self.n_pushed = 0
        self.n_emitted = 0

    def reset(self) -> None:
        self._frames.clear()
        self._indices.clear()
        self.n_pushed = 0
        self.n_emitted = 0

    @property
    def is_ready(self) -> bool:
        return len(self._frames) == self.length

    def push(self, frame_index: int, frame_chw: np.ndarray, timestamp_s: float | None = None,
             original_frame: np.ndarray | None = None) -> FrameSequence | None:
        """Add a frame; return the sequence ending at this frame, or None while warming up."""
        if self._indices and frame_index <= self._indices[-1]:
            raise ValueError(f"frame indices must increase (got {frame_index} after {self._indices[-1]})")
        if self.n_pushed == 0 and self.warmup == "repeat":
            for _ in range(self.length - 1):
                self._frames.append(frame_chw)
                self._indices.append(frame_index)
        self._frames.append(frame_chw)
        self._indices.append(frame_index)
        self.n_pushed += 1
        if not self.is_ready:
            return None
        self.n_emitted += 1
        return FrameSequence(
            frame_indices=tuple(self._indices),
            frames=list(self._frames),
            timestamp_s=timestamp_s,
            original_frame=original_frame if self.keep_original else None,
        )
