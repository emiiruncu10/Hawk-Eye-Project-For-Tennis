"""Gated video -> TrackNet-input pipeline.

Order of operations is deliberate and enforced:

1. ``require_calibration()`` – loads and re-verifies Phase 1's calibration.
   Raises :class:`CalibrationError` if it is missing or not green. **The
   video file is not opened before this succeeds.**
2. Probe the video and assert its resolution matches the calibration
   (or rescale the homography when ``allow_rescale`` is set and the aspect
   ratio is identical).
3. Stream frames, resize, buffer 3 consecutive frames, stack to 9 channels,
   batch, convert to float32 NCHW tensors on the MPS device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..court.calibration import CourtCalibration, require_calibration
from ..court.homography import CalibrationError
from ..device import get_device
from .preprocess import TrackNetInputSpec, preprocess_frame
from .sequence import FrameSequenceBuffer
from .tensorize import BatchCollator, InputBatch
from .video import VideoInfo, VideoReader, probe_video


@dataclass
class PipelineConfig:
    spec: TrackNetInputSpec = field(default_factory=TrackNetInputSpec.tracknet_v2)
    batch_size: int = 8
    device: str | None = None            # None -> auto (mps > cuda > cpu)
    warmup: str = "skip"                 # "skip" | "repeat"  (see FrameSequenceBuffer)
    keep_original_frames: bool = False   # keep the BGR frame t in each batch (for annotation)
    start_frame: int = 0
    end_frame: int | None = None
    max_sequences: int | None = None
    allow_rescale: bool = False          # adapt calibration if the video has another size but same aspect


@dataclass
class PipelineStats:
    frames_read: int = 0
    sequences: int = 0
    batches: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames_read / self.seconds if self.seconds > 0 else 0.0


class TrackNetInputPipeline:
    """Iterable over :class:`InputBatch` objects for a calibrated video."""

    def __init__(self, video_path: str | Path, calibration_path: str | Path,
                 config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.video_path = Path(video_path)
        self.calibration_path = Path(calibration_path)

        # --- 1. calibration gate (before the video is touched) -------------
        self.calibration: CourtCalibration = require_calibration(self.calibration_path)

        # --- 2. video metadata + resolution consistency --------------------
        self.video_info: VideoInfo = probe_video(self.video_path)
        vsize = self.video_info.size
        if tuple(vsize) != tuple(self.calibration.image_size):
            if not self.config.allow_rescale:
                raise CalibrationError(
                    f"Video resolution {vsize} does not match calibration resolution "
                    f"{tuple(self.calibration.image_size)}. Re-run calibrate.py on this video "
                    f"or pass allow_rescale=True (same aspect ratio only).")
            self.calibration = self.calibration.rescaled(vsize)

        self.device = get_device(self.config.device)
        self.stats = PipelineStats()

    # ------------------------------------------------------------------ iter
    def __iter__(self) -> Iterator[InputBatch]:
        cfg = self.config
        buf = FrameSequenceBuffer(cfg.spec.sequence_length, cfg.warmup, cfg.keep_original_frames)
        collator = BatchCollator(cfg.spec, cfg.batch_size, self.device, self.video_info.size)
        self.stats = PipelineStats()
        t0 = time.perf_counter()
        with VideoReader(self.video_path, cfg.start_frame, cfg.end_frame) as reader:
            for idx, frame, ts in reader:
                self.stats.frames_read += 1
                chw = preprocess_frame(frame, cfg.spec)
                seq = buf.push(idx, chw, ts, frame if cfg.keep_original_frames else None)
                if seq is None:
                    continue
                self.stats.sequences += 1
                batch = collator.add(seq)
                if batch is not None:
                    self.stats.batches += 1
                    self.stats.seconds = time.perf_counter() - t0
                    yield batch
                if cfg.max_sequences is not None and self.stats.sequences >= cfg.max_sequences:
                    break
        batch = collator.flush()
        if batch is not None:
            self.stats.batches += 1
            yield batch
        self.stats.seconds = time.perf_counter() - t0

    # --------------------------------------------------------------- helpers
    @property
    def expected_sequences(self) -> int:
        n = len(VideoReader(self.video_path, self.config.start_frame, self.config.end_frame))
        if self.config.warmup == "skip":
            n = max(0, n - (self.config.spec.sequence_length - 1))
        if self.config.max_sequences is not None:
            n = min(n, self.config.max_sequences)
        return n


def iter_tracknet_batches(video_path: str | Path, calibration_path: str | Path,
                          **config_kwargs) -> Iterator[InputBatch]:
    """Convenience: ``for batch in iter_tracknet_batches("m.mp4", "cal.json", batch_size=4)``."""
    return iter(TrackNetInputPipeline(video_path, calibration_path, PipelineConfig(**config_kwargs)))
