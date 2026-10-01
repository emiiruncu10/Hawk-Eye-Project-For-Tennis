"""Per-frame preprocessing and 9-channel stacking for TrackNet.

Two conventions exist in the reference repositories, so both are supported
through :class:`TrackNetInputSpec`:

===========================  ==========  =======  =============
repository                   resolution  colour   frame order
===========================  ==========  =======  =============
ChgygLin/TrackNetV2-pytorch  288 x 512   RGB      t-2, t-1, t   (oldest first)
yastrebksv/TrackNet          360 x 640   BGR      t, t-1, t-2   (newest first)
===========================  ==========  =======  =============

The default is the TrackNetV2 convention (288x512, RGB, chronological),
which is also what the architecture document describes. Pick the preset
that matches the weights you load in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackNetInputSpec:
    height: int = 288
    width: int = 512
    channel_order: str = "rgb"          # "rgb" | "bgr"
    frame_order: str = "oldest_first"   # "oldest_first" (t-2, t-1, t) | "newest_first" (t, t-1, t-2)
    sequence_length: int = 3
    interpolation: int = field(default=cv2.INTER_AREA, repr=False)

    def __post_init__(self):
        if self.channel_order not in ("rgb", "bgr"):
            raise ValueError("channel_order must be 'rgb' or 'bgr'")
        if self.frame_order not in ("oldest_first", "newest_first"):
            raise ValueError("frame_order must be 'oldest_first' or 'newest_first'")
        if self.sequence_length < 1 or self.height < 1 or self.width < 1:
            raise ValueError("invalid spec dimensions")

    # ---------------------------------------------------------------- presets
    @classmethod
    def tracknet_v2(cls) -> "TrackNetInputSpec":
        """ChgygLin/TrackNetV2-pytorch convention."""
        return cls(height=288, width=512, channel_order="rgb", frame_order="oldest_first")

    @classmethod
    def tracknet_yastrebksv(cls) -> "TrackNetInputSpec":
        """yastrebksv/TrackNet convention (360x640, BGR, newest frame first)."""
        return cls(height=360, width=640, channel_order="bgr", frame_order="newest_first")

    @classmethod
    def from_name(cls, name: str) -> "TrackNetInputSpec":
        presets = {"tracknet_v2": cls.tracknet_v2, "v2": cls.tracknet_v2,
                   "yastrebksv": cls.tracknet_yastrebksv, "tracknet_v1": cls.tracknet_yastrebksv}
        if name not in presets:
            raise ValueError(f"unknown preset '{name}', choose from {sorted(presets)}")
        return presets[name]()

    # ------------------------------------------------------------- properties
    @property
    def n_channels(self) -> int:
        return 3 * self.sequence_length

    @property
    def size_wh(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """(C, H, W) of one network input."""
        return self.n_channels, self.height, self.width

    def scale_factors(self, original_size: tuple[int, int]) -> tuple[float, float]:
        """(sx, sy) such that original_px = model_px * s."""
        ow, oh = original_size
        return ow / self.width, oh / self.height


# --------------------------------------------------------------------------- #
def preprocess_frame(frame_bgr: np.ndarray, spec: TrackNetInputSpec) -> np.ndarray:
    """BGR uint8 frame (H0, W0, 3) -> float32 (3, H, W) in [0, 1].

    Steps: ``cv2.resize`` to the model resolution, optional BGR->RGB,
    HWC->CHW, divide by 255.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) BGR frame")
    if frame_bgr.shape[0] != spec.height or frame_bgr.shape[1] != spec.width:
        frame_bgr = cv2.resize(frame_bgr, (spec.width, spec.height), interpolation=spec.interpolation)
    if spec.channel_order == "rgb":
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    chw = np.transpose(frame_bgr, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.float32) / 255.0


def stack_sequence(frames_chw: Sequence[np.ndarray], spec: TrackNetInputSpec) -> np.ndarray:
    """Concatenate ``sequence_length`` preprocessed frames along the channel axis.

    ``frames_chw`` must be in chronological order (t-2, t-1, t); the spec's
    ``frame_order`` decides how they are laid out in the 9-channel tensor.
    Returns float32 (3 * L, H, W).
    """
    if len(frames_chw) != spec.sequence_length:
        raise ValueError(f"expected {spec.sequence_length} frames, got {len(frames_chw)}")
    for f in frames_chw:
        if f.shape != (3, spec.height, spec.width):
            raise ValueError(f"frame shape {f.shape} does not match spec {(3, spec.height, spec.width)}")
    ordered = list(frames_chw) if spec.frame_order == "oldest_first" else list(frames_chw)[::-1]
    return np.concatenate(ordered, axis=0)


def model_to_original_coords(xy_model: np.ndarray, spec: TrackNetInputSpec,
                             original_size: tuple[int, int]) -> np.ndarray:
    """Scale (x, y) points from the model resolution back to original pixels."""
    sx, sy = spec.scale_factors(original_size)
    pts = np.asarray(xy_model, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts


def original_to_model_coords(xy_orig: np.ndarray, spec: TrackNetInputSpec,
                             original_size: tuple[int, int]) -> np.ndarray:
    sx, sy = spec.scale_factors(original_size)
    pts = np.asarray(xy_orig, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] /= sx
    pts[:, 1] /= sy
    return pts
