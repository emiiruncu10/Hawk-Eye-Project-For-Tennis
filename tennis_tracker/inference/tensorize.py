"""NumPy -> torch conversion, batching and device placement (MPS / CUDA / CPU)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..device import get_device
from .preprocess import TrackNetInputSpec, stack_sequence
from .sequence import FrameSequence


def numpy_to_tensor(arr: np.ndarray):
    """``torch.from_numpy`` with a fallback that does not need the NumPy C-API
    bridge (needed when torch was built against NumPy 1.x but NumPy 2.x is
    installed, which makes ``from_numpy`` raise)."""
    import torch

    arr = np.ascontiguousarray(arr)
    try:
        return torch.from_numpy(arr)
    except (RuntimeError, TypeError):
        dtype = {np.dtype("float32"): torch.float32, np.dtype("float64"): torch.float64,
                 np.dtype("uint8"): torch.uint8, np.dtype("int64"): torch.int64,
                 np.dtype("int32"): torch.int32}.get(arr.dtype)
        if dtype is None:
            raise
        return torch.frombuffer(bytearray(arr.tobytes()), dtype=dtype).reshape(arr.shape)


def sequences_to_tensor(stacks: list[np.ndarray] | np.ndarray, device=None,
                        non_blocking: bool = True):
    """List of (9, H, W) float32 arrays -> torch float32 tensor (N, 9, H, W) on ``device``.

    MPS has no float64 support, so the tensor is always float32.
    """
    import torch

    batch = np.stack(stacks, axis=0) if isinstance(stacks, list) else np.asarray(stacks)
    if batch.ndim != 4:
        raise ValueError(f"expected (N, C, H, W), got {batch.shape}")
    batch = batch.astype(np.float32, copy=False)
    t = numpy_to_tensor(batch)
    dev = get_device() if device is None else torch.device(device)
    if dev.type == "cpu":
        return t
    if dev.type == "mps":
        # MPS transfers are async; pin_memory is not supported there.
        return t.to(dev, non_blocking=non_blocking)
    return t.pin_memory().to(dev, non_blocking=non_blocking)


def check_environment(device=None) -> dict:
    """Report torch/NumPy versions, the NumPy bridge status and the device."""
    import torch

    info = {"numpy": np.__version__, "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "cuda_available": bool(torch.cuda.is_available())}
    try:
        torch.from_numpy(np.zeros(2, np.float32))
        info["numpy_bridge"] = "from_numpy"
    except (RuntimeError, TypeError):
        info["numpy_bridge"] = "frombuffer-fallback"
    dev = get_device() if device is None else torch.device(device)
    info["device"] = str(dev)
    try:
        x = sequences_to_tensor(np.zeros((1, 9, 4, 4), np.float32), dev)
        info["device_ok"] = bool((x * 2).sum().item() == 0.0)
    except Exception as e:  # pragma: no cover - depends on hardware
        info["device_ok"] = False
        info["device_error"] = f"{type(e).__name__}: {e}"
    return info


@dataclass
class InputBatch:
    """One network batch plus everything needed to map results back."""

    tensor: "torch.Tensor"                          # noqa: F821  (N, 9, H, W) float32 on device
    frame_indices: np.ndarray                       # (N,) index of frame t for each row
    sequence_indices: list[tuple[int, ...]]         # full (t-2, t-1, t) per row
    timestamps_s: np.ndarray                        # (N,) float, NaN if unknown
    spec: TrackNetInputSpec
    original_size: tuple[int, int]                  # (W, H) of the source video
    original_frames: list[np.ndarray | None] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def device(self) -> str:
        return str(self.tensor.device)


class BatchCollator:
    """Accumulate :class:`FrameSequence` objects into :class:`InputBatch` es."""

    def __init__(self, spec: TrackNetInputSpec, batch_size: int, device, original_size: tuple[int, int]):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.spec = spec
        self.batch_size = batch_size
        self.device = device
        self.original_size = tuple(original_size)
        self._pending: list[FrameSequence] = []

    def __len__(self) -> int:
        return len(self._pending)

    def add(self, seq: FrameSequence) -> InputBatch | None:
        self._pending.append(seq)
        if len(self._pending) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> InputBatch | None:
        if not self._pending:
            return None
        seqs, self._pending = self._pending, []
        stacks = [stack_sequence(s.frames, self.spec) for s in seqs]
        tensor = sequences_to_tensor(stacks, self.device)
        return InputBatch(
            tensor=tensor,
            frame_indices=np.array([s.current_index for s in seqs], dtype=np.int64),
            sequence_indices=[s.frame_indices for s in seqs],
            timestamps_s=np.array([np.nan if s.timestamp_s is None else s.timestamp_s for s in seqs]),
            spec=self.spec,
            original_size=self.original_size,
            original_frames=[s.original_frame for s in seqs],
        )
