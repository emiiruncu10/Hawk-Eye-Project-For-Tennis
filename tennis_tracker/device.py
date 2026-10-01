"""Torch device selection for Apple Silicon (MPS) with CUDA / CPU fallback."""

from __future__ import annotations


def get_device(prefer: str | None = None) -> "torch.device":  # noqa: F821
    """Return the best available torch device.

    Order: explicit ``prefer`` -> MPS (Apple Silicon) -> CUDA -> CPU.
    Torch is imported lazily so that Phase 1 (pure OpenCV) never needs it.
    """
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
