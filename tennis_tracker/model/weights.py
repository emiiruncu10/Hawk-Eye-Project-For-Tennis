"""Safe checkpoint loading with tolerant key matching.

* ``torch.load(..., weights_only=True)`` – never unpickles arbitrary code.
* Accepts raw state dicts or wrappers (``{"state_dict": ...}``,
  ``{"model_state_dict": ...}``, ``{"model": ...}``).
* Strips a ``module.`` prefix left by ``DataParallel``.
* Falls back to *shape-ordered* matching when key names differ (e.g. a port
  of the same architecture with other attribute names) and reports exactly
  what was and was not loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

_WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")


@dataclass
class WeightsReport:
    path: str
    strategy: str                       # "exact" | "shape-order" | "random"
    n_model_tensors: int
    n_checkpoint_tensors: int
    n_loaded: int
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    truncated_keys: list[str] = field(default_factory=list)   # 1-D tensors cut down to the model size

    @property
    def complete(self) -> bool:
        return self.n_loaded == self.n_model_tensors and not self.missing_keys

    def summary(self) -> str:
        s = (f"{self.strategy}: loaded {self.n_loaded}/{self.n_model_tensors} tensors "
             f"from {self.path} ({self.n_checkpoint_tensors} in checkpoint)")
        if self.missing_keys:
            s += f"; missing {len(self.missing_keys)} (e.g. {self.missing_keys[:3]})"
        if self.unexpected_keys:
            s += f"; unexpected {len(self.unexpected_keys)} (e.g. {self.unexpected_keys[:3]})"
        if self.truncated_keys:
            s += f"; TRUNCATED {len(self.truncated_keys)} oversized 1-D tensors (e.g. {self.truncated_keys[:3]})"
        return s


_RENAME_RULES = (
    # ChgygLin/TrackNetV2-pytorch (tf2torch/track.pt): conv2d_N.conv.* / conv2d_N.bn.* / conv2d_18.*
    (re.compile(r"^conv2d_(\d+)\.conv\.(.+)$"), r"conv\1.block.0.\2"),
    (re.compile(r"^conv2d_(\d+)\.bn\.(.+)$"), r"conv\1.block.2.\2"),
    (re.compile(r"^conv2d_(\d+)\.(weight|bias)$"), r"conv\1.\2"),
)


def _rename_keys(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    """Map known third-party key layouts onto this package's names."""
    out, changed = {}, False
    for k, v in sd.items():
        nk = k
        for pat, rep in _RENAME_RULES:
            if pat.match(k):
                nk = pat.sub(rep, k)
                changed = True
                break
        out[nk] = v
    return out, changed


def _extract_state_dict(obj) -> dict[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint is a {type(obj).__name__}, expected a state dict")
    for k in _WRAPPER_KEYS:
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break
    sd = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if not sd:
        raise ValueError("checkpoint contains no tensors")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_weights(model: nn.Module, path: str | Path, strict: bool = False,
                 unsafe_pickle: bool = False, allow_truncate: bool = False) -> WeightsReport:
    """Load ``path`` into ``model`` (in place) and return a :class:`WeightsReport`.

    ``strict=True`` raises if any model tensor is left unloaded. ``unsafe_pickle``
    allows legacy checkpoints that need full unpickling; only use it for files
    you trust. ``allow_truncate`` accepts 1-D tensors (BatchNorm parameters)
    that are *longer* than the model's by keeping their first entries - a
    workaround for faulty converters; the report lists what was truncated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # pickled objects other than tensors
        if not unsafe_pickle:
            raise RuntimeError(
                f"{path} could not be loaded with weights_only=True ({type(e).__name__}: {e}). "
                "Re-save it as a plain state_dict, or pass unsafe_pickle=True if you trust the file.") from e
        raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(raw)
    sd, renamed = _rename_keys(sd)
    model_sd = model.state_dict()

    # 1. exact key match (optionally with oversized 1-D tensors truncated)
    truncated: list[str] = []
    if set(sd) == set(model_sd):
        fixed = {}
        for k, v in sd.items():
            mv = model_sd[k]
            if v.shape == mv.shape:
                fixed[k] = v
            elif allow_truncate and v.ndim == 1 and mv.ndim == 1 and v.numel() > mv.numel():
                fixed[k] = v[: mv.numel()].clone()
                truncated.append(k)
            else:
                fixed = None
                break
        if fixed is not None:
            model.load_state_dict(fixed, strict=True)
            return WeightsReport(str(path), "renamed" if renamed else "exact", len(model_sd), len(sd),
                                 len(model_sd), truncated_keys=truncated)

    # 2. shape-ordered match: walk both lists in order and pair same-shaped tensors
    model_items = list(model_sd.items())
    ckpt_items = list(sd.items())
    mapped: dict[str, torch.Tensor] = {}
    used_ckpt: set[str] = set()
    j = 0
    for mk, mv in model_items:
        # look ahead a little for the next same-shaped checkpoint tensor
        for jj in range(j, min(j + 6, len(ckpt_items))):
            ck, cv = ckpt_items[jj]
            if ck in used_ckpt:
                continue
            if tuple(cv.shape) == tuple(mv.shape) and cv.dtype == mv.dtype or (
                    tuple(cv.shape) == tuple(mv.shape) and mv.dtype.is_floating_point and cv.dtype.is_floating_point):
                mapped[mk] = cv.to(mv.dtype)
                used_ckpt.add(ck)
                j = jj + 1
                break
    missing = [k for k in model_sd if k not in mapped]
    unexpected = [k for k in sd if k not in used_ckpt]
    if strict and missing:
        raise RuntimeError(f"strict load failed: {len(missing)} model tensors unmatched, e.g. {missing[:5]}")
    model.load_state_dict({**model_sd, **mapped}, strict=True)
    return WeightsReport(str(path), "shape-order", len(model_sd), len(sd), len(mapped), missing, unexpected)


def save_weights(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
