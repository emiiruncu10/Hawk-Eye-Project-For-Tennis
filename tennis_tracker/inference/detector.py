"""TrackNet inference + heat-map peak extraction (argmax with confidence gate)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocess import model_to_original_coords
from .tensorize import InputBatch


@dataclass
class BallDetection:
    frame_index: int
    x: float | None                 # original video pixels (None when not visible)
    y: float | None
    confidence: float               # peak heat-map value in [0, 1]
    visible: bool                   # confidence >= threshold
    x_model: float | None = None    # peak in model (heat-map) pixels
    y_model: float | None = None
    # all peaks >= threshold in original pixels, strongest first: [(x, y, conf), ...]
    candidates: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "detected" if self.visible else "occluded"


@dataclass
class DetectorConfig:
    threshold: float = 0.5          # below this the ball is "occluded / out of frame"
    subpixel: bool = True           # refine the integer argmax (see peaks_from_heatmaps)
    method: str = "parabolic"       # "parabolic" (log-quadratic fit) | "centroid"
    window: int = 2                 # centroid half-window (2 -> 5x5); unused for parabolic
    output_frame_index: int | None = None   # which heat-map channel is frame t (None: last)
    max_peaks: int = 3              # candidate peaks per frame (the tracker picks the consistent one)
    nms_radius: int = 8             # model px; peaks closer than this are merged
    border: int = 6                 # model px (~18 px at 1080p); edge peaks are padding artefacts / rig objects


def _parabolic_offset(vm: float, v0: float, vp: float) -> float:
    """Sub-pixel offset of the peak of a quadratic through (-1, vm), (0, v0), (1, vp).
    Applied to log-values this is exact for a Gaussian bump."""
    eps = 1e-6
    lm, l0, lp = np.log(max(vm, eps)), np.log(max(v0, eps)), np.log(max(vp, eps))
    denom = lm - 2.0 * l0 + lp
    if denom >= -1e-12:            # not a local maximum (flat / saturated) -> no shift
        return 0.0
    return float(np.clip(0.5 * (lm - lp) / denom, -1.0, 1.0))


def peaks_from_heatmaps(heatmaps: np.ndarray, threshold: float = 0.5, subpixel: bool = True,
                        window: int = 2, method: str = "parabolic",
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Argmax per heat-map. ``heatmaps``: (N, H, W) in [0, 1].

    Returns (x, y, confidence, visible) arrays of length N. x/y are NaN where
    ``confidence < threshold``. With ``subpixel`` the integer argmax is refined:

    * ``"parabolic"`` – 1-D log-quadratic fit through the peak and its two
      neighbours in x and in y (exact for Gaussian heat-maps, unbiased);
    * ``"centroid"``  – intensity-weighted centroid of a (2w+1)^2 window.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    if hm.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {hm.shape}")
    n, h, w = hm.shape
    flat = hm.reshape(n, -1)
    idx = flat.argmax(axis=1)
    conf = flat[np.arange(n), idx]
    ys, xs = np.divmod(idx, w)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    if subpixel and method == "parabolic":
        for i in range(n):
            xi, yi = int(xs[i]), int(ys[i])
            if 0 < xi < w - 1:
                x[i] += _parabolic_offset(hm[i, yi, xi - 1], hm[i, yi, xi], hm[i, yi, xi + 1])
            if 0 < yi < h - 1:
                y[i] += _parabolic_offset(hm[i, yi - 1, xi], hm[i, yi, xi], hm[i, yi + 1, xi])
    elif subpixel and method == "centroid" and window > 0:
        for i in range(n):
            y0, y1 = max(0, ys[i] - window), min(h, ys[i] + window + 1)
            x0, x1 = max(0, xs[i] - window), min(w, xs[i] + window + 1)
            patch = hm[i, y0:y1, x0:x1]
            patch = patch - patch.min()            # background-subtracted centroid
            wsum = float(patch.sum())
            if wsum > 1e-9:
                gy, gx = np.mgrid[y0:y1, x0:x1]
                x[i] = float((patch * gx).sum() / wsum)
                y[i] = float((patch * gy).sum() / wsum)
    visible = conf >= threshold
    x[~visible] = np.nan
    y[~visible] = np.nan
    return x, y, conf.astype(np.float64), visible


def multi_peaks_from_heatmaps(heatmaps: np.ndarray, threshold: float = 0.5, max_peaks: int = 3,
                              nms_radius: int = 8, border: int = 0,
                              subpixel: bool = True) -> list[list[tuple[float, float, float]]]:
    """Up to ``max_peaks`` local maxima per heat-map (greedy non-maximum
    suppression), each >= ``threshold`` and at least ``border`` px from the
    edge. Returns, per image, a list of (x, y, conf) strongest first."""
    hm = np.asarray(heatmaps, dtype=np.float32)
    if hm.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {hm.shape}")
    n, h, w = hm.shape
    out: list[list[tuple[float, float, float]]] = []
    for i in range(n):
        work = hm[i].copy()
        if border > 0:
            work[:border, :] = 0
            work[-border:, :] = 0
            work[:, :border] = 0
            work[:, -border:] = 0
        peaks: list[tuple[float, float, float]] = []
        for _ in range(max(1, max_peaks)):
            idx = int(work.argmax())
            y, x = divmod(idx, w)
            c = float(work[y, x])
            if c < threshold:
                break
            px, py = float(x), float(y)
            if subpixel:
                if 0 < x < w - 1:
                    px += _parabolic_offset(hm[i, y, x - 1], hm[i, y, x], hm[i, y, x + 1])
                if 0 < y < h - 1:
                    py += _parabolic_offset(hm[i, y - 1, x], hm[i, y, x], hm[i, y + 1, x])
            peaks.append((px, py, c))
            work[max(0, y - nms_radius):y + nms_radius + 1, max(0, x - nms_radius):x + nms_radius + 1] = 0
        out.append(peaks)
    return out


class BallDetector:
    """Runs a TrackNet model on :class:`InputBatch` es and returns detections."""

    def __init__(self, model, device, config: DetectorConfig | None = None):
        import torch

        self.config = config or DetectorConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def heatmaps(self, tensor) -> "torch.Tensor":  # noqa: F821
        """(N, 9, H, W) on device -> (N, H, W) heat-map of frame t in [0, 1]."""
        import torch

        with torch.inference_mode():
            hm = self.model.heatmaps(tensor.to(self.device, non_blocking=True))
        k = self.config.output_frame_index
        if k is None:
            k = hm.shape[1] - 1
        return hm[:, k]

    def detect(self, batch: InputBatch) -> list[BallDetection]:
        hm = self.heatmaps(batch.tensor)
        hm_np = hm.detach().float().cpu().numpy()
        cfg = self.config
        peaks = multi_peaks_from_heatmaps(hm_np, cfg.threshold, cfg.max_peaks, cfg.nms_radius,
                                          cfg.border, cfg.subpixel)
        dets = []
        for i, cands in enumerate(peaks):
            frame = int(batch.frame_indices[i])
            if not cands:
                dets.append(BallDetection(frame, None, None, float(hm_np[i].max()), False))
                continue
            orig = model_to_original_coords(np.array([[c[0], c[1]] for c in cands]), batch.spec, batch.original_size)
            cand_orig = [(float(orig[k, 0]), float(orig[k, 1]), float(cands[k][2])) for k in range(len(cands))]
            x0, y0, c0 = cand_orig[0]
            dets.append(BallDetection(frame, x0, y0, c0, True, cands[0][0], cands[0][1], cand_orig))
        return dets
