"""Ball trajectory tracker: gating, gap interpolation and smoothing.

For every frame the tracker receives a :class:`BallDetection` and returns a
:class:`TrackPoint` with one of three statuses:

* ``detected``  – the measurement passed the gate and updated the filter
* ``predicted`` – no usable measurement; position comes from the Kalman
                  prediction (interpolated trajectory)
* ``lost``      – the ball has been missing for more than ``max_missing``
                  frames (or no track has started yet); no position

Outliers (detections far from the prediction) are rejected; a run of
consecutive rejections re-initialises the track at the new location.
``finish()`` applies RTS smoothing per track segment and drops the
predicted tail after the last detection of each segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..inference.detector import BallDetection
from .kalman import ConstantAccelerationKalman


@dataclass
class TrackerConfig:
    gate_chi2: float = 9.21              # 99 % chi-square gate with 2 DoF
    max_gate_px: float = 200.0           # hard cap on |measurement - prediction|
    max_missing: int = 8                 # frames to coast on prediction before "lost"
    reinit_after_rejections: int = 3     # consecutive outliers -> start a new track there
    min_confidence: float | None = None  # override detection.visible with a stricter threshold
    # Noise levels are in pixels of the *source* video. Tuned on 1080p broadcast
    # footage (ball speed up to ~60 px/frame at 30 fps, abrupt direction changes
    # at racket contact and bounces); scale them down for smaller videos.
    process_noise: float = 6.0
    measurement_noise: float = 6.0
    smooth: bool = True                  # RTS smoothing in finish()
    roi_mask: np.ndarray | None = None   # (H, W) uint8; candidates outside are ignored


@dataclass
class TrackPoint:
    frame_index: int
    x: float | None
    y: float | None
    status: str                          # "detected" | "predicted" | "lost"
    confidence: float = 0.0
    raw_x: float | None = None           # accepted measurement, if any
    raw_y: float | None = None
    segment: int = -1

    @property
    def has_position(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class _Segment:
    id: int
    kf: ConstantAccelerationKalman
    points: list[TrackPoint] = field(default_factory=list)   # one per frame in this segment
    last_detected_pos: int = 0                                 # index into points


class BallTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._segments: list[_Segment] = []
        self._current: _Segment | None = None
        self._missing = 0
        self._rejections = 0
        self._lost_points: list[TrackPoint] = []
        self._order: list[tuple[int, int, int]] = []   # (frame_index, segment_id or -1, pos)
        self.n_rejected = 0                            # frames with candidates but none inside the gate
        self.n_roi_suppressed = 0                      # candidates dropped by the ROI mask

    # ---------------------------------------------------------------- helpers
    def _new_kf(self) -> ConstantAccelerationKalman:
        c = self.config
        return ConstantAccelerationKalman(process_noise=c.process_noise, measurement_noise=c.measurement_noise)

    def _candidates(self, det: BallDetection) -> list[tuple[float, float, float]]:
        """Usable measurement candidates (x, y, conf), strongest first."""
        if not det.visible:
            return []
        cands = list(det.candidates) if det.candidates else (
            [(det.x, det.y, det.confidence)] if det.x is not None and det.y is not None else [])
        if self.config.min_confidence is not None:
            cands = [c for c in cands if c[2] >= self.config.min_confidence]
        if self.config.roi_mask is not None and cands:
            m = self.config.roi_mask
            h, w = m.shape[:2]
            kept = []
            for c in cands:
                xi, yi = int(round(c[0])), int(round(c[1]))
                if 0 <= xi < w and 0 <= yi < h and m[yi, xi]:
                    kept.append(c)
                else:
                    self.n_roi_suppressed += 1
            cands = kept
        return cands

    def _start_segment(self, frame_index: int, x: float, y: float, conf: float) -> TrackPoint:
        kf = self._new_kf()
        kf.init(x, y)
        seg = _Segment(len(self._segments), kf)
        self._segments.append(seg)
        self._current = seg
        self._missing = 0
        self._rejections = 0
        tp = TrackPoint(frame_index, x, y, "detected", conf, x, y, seg.id)
        seg.points.append(tp)
        seg.last_detected_pos = 0
        self._order.append((frame_index, seg.id, 0))
        return tp

    def _close_segment(self) -> None:
        self._current = None
        self._missing = 0
        self._rejections = 0

    def _lost(self, det: BallDetection) -> TrackPoint:
        tp = TrackPoint(det.frame_index, None, None, "lost", det.confidence)
        self._lost_points.append(tp)
        self._order.append((det.frame_index, -1, len(self._lost_points) - 1))
        return tp

    # ------------------------------------------------------------------- step
    def step(self, det: BallDetection) -> TrackPoint:
        cands = self._candidates(det)
        seg = self._current
        if seg is None:
            if cands:
                x, y, c = cands[0]
                return self._start_segment(det.frame_index, x, y, c)
            return self._lost(det)

        pred = seg.kf.predict()
        if cands:
            # data association: the candidate closest to the prediction (Mahalanobis) inside the gate
            best = None
            for (x, y, c) in cands:
                z = np.array([x, y])
                d2 = seg.kf.mahalanobis2(z)
                if d2 <= self.config.gate_chi2 and np.linalg.norm(z - pred) <= self.config.max_gate_px:
                    if best is None or d2 < best[0]:
                        best = (d2, x, y, c)
            if best is not None:
                _, x, y, c = best
                seg.kf.update(np.array([x, y]))
                self._missing = 0
                self._rejections = 0
                p = seg.kf.position
                tp = TrackPoint(det.frame_index, float(p[0]), float(p[1]), "detected", c, x, y, seg.id)
                seg.points.append(tp)
                seg.last_detected_pos = len(seg.points) - 1
                self._order.append((det.frame_index, seg.id, len(seg.points) - 1))
                return tp
            # every candidate is an outlier
            self._rejections += 1
            self.n_rejected += 1
            if self._rejections >= self.config.reinit_after_rejections:
                # the last few "predictions" were wrong: the ball is really at the new place
                self._close_segment()
                x, y, c = cands[0]
                return self._start_segment(det.frame_index, x, y, c)
        else:
            self._rejections = 0

        # no usable measurement: coast on the prediction
        self._missing += 1
        if self._missing > self.config.max_missing:
            self._close_segment()
            return self._lost(det)
        tp = TrackPoint(det.frame_index, float(pred[0]), float(pred[1]), "predicted",
                        det.confidence, None, None, seg.id)
        seg.points.append(tp)
        self._order.append((det.frame_index, seg.id, len(seg.points) - 1))
        return tp

    # ----------------------------------------------------------------- finish
    def finish(self) -> list[TrackPoint]:
        """Return all points in frame order, RTS-smoothed per segment, with
        the predicted tail after each segment's last detection marked lost."""
        for seg in self._segments:
            n_keep = seg.last_detected_pos + 1
            if self.config.smooth and len(seg.points) >= 2:
                xs = seg.kf.rts_smooth()
                for k in range(min(n_keep, len(xs))):
                    seg.points[k].x = float(xs[k, 0])
                    seg.points[k].y = float(xs[k, 1])
            for tp in seg.points[n_keep:]:
                tp.x = tp.y = None
                tp.status = "lost"
        out = []
        for frame_index, sid, pos in self._order:
            out.append(self._lost_points[pos] if sid < 0 else self._segments[sid].points[pos])
        return out

    @property
    def n_segments(self) -> int:
        return len(self._segments)


def track_summary(points: list[TrackPoint]) -> dict[str, int]:
    s = {"frames": len(points), "detected": 0, "predicted": 0, "lost": 0}
    for p in points:
        s[p.status] += 1
    return s
