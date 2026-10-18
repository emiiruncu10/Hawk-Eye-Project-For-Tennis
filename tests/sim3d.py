"""3-D ball flight + pinhole camera simulator for bounce / line-call tests.

World frame: X across the court, Y along it (far baseline 0 -> near baseline
23.77, towards the camera), h = height above the ground. Internally the
camera works in the right-handed frame (X, Y, -h).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis_tracker.court import geometry as G
from tennis_tracker.court.calibration import CourtCalibration
from tennis_tracker.court.homography import apply_homography
from tennis_tracker.tracking.tracker import TrackPoint

BALL_R = 0.033


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray            # centre in (X, Y, -h)
    width: int
    height: int

    @classmethod
    def behind_baseline(cls, width=1280, height=720, focal=1100.0, cam=(5.485, 37.77, 7.0),
                        target=(5.485, 11.0, 0.0)) -> "Camera":
        C = np.array([cam[0], cam[1], -cam[2]], float)
        T = np.array([target[0], target[1], -target[2]], float)
        fwd = (T - C) / np.linalg.norm(T - C)
        right = np.cross([0.0, 0.0, 1.0], fwd)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])
        K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        return cls(K, R, C, width, height)

    def project(self, pts_xyh) -> np.ndarray:
        P = np.atleast_2d(np.asarray(pts_xyh, float))
        W = np.column_stack([P[:, 0], P[:, 1], -P[:, 2]])
        pc = (W - self.C) @ self.R.T
        uv = pc @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    @property
    def H_inv(self) -> np.ndarray:
        """Ground-plane homography metres -> pixels."""
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], -self.R @ self.C])

    def calibration(self) -> CourtCalibration:
        H = np.linalg.inv(self.H_inv)
        kp = apply_homography(self.H_inv, G.COURT_KEYPOINTS_M)
        cal = CourtCalibration(H=H / H[2, 2], image_size=(self.width, self.height), keypoints_px=kp,
                               keypoint_source=["measured"] * 14, method="auto", reproj_rmse_px=0.5,
                               n_inliers=14, line_support=0.9)
        assert cal.verify().passed
        return cal


def simulate(p0, v0, t_end, events=(), e=0.75, mu=0.85, g=9.81):
    """Piecewise-analytic flight. ``events``: [(t, new_velocity)] racket hits.

    Returns a function pos(t) -> (X, Y, h) and the list of true bounces
    [(t, X, Y)] (contact point on the ground).
    """
    segs = []                         # (t0, p0, v0)
    bounces = []
    t, p, v = 0.0, np.array(p0, float), np.array(v0, float)
    events = sorted(events)
    ev = 0
    while t < t_end:
        t_next_ev = events[ev][0] if ev < len(events) else np.inf
        # time until the ball centre reaches the radius height: h + vh*s - g/2 s^2 = R
        a, b, c = -0.5 * g, v[2], p[2] - BALL_R
        disc = b * b - 4 * a * c
        s_hit = (-b - np.sqrt(disc)) / (2 * a) if disc >= 0 else np.inf
        if s_hit <= 1e-9:
            s_hit = np.inf
        segs.append((t, p.copy(), v.copy()))
        if t + s_hit < min(t_next_ev, t_end):
            p = p + v * s_hit + np.array([0, 0, -0.5 * g * s_hit ** 2])
            v = v + np.array([0, 0, -g * s_hit])
            t = t + s_hit
            bounces.append((t, p[0], p[1]))
            v = np.array([v[0] * mu, v[1] * mu, -e * v[2]])
            p[2] = BALL_R
        elif t_next_ev < t_end:
            s = t_next_ev - t
            p = p + v * s + np.array([0, 0, -0.5 * g * s ** 2])
            t = t_next_ev
            v = np.array(events[ev][1], float)
            ev += 1
        else:
            break

    def pos(tq):
        k = max(i for i, sg in enumerate(segs) if sg[0] <= tq + 1e-12)
        t0, q0, w0 = segs[k]
        s = tq - t0
        return q0 + w0 * s + np.array([0, 0, -0.5 * g * s * s])

    return pos, bounces


def track_points(cam: Camera, pos, times, frames, noise_px=0.7, seed=0) -> list[TrackPoint]:
    rng = np.random.default_rng(seed)
    pts = []
    for f, tq in zip(frames, times):
        uv = cam.project(pos(tq))[0] + rng.normal(0, noise_px, 2)
        pts.append(TrackPoint(int(f), float(uv[0]), float(uv[1]), "detected", 0.9, float(uv[0]), float(uv[1]), 0))
    return pts
