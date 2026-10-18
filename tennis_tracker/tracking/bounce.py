"""Bounce detection from a tracked ball trajectory (single static camera).

Physics used
------------
At a bounce the ball's *vertical* velocity reverses while its *horizontal*
velocity carries on almost unchanged. Seen from a camera behind a baseline
this shows up in the image as a sharp corner in the ball's vertical image
coordinate with a smooth horizontal coordinate. Concretely, around every
candidate frame the raw measurements are fitted with a continuous
piecewise-quadratic model that allows a jump in velocity at a knot ``tau``:

    y(t) = a + b*u + c*u^2 + d*max(u, 0) + e*max(u, 0)^2,   u = t - tau

(the same for x). ``d`` is the velocity jump at the knot; the curvature may
differ on both sides (``e``) because perspective bends the image of the
rising and the falling arc differently. In simulation this halves the
depth error and keeps the reported uncertainty honest, compared with a
shared curvature (``e = 0``). The free-curvature model needs at least
``min_points_free`` measurements on each side; with fewer (gaps around the
bounce, end of clip) the shared-curvature model is used. The knot is
searched at sub-frame resolution, which also yields the sub-frame bounce
time and image position.

A knot is accepted as a bounce when

1. the vertical image velocity jumps *upwards* by at least ``min_jump_mps``
   (velocities are converted to metres per second with the local image
   scale at the bounce point) and the jump is statistically significant
   (``|d| / se(d) >= min_tstat``, ``min_tstat_near`` in the near half);
2. the horizontal jump is small compared with the vertical one
   (``max_lateral_ratio``) – racket hits change both;
3. the bounce point, mapped through the ground-plane homography, lies on or
   around the court (a ball high in the air maps far beyond the court);
4. the direction of travel is consistent with a bounce on that half: in the
   far half the ball must move *up* in the image right after the bounce
   (rising and receding) and must not move down fast before it (that is a
   serve or overhead contact); in the near half it must move *down* right
   before it (falling and approaching), and the upward image speed after it
   must stay below ``near_max_rebound_ratio`` times the downward speed
   before it (a bounce loses vertical speed, a racket hit adds some).

After that, a rally-grammar rule drops a second event on the same half
within ``same_half_gap_s``: the ball has to cross the net and come back
before it can bounce there again, so it is a racket hit or a second bounce,
neither of which decides a line call.

Validation (see README): on 8 clips, 69 hand-verified bounces were all found
with no accepted false positive; in simulation, lateral error is ~1 cm and
depth error ~4 cm (near half) / ~12 cm (far half), with honest uncertainty.

Duplicated video frames (common in broadcast footage converted from 25 to
30 fps) are removed first because they fake a sudden stop of the ball. The
remaining distinct frames are re-timed uniformly, which is when they were
actually captured (see :class:`TimeBase`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..court import geometry as G
from ..court.calibration import CourtCalibration
from ..court.homography import apply_homography
from .tracker import TrackPoint

BALL_RADIUS_M = 0.033


@dataclass
class BounceConfig:
    window_s: float = 0.2             # half-width of the fitting window around a candidate
    min_points_per_side: int = 2      # measurements strictly before and after the knot
    min_points_free: int = 3          # per side, to use the free-curvature model
    max_gap_s: float = 0.12           # largest gap between consecutive measurements in a window
    knot_search_frames: float = 1.0   # knot searched within +- this many frames of the candidate
    knot_step_frames: float = 0.1
    min_jump_mps: float = 3.5         # minimum upward jump of the vertical velocity (m/s)
    min_tstat: float = 3.0            # significance of the jump (far half)
    min_tstat_near: float = 3.5       # near half (fast, large image motion; the shared-curvature
                                      # fit gives lower t for very fast serves across tracking gaps)
    max_lateral_ratio: float = 0.5    # |horizontal jump| <= ratio * |vertical jump|
    far_min_rise_mps: float = 1.0     # far half: ball must move up in the image after the bounce
    far_min_fall_mps: float = -1.5    # far half: ...and must not already be rising fast before it (racket hit);
                                      # a fast flat shot can cancel its fall in the image completely
    far_max_fall_mps: float = 8.0     # far half: a receding ball partly cancels its fall in the image;
                                      # a fast downward image motion is a serve/overhead contact instead
    near_min_fall_mps: float = 2.0    # near half: ball must move down in the image before the bounce
    near_max_rebound_ratio: float = 0.8   # near half: upward / downward image speed across the knot.
                                      # A bounce cannot exceed the ball's restitution (~0.75, the limit
                                      # for a camera at ground level); a solid racket hit is ~1. Measured
                                      # with short straight-line fits right next to the knot.
    rebound_points: int = 3           # points per side for those straight-line fits
    court_margin_side_m: float = 3.0  # plausible bounce region around the doubles court
    court_margin_end_m: float = 4.0   # (a ball high in the air near a baseline maps further behind it)
    min_separation_s: float = 0.2     # two bounces cannot be closer than this
    same_half_gap_s: float = 1.0      # a second bounce on the same half sooner than this is a hit or a
                                      # second bounce (the ball must cross the net and come back first)
    pixel_sigma_floor: float = 1.0    # px; floor for the measurement noise (measured ~0.7-1.2 px)
    duplicate_tol_px: float = 0.5     # consecutive frames closer than this count as duplicated


@dataclass
class Bounce:
    frame: float                      # sub-frame time of the bounce, in frames
    time_s: float
    x_px: float                       # fitted ball centre at the bounce (image)
    y_px: float
    contact_x_px: float               # ground contact point (centre shifted down by the radius)
    contact_y_px: float
    court_x_m: float                  # contact point on the court plane
    court_y_m: float
    half: str                         # "near" | "far"
    jump_vy_mps: float                # vertical velocity jump (negative = upwards)
    jump_vx_mps: float
    tstat: float
    vy_before_mps: float
    vy_after_mps: float
    px_per_m: float                   # local image scale at the bounce point
    rebound_ratio: float = float("nan")  # -vy_after / vy_before from straight-line fits next to the knot
    cov_px: np.ndarray = field(repr=False, default_factory=lambda: np.eye(2))  # position covariance (px^2)
    cov_m: np.ndarray = field(repr=False, default_factory=lambda: np.eye(2))   # same, on the court (m^2)

    @property
    def sigma_x_m(self) -> float:
        return float(np.sqrt(max(self.cov_m[0, 0], 0.0)))

    @property
    def sigma_y_m(self) -> float:
        return float(np.sqrt(max(self.cov_m[1, 1], 0.0)))


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def read_track_csv(path: str | Path) -> list[TrackPoint]:
    """Read a ``*_tracked.csv`` written by ``track.py`` back into TrackPoints."""
    def num(v):
        return None if v in ("", None) else float(v)

    pts = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            pts.append(TrackPoint(int(r["frame"]), num(r["x_px"]), num(r["y_px"]), r["status"],
                                  float(r["confidence"]), num(r["raw_x_px"]), num(r["raw_y_px"]),
                                  int(r["segment"])))
    return pts


def measurements(points: list[TrackPoint], duplicate_tol_px: float = 0.5,
                 duplicate_frames=()) -> np.ndarray:
    """Raw measured positions (frame, x, y) of detected frames, duplicated frames removed.

    ``duplicate_frames`` (from :func:`find_duplicate_frames`) are dropped; as a
    fallback, consecutive measurements closer than ``duplicate_tol_px`` are too.
    """
    dup = set(int(f) for f in duplicate_frames)
    rows = [(p.frame_index, p.raw_x, p.raw_y) for p in points
            if p.status == "detected" and p.raw_x is not None and p.raw_y is not None
            and p.frame_index not in dup]
    rows.sort()
    out = []
    for f, x, y in rows:
        if out and f == out[-1][0] + 1 and abs(x - out[-1][1]) <= duplicate_tol_px \
                and abs(y - out[-1][2]) <= duplicate_tol_px:
            continue                       # repeated video frame: no new information
        out.append((f, x, y))
    return np.array(out, dtype=np.float64).reshape(-1, 3)


class TimeBase:
    """Capture time of every video frame, accounting for duplicated frames.

    Without duplicates ``t = frame / fps``. With duplicates (e.g. 25 fps
    material in a 30 fps file) the distinct frames are spaced uniformly at
    ``fps_eff = fps * n_distinct / n_frames`` and a duplicate shares the time
    of the frame it repeats.
    """

    def __init__(self, n_frames: int, fps: float, duplicates=()):
        dup = set(int(f) for f in duplicates)
        n_frames = max(int(n_frames), 1)
        new = np.array([0 if f in dup else 1 for f in range(n_frames)], dtype=np.float64)
        new[0] = 1.0
        k = np.cumsum(new) - 1.0
        self.fps = float(fps)
        self.fps_eff = self.fps * float(new.sum()) / n_frames
        self.duplicates = dup
        self._t = k / self.fps_eff
        self._distinct = np.flatnonzero(new > 0)

    def time(self, frames) -> np.ndarray:
        f = np.asarray(frames, dtype=np.float64)
        return np.interp(f, np.arange(len(self._t)), self._t)

    def frame_at(self, t: float) -> float:
        """Inverse of :meth:`time` on the distinct frames (sub-frame result)."""
        return float(np.interp(t, self._t[self._distinct], self._distinct.astype(np.float64)))


def find_duplicate_frames(video_path: str | Path, max_changed_px: int = 4,
                          size: tuple[int, int] = (480, 270), diff_threshold: int = 12) -> list[int]:
    """Frames that repeat their predecessor: fewer than ``max_changed_px``
    pixels (at ``size``) change by more than ``diff_threshold`` grey levels.
    Real frames of a live scene change hundreds of pixels; repeats change 0-3
    (re-encoding noise). Even a lone small moving ball changes more than 4."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    dups, prev, f = [], None, 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev is not None and int((np.abs(g - prev) > diff_threshold).sum()) <= max_changed_px:
                dups.append(f)
            prev, f = g, f + 1
    finally:
        cap.release()
    return dups


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def local_scale(cal: CourtCalibration, x: float, y: float) -> tuple[float, np.ndarray]:
    """Image pixels per metre at an image point (lateral direction on the ground)."""
    g = apply_homography(cal.H, np.array([[x, y]]))[0]
    a = apply_homography(cal.H_inv, np.array([[g[0] - 0.5, g[1]], [g[0] + 0.5, g[1]]]))
    return float(np.linalg.norm(a[1] - a[0])), g


def pixel_to_court_jacobian(cal: CourtCalibration, x: float, y: float, eps: float = 0.5) -> np.ndarray:
    """2x2 Jacobian d(court metres)/d(image pixels) at an image point."""
    p = np.array([[x - eps, y], [x + eps, y], [x, y - eps], [x, y + eps]])
    g = apply_homography(cal.H, p)
    return np.column_stack([(g[1] - g[0]) / (2 * eps), (g[3] - g[2]) / (2 * eps)])


# --------------------------------------------------------------------------- #
# Piecewise fit
# --------------------------------------------------------------------------- #
@dataclass
class _KnotFit:
    tau: float
    rss: float
    coef_x: np.ndarray
    coef_y: np.ndarray
    cov_x: np.ndarray
    cov_y: np.ndarray
    sigma: float


def _design(u: np.ndarray, free: bool) -> np.ndarray:
    up = np.maximum(u, 0.0)
    cols = [np.ones_like(u), u, u * u, up] + ([up * up] if free else [])
    return np.column_stack(cols)


def _fit_knot(t: np.ndarray, x: np.ndarray, y: np.ndarray, tau: float, sigma_floor: float,
              free: bool = True) -> _KnotFit | None:
    A = _design(t - tau, free)
    n_par = A.shape[1]
    if len(t) <= n_par or np.linalg.matrix_rank(A) < n_par:
        return None
    AtA_inv = np.linalg.inv(A.T @ A)
    cx = AtA_inv @ A.T @ x
    cy = AtA_inv @ A.T @ y
    rss_x = float(np.sum((A @ cx - x) ** 2))
    rss_y = float(np.sum((A @ cy - y) ** 2))
    dof = max(1, len(t) - n_par)
    sx = max(np.sqrt(rss_x / dof), sigma_floor)      # x and y noise differ (motion blur is
    sy = max(np.sqrt(rss_y / dof), sigma_floor)      # mostly along the direction of travel)
    return _KnotFit(tau, rss_x + rss_y, cx, cy, AtA_inv * sx ** 2, AtA_inv * sy ** 2, max(sx, sy))


def detect_bounces(points: list[TrackPoint], cal: CourtCalibration, fps: float,
                   config: BounceConfig | None = None, duplicate_frames=(),
                   n_frames: int | None = None) -> list[Bounce]:
    """Find ball bounces in a tracked trajectory. Returns bounces in time order.

    ``duplicate_frames`` lists repeated video frames (see
    :func:`find_duplicate_frames`); they are ignored and the remaining frames
    are re-timed accordingly.
    """
    cfg = config or BounceConfig()
    m = measurements(points, cfg.duplicate_tol_px, duplicate_frames)
    if len(m) < 2 * cfg.min_points_per_side + 1:
        return []
    if n_frames is None:
        n_frames = max(p.frame_index for p in points) + 1
    tb = TimeBase(n_frames, fps, duplicate_frames)
    frames, xs, ys = m[:, 0], m[:, 1], m[:, 2]
    times = tb.time(frames)
    dt = 1.0 / tb.fps_eff
    candidates: list[Bounce] = []

    for i in range(len(m)):
        idx = np.flatnonzero(np.abs(times - times[i]) <= cfg.window_s + 1e-9)
        # keep the contiguous run (no gap larger than max_gap_s) that contains i
        lo = hi = int(np.flatnonzero(idx == i)[0])
        gap = max(cfg.max_gap_s, 1.5 * dt) + 1e-6       # tolerance for float frame times
        while lo > 0 and times[idx[lo]] - times[idx[lo - 1]] <= gap:
            lo -= 1
        while hi < len(idx) - 1 and times[idx[hi + 1]] - times[idx[hi]] <= gap:
            hi += 1
        idx = idx[lo:hi + 1]
        if len(idx) <= 4:
            continue
        t = times[idx] - times[i]

        def search(lo_off: float, hi_off: float, free: bool) -> _KnotFit | None:
            best_fit = None
            for off in np.arange(lo_off, hi_off + 1e-9, cfg.knot_step_frames):
                tau = off * dt
                n_before = int(np.sum(t < tau - 1e-9))
                n_after = int(np.sum(t > tau + 1e-9))
                need = cfg.min_points_free if free else cfg.min_points_per_side
                if n_before < need or n_after < need:
                    continue
                fit = _fit_knot(t, xs[idx], ys[idx], tau, cfg.pixel_sigma_floor, free)
                if fit is not None and (best_fit is None or fit.rss < best_fit.rss):
                    best_fit = fit
            return best_fit

        # 1) detection: shared-curvature model (robust one-sided velocities). The jump, its
        #    significance and every hit-vs-bounce rule are evaluated on this fit.
        vel = search(-cfg.knot_search_frames, cfg.knot_search_frames, free=False)
        if vel is None:
            continue
        # 2) localisation: free-curvature model, knot re-optimised within +-0.5 frame, when
        #    there are enough points on both sides (about half the depth error in simulation,
        #    with an honest covariance); otherwise the detection fit is used.
        k0 = vel.tau / dt
        best = search(k0 - 0.5, k0 + 0.5, free=True) or vel
        bx, by = float(best.coef_x[0]), float(best.coef_y[0])
        ppm, _ = local_scale(cal, bx, by)
        if not np.isfinite(ppm) or ppm <= 0:
            continue
        d_y, d_x = float(vel.coef_y[3]), float(vel.coef_x[3])
        se_dy = float(np.sqrt(vel.cov_y[3, 3]))
        vy_before = float(vel.coef_y[1])
        vy_after = vy_before + d_y
        # instantaneous vertical speeds right next to the knot (no curvature assumption)
        left = np.flatnonzero(t < vel.tau - 1e-9)[-cfg.rebound_points:]
        right = np.flatnonzero(t > vel.tau + 1e-9)[:cfg.rebound_points]
        rebound = float("nan")
        if len(left) >= 2 and len(right) >= 2:
            v_l = np.polyfit(t[left], ys[idx][left], 1)[0]
            v_r = np.polyfit(t[right], ys[idx][right], 1)[0]
            if v_l > 0:
                rebound = float(-v_r / v_l)
        # ball centre -> ground contact point (radius straight down in the image)
        cx_px, cy_px = bx, by + BALL_RADIUS_M * ppm
        g = apply_homography(cal.H, np.array([[cx_px, cy_px]]))[0]
        if not np.all(np.isfinite(g)):
            continue
        J = pixel_to_court_jacobian(cal, cx_px, cy_px)
        calib_var = cal.reproj_rmse_px ** 2 if np.isfinite(cal.reproj_rmse_px) else 0.0
        cov_px = np.diag([best.cov_x[0, 0] + calib_var, best.cov_y[0, 0] + calib_var])
        cov_m = J @ cov_px @ J.T
        t_bounce = float(times[i] + best.tau)
        candidates.append(Bounce(
            frame=tb.frame_at(t_bounce), time_s=t_bounce,
            x_px=bx, y_px=by, contact_x_px=cx_px, contact_y_px=cy_px,
            court_x_m=float(g[0]), court_y_m=float(g[1]),
            half="far" if g[1] < G.NET_Y else "near",
            jump_vy_mps=d_y / ppm, jump_vx_mps=d_x / ppm,
            tstat=abs(d_y) / se_dy if se_dy > 0 else 0.0,
            vy_before_mps=vy_before / ppm, vy_after_mps=vy_after / ppm,
            px_per_m=ppm, rebound_ratio=rebound, cov_px=cov_px, cov_m=cov_m))

    # Each event is judged by its strongest knot: non-maximum suppression *before* the rules,
    # so that a weaker neighbouring knot of a rejected racket hit cannot slip through.
    kinks = sorted((b for b in candidates if b.jump_vy_mps <= -cfg.min_jump_mps), key=lambda b: b.jump_vy_mps)
    events: list[Bounce] = []
    for b in kinks:
        if all(abs(b.time_s - k.time_s) >= cfg.min_separation_s for k in events):
            events.append(b)
    keep = sorted((b for b in events if is_bounce(b, cfg)), key=lambda b: b.frame)
    # rally grammar: after a bounce the ball must cross the net (be hit back) before it can
    # bounce on that half again; an event on the same half too soon is a racket hit or a
    # second bounce, neither of which decides a line call.
    out: list[Bounce] = []
    for b in keep:
        if out and out[-1].half == b.half and b.time_s - out[-1].time_s < cfg.same_half_gap_s:
            continue
        out.append(b)
    return out


def is_bounce(b: Bounce, cfg: BounceConfig) -> bool:
    """Apply the physical acceptance rules to a fitted knot."""
    if b.jump_vy_mps > -cfg.min_jump_mps:
        return False
    if b.tstat < (cfg.min_tstat_near if b.half == "near" else cfg.min_tstat):
        return False
    if abs(b.jump_vx_mps) > cfg.max_lateral_ratio * abs(b.jump_vy_mps):
        return False
    if not (G.X_DOUBLES_L - cfg.court_margin_side_m <= b.court_x_m <= G.X_DOUBLES_R + cfg.court_margin_side_m):
        return False
    if not (G.Y_BASE_FAR - cfg.court_margin_end_m <= b.court_y_m <= G.Y_BASE_NEAR + cfg.court_margin_end_m):
        return False
    if b.half == "far":
        return (b.vy_after_mps <= -cfg.far_min_rise_mps
                and cfg.far_min_fall_mps <= b.vy_before_mps <= cfg.far_max_fall_mps)
    if b.vy_before_mps < cfg.near_min_fall_mps:
        return False
    ratio = b.rebound_ratio if np.isfinite(b.rebound_ratio) else -b.vy_after_mps / b.vy_before_mps
    return ratio <= cfg.near_max_rebound_ratio
