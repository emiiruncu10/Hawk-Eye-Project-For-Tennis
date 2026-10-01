"""Classical (non-learned) tennis-court line & keypoint detector.

Pipeline
--------
1. Downscale the reference frame to a working width.
2. **White-line mask** – luminance threshold + Farin's "line structure"
   constraint (a pixel must be brighter than its neighbours ``tau`` pixels
   away, horizontally or vertically). This keeps thin bright lines and
   discards large white blobs (banners, shirts, crowd).
3. Optional **court-colour mask** – dominant hue in the centre of the frame
   is taken as the playing-surface colour; white pixels far away from that
   surface are discarded (removes stadium clutter in broadcast footage).
4. **Probabilistic Hough transform** -> segments -> merged into infinite
   lines with a support length; split into horizontal / vertical groups.
5. **Model fitting** – every (2 horizontal, 2 vertical) image-line combination
   is paired with every (2 horizontal, 2 vertical) court-model-line
   combination. Each hypothesis gives a 4-point homography whose projected
   court lines are scored against a distance transform of the white mask.
   This is done fully vectorised in NumPy (tens of thousands of hypotheses
   in ~1-2 s).
6. **Refinement** – with the best hypothesis, every model line is matched to
   a detected image line; keypoints are recomputed as exact line
   intersections and a final RANSAC homography is fitted through
   ``cv2.findHomography``.

The output keypoints follow the 14-point convention in :mod:`geometry`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as G
from .homography import (
    CalibrationError,
    HomographyResult,
    apply_homography,
    estimate_homography,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    work_width: int = 1280               # processing resolution (px)

    # white-line mask
    lum_min: int = 120                   # min grey level of a line pixel (far lines on dusty clay ~145)
    white_delta: int = 24                # min contrast vs. neighbours (thin far lines on bright clay ~25)
    tau_frac: float = 0.0065             # neighbour distance as fraction of width
    use_court_mask: bool = True
    court_mask_dilate_frac: float = 0.03 # grow the court region by this * width
    # keep only lines that have playing-surface colour on at least one side
    # (removes crowd / grandstand / banner edges on courts whose surround has
    # the same colour as the court, e.g. clay)
    surface_filter: bool = True
    surface_offset_frac: float = 0.012   # perpendicular sampling offset as fraction of width
    surface_min_frac: float = 0.5        # fraction of samples on a side that must be surface

    # Hough
    hough_threshold: int = 40
    min_line_length_frac: float = 0.035
    max_line_gap_frac: float = 0.012

    # line merging / classification
    merge_angle_deg: float = 2.0
    merge_dist_px: float = 7.0
    horizontal_max_angle_deg: float = 45.0
    max_lines_per_orientation: int = 10
    # mutual extent pruning: horizontals must cross the vertical span of the
    # strongest vertical lines and vice versa (court lines always do; crowd,
    # banner and grandstand edges usually do not)
    extent_prune: bool = True
    extent_top_n: int = 6
    extent_margin_frac: float = 0.06

    # hypothesis scoring
    sample_step_m: float = 0.10
    score_sigma_px: float = 4.0
    min_quad_area_frac: float = 0.03
    max_quad_area_frac: float = 3.0
    min_corner_sep_frac: float = 0.02    # reject quads with corners closer than this * width
    min_line_sep_frac: float = 0.01      # image-line pairs must be at least this far apart
    n_white_samples: int = 2500          # white pixels used for the recall term
    recall_tol_m: float = 0.25           # a white pixel is "explained" if within this of a model line

    # refinement
    refine_angle_deg: float = 6.0
    refine_dist_px: float = 14.0
    refine_overlap_frac: float = 0.3
    ransac_thresh_px: float = 4.0
    support_dist_px: float = 3.0


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    """Infinite 2-D line with the extent of its detected support."""

    a: float          # normalised line coefficients a*x + b*y + c = 0, a^2+b^2 = 1
    b: float
    c: float
    p0: np.ndarray    # extent start (px)
    p1: np.ndarray    # extent end   (px)
    support: float    # total supporting segment length (px)
    n_segments: int = 1

    @property
    def coeffs(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c], dtype=np.float64)

    @property
    def direction(self) -> np.ndarray:
        return np.array([-self.b, self.a], dtype=np.float64)

    @property
    def angle_deg(self) -> float:
        """Angle of the direction vector from the x-axis in [0, 180)."""
        d = self.direction
        return math.degrees(math.atan2(d[1], d[0])) % 180.0

    def distance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return np.abs(pts @ np.array([self.a, self.b]) + self.c)

    def x_at_y(self, y: float) -> float:
        return (-self.c - self.b * y) / self.a if abs(self.a) > 1e-9 else float("nan")

    def y_at_x(self, x: float) -> float:
        return (-self.c - self.a * x) / self.b if abs(self.b) > 1e-9 else float("nan")


def line_intersection(l1: Line, l2: Line) -> np.ndarray:
    """Intersection of two lines; returns [nan, nan] if parallel."""
    p = np.cross(l1.coeffs, l2.coeffs)
    if abs(p[2]) < 1e-9:
        return np.array([np.nan, np.nan])
    return p[:2] / p[2]


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


@dataclass
class DetectionResult:
    keypoints_px: np.ndarray                     # 14 x 2, ORIGINAL resolution, NaN = not measured
    keypoint_source: list[str]                   # "measured" | "estimated"
    homography: HomographyResult                 # pixels(original) -> metres
    line_support: float                          # fraction of model-line samples on white pixels
    best_hypothesis_score: float
    matched_lines: dict[str, Line]               # model line name -> detected line (work res)
    lines_h: list[Line] = field(default_factory=list)
    lines_v: list[Line] = field(default_factory=list)
    white_mask: np.ndarray | None = None         # work res
    court_mask: np.ndarray | None = None         # work res
    work_scale: float = 1.0                      # work px = original px * work_scale
    n_hypotheses: int = 0

    @property
    def H(self) -> np.ndarray:
        return self.homography.H

    @property
    def keypoints_all_px(self) -> np.ndarray:
        """All 14 keypoints: measured ones as detected, missing ones reprojected."""
        kp = self.keypoints_px.copy()
        miss = ~np.all(np.isfinite(kp), axis=1)
        if miss.any():
            kp[miss] = apply_homography(self.homography.H_inv, G.COURT_KEYPOINTS_M[miss])
        return kp


# --------------------------------------------------------------------------- #
# Step 1-3: masks
# --------------------------------------------------------------------------- #
def white_line_mask(gray: np.ndarray, tau: int, delta: int, lum_min: int) -> np.ndarray:
    """Farin-style line-structure mask: bright *thin* structures only."""
    g = gray.astype(np.int16)
    pad = np.pad(g, tau, mode="edge")
    h, w = g.shape
    left = pad[tau:tau + h, 0:w]
    right = pad[tau:tau + h, 2 * tau:2 * tau + w]
    up = pad[0:h, tau:tau + w]
    down = pad[2 * tau:2 * tau + h, tau:tau + w]
    horiz = (g - left > delta) & (g - right > delta)
    vert = (g - up > delta) & (g - down > delta)
    mask = (g > lum_min) & (horiz | vert)
    return mask.astype(np.uint8) * 255


def surface_color_mask(bgr: np.ndarray) -> np.ndarray | None:
    """Raw per-pixel mask of the dominant (playing-surface) hue in the centre
    of the frame. Returns None when the court is grey / unsaturated."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    roi = hsv[int(0.30 * h):int(0.90 * h), int(0.20 * w):int(0.80 * w)]
    sat, val = roi[..., 1], roi[..., 2]
    good = (sat > 35) & (val > 40)
    if good.mean() < 0.25:
        return None  # grey / unsaturated court, hue is meaningless
    hist = np.bincount(roi[..., 0][good].ravel(), minlength=180).astype(np.float64)
    hist = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(7), "valid")
    hue0 = int(np.argmax(hist))
    if hist[hue0] < 0.15 * good.sum():
        return None
    dh = np.abs(hsv[..., 0].astype(np.int16) - hue0)
    dh = np.minimum(dh, 180 - dh)
    return ((dh <= 12) & (hsv[..., 1] > 30) & (hsv[..., 2] > 35)).astype(np.uint8) * 255


def court_surface_mask(bgr: np.ndarray, dilate_px: int, color_mask: np.ndarray | None = None) -> np.ndarray | None:
    """Mask of the playing-surface region (largest blob of the dominant hue,
    holes filled, dilated). Returns None when no confident surface is found."""
    m = surface_color_mask(bgr) if color_mask is None else color_mask.copy()
    if m is None:
        return None
    h, w = m.shape[:2]
    k = max(3, int(round(w * 0.012)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.08 * h * w:
        return None
    m = (lab == largest).astype(np.uint8) * 255
    # fill holes (players, logos) and grow to include the boundary lines
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, big)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return m


# --------------------------------------------------------------------------- #
# Step 4: segments -> lines
# --------------------------------------------------------------------------- #
def _fit_line(points: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Weighted total-least-squares line through points. Returns (a, b, c, p0, p1)."""
    wsum = weights.sum()
    mean = (points * weights[:, None]).sum(0) / wsum
    d = points - mean
    cov = (d * weights[:, None]).T @ d / wsum
    _, vecs = np.linalg.eigh(cov)
    direction = vecs[:, -1]
    normal = np.array([-direction[1], direction[0]])
    a, b = normal
    c = -normal @ mean
    t = d @ direction
    p0 = mean + direction * t.min()
    p1 = mean + direction * t.max()
    return float(a), float(b), float(c), p0, p1


def merge_segments(segments: np.ndarray, angle_tol_deg: float, dist_tol_px: float) -> list[Line]:
    """Greedy merge of Hough segments (N x 4) into infinite lines with support."""
    if segments is None or len(segments) == 0:
        return []
    segs = segments.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)

    groups: list[dict] = []
    for i in order:
        x0, y0, x1, y1 = segs[i]
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        ends = np.array([[x0, y0], [x1, y1]])
        best = None
        for gi, g in enumerate(groups):
            if angle_diff_deg(ang, g["line"].angle_deg) > angle_tol_deg:
                continue
            if g["line"].distance(ends).max() > dist_tol_px:
                continue
            best = gi
            break
        if best is None:
            groups.append({
                "pts": [ends[0], ends[1]],
                "w": [lengths[i], lengths[i]],
                "line": Line(*_line_from_two(ends[0], ends[1]), ends[0], ends[1], lengths[i], 1),
            })
        else:
            g = groups[best]
            g["pts"] += [ends[0], ends[1]]
            g["w"] += [lengths[i], lengths[i]]
            pts = np.array(g["pts"])
            w = np.array(g["w"])
            a, b, c, p0, p1 = _fit_line(pts, w)
            g["line"] = Line(a, b, c, p0, p1, g["line"].support + lengths[i], g["line"].n_segments + 1)
    lines = [g["line"] for g in groups]
    lines.sort(key=lambda L: -L.support)
    return lines


def _line_from_two(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    d = q - p
    n = np.array([-d[1], d[0]])
    n /= np.linalg.norm(n) + 1e-12
    return float(n[0]), float(n[1]), float(-n @ p)


def filter_lines_by_surface(lines: list[Line], color_mask: np.ndarray, offset_px: float,
                            min_frac: float, n_samples: int = 40) -> list[Line]:
    """Keep lines that run along the playing surface: at least ``min_frac`` of
    the samples on one side (offset perpendicular to the line) must be
    surface-coloured. Court lines always have court on at least one side;
    crowd, banner and grandstand edges do not."""
    h, w = color_mask.shape[:2]
    kept = []
    for L in lines:
        d = L.p1 - L.p0
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d = d / n
        normal = np.array([-d[1], d[0]])
        t = np.linspace(0.05, 0.95, n_samples)[:, None]
        base = L.p0[None, :] + t * (L.p1 - L.p0)[None, :]
        fr = []
        for sign in (1, -1):
            pts = base + sign * offset_px * normal[None, :]
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            fr.append(float(((color_mask[yi, xi] > 0) & inside).mean()))
        if max(fr) >= min_frac:
            kept.append(L)
    return kept


def prune_by_mutual_extent(lines_h: list[Line], lines_v: list[Line], image_size: tuple[int, int],
                           top_n: int = 6, margin_frac: float = 0.06) -> tuple[list[Line], list[Line]]:
    """Keep horizontal lines that pass through the y-span of the strongest
    vertical lines, and vertical lines that pass through the x-span of the
    strongest horizontal lines. Lists must be sorted by support."""
    w, h = image_size
    if len(lines_h) < 2 or len(lines_v) < 2:
        return lines_h, lines_v
    strong_v = lines_v[:top_n]
    strong_h = lines_h[:top_n]
    y_lo = min(min(L.p0[1], L.p1[1]) for L in strong_v) - margin_frac * h
    y_hi = max(max(L.p0[1], L.p1[1]) for L in strong_v) + margin_frac * h
    x_lo = min(min(L.p0[0], L.p1[0]) for L in strong_h) - margin_frac * w
    x_hi = max(max(L.p0[0], L.p1[0]) for L in strong_h) + margin_frac * w
    x_mid = 0.5 * (min(L.x_at_y(0.5 * (y_lo + y_hi)) for L in strong_v if np.isfinite(L.x_at_y(0.5 * (y_lo + y_hi))))
                   + max(L.x_at_y(0.5 * (y_lo + y_hi)) for L in strong_v if np.isfinite(L.x_at_y(0.5 * (y_lo + y_hi)))))
    y_mid = 0.5 * (y_lo + y_hi)
    kept_h = [L for L in lines_h if np.isfinite(L.y_at_x(x_mid)) and y_lo <= L.y_at_x(x_mid) <= y_hi]
    kept_v = [L for L in lines_v if np.isfinite(L.x_at_y(y_mid)) and x_lo <= L.x_at_y(y_mid) <= x_hi]
    if len(kept_h) < 2 or len(kept_v) < 2:
        return lines_h, lines_v
    return kept_h, kept_v


def detect_lines(mask: np.ndarray, cfg: DetectorConfig) -> tuple[list[Line], list[Line]]:
    """White mask -> (horizontal lines, vertical lines), each sorted by support."""
    h, w = mask.shape[:2]
    segs = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 360, threshold=cfg.hough_threshold,
        minLineLength=int(cfg.min_line_length_frac * w),
        maxLineGap=int(cfg.max_line_gap_frac * w),
    )
    lines = merge_segments(segs, cfg.merge_angle_deg, cfg.merge_dist_px)
    horiz, vert = [], []
    for L in lines:
        ang = L.angle_deg
        if min(ang, 180 - ang) <= cfg.horizontal_max_angle_deg:
            horiz.append(L)
        else:
            vert.append(L)
    return horiz, vert


# --------------------------------------------------------------------------- #
# Step 5: model fitting (vectorised)
# --------------------------------------------------------------------------- #
def _batched_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4-point DLT for B hypotheses. src, dst: (B, 4, 2). Returns (B, 3, 3)
    mapping src -> dst. Rows with degenerate input yield NaN matrices."""
    B = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    r1 = np.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], axis=-1)
    r2 = np.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], axis=-1)
    A = np.concatenate([r1, r2], axis=1)  # (B, 8, 9)
    bad = ~np.all(np.isfinite(A.reshape(B, -1)), axis=1)
    A[bad] = np.eye(8, 9)[None]
    _, _, vh = np.linalg.svd(A)
    Hs = vh[:, -1, :].reshape(B, 3, 3)
    Hs = Hs / (Hs[:, 2:3, 2:3] + 1e-18)
    Hs[bad] = np.nan
    return Hs


def _score_hypotheses(H_inv: np.ndarray, samples_m: np.ndarray, dt: np.ndarray,
                      sigma: float, quad_m: np.ndarray, cfg: DetectorConfig,
                      white_px: np.ndarray | None = None,
                      model_dt: tuple[np.ndarray, float, float] | None = None) -> np.ndarray:
    """Score B metres->pixels homographies (B,3,3).

    score = harmonic mean of
      * precision: fraction of projected model-line samples lying on white pixels
      * recall:    fraction of white pixels that map close to a model line
    Recall is what kills degenerate hypotheses (whole court squashed onto one
    bright line), which otherwise reach ~100 % precision.
    """
    B = H_inv.shape[0]
    h, w = dt.shape
    P = np.hstack([samples_m, np.ones((len(samples_m), 1))])  # (N, 3)
    Q = np.hstack([quad_m, np.ones((4, 1))])
    use_recall = white_px is not None and model_dt is not None and len(white_px) > 0
    if use_recall:
        Wp = np.hstack([white_px, np.ones((len(white_px), 1))])
        mdt, m_origin, m_res = model_dt
        mh, mw = mdt.shape
    scores = np.zeros(B)
    chunk = 512
    for s in range(0, B, chunk):
        Hc = H_inv[s:s + chunk]
        ok = np.all(np.isfinite(Hc.reshape(len(Hc), -1)), axis=1)
        # quad plausibility: consistent depth sign, convex, sane area
        q = np.einsum("bij,nj->bni", Hc, Q)
        wq = q[..., 2]
        ok &= (np.all(wq > 1e-9, axis=1) | np.all(wq < -1e-9, axis=1))
        qxy = q[..., :2] / np.where(np.abs(wq[..., None]) < 1e-12, np.nan, wq[..., None])
        # corners must be well separated (rejects squashed / sliver quads)
        dif = qxy[:, :, None, :] - qxy[:, None, :, :]
        cd = np.sqrt(np.nansum(dif ** 2, axis=-1))
        cd[:, np.arange(4), np.arange(4)] = np.inf
        ok &= np.nanmin(cd.reshape(len(Hc), -1), axis=1) > cfg.min_corner_sep_frac * w
        # shoelace area of quad in order 0,1,3,2 (TL, TR, BR, BL)
        poly = qxy[:, [0, 1, 3, 2], :]
        xs, ys = poly[..., 0], poly[..., 1]
        cross = xs * np.roll(ys, -1, axis=1) - np.roll(xs, -1, axis=1) * ys
        area = 0.5 * np.abs(np.nansum(cross, axis=1))
        ok &= (area > cfg.min_quad_area_frac * h * w) & (area < cfg.max_quad_area_frac * h * w)
        # convexity: all consecutive edge cross products share a sign
        e = np.roll(poly, -1, axis=1) - poly
        e2 = np.roll(e, -1, axis=1)
        cz = e[..., 0] * e2[..., 1] - e[..., 1] * e2[..., 0]
        ok &= (np.all(cz > 0, axis=1) | np.all(cz < 0, axis=1))
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        p = np.einsum("bij,nj->bni", Hc[idx], P)
        wp = p[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = p[..., 0] / wp
            py = p[..., 1] / wp
        inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        xi = np.clip(np.nan_to_num(px), 0, w - 1).astype(np.int32)
        yi = np.clip(np.nan_to_num(py), 0, h - 1).astype(np.int32)
        d = dt[yi, xi]
        val = np.exp(-(d / sigma) ** 2) * inside
        precision = val.mean(axis=1)
        if not use_recall:
            scores[s + idx] = precision
            continue
        Hp = np.linalg.inv(Hc[idx])                       # pixels -> metres
        m = np.einsum("bij,nj->bni", Hp, Wp)
        wm = m[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = (m[..., 0] / wm - m_origin) / m_res
            my = (m[..., 1] / wm - m_origin) / m_res
        inb = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx < mw) & (my >= 0) & (my < mh)
        cx = np.clip(np.nan_to_num(mx), 0, mw - 1).astype(np.int32)
        cy = np.clip(np.nan_to_num(my), 0, mh - 1).astype(np.int32)
        explained = (mdt[cy, cx] <= cfg.recall_tol_m) & inb
        recall = explained.mean(axis=1)
        scores[s + idx] = 2 * precision * recall / (precision + recall + 1e-9)
    return scores


def fit_court_model(lines_h: list[Line], lines_v: list[Line], dt: np.ndarray,
                    cfg: DetectorConfig, white_mask: np.ndarray | None = None,
                    rng_seed: int = 0) -> tuple[np.ndarray, float, int, tuple]:
    """Exhaustive search over line-pair assignments. Returns
    (best metres->pixels H_inv, score, n_hypotheses, assignment)."""
    if len(lines_h) < 2 or len(lines_v) < 2:
        raise CalibrationError(
            f"Not enough candidate lines (horizontal={len(lines_h)}, vertical={len(lines_v)})")
    h, w = dt.shape
    Lh = lines_h[:cfg.max_lines_per_orientation]
    Lv = lines_v[:cfg.max_lines_per_orientation]

    # Order image lines: horizontals top->bottom, verticals left->right.
    y_ref = float(np.median([L.y_at_x(w / 2) for L in Lh if np.isfinite(L.y_at_x(w / 2))] or [h / 2]))
    Lh = sorted(Lh, key=lambda L: L.y_at_x(w / 2) if np.isfinite(L.y_at_x(w / 2)) else (L.p0[1] + L.p1[1]) / 2)
    Lv = sorted(Lv, key=lambda L: L.x_at_y(y_ref) if np.isfinite(L.x_at_y(y_ref)) else (L.p0[0] + L.p1[0]) / 2)

    h_pairs = list(itertools.combinations(range(len(Lh)), 2))
    v_pairs = list(itertools.combinations(range(len(Lv)), 2))
    hm_pairs = list(itertools.combinations(range(len(G.HORIZONTAL_MODEL_LINES)), 2))
    vm_pairs = list(itertools.combinations(range(len(G.VERTICAL_MODEL_LINES)), 2))

    # Precompute all image intersections
    inter = np.full((len(Lh), len(Lv), 2), np.nan)
    for i, Hl in enumerate(Lh):
        for j, Vl in enumerate(Lv):
            inter[i, j] = line_intersection(Hl, Vl)

    # Near-coincident line pairs cannot bound a court: drop them up front.
    def _sep_ok(La: Line, Lb: Line, horizontal: bool) -> bool:
        if horizontal:
            xs = np.linspace(0.2 * w, 0.8 * w, 5)
            da = np.array([La.y_at_x(x) for x in xs]); db = np.array([Lb.y_at_x(x) for x in xs])
            return bool(np.nanmax(np.abs(da - db)) > cfg.min_line_sep_frac * h)
        ys = np.linspace(0.2 * h, 0.8 * h, 5)
        da = np.array([La.x_at_y(y) for y in ys]); db = np.array([Lb.x_at_y(y) for y in ys])
        return bool(np.nanmax(np.abs(da - db)) > cfg.min_line_sep_frac * w)

    h_pairs = [(i, j) for i, j in h_pairs if _sep_ok(Lh[i], Lh[j], True)]
    v_pairs = [(k, l) for k, l in v_pairs if _sep_ok(Lv[k], Lv[l], False)]

    src_list, dst_list, assign = [], [], []
    for (i, j) in h_pairs:
        for (k, l) in v_pairs:
            img = np.array([inter[i, k], inter[i, l], inter[j, k], inter[j, l]])
            if not np.all(np.isfinite(img)):
                continue
            for (mi, mj) in hm_pairs:
                y_a, y_b = G.HORIZONTAL_MODEL_LINES[mi][1], G.HORIZONTAL_MODEL_LINES[mj][1]
                for (mk, ml) in vm_pairs:
                    x_a, x_b = G.VERTICAL_MODEL_LINES[mk][1], G.VERTICAL_MODEL_LINES[ml][1]
                    src_list.append([[x_a, y_a], [x_b, y_a], [x_a, y_b], [x_b, y_b]])
                    dst_list.append(img)
                    assign.append((i, j, k, l, mi, mj, mk, ml))
    if not src_list:
        raise CalibrationError("No valid line-pair hypotheses (all candidate lines parallel?)")

    src = np.array(src_list, dtype=np.float64)
    dst = np.array(dst_list, dtype=np.float64) / w    # normalise pixels for the DLT
    Hn = _batched_homography(src, dst)
    S = np.diag([w, w, 1.0])
    H_inv = np.einsum("ij,bjk->bik", S, Hn)

    samples = G.sample_model_lines_m(cfg.sample_step_m)
    quad_m = G.COURT_KEYPOINTS_M[[0, 1, 2, 3]]
    white_px = None
    model_dt = None
    if white_mask is not None:
        ys, xs = np.nonzero(white_mask)
        if len(xs) > 0:
            rng = np.random.default_rng(rng_seed)
            pick = rng.choice(len(xs), size=min(cfg.n_white_samples, len(xs)), replace=False)
            white_px = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float64)
            model_dt = G.model_distance_map()
    scores = _score_hypotheses(H_inv, samples, dt, cfg.score_sigma_px, quad_m, cfg,
                               white_px=white_px, model_dt=model_dt)
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        raise CalibrationError("No court hypothesis matched the white-line evidence")
    return H_inv[best], float(scores[best]), len(scores), (Lh, Lv, assign[best])


# --------------------------------------------------------------------------- #
# Step 6: refinement
# --------------------------------------------------------------------------- #
def match_model_lines(H_inv: np.ndarray, lines: list[Line], cfg: DetectorConfig) -> dict[str, Line]:
    """Assign a detected line to every model line that has support in the image."""
    matched: dict[str, Line] = {}
    used: set[int] = set()
    model = [(n, (np.array([x0, y]), np.array([x1, y]))) for n, y, (x0, x1) in G.HORIZONTAL_MODEL_LINES]
    model += [(n, (np.array([x, y0]), np.array([x, y1]))) for n, x, (y0, y1) in G.VERTICAL_MODEL_LINES]
    candidates = []
    for name, (m0, m1) in model:
        q = apply_homography(H_inv, np.array([m0, m1]))
        if not np.all(np.isfinite(q)):
            continue
        q0, q1 = q
        seg_dir = q1 - q0
        seg_len = np.linalg.norm(seg_dir)
        if seg_len < 1e-6:
            continue
        seg_dir /= seg_len
        seg_ang = math.degrees(math.atan2(seg_dir[1], seg_dir[0])) % 180.0
        for li, L in enumerate(lines):
            if angle_diff_deg(seg_ang, L.angle_deg) > cfg.refine_angle_deg:
                continue
            dist = L.distance(np.array([q0, q1])).max()
            if dist > cfg.refine_dist_px:
                continue
            # extent overlap along the projected segment
            t0, t1 = sorted([(L.p0 - q0) @ seg_dir, (L.p1 - q0) @ seg_dir])
            overlap = max(0.0, min(t1, seg_len) - max(t0, 0.0))
            if overlap < cfg.refine_overlap_frac * seg_len:
                continue
            candidates.append((dist, name, li))
    for dist, name, li in sorted(candidates):
        if name in matched or li in used:
            continue
        matched[name] = lines[li]
        used.add(li)
    return matched


def keypoints_from_lines(matched: dict[str, Line]) -> tuple[np.ndarray, list[str]]:
    kp = np.full((G.NUM_KEYPOINTS, 2), np.nan)
    src = ["estimated"] * G.NUM_KEYPOINTS
    for idx, (hn, vn) in G.KEYPOINT_FROM_LINES.items():
        if hn in matched and vn in matched:
            p = line_intersection(matched[hn], matched[vn])
            if np.all(np.isfinite(p)):
                kp[idx] = p
                src[idx] = "measured"
    return kp, src


def line_support_fraction(H_inv: np.ndarray, dt: np.ndarray, cfg: DetectorConfig) -> float:
    """Fraction of model-line samples that land within ``support_dist_px`` of a white pixel."""
    samples = G.sample_model_lines_m(cfg.sample_step_m)
    p = apply_homography(H_inv, samples)
    h, w = dt.shape
    ok = np.all(np.isfinite(p), axis=1)
    inside = ok & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if inside.sum() == 0:
        return 0.0
    d = dt[p[inside, 1].astype(int), p[inside, 0].astype(int)]
    return float((d <= cfg.support_dist_px).sum() / len(samples))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect_court(image_bgr: np.ndarray, cfg: DetectorConfig | None = None) -> DetectionResult:
    """Detect the tennis court in a single reference frame.

    Raises :class:`CalibrationError` if no plausible court is found.
    """
    cfg = cfg or DetectorConfig()
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("expected a BGR colour image")
    H0, W0 = image_bgr.shape[:2]
    scale = min(1.0, cfg.work_width / W0)
    work = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr.copy()
    h, w = work.shape[:2]

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    tau = max(3, int(round(cfg.tau_frac * w)))
    white = white_line_mask(gray, tau, cfg.white_delta, cfg.lum_min)

    court_mask = None
    color_mask = surface_color_mask(work) if (cfg.use_court_mask or cfg.surface_filter) else None
    if cfg.use_court_mask and color_mask is not None:
        court_mask = court_surface_mask(work, int(cfg.court_mask_dilate_frac * w), color_mask)
        if court_mask is not None:
            white = cv2.bitwise_and(white, court_mask)

    # Bridge tiny gaps so Hough sees continuous lines
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(cv2.bitwise_not(white), cv2.DIST_L2, 5).astype(np.float64)

    lines_h, lines_v = detect_lines(white, cfg)
    if cfg.surface_filter and color_mask is not None:
        off = cfg.surface_offset_frac * w
        fh = filter_lines_by_surface(lines_h, color_mask, off, cfg.surface_min_frac)
        fv = filter_lines_by_surface(lines_v, color_mask, off, cfg.surface_min_frac)
        if len(fh) >= 2 and len(fv) >= 2:
            lines_h, lines_v = fh, fv
    if cfg.extent_prune:
        lines_h, lines_v = prune_by_mutual_extent(lines_h, lines_v, (w, h), cfg.extent_top_n, cfg.extent_margin_frac)
    H_inv0, score, n_hyp, (Lh, Lv, assign) = fit_court_model(lines_h, lines_v, dt, cfg, white)

    # Refinement: match all detected lines to model lines, recompute keypoints
    # as exact intersections, re-fit; iterate once more with the better H.
    all_lines = lines_h + lines_v
    matched = match_model_lines(H_inv0, all_lines, cfg)
    kp_work, src = keypoints_from_lines(matched)
    H_inv_cur = H_inv0
    for _ in range(2):
        if int(np.sum(np.array(src) == "measured")) < 4:
            break
        try:
            res_work = estimate_homography(kp_work, G.COURT_KEYPOINTS_M, cfg.ransac_thresh_px)
        except CalibrationError:
            break
        H_inv_cur = res_work.H_inv
        matched2 = match_model_lines(H_inv_cur, all_lines, cfg)
        kp2, src2 = keypoints_from_lines(matched2)
        if int(np.sum(np.array(src2) == "measured")) < int(np.sum(np.array(src) == "measured")):
            break
        matched, kp_work, src = matched2, kp2, src2

    # Back to original resolution
    kp_orig = kp_work / scale
    S = np.diag([scale, scale, 1.0])          # original px -> work px
    if int(np.sum(np.array(src) == "measured")) >= 4:
        res = estimate_homography(kp_orig, G.COURT_KEYPOINTS_M, cfg.ransac_thresh_px / scale)
    else:
        # Too few keypoint intersections: fall back to the 4 corner
        # correspondences of the winning hypothesis (not necessarily keypoints).
        i, j, k, l, mi, mj, mk, ml = assign
        ya, yb = G.HORIZONTAL_MODEL_LINES[mi][1], G.HORIZONTAL_MODEL_LINES[mj][1]
        xa, xb = G.VERTICAL_MODEL_LINES[mk][1], G.VERTICAL_MODEL_LINES[ml][1]
        img = np.array([line_intersection(Lh[i], Lv[k]), line_intersection(Lh[i], Lv[l]),
                        line_intersection(Lh[j], Lv[k]), line_intersection(Lh[j], Lv[l])]) / scale
        wld = np.array([[xa, ya], [xb, ya], [xa, yb], [xb, yb]])
        res = estimate_homography(img, wld, use_ransac=False)
    support = line_support_fraction(np.linalg.inv(res.H @ np.linalg.inv(S)), dt, cfg)

    return DetectionResult(
        keypoints_px=kp_orig,
        keypoint_source=src,
        homography=res,
        line_support=support,
        best_hypothesis_score=score,
        matched_lines=matched,
        lines_h=lines_h,
        lines_v=lines_v,
        white_mask=white,
        court_mask=court_mask,
        work_scale=scale,
        n_hypotheses=n_hyp,
    )
