"""Homography estimation (pixels <-> court metres) and calibration verification.

The homography ``H`` maps *image pixels* (u, v, 1) to *court metres* (x, y, 1):

    [x', y', w']^T = H @ [u, v, 1]^T ;  x = x'/w', y = y'/w'

``H_inv`` goes the other way. Both are 3x3 float64 matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as G


class CalibrationError(RuntimeError):
    """Raised when a calibration is missing or did not pass verification."""


# --------------------------------------------------------------------------- #
# Basic projection helpers
# --------------------------------------------------------------------------- #
def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project N x 2 points through a 3x3 homography (returns N x 2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts.copy()
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = hom @ H.T
    w = out[:, 2:3]
    # Points at infinity / behind the camera get NaN instead of exploding.
    with np.errstate(divide="ignore", invalid="ignore"):
        res = out[:, :2] / w
    res[np.abs(w[:, 0]) < 1e-12] = np.nan
    return res


def pixel_to_court(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Image pixels -> court metres."""
    return apply_homography(H, pts_px)


def court_to_pixel(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    """Court metres -> image pixels (uses the inverse of H)."""
    return apply_homography(np.linalg.inv(H), pts_m)


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
@dataclass
class HomographyResult:
    H: np.ndarray                       # pixels -> metres
    inlier_mask: np.ndarray             # bool, one per input correspondence
    reproj_errors_px: np.ndarray        # per-point error in pixels (NaN if not used)
    rmse_px: float
    max_err_px: float
    n_points: int
    n_inliers: int

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)


def estimate_homography(
    pts_px: np.ndarray,
    pts_m: np.ndarray,
    ransac_thresh_px: float = 4.0,
    use_ransac: bool = True,
) -> HomographyResult:
    """Compute pixels -> metres homography with ``cv2.findHomography``.

    Parameters
    ----------
    pts_px : N x 2 image points (may contain NaN rows = unknown keypoints)
    pts_m  : N x 2 matching world points in metres
    ransac_thresh_px : RANSAC reprojection threshold *in image pixels*.
        The residual is evaluated in pixel space (metres -> pixels via H_inv)
        because that is where the measurement noise lives.
    """
    pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    pts_m = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
    if pts_px.shape != pts_m.shape:
        raise ValueError("pts_px and pts_m must have the same shape")

    valid = np.all(np.isfinite(pts_px), axis=1) & np.all(np.isfinite(pts_m), axis=1)
    idx = np.flatnonzero(valid)
    if len(idx) < 4:
        raise CalibrationError(
            f"Need at least 4 valid point correspondences, got {len(idx)}"
        )

    src = pts_px[idx].astype(np.float32)
    dst = pts_m[idx].astype(np.float32)

    # RANSAC on the *inverse* direction (metres -> pixels) so the threshold is
    # in pixels. Then invert. cv2 refines the result with Levenberg-Marquardt.
    if use_ransac and len(idx) >= 5:
        H_inv, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransac_thresh_px,
                                         maxIters=5000, confidence=0.999)
    else:
        H_inv, mask = cv2.findHomography(dst, src, 0)
    if H_inv is None or not np.all(np.isfinite(H_inv)) or abs(np.linalg.det(H_inv)) < 1e-12:
        raise CalibrationError("cv2.findHomography failed (degenerate configuration)")

    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(idx), bool)
    if inl.sum() < 4:
        raise CalibrationError("Fewer than 4 RANSAC inliers; keypoints are inconsistent")

    # Re-fit on inliers only (least squares) for the final matrix.
    H_inv, _ = cv2.findHomography(dst[inl], src[inl], 0)
    H_inv = H_inv.astype(np.float64)
    H = np.linalg.inv(H_inv)
    H /= H[2, 2]

    # Per-point reprojection error in pixels
    proj = apply_homography(H_inv, pts_m[idx])
    errs = np.linalg.norm(proj - pts_px[idx], axis=1)

    full_mask = np.zeros(len(pts_px), bool)
    full_mask[idx[inl]] = True
    full_err = np.full(len(pts_px), np.nan)
    full_err[idx] = errs

    return HomographyResult(
        H=H,
        inlier_mask=full_mask,
        reproj_errors_px=full_err,
        rmse_px=float(np.sqrt(np.mean(errs[inl] ** 2))),
        max_err_px=float(np.max(errs[inl])),
        n_points=int(len(idx)),
        n_inliers=int(inl.sum()),
    )


# --------------------------------------------------------------------------- #
# Verification ("green light" logic)
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    critical: bool = True   # non-critical checks only produce warnings


@dataclass
class VerificationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.critical)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.critical]

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.critical]

    def add(self, name: str, passed: bool, detail: str, critical: bool = True) -> None:
        self.checks.append(CheckResult(name, bool(passed), detail, critical))

    def format(self) -> str:
        lines = []
        for c in self.checks:
            tag = "PASS" if c.passed else ("FAIL" if c.critical else "WARN")
            lines.append(f"  [{tag}] {c.name}: {c.detail}")
        lines.append("  => " + ("GREEN LIGHT: calibration accepted"
                                if self.passed else
                                "RED LIGHT: calibration rejected"))
        return "\n".join(lines)


def verify_calibration(
    H: np.ndarray,
    image_size: tuple[int, int],
    keypoints_px: np.ndarray | None = None,
    reproj_rmse_px: float | None = None,
    n_inliers: int | None = None,
    line_support: float | None = None,
    max_rmse_px: float = 6.0,
    min_inliers: int = 4,
    min_line_support: float = 0.6,
    min_court_area_frac: float = 0.04,
    min_measured: int = 4,
) -> VerificationReport:
    """Run all sanity checks on a homography and return a report.

    Parameters
    ----------
    H : pixels -> metres homography
    image_size : (width, height) of the reference image
    keypoints_px : optional 14 x 2 detected keypoints (NaN = missing)
    reproj_rmse_px, n_inliers : from :func:`estimate_homography`
    line_support : fraction of projected model-line samples that landed on
        white pixels (auto detection only)
    """
    rep = VerificationReport()
    W, Hh = image_size

    # 1. numerical sanity ---------------------------------------------------
    finite = H is not None and H.shape == (3, 3) and np.all(np.isfinite(H))
    det = float(np.linalg.det(H)) if finite else float("nan")
    rep.add("matrix_valid", finite and abs(det) > 1e-12,
            f"finite={finite}, det={det:.3e}")
    if not rep.passed:
        return rep
    cond = float(np.linalg.cond(H))
    rep.add("matrix_conditioning", cond < 1e9, f"cond(H)={cond:.2e}", critical=False)

    # 2. reprojection accuracy ---------------------------------------------
    if reproj_rmse_px is not None:
        rep.add("reprojection_rmse", reproj_rmse_px <= max_rmse_px,
                f"{reproj_rmse_px:.2f}px (limit {max_rmse_px:.1f}px)")
    if n_inliers is not None:
        rep.add("inlier_count", n_inliers >= min_inliers,
                f"{n_inliers} inliers (min {min_inliers})")
        rep.add("inlier_count_recommended", n_inliers >= 8,
                f"{n_inliers} inliers (>= 8 recommended for a stable fit)",
                critical=False)

    # 3. geometric plausibility --------------------------------------------
    H_inv = np.linalg.inv(H)
    kp_px = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    rep.add("keypoints_finite", bool(np.all(np.isfinite(kp_px))),
            "all 14 model keypoints project to finite pixels")
    if not rep.passed:
        return rep

    # All court points must lie in front of the camera: same sign of w'.
    hom = np.hstack([G.COURT_KEYPOINTS_M, np.ones((G.NUM_KEYPOINTS, 1))]) @ H_inv.T
    w_sign = np.sign(hom[:, 2])
    rep.add("court_in_front_of_camera", bool(np.all(w_sign == w_sign[0])),
            "projective depth has a consistent sign for the whole court")

    # Orientation: far baseline must appear above the near baseline, and
    # left must be left (no mirroring). The Jacobian determinant of the
    # metres->pixels map must be positive at the court centre.
    far_mid = kp_px[[0, 1]].mean(axis=0)
    near_mid = kp_px[[2, 3]].mean(axis=0)
    rep.add("far_baseline_is_above", far_mid[1] < near_mid[1],
            f"far baseline y={far_mid[1]:.0f}px, near baseline y={near_mid[1]:.0f}px")
    c = np.array([[G.X_CENTRE, G.NET_Y]])
    eps = 0.05
    jx = (apply_homography(H_inv, c + [eps, 0]) - apply_homography(H_inv, c - [eps, 0])) / (2 * eps)
    jy = (apply_homography(H_inv, c + [0, eps]) - apply_homography(H_inv, c - [0, eps])) / (2 * eps)
    jac_det = float(jx[0, 0] * jy[0, 1] - jx[0, 1] * jy[0, 0])
    rep.add("not_mirrored", jac_det > 0,
            f"Jacobian det at court centre = {jac_det:.1f} (must be > 0)")

    # Convexity of the projected doubles rectangle (0,1,3,2 order).
    quad = kp_px[[0, 1, 3, 2]].astype(np.float32)
    convex = bool(cv2.isContourConvex(quad.reshape(-1, 1, 2)))
    area = float(abs(cv2.contourArea(quad)))
    frac = area / float(W * Hh)
    rep.add("court_quad_convex", convex, "projected doubles court is a convex quad")
    rep.add("court_area", frac >= min_court_area_frac,
            f"projected court covers {100 * frac:.1f}% of the frame "
            f"(min {100 * min_court_area_frac:.0f}%)")

    # Court should be at least partially visible.
    inside = ((kp_px[:, 0] >= 0) & (kp_px[:, 0] < W) & (kp_px[:, 1] >= 0) & (kp_px[:, 1] < Hh))
    rep.add("court_visible", int(inside.sum()) >= 4,
            f"{int(inside.sum())}/14 model keypoints fall inside the frame")

    # Perspective sanity: for a camera behind the near baseline the near
    # baseline must appear at least as long as the far one. A clearly longer
    # far baseline means the court model was fitted upside-down / to the
    # wrong lines.
    far_len = np.linalg.norm(kp_px[1] - kp_px[0])
    near_len = np.linalg.norm(kp_px[3] - kp_px[2])
    rep.add("perspective_plausible", near_len >= 0.9 * far_len,
            f"near baseline {near_len:.0f}px vs far baseline {far_len:.0f}px "
            f"(near must be >= 90% of far)")

    # 4. supplied keypoints: count and consistency with H -------------------
    if keypoints_px is not None:
        kp = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, 2)
        ok = np.all(np.isfinite(kp), axis=1)
        rep.add("measured_keypoints", int(ok.sum()) >= min_measured,
                f"{int(ok.sum())}/14 keypoints measured (min {min_measured})")
        if ok.sum() >= 4:
            d = np.linalg.norm(kp[ok] - kp_px[ok], axis=1)
            rep.add("keypoint_consistency", float(np.median(d)) <= max_rmse_px * 1.5,
                    f"median |detected - reprojected| = {np.median(d):.2f}px",
                    critical=False)

    # 5. image evidence (auto mode only) -----------------------------------
    if line_support is not None:
        rep.add("line_support", line_support >= min_line_support,
                f"{100 * line_support:.1f}% of model-line samples lie on white "
                f"pixels (min {100 * min_line_support:.0f}%)")

    return rep
