"""Calibration result container, JSON persistence and the Phase-2 gate.

A :class:`CourtCalibration` is the *only* thing later phases need: it holds
the pixel->metre homography, the 14 keypoints and the verification report.
:func:`require_calibration` is the "green light" check that video processing
must call before touching an MP4.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import geometry as G
from .homography import (
    CalibrationError,
    VerificationReport,
    apply_homography,
    verify_calibration,
)

SCHEMA_VERSION = 1


@dataclass
class CourtCalibration:
    H: np.ndarray                          # 3x3 pixels -> metres
    image_size: tuple[int, int]            # (width, height)
    keypoints_px: np.ndarray               # 14 x 2, NaN where not measured
    keypoint_source: list[str]             # "measured" | "estimated" | "manual" | "skipped"
    method: str                            # "auto" | "manual"
    reproj_rmse_px: float
    n_inliers: int
    line_support: float | None = None
    source: str = ""                       # image / video path
    frame_index: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: str = "unverified"             # "ok" | "failed" | "unverified"
    checks: list[dict] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    # ----------------------------------------------------------------- props
    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    @property
    def is_ready(self) -> bool:
        """True only if verification passed => green light for video processing."""
        return self.status == "ok"

    @property
    def keypoints_all_px(self) -> np.ndarray:
        kp = self.keypoints_px.copy()
        miss = ~np.all(np.isfinite(kp), axis=1)
        if miss.any():
            kp[miss] = apply_homography(self.H_inv, G.COURT_KEYPOINTS_M[miss])
        return kp

    # ------------------------------------------------------------- transforms
    def pixel_to_court(self, pts_px) -> np.ndarray:
        """Image pixels -> court metres (N x 2)."""
        return apply_homography(self.H, pts_px)

    def court_to_pixel(self, pts_m) -> np.ndarray:
        return apply_homography(self.H_inv, pts_m)

    def is_inside_court(self, pts_px, singles: bool = False, margin_m: float = 0.0) -> np.ndarray:
        """Boolean per point: inside the (doubles or singles) court rectangle."""
        m = self.pixel_to_court(pts_px)
        x0, x1 = (G.X_SINGLES_L, G.X_SINGLES_R) if singles else (G.X_DOUBLES_L, G.X_DOUBLES_R)
        return ((m[:, 0] >= x0 - margin_m) & (m[:, 0] <= x1 + margin_m)
                & (m[:, 1] >= G.Y_BASE_FAR - margin_m) & (m[:, 1] <= G.Y_BASE_NEAR + margin_m))

    def plausible_region_mask(self, side_m: float = 5.0, end_m: float = 8.0,
                              lob_band: bool = True) -> np.ndarray:
        """Binary (H, W) mask of image locations where a ball can physically be:
        the court extended by ``side_m`` beside the sidelines and ``end_m``
        behind the baselines (ground plane) plus, when ``lob_band`` is set, the
        band straight above the extended far baseline up to the top of the
        frame (balls in flight project above the far court). Broadcast
        graphics in the frame corners fall outside this region."""
        import cv2

        W, Hh = self.image_size
        corners_m = np.array([
            [G.X_DOUBLES_L - side_m, G.Y_BASE_FAR - end_m], [G.X_DOUBLES_R + side_m, G.Y_BASE_FAR - end_m],
            [G.X_DOUBLES_R + side_m, G.Y_BASE_NEAR + end_m], [G.X_DOUBLES_L - side_m, G.Y_BASE_NEAR + end_m]])
        quad = self.court_to_pixel(corners_m)
        mask = np.zeros((Hh, W), np.uint8)
        if not np.all(np.isfinite(quad)):
            mask[:] = 255
            return mask
        lim_lo, lim_hi = [-4 * W, -4 * Hh], [5 * W, 5 * Hh]
        cv2.fillPoly(mask, [np.clip(quad, lim_lo, lim_hi).astype(np.int32).reshape(-1, 1, 2)], 255)
        if lob_band:
            (xl, yl), (xr, yr) = quad[0], quad[1]
            band = np.array([[xl, yl], [xr, yr], [xr, 0], [xl, 0]])
            cv2.fillPoly(mask, [np.clip(band, lim_lo, lim_hi).astype(np.int32).reshape(-1, 1, 2)], 255)
        return mask

    def rescaled(self, new_size: tuple[int, int], aspect_tol: float = 0.01) -> "CourtCalibration":
        """Return a copy adapted to a video of ``new_size`` (W, H).

        Only a pure resize of the same camera view is supported, so the aspect
        ratio must match; otherwise :class:`CalibrationError` is raised.
        """
        ow, oh = self.image_size
        nw, nh = int(new_size[0]), int(new_size[1])
        if abs(nw / nh - ow / oh) > aspect_tol:
            raise CalibrationError(
                f"Cannot rescale calibration from {(ow, oh)} to {(nw, nh)}: aspect ratio differs")
        sx, sy = nw / ow, nh / oh
        S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])        # new px -> old px
        kp = self.keypoints_px * np.array([sx, sy])
        cal = CourtCalibration(
            H=self.H @ S_inv, image_size=(nw, nh), keypoints_px=kp,
            keypoint_source=list(self.keypoint_source), method=self.method,
            reproj_rmse_px=self.reproj_rmse_px * max(sx, sy), n_inliers=self.n_inliers,
            line_support=self.line_support, source=self.source, frame_index=self.frame_index,
            created_at=self.created_at, status="unverified",
        )
        rep = cal.verify()
        if not rep.passed:
            raise CalibrationError("Rescaled calibration failed verification:\n" + rep.format())
        return cal

    # ----------------------------------------------------------- verification
    def verify(self, **kwargs) -> VerificationReport:
        if "min_measured" not in kwargs:
            kwargs["min_measured"] = 6 if self.method == "auto" else 4
        rep = verify_calibration(
            self.H, self.image_size, keypoints_px=self.keypoints_px,
            reproj_rmse_px=self.reproj_rmse_px, n_inliers=self.n_inliers,
            line_support=self.line_support, **kwargs,
        )
        self.status = "ok" if rep.passed else "failed"
        self.checks = [asdict(c) for c in rep.checks]
        return rep

    # -------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        kp = [[None if not np.isfinite(v) else float(v) for v in p] for p in self.keypoints_px]
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "method": self.method,
            "source": self.source,
            "frame_index": self.frame_index,
            "created_at": self.created_at,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "court_dimensions_m": {"length": G.COURT_LENGTH, "width": G.COURT_WIDTH,
                                   "singles_width": G.SINGLES_WIDTH},
            "homography_px_to_m": self.H.tolist(),
            "homography_m_to_px": self.H_inv.tolist(),
            "keypoints_px": kp,
            "keypoint_source": list(self.keypoint_source),
            "keypoints_world_m": G.COURT_KEYPOINTS_M.tolist(),
            "keypoint_names": G.KEYPOINT_NAMES,
            "reproj_rmse_px": float(self.reproj_rmse_px),
            "n_inliers": int(self.n_inliers),
            "line_support": None if self.line_support is None else float(self.line_support),
            "checks": self.checks,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "CourtCalibration":
        kp = np.array([[np.nan if v is None else v for v in p] for p in d["keypoints_px"]], dtype=np.float64)
        return cls(
            H=np.array(d["homography_px_to_m"], dtype=np.float64),
            image_size=tuple(int(v) for v in d["image_size"]),
            keypoints_px=kp,
            keypoint_source=list(d.get("keypoint_source", ["measured"] * len(kp))),
            method=d.get("method", "unknown"),
            reproj_rmse_px=float(d.get("reproj_rmse_px", float("nan"))),
            n_inliers=int(d.get("n_inliers", 0)),
            line_support=d.get("line_support"),
            source=d.get("source", ""),
            frame_index=d.get("frame_index"),
            created_at=d.get("created_at", ""),
            status=d.get("status", "unverified"),
            checks=list(d.get("checks", [])),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        path = Path(path)
        if not path.exists():
            raise CalibrationError(f"Calibration file not found: {path}")
        return cls.from_dict(json.loads(path.read_text()))


def require_calibration(path: str | Path, image_size: tuple[int, int] | None = None,
                        reverify: bool = True) -> CourtCalibration:
    """Load a calibration and raise :class:`CalibrationError` unless it is green.

    Phase 2 (video processing) must call this first. ``image_size`` lets the
    caller assert that the video has the same resolution as the reference
    image the homography was computed on.
    """
    cal = CourtCalibration.load(path)
    if reverify:
        rep = cal.verify()
        if not rep.passed:
            raise CalibrationError(
                "Calibration failed re-verification:\n" + rep.format())
    if not cal.is_ready:
        raise CalibrationError(f"Calibration status is '{cal.status}', not 'ok' ({path})")
    if image_size is not None and tuple(image_size) != tuple(cal.image_size):
        raise CalibrationError(
            f"Video resolution {tuple(image_size)} does not match calibration "
            f"resolution {tuple(cal.image_size)}; re-run calibrate.py on this video")
    return cal
