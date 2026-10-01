"""Drawing helpers: keypoint overlays, reprojected court lines, top-down minimap."""

from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from .homography import apply_homography

COLOR_MEASURED = (0, 255, 0)
COLOR_ESTIMATED = (0, 200, 255)
COLOR_LINE = (0, 0, 255)
COLOR_NET = (255, 200, 0)


def draw_court_overlay(image_bgr: np.ndarray, H: np.ndarray,
                       keypoints_px: np.ndarray | None = None,
                       keypoint_source: list[str] | None = None,
                       thickness: int = 2, label: bool = True) -> np.ndarray:
    """Draw the court model reprojected through ``H`` (pixels -> metres)."""
    vis = image_bgr.copy()
    H_inv = np.linalg.inv(H)
    kp_model = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    if np.all(np.isfinite(kp_model)):
        for a, b in G.COURT_LINES.values():
            cv2.line(vis, _pt(kp_model[a]), _pt(kp_model[b]), COLOR_LINE, thickness, cv2.LINE_AA)
        net = apply_homography(H_inv, np.array([[G.X_DOUBLES_L, G.NET_Y], [G.X_DOUBLES_R, G.NET_Y]]))
        cv2.line(vis, _pt(net[0]), _pt(net[1]), COLOR_NET, 1, cv2.LINE_AA)
    pts = keypoints_px if keypoints_px is not None else kp_model
    for i, p in enumerate(pts):
        if not np.all(np.isfinite(p)):
            continue
        src = keypoint_source[i] if keypoint_source else "measured"
        col = COLOR_MEASURED if src == "measured" else COLOR_ESTIMATED
        cv2.circle(vis, _pt(p), 6, col, -1, cv2.LINE_AA)
        cv2.circle(vis, _pt(p), 6, (0, 0, 0), 1, cv2.LINE_AA)
        if label:
            cv2.putText(vis, str(i), (_pt(p)[0] + 8, _pt(p)[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(i), (_pt(p)[0] + 8, _pt(p)[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return vis


def draw_minimap(spec: G.MinimapSpec | None = None, highlight: int | None = None,
                 points_m: np.ndarray | None = None) -> np.ndarray:
    """Render the schematic 2-D top-down court."""
    spec = spec or G.MinimapSpec()
    img = np.full((spec.height_px, spec.width_px, 3), (60, 120, 60), np.uint8)
    kp = spec.world_to_minimap(G.COURT_KEYPOINTS_M)
    cv2.rectangle(img, _pt(kp[0]), _pt(kp[3]), (90, 150, 90), -1)
    for a, b in G.COURT_LINES.values():
        cv2.line(img, _pt(kp[a]), _pt(kp[b]), (255, 255, 255), 2, cv2.LINE_AA)
    net = spec.world_to_minimap(np.array([[G.X_DOUBLES_L, G.NET_Y], [G.X_DOUBLES_R, G.NET_Y]]))
    cv2.line(img, _pt(net[0]), _pt(net[1]), (30, 30, 30), 2, cv2.LINE_AA)
    for i, p in enumerate(kp):
        col = (0, 0, 255) if i == highlight else (0, 255, 255)
        r = 7 if i == highlight else 4
        cv2.circle(img, _pt(p), r, col, -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (_pt(p)[0] + 6, _pt(p)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    if points_m is not None:
        for p in spec.world_to_minimap(points_m):
            if np.all(np.isfinite(p)):
                cv2.circle(img, _pt(p), 5, (0, 165, 255), -1, cv2.LINE_AA)
    return img


def warp_top_down(image_bgr: np.ndarray, H: np.ndarray, spec: G.MinimapSpec | None = None) -> np.ndarray:
    """Warp the camera frame into the bird's-eye minimap plane."""
    spec = spec or G.MinimapSpec()
    M = spec.world_to_minimap_matrix() @ H       # pixels -> metres -> minimap px
    warped = cv2.warpPerspective(image_bgr, M, (spec.width_px, spec.height_px))
    kp = spec.world_to_minimap(G.COURT_KEYPOINTS_M)
    for a, b in G.COURT_LINES.values():
        cv2.line(warped, _pt(kp[a]), _pt(kp[b]), (0, 0, 255), 1, cv2.LINE_AA)
    return warped


def side_by_side(left: np.ndarray, right: np.ndarray, target_h: int | None = None) -> np.ndarray:
    target_h = target_h or max(left.shape[0], right.shape[0])
    def rs(im):
        s = target_h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([rs(left), rs(right)])


def draw_debug_lines(image_bgr: np.ndarray, lines_h, lines_v, scale: float = 1.0) -> np.ndarray:
    """Draw candidate lines (work-res coordinates scaled by 1/scale)."""
    vis = image_bgr.copy()
    for L in lines_h:
        cv2.line(vis, _pt(L.p0 / scale), _pt(L.p1 / scale), (255, 0, 0), 2, cv2.LINE_AA)
    for L in lines_v:
        cv2.line(vis, _pt(L.p0 / scale), _pt(L.p1 / scale), (255, 0, 255), 2, cv2.LINE_AA)
    return vis


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))
