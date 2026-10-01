"""Interactive fallback: click the 14 court keypoints by hand.

Used when automatic detection fails (unusual court colour, heavy occlusion)
or when the user wants to correct the automatic result. Also provides a
simple accept/reject review window for automatic results.

Controls (calibration window)
-----------------------------
    left click   place the highlighted keypoint
    s            skip this keypoint (occluded / out of frame)
    u            undo last point
    r            reset all points
    Enter/Space  finish (needs >= 4 points)
    q / Esc      abort
"""

from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from .homography import CalibrationError, apply_homography, estimate_homography
from .visualize import draw_court_overlay, draw_minimap, side_by_side

MAX_DISPLAY_W = 1500
MAX_DISPLAY_H = 850


def _display_scale(w: int, h: int) -> float:
    return min(1.0, MAX_DISPLAY_W / w, MAX_DISPLAY_H / h)


def _put(img, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def click_keypoints(image_bgr: np.ndarray, initial: np.ndarray | None = None,
                    window: str = "TennisTracker calibration") -> tuple[np.ndarray, list[str]]:
    """Open a window and let the user click the 14 keypoints.

    Returns (keypoints_px 14x2 with NaN for skipped, sources list).
    Raises CalibrationError if the user aborts.
    """
    H0, W0 = image_bgr.shape[:2]
    s = _display_scale(W0, H0)
    base = cv2.resize(image_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    minimap_spec = G.MinimapSpec(scale=max(8.0, (base.shape[0] - 40) / (G.COURT_LENGTH + 4)))

    pts = np.full((G.NUM_KEYPOINTS, 2), np.nan)
    src = ["skipped"] * G.NUM_KEYPOINTS
    if initial is not None:
        init = np.asarray(initial, dtype=np.float64).reshape(-1, 2)
        for i in range(min(len(init), G.NUM_KEYPOINTS)):
            if np.all(np.isfinite(init[i])):
                pts[i] = init[i]
                src[i] = "manual"
    state = {"idx": _next_index(src, -1), "mouse": None}

    def on_mouse(event, x, y, flags, _):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and state["idx"] is not None:
            i = state["idx"]
            pts[i] = (x / s, y / s)
            src[i] = "manual"
            state["idx"] = _next_index(src, i)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            frame = _render(base, s, pts, src, state, minimap_spec)
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                raise CalibrationError("Manual calibration aborted by user")
            if key in (13, 32):
                if np.sum(np.array(src) == "manual") >= 4:
                    break
                print("Need at least 4 keypoints before finishing.")
            elif key == ord("s") and state["idx"] is not None:
                i = state["idx"]
                pts[i] = np.nan
                src[i] = "skipped"
                state["idx"] = _next_index(src, i)
            elif key == ord("u"):
                placed = [i for i, v in enumerate(src) if v == "manual"]
                if placed:
                    i = placed[-1]
                    pts[i] = np.nan
                    src[i] = "skipped"
                    state["idx"] = i
            elif key == ord("r"):
                pts[:] = np.nan
                src[:] = ["skipped"] * G.NUM_KEYPOINTS
                state["idx"] = 0
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    return pts, src


def _next_index(src: list[str], after: int) -> int | None:
    order = list(range(after + 1, G.NUM_KEYPOINTS)) + list(range(0, after + 1))
    for i in order:
        if src[i] != "manual":
            return i
    return None


def _render(base, s, pts, src, state, minimap_spec):
    vis = base.copy()
    n_ok = int(np.sum(np.array(src) == "manual"))
    if n_ok >= 4:
        try:
            res = estimate_homography(pts, G.COURT_KEYPOINTS_M, ransac_thresh_px=6.0 / s,
                                      use_ransac=n_ok >= 6)
            H_disp = res.H @ np.diag([1 / s, 1 / s, 1.0])   # display px -> metres
            vis = draw_court_overlay(vis, H_disp, keypoints_px=pts * s,
                                     keypoint_source=["measured" if v == "manual" else "estimated" for v in src],
                                     thickness=1)
            _put(vis, f"live fit: {n_ok} pts, RMSE {res.rmse_px:.1f}px", vis.shape[0] - 14, (0, 255, 0))
        except CalibrationError:
            pass
    for i, p in enumerate(pts):
        if src[i] == "manual":
            q = (int(p[0] * s), int(p[1] * s))
            cv2.circle(vis, q, 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(vis, str(i), (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    idx = state["idx"]
    if idx is not None:
        _put(vis, f"Click keypoint {idx}: {G.KEYPOINT_NAMES[idx]}", 26, (0, 255, 255), 0.7)
    else:
        _put(vis, "All keypoints placed - press Enter to finish", 26, (0, 255, 0), 0.7)
    _put(vis, "s=skip  u=undo  r=reset  Enter=finish  q=abort", 52)
    if state["mouse"] is not None:
        mx, my = state["mouse"]
        cv2.drawMarker(vis, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
    mm = draw_minimap(minimap_spec, highlight=idx)
    return side_by_side(vis, mm, target_h=vis.shape[0])


def review_overlay(image_bgr: np.ndarray, H: np.ndarray, keypoints_px: np.ndarray,
                   keypoint_source: list[str], window: str = "TennisTracker review") -> bool:
    """Show the automatic result; return True if the user accepts (y/Enter)."""
    H0, W0 = image_bgr.shape[:2]
    s = _display_scale(W0, H0)
    vis = draw_court_overlay(image_bgr, H, keypoints_px, keypoint_source)
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    _put(vis, "Automatic calibration - accept?  y/Enter = yes,  n = redo manually,  q = abort", 26, (0, 255, 255), 0.65)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cv2.imshow(window, vis)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("y"), 13, 32):
                return True
            if key == ord("n"):
                return False
            if key in (27, ord("q")):
                raise CalibrationError("Review aborted by user")
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
