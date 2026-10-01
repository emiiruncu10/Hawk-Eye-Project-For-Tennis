"""Synthetic court renderer used by the detector tests."""
import cv2
import numpy as np

from tennis_tracker.court import geometry as G
from tennis_tracker.court.homography import apply_homography


def broadcast_homography(w=1280, h=720, far_y=0.28, near_y=0.82, far_half=0.30, near_half=0.46):
    """Metres -> pixels homography for a typical behind-the-baseline camera."""
    dst = np.array([
        [w * (0.5 - far_half), h * far_y],    # kp0 far/left
        [w * (0.5 + far_half), h * far_y],    # kp1 far/right
        [w * (0.5 - near_half), h * near_y],  # kp2 near/left
        [w * (0.5 + near_half), h * near_y],  # kp3 near/right
    ], np.float32)
    src = G.COURT_KEYPOINTS_M[:4].astype(np.float32)
    H_inv = cv2.getPerspectiveTransform(src, dst)
    return np.linalg.inv(H_inv), H_inv


def render_court(H_inv, w=1280, h=720, line_px=4, noise=8.0, seed=0, clutter=True):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (70, 130, 60), np.uint8)   # green surface
    img[: int(h * 0.2)] = (40, 40, 40)                  # dark stands
    if clutter:
        for _ in range(6):   # bright banners / text at the top
            x0, y0 = rng.integers(0, w - 200), rng.integers(0, int(h * 0.18))
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + 200), int(y0 + 30)), (230, 230, 230), -1)
        for _ in range(3):   # random white streaks on the court
            p = rng.uniform([0, h * 0.3], [w, h], (2,)).astype(int)
            q = p + rng.integers(-150, 150, 2)
            cv2.line(img, tuple(p), tuple(q), (250, 250, 250), 2)
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    for a, b in G.COURT_LINES.values():
        cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (245, 245, 245), line_px, cv2.LINE_AA)
    # net
    net = apply_homography(H_inv, np.array([[0, G.NET_Y], [G.COURT_WIDTH, G.NET_Y]]))
    cv2.line(img, tuple(net[0].astype(int)), tuple(net[1].astype(int)), (20, 20, 20), 3)
    if noise > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
