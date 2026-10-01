"""Standard tennis-court geometry (ITF dimensions) and the 14-keypoint model.

World coordinate frame
----------------------
* Units: metres.
* Origin: top-left corner of the *doubles* court (far baseline, left sideline
  as seen from the camera).
* +x runs across the court (left -> right), +y runs along the court
  (far baseline -> near baseline). The camera is assumed to sit behind the
  *near* baseline (y = COURT_LENGTH), which is the usual broadcast / behind-
  the-court setup.

Keypoint order (14 points) follows the convention used by
`yastrebksv/TennisCourtDetector`, so any model trained with that dataset
can be dropped in as the "deep learning approach" later:

    0  far baseline  x doubles-left      1  far baseline  x doubles-right
    2  near baseline x doubles-left      3  near baseline x doubles-right
    4  far baseline  x singles-left      5  near baseline x singles-left
    6  far baseline  x singles-right     7  near baseline x singles-right
    8  far service   x singles-left      9  far service   x singles-right
    10 near service  x singles-left      11 near service  x singles-right
    12 far service   x centre line (T)   13 near service  x centre line (T)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ITF court dimensions (metres)
COURT_LENGTH = 23.77          # baseline to baseline
COURT_WIDTH = 10.97           # doubles sideline to doubles sideline
SINGLES_WIDTH = 8.23          # singles sideline to singles sideline
SERVICE_LINE_FROM_NET = 6.40  # service line distance from the net
NET_Y = COURT_LENGTH / 2.0    # 11.885
ALLEY_WIDTH = (COURT_WIDTH - SINGLES_WIDTH) / 2.0   # 1.37

# Named coordinates of every painted line
X_DOUBLES_L = 0.0
X_SINGLES_L = ALLEY_WIDTH                 # 1.37
X_CENTRE = COURT_WIDTH / 2.0              # 5.485
X_SINGLES_R = COURT_WIDTH - ALLEY_WIDTH   # 9.60
X_DOUBLES_R = COURT_WIDTH                 # 10.97

Y_BASE_FAR = 0.0
Y_SERVICE_FAR = NET_Y - SERVICE_LINE_FROM_NET     # 5.485
Y_SERVICE_NEAR = NET_Y + SERVICE_LINE_FROM_NET    # 18.285
Y_BASE_NEAR = COURT_LENGTH                        # 23.77

NUM_KEYPOINTS = 14

# 14 keypoints in world metres, in the order documented above.
COURT_KEYPOINTS_M: np.ndarray = np.array(
    [
        (X_DOUBLES_L, Y_BASE_FAR),      # 0
        (X_DOUBLES_R, Y_BASE_FAR),      # 1
        (X_DOUBLES_L, Y_BASE_NEAR),     # 2
        (X_DOUBLES_R, Y_BASE_NEAR),     # 3
        (X_SINGLES_L, Y_BASE_FAR),      # 4
        (X_SINGLES_L, Y_BASE_NEAR),     # 5
        (X_SINGLES_R, Y_BASE_FAR),      # 6
        (X_SINGLES_R, Y_BASE_NEAR),     # 7
        (X_SINGLES_L, Y_SERVICE_FAR),   # 8
        (X_SINGLES_R, Y_SERVICE_FAR),   # 9
        (X_SINGLES_L, Y_SERVICE_NEAR),  # 10
        (X_SINGLES_R, Y_SERVICE_NEAR),  # 11
        (X_CENTRE, Y_SERVICE_FAR),      # 12
        (X_CENTRE, Y_SERVICE_NEAR),     # 13
    ],
    dtype=np.float64,
)

KEYPOINT_NAMES = [
    "far baseline / doubles left",
    "far baseline / doubles right",
    "near baseline / doubles left",
    "near baseline / doubles right",
    "far baseline / singles left",
    "near baseline / singles left",
    "far baseline / singles right",
    "near baseline / singles right",
    "far service line / singles left",
    "far service line / singles right",
    "near service line / singles left",
    "near service line / singles right",
    "far service line / centre (T)",
    "near service line / centre (T)",
]

# Painted line segments as pairs of keypoint indices.
# These are the 9 physical white lines of the court (net is NOT a painted line).
COURT_LINES: dict[str, tuple[int, int]] = {
    "baseline_far": (0, 1),
    "baseline_near": (2, 3),
    "sideline_doubles_left": (0, 2),
    "sideline_doubles_right": (1, 3),
    "sideline_singles_left": (4, 5),
    "sideline_singles_right": (6, 7),
    "service_far": (8, 9),
    "service_near": (10, 11),
    "centre_service": (12, 13),
}

# Lines grouped by orientation in the world frame, sorted by coordinate.
# Used by the detector to enumerate "which image line is which model line".
HORIZONTAL_MODEL_LINES: list[tuple[str, float, tuple[float, float]]] = [
    # (name, y, (x_start, x_end))
    ("baseline_far", Y_BASE_FAR, (X_DOUBLES_L, X_DOUBLES_R)),
    ("service_far", Y_SERVICE_FAR, (X_SINGLES_L, X_SINGLES_R)),
    ("service_near", Y_SERVICE_NEAR, (X_SINGLES_L, X_SINGLES_R)),
    ("baseline_near", Y_BASE_NEAR, (X_DOUBLES_L, X_DOUBLES_R)),
]
VERTICAL_MODEL_LINES: list[tuple[str, float, tuple[float, float]]] = [
    # (name, x, (y_start, y_end))
    ("sideline_doubles_left", X_DOUBLES_L, (Y_BASE_FAR, Y_BASE_NEAR)),
    ("sideline_singles_left", X_SINGLES_L, (Y_BASE_FAR, Y_BASE_NEAR)),
    ("centre_service", X_CENTRE, (Y_SERVICE_FAR, Y_SERVICE_NEAR)),
    ("sideline_singles_right", X_SINGLES_R, (Y_BASE_FAR, Y_BASE_NEAR)),
    ("sideline_doubles_right", X_DOUBLES_R, (Y_BASE_FAR, Y_BASE_NEAR)),
]

# Which (horizontal line, vertical line) pair defines each keypoint.
KEYPOINT_FROM_LINES: dict[int, tuple[str, str]] = {
    0: ("baseline_far", "sideline_doubles_left"),
    1: ("baseline_far", "sideline_doubles_right"),
    2: ("baseline_near", "sideline_doubles_left"),
    3: ("baseline_near", "sideline_doubles_right"),
    4: ("baseline_far", "sideline_singles_left"),
    5: ("baseline_near", "sideline_singles_left"),
    6: ("baseline_far", "sideline_singles_right"),
    7: ("baseline_near", "sideline_singles_right"),
    8: ("service_far", "sideline_singles_left"),
    9: ("service_far", "sideline_singles_right"),
    10: ("service_near", "sideline_singles_left"),
    11: ("service_near", "sideline_singles_right"),
    12: ("service_far", "centre_service"),
    13: ("service_near", "centre_service"),
}


def model_line_segments_m() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the 9 painted lines as (start_xy, end_xy) in metres."""
    segs = []
    for a, b in COURT_LINES.values():
        segs.append((COURT_KEYPOINTS_M[a].copy(), COURT_KEYPOINTS_M[b].copy()))
    return segs


def sample_model_lines_m(step_m: float = 0.10) -> np.ndarray:
    """Densely sample points along all painted lines (N x 2, metres).

    Used to score a homography hypothesis against the white-pixel mask.
    """
    pts = []
    for p0, p1 in model_line_segments_m():
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step_m)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(p0[None, :] * (1 - t) + p1[None, :] * t)
    return np.vstack(pts)


def model_distance_map(res_m: float = 0.05, margin_m: float = 3.0) -> tuple[np.ndarray, float, float]:
    """Distance (metres) from every point of the court plane to the nearest
    painted line, rasterised at ``res_m`` per cell with ``margin_m`` around the
    court. Returns (map, origin_m, res_m); cell (r, c) covers world point
    (c * res_m - margin_m, r * res_m - margin_m).

    Used to measure how much white-pixel evidence a homography hypothesis
    explains (recall), which makes the hypothesis search robust against
    degenerate solutions that squash the whole model onto one bright line.
    """
    import cv2

    w = int(round((COURT_WIDTH + 2 * margin_m) / res_m)) + 1
    h = int(round((COURT_LENGTH + 2 * margin_m) / res_m)) + 1
    img = np.full((h, w), 255, np.uint8)
    for p0, p1 in model_line_segments_m():
        a = np.round((p0 + margin_m) / res_m).astype(int)
        b = np.round((p1 + margin_m) / res_m).astype(int)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 0, 1)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5).astype(np.float64) * res_m
    return dist, -margin_m, res_m


@dataclass(frozen=True)
class MinimapSpec:
    """Pixel layout of the 2D top-down (bird's-eye) court rendering."""

    scale: float = 20.0      # pixels per metre
    margin_m: float = 2.0    # blank border around the court

    @property
    def width_px(self) -> int:
        return int(round((COURT_WIDTH + 2 * self.margin_m) * self.scale))

    @property
    def height_px(self) -> int:
        return int(round((COURT_LENGTH + 2 * self.margin_m) * self.scale))

    def world_to_minimap(self, pts_m: np.ndarray) -> np.ndarray:
        """Metres -> minimap pixels (N x 2)."""
        pts_m = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        return (pts_m + self.margin_m) * self.scale

    def minimap_to_world(self, pts_px: np.ndarray) -> np.ndarray:
        pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        return pts_px / self.scale - self.margin_m

    def world_to_minimap_matrix(self) -> np.ndarray:
        """3x3 similarity: metres -> minimap pixels."""
        s, m = self.scale, self.margin_m
        return np.array([[s, 0, s * m], [0, s, s * m], [0, 0, 1]], dtype=np.float64)
