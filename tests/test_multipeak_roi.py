"""Multi-peak extraction, candidate association and the calibrated ROI mask."""
import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.court.calibration import CourtCalibration
from tennis_tracker.court.homography import apply_homography
from tennis_tracker.inference.detector import BallDetection, multi_peaks_from_heatmaps
from tennis_tracker.tracking.tracker import BallTracker, TrackerConfig
from tests.synthetic import broadcast_homography
from tests.test_ball_detector import gaussian_map
from tests.test_tracking import parabola


def test_multi_peaks_nms_threshold_and_border():
    hm = (gaussian_map(36, 64, 10, 20, amp=0.9) + gaussian_map(36, 64, 50, 5, amp=0.7)
          + gaussian_map(36, 64, 12, 21, amp=0.6)                    # too close to the first peak -> merged
          + gaussian_map(36, 64, 1, 30, sigma=0.7, amp=0.95))        # on the border -> ignored
    peaks = multi_peaks_from_heatmaps(hm[None], threshold=0.5, max_peaks=3, nms_radius=8, border=3)[0]
    assert len(peaks) == 2
    assert peaks[0][2] > peaks[1][2]
    assert abs(peaks[0][0] - 10) < 1.5 and abs(peaks[0][1] - 20) < 1.5
    assert abs(peaks[1][0] - 50) < 0.5 and abs(peaks[1][1] - 5) < 0.5
    assert multi_peaks_from_heatmaps(np.zeros((1, 36, 64), np.float32), threshold=0.5)[0] == []
    assert len(multi_peaks_from_heatmaps(hm[None], threshold=0.5, max_peaks=1, border=3)[0]) == 1


def test_tracker_associates_nearest_candidate_not_strongest():
    traj = parabola(n=40)
    tr = BallTracker(TrackerConfig(smooth=False))
    for i, (x, y) in enumerate(traj):
        cands = [(float(x), float(y), 0.8)]
        if i >= 5:
            cands.insert(0, (120.0, 95.0, 0.9))       # static scoreboard-like false positive, stronger
        tr.step(BallDetection(i, cands[0][0], cands[0][1], cands[0][2], True, candidates=cands))
    pts = tr.finish()
    err = np.array([np.hypot(p.x - traj[k, 0], p.y - traj[k, 1]) for k, p in enumerate(pts)])
    assert all(p.status == "detected" for p in pts)
    assert err.max() < 3.0 and tr.n_segments == 1 and tr.n_rejected == 0


def test_roi_mask_geometry_and_suppression():
    H, H_inv = broadcast_homography(w=1280, h=720)
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    cal = CourtCalibration(H=H, image_size=(1280, 720), keypoints_px=kp, keypoint_source=["measured"] * 14,
                           method="auto", reproj_rmse_px=0.5, n_inliers=14, line_support=0.9)
    mask = cal.plausible_region_mask(side_m=5.0, end_m=8.0)
    assert mask.shape == (720, 1280) and mask.dtype == np.uint8
    def inside(x_m, y_m):
        px = cal.court_to_pixel([[x_m, y_m]])[0]
        xi, yi = int(round(px[0])), int(round(px[1]))
        return bool(0 <= xi < 1280 and 0 <= yi < 720 and mask[yi, xi])

    centre = cal.court_to_pixel([[G.X_CENTRE, G.NET_Y]])[0].astype(int)
    assert mask[centre[1], centre[0]]                       # court centre inside
    assert mask[5, 640]                                      # lob band above the far court inside
    assert inside(G.X_DOUBLES_L - 1.0, G.NET_Y)              # 1 m wide of the sideline: still plausible
    assert inside(G.X_CENTRE, G.Y_BASE_FAR - 5.0)            # 5 m behind the far baseline: plausible
    # without the lob band, the region ends 8 m behind the far baseline
    ground = cal.plausible_region_mask(side_m=5.0, end_m=8.0, lob_band=False)
    top_edge = cal.court_to_pixel([[G.X_CENTRE, G.Y_BASE_FAR - 8.0]])[0]
    assert ground[int(top_edge[1]) + 3, int(top_edge[0])] and not ground[int(top_edge[1]) - 3, int(top_edge[0])]
    assert mask[int(top_edge[1]) - 3, int(top_edge[0])]     # ...but the lob band covers it

    # suppression: a candidate just outside the ground region is ignored, a weaker one inside is used
    ox, oy = float(top_edge[0]), float(int(top_edge[1]) - 3)
    tr = BallTracker(TrackerConfig(roi_mask=ground))
    p = tr.step(BallDetection(0, ox, oy, 0.9, True, candidates=[(ox, oy, 0.9)]))
    assert p.status == "lost" and tr.n_roi_suppressed == 1
    cx, cy = float(centre[0]), float(centre[1])
    p = tr.step(BallDetection(1, ox, oy, 0.9, True, candidates=[(ox, oy, 0.9), (cx, cy, 0.6)]))
    assert p.status == "detected" and p.raw_x == cx and tr.n_roi_suppressed == 2
