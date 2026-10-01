"""Regression tests for the checks added after the clay-court false green light."""
import numpy as np

from tennis_tracker.court import geometry as G
from tennis_tracker.court.calibration import CourtCalibration
from tennis_tracker.court.homography import apply_homography, verify_calibration
from tennis_tracker.court.line_detector import Line, prune_by_mutual_extent
from tests.synthetic import broadcast_homography


def test_upside_down_perspective_is_critical():
    # far baseline wider than near baseline: camera cannot be behind the near baseline
    H, _ = broadcast_homography(far_half=0.46, near_half=0.30)
    rep = verify_calibration(H, (1280, 720), reproj_rmse_px=0.5, n_inliers=14, line_support=0.9)
    assert not rep.passed
    bad = [c for c in rep.checks if c.name == "perspective_plausible"][0]
    assert not bad.passed and bad.critical


def test_too_few_measured_keypoints_fails_auto_but_not_manual():
    H, H_inv = broadcast_homography()
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    kp[4:] = np.nan                                     # only the 4 corners measured
    auto = CourtCalibration(H=H, image_size=(1280, 720), keypoints_px=kp, keypoint_source=["measured"] * 4 + ["estimated"] * 10,
                            method="auto", reproj_rmse_px=0.0, n_inliers=4, line_support=0.9)
    assert not auto.verify().passed
    assert any(c["name"] == "measured_keypoints" and not c["passed"] for c in auto.checks)
    manual = CourtCalibration(H=H, image_size=(1280, 720), keypoints_px=kp, keypoint_source=["manual"] * 4 + ["skipped"] * 10,
                              method="manual", reproj_rmse_px=0.0, n_inliers=4, line_support=None)
    assert manual.verify().passed


def test_low_line_support_fails():
    H, _ = broadcast_homography()
    assert not verify_calibration(H, (1280, 720), reproj_rmse_px=0.5, n_inliers=14, line_support=0.5).passed
    assert verify_calibration(H, (1280, 720), reproj_rmse_px=0.5, n_inliers=14, line_support=0.65).passed


def hline(y, x0, x1, support):
    return Line(0.0, 1.0, -float(y), np.array([x0, y], float), np.array([x1, y], float), support)


def vline(x, y0, y1, support):
    return Line(1.0, 0.0, -float(x), np.array([x, y0], float), np.array([x, y1], float), support)


def test_prune_by_mutual_extent_drops_crowd_lines():
    verticals = [vline(300, 200, 600, 1500), vline(900, 200, 600, 1400), vline(400, 200, 600, 1000), vline(800, 200, 600, 900)]
    horizontals = [hline(50, 100, 1200, 2000),      # banner edge above the court
                   hline(600, 250, 950, 1300),      # near baseline
                   hline(700, 0, 200, 800),         # scoreboard
                   hline(200, 300, 900, 300),       # far baseline (weak)
                   hline(450, 350, 850, 900)]       # service line
    kh, kv = prune_by_mutual_extent(horizontals, verticals, (1280, 720), top_n=4, margin_frac=0.06)
    assert [int(L.y_at_x(600)) for L in kh] == [600, 200, 450]
    assert len(kv) == 4
    # pruning never leaves fewer than two lines per orientation
    kh2, kv2 = prune_by_mutual_extent([hline(50, 0, 1280, 500), hline(60, 0, 1280, 400)], verticals, (1280, 720))
    assert len(kh2) == 2
