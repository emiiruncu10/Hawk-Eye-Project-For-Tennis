import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.court.homography import apply_homography
from tennis_tracker.court.line_detector import DetectorConfig, detect_court
from tests.synthetic import broadcast_homography, render_court


@pytest.mark.parametrize("params", [
    dict(),                                              # standard broadcast
    dict(far_y=0.22, near_y=0.90, far_half=0.22, near_half=0.55),  # strong perspective
    dict(far_y=0.35, near_y=0.75, far_half=0.35, near_half=0.40),  # high camera
])
def test_detects_synthetic_court(params):
    H, H_inv = broadcast_homography(**params)
    img = render_court(H_inv, seed=3)
    res = detect_court(img, DetectorConfig(use_court_mask=True))
    gt = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    pred = apply_homography(res.homography.H_inv, G.COURT_KEYPOINTS_M)
    err = np.linalg.norm(pred - gt, axis=1)
    assert err.max() < 4.0, err
    assert sum(s == "measured" for s in res.keypoint_source) >= 10
    assert res.line_support > 0.7


def test_detector_at_full_hd_scales_back_correctly():
    H, H_inv = broadcast_homography(w=1920, h=1080)
    img = render_court(H_inv, w=1920, h=1080, line_px=6, seed=5)
    res = detect_court(img)
    assert res.work_scale < 1.0
    gt = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    pred = apply_homography(res.homography.H_inv, G.COURT_KEYPOINTS_M)
    assert np.linalg.norm(pred - gt, axis=1).max() < 5.0
    meas = res.keypoints_px[np.all(np.isfinite(res.keypoints_px), axis=1)]
    assert meas[:, 0].max() > 1280       # coordinates are in original pixels


def test_detector_without_court_mask():
    H, H_inv = broadcast_homography()
    img = render_court(H_inv, seed=7, clutter=False)
    res = detect_court(img, DetectorConfig(use_court_mask=False))
    gt = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    pred = apply_homography(res.homography.H_inv, G.COURT_KEYPOINTS_M)
    assert np.linalg.norm(pred - gt, axis=1).max() < 4.0


def test_detector_rejects_blank_image():
    from tennis_tracker.court.homography import CalibrationError
    img = np.full((720, 1280, 3), (70, 130, 60), np.uint8)
    with pytest.raises(CalibrationError):
        detect_court(img)
