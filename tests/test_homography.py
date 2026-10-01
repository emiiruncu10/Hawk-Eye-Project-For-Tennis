import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.court.homography import (
    CalibrationError,
    apply_homography,
    estimate_homography,
    verify_calibration,
)
from tests.synthetic import broadcast_homography


def test_estimate_recovers_ground_truth():
    H, H_inv = broadcast_homography()
    px = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    res = estimate_homography(px, G.COURT_KEYPOINTS_M)
    assert res.rmse_px < 1e-3
    assert res.n_inliers == 14
    back = apply_homography(res.H, px)
    assert np.allclose(back, G.COURT_KEYPOINTS_M, atol=1e-4)


def test_estimate_with_missing_and_outlier_points():
    H, H_inv = broadcast_homography()
    px = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    px[[1, 6]] = np.nan                 # missing
    px[12] += (60, -40)                 # gross outlier
    px += np.random.default_rng(1).normal(0, 0.5, px.shape)
    res = estimate_homography(px, G.COURT_KEYPOINTS_M, ransac_thresh_px=4.0)
    assert res.n_points == 12
    assert not res.inlier_mask[12]
    assert res.rmse_px < 2.0


def test_too_few_points_raises():
    H, H_inv = broadcast_homography()
    px = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    px[3:] = np.nan
    with pytest.raises(CalibrationError):
        estimate_homography(px, G.COURT_KEYPOINTS_M)


def test_verify_passes_for_good_homography():
    H, H_inv = broadcast_homography()
    rep = verify_calibration(H, (1280, 720), reproj_rmse_px=0.8, n_inliers=14, line_support=0.9)
    assert rep.passed, rep.format()


def test_verify_fails_when_mirrored_or_upside_down():
    H, H_inv = broadcast_homography()
    flip = np.diag([-1.0, 1.0, 1.0]) @ np.array([[1, 0, -G.COURT_WIDTH], [0, 1, 0], [0, 0, 1]], float)
    H_mirror = flip @ H     # x -> width - x
    rep = verify_calibration(H_mirror, (1280, 720), reproj_rmse_px=0.5, n_inliers=14)
    assert not rep.passed
    assert any(c.name == "not_mirrored" and not c.passed for c in rep.checks)

    rot = np.array([[-1, 0, G.COURT_WIDTH], [0, -1, G.COURT_LENGTH], [0, 0, 1]], float)
    rep2 = verify_calibration(rot @ H, (1280, 720), reproj_rmse_px=0.5, n_inliers=14)
    assert not rep2.passed
    assert any(c.name == "far_baseline_is_above" and not c.passed for c in rep2.checks)


def test_verify_fails_on_large_error_or_low_support():
    H, H_inv = broadcast_homography()
    assert not verify_calibration(H, (1280, 720), reproj_rmse_px=25.0, n_inliers=14).passed
    assert not verify_calibration(H, (1280, 720), reproj_rmse_px=1.0, n_inliers=14, line_support=0.1).passed
    assert not verify_calibration(H, (1280, 720), reproj_rmse_px=1.0, n_inliers=3).passed


def test_verify_fails_for_garbage_matrix():
    assert not verify_calibration(np.full((3, 3), np.nan), (1280, 720)).passed
    assert not verify_calibration(np.zeros((3, 3)), (1280, 720)).passed
