import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.court.calibration import CourtCalibration, require_calibration
from tennis_tracker.court.homography import CalibrationError, apply_homography
from tests.synthetic import broadcast_homography


def make_cal():
    H, H_inv = broadcast_homography()
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    kp[5] = np.nan
    src = ["measured"] * 14
    src[5] = "estimated"
    return CourtCalibration(H=H, image_size=(1280, 720), keypoints_px=kp, keypoint_source=src,
                            method="auto", reproj_rmse_px=0.7, n_inliers=13, line_support=0.88)


def test_save_load_roundtrip(tmp_path):
    cal = make_cal()
    cal.verify()
    assert cal.is_ready
    p = cal.save(tmp_path / "cal.json")
    cal2 = CourtCalibration.load(p)
    assert np.allclose(cal.H, cal2.H)
    assert np.isnan(cal2.keypoints_px[5]).all()
    assert cal2.keypoint_source[5] == "estimated"
    assert cal2.status == "ok"
    assert len(cal2.checks) > 0


def test_require_calibration_green_light(tmp_path):
    cal = make_cal()
    cal.verify()
    p = cal.save(tmp_path / "cal.json")
    got = require_calibration(p, image_size=(1280, 720))
    assert got.is_ready


def test_require_calibration_rejects_bad_or_mismatched(tmp_path):
    with pytest.raises(CalibrationError):
        require_calibration(tmp_path / "missing.json")
    cal = make_cal()
    cal.reproj_rmse_px = 50.0
    cal.verify()
    assert not cal.is_ready
    p = cal.save(tmp_path / "bad.json")
    with pytest.raises(CalibrationError):
        require_calibration(p)
    good = make_cal()
    good.verify()
    p2 = good.save(tmp_path / "good.json")
    with pytest.raises(CalibrationError):
        require_calibration(p2, image_size=(1920, 1080))


def test_transforms_and_inside_court():
    cal = make_cal()
    corners_px = cal.court_to_pixel(G.COURT_KEYPOINTS_M[:4])
    centre_px = cal.court_to_pixel(np.array([[G.X_CENTRE, G.NET_Y]]))
    assert np.allclose(cal.pixel_to_court(centre_px), [[G.X_CENTRE, G.NET_Y]], atol=1e-6)
    assert cal.is_inside_court(centre_px)[0]
    outside_px = cal.court_to_pixel(np.array([[-1.0, 5.0]]))
    assert not cal.is_inside_court(outside_px)[0]
    assert cal.is_inside_court(corners_px, margin_m=0.01).all()
