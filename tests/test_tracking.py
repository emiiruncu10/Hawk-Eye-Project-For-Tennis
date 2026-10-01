import numpy as np
import pytest

from tennis_tracker.inference.detector import BallDetection
from tennis_tracker.tracking.kalman import ConstantAccelerationKalman
from tennis_tracker.tracking.tracker import BallTracker, TrackerConfig, track_summary


def parabola(n=60, x0=100.0, y0=600.0, vx=18.0, vy=-25.0, g=1.2):
    t = np.arange(n, dtype=float)
    return np.stack([x0 + vx * t, y0 + vy * t + 0.5 * g * t * t], axis=1)


def dets_from(traj, visible=None, noise=0.0, seed=0, conf=0.9):
    rng = np.random.default_rng(seed)
    out = []
    for i, (x, y) in enumerate(traj):
        v = True if visible is None else visible[i]
        if v:
            out.append(BallDetection(i, float(x + rng.normal(0, noise)), float(y + rng.normal(0, noise)), conf, True))
        else:
            out.append(BallDetection(i, None, None, 0.1, False))
    return out


def positions(points):
    return np.array([[p.x, p.y] if p.has_position else [np.nan, np.nan] for p in points])


# ----------------------------------------------------------------- Kalman
def test_kalman_tracks_parabola_and_smoothing_reduces_noise():
    traj = parabola()
    rng = np.random.default_rng(1)
    meas = traj + rng.normal(0, 3.0, traj.shape)
    kf = ConstantAccelerationKalman(process_noise=1.0, measurement_noise=3.0)
    kf.init(*meas[0])
    filt = [kf.position]
    for z in meas[1:]:
        kf.predict()
        kf.update(z)
        filt.append(kf.position)
    filt = np.array(filt)
    err_meas = np.linalg.norm(meas[10:] - traj[10:], axis=1).mean()
    err_filt = np.linalg.norm(filt[10:] - traj[10:], axis=1).mean()
    smooth = kf.rts_smooth()[:, :2]
    err_smooth = np.linalg.norm(smooth[10:] - traj[10:], axis=1).mean()
    assert err_filt < err_meas
    assert err_smooth < err_filt
    assert err_smooth < 2.0


def test_kalman_prediction_follows_acceleration():
    traj = parabola(n=40)
    kf = ConstantAccelerationKalman(process_noise=0.5, measurement_noise=1.0)
    kf.init(*traj[0])
    for z in traj[1:30]:
        kf.predict(); kf.update(z)
    preds = [kf.predict() for _ in range(5)]          # coast 5 frames
    err = np.linalg.norm(np.array(preds) - traj[30:35], axis=1)
    assert err.max() < 6.0, err


def test_mahalanobis_grows_with_distance():
    kf = ConstantAccelerationKalman()
    kf.init(0, 0)
    kf.predict()
    assert kf.mahalanobis2([1, 1]) < kf.mahalanobis2([50, 50])


# ---------------------------------------------------------------- tracker
def test_tracker_all_detected():
    traj = parabola()
    tr = BallTracker(TrackerConfig(smooth=True))
    for d in dets_from(traj, noise=1.0):
        tr.step(d)
    pts = tr.finish()
    s = track_summary(pts)
    assert s == {"frames": 60, "detected": 60, "predicted": 0, "lost": 0}
    assert tr.n_segments == 1
    assert np.linalg.norm(positions(pts) - traj, axis=1).mean() < 1.5
    assert [p.frame_index for p in pts] == list(range(60))


def test_tracker_interpolates_short_gap():
    traj = parabola()
    vis = np.ones(60, bool)
    vis[25:30] = False                                  # 5 missing frames
    tr = BallTracker(TrackerConfig(max_missing=8, smooth=True))
    for d in dets_from(traj, vis, noise=0.5):
        tr.step(d)
    pts = tr.finish()
    assert [p.status for p in pts[25:30]] == ["predicted"] * 5
    gap_err = np.linalg.norm(positions(pts)[25:30] - traj[25:30], axis=1)
    assert gap_err.max() < 5.0, gap_err
    assert pts[30].status == "detected"


def test_tracker_marks_long_gap_lost_and_restarts():
    traj = parabola()
    vis = np.ones(60, bool)
    vis[20:40] = False                                  # 20 missing frames > max_missing
    tr = BallTracker(TrackerConfig(max_missing=5, smooth=False))
    for d in dets_from(traj, vis):
        tr.step(d)
    pts = tr.finish()
    statuses = [p.status for p in pts]
    assert statuses[:20] == ["detected"] * 20
    assert statuses[20:40] == ["lost"] * 20             # predicted tail after last detection -> lost
    assert statuses[40:] == ["detected"] * 20
    assert tr.n_segments == 2 and pts[40].segment == 1
    assert all(not p.has_position for p in pts[20:40])


def test_tracker_rejects_single_outlier():
    traj = parabola()
    dets = dets_from(traj, noise=0.5)
    dets[30] = BallDetection(30, traj[30, 0] + 400, traj[30, 1] - 300, 0.95, True)   # false positive
    tr = BallTracker(TrackerConfig(smooth=False))
    for d in dets:
        tr.step(d)
    pts = tr.finish()
    assert pts[30].status == "predicted"
    assert np.linalg.norm(positions(pts)[30] - traj[30]) < 8.0
    assert tr.n_segments == 1


def test_tracker_reinitialises_after_persistent_jump():
    traj = parabola()
    jumped = traj.copy()
    jumped[30:] += [500, -200]                          # ball really is somewhere else now
    tr = BallTracker(TrackerConfig(reinit_after_rejections=3, smooth=False))
    for d in dets_from(jumped):
        tr.step(d)
    pts = tr.finish()
    assert tr.n_segments == 2
    assert pts[32].status == "detected" and pts[32].segment == 1
    assert np.linalg.norm(positions(pts)[40] - jumped[40]) < 5.0


def test_tracker_starts_lost_and_uses_min_confidence():
    tr = BallTracker(TrackerConfig(min_confidence=0.8))
    p0 = tr.step(BallDetection(0, None, None, 0.0, False))
    p1 = tr.step(BallDetection(1, 10.0, 10.0, 0.6, True))    # visible but below min_confidence
    p2 = tr.step(BallDetection(2, 12.0, 11.0, 0.9, True))
    assert p0.status == "lost" and p1.status == "lost" and p2.status == "detected"
    pts = tr.finish()
    assert [p.frame_index for p in pts] == [0, 1, 2]
