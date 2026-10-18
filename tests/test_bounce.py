import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.tracking.bounce import (
    BounceConfig,
    TimeBase,
    detect_bounces,
    find_duplicate_frames,
    measurements,
    read_track_csv,
)
from tennis_tracker.tracking.tracker import TrackPoint
from tests.sim3d import Camera, simulate, track_points

FPS = 50.0


def run(p0, v0, t_end, events=(), fps=FPS, noise=0.7, seed=0):
    cam = Camera.behind_baseline()
    pos, true = simulate(p0, v0, t_end, events)
    n = int(t_end * fps)
    pts = track_points(cam, pos, np.arange(n) / fps, np.arange(n), noise, seed)
    return detect_bounces(pts, cam.calibration(), fps), true, cam


def test_far_court_bounce_position_and_time():
    found, true, _ = run((4.0, 24.5, 1.0), (0.5, -24.0, 3.5), 1.4)
    assert len(true) == 1 and true[0][2] < G.NET_Y
    assert len(found) == 1
    b = found[0]
    t_true, x_true, y_true = true[0]
    assert b.half == "far"
    assert abs(b.time_s - t_true) < 1.0 / FPS
    assert abs(b.court_x_m - x_true) < 0.06            # lateral: a few cm
    assert abs(b.court_y_m - y_true) < 0.40            # depth: foreshortened far court ...
    assert abs(b.court_y_m - y_true) < 3 * b.sigma_y_m + 0.02   # ... and honestly reported
    assert b.jump_vy_mps < -3.5 and abs(b.jump_vx_mps) < 0.5 * abs(b.jump_vy_mps)


def test_near_court_bounce_position_and_time():
    found, true, _ = run((6.0, -1.0, 1.0), (-0.3, 24.0, 3.5), 1.3)
    assert len(true) == 1 and true[0][2] > G.NET_Y
    assert len(found) == 1
    b = found[0]
    assert b.half == "near"
    assert abs(b.time_s - true[0][0]) < 1.0 / FPS
    assert abs(b.court_x_m - true[0][1]) < 0.05
    assert abs(b.court_y_m - true[0][2]) < 0.15
    # reported uncertainty is honest: true error within 4 sigma
    assert abs(b.court_y_m - true[0][2]) < 4 * b.sigma_y_m + 0.02


@pytest.mark.parametrize("p0,v0,hit", [
    ((5.0, 14.0, 1.8), (0.0, 18.0, -1.0), (0.35, (0.5, -22.0, 4.0))),     # solid return volley
    ((4.0, 13.0, 1.5), (0.3, 20.0, 0.5), (0.40, (-0.4, -18.0, 5.0))),     # rising return
])
def test_racket_hit_is_not_a_bounce(p0, v0, hit):
    # ball approaching the near player, hit back before it lands (no ground contact at all)
    for seed in range(3):
        found, true, _ = run(p0, v0, hit[0] + 0.6, events=[hit], seed=seed)
        assert true == []
        assert found == [], (seed, found)


def test_hit_right_after_a_bounce_is_removed_by_rally_grammar():
    # far player's shot bounces in the near half, the near player hits it 0.45 s later
    pos, true = simulate((6.0, -1.0, 1.0), (-0.3, 24.0, 3.5), 1.8, events=[(1.37, (0.3, -16.0, 3.0))])
    cam = Camera.behind_baseline()
    n = int(1.8 * FPS)
    pts = track_points(cam, pos, np.arange(n) / FPS, np.arange(n), 0.7, 2)
    found = detect_bounces(pts, cam.calibration(), FPS)
    near = [b for b in found if b.half == "near"]
    assert len(near) == 1 and abs(near[0].time_s - true[0][0]) < 1.0 / FPS


def test_rally_with_two_bounces_and_a_hit():
    # near player -> far bounce -> far player hits -> near bounce
    p0, v0 = (4.0, 24.5, 1.0), (0.5, -24.0, 3.5)
    pos, true = simulate(p0, v0, 2.4, events=[(1.15, (-0.2, 24.0, 3.0))])
    assert len(true) == 2
    cam = Camera.behind_baseline()
    n = int(2.4 * FPS)
    pts = track_points(cam, pos, np.arange(n) / FPS, np.arange(n), 0.7, 3)
    found = detect_bounces(pts, cam.calibration(), FPS)
    assert [b.half for b in found] == ["far", "near"]
    for b, (t, x, y) in zip(found, true):
        assert abs(b.time_s - t) < 1.0 / FPS
        assert abs(b.court_x_m - x) < 0.06


def test_flight_without_bounce_gives_nothing():
    found, true, _ = run((5.0, 24.0, 1.2), (0.0, -20.0, 6.0), 0.9)
    assert true == [] and found == []


def test_duplicate_frames_25_in_30_fps():
    """25 fps material in a 30 fps file: one frame in six repeats its predecessor."""
    cam = Camera.behind_baseline()
    pos, true = simulate((4.0, 24.5, 1.0), (0.5, -24.0, 3.5), 1.4)
    n = int(1.4 * 30)
    src = np.floor(np.arange(n) * 25 / 30).astype(int)              # source frame shown by each video frame
    dups = [k for k in range(1, n) if src[k] == src[k - 1]]
    assert len(dups) == n // 6
    pts = track_points(cam, pos, src / 25.0, np.arange(n), 0.7, 5)   # fresh noise on every video frame
    found = detect_bounces(pts, cam.calibration(), 30.0, duplicate_frames=dups, n_frames=n)
    assert len(found) == 1
    assert abs(found[0].time_s - true[0][0]) < 1.0 / 25
    assert abs(found[0].court_x_m - true[0][1]) < 0.08


def test_timebase():
    tb = TimeBase(10, 30.0)
    assert np.allclose(tb.time([0, 3, 9]), [0, 0.1, 0.3])
    assert tb.frame_at(0.1) == pytest.approx(3.0)
    tb = TimeBase(12, 30.0, duplicates=[5, 11])
    assert tb.fps_eff == pytest.approx(25.0)
    assert tb.time([4])[0] == tb.time([5])[0]                         # a repeat shares its time
    assert tb.time([6])[0] == pytest.approx(5 / 25.0)
    assert tb.frame_at(tb.time([6])[0]) == pytest.approx(6.0)


def test_measurements_drop_duplicates_and_non_detected():
    pts = [TrackPoint(0, 10, 10, "detected", .9, 10, 10), TrackPoint(1, 10.2, 10.1, "detected", .9, 10.2, 10.1),
           TrackPoint(2, 30, 30, "predicted", .1), TrackPoint(3, 50, 50, "detected", .9, 50, 50),
           TrackPoint(4, 60, 60, "detected", .9, 60, 60)]
    m = measurements(pts, duplicate_tol_px=0.5)
    assert m[:, 0].tolist() == [0, 3, 4]                               # frame 1 repeats frame 0
    assert measurements(pts, 0.5, duplicate_frames=[3])[:, 0].tolist() == [0, 4]


def test_find_duplicate_frames_on_video(tmp_path):
    import cv2
    path = tmp_path / "dup.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    frames = []
    for k in range(10):
        img = np.full((180, 320, 3), (60, 120, 60), np.uint8)
        cv2.circle(img, (20 + 25 * k, 90), 10, (0, 255, 255), -1)
        frames.append(img)
    seq = [frames[0], frames[1], frames[2], frames[2], frames[3], frames[4], frames[5], frames[5], frames[6]]
    for img in seq:
        w.write(img)
    w.release()
    assert find_duplicate_frames(path) == [3, 7]


def test_read_track_csv_roundtrip(tmp_path):
    from tennis_tracker.tracking.annotate import export_csv
    pts = [TrackPoint(2, 10.5, 20.25, "detected", 0.9, 10.0, 20.0, 0), TrackPoint(3, None, None, "lost", 0.1),
           TrackPoint(4, 12.0, 22.0, "predicted", 0.3, None, None, 0)]
    p = export_csv(pts, tmp_path / "t.csv")
    back = read_track_csv(p)
    assert [b.frame_index for b in back] == [2, 3, 4]
    assert back[0].raw_x == 10.0 and back[0].x == 10.5 and back[1].x is None and back[2].raw_x is None
    assert [b.status for b in back] == ["detected", "lost", "predicted"]


def test_config_rules_individually():
    from tennis_tracker.tracking.bounce import Bounce, is_bounce
    cfg = BounceConfig()
    base = dict(frame=10, time_s=0.2, x_px=0, y_px=0, contact_x_px=0, contact_y_px=0, court_x_m=5, court_y_m=5,
                half="far", jump_vy_mps=-8, jump_vx_mps=0.5, tstat=10, vy_before_mps=2.5, vy_after_mps=-5.5, px_per_m=50)
    assert is_bounce(Bounce(**base), cfg)
    assert not is_bounce(Bounce(**{**base, "jump_vy_mps": -2}), cfg)            # too weak
    assert not is_bounce(Bounce(**{**base, "tstat": 2}), cfg)                   # not significant
    assert not is_bounce(Bounce(**{**base, "jump_vx_mps": 6}), cfg)             # lateral change = hit
    assert not is_bounce(Bounce(**{**base, "court_y_m": -9}), cfg)              # high in the air
    assert not is_bounce(Bounce(**{**base, "vy_before_mps": -2}), cfg)          # far: rising before = hit
    assert not is_bounce(Bounce(**{**base, "vy_before_mps": 12}), cfg)          # far: serve contact
    near = {**base, "half": "near", "court_y_m": 20, "vy_before_mps": 10, "vy_after_mps": -2}
    assert is_bounce(Bounce(**near), cfg)
    assert not is_bounce(Bounce(**{**near, "vy_before_mps": 1}), cfg)           # near: not falling
    assert not is_bounce(Bounce(**{**near, "vy_after_mps": -14}), cfg)          # near: gained energy = hit
    assert not is_bounce(Bounce(**{**near, "rebound_ratio": 1.0}), cfg)         # ... measured next to the knot
    assert is_bounce(Bounce(**{**near, "vy_after_mps": -14, "rebound_ratio": 0.5}), cfg)  # local value wins
    assert not is_bounce(Bounce(**{**near, "tstat": 3}), cfg)                   # near: weak corner = noise
    assert not is_bounce(Bounce(**{**base, "court_y_m": -5}), cfg)              # far: 5 m behind = in the air
