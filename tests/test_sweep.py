import numpy as np

from sweep_threshold import choose, evaluate
from tennis_tracker.tracking.tracker import TrackerConfig
from tests.test_tracking import parabola


def make_rows():
    traj = parabola(n=50)
    rows = []
    for i, (x, y) in enumerate(traj):
        conf = 0.9
        if 20 <= i < 24:
            conf = 0.05                                   # occluded
        rows.append((i, conf, float(x), float(y)))
    rows[30] = (30, 0.4, rows[30][2] + 600.0, rows[30][3] - 300.0)   # low-confidence false positive
    return rows


def test_evaluate_counts_and_score():
    rows = make_rows()
    r_low = evaluate(rows, 0.3, TrackerConfig())      # FP visible -> gate rejects it
    r_high = evaluate(rows, 0.5, TrackerConfig())     # FP filtered by threshold
    assert r_low["visible"] == 46 and r_high["visible"] == 45
    assert r_low["rejected"] == 1 and r_high["rejected"] == 0
    assert r_high["predicted"] == 5 and r_high["lost"] == 0 and r_high["segments"] == 1
    assert r_high["score"] > r_low["score"]
    assert 5 < r_high["jump_px"] < 40


def test_choose_prefers_middle_of_best_plateau():
    res = [{"threshold": t, "score": s} for t, s in [(0.3, 10), (0.4, 12), (0.5, 12), (0.6, 12), (0.7, 5)]]
    assert choose(res)["threshold"] == 0.5


def test_tracker_counts_rejections():
    from tennis_tracker.inference.detector import BallDetection
    from tennis_tracker.tracking.tracker import BallTracker
    tr = BallTracker(TrackerConfig())
    for i, (x, y) in enumerate(parabola(n=20)):
        if i == 10:
            x += 900
        tr.step(BallDetection(i, float(x), float(y), 0.9, True))
    tr.finish()
    assert tr.n_rejected == 1 and tr.n_segments == 1
