import csv
import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import track  # noqa: E402
from tennis_tracker.court import geometry as G  # noqa: E402
from tennis_tracker.court.calibration import CourtCalibration  # noqa: E402
from tennis_tracker.court.homography import apply_homography  # noqa: E402
from tennis_tracker.inference.video import probe_video  # noqa: E402
from tennis_tracker.model import save_weights, tracknet_v2  # noqa: E402
from tests.synthetic import broadcast_homography, render_court  # noqa: E402

W, H, N = 320, 176, 10     # H must be a multiple of 8 only at model resolution, not here


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    Hm, H_inv = broadcast_homography(w=W, h=H)
    v = d / "clip.mp4"
    out = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for i in range(N):
        out.write(render_court(H_inv, w=W, h=H, noise=0, clutter=False))
    out.release()
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    cal = CourtCalibration(H=Hm, image_size=(W, H), keypoints_px=kp, keypoint_source=["measured"] * 14,
                           method="auto", reproj_rmse_px=0.5, n_inliers=14, line_support=0.9)
    cal.verify()
    c = cal.save(d / "cal.json")
    w = save_weights(tracknet_v2(), d / "w.pt")
    return d, v, c, w


def test_cli_end_to_end_with_weights(setup):
    d, v, c, w = setup
    out = d / "tracked.mp4"
    rc = track.main(["--video", str(v), "--calibration", str(c), "--weights", str(w), "--arch", "v2",
                     "--preset", "tracknet_v2", "--device", "cpu", "--batch-size", "4", "--threshold", "0.0",
                     "--output", str(out), "--quiet"])
    assert rc == 0
    assert probe_video(out).frame_count == N
    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert len(rows) == N - 2 and rows[0]["frame"] == "2"
    assert all(r["status"] in ("detected", "predicted", "lost") for r in rows)


def test_cli_uses_default_weights_when_present(setup, monkeypatch):
    d, v, c, w = setup
    monkeypatch.setattr(track, "DEFAULT_WEIGHTS", w)
    out = d / "default_w.mp4"
    rc = track.main(["--video", str(v), "--calibration", str(c), "--arch", "v2", "--device", "cpu",
                     "--max-frames", "5", "--output", str(out), "--quiet"])
    assert rc == 0 and probe_video(out).frame_count == 5


def test_cli_random_weights_and_max_frames(setup):
    d, v, c, _ = setup
    out = d / "rand.mp4"
    rc = track.main(["--video", str(v), "--calibration", str(c), "--allow-random-weights", "--device", "cpu",
                     "--max-frames", "6", "--output", str(out), "--no-minimap", "--quiet"])
    assert rc == 0 and probe_video(out).frame_count == 6


def test_cli_requires_weights_and_gate(setup, monkeypatch):
    d, v, c, w = setup
    monkeypatch.setattr(track, "DEFAULT_WEIGHTS", d / "absent.pt")     # ignore weights/model_best.pt
    assert track.main(["--video", str(v), "--calibration", str(c), "--device", "cpu", "--quiet"]) == 2
    bad = json.loads(c.read_text()); bad["reproj_rmse_px"] = 99.0
    b = d / "bad.json"; b.write_text(json.dumps(bad))
    assert track.main(["--video", str(v), "--calibration", str(b), "--weights", str(w), "--device", "cpu", "--quiet"]) == 1
    assert track.main(["--video", str(v), "--calibration", str(c), "--weights", str(d / "none.pt"), "--device", "cpu", "--quiet"]) == 2
