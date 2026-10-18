import csv
import json

import cv2
import numpy as np
import pytest

import line_calls
from tennis_tracker.inference.video import probe_video
from tennis_tracker.tracking.annotate import export_csv
from tests.sim3d import Camera, simulate, track_points
from tests.synthetic import render_court

FPS, N = 50.0, 70


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("calls")
    cam = Camera.behind_baseline(width=640, height=360, focal=550.0)
    pos, true = simulate((6.0, -1.0, 1.0), (-0.3, 24.0, 3.5), N / FPS)
    pts = track_points(cam, pos, np.arange(N) / FPS, np.arange(N), 0.5, 1)
    v = d / "clip.mp4"
    out = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    for p in pts:
        img = render_court(cam.H_inv, w=640, h=360, line_px=2, noise=0, clutter=False)
        cv2.circle(img, (int(p.raw_x), int(p.raw_y)), 4, (0, 255, 255), -1)
        out.write(img)
    out.release()
    track = export_csv(pts, d / "clip_tracked.csv")
    cal = cam.calibration().save(d / "cal.json")
    return d, v, track, cal, true


def test_cli_end_to_end(clip):
    d, v, track, cal, true = clip
    out = d / "calls.mp4"
    rc = line_calls.main(["--video", str(v), "--calibration", str(cal), "--track-csv", str(track),
                          "--output", str(out), "--quiet"])
    assert rc == 0
    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert len(rows) == 1 and rows[0]["call"] == "IN" and rows[0]["half"] == "near"
    assert abs(float(rows[0]["court_y_m"]) - true[0][2]) < 0.3
    assert probe_video(out).frame_count == N


def test_cli_doubles_csv_only(clip):
    d, v, track, cal, _ = clip
    ev = d / "only.csv"
    rc = line_calls.main(["--video", str(v), "--calibration", str(cal), "--track-csv", str(track),
                          "--events-csv", str(ev), "--output", str(d / "none.mp4"), "--no-video",
                          "--game", "doubles", "--quiet"])
    assert rc == 0 and ev.exists() and not (d / "none.mp4").exists()
    assert list(csv.DictReader(ev.open()))[0]["game"] == "doubles"


def test_cli_gate_and_missing_inputs(clip):
    d, v, track, cal, _ = clip
    bad = json.loads(cal.read_text()); bad["reproj_rmse_px"] = 99.0
    b = d / "bad.json"; b.write_text(json.dumps(bad))
    assert line_calls.main(["--video", str(v), "--calibration", str(b), "--track-csv", str(track), "--quiet"]) == 1
    assert line_calls.main(["--video", str(v), "--calibration", str(cal), "--track-csv", str(d / "x.csv"), "--quiet"]) == 2
    assert line_calls.main(["--video", str(d / "missing.mp4"), "--calibration", str(cal), "--quiet"]) == 2
