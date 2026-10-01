import csv

import cv2
import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.court.calibration import CourtCalibration
from tennis_tracker.court.homography import apply_homography
from tennis_tracker.inference.video import probe_video
from tennis_tracker.tracking.annotate import (
    AnnotatorConfig,
    TrajectoryAnnotator,
    VideoWriterMP4,
    export_csv,
    render_annotated_video,
)
from tennis_tracker.tracking.tracker import TrackPoint
from tests.synthetic import broadcast_homography, render_court

W, H = 320, 180


@pytest.fixture
def cal():
    Hm, H_inv = broadcast_homography(w=W, h=H)
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    c = CourtCalibration(H=Hm, image_size=(W, H), keypoints_px=kp, keypoint_source=["measured"] * 14,
                         method="auto", reproj_rmse_px=0.5, n_inliers=14, line_support=0.9)
    assert c.verify().passed
    return c


def make_points(n=15):
    pts = []
    for i in range(n):
        st = "detected" if i % 5 else "predicted"
        pts.append(TrackPoint(i, 40.0 + 12 * i, 120.0 - 3 * i, st, 0.8))
    pts.append(TrackPoint(n, None, None, "lost", 0.1))
    return pts


def test_annotator_draws_ball_tail_and_minimap(cal):
    frame = np.zeros((H, W, 3), np.uint8)
    ann = TrajectoryAnnotator(cal, AnnotatorConfig(tail_length=5, minimap_scale=3.0, draw_court_lines=True))
    pts = make_points()
    out = ann.draw(frame, pts[:10])
    assert out.shape == frame.shape and out is not frame
    cx, cy = int(pts[9].x), int(pts[9].y)
    assert tuple(out[cy, cx]) == (0, 255, 0)                     # current ball in green
    assert out.sum() > 0
    mm_w = ann._minimap_base.shape[1]
    assert out[20, W - mm_w // 2].sum() > 0                       # minimap pasted top-right
    out_lost = ann.draw(frame, pts)                               # last point lost: no circle, no crash
    assert out_lost.shape == frame.shape
    m = ann.to_court_m(*cal.court_to_pixel([[G.X_CENTRE, G.NET_Y]])[0])
    assert np.allclose(m, [G.X_CENTRE, G.NET_Y], atol=1e-6)


def test_annotator_without_calibration():
    ann = TrajectoryAnnotator(None, AnnotatorConfig(draw_minimap=True))
    out = ann.draw(np.zeros((H, W, 3), np.uint8), make_points()[:3])
    assert out.shape == (H, W, 3)


def test_video_writer_and_render(tmp_path, cal):
    src = tmp_path / "src.mp4"
    _, H_inv = broadcast_homography(w=W, h=H)
    with VideoWriterMP4(src, 30.0, (W, H)) as w:
        for _ in range(16):
            w.write(render_court(H_inv, w=W, h=H, noise=0, clutter=False))
        assert w.codec in ("avc1", "mp4v") and w.frames_written == 16
    assert probe_video(src).frame_count == 16
    pts = make_points()
    out = tmp_path / "out.mp4"
    seen = []
    n = render_annotated_video(src, pts, out, cal, AnnotatorConfig(tail_length=6), progress=seen.append)
    assert n == 16 and seen == list(range(16))
    info = probe_video(out)
    assert info.size == (W, H) and info.frame_count == 16
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9)
    ok, f = cap.read()
    cap.release()
    assert ok and f.shape == (H, W, 3)


def test_export_csv(tmp_path, cal):
    pts = make_points(6)
    p = export_csv(pts, tmp_path / "t.csv", cal)
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 7
    assert rows[0]["status"] == "predicted" and rows[1]["status"] == "detected"
    assert rows[-1]["status"] == "lost" and rows[-1]["x_px"] == "" and rows[-1]["court_x_m"] == ""
    assert float(rows[1]["x_px"]) == pytest.approx(52.0)
    assert rows[1]["court_x_m"] != ""
