"""Whole-chain check without trained weights.

A synthetic clip contains a yellow ball flying along a known parabola. An
"oracle" model locates the yellow blob *in the 9-channel input tensor* and
emits a Gaussian heat-map there. If preprocessing (resize, RGB order, frame
order), peak extraction, scaling back to video pixels, tracking and CSV
export are all consistent, the recovered trajectory matches ground truth.
"""

import csv

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.court import geometry as G  # noqa: E402
from tennis_tracker.court.calibration import CourtCalibration  # noqa: E402
from tennis_tracker.court.homography import apply_homography  # noqa: E402
from tennis_tracker.inference.detector import BallDetector, DetectorConfig  # noqa: E402
from tennis_tracker.inference.pipeline import PipelineConfig, TrackNetInputPipeline  # noqa: E402
from tennis_tracker.inference.preprocess import TrackNetInputSpec  # noqa: E402
from tennis_tracker.tracking.annotate import export_csv, render_annotated_video  # noqa: E402
from tennis_tracker.tracking.tracker import BallTracker, TrackerConfig, track_summary  # noqa: E402
from tests.synthetic import broadcast_homography, render_court  # noqa: E402

W, H, N = 640, 360, 40


class OracleModel(torch.nn.Module):
    """Finds the yellow ball in the *newest* frame of the stack (channels 6:9 for
    oldest-first RGB input) and returns 3 heat-maps with a Gaussian at that spot."""

    def heatmaps(self, x):
        n, _, h, w = x.shape
        rgb = x[:, 6:9]
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        mask = ((r > 0.8) & (g > 0.8) & (b < 0.3)).float()          # (n, h, w)
        ys = torch.arange(h, device=x.device, dtype=torch.float32).view(1, h, 1)
        xs = torch.arange(w, device=x.device, dtype=torch.float32).view(1, 1, w)
        tot = mask.sum(dim=(1, 2)).clamp(min=1e-6)
        cx = (mask * xs).sum(dim=(1, 2)) / tot
        cy = (mask * ys).sum(dim=(1, 2)) / tot
        found = (mask.sum(dim=(1, 2)) > 0).float()
        d2 = (xs - cx.view(n, 1, 1)) ** 2 + (ys - cy.view(n, 1, 1)) ** 2
        hm = torch.exp(-d2 / (2 * 2.0 ** 2)) * found.view(n, 1, 1)
        return torch.stack([torch.zeros_like(hm), torch.zeros_like(hm), hm], dim=1)


def ground_truth():
    t = np.arange(N, dtype=float)
    x = 60 + 12.5 * t
    y = 300 - 16 * t + 0.42 * t * t          # stays inside the frame (border peaks are suppressed)
    return np.stack([x, y], axis=1)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("oracle")
    Hm, H_inv = broadcast_homography(w=W, h=H)
    gt = ground_truth()
    v = d / "ball.mp4"
    out = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for i in range(N):
        img = render_court(H_inv, w=W, h=H, noise=0, clutter=False)
        if 18 <= i < 22:                      # occlusion: ball hidden for 4 frames
            pass
        else:
            cv2.circle(img, (int(round(gt[i, 0])), int(round(gt[i, 1]))), 6, (0, 255, 255), -1)
        out.write(img)
    out.release()
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    cal = CourtCalibration(H=Hm, image_size=(W, H), keypoints_px=kp, keypoint_source=["measured"] * 14,
                           method="auto", reproj_rmse_px=0.5, n_inliers=14, line_support=0.9)
    cal.verify()
    return v, cal.save(d / "cal.json"), d, gt


def test_oracle_chain_matches_ground_truth(clip):
    v, c, d, gt = clip
    spec = TrackNetInputSpec(height=144, width=256)               # 2.5x downscale
    pipe = TrackNetInputPipeline(v, c, PipelineConfig(spec=spec, batch_size=6, device="cpu"))
    det = BallDetector(OracleModel(), "cpu", DetectorConfig(threshold=0.5, output_frame_index=2))
    tracker = BallTracker(TrackerConfig(max_missing=8))
    raw = []
    for batch in pipe:
        for dd in det.detect(batch):
            raw.append(dd)
            tracker.step(dd)
    pts = tracker.finish()
    assert [p.frame_index for p in pts] == list(range(2, N))
    summ = track_summary(pts)
    assert summ["predicted"] == 4 and summ["lost"] == 0 and summ["detected"] == N - 2 - 4
    assert [r.visible for r in raw][16:20] == [False] * 4          # frames 18..21 occluded

    pos = np.array([[p.x, p.y] for p in pts])
    err = np.linalg.norm(pos - gt[2:], axis=1)
    detected = np.array([p.status == "detected" for p in pts])
    assert err[detected].mean() < 1.5, err[detected].mean()       # sub-2px after 2.5x downscale
    assert err[~detected].max() < 8.0, err[~detected]             # interpolated gap

    out = d / "ball_tracked.mp4"
    assert render_annotated_video(v, pts, out, pipe.calibration) == N
    export_csv(pts, d / "ball.csv", pipe.calibration)
    rows = list(csv.DictReader((d / "ball.csv").open()))
    assert len(rows) == N - 2
    r = rows[10]
    m = pipe.calibration.pixel_to_court([[float(r["x_px"]), float(r["y_px"])]])[0]
    assert float(r["court_x_m"]) == pytest.approx(m[0], abs=1e-3)
