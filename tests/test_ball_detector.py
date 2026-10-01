import numpy as np
import pytest

from tennis_tracker.inference.detector import BallDetection, DetectorConfig, peaks_from_heatmaps
from tennis_tracker.inference.preprocess import TrackNetInputSpec


def gaussian_map(h, w, cx, cy, sigma=2.0, amp=1.0):
    ys, xs = np.mgrid[0:h, 0:w]
    return (amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def test_argmax_finds_integer_peaks():
    hm = np.stack([gaussian_map(36, 64, 10, 20), gaussian_map(36, 64, 50, 5)])
    x, y, conf, vis = peaks_from_heatmaps(hm, threshold=0.5, subpixel=False)
    assert x.tolist() == [10, 50] and y.tolist() == [20, 5]
    assert np.allclose(conf, 1.0) and vis.all()


def test_subpixel_refinement_recovers_fractional_centre():
    hm = gaussian_map(36, 64, 30.4, 12.7, sigma=2.5)[None]
    x, y, conf, vis = peaks_from_heatmaps(hm, threshold=0.5, subpixel=True)
    assert abs(x[0] - 30.4) < 0.02 and abs(y[0] - 12.7) < 0.02          # log-parabolic: exact
    xc, yc, _, _ = peaks_from_heatmaps(hm, threshold=0.5, subpixel=True, method="centroid", window=3)
    assert abs(xc[0] - 30.4) < 0.25 and abs(yc[0] - 12.7) < 0.25
    x2, y2, _, _ = peaks_from_heatmaps(hm, threshold=0.5, subpixel=False)
    assert (x2[0], y2[0]) == (30, 13)
    flat = np.full((1, 36, 64), 0.7, np.float32)                          # saturated plateau: no shift
    xf, yf, _, _ = peaks_from_heatmaps(flat, threshold=0.5, subpixel=True)
    assert xf[0] == 0 and yf[0] == 0


def test_threshold_marks_occluded():
    hm = np.stack([gaussian_map(36, 64, 10, 20, amp=0.9), gaussian_map(36, 64, 10, 20, amp=0.3),
                   np.zeros((36, 64), np.float32)])
    x, y, conf, vis = peaks_from_heatmaps(hm, threshold=0.5)
    assert vis.tolist() == [True, False, False]
    assert np.isnan(x[1]) and np.isnan(y[2])
    assert conf[1] == pytest.approx(0.3, abs=1e-6) and conf[2] == 0.0


def test_peak_at_border_does_not_crash():
    hm = gaussian_map(36, 64, 0, 35, sigma=1.0)[None]
    x, y, conf, vis = peaks_from_heatmaps(hm, threshold=0.5)
    assert vis[0] and 0 <= x[0] < 2 and 33 < y[0] <= 35


def test_detection_scaling_through_batch():
    torch = pytest.importorskip("torch")
    from tennis_tracker.inference.detector import BallDetector
    from tennis_tracker.inference.tensorize import InputBatch

    class FakeModel(torch.nn.Module):
        def heatmaps(self, x):
            n, _, h, w = x.shape
            hm = torch.zeros(n, 3, h, w)
            hm[0, 2, 9, 16] = 1.0          # frame-t channel: peak at x=16, y=9
            hm[0, 0, 0, 0] = 1.0           # a peak in another channel must be ignored
            hm[1, 2, 3, 3] = 0.2           # below threshold
            return hm

    spec = TrackNetInputSpec(height=36, width=64)
    batch = InputBatch(tensor=torch.zeros(2, 9, 36, 64), frame_indices=np.array([7, 8]),
                       sequence_indices=[(5, 6, 7), (6, 7, 8)], timestamps_s=np.array([0.0, 0.0]),
                       spec=spec, original_size=(1920, 1080))
    det = BallDetector(FakeModel(), "cpu", DetectorConfig(threshold=0.5, subpixel=False, output_frame_index=2))
    out = det.detect(batch)
    assert isinstance(out[0], BallDetection)
    assert out[0].frame_index == 7 and out[0].visible
    assert out[0].x == pytest.approx(16 * 30) and out[0].y == pytest.approx(9 * 30)   # 1920/64, 1080/36
    assert out[0].x_model == 16 and out[0].y_model == 9
    assert out[1].frame_index == 8 and not out[1].visible and out[1].x is None
    assert out[1].status == "occluded" and out[1].confidence == pytest.approx(0.2)


def test_detector_runs_real_model_on_devices():
    torch = pytest.importorskip("torch")
    from tennis_tracker.inference.detector import BallDetector
    from tennis_tracker.inference.tensorize import InputBatch
    from tennis_tracker.model import tracknet_v2

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    spec = TrackNetInputSpec(height=32, width=64)
    for dev in devices:
        batch = InputBatch(tensor=torch.rand(2, 9, 32, 64, device=dev), frame_indices=np.array([2, 3]),
                           sequence_indices=[(0, 1, 2), (1, 2, 3)], timestamps_s=np.zeros(2),
                           spec=spec, original_size=(640, 320))
        det = BallDetector(tracknet_v2(), dev, DetectorConfig(threshold=0.0, output_frame_index=2))
        out = det.detect(batch)
        assert len(out) == 2 and all(d.visible for d in out)
        assert 0 <= out[0].x <= 640 and 0 <= out[0].y <= 320
