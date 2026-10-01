import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.court import geometry as G  # noqa: E402
from tennis_tracker.court.calibration import CourtCalibration  # noqa: E402
from tennis_tracker.court.homography import CalibrationError, apply_homography  # noqa: E402
from tennis_tracker.inference.pipeline import PipelineConfig, TrackNetInputPipeline, iter_tracknet_batches  # noqa: E402
from tennis_tracker.inference.preprocess import TrackNetInputSpec  # noqa: E402
from tennis_tracker.inference.video import VideoError, VideoReader, probe_video  # noqa: E402
from tests.synthetic import broadcast_homography, render_court  # noqa: E402

W, H, N_FRAMES = 320, 180, 12


@pytest.fixture(scope="module")
def video_and_cal(tmp_path_factory):
    d = tmp_path_factory.mktemp("phase2")
    Hm, H_inv = broadcast_homography(w=W, h=H)
    vpath = d / "clip.mp4"
    out = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert out.isOpened(), "mp4v VideoWriter unavailable"
    for i in range(N_FRAMES):
        img = render_court(H_inv, w=W, h=H, line_px=2, noise=0, clutter=False)
        cv2.circle(img, (20 + 20 * i, 90), 4, (0, 255, 255), -1)  # moving "ball"
        out.write(img)
    out.release()
    kp = apply_homography(H_inv, G.COURT_KEYPOINTS_M)
    cal = CourtCalibration(H=Hm, image_size=(W, H), keypoints_px=kp, keypoint_source=["measured"] * 14,
                           method="auto", reproj_rmse_px=0.5, n_inliers=14, line_support=0.9)
    assert cal.verify().passed
    cpath = cal.save(d / "calibration.json")
    return vpath, cpath, d


def test_probe_and_reader(video_and_cal):
    vpath, _, _ = video_and_cal
    info = probe_video(vpath)
    assert info.size == (W, H) and info.frame_count == N_FRAMES and info.fps == pytest.approx(30.0)
    with VideoReader(vpath) as vr:
        frames = list(vr)
    assert len(frames) == N_FRAMES
    assert frames[0][0] == 0 and frames[-1][0] == N_FRAMES - 1
    assert frames[5][1].shape == (H, W, 3) and frames[5][2] == pytest.approx(5 / 30)
    with VideoReader(vpath, start=4, end=9, step=2) as vr:
        idx = [i for i, _, _ in vr]
    assert idx == [4, 6, 8]
    with pytest.raises(VideoError):
        probe_video(vpath.parent / "missing.mp4")


def test_gate_blocks_before_video_is_opened(video_and_cal, monkeypatch):
    vpath, cpath, d = video_and_cal
    opened = []
    real = cv2.VideoCapture

    def spy(*a, **k):
        opened.append(a)
        return real(*a, **k)
    monkeypatch.setattr(cv2, "VideoCapture", spy)

    with pytest.raises(CalibrationError):
        TrackNetInputPipeline(vpath, d / "missing.json")
    bad = json.loads(cpath.read_text())
    bad["reproj_rmse_px"] = 99.0
    bad_path = d / "bad.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(CalibrationError):
        TrackNetInputPipeline(vpath, bad_path)
    assert opened == []          # the video was never touched


def test_resolution_mismatch_and_rescale(video_and_cal):
    vpath, cpath, d = video_and_cal
    cal = CourtCalibration.load(cpath)
    small = cal.rescaled((160, 90))
    assert small.is_ready and np.allclose(small.keypoints_px, cal.keypoints_px / 2)
    spath = small.save(d / "small.json")
    with pytest.raises(CalibrationError):
        TrackNetInputPipeline(vpath, spath)
    pipe = TrackNetInputPipeline(vpath, spath, PipelineConfig(allow_rescale=True, device="cpu"))
    assert pipe.calibration.image_size == (W, H)
    assert np.allclose(pipe.calibration.H, cal.H, rtol=1e-6)
    with pytest.raises(CalibrationError):
        cal.rescaled((300, 300))


def test_pipeline_end_to_end_cpu(video_and_cal):
    vpath, cpath, _ = video_and_cal
    spec = TrackNetInputSpec(height=72, width=128)
    cfg = PipelineConfig(spec=spec, batch_size=4, device="cpu", keep_original_frames=True)
    pipe = TrackNetInputPipeline(vpath, cpath, cfg)
    assert pipe.expected_sequences == N_FRAMES - 2
    batches = list(pipe)
    n = sum(len(b) for b in batches)
    assert n == N_FRAMES - 2 and len(batches) == 3
    assert [len(b) for b in batches] == [4, 4, 2]
    b0 = batches[0]
    assert tuple(b0.tensor.shape) == (4, 9, 72, 128) and b0.tensor.dtype == torch.float32
    assert b0.frame_indices.tolist() == [2, 3, 4, 5] and b0.sequence_indices[0] == (0, 1, 2)
    assert float(b0.tensor.min()) >= 0 and float(b0.tensor.max()) <= 1
    assert b0.original_frames[0].shape == (H, W, 3) and b0.original_size == (W, H)
    # channels 6:9 (frame t) of row 0 must equal channels 3:6 (frame t-1) of row 1
    assert torch.allclose(b0.tensor[0, 6:9], b0.tensor[1, 3:6])
    assert pipe.stats.frames_read == N_FRAMES and pipe.stats.sequences == N_FRAMES - 2


def test_pipeline_options(video_and_cal):
    vpath, cpath, _ = video_and_cal
    spec = TrackNetInputSpec(height=36, width=64)
    seqs = [b for b in iter_tracknet_batches(vpath, cpath, spec=spec, batch_size=100, device="cpu", warmup="repeat")]
    assert sum(len(b) for b in seqs) == N_FRAMES
    assert seqs[0].sequence_indices[0] == (0, 0, 0)
    seqs = list(iter_tracknet_batches(vpath, cpath, spec=spec, batch_size=2, device="cpu",
                                      start_frame=3, end_frame=9, max_sequences=3))
    assert sum(len(b) for b in seqs) == 3
    assert seqs[0].frame_indices.tolist() == [5, 6]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_pipeline_on_mps(video_and_cal):
    vpath, cpath, _ = video_and_cal
    spec = TrackNetInputSpec(height=36, width=64)
    b = next(iter(iter_tracknet_batches(vpath, cpath, spec=spec, batch_size=4, device="mps")))
    assert b.tensor.device.type == "mps" and b.tensor.dtype == torch.float32
    assert tuple(b.tensor.shape) == (4, 9, 36, 64)
    assert float(b.tensor.max()) <= 1.0
