import cv2
import numpy as np
import pytest

from tennis_tracker.inference.preprocess import (
    TrackNetInputSpec,
    model_to_original_coords,
    original_to_model_coords,
    preprocess_frame,
    stack_sequence,
)


def solid_frame(bgr, h=90, w=160):
    return np.tile(np.array(bgr, np.uint8), (h, w, 1))


def test_default_spec_is_tracknet_v2():
    s = TrackNetInputSpec()
    assert (s.height, s.width) == (288, 512)
    assert s.n_channels == 9 and s.input_shape == (9, 288, 512)
    assert TrackNetInputSpec.from_name("yastrebksv") == TrackNetInputSpec.tracknet_yastrebksv()
    with pytest.raises(ValueError):
        TrackNetInputSpec.from_name("nope")
    with pytest.raises(ValueError):
        TrackNetInputSpec(channel_order="bgra")


def test_preprocess_shape_dtype_range():
    spec = TrackNetInputSpec()
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), np.uint8)
    out = preprocess_frame(frame, spec)
    assert out.shape == (3, 288, 512)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.flags["C_CONTIGUOUS"]


def test_preprocess_channel_order():
    frame = solid_frame((255, 0, 0))                     # pure blue in BGR
    rgb = preprocess_frame(frame, TrackNetInputSpec(channel_order="rgb", height=45, width=80))
    bgr = preprocess_frame(frame, TrackNetInputSpec(channel_order="bgr", height=45, width=80))
    assert np.allclose(rgb[:, 0, 0], [0, 0, 1])          # blue lands in channel 2 for RGB
    assert np.allclose(bgr[:, 0, 0], [1, 0, 0])          # channel 0 for BGR


def test_preprocess_matches_manual_resize():
    spec = TrackNetInputSpec(height=72, width=128, channel_order="bgr")
    frame = np.random.default_rng(1).integers(0, 256, (144, 256, 3), np.uint8)
    ref = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    assert np.allclose(preprocess_frame(frame, spec), np.transpose(ref, (2, 0, 1)))


def test_stack_sequence_orders_and_shape():
    spec = TrackNetInputSpec(height=45, width=80)
    frames = [np.full((3, 45, 80), v, np.float32) for v in (0.1, 0.5, 0.9)]   # t-2, t-1, t
    old = stack_sequence(frames, TrackNetInputSpec(height=45, width=80, frame_order="oldest_first"))
    new = stack_sequence(frames, TrackNetInputSpec(height=45, width=80, frame_order="newest_first"))
    assert old.shape == (9, 45, 80) and old.dtype == np.float32
    assert np.allclose(old[0:3], 0.1) and np.allclose(old[6:9], 0.9)
    assert np.allclose(new[0:3], 0.9) and np.allclose(new[6:9], 0.1)
    with pytest.raises(ValueError):
        stack_sequence(frames[:2], spec)
    with pytest.raises(ValueError):
        stack_sequence([np.zeros((3, 10, 10), np.float32)] * 3, spec)


def test_coordinate_scaling_roundtrip():
    spec = TrackNetInputSpec()
    orig = (1920, 1080)
    pts = np.array([[256.0, 144.0], [0.0, 0.0]])
    back = model_to_original_coords(pts, spec, orig)
    assert np.allclose(back[0], [960, 540])
    assert np.allclose(original_to_model_coords(back, spec, orig), pts)
