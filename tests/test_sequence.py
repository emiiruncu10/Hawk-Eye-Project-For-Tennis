import numpy as np
import pytest

from tennis_tracker.inference.sequence import FrameSequenceBuffer


def frame(v):
    return np.full((3, 4, 6), v, np.float32)


def test_skip_warmup_emits_from_third_frame():
    buf = FrameSequenceBuffer(length=3, warmup="skip")
    assert buf.push(0, frame(0)) is None
    assert buf.push(1, frame(1)) is None
    assert not buf.is_ready or buf.n_emitted == 0
    seq = buf.push(2, frame(2), timestamp_s=2 / 30)
    assert seq is not None
    assert seq.frame_indices == (0, 1, 2) and seq.current_index == 2
    assert [float(f[0, 0, 0]) for f in seq.frames] == [0.0, 1.0, 2.0]
    assert seq.timestamp_s == pytest.approx(2 / 30)
    seq = buf.push(3, frame(3))
    assert seq.frame_indices == (1, 2, 3)
    assert buf.n_pushed == 4 and buf.n_emitted == 2


def test_repeat_warmup_emits_from_first_frame():
    buf = FrameSequenceBuffer(length=3, warmup="repeat")
    seq = buf.push(0, frame(0))
    assert seq is not None and seq.frame_indices == (0, 0, 0)
    seq = buf.push(1, frame(1))
    assert seq.frame_indices == (0, 0, 1)
    seq = buf.push(2, frame(2))
    assert seq.frame_indices == (0, 1, 2)


def test_buffer_keeps_only_last_three_and_rejects_non_increasing():
    buf = FrameSequenceBuffer()
    for i in range(10):
        seq = buf.push(i, frame(i))
    assert seq.frame_indices == (7, 8, 9)
    assert seq.length == 3
    with pytest.raises(ValueError):
        buf.push(9, frame(9))


def test_reset_and_original_frames():
    buf = FrameSequenceBuffer(keep_original=True)
    orig = np.zeros((8, 12, 3), np.uint8)
    for i in range(3):
        seq = buf.push(i, frame(i), original_frame=orig)
    assert seq.original_frame is orig
    buf.reset()
    assert buf.push(0, frame(0)) is None and buf.n_pushed == 1
    buf2 = FrameSequenceBuffer(keep_original=False)
    for i in range(3):
        seq = buf2.push(i, frame(i), original_frame=orig)
    assert seq.original_frame is None


def test_invalid_args():
    with pytest.raises(ValueError):
        FrameSequenceBuffer(length=0)
    with pytest.raises(ValueError):
        FrameSequenceBuffer(warmup="pad")
