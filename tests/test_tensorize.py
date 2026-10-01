import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.inference.preprocess import TrackNetInputSpec  # noqa: E402
from tennis_tracker.inference.sequence import FrameSequence  # noqa: E402
from tennis_tracker.inference.tensorize import (  # noqa: E402
    BatchCollator,
    check_environment,
    numpy_to_tensor,
    sequences_to_tensor,
)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_sequences_to_tensor_shape_dtype_device(device):
    stacks = [np.random.default_rng(i).random((9, 32, 48), dtype=np.float32) for i in range(4)]
    t = sequences_to_tensor(stacks, device)
    assert tuple(t.shape) == (4, 9, 32, 48)
    assert t.dtype == torch.float32
    assert t.device.type == device
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    assert np.allclose(t.cpu().numpy(), np.stack(stacks))


def test_float64_input_is_cast_to_float32():
    t = sequences_to_tensor(np.zeros((1, 9, 4, 4), np.float64), "cpu")
    assert t.dtype == torch.float32


def test_numpy_to_tensor_fallback(monkeypatch):
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ref = numpy_to_tensor(arr)

    def boom(_):
        raise RuntimeError("Numpy is not available")
    monkeypatch.setattr(torch, "from_numpy", boom)
    t = numpy_to_tensor(arr)
    assert torch.equal(t, ref) and t.dtype == torch.float32 and tuple(t.shape) == (2, 3, 4)


def test_check_environment_reports_bridge_and_device():
    info = check_environment("cpu")
    assert info["numpy_bridge"] in ("from_numpy", "frombuffer-fallback")
    assert info["device"] == "cpu" and info["device_ok"]


def test_batch_collator_batches_and_flushes():
    spec = TrackNetInputSpec(height=16, width=24)
    col = BatchCollator(spec, batch_size=3, device="cpu", original_size=(1920, 1080))

    def seq(i):
        return FrameSequence((i - 2, i - 1, i), [np.full((3, 16, 24), i / 10, np.float32)] * 3, i / 30.0)

    assert col.add(seq(2)) is None and col.add(seq(3)) is None
    b = col.add(seq(4))
    assert b is not None and len(b) == 3
    assert tuple(b.tensor.shape) == (3, 9, 16, 24)
    assert b.frame_indices.tolist() == [2, 3, 4]
    assert b.sequence_indices[0] == (0, 1, 2)
    assert np.allclose(b.timestamps_s, [2 / 30, 3 / 30, 4 / 30])
    assert b.original_size == (1920, 1080) and b.spec == spec
    assert col.flush() is None
    col.add(seq(5))
    tail = col.flush()
    assert tail is not None and len(tail) == 1 and float(tail.tensor.mean()) == pytest.approx(0.5)
