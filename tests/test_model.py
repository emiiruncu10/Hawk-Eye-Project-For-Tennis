import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.model import build_model, load_weights, save_weights, tracknet_v1, tracknet_v2  # noqa: E402
from tennis_tracker.model.tracknet import TrackNet  # noqa: E402


def test_v2_forward_and_heatmaps_shape_range():
    m = tracknet_v2().eval()
    x = torch.rand(2, 9, 32, 64)
    with torch.inference_mode():
        logits = m(x)
        hm = m.heatmaps(x)
    assert tuple(logits.shape) == (2, 3, 32, 64)
    assert tuple(hm.shape) == (2, 3, 32, 64)
    assert float(hm.min()) >= 0 and float(hm.max()) <= 1


def test_v1_forward_and_heatmaps_shape_range():
    m = tracknet_v1().eval()
    x = torch.rand(1, 9, 24, 40)
    with torch.inference_mode():
        logits = m(x)
        hm = m.heatmaps(x)
    assert tuple(logits.shape) == (1, 256, 24, 40)
    assert tuple(hm.shape) == (1, 1, 24, 40)
    assert float(hm.min()) >= 0 and float(hm.max()) <= 1


def test_v1_state_dict_keys_match_yastrebksv_layout():
    keys = list(tracknet_v1().state_dict().keys())
    assert keys[0] == "conv1.block.0.weight"            # conv
    assert "conv1.block.2.running_mean" in keys         # batchnorm after relu
    assert "conv18.block.0.weight" in keys
    assert not any("ups" in k or "pool" in k for k in keys)
    assert tracknet_v1().state_dict()["conv18.block.0.weight"].shape == (256, 64, 3, 3)
    assert tracknet_v2().state_dict()["conv11.block.0.weight"].shape == (256, 768, 3, 3)


def test_input_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        tracknet_v2()(torch.rand(1, 9, 30, 64))
    with pytest.raises(ValueError):
        build_model("v9")
    with pytest.raises(ValueError):
        TrackNet(head="softmax256", out_channels=3)


def test_save_load_roundtrip_exact(tmp_path):
    m = tracknet_v2()
    p = save_weights(m, tmp_path / "w.pt")
    m2 = tracknet_v2()
    rep = load_weights(m2, p)
    assert rep.strategy == "exact" and rep.complete
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_load_wrapped_and_dataparallel_checkpoint(tmp_path):
    m = tracknet_v2()
    sd = {"module." + k: v for k, v in m.state_dict().items()}
    torch.save({"epoch": 3, "model_state_dict": sd}, tmp_path / "wrapped.pt")
    m2 = tracknet_v2()
    rep = load_weights(m2, tmp_path / "wrapped.pt")
    assert rep.complete
    assert torch.equal(m.conv18.weight, m2.conv18.weight)          # v2 ends with a plain 1x1 conv


def test_load_by_shape_order_with_renamed_keys(tmp_path):
    m = tracknet_v2()
    sd = {f"layer{i}": v for i, (k, v) in enumerate(m.state_dict().items())}
    torch.save(sd, tmp_path / "renamed.pt")
    m2 = tracknet_v2()
    rep = load_weights(m2, tmp_path / "renamed.pt")
    assert rep.strategy == "shape-order" and rep.complete, rep.summary()
    assert torch.equal(m.conv5.block[0].weight, m2.conv5.block[0].weight)


def test_load_missing_or_incompatible(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_weights(tracknet_v2(), tmp_path / "nope.pt")
    torch.save(tracknet_v1().state_dict(), tmp_path / "v1.pt")
    rep = load_weights(tracknet_v2(), tmp_path / "v1.pt")
    assert not rep.complete and rep.missing_keys
    with pytest.raises(RuntimeError):
        load_weights(tracknet_v2(), tmp_path / "v1.pt", strict=True)
    torch.save({"not": "tensors", "n": 3}, tmp_path / "junk.pt")
    with pytest.raises(ValueError):
        load_weights(tracknet_v2(), tmp_path / "junk.pt")
