"""Checkpoint compatibility: third-party key layouts and oversized BN tensors."""
import pytest

torch = pytest.importorskip("torch")

from tennis_tracker.model import load_weights, tracknet_v1, tracknet_v2  # noqa: E402
from tennis_tracker.model.weights import _rename_keys  # noqa: E402


def chgyglin_style_state_dict(model):
    """Re-key a tracknet_v2 state dict the way ChgygLin/TrackNetV2-pytorch names it."""
    out = {}
    for k, v in model.state_dict().items():
        if k.startswith("conv18."):
            out["conv2d_18." + k.split(".", 1)[1]] = v
        else:
            layer, _, part, rest = k.split(".", 3)          # convN . block . 0|2 . name
            sub = "conv" if part == "0" else "bn"
            out[f"conv2d_{layer[4:]}.{sub}.{rest}"] = v
    return out


def test_rename_rules_map_chgyglin_keys():
    m = tracknet_v2()
    sd, changed = _rename_keys(chgyglin_style_state_dict(m))
    assert changed and set(sd) == set(m.state_dict())


def test_load_chgyglin_layout_exact(tmp_path):
    m = tracknet_v2()
    torch.save(chgyglin_style_state_dict(m), tmp_path / "track.pt")
    m2 = tracknet_v2()
    rep = load_weights(m2, tmp_path / "track.pt")
    assert rep.strategy == "renamed" and rep.complete
    assert torch.equal(m.conv18.weight, m2.conv18.weight)
    assert tuple(m2.conv18.weight.shape) == (3, 64, 1, 1)


def test_allow_truncate_oversized_bn(tmp_path):
    m = tracknet_v2()
    sd = m.state_dict()
    big = {k: (torch.cat([v, torch.full((100,), 7.0)]) if k == "conv1.block.2.weight" else v) for k, v in sd.items()}
    torch.save(big, tmp_path / "big.pt")
    rep = load_weights(tracknet_v2(), tmp_path / "big.pt")             # without the flag: partial load
    assert rep.strategy == "shape-order" and not rep.complete
    m2 = tracknet_v2()
    rep = load_weights(m2, tmp_path / "big.pt", allow_truncate=True)
    assert rep.complete and rep.truncated_keys == ["conv1.block.2.weight"]
    assert torch.equal(m2.conv1.block[2].weight, sd["conv1.block.2.weight"])
    assert "TRUNCATED" in rep.summary()


def test_v1_layout_still_exact(tmp_path):
    m = tracknet_v1()
    torch.save(m.state_dict(), tmp_path / "v1.pt")
    rep = load_weights(tracknet_v1(), tmp_path / "v1.pt", strict=True)
    assert rep.strategy == "exact" and rep.complete
