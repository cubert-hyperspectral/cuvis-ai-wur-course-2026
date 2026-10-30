"""Tests for UNetChemicalMapEnsemble.

All tests use random weights (``weights=None``); nothing downloads a
checkpoint. A minimal valid input size is derived from the study's
valid-convolution size arithmetic (no padding anywhere in the U-Net, a
2x2 max-pool at each of the four contraction levels), so the chosen size
is the smallest multiple that survives every stage without an odd
dimension reaching a pooling layer.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_wur_course.node.unet import UNetChemicalMapEnsemble

pytestmark = pytest.mark.unit

_INPUT_CHANNELS = 62
# Smallest square input size (post Conv3D front-end stride-2 compression and
# four more stride-2 poolings) that keeps every intermediate size even and
# positive; derived from the study's unet_size_table arithmetic in the
# notebook (section 4.4): a size_in of 376 survives all five halvings.
_MIN_HW = 376
_EXPECTED_OUT_HW = 4


def _small_cube(batch: int = 1) -> torch.Tensor:
    return torch.rand(batch, _MIN_HW, _MIN_HW, _INPUT_CHANNELS, dtype=torch.float32)


# --------------------------------------------------------------------------
# 1. golden-reference / parity: ensemble mean equals mean of direct member calls
# --------------------------------------------------------------------------


def test_two_member_mean_equals_mean_of_direct_calls():
    torch.manual_seed(0)
    node = UNetChemicalMapEnsemble(n_members=2, weights=None, input_channels=_INPUT_CHANNELS)
    cube = _small_cube()

    out = node(cube=cube)["map"]

    x = cube.permute(0, 3, 1, 2)
    member_outputs = [member(x) for member in node.members]
    expected = torch.stack(member_outputs, dim=0).mean(dim=0).permute(0, 2, 3, 1)

    assert torch.allclose(out, expected, atol=1e-5)


def test_single_member_equals_compound_model_call():
    """n_members=1 output equals calling that one member directly."""
    torch.manual_seed(1)
    node = UNetChemicalMapEnsemble(n_members=1, weights=None, input_channels=_INPUT_CHANNELS)
    cube = _small_cube()

    out = node(cube=cube)["map"]
    x = cube.permute(0, 3, 1, 2)
    expected = node.members[0](x).permute(0, 2, 3, 1)

    assert torch.allclose(out, expected, atol=1e-6)


# --------------------------------------------------------------------------
# 2. output shape matches the crop arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_members", [1, 2])
def test_output_shape_matches_crop_arithmetic(n_members: int):
    torch.manual_seed(0)
    node = UNetChemicalMapEnsemble(
        n_members=n_members, weights=None, input_channels=_INPUT_CHANNELS
    )
    cube = _small_cube()
    out = node(cube=cube)["map"]
    assert out.shape == (1, _EXPECTED_OUT_HW, _EXPECTED_OUT_HW, 1)


# --------------------------------------------------------------------------
# 3. port contract
# --------------------------------------------------------------------------


def test_port_contract():
    torch.manual_seed(0)
    node = UNetChemicalMapEnsemble(n_members=1, weights=None, input_channels=_INPUT_CHANNELS)
    cube = _small_cube()
    out = node(cube=cube)
    assert "map" in out
    assert out["map"].dtype == torch.float32
    assert out["map"].shape[0] == cube.shape[0]
    assert out["map"].shape[-1] == 1


# --------------------------------------------------------------------------
# 4. construction validation
# --------------------------------------------------------------------------


def test_n_members_out_of_range_rejected():
    with pytest.raises(ValueError):
        UNetChemicalMapEnsemble(n_members=0, weights=None)
    with pytest.raises(ValueError):
        UNetChemicalMapEnsemble(n_members=6, weights=None)


def test_member_ids_length_mismatch_rejected():
    with pytest.raises(ValueError):
        UNetChemicalMapEnsemble(n_members=2, member_ids=[1, 2, 3], weights=None)


def test_default_member_ids():
    node = UNetChemicalMapEnsemble(n_members=3, weights=None)
    assert node.member_ids == [1, 2, 3]


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        UNetChemicalMapEnsemble(n_members=1, weights="bogus")


# --------------------------------------------------------------------------
# 5. reload with weights=None (hparams round-trip through the YAML contract)
# --------------------------------------------------------------------------


def _is_json_safe(value: object) -> bool:
    if isinstance(value, (int, float, str, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


def test_hparams_are_json_serializable():
    """Every captured constructor kwarg (node.hparams, via Serializable) round-trips."""
    node = UNetChemicalMapEnsemble(n_members=2, weights=None, input_channels=_INPUT_CHANNELS)
    assert node.hparams, "hparams must be populated by Serializable.__init__"
    for key, value in node.hparams.items():
        assert _is_json_safe(value), f"hparams[{key!r}] = {value!r} is not JSON-safe"


def test_rebuild_from_hparams_matches_architecture():
    """A fresh node built from the same hparams has the same member architecture."""
    torch.manual_seed(0)
    node_a = UNetChemicalMapEnsemble(n_members=2, weights=None, input_channels=_INPUT_CHANNELS)
    node_b = UNetChemicalMapEnsemble(
        n_members=node_a.n_members,
        member_ids=node_a.member_ids,
        weights=node_a.weights,
        hf_repo=node_a.hf_repo,
        hf_revision=node_a.hf_revision,
        input_channels=node_a.input_channels,
        conv3d_kernel=node_a.conv3d_kernel,
        conv3d_filters=node_a.conv3d_filters,
        out_channels=node_a.out_channels,
        pad=node_a.pad,
        bilinear=node_a.bilinear,
        normalization=node_a.normalization,
    )
    params_a = sum(p.numel() for p in node_a.parameters())
    params_b = sum(p.numel() for p in node_b.parameters())
    assert params_a == params_b


# --------------------------------------------------------------------------
# 6. freeze / unfreeze (learnable submodules, no TRAINABLE_BUFFERS)
# --------------------------------------------------------------------------


def test_freeze_unfreeze_toggles_all_submodule_parameters():
    node = UNetChemicalMapEnsemble(n_members=1, weights=None, input_channels=_INPUT_CHANNELS)
    assert all(p.requires_grad for p in node.parameters())

    node.freeze()
    assert node.frozen
    assert all(not p.requires_grad for p in node.parameters())

    node.unfreeze()
    assert not node.frozen
    assert all(p.requires_grad for p in node.parameters())


# --------------------------------------------------------------------------
# 7. export smoke
# --------------------------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    def __init__(self, node: UNetChemicalMapEnsemble):
        super().__init__()
        self.module = node

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        return self.module(cube=cube)["map"]


def test_export_smoke():
    torch.manual_seed(0)
    node = UNetChemicalMapEnsemble(n_members=1, weights=None, input_channels=_INPUT_CHANNELS).eval()
    example = _small_cube()
    program = torch.export.export(_ExportProbe(node), (example,))

    other = _small_cube()
    traced = program.module()(other)
    direct = node(cube=other)["map"]
    assert torch.allclose(traced, direct, atol=1e-5)
