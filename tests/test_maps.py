"""Tests for the chemical-map nodes: Dequantize, CenterCrop, GroupWeightedMean,
ChemicalMapStatistics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.maps import (
    CenterCrop,
    ChemicalMapStatistics,
    Dequantize,
    GroupWeightedMean,
    MaskToMapResolution,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Dequantize
# --------------------------------------------------------------------------


def test_dequantize_golden():
    """q / scale + bias, per frame, matches a hand-computed reference."""
    cube = torch.tensor(
        [[[[10.0, 20.0, 30.0]]], [[[0.0, 65535.0, 100.0]]]], dtype=torch.float32
    )  # [2, 1, 1, 3]
    scale = torch.tensor([2.0, 100.0], dtype=torch.float32)
    bias = torch.tensor([1.0, -5.0], dtype=torch.float32)

    node = Dequantize()
    out = node(cube=cube, quant_scale=scale, quant_bias=bias)["cube"]

    expected = torch.stack([cube[0] / scale[0] + bias[0], cube[1] / scale[1] + bias[1]], dim=0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_dequantize_port_contract():
    node = Dequantize()
    cube = torch.rand(2, 4, 4, 5)
    scale = torch.rand(2) + 0.5
    bias = torch.rand(2)
    out = node(cube=cube, quant_scale=scale, quant_bias=bias)
    assert out["cube"].shape == cube.shape
    assert out["cube"].dtype == torch.float32


def test_dequantize_export_smoke():
    class _Probe(torch.nn.Module):
        def __init__(self, node):
            super().__init__()
            self.module = node

        def forward(self, cube, quant_scale, quant_bias):
            return self.module(cube=cube, quant_scale=quant_scale, quant_bias=quant_bias)["cube"]

    node = Dequantize().eval()
    example = (torch.rand(2, 3, 3, 4), torch.rand(2) + 0.5, torch.rand(2))
    program = torch.export.export(_Probe(node), example)

    other = (torch.rand(2, 3, 3, 4), torch.rand(2) + 0.5, torch.rand(2))
    traced = program.module()(*other)
    direct = node(cube=other[0], quant_scale=other[1], quant_bias=other[2])["cube"]
    assert torch.allclose(traced, direct, atol=1e-6)


# --------------------------------------------------------------------------
# CenterCrop
# --------------------------------------------------------------------------


def test_center_crop_golden():
    """Crops the centred region of a hand-built cube."""
    cube = torch.arange(4 * 4, dtype=torch.float32).reshape(1, 4, 4, 1)

    node = CenterCrop(crop_height=2, crop_width=2)
    out = node(cube=cube)

    expected_cube = torch.tensor([[5.0, 6.0], [9.0, 10.0]]).view(1, 2, 2, 1)
    assert torch.equal(out["cube"], expected_cube)
    assert set(out) == {"cube"}


def test_center_crop_declares_no_mask_port():
    """A mask port would be filled from the batch's coarse mask and cropped to the wrong size."""
    assert "mask" not in CenterCrop.INPUT_SPECS
    assert "mask" not in CenterCrop.OUTPUT_SPECS
    cube = torch.rand(1, 6, 6, 3)
    coarse_mask = torch.ones(1, 2, 2, dtype=torch.int32)
    out = CenterCrop(crop_height=4, crop_width=4)(cube=cube, mask=coarse_mask)
    assert out["cube"].shape == (1, 4, 4, 3)
    assert "mask" not in out


def test_center_crop_port_contract():
    node = CenterCrop(crop_height=2, crop_width=3)
    cube = torch.rand(2, 8, 9, 4)
    out = node(cube=cube)
    assert out["cube"].shape == (2, 2, 3, 4)
    assert out["cube"].dtype == torch.float32


def test_center_crop_export_smoke():
    class _Probe(torch.nn.Module):
        def __init__(self, node):
            super().__init__()
            self.module = node

        def forward(self, cube):
            return self.module(cube=cube)["cube"]

    node = CenterCrop(crop_height=2, crop_width=2).eval()
    example = torch.rand(1, 4, 4, 3)
    program = torch.export.export(_Probe(node), (example,))

    other = torch.rand(1, 4, 4, 3)
    traced = program.module()(other)
    direct = node(cube=other)["cube"]
    assert torch.allclose(traced, direct, atol=1e-6)


# --------------------------------------------------------------------------
# GroupWeightedMean
# --------------------------------------------------------------------------


def test_group_weighted_mean_golden_three_frames_two_groups():
    """Mask-weighted mean over three frames split across two groups."""
    node = GroupWeightedMean(max_groups=2, n_targets=1)

    # Group 0: frame A (4 valid pixels, value 10), frame B (2 valid pixels, value 20).
    # Group 1: frame C (3 valid pixels, value 100).
    maps = torch.zeros(3, 2, 2, 1)
    maps[0, :, :, 0] = 10.0
    maps[1, 0, :, 0] = 20.0  # only top row contributes once masked
    maps[2, :, :, 0] = 100.0

    masks = torch.zeros(3, 2, 2, dtype=torch.int32)
    masks[0] = 1  # all 4 pixels valid
    masks[1, 0, :] = 1  # 2 valid pixels (top row)
    masks[2] = 1
    masks[2, 1, 1] = 0  # drop one pixel -> 3 valid pixels

    groups = torch.tensor([0, 0, 1], dtype=torch.int32)

    out = node(map=maps, mask=masks, group=groups)

    # Group 0: sum = 4*10 + 2*20 = 80, count = 6 -> mean 13.333...
    expected_group0 = (4 * 10.0 + 2 * 20.0) / 6
    # Group 1: sum = 3*100 = 300, count = 3 -> mean 100
    expected_group1 = 100.0

    assert out["values"].shape == (2, 1)
    assert out["counts"].shape == (2,)
    assert torch.allclose(out["values"][0, 0], torch.tensor(expected_group0), atol=1e-5)
    assert torch.allclose(out["values"][1, 0], torch.tensor(expected_group1), atol=1e-5)
    assert torch.allclose(out["counts"], torch.tensor([6.0, 3.0]))

    computed = node.compute()
    assert set(computed.keys()) == {0, 1}
    assert torch.allclose(computed[0], torch.tensor([expected_group0]), atol=1e-5)
    assert torch.allclose(computed[1], torch.tensor([expected_group1]), atol=1e-5)


def test_group_weighted_mean_reset():
    node = GroupWeightedMean(max_groups=2, n_targets=1)
    maps = torch.full((1, 2, 2, 1), 5.0)
    masks = torch.ones(1, 2, 2, dtype=torch.int32)
    groups = torch.tensor([0], dtype=torch.int32)
    node(map=maps, mask=masks, group=groups)
    assert node.compute()
    node.reset()
    assert node.compute() == {}


def test_group_weighted_mean_port_contract():
    node = GroupWeightedMean(max_groups=4, n_targets=2)
    maps = torch.rand(3, 2, 2, 2)
    masks = torch.ones(3, 2, 2, dtype=torch.int32)
    groups = torch.tensor([0, 1, 2], dtype=torch.int32)
    out = node(map=maps, mask=masks, group=groups)
    assert out["values"].shape == (4, 2)
    assert out["values"].dtype == torch.float32
    assert out["counts"].shape == (4,)
    assert out["counts"].dtype == torch.float32


# --------------------------------------------------------------------------
# ChemicalMapStatistics
# --------------------------------------------------------------------------


def _toy_maps() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rebuild the notebook's four toy chemical maps (section 5.6/5.7)."""
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w]
    toy_mask = (((yy - 32) / 26) ** 2 + ((xx - 32) / 20) ** 2 < 1).astype(np.float32)
    rng = np.random.default_rng(1)
    ref = 35.0

    def recentre(mp: np.ndarray) -> np.ndarray:
        return mp - mp[toy_mask > 0].mean() + ref

    maps = {
        "smooth_gradient": recentre(35 + 15 * (32 - yy) / 32),
        "constant": np.full((h, w), ref),
        "pls_like_noise": recentre(ref + rng.normal(0, 300, (h, w))),
        "out_of_range": recentre(35 + 80 * np.exp(-((yy - 15) ** 2 + (xx - 32) ** 2) / 60)),
    }
    return toy_mask, maps


# Published numbers from the notebook (section 5.6/5.7): C0 for each toy map.
_PUBLISHED_NUGGETS = {
    "smooth_gradient": 0.0549,
    "constant": 0.0,
    "pls_like_noise": 94982.6294,
    "out_of_range": 2.8267,
}


@pytest.mark.parametrize("name", list(_PUBLISHED_NUGGETS))
def test_chemical_map_statistics_golden_toy_maps(name: str):
    """Nugget (C0) on the notebook's toy maps matches the published numbers."""
    toy_mask, maps = _toy_maps()
    mp = maps[name]

    node = ChemicalMapStatistics(mask_scale=1)
    y = torch.tensor(mp, dtype=torch.float32)[None, :, :, None]
    mask = torch.tensor(toy_mask, dtype=torch.int32)[None]
    out = node(map=y, mask=mask)

    assert out["nugget"].item() == pytest.approx(_PUBLISHED_NUGGETS[name], rel=1e-4, abs=2e-3)

    if name == "constant":
        assert out["variance"].item() == pytest.approx(0.0, abs=1e-6)
        # 1 - C0 / variance is defined as 1 when variance is (numerically) zero.
        assert out["structured_share"].item() == pytest.approx(1.0, abs=1e-6)
    else:
        expected_variance = float(np.var(mp[toy_mask > 0]))
        expected_structured_share = 1.0 - _PUBLISHED_NUGGETS[name] / expected_variance
        assert out["variance"].item() == pytest.approx(expected_variance, rel=1e-3)
        assert out["structured_share"].item() == pytest.approx(expected_structured_share, abs=2e-3)


def test_chemical_map_statistics_out_of_range_share():
    """A map with pixels outside 0 to 100 reports the correct masked share."""
    node = ChemicalMapStatistics(mask_scale=1)
    y = torch.zeros(1, 2, 2, 1)
    y[0, 0, 0, 0] = -5.0  # out of range
    y[0, 0, 1, 0] = 50.0
    y[0, 1, 0, 0] = 150.0  # out of range
    y[0, 1, 1, 0] = 50.0
    mask = torch.ones(1, 2, 2, dtype=torch.int32)

    out = node(map=y, mask=mask)
    assert out["out_of_range_share"].item() == pytest.approx(0.5, abs=1e-6)


def test_chemical_map_statistics_mask_scale_2_upsamples_nearest():
    """mask_scale=2 upsamples a half-resolution mask with nearest interpolation."""
    node_full = ChemicalMapStatistics(mask_scale=1)
    node_half = ChemicalMapStatistics(mask_scale=2)

    torch.manual_seed(0)
    y = torch.rand(1, 4, 4, 1) * 100

    full_mask = torch.ones(1, 4, 4, dtype=torch.int32)
    half_mask = torch.ones(1, 2, 2, dtype=torch.int32)  # upsamples to all-ones 4x4

    out_full = node_full(map=y, mask=full_mask)
    out_half = node_half(map=y, mask=half_mask)

    assert torch.allclose(out_full["variance"], out_half["variance"], atol=1e-6)
    assert torch.allclose(out_full["nugget"], out_half["nugget"], atol=1e-6)


def test_chemical_map_statistics_no_mask_uses_all_pixels():
    node = ChemicalMapStatistics()
    y = torch.full((1, 4, 4, 1), 42.0)
    out = node(map=y)
    assert out["variance"].item() == pytest.approx(0.0, abs=1e-6)


def test_chemical_map_statistics_port_contract():
    node = ChemicalMapStatistics()
    y = torch.rand(2, 8, 8, 1) * 100
    mask = torch.ones(2, 8, 8, dtype=torch.int32)
    out = node(map=y, mask=mask)
    for port in ("variance", "nugget", "structured_share", "out_of_range_share"):
        assert out[port].shape == (2,)
        assert out[port].dtype == torch.float32


def test_chemical_map_statistics_invalid_mask_scale_rejected():
    with pytest.raises(ValueError):
        ChemicalMapStatistics(mask_scale=3)


def test_chemical_map_statistics_export_smoke():
    class _Probe(torch.nn.Module):
        def __init__(self, node):
            super().__init__()
            self.module = node

        def forward(self, map, mask):  # noqa: A002
            out = self.module(map=map, mask=mask)
            return (
                out["variance"],
                out["nugget"],
                out["structured_share"],
                out["out_of_range_share"],
            )

    node = ChemicalMapStatistics(mask_scale=1).eval()
    example = (torch.rand(1, 8, 8, 1) * 100, torch.ones(1, 8, 8, dtype=torch.int32))
    program = torch.export.export(_Probe(node), example)

    other = (torch.rand(1, 8, 8, 1) * 100, torch.ones(1, 8, 8, dtype=torch.int32))
    traced = program.module()(*other)
    direct = node(map=other[0], mask=other[1])
    for traced_value, direct_value in zip(traced, direct.values(), strict=True):
        assert torch.allclose(traced_value, direct_value, atol=1e-5)


# ----- MaskToMapResolution --------------------------------------------------


def _blob_mask(h: int, w: int, seed: int = 11) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    mask = torch.zeros(1, h, w, dtype=torch.int32)
    mask[0, h // 6 : h - h // 6, w // 6 : w - w // 6] = 1
    holes = rng.random((h, w)) < 0.08
    mask[0][torch.from_numpy(holes)] = 0
    return mask


def _reference_mask_to_map(mask: np.ndarray, map_hw, crop_factor, downsample, erode_px):
    from scipy import ndimage

    h, w = mask.shape
    crop_h = min(map_hw[0] * crop_factor, h)
    crop_w = min(map_hw[1] * crop_factor, w)
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = (mask[top : top + crop_h, left : left + crop_w] > 0).astype(np.float64)
    out_h, out_w = crop_h // downsample, crop_w // downsample
    # Bilinear by an exact factor of two with align_corners False is the 2 x 2 block mean.
    assert downsample == 2
    block = cropped[: out_h * 2, : out_w * 2].reshape(out_h, 2, out_w, 2).mean(axis=(1, 3))
    binary = np.round(block).astype(bool)
    for _ in range(erode_px):
        binary = ndimage.binary_erosion(binary, border_value=0)
    return binary.astype(np.int32)


@pytest.mark.parametrize(("crop_factor", "erode_px"), [(2, 1), (1, 1), (2, 0)])
def test_mask_to_map_matches_scipy_reference(crop_factor: int, erode_px: int) -> None:
    mask = _blob_mask(40, 28)
    map_hw = (10, 6) if crop_factor == 2 else (20, 12)
    chem_map = torch.zeros(1, map_hw[0], map_hw[1], 1)
    node = MaskToMapResolution(crop_factor=crop_factor, downsample=2, erode_px=erode_px)
    out = node(mask=mask, map=chem_map)["mask"]
    expected = _reference_mask_to_map(mask[0].numpy(), map_hw, crop_factor, 2, erode_px)
    assert out.shape == (1, *expected.shape)
    assert np.array_equal(out[0].numpy(), expected)


def test_mask_to_map_clamps_crop_to_the_slice() -> None:
    mask = _blob_mask(16, 12)
    chem_map = torch.zeros(1, 20, 20, 1)  # asks for a 40 x 40 crop out of a 16 x 12 mask
    out = MaskToMapResolution(crop_factor=2, downsample=2, erode_px=0)(mask=mask, map=chem_map)
    assert out["mask"].shape == (1, 8, 6)


def test_mask_to_map_port_contract() -> None:
    node = MaskToMapResolution()
    out = node(mask=_blob_mask(24, 20), map=torch.zeros(1, 6, 5, 1))
    assert out["mask"].dtype == torch.int32
    assert out["mask"].shape == (1, 6, 5)
    assert set(out) == set(node.OUTPUT_SPECS)


class _MaskMapSource(Node):
    """Module-scope source so the saved pipeline can be rebuilt by class path."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.MASK})
    OUTPUT_SPECS = {
        "mask": PortSpec(dtype=torch.int32, shape=(-1, -1, -1), description="m"),
        "map": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1), description="p"),
    }

    def forward(self, **_):
        return {"mask": _blob_mask(24, 20), "map": torch.zeros(1, 6, 5, 1)}


def test_mask_to_map_reload_smoke(tmp_path) -> None:
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    pipe = CuvisPipeline("mask_to_map_reload")
    src = _MaskMapSource(name="src")
    node = MaskToMapResolution(name="mask_map", crop_factor=2, downsample=2, erode_px=1)
    pipe.connect((src.outputs.mask, node.inputs.mask), (src.outputs.map, node.inputs.map))
    before = pipe.forward(batch={})
    path = tmp_path / "mask_to_map.yaml"
    pipe.save_to_file(str(path))
    after = PipelineBuilder().build_from_config(str(path)).forward(batch={})
    assert torch.equal(before[(node.name, "mask")], after[(node.name, "mask")])


def test_mask_to_map_export_smoke() -> None:
    node = MaskToMapResolution().eval()

    class _Probe(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, mask, chem_map):
            return self.inner(mask=mask, map=chem_map)["mask"]

    example = (_blob_mask(24, 20, seed=1), torch.zeros(1, 6, 5, 1))
    program = torch.export.export(_Probe(node), example)
    other = (_blob_mask(24, 20, seed=2), torch.zeros(1, 6, 5, 1))
    assert torch.equal(program.module()(*other), node(mask=other[0], map=other[1])["mask"])
