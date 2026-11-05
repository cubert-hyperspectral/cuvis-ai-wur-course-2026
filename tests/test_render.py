"""Tests for BandSweep (cuvis_ai_wur_course.node.render).

Golden cases are tiny random cubes whose per-band percentile stretch is computed here with
``torch.quantile``, so the frame order (one frame per band, every ``step``-th band, batch
elements one after another) and the stretch are known by construction. The wavelength label
is checked by its footprint, not by its pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.render import BandSweep

pytestmark = pytest.mark.unit

WAVELENGTHS = np.array([450, 500, 550, 600], dtype=np.int32)


class _CubeSource(Node):
    """Minimal source emitting a fixed cube and its wavelengths, for the reload smoke test.

    Defined at module scope so the class resolves as ``<module>.<ClassName>`` after a
    pipeline save/reload round trip.
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.HYPERSPECTRAL})

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "wavelengths": PortSpec(dtype=np.int32, shape=(-1,)),
    }

    def forward(self, **_):
        return {"cube": _cube(), "wavelengths": WAVELENGTHS}


def _cube(b: int = 1, h: int = 6, w: int = 7, c: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(b, h, w, c, dtype=torch.float32) * 3.0 - 1.0  # outside [0, 1] on purpose


def _stretch(band: torch.Tensor, low: float = 0.005, high: float = 0.995) -> torch.Tensor:
    q = torch.quantile(band.flatten(), torch.tensor([low, high]))
    return ((band - q[0]) / (q[1] - q[0]).clamp_min(1e-6)).clamp(0.0, 1.0)


def test_one_frame_per_band_in_band_order():
    cube = _cube()

    out = BandSweep(label=False)(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]

    assert out.shape == (4, 6, 7, 3)
    assert out.dtype == torch.float32
    for k in range(4):
        expected = _stretch(cube[0, ..., k])
        assert torch.allclose(out[k, ..., 0], expected, atol=1e-6)
        assert torch.equal(out[k, ..., 0], out[k, ..., 1])
        assert torch.equal(out[k, ..., 0], out[k, ..., 2])


def test_step_renders_every_kth_band():
    cube = _cube()

    out = BandSweep(step=3, label=False)(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]

    assert out.shape[0] == 2  # bands 0 and 3
    assert torch.allclose(out[1, ..., 0], _stretch(cube[0, ..., 3]), atol=1e-6)


def test_batch_elements_follow_one_another():
    cube = _cube(b=2)

    out = BandSweep(label=False)(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]

    assert out.shape[0] == 8
    assert torch.allclose(out[5, ..., 0], _stretch(cube[1, ..., 1]), atol=1e-6)


def test_custom_percentiles_change_the_stretch():
    cube = _cube()

    out = BandSweep(low=0.1, high=0.9, label=False)(cube=cube, wavelengths=WAVELENGTHS)
    expected = _stretch(cube[0, ..., 2], low=0.1, high=0.9)

    assert torch.allclose(out["rgb_image"][2, ..., 0], expected, atol=1e-6)


def test_constant_band_renders_black_not_nan():
    cube = torch.ones(1, 6, 7, 1, dtype=torch.float32)

    out = BandSweep(label=False)(cube=cube, wavelengths=WAVELENGTHS[:1])["rgb_image"]

    assert bool(torch.isfinite(out).all())
    assert float(out.max()) == 0.0


def test_label_sits_in_the_bottom_left_corner_and_differs_per_band():
    cube = _cube(h=60, w=240)

    plain = BandSweep(label=False)(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]
    labelled = BandSweep(label=True)(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]

    assert float(labelled.min()) >= 0.0 and float(labelled.max()) <= 1.0
    diff = (labelled - plain).abs().amax(dim=-1)  # [N, H, W]
    quantisation = 1.0 / 255.0  # the labelled frame went through uint8 once
    assert bool((diff[:, 20:, :160] > 2 * quantisation).any())
    assert float(diff[:, :20, :].max()) <= quantisation + 1e-6
    assert float(diff[:, :, 160:].max()) <= quantisation + 1e-6
    assert not torch.equal(labelled[0, 20:, :160], labelled[1, 20:, :160])


def test_rejects_a_step_below_one():
    with pytest.raises(ValueError):
        BandSweep(step=0)


def test_reload_smoke(tmp_path):
    source = _CubeSource(name="source")
    sweep = BandSweep(name="sweep", step=2, label=False)
    pipeline = CuvisPipeline("band_sweep")
    pipeline.connect(
        (source.outputs.cube, sweep.inputs.cube),
        (source.outputs.wavelengths, sweep.inputs.wavelengths),
    )
    path = tmp_path / "band_sweep.yaml"
    pipeline.save_to_file(str(path))

    rebuilt = PipelineBuilder().build_from_config(str(path))

    names = (
        {n.name: n for n in rebuilt.nodes.values()}
        if isinstance(rebuilt.nodes, dict)
        else {n.name: n for n in rebuilt.nodes}
    )
    assert isinstance(names["sweep"], BandSweep)
    assert names["sweep"].step == 2
    assert names["sweep"].label is False
