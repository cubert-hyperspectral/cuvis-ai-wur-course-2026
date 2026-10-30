"""Tests for ``cuvis_ai_wur_course.node.source.CourseFrame``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.source import CourseFrame

pytestmark = pytest.mark.unit


def _mean_spectra_batch(b: int = 2, c: int = 4) -> dict[str, torch.Tensor]:
    """A mean-spectra-like batch: target + fold, no mask/group/quant."""
    return {
        "cube": torch.rand(b, 1, 1, c, dtype=torch.float64),  # upstream dtype before the node casts
        "wavelengths": torch.linspace(747.92, 1015.78, c, dtype=torch.float64)
        .unsqueeze(0)
        .expand(b, -1),
        "target": torch.rand(b, 1, dtype=torch.float64) * 50.0,
        "fold": torch.randint(0, 5, (b,), dtype=torch.int64),
    }


def _image_batch(b: int = 1, h: int = 5, w: int = 6, c: int = 4) -> dict[str, torch.Tensor]:
    """An image-slice-like batch: mask + group + quant, no target."""
    return {
        "cube": torch.rand(b, h, w, c, dtype=torch.float32),
        "wavelengths": torch.linspace(747.92, 1015.78, c, dtype=torch.float32)
        .unsqueeze(0)
        .expand(b, -1),
        "mask": torch.randint(0, 2, (b, h, w), dtype=torch.int32),
        "group": torch.full((b,), 101, dtype=torch.int32),
        "quant_scale": torch.full((b,), 1234.5, dtype=torch.float32),
        "quant_bias": torch.full((b,), 0.02, dtype=torch.float32),
    }


# ----- 1. golden-reference ---------------------------------------------


def test_mean_spectra_batch_golden() -> None:
    node = CourseFrame()
    batch = _mean_spectra_batch()
    out = node(**batch)

    assert torch.allclose(out["cube"], batch["cube"].to(torch.float32))
    expected_wl = batch["wavelengths"][0].round().to(torch.int32).cpu().numpy()
    assert out["wavelengths"].dtype == np.int32
    assert (out["wavelengths"] == expected_wl).all()
    assert torch.allclose(out["target"], batch["target"].to(torch.float32))
    assert torch.equal(out["fold"], batch["fold"].to(torch.int32))
    assert "mask" not in out
    assert "group" not in out
    assert "quant_scale" not in out
    assert "quant_bias" not in out


def test_image_batch_golden() -> None:
    node = CourseFrame()
    batch = _image_batch()
    out = node(**batch)

    assert torch.allclose(out["cube"], batch["cube"])
    assert torch.equal(out["mask"], batch["mask"])
    assert torch.equal(out["group"], batch["group"])
    assert torch.allclose(out["quant_scale"], batch["quant_scale"])
    assert torch.allclose(out["quant_bias"], batch["quant_bias"])
    assert "target" not in out
    assert "fold" not in out


# ----- 2. port contract --------------------------------------------------


def test_port_contract_mean_spectra() -> None:
    node = CourseFrame()
    out = node(**_mean_spectra_batch())
    for port_name in ("cube", "wavelengths", "target", "fold"):
        spec = node.OUTPUT_SPECS[port_name]
        assert port_name in out
        if port_name == "wavelengths":
            # Handed on as the builtin source does: a NumPy int32 vector [C].
            assert isinstance(out[port_name], np.ndarray)
            assert out[port_name].dtype == np.int32
            continue
        assert isinstance(out[port_name], torch.Tensor)
        assert out[port_name].dtype == spec.dtype


def test_port_contract_image() -> None:
    node = CourseFrame()
    out = node(**_image_batch())
    for port_name in ("cube", "wavelengths", "mask", "group", "quant_scale", "quant_bias"):
        spec = node.OUTPUT_SPECS[port_name]
        assert port_name in out
        if port_name == "wavelengths":
            # Handed on as the builtin source does: a NumPy int32 vector [C].
            assert isinstance(out[port_name], np.ndarray)
            assert out[port_name].dtype == np.int32
            continue
        assert isinstance(out[port_name], torch.Tensor)
        assert out[port_name].dtype == spec.dtype


# ----- 3. reload smoke -----------------------------------------------------


class _CubeSink(Node):
    """Minimal terminal node so CourseFrame has a consumer and joins the graph."""

    INPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), description="Cube in."),
    }
    OUTPUT_SPECS = {
        "seen": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), description="Cube out."),
    }

    def forward(self, cube: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        return {"seen": cube}


def test_reload_smoke(tmp_path: Path) -> None:
    pipeline = CuvisPipeline("course_frame_reload_test")
    node = CourseFrame(name="frame")
    sink = _CubeSink(name="sink")
    pipeline.connect(node.outputs.cube, sink.inputs.cube)

    out_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(str(out_path))

    rebuilt = PipelineBuilder().build_from_config(str(out_path))
    rebuilt_node = {n.name: n for n in rebuilt.nodes()}["frame"]
    assert isinstance(rebuilt_node, CourseFrame)

    batch = _image_batch()
    before = node(**batch)
    after = rebuilt_node(**batch)
    assert torch.allclose(before["cube"], after["cube"])


# ----- 4. export smoke ------------------------------------------------------
#
# Not applicable: CourseFrame is a source node and, like the builtin CU3SDataNode, hands
# the wavelengths on as a NumPy array, so it sits in front of an exported span, never
# inside one (see the class docstring).
