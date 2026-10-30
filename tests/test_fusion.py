"""Tests for cuvis_ai_wur_course.node.fusion: ChannelSlice, PredictionMean, SpatialMean."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.fusion import ChannelSlice, PredictionMean, SpatialMean

pytestmark = pytest.mark.unit


# ----- shared minimal source/sink for reload and pipeline tests ---------


class _CubeSource(Node):
    """Minimal source node: no inputs, emits a fixed cube."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), description="cube"),
    }

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(channels=channels, **kwargs)
        self.channels = channels

    def forward(self, **_):
        torch.manual_seed(0)
        return {"cube": torch.rand(2, 3, 3, self.channels, dtype=torch.float32)}


class _CubeSink(Node):
    """Minimal sink node: consumes a cube, passes it through as 'seen'."""

    INPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), description="cube"),
    }
    OUTPUT_SPECS = {
        "seen": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), description="seen"),
    }

    def forward(self, cube, **_):
        return {"seen": cube}


class _PredictionSource(Node):
    """Minimal source node: emits a fixed [B, T] prediction."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "prediction": PortSpec(dtype=torch.float32, shape=(-1, -1), description="prediction"),
    }

    def __init__(self, value: float, batch: int = 2, n_targets: int = 1, **kwargs) -> None:
        super().__init__(value=value, batch=batch, n_targets=n_targets, **kwargs)
        self.value = value
        self.batch = batch
        self.n_targets = n_targets

    def forward(self, **_):
        return {
            "prediction": torch.full((self.batch, self.n_targets), self.value, dtype=torch.float32)
        }


class _PredictionSink(Node):
    INPUT_SPECS = {
        "prediction": PortSpec(dtype=torch.float32, shape=(-1, -1), description="prediction"),
    }
    OUTPUT_SPECS = {
        "seen": PortSpec(dtype=torch.float32, shape=(-1, -1), description="seen"),
    }

    def forward(self, prediction, **_):
        return {"seen": prediction}


# ----- ChannelSlice --------------------------------------------------------


def test_channel_slice_golden_reference():
    node = ChannelSlice(start=2, stop=5)
    cube = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 2, 2, 8).contiguous()

    out = node(cube=cube)["cube"]
    expected = cube[..., 2:5]
    assert torch.allclose(out, expected)
    assert out.shape[-1] == 3


def test_channel_slice_rejects_bad_range():
    with pytest.raises(ValueError):
        ChannelSlice(start=5, stop=5)
    with pytest.raises(ValueError):
        ChannelSlice(start=-1, stop=3)


def test_channel_slice_slices_wavelengths_alike():
    node = ChannelSlice(start=1, stop=3)
    cube = torch.rand(1, 1, 1, 5, dtype=torch.float32)
    wavelengths = np.array([700, 710, 720, 730, 740], dtype=np.int32)

    out = node(cube=cube, wavelengths=wavelengths)
    np.testing.assert_array_equal(out["wavelengths"], np.array([710, 720], dtype=np.int32))


def test_channel_slice_812_block_into_spatial_and_spectral():
    """Matches the course's 812-wide pork feature block: spatial 0:512, spectral 512:812."""
    spatial = ChannelSlice(start=0, stop=512)
    spectral = ChannelSlice(start=512, stop=812)
    block = torch.rand(1, 1, 1, 812, dtype=torch.float32)

    spatial_out = spatial(cube=block)["cube"]
    spectral_out = spectral(cube=block)["cube"]
    assert spatial_out.shape[-1] == 512
    assert spectral_out.shape[-1] == 300
    assert torch.allclose(spatial_out, block[..., :512])
    assert torch.allclose(spectral_out, block[..., 512:812])


def test_channel_slice_port_contract():
    node = ChannelSlice(start=0, stop=4)
    cube = torch.rand(2, 3, 3, 6, dtype=torch.float32)
    out = node(cube=cube)
    for port_name, spec in node.OUTPUT_SPECS.items():
        if port_name not in out:
            assert spec.optional
            continue
        tensor = out[port_name]
        if isinstance(tensor, torch.Tensor):
            assert tensor.dtype == spec.dtype
    assert out["cube"].shape == (2, 3, 3, 4)


def test_channel_slice_reload_smoke(tmp_path):
    pipe = CuvisPipeline("channel_slice_reload_test")
    src = _CubeSource(channels=8, name="src")
    node = ChannelSlice(start=2, stop=6, name="slice")
    sink = _CubeSink(name="sink")
    pipe.connect(src.outputs.cube, node.inputs.cube)
    pipe.connect(node.outputs.cube, sink.inputs.cube)

    yaml_path = tmp_path / "channel_slice_pipeline.yaml"
    pipe.save_to_file(yaml_path)

    rebuilt = PipelineBuilder().build_from_config(yaml_path)
    assert rebuilt is not None


def test_channel_slice_export_smoke():
    node = ChannelSlice(start=1, stop=5).eval()

    class _Probe(torch.nn.Module):
        def forward(self, cube: torch.Tensor) -> torch.Tensor:
            return node(cube=cube)["cube"]

    example = torch.rand(2, 3, 3, 8, dtype=torch.float32)
    program = torch.export.export(_Probe(), (example,))

    other = torch.rand(2, 3, 3, 8, dtype=torch.float32)
    traced = program.module()(other)
    assert torch.allclose(traced, node(cube=other)["cube"], atol=1e-6)


# ----- PredictionMean --------------------------------------------------


def test_prediction_mean_two_producers():
    node = PredictionMean()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[3.0, 4.0], [5.0, 6.0]])

    out = node(prediction=[a, b])["prediction"]
    expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    assert torch.allclose(out, expected)


def test_prediction_mean_three_producers():
    node = PredictionMean()
    a = torch.full((3, 1), 1.0)
    b = torch.full((3, 1), 2.0)
    c = torch.full((3, 1), 3.0)

    out = node(prediction=[a, b, c])["prediction"]
    assert torch.allclose(out, torch.full((3, 1), 2.0))


def test_prediction_mean_port_contract():
    node = PredictionMean()
    a = torch.rand(4, 2)
    b = torch.rand(4, 2)
    out = node(prediction=[a, b])
    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == spec.dtype
    assert out["prediction"].shape == (4, 2)


def test_prediction_mean_rejects_mismatched_shapes():
    node = PredictionMean()
    a = torch.rand(2, 1)
    b = torch.rand(3, 1)
    with pytest.raises(ValueError):
        node(prediction=[a, b])


def test_prediction_mean_in_small_pipeline_with_two_and_three_producers():
    for n_producers in (2, 3):
        pipe = CuvisPipeline(f"prediction_mean_{n_producers}_test")
        sources = [
            _PredictionSource(value=float(i + 1), name=f"src{i}") for i in range(n_producers)
        ]
        node = PredictionMean(name="mean")
        sink = _PredictionSink(name="sink")
        for src in sources:
            pipe.connect(src.outputs.prediction, node.inputs.prediction)
        pipe.connect(node.outputs.prediction, sink.inputs.prediction)

        out = pipe.forward(batch={})
        seen = out[("sink", "seen")] if ("sink", "seen") in out else None
        if seen is None:
            # Fallback for a differently-shaped forward() return; locate the sink's output.
            for key, value in out.items():
                if key[0] == "sink" and key[1] == "seen":
                    seen = value
                    break
        assert seen is not None
        expected_value = sum(float(i + 1) for i in range(n_producers)) / n_producers
        assert torch.allclose(seen, torch.full_like(seen, expected_value))


def test_prediction_mean_export_smoke():
    node = PredictionMean().eval()

    class _Probe(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return node(prediction=[a, b])["prediction"]

    example_a = torch.rand(2, 1, dtype=torch.float32)
    example_b = torch.rand(2, 1, dtype=torch.float32)
    program = torch.export.export(_Probe(), (example_a, example_b))

    other_a = torch.rand(2, 1, dtype=torch.float32)
    other_b = torch.rand(2, 1, dtype=torch.float32)
    traced = program.module()(other_a, other_b)
    assert torch.allclose(traced, node(prediction=[other_a, other_b])["prediction"], atol=1e-6)


# ----- SpatialMean --------------------------------------------------------


class _MapSource(Node):
    """Minimal source node: no inputs, emits a fixed per-pixel prediction map."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "prediction": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), description="map"),
    }

    def forward(self, **_):
        return {"prediction": torch.rand(2, 3, 3, 1, dtype=torch.float32)}


def test_spatial_mean_golden_reference_unmasked():
    node = SpatialMean()
    pred_map = torch.rand(2, 4, 4, 3, dtype=torch.float32)

    out = node(prediction=pred_map)["prediction"]
    expected = pred_map.mean(dim=(1, 2))
    assert torch.allclose(out, expected, atol=1e-6)


def test_spatial_mean_port_contract():
    node = SpatialMean()
    pred_map = torch.rand(2, 3, 3, 4, dtype=torch.float32)
    out = node(prediction=pred_map)
    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == spec.dtype
    assert out["prediction"].shape == (2, 4)


def test_spatial_mean_reload_smoke(tmp_path):
    pipe = CuvisPipeline("spatial_mean_reload_test")

    src = _MapSource(name="src")
    node = SpatialMean(name="spatial_mean")
    sink = _PredictionSink(name="sink")
    pipe.connect(src.outputs.prediction, node.inputs.prediction)
    pipe.connect(node.outputs.prediction, sink.inputs.prediction)

    yaml_path = tmp_path / "spatial_mean_pipeline.yaml"
    pipe.save_to_file(yaml_path)

    rebuilt = PipelineBuilder().build_from_config(yaml_path)
    assert rebuilt is not None


def test_spatial_mean_export_smoke():
    node = SpatialMean().eval()

    class _Probe(torch.nn.Module):
        def forward(self, pred_map: torch.Tensor) -> torch.Tensor:
            return node(prediction=pred_map)["prediction"]

    example = torch.rand(2, 3, 3, 2, dtype=torch.float32)
    program = torch.export.export(_Probe(), (example,))

    other = torch.rand(2, 3, 3, 2, dtype=torch.float32)
    traced = program.module()(other)
    assert torch.allclose(traced, node(prediction=other)["prediction"], atol=1e-6)
