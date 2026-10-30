"""Tests for cuvis_ai_wur_course.node.pretreatment: SpectralBinning, SavitzkyGolayInterp."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec
from scipy.signal import savgol_filter

from cuvis_ai_wur_course.node.pretreatment import SavitzkyGolayInterp, SpectralBinning

pytestmark = pytest.mark.unit


# ----- shared minimal source/sink for reload smoke tests ---------------


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


# ----- SpectralBinning ---------------------------------------------------


def test_spectral_binning_golden_reference_pairwise():
    node = SpectralBinning(input_channels=6, bin_size=2)
    cube = torch.rand(2, 3, 3, 6, dtype=torch.float32)

    expected = cube.reshape(2, 3, 3, 3, 2).mean(dim=-1)

    out = node(cube=cube)["cube"]
    assert torch.allclose(out, expected, atol=1e-6)


def test_spectral_binning_drops_trailing_remainder():
    # 7 channels, bin_size 2: last channel is dropped, 3 bins remain.
    node = SpectralBinning(input_channels=7, bin_size=2)
    cube = torch.arange(7, dtype=torch.float32).view(1, 1, 1, 7)

    out = node(cube=cube)["cube"]
    expected = torch.tensor([[[[0.5, 2.5, 4.5]]]])
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.shape[-1] == 3


def test_spectral_binning_124_to_62_matches_course_shape():
    node = SpectralBinning(input_channels=124, bin_size=2)
    cube = torch.rand(1, 2, 2, 124, dtype=torch.float32)
    out = node(cube=cube)["cube"]
    assert out.shape == (1, 2, 2, 62)


def test_spectral_binning_bins_wavelengths_alongside_cube():
    node = SpectralBinning(input_channels=6, bin_size=2)
    cube = torch.rand(1, 1, 1, 6, dtype=torch.float32)
    wavelengths = np.array([700, 710, 720, 730, 740, 750], dtype=np.int32)

    out = node(cube=cube, wavelengths=wavelengths)
    assert "wavelengths" in out
    expected_wl = np.array([705, 725, 745], dtype=np.int32)
    np.testing.assert_array_equal(out["wavelengths"], expected_wl)


def test_spectral_binning_port_contract():
    node = SpectralBinning(input_channels=6, bin_size=2)
    cube = torch.rand(2, 4, 4, 6, dtype=torch.float32)

    out = node(cube=cube)
    for port_name, spec in node.OUTPUT_SPECS.items():
        if port_name not in out:
            assert spec.optional
            continue
        tensor = out[port_name]
        if isinstance(tensor, torch.Tensor):
            assert tensor.dtype == spec.dtype
    assert out["cube"].shape == (2, 4, 4, 3)


def test_spectral_binning_reload_smoke(tmp_path):
    pipe = CuvisPipeline("binning_reload_test")
    src = _CubeSource(channels=6, name="src")
    node = SpectralBinning(input_channels=6, bin_size=2, name="binning")
    sink = _CubeSink(name="sink")
    pipe.connect(src.outputs.cube, node.inputs.cube)
    pipe.connect(node.outputs.cube, sink.inputs.cube)

    yaml_path = tmp_path / "binning_pipeline.yaml"
    pipe.save_to_file(yaml_path)

    rebuilt = PipelineBuilder().build_from_config(yaml_path)
    assert rebuilt is not None


def test_spectral_binning_export_smoke():
    node = SpectralBinning(input_channels=6, bin_size=2).eval()

    class _Probe(torch.nn.Module):
        def forward(self, cube: torch.Tensor) -> torch.Tensor:
            return node(cube=cube)["cube"]

    example = torch.rand(2, 3, 3, 6, dtype=torch.float32)
    program = torch.export.export(_Probe(), (example,))

    other = torch.rand(2, 3, 3, 6, dtype=torch.float32)
    traced = program.module()(other)
    assert torch.allclose(traced, node(cube=other)["cube"], atol=1e-6)


# ----- SavitzkyGolayInterp -----------------------------------------------


def test_savitzky_golay_interp_parity_with_scipy_including_edges():
    num_channels = 20
    node = SavitzkyGolayInterp(
        num_channels=num_channels, window_length=7, polyorder=2, deriv=2, delta=1.0
    )
    torch.manual_seed(1)
    spectra = torch.randn(5, num_channels, dtype=torch.float64)

    expected = savgol_filter(
        spectra.numpy(),
        window_length=7,
        polyorder=2,
        deriv=2,
        delta=1.0,
        axis=-1,
        mode="interp",
    )

    out = node(cube=spectra.to(torch.float32).view(5, 1, 1, num_channels))["cube"]
    out_flat = out.view(5, num_channels).detach().numpy()
    assert np.allclose(out_flat, expected, atol=1e-6), np.max(np.abs(out_flat - expected))


def test_savitzky_golay_interp_edge_channels_match_scipy():
    """Specifically checks the first and last channels, where interp edge handling differs."""
    num_channels = 16
    node = SavitzkyGolayInterp(num_channels=num_channels, window_length=7, polyorder=2, deriv=2)
    torch.manual_seed(2)
    spectra = torch.randn(3, num_channels, dtype=torch.float32)

    expected = savgol_filter(
        spectra.numpy(), window_length=7, polyorder=2, deriv=2, axis=-1, mode="interp"
    )

    out = node(cube=spectra.view(3, 1, 1, num_channels))["cube"].view(3, num_channels)
    out_np = out.detach().numpy()

    # Edge channels (first 3 and last 3, half the window) are where mode="interp"
    # diverges most from a padded convolution.
    edge_idx = [0, 1, 2, num_channels - 3, num_channels - 2, num_channels - 1]
    assert np.allclose(out_np[:, edge_idx], expected[:, edge_idx], atol=1e-6)


def test_builtin_savitzky_golay_differs_at_edges_from_scipy_interp():
    """Documents the gap: the builtin SavitzkyGolay (padded convolution) does not
    reproduce scipy's mode="interp" edges, which is why this node exists."""
    from cuvis_ai.node.pretreatments.savitzky_golay import SavitzkyGolay

    num_channels = 16
    window_length, polyorder, deriv = 7, 2, 2
    torch.manual_seed(3)
    spectra = torch.randn(1, num_channels, dtype=torch.float32)

    expected_interp = savgol_filter(
        spectra.numpy(),
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        axis=-1,
        mode="interp",
    )

    builtin = SavitzkyGolay(
        window_length=window_length, polyorder=polyorder, deriv=deriv, mode="nearest"
    )
    builtin_out = builtin(cube=spectra.view(1, 1, 1, num_channels))["cube"]
    builtin_out_np = builtin_out.view(1, num_channels).detach().numpy()

    # The two outputs must NOT match at the edge channels: this is the gap that
    # SavitzkyGolayInterp closes.
    first_last = [0, num_channels - 1]
    assert not np.allclose(builtin_out_np[:, first_last], expected_interp[:, first_last], atol=1e-6)


def test_savitzky_golay_interp_port_contract():
    num_channels = 10
    node = SavitzkyGolayInterp(num_channels=num_channels)
    cube = torch.rand(2, 4, 4, num_channels, dtype=torch.float32)

    out = node(cube=cube)
    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == spec.dtype
    assert out["cube"].shape == cube.shape


def test_savitzky_golay_interp_reload_smoke(tmp_path):
    num_channels = 10
    pipe = CuvisPipeline("sg_interp_reload_test")
    src = _CubeSource(channels=num_channels, name="src")
    node = SavitzkyGolayInterp(num_channels=num_channels, name="sg_interp")
    sink = _CubeSink(name="sink")
    pipe.connect(src.outputs.cube, node.inputs.cube)
    pipe.connect(node.outputs.cube, sink.inputs.cube)

    yaml_path = tmp_path / "sg_interp_pipeline.yaml"
    pipe.save_to_file(yaml_path)

    rebuilt = PipelineBuilder().build_from_config(yaml_path)
    assert rebuilt is not None


def test_savitzky_golay_interp_export_smoke():
    num_channels = 10
    node = SavitzkyGolayInterp(num_channels=num_channels).eval()

    class _Probe(torch.nn.Module):
        def forward(self, cube: torch.Tensor) -> torch.Tensor:
            return node(cube=cube)["cube"]

    example = torch.rand(2, 3, 3, num_channels, dtype=torch.float32)
    program = torch.export.export(_Probe(), (example,))

    other = torch.rand(2, 3, 3, num_channels, dtype=torch.float32) * 2
    traced = program.module()(other)
    assert torch.allclose(traced, node(cube=other)["cube"], atol=1e-6)


def test_savitzky_golay_interp_rejects_even_window():
    with pytest.raises(ValueError):
        SavitzkyGolayInterp(num_channels=10, window_length=6)


def test_savitzky_golay_interp_rejects_polyorder_too_high():
    with pytest.raises(ValueError):
        SavitzkyGolayInterp(num_channels=10, window_length=5, polyorder=5)
