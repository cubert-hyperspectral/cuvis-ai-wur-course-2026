"""Tests for ``SpectrumRegressor``: parity with the course module, standardisation
round-trip, port contract, freeze/unfreeze, YAML reload, export smoke, and strict
state-dict loading from a course checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.models import SpectrumRegressor
from cuvis_ai_wur_course.upstream import course_models

pytestmark = pytest.mark.unit


# ----- helpers -----------------------------------------------------------

SPECTRUM_LENGTH = 16

X_MEAN = 0.031223
X_STD = 0.049720
Y_MEAN = 37.9531
Y_STD = 13.7968


def _tiny_spectrum_cube(batch: int = 2, channels: int = SPECTRUM_LENGTH) -> torch.Tensor:
    """A BHWC cube with H=W=1, matching one mean spectrum per pixel."""
    torch.manual_seed(0)
    return torch.rand(batch, 1, 1, channels, dtype=torch.float32)


def _course_mlp_backbone(input_dim: int = SPECTRUM_LENGTH) -> course_models.MLP:
    return course_models.MLP(
        input_dim=input_dim, hidden_dims=(32, 16), output_dim=1, normalization=None, dropout=0.0
    )


def _course_resnet1d_backbone(input_dim: int = SPECTRUM_LENGTH):
    return course_models.resnet18(input_channels=1, dim=1, num_classes=1, normalization=None)


def _course_regressor(
    arch: str, input_dim: int = SPECTRUM_LENGTH
) -> course_models.SpectrumRegressor:
    if arch == "mlp":
        backbone = _course_mlp_backbone(input_dim)
        add_channel_dim = False
    elif arch == "resnet1d":
        backbone = _course_resnet1d_backbone(input_dim)
        add_channel_dim = True
    else:
        raise ValueError(arch)
    return course_models.SpectrumRegressor(
        backbone,
        add_channel_dim=add_channel_dim,
        mean=X_MEAN,
        std=X_STD,
        out_mean=Y_MEAN,
        out_std=Y_STD,
    )


def _node(
    arch: str, input_channels: int = SPECTRUM_LENGTH, weights_path: str | None = None
) -> SpectrumRegressor:
    kwargs = {"hidden_dims": (32, 16) if arch == "mlp" else ()}
    return SpectrumRegressor(
        arch=arch,
        input_channels=input_channels,
        weights_path=weights_path,
        **kwargs,
    )


def _copy_state(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=True)


# ----- 1. golden-reference: parity with the course module on the same weights ----


@pytest.mark.parametrize("arch", ["mlp", "resnet1d"])
def test_parity_with_course_module(arch: str):
    torch.manual_seed(42)
    course = _course_regressor(arch)
    course.eval()

    node = _node(arch)
    node.eval()
    _copy_state(node.backbone, course.backbone)

    cube = _tiny_spectrum_cube()
    spectrum = cube.reshape(cube.shape[0], -1)  # (B, C), matches the course's (B, L) input

    with torch.no_grad():
        expected = course(spectrum).reshape(-1)  # course output is already de-standardised
        out = node(cube=cube)["prediction"].reshape(-1)

    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


# ----- 2. standardisation round-trip --------------------------------------


def test_standardization_round_trip_identity_backbone():
    """A backbone that returns its (standardised) input unchanged yields
    y_mean + y_std * standardised_x, which is x itself de-standardised."""

    class _Identity(torch.nn.Module):
        def forward(self, x):
            return x

    node = _node("mlp")
    node.backbone = _Identity()  # swap in an identity backbone for this check

    cube = _tiny_spectrum_cube(batch=3, channels=1)  # single output channel to match T=1
    with torch.no_grad():
        out = node(cube=cube)["prediction"]

    standardized = (cube - node.x_mean) / node.x_std
    expected = node.y_mean + node.y_std * standardized

    assert torch.allclose(out, expected, atol=1e-6)


# ----- 3. port contract ----------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "resnet1d"])
def test_port_contract(arch: str):
    node = _node(arch).eval()
    cube = _tiny_spectrum_cube(batch=2)

    out = node(cube=cube)

    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out, f"missing output port: {port_name}"
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == spec.dtype
        assert tensor.shape == (cube.shape[0], cube.shape[1], cube.shape[2], 1)


# ----- 4. freeze / unfreeze -------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "resnet1d"])
def test_freeze_unfreeze_cycle(arch: str):
    node = _node(arch)

    before = [p.detach().clone() for p in node.backbone.parameters()]
    assert all(not p.requires_grad for p in node.backbone.parameters())

    node.unfreeze()
    after_unfreeze = list(node.backbone.parameters())
    assert len(after_unfreeze) > 0
    assert all(p.requires_grad for p in after_unfreeze)
    for b, a in zip(before, after_unfreeze, strict=True):
        assert torch.allclose(b, a.detach())

    node.freeze()
    after_freeze = list(node.backbone.parameters())
    assert all(not p.requires_grad for p in after_freeze)
    for b, a in zip(before, after_freeze, strict=True):
        assert torch.allclose(b, a.detach())


# ----- 5. reload from YAML with weights_path None --------------------------


class _Src(Node):
    """Minimal source node: no inputs, one cube out."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
    }

    def forward(self, **_):
        return {"cube": _tiny_spectrum_cube()}


def test_reload_from_yaml(tmp_path: Path):
    pipe = CuvisPipeline("spectrum_regressor_reload_test")
    src = _Src(name="src")
    node = SpectrumRegressor(
        name="regressor",
        arch="mlp",
        input_channels=SPECTRUM_LENGTH,
        weights_path=None,
        hidden_dims=(32, 16),
    )
    pipe.connect(src.outputs.cube, node.inputs.cube)

    config_path = tmp_path / "pipeline.yaml"
    pipe.save_to_file(config_path)
    weights_path = config_path.with_suffix(".pt")

    rebuilt = CuvisPipeline.load_pipeline(config_path, weights_path=weights_path)
    rebuilt_node = next(n for n in rebuilt.nodes if n.name.startswith("regressor"))

    cube = _tiny_spectrum_cube()
    original_node = next(n for n in pipe.nodes if n.name.startswith("regressor"))
    original_node.eval()
    rebuilt_node.eval()
    with torch.no_grad():
        expected = original_node(cube=cube)["prediction"]
        out = rebuilt_node(cube=cube)["prediction"]
    assert torch.allclose(out, expected, atol=1e-6)


# ----- 6. export smoke ------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    def __init__(self, node: SpectrumRegressor):
        super().__init__()
        adapter = getattr(node, "export_module", lambda: None)()
        self.module = adapter if adapter is not None else node

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        return self.module(cube=cube)["prediction"]


@pytest.mark.parametrize("arch", ["mlp", "resnet1d"])
def test_export_smoke(arch: str):
    node = _node(arch).eval()
    example = _tiny_spectrum_cube()
    program = torch.export.export(_ExportProbe(node), (example,))

    other = _tiny_spectrum_cube() * 2
    traced = program.module()(other)
    assert torch.allclose(traced, node(cube=other)["prediction"], atol=1e-5)


# ----- 7. strict load from a course checkpoint ------------------------------


@pytest.mark.parametrize("arch", ["mlp", "resnet1d"])
def test_strict_load_from_course_checkpoint(arch: str, tmp_path: Path):
    torch.manual_seed(7)
    course = _course_regressor(arch)
    weights_path = tmp_path / "best_weights.pt"
    torch.save(course.state_dict(), weights_path)

    kwargs = {"hidden_dims": (32, 16)} if arch == "mlp" else {}
    node = SpectrumRegressor(
        arch=arch,
        input_channels=SPECTRUM_LENGTH,
        weights_path=str(weights_path),
        x_mean=X_MEAN,
        x_std=X_STD,
        y_mean=Y_MEAN,
        y_std=Y_STD,
        **kwargs,
    )
    node.eval()

    cube = _tiny_spectrum_cube()
    spectrum = cube.reshape(cube.shape[0], -1)
    with torch.no_grad():
        expected = course.eval()(spectrum).reshape(-1)
        out = node(cube=cube)["prediction"].reshape(-1)
    assert torch.allclose(out, expected, atol=1e-5)
