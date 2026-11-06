"""Tests for cuvis_ai_wur_course.node.losses.RegressionMSELoss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from cuvis_ai.node.losses import LossNode
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.losses import RegressionMSELoss

pytestmark = pytest.mark.unit


class _RegressionSource(Node):
    """Minimal source node emitting fixed prediction/target tensors, for reload smoke."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TABULAR})

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "prediction": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "target": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {
            "prediction": torch.rand(4, 1, 1, 2, dtype=torch.float32),
            "target": torch.rand(4, 2, dtype=torch.float32),
        }


# ----- helpers -----------------------------------------------------------


def _tensors(batch: int = 5, targets: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return (
        torch.randn(batch, 1, 1, targets, dtype=torch.float32),
        torch.randn(batch, targets, dtype=torch.float32),
    )


# ----- 1. golden-reference -------------------------------------------------


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_golden_reference_matches_torch_mse(reduction: str) -> None:
    prediction, target = _tensors()
    node = RegressionMSELoss(reduction=reduction)

    out = node(prediction=prediction, target=target)
    expected = F.mse_loss(prediction.mean(dim=(1, 2)), target, reduction=reduction)

    assert torch.allclose(out["loss"], expected, atol=1e-6)


def test_golden_reference_hand_built_single_target() -> None:
    prediction = torch.tensor([[1.0], [2.0], [3.0]])[:, None, None, :]
    target = torch.tensor([[1.5], [1.5], [2.0]])
    node = RegressionMSELoss()

    out = node(prediction=prediction, target=target)
    expected = ((prediction.mean(dim=(1, 2)) - target) ** 2).mean()

    assert torch.allclose(out["loss"], expected, atol=1e-6)


def test_invalid_reduction_rejected() -> None:
    with pytest.raises(ValueError):
        RegressionMSELoss(reduction="bogus")


# ----- 2. port contract -----------------------------------------------------


def test_port_contract() -> None:
    node = RegressionMSELoss()
    prediction, target = _tensors(batch=4, targets=2)

    out = node(prediction=prediction, target=target)

    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out, f"missing output port: {port_name}"
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        if isinstance(spec, PortSpec):
            assert tensor.dtype == spec.dtype
        assert tensor.shape == ()


# ----- base class and category checks ---------------------------------------


def test_inherits_loss_node_stages_and_category() -> None:
    node = RegressionMSELoss()
    assert isinstance(node, LossNode)
    assert node.execution_stages == {
        ExecutionStage.TRAIN,
        ExecutionStage.VAL,
        ExecutionStage.TEST,
    }
    assert node.get_category() == NodeCategory.LOSS


# ----- 3. reload smoke -------------------------------------------------------


def test_reload_smoke(tmp_path) -> None:
    """A pipeline carrying the node saves and reloads with the same forward output."""
    pipeline = CuvisPipeline("regression_mse_loss_reload")
    source = _RegressionSource(name="source")
    node = RegressionMSELoss(reduction="mean", name="loss")
    pipeline.connect(
        (source.outputs.prediction, node.inputs.prediction),
        (source.outputs.target, node.inputs.target),
    )

    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path, validate_nodes=False)

    rebuilt = PipelineBuilder().build_from_config(config_path)
    rebuilt_node = next(n for n in rebuilt.nodes if isinstance(n, RegressionMSELoss))
    assert rebuilt_node.reduction == "mean"

    prediction = torch.rand(3, 1, 1, 2, dtype=torch.float32)
    target = torch.rand(3, 2, dtype=torch.float32)
    out = rebuilt_node(prediction=prediction, target=target)
    expected = F.mse_loss(prediction.mean(dim=(1, 2)), target)
    assert torch.allclose(out["loss"], expected, atol=1e-6)


# ----- 4. export smoke -------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    """The node, or its export_module() adapter, with tensors in and a tensor out."""

    def __init__(self, node: RegressionMSELoss) -> None:
        super().__init__()
        adapter = getattr(node, "export_module", lambda: None)()
        self.module = adapter if adapter is not None else node

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.module(prediction=prediction, target=target)["loss"]


def test_export_smoke() -> None:
    node = RegressionMSELoss().eval()
    prediction, target = _tensors(batch=4, targets=2)
    program = torch.export.export(_ExportProbe(node), (prediction, target))

    other_prediction, other_target = _tensors(batch=4, targets=2)
    other_prediction = other_prediction * 2
    traced = program.module()(other_prediction, other_target)
    expected = node(prediction=other_prediction, target=other_target)["loss"]
    assert torch.allclose(traced, expected, atol=1e-6)


# ----- prediction maps ---------------------------------------------------------


def test_prediction_port_takes_a_map() -> None:
    assert RegressionMSELoss.INPUT_SPECS["prediction"].shape == (-1, -1, -1, -1)


def test_a_one_by_one_map_gives_the_loss_of_its_squeezed_prediction_exactly() -> None:
    torch.manual_seed(4)
    prediction = torch.randn(5, 2)
    target = torch.randn(5, 2)
    out = RegressionMSELoss()(prediction=prediction[:, None, None, :], target=target)
    assert torch.equal(out["loss"], F.mse_loss(prediction, target))


def test_a_map_is_averaged_per_sample_before_the_loss() -> None:
    torch.manual_seed(5)
    prediction_map = torch.randn(3, 4, 2, 2)
    target = torch.randn(3, 2)
    out = RegressionMSELoss()(prediction=prediction_map, target=target)
    expected = F.mse_loss(prediction_map.mean(dim=(1, 2)), target)
    assert torch.allclose(out["loss"], expected, atol=1e-6)
