"""Tests for cuvis_ai_wur_course.node.metrics.RegressionMetrics."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cuvis_ai_wur_course.node.metrics import RegressionMetrics

pytestmark = pytest.mark.unit


class _RegressionSource(Node):
    """Minimal source node emitting fixed prediction/target tensors, for reload smoke."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TABULAR})

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "prediction": PortSpec(dtype=torch.float32, shape=(-1, -1)),
        "target": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {
            "prediction": torch.rand(3, 1, dtype=torch.float32),
            "target": torch.rand(3, 1, dtype=torch.float32),
        }


# ----- helpers ---------------------------------------------------------


def _batches(
    n_batches: int, batch_size: int, n_targets: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A handful of (prediction, target) batches with fixed shapes."""
    return [
        (
            torch.rand(batch_size, n_targets, dtype=torch.float32),
            torch.rand(batch_size, n_targets, dtype=torch.float32),
        )
        for _ in range(n_batches)
    ]


# ----- 1. golden-reference ----------------------------------------------


def test_golden_reference_pooled_over_several_batches() -> None:
    """RMSE, R2 and MAE pooled across batches match sklearn on the concatenated data."""
    node = RegressionMetrics()
    node.reset()
    batches = _batches(n_batches=4, batch_size=6, n_targets=3)

    out = None
    for prediction, target in batches:
        out = node(prediction=prediction, target=target)
    assert out is not None

    all_pred = torch.cat([p for p, _ in batches]).numpy()
    all_target = torch.cat([t for _, t in batches]).numpy()

    expected_rmse = mean_squared_error(all_target, all_pred) ** 0.5
    expected_r2 = r2_score(all_target, all_pred)
    expected_mae = mean_absolute_error(all_target, all_pred)

    assert out["rmse"].item() == pytest.approx(expected_rmse, abs=1e-5)
    assert out["r2"].item() == pytest.approx(expected_r2, abs=1e-5)
    assert out["mae"].item() == pytest.approx(expected_mae, abs=1e-5)


def test_golden_reference_single_target_hand_values() -> None:
    """A tiny hand-built single-target batch matches a closed-form computation."""
    node = RegressionMetrics()
    node.reset()
    prediction = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.5], [2.0], [2.5]])

    out = node(prediction=prediction, target=target)

    errors = (prediction - target).squeeze(-1)
    expected_mae = errors.abs().mean().item()
    expected_rmse = (errors**2).mean().sqrt().item()

    assert out["mae"].item() == pytest.approx(expected_mae, abs=1e-6)
    assert out["rmse"].item() == pytest.approx(expected_rmse, abs=1e-6)


def test_reset_clears_pooled_state() -> None:
    """reset() drops accumulated state so a fresh pass starts from zero batches seen."""
    node = RegressionMetrics()
    node.reset()
    node(prediction=torch.rand(4, 2), target=torch.rand(4, 2))

    node.reset()
    prediction = torch.rand(4, 2)
    target = torch.rand(4, 2)
    out = node(prediction=prediction, target=target)

    expected_mae = (prediction - target).abs().mean().item()
    assert out["mae"].item() == pytest.approx(expected_mae, abs=1e-6)


# ----- 2. port contract --------------------------------------------------


def test_port_contract() -> None:
    node = RegressionMetrics()
    node.reset()
    prediction = torch.rand(5, 2, dtype=torch.float32)
    target = torch.rand(5, 2, dtype=torch.float32)

    out = node(prediction=prediction, target=target)

    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out, f"missing output port: {port_name}"
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        if isinstance(spec, PortSpec):
            assert tensor.dtype == spec.dtype, (
                f"{port_name}: expected {spec.dtype}, got {tensor.dtype}"
            )
        assert tensor.shape == (), f"{port_name}: expected a scalar, got {tensor.shape}"


# ----- 3. reload smoke ----------------------------------------------------


def test_reload_smoke(tmp_path) -> None:
    """A pipeline carrying the node saves and reloads without error."""
    pipeline = CuvisPipeline("regression_metrics_reload")
    source = _RegressionSource(name="source")
    node = RegressionMetrics(name="metrics")
    pipeline.connect(
        (source.outputs.prediction, node.inputs.prediction),
        (source.outputs.target, node.inputs.target),
    )

    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path, validate_nodes=False)

    rebuilt = PipelineBuilder().build_from_config(config_path)
    rebuilt_node = next(n for n in rebuilt.nodes if isinstance(n, RegressionMetrics))
    rebuilt_node.reset()

    prediction = torch.rand(3, 1, dtype=torch.float32)
    target = torch.rand(3, 1, dtype=torch.float32)
    out = rebuilt_node(prediction=prediction, target=target)
    assert "rmse" in out and "r2" in out and "mae" in out


# ----- 4. export smoke: declared non-exportable --------------------------
#
# RegressionMetrics wraps stateful torchmetrics objects that accumulate across
# calls (update() mutates internal counters); torch.export traces a single
# forward() call and cannot capture that running accumulation, so this node
# is non-exportable by design (see the class docstring) and carries no
# NodeTag.ONNX. No export smoke test is written for it.


def test_pooled_metric_contract_exposes_live_objects_after_a_batch() -> None:
    node = RegressionMetrics()
    assert node.POOLED_METRIC_NAMES == frozenset({"rmse", "r2", "mae"})
    assert node.pooled_metrics() == {}
    node(prediction=torch.tensor([[1.0], [2.0]]), target=torch.tensor([[1.5], [2.5]]))
    pooled = node.pooled_metrics()
    assert set(pooled) == node.POOLED_METRIC_NAMES
    assert pooled["rmse"] is node.rmse_metric
    assert pooled["rmse"].compute().item() == pytest.approx(0.5)
    node.reset()
    assert node.pooled_metrics() == {}


def test_compute_returns_pooled_floats_after_batches() -> None:
    node = RegressionMetrics()
    assert node.compute() == {}
    node(prediction=torch.tensor([[1.0], [2.0]]), target=torch.tensor([[1.5], [2.5]]))
    node(prediction=torch.tensor([[3.0]]), target=torch.tensor([[3.5]]))
    values = node.compute()
    assert set(values) == {"rmse", "r2", "mae"}
    assert values["rmse"] == pytest.approx(0.5)
    assert values["mae"] == pytest.approx(0.5)
