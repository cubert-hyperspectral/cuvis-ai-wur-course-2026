"""Regression metric node for the WUR course pipelines."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics import Metric as TorchMetric


class RegressionMetrics(Node):
    """Pooled RMSE, R2 and MAE for a scalar or multi-target regression head.

    Mirrors the pooling contract of ``cuvis_ai.node.metrics.AnomalyDetectionMetrics``:
    the three wrapped ``torchmetrics`` objects accumulate with ``update()`` over every
    batch of a validation or test pass and are read with a single ``compute()`` at the
    end of the pass, not averaged per batch. ``reset()`` clears all three at once, which
    the caller runs at the start of each pass (the trainer does this at the
    ``(stage, epoch)`` boundary; a notebook calling the node directly resets it the same
    way before iterating a split).

    ``rmse`` comes from ``torchmetrics.MeanSquaredError(squared=False)``, ``r2`` from
    ``torchmetrics.R2Score(multioutput="uniform_average")`` (equal weight per target,
    matching ``sklearn.metrics.r2_score`` on multi-column input), and ``mae`` from
    ``torchmetrics.MeanAbsoluteError()``. All three are read directly off the live
    metric objects via ``.compute()``, the pattern the notebook skill expects, rather
    than emitted as a ``metrics`` output port of ``Metric`` records.

    Stateful across calls (the three metrics accumulate), so this node is declared
    non-exportable: ``torch.export`` traces one ``forward`` call and cannot capture the
    running accumulation ``update()`` performs across calls.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.REGRESSION})

    INPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Predicted regression values [B, T].",
        ),
        "target": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Ground-truth regression values [B, T].",
        ),
    }

    OUTPUT_SPECS = {
        "rmse": PortSpec(
            dtype=torch.float32, shape=(), description="Pooled root-mean-squared error so far."
        ),
        "r2": PortSpec(dtype=torch.float32, shape=(), description="Pooled R2 score so far."),
        "mae": PortSpec(
            dtype=torch.float32, shape=(), description="Pooled mean absolute error so far."
        ),
    }

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    # The trainer skips these names in per-batch logging and logs the live torchmetrics
    # objects from pooled_metrics() with on_epoch=True instead, so Lightning computes one
    # pooled value per epoch and resets it, exact and batch-size-invariant.
    POOLED_METRIC_NAMES: ClassVar[frozenset[str]] = frozenset({"rmse", "r2", "mae"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._seen_batch = False
        self.rmse_metric = MeanSquaredError(squared=False)
        self.r2_metric = R2Score(multioutput="uniform_average")
        self.mae_metric = MeanAbsoluteError()

    def reset(self) -> None:
        """Clear the accumulated state of all three wrapped metrics."""
        self.rmse_metric.reset()
        self.r2_metric.reset()
        self.mae_metric.reset()
        self._seen_batch = False

    def compute(self) -> dict[str, float]:
        """The pooled RMSE, R2 and MAE over everything seen since the last reset, as floats."""
        if not self._seen_batch:
            return {}
        return {
            "rmse": float(self.rmse_metric.compute()),
            "r2": float(self.r2_metric.compute()),
            "mae": float(self.mae_metric.compute()),
        }

    def pooled_metrics(self) -> dict[str, TorchMetric]:
        """Live torchmetrics objects for the epoch-pooled metrics, keyed by name.

        Empty until the node has seen a batch, as in the builtin metric nodes, so a pass
        that never reached this node logs nothing instead of computing on empty state.
        """
        if not self._seen_batch:
            return {}
        return {"rmse": self.rmse_metric, "r2": self.r2_metric, "mae": self.mae_metric}

    def forward(self, prediction: Tensor, target: Tensor, **_: Any) -> dict[str, Tensor]:
        """Update the pooled metrics with one batch and return the running values.

        Parameters
        ----------
        prediction : Tensor
            Predicted regression values [B, T].
        target : Tensor
            Ground-truth regression values [B, T].

        Returns
        -------
        dict[str, Tensor]
            The running ``rmse``, ``r2`` and ``mae`` after this batch, each a scalar
            tensor read straight off the corresponding metric's ``.compute()``.
        """
        self._seen_batch = True
        self.rmse_metric.update(prediction, target)
        self.r2_metric.update(prediction, target)
        self.mae_metric.update(prediction, target)
        return {
            "rmse": self.rmse_metric.compute(),
            "r2": self.r2_metric.compute(),
            "mae": self.mae_metric.compute(),
        }


__all__ = ["RegressionMetrics"]
