"""Regression loss node for the WUR course pipelines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai.node.losses import LossNode
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class RegressionMSELoss(LossNode):
    """Mean squared error between a regression prediction and its target.

    Inherits ``cuvis_ai.node.losses.LossNode``, so ``EXECUTION_STAGES`` (train, val,
    test) and the ``LOSS`` category come from the base class rather than being
    redeclared here. Plain ``torch.nn.functional.mse_loss`` under the ``reduction``
    hparam; no class-imbalance or per-target weighting, unlike
    ``cuvis_ai.node.losses.MSEReconstructionLoss`` which targets dense reconstructions
    of arbitrary rank rather than a ``[B, T]`` regression head.

    Parameters
    ----------
    reduction : str, optional
        Reduction passed to ``F.mse_loss``: ``"mean"``, ``"sum"`` or ``"none"``
        (default: ``"mean"``). ``"none"`` returns the per-element squared error
        rather than a scalar, which breaks the ``loss`` port's contract of a single
        scalar; it is accepted (matching the PyTorch reduction vocabulary) but only
        ``"mean"`` and ``"sum"`` are meaningful choices for the trainer.
    """

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
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Scalar MSE loss value."),
    }

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.reduction = reduction
        super().__init__(reduction=reduction, **kwargs)

    def forward(self, prediction: Tensor, target: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute the MSE loss between prediction and target.

        Parameters
        ----------
        prediction : Tensor
            Predicted regression values [B, T].
        target : Tensor
            Ground-truth regression values [B, T].

        Returns
        -------
        dict[str, Tensor]
            Dictionary with a single key ``"loss"``.
        """
        loss = F.mse_loss(prediction, target, reduction=self.reduction)
        return {"loss": loss}


__all__ = ["RegressionMSELoss"]
