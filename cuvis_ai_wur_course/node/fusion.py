"""Channel and prediction fusion nodes for the course regression pipelines.

``ChannelSlice`` splits a stacked feature block into its spatial and spectral
parts (the 812-wide pork feature vector), and ``PredictionMean`` averages several
prediction maps into one (the course's average-fusion row).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class ChannelSlice(Node):
    """Slice a contiguous range of channels out of a hyperspectral cube.

    Takes channels ``[start, stop)`` of the last axis, e.g. splitting the
    course's 812-wide pork feature block into its spatial (``0:512``) and
    spectral (``512:812``) halves after one shared autoscale, so no concat
    node is needed to put the block together beforehand. An optional
    ``wavelengths`` port is sliced the same way, for consistency with
    ``SpectralBinning`` when both feed off the same channel axis.
    `cuvis_ai.node.channel_selector` was checked for a reusable slice: its
    selectors are per-channel gates or weighted mixers (`ChannelSelectorBase`
    subclasses apply ``weight[c] * input[c]``, preserving the channel count),
    not a contiguous range extraction that changes the channel count, so none
    of them fit this operation.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Optional per-channel wavelengths [C], sliced the same way as the cube.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Sliced cube [B, H, W, stop - start].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Sliced wavelengths [stop - start]; only present when the "
            "wavelengths input is connected.",
            optional=True,
        ),
    }

    def __init__(self, start: int, stop: int, **kwargs) -> None:
        """Create the node.

        Parameters
        ----------
        start : int
            First channel index kept (inclusive).
        stop : int
            Last channel index kept (exclusive); must be greater than ``start``.
        """
        if stop <= start:
            raise ValueError(f"stop ({stop}) must be greater than start ({start})")
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        super().__init__(start=start, stop=stop, **kwargs)
        self.start = int(start)
        self.stop = int(stop)

    def forward(self, cube: torch.Tensor, wavelengths=None, **_) -> dict[str, torch.Tensor]:
        """Return the ``[start, stop)`` channel slice of the cube (and wavelengths)."""
        out: dict[str, torch.Tensor] = {"cube": cube[..., self.start : self.stop]}
        if wavelengths is not None:
            # Wavelengths are a np.int32 port; plain slicing works whether the
            # runtime value is a numpy array or array-like sequence.
            out["wavelengths"] = np.asarray(wavelengths)[self.start : self.stop]
        return out


def _stack_predictions(values: Sequence[Tensor] | Tensor, node: str) -> Tensor:
    """Stack the inbound tensors of a variadic prediction port along a new first axis."""
    items = list(values) if isinstance(values, (list, tuple)) else [values]
    if not items:
        raise ValueError(f"{node}: no predictions connected.")
    shape = items[0].shape
    for i, item in enumerate(items[1:], start=1):
        if item.shape != shape:
            raise ValueError(
                f"{node}: prediction {i} has shape {tuple(item.shape)}, expected {tuple(shape)}."
            )
    return torch.stack(items, dim=0)


class PredictionMean(Node):
    """Average several prediction maps into one, fan-in on one variadic port.

    One per-target prediction map ``[B, H, W, T]`` per inbound connection
    (connection order is not significant: the result is their plain elementwise
    mean), the course's average-fusion row over the spatial, spectral and stacked
    PLS predictions. The fused map goes on to ``RegressionMetrics``, which averages
    it per sample. ``cuvis_ai.node.fusion.ScoreMapFusion`` (mode ``mean``) computes
    the same mean, but its port is named ``scores``, which reads wrong for a fat
    prediction in the course notebook, so this node keeps the regression name.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.REGRESSION, NodeTag.DIFFERENTIABLE, NodeTag.TORCH})

    INPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            variadic=True,
            description="Per-target prediction maps [B, H, W, T] of one shape, one per "
            "inbound connection (fan-in).",
        ),
    }

    OUTPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Elementwise mean prediction map [B, H, W, T] across the inbound "
            "connections.",
        ),
    }

    def forward(self, prediction: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the elementwise mean of the inbound prediction maps."""
        stack = _stack_predictions(prediction, "PredictionMean")  # [N, B, H, W, T]
        return {"prediction": stack.mean(dim=0)}
