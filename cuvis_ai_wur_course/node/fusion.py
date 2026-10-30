"""Channel and prediction fusion nodes for the course regression pipelines.

``ChannelSlice`` splits a stacked feature block into its spatial and spectral
parts (the 812-wide pork feature vector), ``PredictionMean`` averages several
per-target predictions into one (the course's average-fusion row), and
``SpatialMean`` collapses a per-pixel regression map into one value per frame
and target, the step that turns a pixel-wise PLS map into the same ``[B, T]``
shape the regression metrics and losses expect.
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
    """Stack the inbound tensors of a variadic prediction port into ``[N, B, T]``."""
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
    """Average several ``[B, T]`` predictions into one, fan-in on one variadic port.

    One per-target prediction per inbound connection (connection order is not
    significant: the result is their plain mean), the course's average-fusion
    row over the spatial, spectral and stacked PLS predictions.
    ``cuvis_ai.node.fusion.ScoreMapFusion`` was checked first: it fuses
    ``[B, H, W, C]`` score maps with several modes (mean, min, max, weighted,
    geometric, soft-min), the right shape for anomaly and segmentation scores
    but not for a ``[B, T]`` per-sample regression prediction; this node
    follows the same variadic fan-in and stacking pattern at the
    regression-prediction shape, with only the one fusion rule the course
    uses.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.REGRESSION, NodeTag.DIFFERENTIABLE, NodeTag.TORCH})

    INPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            variadic=True,
            description="Per-target predictions [B, T] of one shape, one per inbound "
            "connection (fan-in).",
        ),
    }

    OUTPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Mean prediction [B, T] across the inbound connections.",
        ),
    }

    def forward(self, prediction: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the elementwise mean of the inbound predictions."""
        stack = _stack_predictions(prediction, "PredictionMean")  # [N, B, T]
        return {"prediction": stack.mean(dim=0)}


class SpatialMean(Node):
    """Collapse a per-pixel regression map into one value per frame and target.

    Input ``prediction [B, H, W, T]`` is a per-pixel regression map, such as
    the pixel-wise PLS prediction on a 1x1-spectrum frame broadcast over an
    image, or the per-pixel output of a model like ``PLSRegression`` run on a
    full cube; output ``prediction [B, T]`` is the mean over ``H`` and ``W``,
    which is the shape the regression metrics and losses take.

    The node takes no mask on purpose: a batch's ``mask`` key is the ground-truth
    mask the data module always emits (zeros for a frame without one), and the
    pipeline fills an unconnected port of that name from the batch. A
    mask-weighted mean over a map belongs to ``GroupWeightedMean``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.REGRESSION, NodeTag.DIFFERENTIABLE, NodeTag.TORCH})

    INPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel regression map [B, H, W, T].",
        ),
    }

    OUTPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Mean prediction [B, T] over H and W.",
        ),
    }

    def forward(self, prediction: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Mean the per-pixel map over H and W."""
        return {"prediction": prediction.mean(dim=(1, 2))}
