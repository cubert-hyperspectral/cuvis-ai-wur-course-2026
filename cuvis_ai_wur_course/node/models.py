"""Spectral regression model node for the WUR course rebuild.

Wraps the course's spectral regressor (notebook 03,
``end2end_spectral_vs_spatial_models.ipynb``): a global scalar standardisation of
the spectrum, an MLP or 1D ResNet backbone, and a de-standardisation of the
predicted target back to its own units.
"""

from __future__ import annotations

from typing import Literal

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import nn

from cuvis_ai_wur_course.upstream.course_models import MLP, resnet18

# Course defaults: a single scalar mean/std over all training spectra and
# wavelengths, and a single scalar mean/std over the training target values.
_DEFAULT_X_MEAN = 0.031223
_DEFAULT_X_STD = 0.049720
_DEFAULT_Y_MEAN = 37.9531
_DEFAULT_Y_STD = 13.7968

_DEFAULT_MLP_HIDDEN_DIMS = (256, 128)


class SpectrumRegressor(Node):
    """Predict a scalar target per pixel from its spectrum, in the target's own units.

    Standardises the spectrum with a global scalar mean/std, runs it through an
    MLP or 1D ResNet backbone (the course's two spectral architectures), and
    de-standardises the backbone output back to the target's units. The input
    cube is applied per pixel: ``[B, H, W, C]`` is reshaped to ``[N, C]`` for the
    backbone and reshaped back to ``[B, H, W, T]`` on the way out, ``T = 1``.

    Loading from ``weights_path`` follows the course's ``load_trained_model``:
    the file holds a plain ``state_dict`` of the whole wrapper (standardiser,
    backbone, destandardiser) and is loaded with ``strict=True``.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.REGRESSION, NodeTag.LEARNABLE, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "input_channels"),
            description="Hyperspectral cube, BHWC float32; one spectrum per pixel.",
        ),
    }
    OUTPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Predicted target per pixel, de-standardised to its own units. T = 1.",
        ),
    }

    TRAINABLE_BUFFERS = ()

    def __init__(
        self,
        arch: Literal["mlp", "resnet1d"],
        input_channels: int,
        weights_path: str | None = None,
        hidden_dims: tuple[int, ...] = _DEFAULT_MLP_HIDDEN_DIMS,
        dropout: float = 0.0,
        x_mean: float = _DEFAULT_X_MEAN,
        x_std: float = _DEFAULT_X_STD,
        y_mean: float = _DEFAULT_Y_MEAN,
        y_std: float = _DEFAULT_Y_STD,
        **kwargs,
    ) -> None:
        if arch not in ("mlp", "resnet1d"):
            raise ValueError(f"arch must be 'mlp' or 'resnet1d', got {arch!r}")

        super().__init__(
            arch=arch,
            input_channels=input_channels,
            weights_path=weights_path,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            **kwargs,
        )

        self.arch = arch
        self.input_channels = input_channels
        self.weights_path = weights_path
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        self.register_buffer("x_mean", torch.tensor(float(x_mean)))
        self.register_buffer("x_std", torch.tensor(float(x_std)))
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

        if arch == "mlp":
            self.backbone: nn.Module = MLP(
                input_dim=input_channels,
                hidden_dims=self.hidden_dims,
                output_dim=1,
                normalization=None,
                dropout=dropout,
            )
            self._add_channel_dim = False
        else:
            self.backbone = resnet18(input_channels=1, dim=1, num_classes=1, normalization=None)
            self._add_channel_dim = True

        # Backbone weights start frozen like every other learnable node; unfreeze()
        # promotes them for gradient training.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if weights_path is not None:
            self._load_course_checkpoint(weights_path)

    def _load_course_checkpoint(self, weights_path: str) -> None:
        """Load a course ``best_weights.pt`` state dict, following ``load_trained_model``.

        The checkpoint holds the whole course ``SpectrumRegressor`` wrapper
        (``standardizer.*``, ``backbone.*``, ``destandardizer.*``); this method
        remaps those keys onto this node's own buffers and backbone and loads
        with ``strict=True``, so a mismatched architecture or stale checkpoint
        fails loudly instead of silently loading a partial state.
        """
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        remapped = {}
        for key, value in state_dict.items():
            if key == "standardizer.mean":
                remapped["x_mean"] = value
            elif key == "standardizer.std":
                remapped["x_std"] = value
            elif key == "destandardizer.mean":
                remapped["y_mean"] = value
            elif key == "destandardizer.std":
                remapped["y_std"] = value
            elif key.startswith("backbone."):
                remapped[key] = value
            else:
                raise KeyError(
                    f"{type(self).__name__}: unexpected key {key!r} in checkpoint {weights_path!r}"
                )
        self.load_state_dict(remapped, strict=True)

    def freeze(self) -> None:
        """Demote the backbone's parameters back to non-trainable."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        super().freeze()

    def unfreeze(self) -> None:
        """Promote the backbone's parameters to trainable for gradient training."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        super().unfreeze()

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        batch, height, width, channels = cube.shape

        pixels = cube.reshape(-1, channels)  # (N, C), N = B*H*W
        standardized = (pixels - self.x_mean) / self.x_std
        if self._add_channel_dim:
            backbone_input = standardized.unsqueeze(1)  # (N, 1, C)
        else:
            backbone_input = standardized  # (N, C)

        backbone_output = self.backbone(backbone_input)  # (N, 1)
        prediction = backbone_output * self.y_std + self.y_mean  # de-standardise

        return {"prediction": prediction.reshape(batch, height, width, 1)}
