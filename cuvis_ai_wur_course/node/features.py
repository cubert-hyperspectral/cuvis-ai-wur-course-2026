"""ResNet-18 deep feature extraction, the ImageNet way.

Reproduces the deep-feature step of ``using_pretrained_ai.ipynb``: a
torchvision ``resnet18`` with its final classification layer removed gives a
512-dimensional feature vector per image, the same "fingerprint" the course
then feeds to PLS instead of a raw spectrum.

Preprocessing (resize, centre crop, ImageNet normalization) is implemented in
torch inside ``forward`` rather than through ``torchvision.transforms`` on a
PIL image, so the whole node is one differentiable, traceable graph that
never leaves the pixel cube already living on a device as a tensor.

Deviation from the course notebook: the course calls
``torchvision.transforms.Resize`` and ``CenterCrop`` on a PIL image, whose
resize kernel (PIL's own resampling, Lanczos for downscaling by default) is
not bit-identical to ``torch.nn.functional.interpolate``'s bilinear
antialiased kernel used here. For the square 256x256 canvas this node
actually receives from ``LabelCropPad``, ``Resize(256)`` is a no-op (the
shorter edge is already 256), so the two paths agree exactly in the pipeline
this node ships in; the difference only shows up for a non-square or
differently-sized input, and only in the resize step, never in the backbone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torchvision import models

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESIZE_EDGE = 256
_CROP_SIZE = 224


class ResNet18Features(Node):
    """512-dimensional ResNet-18 feature vector for an RGB image.

    Loads a torchvision ``resnet18`` and drops its final fully connected
    layer, exactly as the course does (``torch.nn.Sequential(*list(model
    .children())[:-1])``). ImageNet preprocessing (resize the shorter edge to
    256, centre crop to 224, normalize with the standard ImageNet per-channel
    mean/std) runs inside ``forward`` in torch, so the node stays a single
    traceable graph from a ``[0, 1]`` RGB image to a feature vector.

    The backbone's weights are frozen by default (a pretrained or randomly
    initialized feature extractor, used as a fixed "fingerprint" generator);
    call :meth:`unfreeze` to fine-tune it end-to-end.

    Parameters
    ----------
    weights : str or None
        ``"imagenet"`` loads ``ResNet18_Weights.DEFAULT`` (downloads on first
        use); ``None`` starts from randomly initialized weights and performs
        no download, which is what every test in this module uses. Default
        ``None``.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.IMAGE, NodeTag.RGB, NodeTag.EMBEDDING, NodeTag.TORCH, NodeTag.LEARNABLE}
    )

    INPUT_SPECS = {
        "image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] float32 in [0, 1].",
        ),
    }
    OUTPUT_SPECS = {
        "features": PortSpec(
            dtype=torch.float32,
            shape=(-1, 512),
            description="[B, 512] ResNet-18 feature vector (penultimate layer, pre-fc).",
        ),
    }

    def __init__(self, weights: str | None = None, **kwargs: Any) -> None:
        if weights not in (None, "imagenet"):
            raise ValueError(f"weights must be 'imagenet' or None, got {weights!r}")

        self.weights = weights

        super().__init__(weights=weights, **kwargs)

        backbone_weights = models.ResNet18_Weights.DEFAULT if weights == "imagenet" else None
        full_model = models.resnet18(weights=backbone_weights)
        full_model.fc = torch.nn.Identity()
        self.backbone = full_model

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

        self.freeze()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """BHWC [0, 1] image -> BCHW ImageNet-normalized [224, 224] tensor.

        Resize the shorter edge to ``_RESIZE_EDGE`` with bilinear
        interpolation (antialiased, matching ``torchvision.transforms
        .Resize``'s default interpolation family as closely as a pure-torch
        op can; see the module docstring for the documented PIL-vs-torch
        deviation), centre crop to ``_CROP_SIZE``, then normalize per
        channel.
        """
        chw = image.permute(0, 3, 1, 2)
        _, _, h, w = chw.shape
        if h <= w:
            new_h = _RESIZE_EDGE
            new_w = max(int(round(w * _RESIZE_EDGE / h)), 1)
        else:
            new_w = _RESIZE_EDGE
            new_h = max(int(round(h * _RESIZE_EDGE / w)), 1)
        resized = F.interpolate(
            chw, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
        )

        top = max((new_h - _CROP_SIZE) // 2, 0)
        left = max((new_w - _CROP_SIZE) // 2, 0)
        cropped = resized[:, :, top : top + _CROP_SIZE, left : left + _CROP_SIZE]
        return (cropped - self._mean) / self._std

    def forward(self, image: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Preprocess and run the ResNet-18 backbone, returning a flat feature vector."""
        preprocessed = self._preprocess(image)
        features = self.backbone(preprocessed)
        # Under a bfloat16 autocast the backbone emits bfloat16; the port promises float32
        # whatever autocast state the caller runs under.
        return {"features": features.reshape(features.shape[0], -1).to(torch.float32)}

    def freeze(self) -> None:
        """Set the backbone to non-trainable (the default: a fixed feature extractor)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        super().freeze()

    def unfreeze(self) -> None:
        """Make the backbone trainable, for end-to-end fine-tuning."""
        super().unfreeze()
        for p in self.backbone.parameters():
            p.requires_grad_(True)


__all__ = ["ResNet18Features"]
