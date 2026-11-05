"""Rendering nodes for the course videos."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai.utils.torch_draw import draw_text
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


class BandSweep(Node):
    """Render one grayscale frame per band, so a video sink plays a cube as a sweep through
    its wavelengths.

    Each band is stretched with its own percentiles (``low`` and ``high``, the 0.5 and 99.5
    percent of the cuvis-ai false-colour selectors) to [0, 1], repeated into three channels and,
    with ``label``, stamped with its wavelength in the bottom-left corner (the top-left is where
    ``ToVideoNode`` prints its frame counter). Every ``step``-th band is
    rendered in band order, batch elements one after another, so ``rgb_image`` carries
    ``B * ceil(C / step)`` frames for ``ToVideoNode``. A band without spread (a constant image)
    renders black instead of dividing by zero.

    The label is drawn with the eager bitmap helper of cuvis-ai on a uint8 copy of the frame, so
    this node is a visualiser outside any exported span: it carries no ONNX tag and the pipeline
    author keeps it out of an exported span.

    Parameters
    ----------
    step : int, optional
        Render every ``step``-th band (default 1, every band).
    low, high : float, optional
        Percentile fractions of the per-band stretch (default 0.005 and 0.995).
    label : bool, optional
        Stamp the wavelength in nanometres on each frame (default True).
    label_scale : int, optional
        Bitmap font scale of the label (default 3, a 21 px tall line).
    eps : float, optional
        Floor of the percentile range before dividing (default 1e-6).
    """

    _category = NodeCategory.VISUALIZER
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.IMAGE, NodeTag.VIDEO})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelength array [C] in nanometres, one entry per band.",
        ),
    }
    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="One grayscale frame per rendered band [B * ceil(C / step), H, W, 3] "
            "in [0, 1], in band order, batch elements one after another.",
        ),
    }

    def __init__(
        self,
        step: int = 1,
        low: float = 0.005,
        high: float = 0.995,
        label: bool = True,
        label_scale: int = 3,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if int(step) < 1:
            raise ValueError("step must be at least 1")
        if not 0.0 <= float(low) < float(high) <= 1.0:
            raise ValueError("percentile fractions must satisfy 0 <= low < high <= 1")
        self.step = int(step)
        self.low = float(low)
        self.high = float(high)
        self.label = bool(label)
        self.label_scale = int(label_scale)
        self.eps = float(eps)
        super().__init__(
            step=self.step,
            low=self.low,
            high=self.high,
            label=self.label,
            label_scale=self.label_scale,
            eps=self.eps,
            **kwargs,
        )
        self.register_buffer("_quantiles", torch.tensor([self.low, self.high], dtype=torch.float32))

    def forward(self, cube: torch.Tensor, wavelengths: np.ndarray, **_: Any) -> dict[str, Any]:
        """Stretch every ``step``-th band of every batch element into a labelled gray frame."""
        batch, height, width, channels = cube.shape
        bands = range(0, int(channels), self.step)
        frames = torch.empty(
            (batch * len(bands), height, width, 3), dtype=torch.float32, device=cube.device
        )
        index = 0
        for b in range(int(batch)):
            for k in bands:
                band = cube[b, :, :, k]
                q = torch.quantile(band.reshape(-1), self._quantiles)
                gray = ((band - q[0]) / (q[1] - q[0]).clamp_min(self.eps)).clamp(0.0, 1.0)
                frame = gray.unsqueeze(-1).expand(height, width, 3)
                if self.label:
                    frame = self._stamp(frame, int(wavelengths[k]))
                frames[index].copy_(frame)
                index += 1
        return {"rgb_image": frames}

    def _stamp(self, frame: torch.Tensor, nanometres: int) -> torch.Tensor:
        """Return a copy of ``frame`` with the wavelength drawn in the bottom-left corner."""
        image = (frame * 255.0).round().to(torch.uint8).contiguous()
        y = int(image.shape[0]) - 8 - 7 * self.label_scale  # the bitmap font is 7 rows tall
        draw_text(image, 8, y, f"{nanometres} nm", (255, 255, 255), scale=self.label_scale)
        return image.to(torch.float32) / 255.0
