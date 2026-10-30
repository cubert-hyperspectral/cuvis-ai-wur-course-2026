"""Source node for the WUR course data module batches."""

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


class CourseFrame(Node):
    """Normalizes a ``CourseNpzDataModule`` batch into pipeline ports.

    Mirrors ``cuvis_ai.node.data.CU3SDataNode``: casts the cube to float32 and
    passes wavelengths and mask through. In addition it exposes the course
    data's optional per-frame keys (``target``, ``fold``, ``group``,
    ``quant_scale``, ``quant_bias``) as their own output ports. A key the
    batch does not carry is simply absent from the input, and the matching
    output is then also absent (the port is optional on both sides), so a
    pipeline wired against mean-spectra frames (which carry ``target`` and
    ``fold`` but not ``group`` or the quantisation pair) and one wired
    against image frames (the reverse) both run unmodified.

    Like the builtin source it hands the wavelengths on as a NumPy array, so it is not
    part of any exported span: source nodes sit in front of the span, never inside it.
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.TABULAR, NodeTag.METADATA})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1, -1, -1),
            description="Input cube [B, H, W, C].",
        ),
        "wavelengths": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1),
            description="Wavelength axis in nm, batched [B, C] as the data module delivers it.",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Segmentation mask [B, H, W].",
            optional=True,
        ),
        "target": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1),
            description="Regression target(s) per frame [B, T].",
            optional=True,
        ),
        "fold": PortSpec(
            dtype=torch.Tensor,
            shape=(-1,),
            description="Cross-validation fold id per frame [B].",
            optional=True,
        ),
        "group": PortSpec(
            dtype=torch.Tensor,
            shape=(-1,),
            description="Leakage-grouping id per frame (e.g. a belly id) [B].",
            optional=True,
        ),
        "quant_scale": PortSpec(
            dtype=torch.Tensor,
            shape=(-1,),
            description="Dequantization scale per frame [B].",
            optional=True,
        ),
        "quant_bias": PortSpec(
            dtype=torch.Tensor,
            shape=(-1,),
            description="Dequantization bias per frame [B].",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Cube [B, H, W, C] as float32.",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelengths [C] in nm as np.int32, as the builtin source emits them; "
            "Wavelength axis in nm [C] (first batch element).",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Segmentation mask passthrough [B, H, W].",
            optional=True,
        ),
        "target": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Regression target(s) per frame [B, T].",
            optional=True,
        ),
        "fold": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Cross-validation fold id per frame [B].",
            optional=True,
        ),
        "group": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Leakage-grouping id per frame [B].",
            optional=True,
        ),
        "quant_scale": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Dequantization scale per frame [B].",
            optional=True,
        ),
        "quant_bias": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Dequantization bias per frame [B].",
            optional=True,
        ),
    }

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: torch.Tensor,
        mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        fold: torch.Tensor | None = None,
        group: torch.Tensor | None = None,
        quant_scale: torch.Tensor | None = None,
        quant_bias: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Normalize a course batch into pipeline ports, passing optional keys through."""
        result: dict[str, torch.Tensor] = {
            "cube": cube.to(torch.float32),
            # The builtin source emits np.int32 [C]; the course grid (747.92 nm steps) rounds to
            # whole nanometres, which every wavelength consumer here only uses for lookup.
            "wavelengths": wavelengths[0].round().to(torch.int32).cpu().numpy(),
        }
        if mask is not None:
            result["mask"] = mask.to(torch.int32)
        if target is not None:
            result["target"] = target.to(torch.float32)
        if fold is not None:
            result["fold"] = fold.to(torch.int32)
        if group is not None:
            result["group"] = group.to(torch.int32)
        if quant_scale is not None:
            result["quant_scale"] = quant_scale.to(torch.float32)
        if quant_bias is not None:
            result["quant_bias"] = quant_bias.to(torch.float32)
        return result
