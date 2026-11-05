"""Source node for the WUR course data module batches."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

# The batch keys besides ``cube`` that a frame can declare, in port order.
_OPTIONAL_KEYS = ("wavelengths", "mask", "target", "fold", "group", "quant_scale", "quant_bias")


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

    def __init__(self, keys: Sequence[str] | None = None, **kwargs: Any) -> None:
        """Create the frame, optionally trimmed to the ports one pipeline uses.

        Parameters
        ----------
        keys : sequence of str, optional
            Batch keys this instance declares besides ``cube``, from ``wavelengths``,
            ``mask``, ``target``, ``fold``, ``group``, ``quant_scale`` and
            ``quant_bias``. ``None`` (the default) declares all eight ports. With keys
            the instance's input and output ports are trimmed before the node is
            built, so a pipeline diagram shows only the ports that pipeline wires and
            the batch keys it does not name are never fed in. The class attributes
            keep every port, so the plugin manifest still lists all eight.
        """
        if keys is not None:
            keys = [str(key) for key in keys]
            unknown = sorted(set(keys) - set(_OPTIONAL_KEYS) - {"cube"})
            if unknown:
                raise ValueError(
                    f"CourseFrame keys must be among {list(_OPTIONAL_KEYS)}, got {unknown}"
                )
            kept = ["cube", *(key for key in _OPTIONAL_KEYS if key in keys)]
            self.INPUT_SPECS = {key: type(self).INPUT_SPECS[key] for key in kept}
            self.OUTPUT_SPECS = {key: type(self).OUTPUT_SPECS[key] for key in kept}
        super().__init__(keys=keys, **kwargs)

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        fold: torch.Tensor | None = None,
        group: torch.Tensor | None = None,
        quant_scale: torch.Tensor | None = None,
        quant_bias: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Normalize a course batch into pipeline ports, passing optional keys through.

        Only the ports this instance declares are emitted, so a frame trimmed with
        ``keys`` hands on nothing else even when the batch carries more keys.
        """
        result: dict[str, Any] = {"cube": cube.to(torch.float32)}
        if wavelengths is not None:
            # The builtin source emits np.int32 [C]; the course grid (747.92 nm steps) rounds to
            # whole nanometres, which every wavelength consumer here only uses for lookup.
            result["wavelengths"] = wavelengths[0].round().to(torch.int32).cpu().numpy()
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
        return {key: value for key, value in result.items() if key in self.OUTPUT_SPECS}
