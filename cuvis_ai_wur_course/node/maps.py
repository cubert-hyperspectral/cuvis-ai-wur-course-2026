"""Chemical-map nodes for the end-to-end U-Net course notebook.

Dequantizes the stored uint16 counts, crops the centre region a U-Net head
predicts out of a padded slice, accumulates per-belly weighted means across
the slices of one pass, and measures how spatially structured a predicted
map is (the semivariogram nugget used by Engstrom et al., 2025).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


class Dequantize(Node):
    """Converts a quantized uint16-valued cube back to reflectance.

    Reverses the per-frame affine quantization ``q = (R - bias) * scale``
    the course data ships with: ``R = q / scale + bias``. ``scale`` and
    ``bias`` are stored per frame (one scalar pair per slice), not per
    pixel, matching the course's ``quant_scales`` / ``quant_biases`` files.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Quantized counts held as float32, BHWC.",
        ),
        "quant_scale": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-frame quantization scale.",
        ),
        "quant_bias": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-frame quantization bias.",
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Dequantized reflectance, BHWC float32.",
        ),
    }

    def forward(
        self,
        cube: torch.Tensor,
        quant_scale: torch.Tensor,
        quant_bias: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        """Apply ``cube / scale + bias`` with per-frame broadcasting."""
        scale = quant_scale.view(-1, 1, 1, 1)
        bias = quant_bias.view(-1, 1, 1, 1)
        return {"cube": cube / scale + bias}


class CenterCrop(Node):
    """Crops the centre ``crop_height x crop_width`` region of a cube.

    Used to cut the region a valid-convolution U-Net predicts out of a
    larger, padded slice, so a full-resolution map (such as a pixel-wise
    PLS prediction) can be compared against the smaller region the network
    actually sees. The node takes no mask: the course ships its slice masks
    on the U-Net output grid, and the pipeline fills any input port named
    ``mask`` from the batch whether or not an edge leads into it, so an
    optional mask port here would receive that coarse mask and crop it to
    the wrong size. ``MaskToMapResolution`` brings a mask stored at the
    slice resolution to the map's region.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube, BHWC float32.",
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, "crop_height", "crop_width", -1),
            description="Centre-cropped cube.",
        ),
    }

    def __init__(self, crop_height: int, crop_width: int, **kwargs) -> None:
        super().__init__(crop_height=crop_height, crop_width=crop_width, **kwargs)
        self.crop_height = crop_height
        self.crop_width = crop_width

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Cut the centred ``crop_height x crop_width`` region from the cube."""
        top, left = self._offsets(cube.shape[-3], cube.shape[-2])
        return {"cube": cube[:, top : top + self.crop_height, left : left + self.crop_width, :]}

    def _offsets(self, height: int, width: int) -> tuple[int, int]:
        return (height - self.crop_height) // 2, (width - self.crop_width) // 2


class GroupWeightedMean(Node):
    """Accumulates a mask-weighted per-group mean across a pass of frames.

    Each forward call adds one batch of maps to a running, per-group sum and
    pixel count, keyed by an integer group id (the belly id in the course
    notebook). The running mean equals the pooled mean over every masked
    pixel of every frame seen for that group so far, which is exactly how
    the notebook turns per-slice chemical maps into one belly prediction:
    larger slices (more masked pixels) weigh proportionally more.

    Stateful and non-exportable: it carries fixed-capacity buffers across
    calls and is meant to run once per pass of a pipeline, not be traced
    into a single ONNX graph. Call :meth:`reset` before each such pass.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.STATEFUL, NodeTag.TORCH})

    INPUT_SPECS = {
        "map": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "n_targets"),
            description="Predicted map, BHWC float32 (one channel per target).",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Labelled mask at the map's resolution, 0 = background.",
        ),
        "group": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Group id per frame (0 <= id < max_groups), e.g. the belly id.",
        ),
    }
    OUTPUT_SPECS = {
        "values": PortSpec(
            dtype=torch.float32,
            shape=("max_groups", "n_targets"),
            description="Running mask-weighted mean per group, over all frames seen so far.",
        ),
        "counts": PortSpec(
            dtype=torch.float32,
            shape=("max_groups",),
            description="Running count of masked pixels seen per group.",
        ),
    }

    def __init__(self, max_groups: int, n_targets: int = 1, **kwargs) -> None:
        super().__init__(max_groups=max_groups, n_targets=n_targets, **kwargs)
        self.max_groups = max_groups
        self.n_targets = n_targets
        self.register_buffer("_sum", torch.zeros(max_groups, n_targets, dtype=torch.float64))
        self.register_buffer("_count", torch.zeros(max_groups, dtype=torch.float64))

    def reset(self) -> None:
        """Zero the running sums and counts, starting a new pass."""
        self._sum.zero_()
        self._count.zero_()

    def compute(self) -> dict[int, torch.Tensor]:
        """Return ``{group_id: mean}`` for every group with at least one pixel seen."""
        result: dict[int, torch.Tensor] = {}
        for group_id in range(self.max_groups):
            count = self._count[group_id]
            if count > 0:
                result[group_id] = (self._sum[group_id] / count).to(torch.float32)
        return result

    def forward(
        self,
        map: torch.Tensor,  # noqa: A002 - port name matches the pipeline contract
        mask: torch.Tensor,
        group: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        """Add one batch of masked maps to the running per-group accumulators."""
        valid = (mask > 0).unsqueeze(-1).to(map.dtype)  # [B, H, W, 1]
        masked = map * valid
        batch_size = map.shape[0]
        for b in range(batch_size):
            group_id = int(group[b].item())
            pixel_sum = masked[b].sum(dim=(0, 1)).to(torch.float64)
            pixel_count = valid[b].sum().to(torch.float64)
            self._sum[group_id] += pixel_sum
            self._count[group_id] += pixel_count
        return {
            "values": (self._sum / self._count.clamp_min(1.0).unsqueeze(-1)).to(torch.float32),
            "counts": self._count.to(torch.float32),
        }


class ChemicalMapStatistics(Node):
    """Measures how spatially structured a predicted chemical map is.

    Implements the geostatistics link used in section 5.7 / 8.3 of the
    notebook: the semivariogram at unit lag in each direction,

        gamma(dh, dw) = 0.5 * sum(M_i * (Y - shift(Y, dh, dw))^2) / sum(M_i),

    the nugget effect C0 = 0.5 * (gamma(0, 1) + gamma(1, 0)) (the variance
    between direct neighbours, i.e. variance with no spatial correlation),
    and the share of that total variance which is spatially correlated,
    ``1 - C0 / variance``. It also reports the share of masked pixels whose
    prediction falls outside the physically valid 0 to 100 range.

    ``mask`` may be given at the map's own resolution (``mask_scale=1``) or
    at half that resolution (``mask_scale=2``, the course's convention for
    the full-resolution pixel-wise PLS map, whose mask is the U-Net's
    output-resolution mask): the mask is then upsampled with nearest-
    neighbour interpolation to the map's size before use, matching the
    notebook's ``np.kron(mask, np.ones((2, 2)))``.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.TORCH})

    INPUT_SPECS = {
        "map": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Predicted chemical map, BHWC float32, one channel.",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Labelled mask, at the map's resolution or mask_scale times coarser.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "variance": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Variance of the map over the masked pixels.",
        ),
        "nugget": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Nugget effect C0: variance between direct neighbours.",
        ),
        "structured_share": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="1 - C0 / variance: the share of variance that is spatially correlated.",
        ),
        "out_of_range_share": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Share of masked pixels outside the 0 to 100 range.",
        ),
    }

    def __init__(self, mask_scale: int = 1, eps: float = 1e-12, **kwargs) -> None:
        if mask_scale not in (1, 2):
            raise ValueError(f"mask_scale must be 1 or 2, got {mask_scale!r}")
        super().__init__(mask_scale=mask_scale, eps=eps, **kwargs)
        self.mask_scale = mask_scale
        self.eps = eps

    def forward(
        self,
        map: torch.Tensor,
        mask: torch.Tensor | None = None,
        **_,  # noqa: A002
    ) -> dict[str, torch.Tensor]:
        """Compute variance, nugget, structured share, and out-of-range share per frame."""
        y = map[..., 0]  # [B, H, W]
        if mask is not None:
            m = mask
            if self.mask_scale != 1:
                m = (
                    torch.nn.functional.interpolate(
                        m.unsqueeze(1).to(torch.float32),
                        scale_factor=self.mask_scale,
                        mode="nearest",
                    )
                    .squeeze(1)
                    .to(torch.int32)
                )
            valid = (m > 0).to(y.dtype)
        else:
            valid = torch.ones_like(y)

        count = valid.sum(dim=(1, 2)).clamp_min(self.eps)
        mean = (y * valid).sum(dim=(1, 2)) / count
        variance = ((y - mean.view(-1, 1, 1)) ** 2 * valid).sum(dim=(1, 2)) / count

        gamma_h = self._semivariogram(y, valid, dh=0, dw=1)
        gamma_w = self._semivariogram(y, valid, dh=1, dw=0)
        nugget = 0.5 * (gamma_h + gamma_w)

        structured_share = 1.0 - nugget / variance.clamp_min(self.eps)
        out_of_range = ((y < 0.0) | (y > 100.0)).to(y.dtype) * valid
        out_of_range_share = out_of_range.sum(dim=(1, 2)) / count

        return {
            "variance": variance,
            "nugget": nugget,
            "structured_share": structured_share,
            "out_of_range_share": out_of_range_share,
        }

    def _semivariogram(self, y: torch.Tensor, mask: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
        """``gamma(dh, dw)`` for a unit lag, per frame (batched over dim 0)."""
        height, width = y.shape[-2], y.shape[-1]
        # Shifted vs. base pair, both cropped to the overlapping region.
        y_base = y[:, : height - dh, : width - dw]
        y_shift = y[:, dh:, dw:]
        m_base = mask[:, : height - dh, : width - dw]
        diff_sq = (y_base - y_shift) ** 2
        return 0.5 * (m_base * diff_sq).sum(dim=(1, 2)) / m_base.sum(dim=(1, 2)).clamp_min(self.eps)


class MaskToMapResolution(Node):
    """Bring a slice mask to the region and resolution of a predicted chemical map.

    A chemical map covers only the centre region of a slice that the valid convolutions
    keep, usually at a coarser resolution. The course therefore crops the slice mask to
    that region, resizes it to the map size (bilinear, then rounded) and erodes it by one
    pixel, so the border pixels of a belly, whose receptive field mixes belly and
    background, stay out of the statistics. This node does the same with torch ops: a
    centre crop of ``crop_factor`` times the map size, a bilinear resize by ``downsample``
    followed by rounding (for a factor of two this is the mean of each 2 x 2 block), and
    ``erode_px`` rounds of erosion with the four-connected cross that
    ``scipy.ndimage.binary_erosion`` uses by default, with the border counted as background.

    The course's own ``HSI_data_binned_spectra`` masks are not a slice-resolution mask that
    needs this treatment: they already sit on the U-Net's output grid, and go straight into
    ``ChemicalMapStatistics`` (``mask_scale`` 2 for the full-resolution pixel-wise PLS map,
    which upsamples that mask by two, nearest neighbour). The 02 pipelines therefore do not
    use this node at all. It serves a mask stored at the slice's own resolution instead, the
    way the course's ``HSI_binned_images`` folder ships them.

    The crop is clamped to the slice, so a slice smaller than the requested region is used
    whole on that axis. All sizes come from tensor shapes, never from values, so the node
    traces with ``torch.export`` at fixed shapes.

    Parameters
    ----------
    crop_factor : int
        Centre-crop size as a multiple of the map size. Default ``2``.
    downsample : int
        Resize factor from the crop to the output mask. Default ``2``.
    erode_px : int
        Rounds of one-pixel cross erosion applied after the resize. Default ``1``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Slice mask [B, H, W] int32; 0 is background.",
        ),
        "map": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Predicted map [B, H', W', 1]; only its spatial size is used.",
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Binary mask [B, Hc / downsample, Wc / downsample] int32 at the map's "
            "region, 1 for belly pixels.",
        ),
    }

    def __init__(
        self, crop_factor: int = 2, downsample: int = 2, erode_px: int = 1, **kwargs: Any
    ) -> None:
        if crop_factor < 1 or downsample < 1 or erode_px < 0:
            raise ValueError("crop_factor and downsample must be >= 1, erode_px >= 0")
        self.crop_factor = int(crop_factor)
        self.downsample = int(downsample)
        self.erode_px = int(erode_px)
        super().__init__(
            crop_factor=self.crop_factor,
            downsample=self.downsample,
            erode_px=self.erode_px,
            **kwargs,
        )

    @staticmethod
    def _erode_cross(binary: torch.Tensor) -> torch.Tensor:
        """One round of erosion with the four-connected cross, border as background."""
        padded = F.pad(binary, (1, 1, 1, 1), value=0.0)
        centre = padded[:, :, 1:-1, 1:-1]
        up = padded[:, :, :-2, 1:-1]
        down = padded[:, :, 2:, 1:-1]
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]
        return centre * up * down * left * right

    def forward(self, mask: torch.Tensor, map: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Crop, resize, round and erode the slice mask to the map's grid."""
        _, h, w = mask.shape
        crop_h = min(int(map.shape[1]) * self.crop_factor, h)
        crop_w = min(int(map.shape[2]) * self.crop_factor, w)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = mask[:, top : top + crop_h, left : left + crop_w]

        binary = (cropped > 0).to(torch.float32).unsqueeze(1)
        out_h = max(crop_h // self.downsample, 1)
        out_w = max(crop_w // self.downsample, 1)
        resized = F.interpolate(binary, size=(out_h, out_w), mode="bilinear", align_corners=False)
        rounded = torch.round(resized)
        for _ in range(self.erode_px):
            rounded = self._erode_cross(rounded)
        return {"mask": rounded.squeeze(1).to(torch.int32)}
