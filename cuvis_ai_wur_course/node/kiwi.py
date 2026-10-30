"""Bounding-box filtering and cropping for labelled instance masks.

``LabelBBoxFilter`` keeps the labels of a labelled mask whose axis-aligned
bounding box falls inside a size window, mirroring the size filter the course
applies to SAM's region proposals before any spectral check runs
(``filter_masks_by_size`` in ``using_pretrained_ai.ipynb``).

``LabelCropPad`` crops a single selected label's bounding box out of a cube
and centres it on a fixed-size canvas with the label's own mask applied, the
same placement the course performs by hand before feeding a crop to a
ResNet-18 feature extractor.

Both nodes compute every label's row/column extent in one batched
``scatter_reduce`` pass over flattened ``(batch, label)`` slots, never a
Python loop over objects or ``torch.unique``/boolean-mask indexing, so the
shapes stay fixed and the nodes trace with ``torch.export``.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


def _label_extents(
    mask: torch.Tensor, max_labels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(batch, label) row/column extents and presence, in one scatter pass.

    Parameters
    ----------
    mask : torch.Tensor
        Labelled mask ``[B, H, W]`` int32/int64; label ``0`` is background.
        Values ``>= max_labels`` have no slot and are excluded from every
        returned tensor (their pixels are reported as not present).
    max_labels : int
        Fixed label capacity; also the size of the per-batch label axis of
        every returned tensor.

    Returns
    -------
    min_y, max_y, min_x, max_x : torch.Tensor
        ``[B, max_labels]`` float32 pixel-coordinate extents. Undefined
        (sentinel) where ``present`` is False.
    present : torch.Tensor
        ``[B, max_labels]`` bool; True where at least one pixel carries that
        label (label ``0`` is never marked present).
    """
    b, h, w = mask.shape
    device = mask.device
    labels = mask.to(torch.int64)

    ys = torch.arange(h, dtype=torch.float32, device=device).view(1, h, 1).expand(b, h, w)
    xs = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, w).expand(b, h, w)
    batch_ids = torch.arange(b, dtype=torch.int64, device=device).view(b, 1, 1).expand(b, h, w)

    # Every pixel gets a flat slot; background (0) and out-of-capacity labels
    # share slot 0 of batch 0 as a harmless sink (never read back as "present").
    in_range = (labels > 0) & (labels < max_labels)
    safe_labels = torch.where(in_range, labels, torch.zeros_like(labels))
    flat_idx = (batch_ids * max_labels + safe_labels).reshape(-1)
    weight = in_range.to(torch.float32).reshape(-1)

    n_slots = b * max_labels
    sentinel_hi = float(max(h, w))
    min_y = torch.full((n_slots,), sentinel_hi, dtype=torch.float32, device=device)
    max_y = torch.full((n_slots,), -1.0, dtype=torch.float32, device=device)
    min_x = torch.full((n_slots,), sentinel_hi, dtype=torch.float32, device=device)
    max_x = torch.full((n_slots,), -1.0, dtype=torch.float32, device=device)
    count = torch.zeros((n_slots,), dtype=torch.float32, device=device)

    # Pixels outside `in_range` (background, or a label beyond capacity) would
    # otherwise pollute slot 0 with their real coordinates; route each one to
    # the reduce's own identity value instead (never the minimum for an amin
    # pass, never the maximum for an amax pass) so it cannot shift the result.
    flat_y, flat_x = ys.reshape(-1), xs.reshape(-1)
    in_range_flat = in_range.reshape(-1)
    y_for_min = torch.where(in_range_flat, flat_y, torch.full_like(flat_y, sentinel_hi))
    y_for_max = torch.where(in_range_flat, flat_y, torch.full_like(flat_y, -1.0))
    x_for_min = torch.where(in_range_flat, flat_x, torch.full_like(flat_x, sentinel_hi))
    x_for_max = torch.where(in_range_flat, flat_x, torch.full_like(flat_x, -1.0))

    min_y.scatter_reduce_(0, flat_idx, y_for_min, reduce="amin", include_self=True)
    max_y.scatter_reduce_(0, flat_idx, y_for_max, reduce="amax", include_self=True)
    min_x.scatter_reduce_(0, flat_idx, x_for_min, reduce="amin", include_self=True)
    max_x.scatter_reduce_(0, flat_idx, x_for_max, reduce="amax", include_self=True)
    count.scatter_add_(0, flat_idx, weight)

    min_y = min_y.view(b, max_labels)
    max_y = max_y.view(b, max_labels)
    min_x = min_x.view(b, max_labels)
    max_x = max_x.view(b, max_labels)
    present = count.view(b, max_labels) > 0
    # Slot 0 (background / overflow sink) is never a real label.
    present = present & (
        torch.arange(max_labels, device=device).view(1, max_labels).expand(b, max_labels) > 0
    )
    return min_y, max_y, min_x, max_x, present


def _batched_roll_2d(
    frames: torch.Tensor, shift_y: torch.Tensor, shift_x: torch.Tensor
) -> torch.Tensor:
    """Circularly shift each batch element of a BHWC tensor by its own (dy, dx).

    Equivalent to ``torch.roll(frames[i], (shift_y[i], shift_x[i]), dims=(0, 1))``
    per batch index ``i``, but vectorised over the batch with a single
    ``gather`` so no Python loop or ``.item()`` call is needed: the per-sample
    shift stays a tensor value throughout, which keeps the op traceable.

    Parameters
    ----------
    frames : torch.Tensor
        ``[B, H, W, C]``.
    shift_y, shift_x : torch.Tensor
        ``[B]`` int64 shift amounts, one pair per batch element.
    """
    b, h, w, c = frames.shape
    device = frames.device

    row_idx = torch.arange(h, dtype=torch.int64, device=device).view(1, h, 1)
    col_idx = torch.arange(w, dtype=torch.int64, device=device).view(1, 1, w)
    src_row = (row_idx - shift_y.view(b, 1, 1)) % h  # [B, H, 1]
    src_col = (col_idx - shift_x.view(b, 1, 1)) % w  # [B, 1, W]

    src_row = src_row.expand(b, h, w)
    src_col = src_col.expand(b, h, w)

    # Gather rows first (dim=1), then columns (dim=2); both via expand+gather
    # so the whole thing stays batched tensor ops with fixed output shapes.
    # Modulo already guarantees 0 <= src_row < h and 0 <= src_col < w.
    row_gather_idx = src_row.unsqueeze(-1).expand(b, h, w, c)
    rows_shifted = torch.gather(frames, 1, row_gather_idx)
    col_gather_idx = src_col.unsqueeze(-1).expand(b, h, w, c)
    out = torch.gather(rows_shifted, 2, col_gather_idx)
    return out


class LabelBBoxFilter(Node):
    """Keep the labels of a labelled mask whose bounding box fits a size window.

    Mirrors the course's ``filter_masks_by_size``: a SAM mask (one label id
    per proposed object) is kept only if its tight axis-aligned bounding box
    height and width both fall within configured bounds. Labels outside the
    bounds are zeroed out of the mask; the surviving ones keep their original
    ids.

    The node stays torch-native and fixed-shape: every label's bounding box
    is computed in one batched ``scatter_reduce`` pass (see
    :func:`_label_extents`), never ``torch.unique`` or boolean-mask indexing,
    so it traces with ``torch.export``.

    Parameters
    ----------
    min_height : int
        Smallest allowed bounding-box height in pixels, inclusive.
    max_height : int
        Largest allowed bounding-box height in pixels, inclusive.
    min_width : int
        Smallest allowed bounding-box width in pixels, inclusive.
    max_width : int
        Largest allowed bounding-box width in pixels, inclusive.
    max_labels : int
        Fixed label capacity; also the length of the ``kept`` output's label
        axis. Label ids ``0`` is background. A label id ``>= max_labels`` has
        no tracking slot: its pixels are zeroed out of the output mask (there
        is nowhere to record whether its bbox qualifies), and it never
        appears in ``kept``. Default ``1024``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.BBOX, NodeTag.SEGMENTATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Labelled mask [B, H, W] int32; 0 is background, "
            "each other value one object id (e.g. SAM3SegmentEverything output).",
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Same mask with out-of-bounds labels zeroed; surviving labels "
            "keep their original ids.",
        ),
        "kept": PortSpec(
            dtype=torch.int32,
            shape=(-1, "max_labels"),
            description="[B, max_labels] flags; 1 where that label id's bbox is within "
            "bounds, 0 otherwise (including absent and background label 0).",
        ),
    }

    def __init__(
        self,
        min_height: int,
        max_height: int,
        min_width: int,
        max_width: int,
        max_labels: int = 1024,
        **kwargs: Any,
    ) -> None:
        if min_height < 0 or min_width < 0:
            raise ValueError("min_height and min_width must be >= 0")
        if max_height < min_height:
            raise ValueError("max_height must be >= min_height")
        if max_width < min_width:
            raise ValueError("max_width must be >= min_width")
        if max_labels < 1:
            raise ValueError("max_labels must be >= 1")

        self.min_height = int(min_height)
        self.max_height = int(max_height)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.max_labels = int(max_labels)

        super().__init__(
            min_height=self.min_height,
            max_height=self.max_height,
            min_width=self.min_width,
            max_width=self.max_width,
            max_labels=self.max_labels,
            **kwargs,
        )

    def forward(self, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Zero out labels whose bbox lies outside the configured size window."""
        min_y, max_y, min_x, max_x, present = _label_extents(mask, self.max_labels)

        height = max_y - min_y + 1.0
        width = max_x - min_x + 1.0
        in_bounds = (
            (height >= float(self.min_height))
            & (height <= float(self.max_height))
            & (width >= float(self.min_width))
            & (width <= float(self.max_width))
        )
        kept_flags = (present & in_bounds).to(torch.int32)  # [B, max_labels]

        # Gather each pixel's own label's kept flag back onto the mask shape.
        b, h, w = mask.shape
        labels = mask.to(torch.int64)
        in_range = (labels > 0) & (labels < self.max_labels)
        safe_labels = torch.where(in_range, labels, torch.zeros_like(labels))
        per_pixel_kept = torch.gather(kept_flags, 1, safe_labels.reshape(b, h * w)).reshape(b, h, w)
        survives = in_range & (per_pixel_kept > 0)

        filtered_mask = torch.where(survives, mask, torch.zeros_like(mask))
        return {"mask": filtered_mask, "kept": kept_flags}


class LabelCropPad(Node):
    """Crop one labelled object out of a cube and centre it on a fixed canvas.

    Reproduces the course's crop-and-pad step: the chosen label's tight
    bounding box is cut out of the cube, pixels outside that label's own mask
    within the box are zeroed, and the result is centred on a
    ``canvas_size x canvas_size`` black canvas (``padded_kiwi_rgb`` in
    ``using_pretrained_ai.ipynb``, generalised from 3-channel pseudo-RGB to
    an arbitrary channel count since this node runs on the cube, not the
    rendered preview).

    A bounding box larger than the canvas in either dimension is centre
    cropped to fit instead of overflowing it; the course notebook only warns
    about this case; the node makes the shape guarantee hold unconditionally.
    In that case the label's presence and centring are recomputed from the
    label pixels that actually remain after the centre crop (not from the
    pre-crop bounding box), so a label whose extremes fall outside the
    retained window is centred correctly on whatever of it survives, or
    yields an all-zero canvas when none of it does.

    Centring is implemented as an integer ``torch.roll`` of the whole frame
    (translate the bbox centre to the canvas centre, after placing the source
    cube on a same-size zero canvas), which keeps the node a fixed graph of
    torch ops with no data-dependent shapes.

    The cube sets the device: the SAM3 node emits its mask on the cpu whatever
    device the pipeline runs on, so the mask is moved onto the cube device
    before the extents are computed and the crop comes out next to the cube.

    Parameters
    ----------
    canvas_size : int
        Output canvas height and width in pixels. Default ``256``.
    label_selection : str or int
        ``"first"`` picks the smallest non-zero label id present in each
        frame (matches "pick a kiwi" once upstream filtering has left exactly
        the wanted objects); an int picks that exact label id. Default
        ``"first"``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.MASK, NodeTag.BBOX, NodeTag.IMAGE, NodeTag.HYPERSPECTRAL, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral or RGB cube, BHWC float32.",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Labelled mask [B, H, W] int32; 0 is background.",
        ),
    }
    OUTPUT_SPECS = {
        "crop": PortSpec(
            dtype=torch.float32,
            shape=(-1, "canvas_size", "canvas_size", -1),
            description="[B, canvas_size, canvas_size, C]; the selected label's bbox "
            "centred on a black canvas, zeros outside the label's own mask.",
        ),
    }

    _LABEL_CAPACITY = 4096  # fixed scan range for "first" selection; see docstring below.

    def __init__(
        self,
        canvas_size: int = 256,
        label_selection: str | int = "first",
        **kwargs: Any,
    ) -> None:
        if canvas_size < 1:
            raise ValueError("canvas_size must be >= 1")
        if isinstance(label_selection, bool) or not isinstance(label_selection, (str, int)):
            raise TypeError("label_selection must be 'first' or an int label id")
        if isinstance(label_selection, str) and label_selection != "first":
            raise ValueError("label_selection string value must be 'first'")
        if isinstance(label_selection, int) and label_selection < 1:
            raise ValueError("label_selection as an int must be a positive label id")

        self.canvas_size = int(canvas_size)
        self.label_selection = label_selection

        super().__init__(
            canvas_size=self.canvas_size,
            label_selection=self.label_selection,
            **kwargs,
        )

    def forward(self, cube: torch.Tensor, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Crop the selected label's bbox and centre it on a zero canvas.

        ``label_selection="first"`` scans a fixed id range (``_LABEL_CAPACITY``)
        for the smallest present label per frame: a compile-time-bounded loop
        over a Python range, not a data-dependent shape, so the node stays
        exportable. A label id at or beyond that range is never selected by
        ``"first"``. The per-frame integer translation onto the canvas is
        applied with a batched modular index gather (no ``.item()``, no
        per-sample Python branch), keeping the whole node one fixed graph of
        torch ops.
        """
        b, h, w, c = cube.shape
        device = cube.device
        mask = mask.to(device)

        if isinstance(self.label_selection, int):
            selected = torch.full((b,), self.label_selection, dtype=torch.int64, device=device)
        else:
            selected = self._first_present_label(mask)

        min_y, max_y, min_x, max_x, present = _label_extents(mask, max_labels=self._LABEL_CAPACITY)
        # Gather this frame's selected label's own extent row.
        idx = selected.clamp(max=self._LABEL_CAPACITY - 1).view(b, 1)
        s_min_y = torch.gather(min_y, 1, idx).squeeze(1)
        s_max_y = torch.gather(max_y, 1, idx).squeeze(1)
        s_min_x = torch.gather(min_x, 1, idx).squeeze(1)
        s_max_x = torch.gather(max_x, 1, idx).squeeze(1)
        s_present = torch.gather(present, 1, idx).squeeze(1)

        labels = mask.to(torch.int64)
        label_match = (labels == selected.view(b, 1, 1)) & s_present.view(b, 1, 1)
        masked_cube = cube * label_match.unsqueeze(-1).to(cube.dtype)

        # Place the masked cube (unshifted) on a canvas-sized zero frame.
        # When the cube is smaller than the canvas, pad with zeros; when
        # larger, a centre crop keeps the output at exactly canvas_size (the
        # course only warns in this case; the node guarantees the shape
        # instead of merely warning).
        canvas = self.canvas_size
        if h <= canvas and w <= canvas:
            base = torch.zeros((b, canvas, canvas, c), dtype=cube.dtype, device=device)
            base[:, :h, :w, :] = masked_cube
        else:
            top = max((h - canvas) // 2, 0)
            left = max((w - canvas) // 2, 0)
            cropped = masked_cube[:, top : top + canvas, left : left + canvas, :]
            ch, cw = cropped.shape[1], cropped.shape[2]
            base = torch.zeros((b, canvas, canvas, c), dtype=cube.dtype, device=device)
            base[:, :ch, :cw, :] = cropped

            # The pre-crop extents (and presence) describe the label's full
            # bounding box, which can disagree with what actually survives
            # the centre crop: a label whose extremes sit outside the
            # retained window can have zero pixels left, or pixels whose
            # centre differs from the pre-crop bbox centre. Recompute both
            # from the cropped label mask itself (same fixed scatter-reduce
            # extents, applied to the already-cropped region) so centring
            # and presence always match what the canvas actually holds.
            cropped_label_match = label_match[:, top : top + canvas, left : left + canvas]
            cropped_mask_for_label = torch.where(
                cropped_label_match,
                torch.ones((b, ch, cw), dtype=torch.int32, device=device),
                torch.zeros((b, ch, cw), dtype=torch.int32, device=device),
            )
            s_min_y, s_max_y, s_min_x, s_max_x, s_present = _label_extents(
                cropped_mask_for_label, max_labels=2
            )
            s_min_y = s_min_y[:, 1]
            s_max_y = s_max_y[:, 1]
            s_min_x = s_min_x[:, 1]
            s_max_x = s_max_x[:, 1]
            s_present = s_present[:, 1]

        # The course pads the crop with an offset of (canvas - size) // 2, a floor;
        # a half-to-even round would land one pixel off whenever canvas - size is odd.
        box_h = s_max_y - s_min_y + 1
        box_w = s_max_x - s_min_x + 1
        shift_y = torch.floor((canvas - box_h) * 0.5 - s_min_y).to(torch.int64)
        shift_x = torch.floor((canvas - box_w) * 0.5 - s_min_x).to(torch.int64)
        # No label present in this frame: leave it centred at the origin
        # (an all-zero canvas either way, since masked_cube is all zero).
        shift_y = torch.where(s_present, shift_y, torch.zeros_like(shift_y))
        shift_x = torch.where(s_present, shift_x, torch.zeros_like(shift_x))

        out = _batched_roll_2d(base, shift_y, shift_x)
        return {"crop": out}

    def _first_present_label(self, mask: torch.Tensor) -> torch.Tensor:
        """Smallest non-zero label id present per frame, or 0 if none.

        Scans the fixed id range ``1 .. _LABEL_CAPACITY - 1`` (a compile-time
        Python loop, not a data-dependent shape) and keeps the smallest id
        whose presence flag is set.
        """
        b = mask.shape[0]
        device = mask.device
        best = torch.zeros(b, dtype=torch.int64, device=device)
        found = torch.zeros(b, dtype=torch.bool, device=device)
        labels = mask.to(torch.int64)
        for label_id in range(1, self._LABEL_CAPACITY):
            present = (labels == label_id).any(dim=(1, 2))
            take = present & (~found)
            best = torch.where(take, torch.full_like(best, label_id), best)
            found = found | present
        return best


__all__ = ["LabelBBoxFilter", "LabelCropPad"]


class LabelKeep(Node):
    """Write one keep-or-drop decision per object back onto a labelled mask.

    Closes the loop of the course's spectral-match filter: ``SpectralSignatureExtractor``
    turns the SAM mask into one mean spectrum per object, ``SignaturesToReferences`` and
    ``SpectralAngleMapper`` score every object against the kiwi reference spectrum, and
    ``ScoreThreshold`` turns the angles into one decision per object. This node sets the
    labels of the rejected objects to 0 and leaves the kept labels as they are.

    The decisions arrive on the batch axis, one ``[1, 1, 1]`` entry per object in
    ``object_ids`` order (that is how ``SignaturesToReferences`` lays the objects out), so
    the node processes one frame per call, which is also what the SAM3 node emits.

    The mapping from label id to decision goes through a fixed-size lookup table of
    ``max_labels`` entries (a scatter, then a gather by label id), so the node is one fixed
    graph of torch ops and traces with ``torch.export``. A label id at or above
    ``max_labels`` has no table slot and is dropped.

    The mask sets the device. The SAM3 node emits its mask and object ids on the cpu
    whatever device the pipeline runs on, while the decisions arrive from the threshold
    node on the pipeline device, so the two small inputs are moved onto the mask device
    before the lookup (the adapter cuvis-ai's ``SpectralSignatureExtractor`` applies to
    the same mask).

    Parameters
    ----------
    max_labels : int
        Capacity of the label lookup table. Default ``1024``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Labelled mask [1, H, W] int32; 0 is background.",
        ),
        "object_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Label id of every object [1, N], in the order of the decisions.",
        ),
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, 1, 1, 1),
            description="One decision per object [N, 1, 1, 1]; true keeps the label.",
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="The mask with every rejected label set to 0.",
        ),
        "kept": PortSpec(
            dtype=torch.int32,
            shape=(1, -1),
            description="1 for every object whose label survived, else 0 [1, N].",
        ),
    }

    def __init__(self, max_labels: int = 1024, **kwargs: Any) -> None:
        if max_labels < 1:
            raise ValueError("max_labels must be >= 1")
        self.max_labels = int(max_labels)
        super().__init__(max_labels=self.max_labels, **kwargs)

    def forward(
        self,
        mask: torch.Tensor,
        object_ids: torch.Tensor,
        decisions: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Zero the labels whose decision is false; keep the others unchanged."""
        device = mask.device
        keep = decisions.reshape(-1).to(device=device, dtype=torch.int32)
        raw_ids = object_ids.reshape(-1).to(device)
        ids = raw_ids.clamp(min=0, max=self.max_labels - 1)
        table = torch.zeros(self.max_labels, dtype=torch.int32, device=device)
        table = table.scatter(0, ids, keep)
        # Background never counts as an object, whatever a stray decision says.
        slot_is_object = (torch.arange(self.max_labels, device=device) > 0).to(torch.int32)
        table = table * slot_is_object

        labels = mask.to(torch.int64)
        in_range = (labels < self.max_labels).to(torch.int32)
        keep_map = table[labels.clamp(max=self.max_labels - 1)] * in_range
        id_in_range = (raw_ids < self.max_labels).to(torch.int32)
        kept = (table[ids] * id_in_range).reshape(1, -1)
        return {"mask": mask * keep_map, "kept": kept}
