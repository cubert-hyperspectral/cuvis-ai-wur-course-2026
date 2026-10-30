"""Tests for LabelBBoxFilter and LabelCropPad (cuvis_ai_wur_course.node.kiwi).

Golden cases are hand-built labelled masks so the bounding-box extents and the
kept/dropped outcome are known by construction, matching the notebook's size
filter (``filter_masks_by_size``) and crop/pad step for a single kiwi.
"""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.kiwi import LabelBBoxFilter, LabelCropPad, LabelKeep

pytestmark = pytest.mark.unit


# ----- helpers ---------------------------------------------------------


class _MaskSource(Node):
    """Minimal source node emitting a fixed labelled mask, for reload smoke tests.

    Defined at module scope (not inside a test function) so the class resolves
    as ``<module>.<ClassName>`` after a pipeline save/reload round trip.
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.MASK})

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"mask": PortSpec(dtype=torch.int32, shape=(-1, -1, -1))}

    def forward(self, **_):
        return {"mask": _mask_with_boxes(32, 32, {1: (2, 12, 3, 13)})}


class _CubeMaskSource(Node):
    """Minimal source node emitting a fixed cube + labelled mask, for reload smoke tests.

    Defined at module scope for the same reason as :class:`_MaskSource`.
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.MASK})

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "mask": PortSpec(dtype=torch.int32, shape=(-1, -1, -1)),
    }

    def forward(self, **_):
        cube = torch.zeros(1, 8, 8, 2, dtype=torch.float32)
        cube[0, 1:4, 2:7, :] = 1.0
        mask = torch.zeros(1, 8, 8, dtype=torch.int32)
        mask[0, 1:4, 2:7] = 1
        return {"cube": cube, "mask": mask}


def _mask_with_boxes(
    height: int, width: int, boxes: dict[int, tuple[int, int, int, int]]
) -> torch.Tensor:
    """Build a [1, height, width] int32 mask with one filled rectangle per label.

    ``boxes`` maps label id -> (y0, y1, x0, x1) inclusive pixel bounds.
    """
    mask = torch.zeros(1, height, width, dtype=torch.int32)
    for label, (y0, y1, x0, x1) in boxes.items():
        mask[0, y0 : y1 + 1, x0 : x1 + 1] = label
    return mask


# ----- LabelBBoxFilter ---------------------------------------------------


def test_bbox_filter_keeps_label_within_bounds():
    # height 10..20, width 10..20; label 1 is an 11x11 box -> kept.
    mask = _mask_with_boxes(32, 32, {1: (2, 12, 3, 13)})
    node = LabelBBoxFilter(min_height=10, max_height=20, min_width=10, max_width=20, max_labels=8)

    out = node(mask=mask)

    assert torch.equal(out["mask"], mask)
    expected_kept = torch.zeros(1, 8, dtype=torch.int32)
    expected_kept[0, 1] = 1
    assert torch.equal(out["kept"], expected_kept)


def test_bbox_filter_drops_label_by_height():
    # height 30 (= 29 - 0 + 1) is outside [10, 20]; width 11 is inside.
    mask = _mask_with_boxes(40, 32, {1: (0, 29, 3, 13)})
    node = LabelBBoxFilter(min_height=10, max_height=20, min_width=10, max_width=20, max_labels=8)

    out = node(mask=mask)

    assert torch.equal(out["mask"], torch.zeros_like(mask))
    assert torch.equal(out["kept"], torch.zeros(1, 8, dtype=torch.int32))


def test_bbox_filter_drops_label_by_width():
    # width 25 is outside [10, 20]; height 11 is inside.
    mask = _mask_with_boxes(32, 40, {1: (2, 12, 0, 24)})
    node = LabelBBoxFilter(min_height=10, max_height=20, min_width=10, max_width=20, max_labels=8)

    out = node(mask=mask)

    assert torch.equal(out["mask"], torch.zeros_like(mask))
    assert torch.equal(out["kept"], torch.zeros(1, 8, dtype=torch.int32))


def test_bbox_filter_mixed_labels_kept_and_dropped():
    # label 1: 11x11 (kept); label 2: height 30 (dropped); label 3: width 25 (dropped).
    mask = _mask_with_boxes(
        64,
        64,
        {
            1: (2, 12, 3, 13),
            2: (20, 49, 20, 30),
            3: (40, 50, 35, 59),
        },
    )
    node = LabelBBoxFilter(min_height=10, max_height=20, min_width=10, max_width=20, max_labels=8)

    out = node(mask=mask)

    expected_mask = _mask_with_boxes(64, 64, {1: (2, 12, 3, 13)})
    assert torch.equal(out["mask"], expected_mask)
    expected_kept = torch.zeros(1, 8, dtype=torch.int32)
    expected_kept[0, 1] = 1
    assert torch.equal(out["kept"], expected_kept)


def test_bbox_filter_empty_mask():
    mask = torch.zeros(2, 16, 16, dtype=torch.int32)
    node = LabelBBoxFilter(min_height=1, max_height=20, min_width=1, max_width=20, max_labels=8)

    out = node(mask=mask)

    assert torch.equal(out["mask"], mask)
    assert torch.equal(out["kept"], torch.zeros(2, 8, dtype=torch.int32))


def test_bbox_filter_label_id_above_max_labels_is_dropped_from_kept_tracking():
    # max_labels=4 means valid label ids are 0..3; label id 5 is above capacity.
    # Documented behavior: pixels with a label >= max_labels are zeroed in the
    # output mask (treated as background) because there is no slot to track
    # whether that label's bbox qualifies.
    mask = _mask_with_boxes(32, 32, {5: (2, 12, 3, 13)})
    node = LabelBBoxFilter(min_height=1, max_height=20, min_width=1, max_width=20, max_labels=4)

    out = node(mask=mask)

    assert torch.equal(out["mask"], torch.zeros_like(mask))
    assert out["kept"].shape == (1, 4)


def test_bbox_filter_port_contract():
    mask = _mask_with_boxes(32, 32, {1: (2, 12, 3, 13)})
    node = LabelBBoxFilter(min_height=1, max_height=20, min_width=1, max_width=20, max_labels=16)

    out = node(mask=mask)

    for port_name in node.OUTPUT_SPECS:
        assert port_name in out
    assert out["mask"].dtype == torch.int32
    assert out["mask"].shape == mask.shape
    assert out["kept"].dtype == torch.int32
    assert out["kept"].shape == (1, 16)


def test_bbox_filter_reload_smoke(tmp_path):
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    pipe = CuvisPipeline("bbox_reload_test")
    src = _MaskSource(name="src")
    node = LabelBBoxFilter(
        name="bbox_filter", min_height=1, max_height=20, min_width=1, max_width=20, max_labels=8
    )
    pipe.connect(src.outputs.mask, node.inputs.mask)

    before = pipe.forward(batch={})

    save_path = tmp_path / "bbox_filter_pipeline.yaml"
    pipe.save_to_file(str(save_path))
    rebuilt = PipelineBuilder().build_from_config(str(save_path))
    after = rebuilt.forward(batch={})

    assert torch.equal(
        before[(node.name, "mask")],
        after[(node.name, "mask")],
    )


def test_bbox_filter_export_smoke():
    mask = _mask_with_boxes(32, 32, {1: (2, 12, 3, 13)})
    node = LabelBBoxFilter(
        min_height=1, max_height=20, min_width=1, max_width=20, max_labels=8
    ).eval()

    class _Probe(torch.nn.Module):
        def __init__(self, inner: LabelBBoxFilter):
            super().__init__()
            self.inner = inner

        def forward(self, mask: torch.Tensor) -> torch.Tensor:
            out = self.inner(mask=mask)
            return out["mask"].to(torch.float32)

    program = torch.export.export(_Probe(node), (mask,))

    other = _mask_with_boxes(32, 32, {1: (5, 15, 5, 15)})
    traced = program.module()(other)
    expected = node(mask=other)["mask"].to(torch.float32)
    assert torch.allclose(traced, expected)


# ----- LabelCropPad ------------------------------------------------------


def test_crop_pad_places_label_centered_on_canvas():
    # A single 3x5 block of label 1 at rows 1..3, cols 2..6 in an 8x10 frame.
    cube = torch.zeros(1, 8, 10, 2, dtype=torch.float32)
    cube[0, 1:4, 2:7, :] = 1.0
    mask = torch.zeros(1, 8, 10, dtype=torch.int32)
    mask[0, 1:4, 2:7] = 1

    node = LabelCropPad(canvas_size=16, label_selection="first")
    out = node(cube=cube, mask=mask)

    crop = out["crop"][0]
    # bbox height=3, width=5; the course pads with an offset of (canvas - size) // 2
    # on each axis, so the block starts at row (16 - 3) // 2 and col (16 - 5) // 2.
    row0 = (16 - 3) // 2
    col0 = (16 - 5) // 2
    expected = torch.zeros(16, 16, 2, dtype=torch.float32)
    expected[row0 : row0 + 3, col0 : col0 + 5, :] = 1.0
    assert torch.equal(crop, expected)


def test_crop_pad_zeros_outside_label_mask_within_bbox():
    # bbox is a 3x3 block but only the center pixel belongs to the label;
    # non-label pixels inside the bbox must be zeroed in the crop.
    cube = torch.ones(1, 8, 8, 1, dtype=torch.float32) * 9.0
    mask = torch.zeros(1, 8, 8, dtype=torch.int32)
    mask[0, 2:5, 2:5] = 0
    mask[0, 3, 3] = 1  # single foreground pixel inside the 3x3 extent

    node = LabelCropPad(canvas_size=8, label_selection="first")
    out = node(cube=cube, mask=mask)

    crop = out["crop"][0]
    assert crop.sum().item() == pytest.approx(9.0)
    nonzero = (crop != 0).nonzero(as_tuple=False)
    assert nonzero.shape[0] == 1


def test_crop_pad_explicit_label_id_selection():
    cube = torch.zeros(1, 8, 8, 1, dtype=torch.float32)
    cube[0, 1:3, 1:3, :] = 2.0
    cube[0, 5:7, 5:7, :] = 3.0
    mask = torch.zeros(1, 8, 8, dtype=torch.int32)
    mask[0, 1:3, 1:3] = 1
    mask[0, 5:7, 5:7] = 2

    node = LabelCropPad(canvas_size=8, label_selection=2)
    out = node(cube=cube, mask=mask)

    crop = out["crop"][0]
    assert crop.sum().item() == pytest.approx(3.0 * 4)
    # the label-1 region's value (2.0) must not appear in the crop.
    assert not torch.any(crop == 2.0)


def test_crop_pad_empty_mask_yields_blank_canvas():
    cube = torch.rand(1, 8, 8, 3, dtype=torch.float32)
    mask = torch.zeros(1, 8, 8, dtype=torch.int32)

    node = LabelCropPad(canvas_size=8, label_selection="first")
    out = node(cube=cube, mask=mask)

    assert torch.equal(out["crop"], torch.zeros(1, 8, 8, 3, dtype=torch.float32))


def test_crop_pad_port_contract():
    cube = torch.rand(2, 10, 10, 4, dtype=torch.float32)
    mask = torch.zeros(2, 10, 10, dtype=torch.int32)
    mask[0, 2:5, 2:5] = 1
    mask[1, 3:6, 3:6] = 1

    node = LabelCropPad(canvas_size=16, label_selection="first")
    out = node(cube=cube, mask=mask)

    for port_name in node.OUTPUT_SPECS:
        assert port_name in out
    assert out["crop"].dtype == torch.float32
    assert out["crop"].shape == (2, 16, 16, 4)


def test_crop_pad_reload_smoke(tmp_path):
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    pipe = CuvisPipeline("crop_pad_reload_test")
    src = _CubeMaskSource(name="src")
    node = LabelCropPad(name="crop_pad", canvas_size=16, label_selection="first")
    pipe.connect(src.outputs.cube, node.inputs.cube)
    pipe.connect(src.outputs.mask, node.inputs.mask)

    before = pipe.forward(batch={})

    save_path = tmp_path / "crop_pad_pipeline.yaml"
    pipe.save_to_file(str(save_path))
    rebuilt = PipelineBuilder().build_from_config(str(save_path))
    after = rebuilt.forward(batch={})

    assert torch.equal(
        before[(node.name, "crop")],
        after[(node.name, "crop")],
    )


def test_crop_pad_export_smoke():
    cube = torch.zeros(1, 8, 10, 2, dtype=torch.float32)
    cube[0, 1:4, 2:7, :] = 1.0
    mask = torch.zeros(1, 8, 10, dtype=torch.int32)
    mask[0, 1:4, 2:7] = 1
    node = LabelCropPad(canvas_size=16, label_selection="first").eval()

    class _Probe(torch.nn.Module):
        def __init__(self, inner: LabelCropPad):
            super().__init__()
            self.inner = inner

        def forward(self, cube: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.inner(cube=cube, mask=mask)["crop"]

    program = torch.export.export(_Probe(node), (cube, mask))

    other_cube = torch.zeros(1, 8, 10, 2, dtype=torch.float32)
    other_cube[0, 0:2, 0:3, :] = 2.0
    other_mask = torch.zeros(1, 8, 10, dtype=torch.int32)
    other_mask[0, 0:2, 0:3] = 1
    traced = program.module()(other_cube, other_mask)
    expected = node(cube=other_cube, mask=other_mask)["crop"]
    assert torch.allclose(traced, expected)


# ----- LabelCropPad: oversized bounding box (bbox larger than canvas) ---


def test_crop_pad_oversized_bbox_centers_surviving_pixel():
    # Label 1's two extreme pixels sit in opposite corners of a 20x20 frame,
    # inflating its pre-crop bbox to the full frame; with canvas_size=8 the
    # centre crop keeps only rows/cols 6..13. A third label-1 pixel at (6, 6)
    # sits just inside that retained window and is the only one that
    # survives. Presence and centring must be recomputed from what actually
    # survives the crop, not from the pre-crop (now stale) bbox: the
    # surviving single pixel is its own bbox, so it must land exactly on the
    # canvas centre.
    h, w, canvas = 20, 20, 8
    cube = torch.zeros(1, h, w, 1, dtype=torch.float32)
    mask = torch.zeros(1, h, w, dtype=torch.int32)
    mask[0, 0, 0] = 1
    mask[0, 19, 19] = 1
    mask[0, 6, 6] = 1
    cube[0, 6, 6, 0] = 42.0

    node = LabelCropPad(canvas_size=canvas, label_selection="first")
    out = node(cube=cube, mask=mask)
    crop = out["crop"][0, :, :, 0]

    assert crop.sum().item() == pytest.approx(42.0)
    nonzero = (crop != 0).nonzero(as_tuple=False)
    assert nonzero.shape[0] == 1
    center = (canvas - 1) // 2
    assert nonzero[0].tolist() == [center, center]


def test_crop_pad_oversized_bbox_label_entirely_outside_crop_window_yields_blank():
    # Label 1's only pixel sits outside the retained centre-crop window
    # entirely: the pre-crop bbox says the label is present, but nothing of
    # it survives the crop, so the output must be an all-zero canvas rather
    # than a bogus non-zero placement derived from the stale pre-crop bbox.
    h, w, canvas = 20, 20, 8
    cube = torch.zeros(1, h, w, 1, dtype=torch.float32)
    mask = torch.zeros(1, h, w, dtype=torch.int32)
    mask[0, 0, 0] = 1
    cube[0, 0, 0, 0] = 5.0

    node = LabelCropPad(canvas_size=canvas, label_selection="first")
    out = node(cube=cube, mask=mask)

    assert torch.equal(out["crop"], torch.zeros(1, canvas, canvas, 1, dtype=torch.float32))


def test_crop_pad_oversized_bbox_shape_contract():
    # The oversized branch (h > canvas_size or w > canvas_size) must still
    # honor the fixed output shape contract.
    h, w, canvas = 20, 24, 8
    cube = torch.rand(2, h, w, 3, dtype=torch.float32)
    mask = torch.zeros(2, h, w, dtype=torch.int32)
    mask[0, 8:12, 8:12] = 1
    mask[1, 9:11, 10:14] = 1

    node = LabelCropPad(canvas_size=canvas, label_selection="first")
    out = node(cube=cube, mask=mask)

    assert out["crop"].shape == (2, canvas, canvas, 3)
    assert out["crop"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_crop_pad_follows_the_cube_device():
    # The SAM3 node hands its mask over on the cpu whatever device the pipeline
    # runs on; the crop must land on the cube device with the same values.
    cube = torch.zeros(1, 8, 10, 2, dtype=torch.float32)
    cube[0, 1:4, 2:7, :] = 1.0
    mask = torch.zeros(1, 8, 10, dtype=torch.int32)
    mask[0, 1:4, 2:7] = 1
    node = LabelCropPad(canvas_size=16, label_selection="first")
    expected = node(cube=cube, mask=mask)["crop"]

    out = node(cube=cube.cuda(), mask=mask)["crop"]
    assert out.device.type == "cuda"
    assert torch.equal(out.cpu(), expected)


# ----- LabelKeep -----------------------------------------------------------


def _three_object_mask() -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(1, 6, 6, dtype=torch.int32)
    mask[0, 0:2, 0:2] = 1
    mask[0, 2:4, 2:4] = 2
    mask[0, 4:6, 4:6] = 7
    object_ids = torch.tensor([[1, 2, 7]], dtype=torch.int64)
    return mask, object_ids


def test_label_keep_zeroes_rejected_labels_and_reports_kept():
    mask, object_ids = _three_object_mask()
    decisions = torch.tensor([True, False, True]).reshape(3, 1, 1, 1)
    out = LabelKeep(max_labels=16)(mask=mask, object_ids=object_ids, decisions=decisions)

    expected = mask.clone()
    expected[mask == 2] = 0
    assert torch.equal(out["mask"], expected)
    assert out["kept"].tolist() == [[1, 0, 1]]


def test_label_keep_drops_label_at_or_above_capacity():
    mask, object_ids = _three_object_mask()
    decisions = torch.ones(3, 1, 1, 1, dtype=torch.bool)
    out = LabelKeep(max_labels=7)(mask=mask, object_ids=object_ids, decisions=decisions)

    assert (out["mask"][mask == 7] == 0).all()
    assert (out["mask"][mask == 1] == 1).all()
    assert out["kept"].tolist() == [[1, 1, 0]]


def test_label_keep_without_objects_returns_empty_mask():
    mask, _ = _three_object_mask()
    out = LabelKeep(max_labels=16)(
        mask=mask,
        object_ids=torch.zeros(1, 0, dtype=torch.int64),
        decisions=torch.zeros(0, 1, 1, 1, dtype=torch.bool),
    )
    assert torch.equal(out["mask"], torch.zeros_like(mask))
    assert out["kept"].shape == (1, 0)


def test_label_keep_port_contract():
    mask, object_ids = _three_object_mask()
    decisions = torch.ones(3, 1, 1, 1, dtype=torch.bool)
    node = LabelKeep(max_labels=16)
    out = node(mask=mask, object_ids=object_ids, decisions=decisions)
    assert out["mask"].shape == mask.shape
    assert out["mask"].dtype == torch.int32
    assert out["kept"].shape == (1, 3)
    assert out["kept"].dtype == torch.int32
    assert set(out) == set(node.OUTPUT_SPECS)


class _KeepSource(Node):
    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.MASK})
    OUTPUT_SPECS = {
        "mask": PortSpec(dtype=torch.int32, shape=(1, -1, -1), description="mask"),
        "object_ids": PortSpec(dtype=torch.int64, shape=(1, -1), description="ids"),
        "decisions": PortSpec(dtype=torch.bool, shape=(-1, 1, 1, 1), description="keep"),
    }

    def forward(self, **_):
        mask, object_ids = _three_object_mask()
        return {
            "mask": mask,
            "object_ids": object_ids,
            "decisions": torch.tensor([True, False, True]).reshape(3, 1, 1, 1),
        }


def test_label_keep_reload_smoke(tmp_path):
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    pipe = CuvisPipeline("label_keep_reload_test")
    src = _KeepSource(name="src")
    node = LabelKeep(name="label_keep", max_labels=16)
    pipe.connect(
        (src.outputs.mask, node.inputs.mask),
        (src.outputs.object_ids, node.inputs.object_ids),
        (src.outputs.decisions, node.inputs.decisions),
    )
    before = pipe.forward(batch={})
    save_path = tmp_path / "label_keep_pipeline.yaml"
    pipe.save_to_file(str(save_path))
    after = PipelineBuilder().build_from_config(str(save_path)).forward(batch={})
    assert torch.equal(before[(node.name, "mask")], after[(node.name, "mask")])


class _KeepProbe(torch.nn.Module):
    def __init__(self, inner: LabelKeep) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, mask, object_ids, decisions):
        out = self.inner(mask=mask, object_ids=object_ids, decisions=decisions)
        return out["mask"], out["kept"]


def test_label_keep_export_smoke():
    node = LabelKeep(max_labels=16).eval()
    mask, object_ids = _three_object_mask()
    decisions = torch.tensor([True, False, True]).reshape(3, 1, 1, 1)
    program = torch.export.export(_KeepProbe(node), (mask, object_ids, decisions))

    other_mask = mask.flip(-1).contiguous()
    other_decisions = torch.tensor([False, True, True]).reshape(3, 1, 1, 1)
    traced_mask, traced_kept = program.module()(other_mask, object_ids, other_decisions)
    eager = node(mask=other_mask, object_ids=object_ids, decisions=other_decisions)
    assert torch.equal(traced_mask, eager["mask"])
    assert torch.equal(traced_kept, eager["kept"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_label_keep_follows_the_mask_device():
    # Mask and ids come from the SAM3 node on the cpu, the decisions from the
    # threshold node on the pipeline device; the mask device wins either way.
    mask, object_ids = _three_object_mask()
    decisions = torch.tensor([True, False, True]).reshape(3, 1, 1, 1)
    node = LabelKeep(max_labels=16)
    expected = node(mask=mask, object_ids=object_ids, decisions=decisions)

    cpu_mask = node(mask=mask, object_ids=object_ids, decisions=decisions.cuda())
    assert cpu_mask["mask"].device.type == "cpu"
    assert torch.equal(cpu_mask["mask"], expected["mask"])
    assert torch.equal(cpu_mask["kept"], expected["kept"])

    cuda_mask = node(mask=mask.cuda(), object_ids=object_ids, decisions=decisions)
    assert cuda_mask["mask"].device.type == "cuda"
    assert torch.equal(cuda_mask["mask"].cpu(), expected["mask"])
    assert torch.equal(cuda_mask["kept"].cpu(), expected["kept"])
