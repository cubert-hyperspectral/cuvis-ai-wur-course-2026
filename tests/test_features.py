"""Tests for ResNet18Features (cuvis_ai_wur_course.node.features).

The parity check compares the node's own ImageNet-preprocessing-plus-backbone
forward against a bare torchvision ``resnet18`` fed the exact same
preprocessed tensor, so it isolates the backbone computation from the
PIL-versus-torch resize kernel difference (see the module docstring for why
that difference is a documented deviation, not a numeric target).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torchvision import models

from cuvis_ai_wur_course.node.features import ResNet18Features

pytestmark = pytest.mark.unit


class _ImageSource(Node):
    """Minimal source node emitting a fixed image, for reload smoke tests.

    Defined at module scope (not inside a test function) so the class
    resolves as ``<module>.<ClassName>`` after a pipeline save/reload round
    trip (the class-location trap in node-anatomy.md).
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.IMAGE})

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"image": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 3))}

    def forward(self, **_):
        return {"image": _tiny_image(b=1, h=256, w=256)}


def _tiny_image(b: int = 2, h: int = 256, w: int = 256) -> torch.Tensor:
    """Deterministic BHWC float32 image in [0, 1]."""
    torch.manual_seed(0)
    return torch.rand(b, h, w, 3, dtype=torch.float32)


# ----- 1. golden-reference / parity ------------------------------------


def test_matches_plain_torchvision_backbone_on_identical_preprocessing():
    torch.manual_seed(42)
    node = ResNet18Features(weights=None)
    node.eval()

    reference = models.resnet18(weights=None)
    # node.backbone.fc is Identity(); only copy the shared trunk's weights.
    reference.load_state_dict(node.backbone.state_dict(), strict=False)
    reference_extractor = torch.nn.Sequential(*list(reference.children())[:-1]).eval()

    image = _tiny_image(b=2, h=256, w=256)

    # Reproduce the node's own preprocessing to feed the reference model the
    # exact same tensor (this is what makes it a parity test rather than an
    # end-to-end one: the resize kernel is identical on both sides).
    preprocessed = node._preprocess(image)
    with torch.no_grad():
        expected = reference_extractor(preprocessed).view(preprocessed.shape[0], -1)

    out = node(image=image)["features"]
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_preprocessing_matches_hand_built_resize_and_normalize():
    node = ResNet18Features(weights=None)
    image = _tiny_image(b=1, h=256, w=256)

    preprocessed = node._preprocess(image)
    assert preprocessed.shape == (1, 3, 224, 224)

    # Hand-built reference: BHWC -> BCHW, resize to 256 on the shorter edge
    # (a no-op here since the input is already 256x256 square), center crop
    # to 224, then ImageNet mean/std normalization.
    chw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(
        chw, size=(256, 256), mode="bilinear", align_corners=False, antialias=True
    )
    top = (256 - 224) // 2
    left = (256 - 224) // 2
    cropped = resized[:, :, top : top + 224, left : left + 224]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    expected = (cropped - mean) / std

    assert torch.allclose(preprocessed, expected, atol=1e-6)


# ----- 2. port contract --------------------------------------------------


def test_port_contract():
    node = ResNet18Features(weights=None)
    image = _tiny_image(b=3, h=256, w=256)

    out = node(image=image)

    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out
        tensor = out[port_name]
        assert tensor.dtype == spec.dtype
    assert out["features"].shape == (3, 512)


# ----- 3. freeze / unfreeze ----------------------------------------------


def test_features_stay_float32_under_bfloat16_autocast():
    # The SAM3 plugin (cuvis-ai-sam3 0.5.1) leaves bfloat16 autocast switched on after it
    # built its model; the feature port still promises float32 to the next node.
    node = ResNet18Features(weights=None).eval()
    image = _tiny_image(b=1, h=256, w=256)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = node(image=image)
    assert out["features"].dtype == torch.float32


def test_freeze_unfreeze_cycle_toggles_backbone_grad():
    node = ResNet18Features(weights=None)

    # Default state: frozen (pretrained-style feature extractor).
    assert all(not p.requires_grad for p in node.backbone.parameters())

    node.unfreeze()
    assert all(p.requires_grad for p in node.backbone.parameters())

    node.freeze()
    assert all(not p.requires_grad for p in node.backbone.parameters())


def test_freeze_unfreeze_preserves_weight_values():
    node = ResNet18Features(weights=None)
    before = {name: p.clone() for name, p in node.backbone.named_parameters()}

    node.unfreeze()
    node.freeze()

    for name, p in node.backbone.named_parameters():
        assert torch.allclose(p, before[name])


# ----- 4. export smoke ----------------------------------------------------


def test_export_smoke():
    node = ResNet18Features(weights=None).eval()
    image = _tiny_image(b=2, h=256, w=256)

    class _Probe(torch.nn.Module):
        def __init__(self, inner: ResNet18Features):
            super().__init__()
            self.inner = inner

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.inner(image=image)["features"]

    program = torch.export.export(_Probe(node), (image,))

    other = _tiny_image(b=2, h=256, w=256) * 0.5
    traced = program.module()(other)
    expected = node(image=other)["features"]
    assert torch.allclose(traced, expected, atol=1e-5)


# ----- 5. reload smoke -----------------------------------------------------


def test_reload_smoke(tmp_path):
    """Structural reload: the saved config resolves and the graph rewires.

    Weight identity is checked separately by loading the saved state_dict
    back (below), since ``build_from_config`` alone only reconstructs the
    graph from the YAML and does not load the co-saved ``.pt`` weights
    (that is a separate restore step, out of scope for a node-level test).
    """
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    pipe = CuvisPipeline("resnet_features_reload_test")
    src = _ImageSource(name="src")
    node = ResNet18Features(name="features", weights=None)
    pipe.connect(src.outputs.image, node.inputs.image)

    before = pipe.forward(batch={})

    save_path = tmp_path / "resnet_features_pipeline.yaml"
    pipe.save_to_file(str(save_path))
    rebuilt = PipelineBuilder().build_from_config(str(save_path))
    after = rebuilt.forward(batch={})

    assert before[(node.name, "features")].shape == after[(node.name, "features")].shape

    weights_path = save_path.with_suffix(".pt")
    checkpoint = torch.load(weights_path, weights_only=True)
    rebuilt_node = next(n for n in rebuilt.nodes if n.name == node.name)
    rebuilt_node.load_state_dict(checkpoint["state_dict"][node.name])

    after_with_weights = rebuilt.forward(batch={})
    assert torch.allclose(
        before[(node.name, "features")],
        after_with_weights[(node.name, "features")],
        atol=1e-6,
    )
