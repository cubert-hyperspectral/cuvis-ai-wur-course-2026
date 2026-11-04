"""The notebook diagram helper labels every card with the node's name as well as its class."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from cuvis_ai.node.pretreatments.scaling import MeanCenter, UnitVarianceScaling
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import utils  # noqa: E402


def test_show_pipeline_writes_the_node_names_under_the_class_titles() -> None:
    graphviz = pytest.importorskip("graphviz")
    center, scale = MeanCenter(name="center_pork"), UnitVarianceScaling(name="scale_pork")
    pipeline = CuvisPipeline("named")
    pipeline.connect((center.outputs.cube, scale.inputs.cube))
    try:
        shown = utils.show_pipeline(pipeline)
    except graphviz.ExecutableNotFound:
        pytest.skip("the Graphviz dot binary is not installed")
    assert "<svg" in shown.data
    for text in ("MeanCenter", "center_pork", "UnitVarianceScaling", "scale_pork"):
        assert text in _visible_text(shown.data)
    # The bare repr is what the helper improves on: it draws the class titles only (the name
    # is still in the SVG as the node's id, which no reader sees).
    assert "center_pork" not in _visible_text(pipeline._repr_html_())


def _visible_text(svg: str) -> set[str]:
    return set(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
