"""Round-trip every pipeline YAML under ``pipelines/`` through the builder.

Each YAML is loaded, built with ``PipelineBuilder`` (which validates every port
connection), saved again and rebuilt from the saved copy. Nodes that open a file at
construction get a placeholder written into ``tmp_path`` so the check runs without the
course data; the pipelines that load large pretrained models are marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from cuvis_ai_core.pipeline.factory import PipelineBuilder

from cuvis_ai_wur_course.node.models import SpectrumRegressor

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
PIPELINES = sorted((REPO / "pipelines").rglob("*.yaml"))
HEAVY = {"kiwi_sam3_spectral_match", "yolo_rgb", "resnet18_features", "unet_ensemble_maps"}
MEAN_SPECTRA_CHANNELS = 124


def _node_names(pipeline) -> set[str]:
    nodes = pipeline.nodes
    members = nodes.values() if isinstance(nodes, dict) else nodes
    return {getattr(n, "name", str(n)) for n in members}


def _placeholder_checkpoint(arch: str, path: Path) -> None:
    node = SpectrumRegressor(arch=arch, input_channels=MEAN_SPECTRA_CHANNELS)
    renamed = {
        "x_mean": "standardizer.mean",
        "x_std": "standardizer.std",
        "y_mean": "destandardizer.mean",
        "y_std": "destandardizer.std",
    }
    torch.save({renamed.get(k, k): v for k, v in node.state_dict().items()}, path)


def _with_placeholders(doc: dict, tmp_path: Path) -> dict:
    """Point file-reading hparams at placeholders and disable weight downloads."""
    for node in doc["nodes"]:
        hparams = node.get("hparams") or {}
        if "weights_path" in hparams and hparams["weights_path"]:
            target = tmp_path / f"{node['name']}_best_weights.pt"
            _placeholder_checkpoint(hparams["arch"], target)
            hparams["weights_path"] = str(target)
        if "file_path" in hparams:
            target = tmp_path / f"{node['name']}.npy"
            np.save(target, np.ones((1, 1, 1, 4), dtype=np.float32))
            hparams["file_path"] = str(target)
        if hparams.get("weights") in ("hf", "imagenet"):
            hparams["weights"] = None
    return doc


def _round_trip(yaml_path: Path, tmp_path: Path) -> None:
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert doc["plugins"], f"{yaml_path.name} declares no plugins"
    assert "cuvis_ai_wur_course" in doc["plugins"] or yaml_path.stem == "yolo_rgb"
    doc = _with_placeholders(doc, tmp_path)
    first = tmp_path / yaml_path.name
    first.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    pipeline = PipelineBuilder().build_from_config(str(first))
    second = tmp_path / f"roundtrip_{yaml_path.name}"
    pipeline.save_to_file(str(second))
    rebuilt = PipelineBuilder().build_from_config(str(second))

    assert _node_names(rebuilt) == _node_names(pipeline)
    assert len(_node_names(rebuilt)) == len(doc["nodes"])


@pytest.mark.parametrize(
    "yaml_path",
    [p for p in PIPELINES if p.stem not in HEAVY],
    ids=lambda p: p.stem,
)
def test_pipeline_round_trip(yaml_path: Path, tmp_path: Path) -> None:
    _round_trip(yaml_path, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize(
    "yaml_path",
    [p for p in PIPELINES if p.stem in HEAVY],
    ids=lambda p: p.stem,
)
def test_heavy_pipeline_round_trip(yaml_path: Path, tmp_path: Path) -> None:
    _round_trip(yaml_path, tmp_path)


def test_every_pipeline_names_the_course_plugin_catalog() -> None:
    assert PIPELINES, "no pipeline YAML found"
    for yaml_path in PIPELINES:
        doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert doc["metadata"]["name"] == yaml_path.stem
        assert doc["plugins"][0] == "cuvis_ai_builtin"


@pytest.mark.parametrize("pipeline_name", ["unet_ensemble_maps", "pls_pixel_map"])
def test_02_pipelines_feed_the_frame_mask_straight_to_the_map_statistics(
    pipeline_name: str,
) -> None:
    """The course ships slice masks already on the U-Net output grid, so the 02
    pipelines must not resize them with ``MaskToMapResolution`` any more."""
    yaml_path = REPO / "pipelines" / "02_unet_chemical_maps" / f"{pipeline_name}.yaml"
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    class_names = {node["class_name"] for node in doc["nodes"]}
    assert "cuvis_ai_wur_course.node.maps.MaskToMapResolution" not in class_names

    mask_targets = {
        conn["target"] for conn in doc["connections"] if conn["source"] == "frame.outputs.mask"
    }
    assert any(target.endswith("stats.inputs.mask") for target in mask_targets)
