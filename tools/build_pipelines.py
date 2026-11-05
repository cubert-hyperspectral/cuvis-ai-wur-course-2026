"""Build the course pipelines in memory and write them as YAML under ``pipelines/``.

The YAML files are the deliverable; this script is how they are regenerated. Every
pipeline is wired with ``CuvisPipeline.connect`` so the port validator checks each
connection, saved with ``save_to_file``, rebuilt from the YAML with ``PipelineBuilder``
as a round-trip check, and given its ``plugins`` list and metadata.

Nodes that read a file or weights at construction (the kiwi reference spectrum, the
course model checkpoints) need those files to exist. When the course data has not been
converted yet, the script writes small placeholders under ``data/course`` so the wiring
can be validated; ``prepare-course-data`` and the real checkpoints replace them.

Run from the repository root:

    uv run python tools/build_pipelines.py [--skip-models]

``--skip-models`` leaves out the pipelines whose construction loads large pretrained
models (SAM3, YOLO, the U-Net ensemble, ImageNet ResNet-18).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from cuvis_ai.node.channel_selector import FixedWavelengthSelector
from cuvis_ai.node.deciders.binary_decider import ScoreThreshold
from cuvis_ai.node.metrics import DistinctLabelCount, ExplainedVarianceMetric
from cuvis_ai.node.numpy_file import NpyReader
from cuvis_ai.node.pretreatments.logarithm import Logarithm
from cuvis_ai.node.pretreatments.standardize import Standardize
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.spectral_extractor import SignaturesToReferences, SpectralSignatureExtractor
from cuvis_ai.node.video import VideoFrameNode
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import PipelineMetadata

from cuvis_ai_wur_course.node.features import ResNet18Features
from cuvis_ai_wur_course.node.fusion import ChannelSlice, PredictionMean, SpatialMean
from cuvis_ai_wur_course.node.kiwi import LabelBBoxFilter, LabelCropPad, LabelKeep
from cuvis_ai_wur_course.node.losses import RegressionMSELoss
from cuvis_ai_wur_course.node.maps import (
    CenterCrop,
    ChemicalMapStatistics,
    Dequantize,
    GroupWeightedMean,
)
from cuvis_ai_wur_course.node.metrics import RegressionMetrics
from cuvis_ai_wur_course.node.models import SpectrumRegressor
from cuvis_ai_wur_course.node.pca import DatasetPCA
from cuvis_ai_wur_course.node.pretreatment import SavitzkyGolayInterp, SpectralBinning
from cuvis_ai_wur_course.node.regression import PLSRegression
from cuvis_ai_wur_course.node.source import CourseFrame
from cuvis_ai_wur_course.node.unet import UNetChemicalMapEnsemble

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "pipelines"
DATA = REPO / "data" / "course"

KIWI_REFERENCE = "data/course/kiwi/kiwi_reference.npy"
MLP_WEIGHTS = "data/course/models/mlp/best_weights.pt"
RESNET1D_WEIGHTS = "data/course/models/resnet1d/best_weights.pt"

BASE_PLUGINS = ["cuvis_ai_builtin", "cuvis_ai_wur_course"]
AUTHOR = "Cubert GmbH"

# Course constants.
PSEUDO_RGB_NM = (575.0, 535.0, 445.0)
KIWI_ANGLE_CUTOFF = 0.035
PORK_SPATIAL, PORK_SPECTRAL = 512, 300
MEAN_SPECTRA_CHANNELS = 124
BINNED_CHANNELS = 62
MAX_COMPONENTS = 20
PLS_MAP_CROP = (1992, 904)


def _ensure_placeholders() -> None:
    """Write small stand-ins for the course files that nodes open at construction."""
    reference = REPO / KIWI_REFERENCE
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        np.save(reference, np.ones((1, 1, 1, 1), dtype=np.float32))
        print(f"placeholder written: {reference}")
    for arch, rel in (("mlp", MLP_WEIGHTS), ("resnet1d", RESNET1D_WEIGHTS)):
        path = REPO / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        node = SpectrumRegressor(arch=arch, input_channels=MEAN_SPECTRA_CHANNELS)
        renamed = {
            "x_mean": "standardizer.mean",
            "x_std": "standardizer.std",
            "y_mean": "destandardizer.mean",
            "y_std": "destandardizer.std",
        }
        state = {renamed.get(k, k): v for k, v in node.state_dict().items()}
        torch.save(state, path)
        print(f"placeholder written: {path}")


def _canonical_class_name(class_name: str) -> str:
    """Use the package-level class path the sam3 and ultralytics manifests list.

    ``save_to_file`` records the defining module (``cuvis_ai_sam3.node.sam3_segment_everything.
    SAM3SegmentEverything``); the plugin resolver matches class names against the manifest
    entries literally, which are ``cuvis_ai_sam3.node.SAM3SegmentEverything``.
    """
    for package in ("cuvis_ai_sam3", "cuvis_ai_ultralytics"):
        prefix = f"{package}.node."
        if class_name.startswith(prefix):
            return prefix + class_name.rsplit(".", 1)[1]
    return class_name


def _emit(
    pipeline: CuvisPipeline,
    relative: str,
    plugins: list[str],
    name: str,
    description: str,
    tags: list[str],
    overrides: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Save, post-process (plugins, hparam overrides), round-trip and report one pipeline."""
    target = PIPELINES / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / target.name
        pipeline.save_to_file(
            str(scratch),
            metadata=PipelineMetadata(name=name, description=description, tags=tags, author=AUTHOR),
        )
        doc = yaml.safe_load(scratch.read_text(encoding="utf-8"))
    doc["plugins"] = list(plugins)
    for node in doc["nodes"]:
        node["class_name"] = _canonical_class_name(node["class_name"])
        for key, value in (overrides or {}).get(node["name"], {}).items():
            node.setdefault("hparams", {})[key] = value
    ordered = {k: doc[k] for k in ("metadata", "plugins", "nodes", "connections") if k in doc}
    target.write_text(
        yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    shutil.rmtree(target.with_suffix(".pt"), ignore_errors=True)
    PipelineBuilder().build_from_config(str(target))
    n_nodes, n_edges = len(doc["nodes"]), len(doc["connections"])
    print(f"ok  {target.relative_to(REPO)}  ({n_nodes} nodes, {n_edges} connections)")
    return target


# ----------------------------------------------------------------------------
# Notebook 01: using_pretrained_ai
# ----------------------------------------------------------------------------


def build_kiwi_sam3_spectral_match() -> None:
    pipe = CuvisPipeline("kiwi_sam3_spectral_match")
    from cuvis_ai_sam3.node import SAM3SegmentEverything

    frame = CourseFrame(name="frame")
    pseudo_rgb = FixedWavelengthSelector(name="pseudo_rgb", target_wavelengths=PSEUDO_RGB_NM)
    sam3 = SAM3SegmentEverything(name="sam3")
    size_filter = LabelBBoxFilter(
        name="size_filter", min_height=50, max_height=120, min_width=150, max_width=210
    )
    count_size = DistinctLabelCount(name="count_size_filter")
    object_spectra = SpectralSignatureExtractor(
        name="object_spectra", trim_fraction=0.0, min_mask_pixels=1
    )
    object_references = SignaturesToReferences(name="object_references")
    kiwi_reference = NpyReader(name="kiwi_reference", file_path=KIWI_REFERENCE)
    kiwi_angle = SpectralAngleMapper(name="kiwi_angle")
    kiwi_decision = ScoreThreshold(name="kiwi_decision", threshold=KIWI_ANGLE_CUTOFF, keep="below")
    spectral_filter = LabelKeep(name="spectral_filter")
    count_spectral = DistinctLabelCount(name="count_spectral_filter")
    kiwi_crop = LabelCropPad(name="kiwi_crop", canvas_size=256, label_selection="first")
    crop_rgb = FixedWavelengthSelector(name="crop_rgb", target_wavelengths=PSEUDO_RGB_NM)
    features = ResNet18Features(name="features", weights="imagenet")

    pipe.connect(
        (frame.outputs.cube, pseudo_rgb.inputs.cube),
        (frame.outputs.wavelengths, pseudo_rgb.inputs.wavelengths),
        (pseudo_rgb.outputs.rgb_image, sam3.inputs.rgb_image),
        # Branch A: the course's size filter (height 50 to 120, width 150 to 210 px).
        (sam3.outputs.mask, size_filter.inputs.mask),
        (size_filter.outputs.mask, count_size.inputs.mask),
        # Branch B: the course's spectral-angle filter against the saved kiwi spectrum.
        (frame.outputs.cube, object_spectra.inputs.cube),
        (sam3.outputs.mask, object_spectra.inputs.mask),
        (sam3.outputs.object_ids, object_spectra.inputs.object_ids),
        (frame.outputs.wavelengths, object_spectra.inputs.wavelengths),
        (object_spectra.outputs.signatures, object_references.inputs.signatures),
        (object_references.outputs.spectral_signature, kiwi_angle.inputs.cube),
        (kiwi_reference.outputs.data, kiwi_angle.inputs.spectral_signature),
        (kiwi_angle.outputs.best_scores, kiwi_decision.inputs.scores),
        (sam3.outputs.mask, spectral_filter.inputs.mask),
        (sam3.outputs.object_ids, spectral_filter.inputs.object_ids),
        (kiwi_decision.outputs.decisions, spectral_filter.inputs.decisions),
        (spectral_filter.outputs.mask, count_spectral.inputs.mask),
        # One kiwi out of branch B, padded to 256 x 256, rendered and embedded.
        (frame.outputs.cube, kiwi_crop.inputs.cube),
        (spectral_filter.outputs.mask, kiwi_crop.inputs.mask),
        (kiwi_crop.outputs.crop, crop_rgb.inputs.cube),
        (frame.outputs.wavelengths, crop_rgb.inputs.wavelengths),
        (crop_rgb.outputs.rgb_image, features.inputs.image),
    )
    _emit(
        pipe,
        "01_pretrained_ai/kiwi_sam3_spectral_match.yaml",
        [*BASE_PLUGINS, "sam3"],
        "kiwi_sam3_spectral_match",
        "Kiwi cube to pseudo-RGB, SAM3 segment-everything, the course's size filter and "
        "spectral-angle filter side by side, one kiwi cropped to 256 px and embedded with "
        "ResNet-18.",
        ["course", "sam3", "spectral-match", "resnet18"],
    )


def build_yolo_rgb() -> None:
    from cuvis_ai_ultralytics.node import YOLO26Detection, YOLOPostprocess, YOLOPreprocess

    pipe = CuvisPipeline("yolo_rgb")
    frame = VideoFrameNode(name="frame")
    preprocess = YOLOPreprocess(name="preprocess", imgsz=640)
    detector = YOLO26Detection(name="detector", model_path="yolov8n.pt")
    postprocess = YOLOPostprocess(name="postprocess", confidence_threshold=0.25)
    pipe.connect(
        (frame.outputs.rgb_image, preprocess.inputs.rgb_image),
        (preprocess.outputs.preprocessed, detector.inputs.preprocessed),
        (detector.outputs.raw_preds, postprocess.inputs.raw_preds),
        (preprocess.outputs.model_input_hw, postprocess.inputs.model_input_hw),
        (preprocess.outputs.orig_hw, postprocess.inputs.orig_hw),
    )
    _emit(
        pipe,
        "01_pretrained_ai/yolo_rgb.yaml",
        ["cuvis_ai_builtin", "ultralytics"],
        "yolo_rgb",
        "An RGB frame through the ultralytics nodes with the course's yolov8n weights at imgsz 640 "
        "(the detector node loads any ultralytics weight file).",
        ["course", "yolo", "detection"],
    )


def build_resnet18_features() -> None:
    pipe = CuvisPipeline("resnet18_features")
    frame = VideoFrameNode(name="frame")
    features = ResNet18Features(name="features", weights="imagenet")
    pipe.connect((frame.outputs.rgb_image, features.inputs.image))
    _emit(
        pipe,
        "01_pretrained_ai/resnet18_features.yaml",
        BASE_PLUGINS,
        "resnet18_features",
        "An RGB frame through ImageNet ResNet-18 without its classifier: one 512-vector.",
        ["course", "resnet18", "features"],
    )


def _regression_head(
    pipe: CuvisPipeline, frame: CourseFrame, pls: PLSRegression, tag: str
) -> SpatialMean:
    mean = SpatialMean(name=f"mean_{tag}")
    metrics = RegressionMetrics(name=f"metrics_{tag}")
    pipe.connect(
        (pls.outputs.prediction, mean.inputs.prediction),
        (mean.outputs.prediction, metrics.inputs.prediction),
        (frame.outputs.target, metrics.inputs.target),
    )
    return mean


def build_pork_pls_fusion() -> None:
    pipe = CuvisPipeline("pork_pls_fusion")
    frame = CourseFrame(name="frame")
    standardize = Standardize(name="standardize", axis="channel")
    spatial_block = ChannelSlice(name="spatial_block", start=0, stop=PORK_SPATIAL)
    spectral_block = ChannelSlice(
        name="spectral_block", start=PORK_SPATIAL, stop=PORK_SPATIAL + PORK_SPECTRAL
    )
    pls_kwargs = {
        "max_components": MAX_COMPONENTS,
        "holdout_folds": "each",
        "refit_on_holdout": True,
    }
    pls_spatial = PLSRegression(name="pls_spatial", input_channels=PORK_SPATIAL, **pls_kwargs)
    pls_spectral = PLSRegression(name="pls_spectral", input_channels=PORK_SPECTRAL, **pls_kwargs)
    pls_stacked = PLSRegression(
        name="pls_stacked", input_channels=PORK_SPATIAL + PORK_SPECTRAL, **pls_kwargs
    )
    pipe.connect(
        (frame.outputs.cube, standardize.inputs.cube),
        (standardize.outputs.cube, spatial_block.inputs.cube),
        (standardize.outputs.cube, spectral_block.inputs.cube),
        # Connected on purpose: an unconnected optional `wavelengths` port would otherwise be
        # auto-filled from the batch's [B, C] float wavelengths and fail its int32 [C] spec.
        (frame.outputs.wavelengths, spatial_block.inputs.wavelengths),
        (frame.outputs.wavelengths, spectral_block.inputs.wavelengths),
        (spatial_block.outputs.cube, pls_spatial.inputs.cube),
        (spectral_block.outputs.cube, pls_spectral.inputs.cube),
        (standardize.outputs.cube, pls_stacked.inputs.cube),
    )
    for pls in (pls_spatial, pls_spectral, pls_stacked):
        pipe.connect(
            (frame.outputs.target, pls.inputs.target),
            (frame.outputs.fold, pls.inputs.fold),
        )
    mean_spatial = _regression_head(pipe, frame, pls_spatial, "spatial")
    mean_spectral = _regression_head(pipe, frame, pls_spectral, "spectral")
    _regression_head(pipe, frame, pls_stacked, "stacked")
    fusion = PredictionMean(name="fusion")
    metrics_fusion = RegressionMetrics(name="metrics_fusion")
    pipe.connect(
        (mean_spatial.outputs.prediction, fusion.inputs.prediction),
        (mean_spectral.outputs.prediction, fusion.inputs.prediction),
        (fusion.outputs.prediction, metrics_fusion.inputs.prediction),
        (frame.outputs.target, metrics_fusion.inputs.target),
    )
    _emit(
        pipe,
        "01_pretrained_ai/pork_pls_fusion.yaml",
        BASE_PLUGINS,
        "pork_pls_fusion",
        "Pork belly fat from the 512 ResNet-18 spatial and 300 spectral features: three PLS "
        "models (spatial, spectral, stacked) with five-fold component selection, and the "
        "average of the spatial and spectral predictions.",
        ["course", "pls", "fusion"],
    )


# ----------------------------------------------------------------------------
# Notebook 02: end2end_unet_chemical_maps
# ----------------------------------------------------------------------------


def build_unet_ensemble_maps(unet_weights: str | None) -> None:
    pipe = CuvisPipeline("unet_ensemble_maps")
    frame = CourseFrame(name="frame")
    dequantize = Dequantize(name="dequantize")
    unet = UNetChemicalMapEnsemble(name="unet", n_members=5, weights=unet_weights)
    stats = ChemicalMapStatistics(name="stats", mask_scale=1)
    belly_mean = GroupWeightedMean(name="belly_mean", max_groups=256)
    pipe.connect(
        (frame.outputs.cube, dequantize.inputs.cube),
        (frame.outputs.quant_scale, dequantize.inputs.quant_scale),
        (frame.outputs.quant_bias, dequantize.inputs.quant_bias),
        (dequantize.outputs.cube, unet.inputs.cube),
        (unet.outputs.map, stats.inputs.map),
        (frame.outputs.mask, stats.inputs.mask),
        (unet.outputs.map, belly_mean.inputs.map),
        (frame.outputs.mask, belly_mean.inputs.mask),
        (frame.outputs.group, belly_mean.inputs.group),
    )
    _emit(
        pipe,
        "02_unet_chemical_maps/unet_ensemble_maps.yaml",
        BASE_PLUGINS,
        "unet_ensemble_maps",
        "Dequantised slices through the five-member U-Net ensemble of Engstrom et al.: fat maps, "
        "map statistics against the slice mask the course already ships at the U-Net output "
        "grid, and the mask-weighted belly value.",
        ["course", "unet", "chemical-map"],
        overrides={"unet": {"weights": "hf"}},
    )


def _spectral_pls_chain(
    pipe: CuvisPipeline, upstream, num_channels: int, fit: bool, frame
) -> PLSRegression:
    snv = Standardize(name="snv", axis="spectrum", ddof=0)
    sg = SavitzkyGolayInterp(
        name="sg", num_channels=num_channels, window_length=7, polyorder=2, deriv=2
    )
    pls = PLSRegression(
        name="pls",
        input_channels=num_channels,
        max_components=MAX_COMPONENTS,
        holdout_folds="each",
        refit_on_holdout=True,
    )
    pipe.connect(
        (upstream, snv.inputs.cube),
        (snv.outputs.cube, sg.inputs.cube),
        (sg.outputs.cube, pls.inputs.cube),
    )
    if fit:
        pipe.connect(
            (frame.outputs.target, pls.inputs.target), (frame.outputs.fold, pls.inputs.fold)
        )
    return pls


def build_pls_mean_spectra_fit() -> None:
    pipe = CuvisPipeline("pls_mean_spectra_fit")
    frame = CourseFrame(name="frame")
    binning = SpectralBinning(name="binning", input_channels=MEAN_SPECTRA_CHANNELS, bin_size=2)
    pipe.connect(
        (frame.outputs.cube, binning.inputs.cube),
        (frame.outputs.wavelengths, binning.inputs.wavelengths),
    )
    pls = _spectral_pls_chain(pipe, binning.outputs.cube, BINNED_CHANNELS, fit=True, frame=frame)
    _regression_head(pipe, frame, pls, "pls")
    _emit(
        pipe,
        "02_unet_chemical_maps/pls_mean_spectra_fit.yaml",
        BASE_PLUGINS,
        "pls_mean_spectra_fit",
        "Mean spectra binned to 62 bands, SNV, Savitzky-Golay second derivative (window 7, "
        "order 2, scipy interp edges), PLS with the component count chosen by leave-one-split-out "
        "pooled SSE over the five CV splits.",
        ["course", "pls", "fit"],
    )


def build_pls_pixel_map() -> None:
    pipe = CuvisPipeline("pls_pixel_map")
    frame = CourseFrame(name="frame")
    dequantize = Dequantize(name="dequantize")
    crop = CenterCrop(name="crop", crop_height=PLS_MAP_CROP[0], crop_width=PLS_MAP_CROP[1])
    absorbance = Logarithm(name="absorbance", mode="log10", negate=True, eps=1e-6)
    pipe.connect(
        (frame.outputs.cube, dequantize.inputs.cube),
        (frame.outputs.quant_scale, dequantize.inputs.quant_scale),
        (frame.outputs.quant_bias, dequantize.inputs.quant_bias),
        (dequantize.outputs.cube, crop.inputs.cube),
        (crop.outputs.cube, absorbance.inputs.cube),
    )
    pls = _spectral_pls_chain(
        pipe, absorbance.outputs.cube, BINNED_CHANNELS, fit=False, frame=frame
    )
    stats = ChemicalMapStatistics(name="stats", mask_scale=2)
    pipe.connect(
        (pls.outputs.prediction, stats.inputs.map),
        (frame.outputs.mask, stats.inputs.mask),
    )
    _emit(
        pipe,
        "02_unet_chemical_maps/pls_pixel_map.yaml",
        BASE_PLUGINS,
        "pls_pixel_map",
        "A dequantised slice, centre-cropped, to absorbance, SNV and Savitzky-Golay per pixel, "
        "through the PLS model fitted on the mean spectra (same node name, weights restored by "
        "name), with map statistics against the slice mask the course already ships at the "
        "U-Net output grid, upsampled by two as the course's np.kron does.",
        ["course", "pls", "chemical-map"],
    )


# ----------------------------------------------------------------------------
# Notebook 03: end2end_spectral_vs_spatial_models
# ----------------------------------------------------------------------------


def _model_head(pipe: CuvisPipeline, frame: CourseFrame, model, tag: str) -> SpatialMean:
    mean = SpatialMean(name=f"mean_{tag}")
    metrics = RegressionMetrics(name=f"metrics_{tag}")
    pipe.connect(
        (frame.outputs.cube, model.inputs.cube),
        (model.outputs.prediction, mean.inputs.prediction),
        (mean.outputs.prediction, metrics.inputs.prediction),
        (frame.outputs.target, metrics.inputs.target),
    )
    return mean


def build_spectral_models() -> None:
    pipe = CuvisPipeline("spectral_models")
    frame = CourseFrame(name="frame")
    standardize = Standardize(
        name="standardize", axis="global", ddof=0, eps=1e-6, exclude_folds=[1]
    )
    pca = DatasetPCA(
        name="pca", input_channels=MEAN_SPECTRA_CHANNELS, n_components=3, exclude_folds=[1]
    )
    explained = ExplainedVarianceMetric(name="explained_variance")
    pls = PLSRegression(
        name="pls",
        input_channels=MEAN_SPECTRA_CHANNELS,
        max_components=MAX_COMPONENTS,
        holdout_folds=[1],
        refit_on_holdout=False,
    )
    mlp = SpectrumRegressor(
        name="mlp", arch="mlp", input_channels=MEAN_SPECTRA_CHANNELS, weights_path=MLP_WEIGHTS
    )
    resnet1d = SpectrumRegressor(
        name="resnet1d",
        arch="resnet1d",
        input_channels=MEAN_SPECTRA_CHANNELS,
        weights_path=RESNET1D_WEIGHTS,
    )
    pipe.connect(
        (frame.outputs.cube, standardize.inputs.cube),
        (frame.outputs.cube, pca.inputs.cube),
        (pca.outputs.explained_variance_ratio, explained.inputs.explained_variance_ratio),
        (frame.outputs.target, pls.inputs.target),
        (frame.outputs.fold, pls.inputs.fold),
        (frame.outputs.fold, standardize.inputs.fold),
        (frame.outputs.fold, pca.inputs.fold),
    )
    _model_head(pipe, frame, pls, "pls")
    _model_head(pipe, frame, mlp, "mlp")
    _model_head(pipe, frame, resnet1d, "resnet1d")
    _emit(
        pipe,
        "03_spectral_vs_spatial/spectral_models.yaml",
        BASE_PLUGINS,
        "spectral_models",
        "Mean spectra: the course's global scalar standardisation, a three-component dataset "
        "PCA, PLS selected on hold-out split 1, and the course's MLP and 1D ResNet from their "
        "trained weights, each with pooled regression metrics.",
        ["course", "pls", "mlp", "resnet1d", "pca"],
    )


def build_spectrum_training(arch: str) -> None:
    """A trainable copy of one course model with its loss and metrics for the gradient trainrun."""
    pipe = CuvisPipeline(f"{arch}_train")
    frame = CourseFrame(name="frame")
    model = SpectrumRegressor(name=arch, arch=arch, input_channels=MEAN_SPECTRA_CHANNELS)
    mean = SpatialMean(name="mean")
    loss = RegressionMSELoss(name="loss")
    metrics = RegressionMetrics(name="metrics")
    pipe.connect(
        (frame.outputs.cube, model.inputs.cube),
        (model.outputs.prediction, mean.inputs.prediction),
        (mean.outputs.prediction, loss.inputs.prediction),
        (frame.outputs.target, loss.inputs.target),
        (mean.outputs.prediction, metrics.inputs.prediction),
        (frame.outputs.target, metrics.inputs.target),
    )
    _emit(
        pipe,
        f"03_spectral_vs_spatial/{arch}_train.yaml",
        BASE_PLUGINS,
        f"{arch}_train",
        f"The course's {arch} spectrum regressor from random weights with the MSE loss in fat "
        "percent and pooled regression metrics, for the gradient trainrun.",
        ["course", arch, "training"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    _ensure_placeholders()
    build_pork_pls_fusion()
    build_pls_mean_spectra_fit()
    build_pls_pixel_map()
    build_spectral_models()
    build_spectrum_training("mlp")
    build_spectrum_training("resnet1d")
    if args.skip_models:
        print("skipped: kiwi_sam3_spectral_match, yolo_rgb, resnet18_features, unet_ensemble_maps")
        return
    build_resnet18_features()
    try:
        build_unet_ensemble_maps("hf")
    except Exception as exc:  # noqa: BLE001
        print(
            f"U-Net weights from Hugging Face unavailable ({exc}); wiring built with random weights"
        )
        build_unet_ensemble_maps(None)
    build_yolo_rgb()
    build_kiwi_sam3_spectral_match()


if __name__ == "__main__":
    main()
