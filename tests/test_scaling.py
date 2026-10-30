"""Tests for cuvis_ai_wur_course.node.scaling.GlobalScalarStandardizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.node.scaling import GlobalScalarStandardizer

pytestmark = pytest.mark.unit


# ----- helpers -------------------------------------------------------------


def _stream(seed: int, n_batches: int = 3, shape: tuple[int, ...] = (2, 4, 4, 5)):
    rng = np.random.default_rng(seed)
    cubes = [rng.random(shape).astype(np.float32) for _ in range(n_batches)]
    stream = [{"cube": torch.tensor(c)} for c in cubes]
    all_values = np.concatenate([c.reshape(-1) for c in cubes])
    return stream, all_values


def _stream_with_folds(seed: int, n_batches: int = 3, shape: tuple[int, ...] = (2, 4, 4, 5)):
    """Like ``_stream`` but each batch also carries a per-frame ``fold`` tensor.

    Fold ids cycle ``0, 1, 2`` across the frames of the whole stream. Returns the
    batch stream and the flat array of values belonging only to frames whose fold
    id is *not* in ``exclude_folds``, for a given ``exclude_folds`` list.
    """
    rng = np.random.default_rng(seed)
    cubes = [rng.random(shape).astype(np.float32) for _ in range(n_batches)]
    n_per_batch = shape[0]
    fold_ids = [i % 3 for i in range(n_batches * n_per_batch)]

    stream = []
    for b, cube in enumerate(cubes):
        fold = torch.tensor(fold_ids[b * n_per_batch : (b + 1) * n_per_batch], dtype=torch.int32)
        stream.append({"cube": torch.tensor(cube), "fold": fold})

    def kept_values(exclude_folds: list[int]) -> np.ndarray:
        kept_frames = []
        for b, cube in enumerate(cubes):
            for f in range(n_per_batch):
                if fold_ids[b * n_per_batch + f] not in exclude_folds:
                    kept_frames.append(cube[f])
        return np.stack(kept_frames).reshape(-1)

    return stream, kept_values


# ----- 1. golden-reference ---------------------------------------------------


def test_golden_reference_ddof0_matches_numpy_default() -> None:
    """Default ddof=0 matches numpy's default population standard deviation."""
    stream, values = _stream(seed=10)
    node = GlobalScalarStandardizer()
    node.statistical_initialization(iter(stream))

    assert node.mean.item() == pytest.approx(values.mean(), abs=1e-5)
    assert node.std.item() == pytest.approx(values.std(), abs=1e-5)

    cube = stream[0]["cube"].clone()
    out = node(cube=cube)["cube"]
    expected = (cube - values.mean()) / max(values.std(), 1e-6)
    assert torch.allclose(out, expected, atol=1e-5)


def test_golden_reference_ddof1_matches_numpy_sample_std() -> None:
    stream, values = _stream(seed=11)
    node = GlobalScalarStandardizer(ddof=1)
    node.statistical_initialization(iter(stream))

    assert node.std.item() == pytest.approx(values.std(ddof=1), abs=1e-5)


def test_invalid_ddof_rejected() -> None:
    with pytest.raises(ValueError):
        GlobalScalarStandardizer(ddof=2)


def test_unfit_forward_raises() -> None:
    node = GlobalScalarStandardizer()
    with pytest.raises(RuntimeError):
        node(cube=torch.rand(1, 2, 2, 3))


# ----- 1b. exclude_folds ------------------------------------------------------


def test_exclude_folds_fits_only_on_kept_frames() -> None:
    """Fitting with exclude_folds=[1] matches a direct fit over the kept frames."""
    stream, kept_values = _stream_with_folds(seed=20)
    expected = kept_values([1])

    node = GlobalScalarStandardizer(exclude_folds=[1])
    node.statistical_initialization(iter(stream))

    assert node.mean.item() == pytest.approx(expected.mean(), abs=1e-5)
    assert node.std.item() == pytest.approx(expected.std(), abs=1e-5)


def test_exclude_folds_requires_fold_input() -> None:
    """A stream without 'fold' must raise when exclude_folds is set."""
    stream, _ = _stream(seed=21)
    node = GlobalScalarStandardizer(exclude_folds=[1])
    with pytest.raises(ValueError, match="GlobalScalarStandardizer.*exclude_folds"):
        node.statistical_initialization(iter(stream))


def test_exclude_folds_none_with_fold_present_is_unchanged() -> None:
    """exclude_folds=None ignores a fold tensor and fits on every frame."""
    stream, kept_values = _stream_with_folds(seed=22)
    all_values = kept_values([])

    node = GlobalScalarStandardizer()
    node.statistical_initialization(iter(stream))

    assert node.mean.item() == pytest.approx(all_values.mean(), abs=1e-5)
    assert node.std.item() == pytest.approx(all_values.std(), abs=1e-5)


def test_exclude_folds_round_trips_through_yaml(tmp_path) -> None:
    """exclude_folds survives a save/reload the way the other hyperparameters do."""
    stream, _ = _stream_with_folds(seed=23)
    node = GlobalScalarStandardizer(exclude_folds=[1], name="scaler")
    node.statistical_initialization(iter(stream))

    pipeline = CuvisPipeline("global_scalar_standardizer_exclude_folds_reload")
    pipeline._assign_counter_and_add_node(node)

    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path, validate_nodes=False)
    weights_path = config_path.with_suffix(".pt")

    rebuilt = CuvisPipeline.load_pipeline(config_path, weights_path=weights_path)
    rebuilt_node = next(n for n in rebuilt.nodes if isinstance(n, GlobalScalarStandardizer))

    assert rebuilt_node.exclude_folds == [1]
    cube = stream[0]["cube"].clone()
    assert torch.allclose(node(cube=cube)["cube"], rebuilt_node(cube=cube)["cube"], atol=1e-6)


# ----- 2. port contract ------------------------------------------------------


def test_port_contract() -> None:
    stream, _ = _stream(seed=12)
    node = GlobalScalarStandardizer()
    node.statistical_initialization(iter(stream))

    cube = torch.rand(2, 4, 4, 5, dtype=torch.float32)
    out = node(cube=cube)

    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        if isinstance(spec, PortSpec):
            assert tensor.dtype == spec.dtype
        assert tensor.shape == cube.shape


# ----- 3. reload smoke --------------------------------------------------------


def test_reload_smoke(tmp_path) -> None:
    """Save the fitted node, then reload structure and weights together.

    ``PipelineBuilder.build_from_config`` only reconstructs the graph structure (no
    ``.pt`` is read), so a freshly rebuilt node's ``_statistically_initialized`` flag
    (not a persisted buffer) comes back ``False``. ``CuvisPipeline.load_pipeline``
    additionally restores the co-located weights checkpoint that ``save_to_file``
    writes, which is the reload path appropriate for an already-fitted node.
    """
    stream, _ = _stream(seed=13)
    node = GlobalScalarStandardizer(ddof=0, name="scaler")
    node.statistical_initialization(iter(stream))

    pipeline = CuvisPipeline("global_scalar_standardizer_reload")
    pipeline._assign_counter_and_add_node(node)

    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path, validate_nodes=False)
    weights_path = config_path.with_suffix(".pt")

    rebuilt = CuvisPipeline.load_pipeline(config_path, weights_path=weights_path)
    rebuilt_node = next(n for n in rebuilt.nodes if isinstance(n, GlobalScalarStandardizer))

    cube = stream[0]["cube"].clone()
    out = rebuilt_node(cube=cube)["cube"]
    expected = node(cube=cube)["cube"]
    assert torch.allclose(out, expected, atol=1e-6)


# ----- 4. export smoke ---------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    """The node, or its export_module() adapter, with tensors in and a tensor out."""

    def __init__(self, node: GlobalScalarStandardizer) -> None:
        super().__init__()
        adapter = getattr(node, "export_module", lambda: None)()
        self.module = adapter if adapter is not None else node

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        return self.module(cube=cube)["cube"]


def test_export_smoke() -> None:
    stream, _ = _stream(seed=14)
    node = GlobalScalarStandardizer()
    node.statistical_initialization(iter(stream))
    node.eval()

    example = stream[0]["cube"]
    program = torch.export.export(_ExportProbe(node), (example,))

    other = stream[1]["cube"]
    traced = program.module()(other)
    assert torch.allclose(traced, node(cube=other)["cube"], atol=1e-6)
