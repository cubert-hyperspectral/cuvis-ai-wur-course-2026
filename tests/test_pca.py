"""Tests for cuvis_ai_wur_course.node.pca.DatasetPCA."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PortSpec
from sklearn.decomposition import PCA as SklearnPCA

from cuvis_ai_wur_course.node.pca import DatasetPCA

pytestmark = pytest.mark.unit


# ----- helpers ---------------------------------------------------------


def _random_cube(seed: int, shape: tuple[int, int, int, int]) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal(shape).astype(np.float32))


def _fitted_node(cube: torch.Tensor, n_components: int) -> DatasetPCA:
    channels = cube.shape[-1]
    node = DatasetPCA(input_channels=channels, n_components=n_components)
    node.statistical_initialization(iter([{"cube": cube}]))
    return node


def _stream_with_folds(cube: torch.Tensor, fold_ids: list[int]) -> list[dict[str, torch.Tensor]]:
    """Split ``cube`` into one batch per frame, each tagged with its fold id.

    ``len(fold_ids)`` must equal ``cube.shape[0]``.
    """
    assert len(fold_ids) == cube.shape[0]
    return [
        {"cube": cube[i : i + 1], "fold": torch.tensor([fold_ids[i]], dtype=torch.int32)}
        for i in range(cube.shape[0])
    ]


# ----- 1. golden-reference (parity with sklearn, up to sign) -----------


def test_parity_with_sklearn_projection_and_explained_variance() -> None:
    cube = _random_cube(seed=1, shape=(5, 3, 3, 8))
    n_components = 3
    node = _fitted_node(cube, n_components)

    pixels = cube.reshape(-1, cube.shape[-1]).numpy()
    sk = SklearnPCA(n_components=n_components)
    sk_projected = sk.fit_transform(pixels)

    out = node(cube=cube)
    projected = out["cube"].reshape(-1, n_components).detach().numpy()

    for k in range(n_components):
        ours = projected[:, k]
        theirs = sk_projected[:, k]
        matches_positive = np.allclose(ours, theirs, atol=1e-3)
        matches_negative = np.allclose(ours, -theirs, atol=1e-3)
        assert matches_positive or matches_negative, f"component {k} does not match up to sign"

    np.testing.assert_allclose(
        out["explained_variance_ratio"].detach().numpy(),
        sk.explained_variance_ratio_,
        atol=1e-6,
    )


def test_sign_convention_is_deterministic_across_fits() -> None:
    """Fitting twice from the same data gives bit-identical component signs."""
    cube = _random_cube(seed=2, shape=(4, 2, 2, 6))
    node_a = _fitted_node(cube, n_components=2)
    node_b = _fitted_node(cube, n_components=2)

    assert torch.allclose(node_a._components, node_b._components, atol=1e-6)

    # Every component's largest-magnitude entry is positive by construction.
    max_idx = node_a._components.abs().argmax(dim=0)
    picked = node_a._components[max_idx, torch.arange(node_a._components.shape[1])]
    assert torch.all(picked > 0)


def test_n_components_cannot_exceed_input_channels() -> None:
    with pytest.raises(ValueError):
        DatasetPCA(input_channels=4, n_components=5)


def test_unfit_forward_raises() -> None:
    node = DatasetPCA(input_channels=4, n_components=2)
    with pytest.raises(RuntimeError):
        node(cube=torch.rand(1, 2, 2, 4))


# ----- 1b. exclude_folds --------------------------------------------------


def test_exclude_folds_fits_only_on_kept_frames() -> None:
    """Fitting with exclude_folds=[1] matches a direct fit over the kept frames."""
    cube = _random_cube(seed=30, shape=(6, 2, 2, 5))
    fold_ids = [0, 1, 0, 1, 2, 2]
    stream = _stream_with_folds(cube, fold_ids)

    node = DatasetPCA(input_channels=5, n_components=2, exclude_folds=[1])
    node.statistical_initialization(iter(stream))

    kept = cube[[i for i, f in enumerate(fold_ids) if f != 1]]
    expected = _fitted_node(kept, n_components=2)

    assert torch.allclose(node._mean, expected._mean, atol=1e-5)
    for k in range(2):
        same = torch.allclose(node._components[:, k], expected._components[:, k], atol=1e-4)
        flipped = torch.allclose(node._components[:, k], -expected._components[:, k], atol=1e-4)
        assert same or flipped, f"component {k} does not match up to sign"
    assert torch.allclose(
        node._explained_variance_ratio, expected._explained_variance_ratio, atol=1e-4
    )


def test_exclude_folds_requires_fold_input() -> None:
    """A stream without 'fold' must raise when exclude_folds is set."""
    cube = _random_cube(seed=31, shape=(4, 2, 2, 5))
    node = DatasetPCA(input_channels=5, n_components=2, exclude_folds=[1])
    with pytest.raises(ValueError, match="DatasetPCA.*exclude_folds"):
        node.statistical_initialization(iter([{"cube": cube}]))


def test_exclude_folds_none_with_fold_present_is_unchanged() -> None:
    """exclude_folds=None ignores a fold tensor and fits on every frame."""
    cube = _random_cube(seed=32, shape=(6, 2, 2, 5))
    fold_ids = [0, 1, 0, 1, 2, 2]
    stream = _stream_with_folds(cube, fold_ids)

    node = DatasetPCA(input_channels=5, n_components=2)
    node.statistical_initialization(iter(stream))

    expected = _fitted_node(cube, n_components=2)

    assert torch.allclose(node._mean, expected._mean, atol=1e-5)
    assert torch.allclose(
        node._explained_variance_ratio, expected._explained_variance_ratio, atol=1e-4
    )


def test_exclude_folds_round_trips_through_yaml(tmp_path) -> None:
    """exclude_folds survives a save/reload the way the other hyperparameters do."""
    cube = _random_cube(seed=33, shape=(6, 2, 2, 5))
    fold_ids = [0, 1, 0, 1, 2, 2]
    stream = _stream_with_folds(cube, fold_ids)

    node = DatasetPCA(input_channels=5, n_components=2, exclude_folds=[1])
    node.statistical_initialization(iter(stream))
    node._name = "pca"

    pipeline = CuvisPipeline("dataset_pca_exclude_folds_reload")
    pipeline._assign_counter_and_add_node(node)

    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path, validate_nodes=False)
    weights_path = config_path.with_suffix(".pt")

    rebuilt = CuvisPipeline.load_pipeline(config_path, weights_path=weights_path)
    rebuilt_node = next(n for n in rebuilt.nodes if isinstance(n, DatasetPCA))

    assert rebuilt_node.exclude_folds == [1]
    out = rebuilt_node(cube=cube)
    expected = node(cube=cube)
    assert torch.allclose(out["cube"], expected["cube"], atol=1e-6)


# ----- 2. port contract --------------------------------------------------


def test_port_contract() -> None:
    cube = _random_cube(seed=3, shape=(2, 4, 4, 6))
    node = _fitted_node(cube, n_components=3)

    out = node(cube=cube)

    for port_name, spec in node.OUTPUT_SPECS.items():
        assert port_name in out, f"missing output port: {port_name}"
        tensor = out[port_name]
        assert isinstance(tensor, torch.Tensor)
        if isinstance(spec, PortSpec):
            assert tensor.dtype == spec.dtype

    assert out["cube"].shape == (2, 4, 4, 3)
    assert out["explained_variance_ratio"].shape == (3,)


# ----- explained_variance_ratio port name matches ExplainedVarianceMetric ---


def test_explained_variance_port_name_matches_metric_consumer() -> None:
    from cuvis_ai.node.metrics import ExplainedVarianceMetric

    assert "explained_variance_ratio" in ExplainedVarianceMetric.INPUT_SPECS
    assert "explained_variance_ratio" in DatasetPCA.OUTPUT_SPECS


# ----- 3. reload smoke -----------------------------------------------------


def test_reload_smoke(tmp_path) -> None:
    """Save the fitted node, then reload structure and weights together.

    As with ``Standardize``, ``_statistically_initialized`` is not a
    persisted buffer, so the fitted mean/components/ratio buffers must come back via
    the companion weights checkpoint, not structure alone.
    """
    cube = _random_cube(seed=4, shape=(3, 2, 2, 5))
    node = _fitted_node(cube, n_components=2)
    node._name = "pca"

    pipeline = CuvisPipeline("dataset_pca_reload")
    pipeline._assign_counter_and_add_node(node)

    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path, validate_nodes=False)
    weights_path = config_path.with_suffix(".pt")

    rebuilt = CuvisPipeline.load_pipeline(config_path, weights_path=weights_path)
    rebuilt_node = next(n for n in rebuilt.nodes if isinstance(n, DatasetPCA))

    out = rebuilt_node(cube=cube)
    expected = node(cube=cube)
    assert torch.allclose(out["cube"], expected["cube"], atol=1e-6)
    assert torch.allclose(
        out["explained_variance_ratio"], expected["explained_variance_ratio"], atol=1e-6
    )


# ----- 4. export smoke -------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    """The node, or its export_module() adapter, with tensors in and tensors out."""

    def __init__(self, node: DatasetPCA) -> None:
        super().__init__()
        adapter = getattr(node, "export_module", lambda: None)()
        self.module = adapter if adapter is not None else node

    def forward(self, cube: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.module(cube=cube)
        return out["cube"], out["explained_variance_ratio"]


def test_export_smoke() -> None:
    cube = _random_cube(seed=5, shape=(3, 3, 3, 6))
    node = _fitted_node(cube, n_components=2).eval()

    program = torch.export.export(_ExportProbe(node), (cube,))

    other = _random_cube(seed=6, shape=(3, 3, 3, 6))
    traced_cube, traced_ratio = program.module()(other)
    expected = node(cube=other)
    assert torch.allclose(traced_cube, expected["cube"], atol=1e-6)
    assert torch.allclose(traced_ratio, expected["explained_variance_ratio"], atol=1e-6)
