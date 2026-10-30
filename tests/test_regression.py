"""Tests for cuvis_ai_wur_course.node.regression.PLSRegression."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

ikpls = pytest.importorskip("ikpls.numpy")
from cuvis_ai_core.node import Node as _NodeBase  # noqa: E402
from cuvis_ai_schemas.pipeline import PortSpec  # noqa: E402
from sklearn.cross_decomposition import PLSRegression as SklearnPLS  # noqa: E402

from cuvis_ai_wur_course.node.regression import PLSRegression  # noqa: E402

# ----- helpers -----------------------------------------------------------


def _random_pls1_data(
    n: int = 60, c: int = 12, t: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random spectra and a target with a genuine linear relationship."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, c))
    true_coef = rng.normal(size=(c, t))
    y = x @ true_coef + 0.05 * rng.normal(size=(n, t))
    return x.astype(np.float64), y.astype(np.float64)


def _frames_from_spectra(x: np.ndarray) -> torch.Tensor:
    """Turn an (N, C) spectra matrix into 1x1 BHWC frames, float32."""
    n, c = x.shape
    return torch.as_tensor(x, dtype=torch.float32).reshape(n, 1, 1, c)


def _fixed_node(c: int, t: int, n_components: int) -> PLSRegression:
    return PLSRegression(input_channels=c, n_targets=t, n_components=n_components)


def _fit_fixed(node: PLSRegression, x: np.ndarray, y: np.ndarray) -> None:
    cube = _frames_from_spectra(x)
    target = torch.as_tensor(y, dtype=torch.float32)
    stream = iter([{"cube": cube, "target": target}])
    node.statistical_initialization(stream)


# ----- 1. golden-reference: parity with sklearn and ikpls -----------------


def test_parity_with_sklearn_fixed_components():
    x, y = _random_pls1_data(n=60, c=12, t=1, seed=1)
    a = 4

    node = _fixed_node(c=12, t=1, n_components=a)
    _fit_fixed(node, x, y)

    sk = SklearnPLS(n_components=a, scale=False)
    sk.fit(x, y)
    expected = sk.predict(x)

    cube = _frames_from_spectra(x)
    out = node(cube=cube)["prediction"].reshape(-1, 1).detach().numpy()
    assert np.allclose(out, expected, atol=1e-5), np.abs(out - expected).max()


def test_parity_with_ikpls_fixed_components():
    x, y = _random_pls1_data(n=60, c=12, t=1, seed=2)
    a = 5

    node = _fixed_node(c=12, t=1, n_components=a)
    _fit_fixed(node, x, y)

    ref = ikpls.PLS(algorithm=1, center_X=True, center_Y=True, scale_X=False, scale_Y=False)
    ref.fit(x, y, A=a)
    expected = ref.predict(x, n_components=a)

    cube = _frames_from_spectra(x)
    out = node(cube=cube)["prediction"].reshape(-1, 1).detach().numpy()
    assert np.allclose(out, expected, atol=1e-5), np.abs(out - expected).max()

    expected_coef = ref.B[a - 1]
    assert np.allclose(node.coef.detach().numpy(), expected_coef, atol=1e-5)


def test_parity_with_ikpls_two_targets():
    x, y = _random_pls1_data(n=60, c=12, t=2, seed=3)
    a = 4

    node = _fixed_node(c=12, t=2, n_components=a)
    _fit_fixed(node, x, y)

    ref = ikpls.PLS(algorithm=1, center_X=True, center_Y=True, scale_X=False, scale_Y=False)
    ref.fit(x, y, A=a)
    expected = ref.predict(x, n_components=a)

    cube = _frames_from_spectra(x)
    out = node(cube=cube)["prediction"].reshape(-1, 2).detach().numpy()
    assert np.allclose(out, expected, atol=1e-5), np.abs(out - expected).max()


# ----- 2. port contract ----------------------------------------------------


def test_port_contract():
    x, y = _random_pls1_data(n=20, c=8, t=1, seed=4)
    node = _fixed_node(c=8, t=1, n_components=3)
    _fit_fixed(node, x, y)

    cube = _frames_from_spectra(x)
    out = node(cube=cube)

    assert "prediction" in out
    pred = out["prediction"]
    assert isinstance(pred, torch.Tensor)
    spec = node.OUTPUT_SPECS["prediction"]
    assert pred.dtype == spec.dtype
    assert pred.shape == (20, 1, 1, 1)


def test_port_contract_image_frame_multi_target():
    c, t = 6, 2
    node = PLSRegression(input_channels=c, n_targets=t, n_components=2)
    x = np.random.default_rng(5).normal(size=(10, c))
    y = np.random.default_rng(6).normal(size=(10, t))
    _fit_fixed(node, x, y)

    cube = torch.rand(3, 5, 5, c, dtype=torch.float32)
    out = node(cube=cube)["prediction"]
    assert out.shape == (3, 5, 5, t)
    assert out.dtype == torch.float32


# ----- 3. selection modes on toy data with a known right answer -----------


def _toy_selection_data(n_per_fold: int = 20, c: int = 6, n_folds: int = 5, seed: int = 10):
    """Spectra with exactly 2 informative directions; more components overfit noise.

    Folds are i.i.d. draws from the same generative model, so the pooled
    held-out SSE should bottom out near the true number of components (2)
    and not keep improving as more (noise-fitting) components are added.
    """
    rng = np.random.default_rng(seed)
    true_coef = rng.normal(size=(c, 1))
    true_coef[2:] = 0.0  # only 2 informative channels really drive y... but
    # make it rank reducible: y depends on exactly 2 latent directions
    w1 = rng.normal(size=(c, 1))
    w2 = rng.normal(size=(c, 1))
    xs, ys, folds = [], [], []
    for fold in range(1, n_folds + 1):
        z1 = rng.normal(size=(n_per_fold, 1))
        z2 = rng.normal(size=(n_per_fold, 1))
        x = z1 @ w1.T + z2 @ w2.T + 0.01 * rng.normal(size=(n_per_fold, c))
        y = 3.0 * z1 + -2.0 * z2 + 0.01 * rng.normal(size=(n_per_fold, 1))
        xs.append(x)
        ys.append(y)
        folds.append(np.full(n_per_fold, fold, dtype=np.int32))
    return (
        np.concatenate(xs).astype(np.float64),
        np.concatenate(ys).astype(np.float64),
        np.concatenate(folds),
    )


def _fit_with_folds(node: PLSRegression, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> None:
    cube = _frames_from_spectra(x)
    target = torch.as_tensor(y, dtype=torch.float32)
    fold = torch.as_tensor(folds, dtype=torch.int32)
    stream = iter([{"cube": cube, "target": target, "fold": fold}])
    node.statistical_initialization(stream)


def test_selection_mode_each_refit_picks_near_true_rank():
    x, y, folds = _toy_selection_data()
    node = PLSRegression(
        input_channels=6,
        n_targets=1,
        max_components=6,
        holdout_folds="each",
        refit_on_holdout=True,
    )
    _fit_with_folds(node, x, y, folds)

    assert int(node.n_selected.item()) == 2
    assert node.cv_sse.shape == (6,)


def test_selection_mode_list_no_refit():
    x, y, folds = _toy_selection_data()
    node = PLSRegression(
        input_channels=6,
        n_targets=1,
        max_components=6,
        holdout_folds=[1],
        refit_on_holdout=False,
    )
    _fit_with_folds(node, x, y, folds)

    assert int(node.n_selected.item()) == 2

    # No-refit: the kept fit must match a direct fit on folds != 1 only.
    keep = folds != 1
    direct = PLSRegression(input_channels=6, n_targets=1, n_components=2)
    _fit_fixed(direct, x[keep], y[keep])

    assert torch.allclose(node.coef, direct.coef, atol=1e-5)
    assert torch.allclose(node.x_mean, direct.x_mean, atol=1e-5)


def test_fixed_n_components_ignores_folds():
    x, y, folds = _toy_selection_data()
    node = PLSRegression(input_channels=6, n_targets=1, n_components=3)
    _fit_with_folds(node, x, y, folds)

    direct = PLSRegression(input_channels=6, n_targets=1, n_components=3)
    _fit_fixed(direct, x, y)

    assert torch.allclose(node.coef, direct.coef, atol=1e-6)
    assert int(node.n_selected.item()) == 3
    assert torch.all(node.cv_sse == 0)


def test_selection_mode_each_single_fold_raises():
    """A single unique fold id under holdout_folds='each' holds out every frame.

    Leaving out the only fold present makes the training set empty, which
    must be refused loudly instead of silently returning a NaN-SSE fit with
    an arbitrary selected component count.
    """
    x, y = _random_pls1_data(n=10, c=4, t=1, seed=20)
    folds = np.zeros(10, dtype=np.int32)
    node = PLSRegression(input_channels=4, n_targets=1, max_components=2, holdout_folds="each")
    with pytest.raises(ValueError, match="fold"):
        _fit_with_folds(node, x, y, folds)


def test_selection_mode_list_absent_fold_raises():
    """A holdout_folds id missing from the batch must be refused, not silently SSE=0."""
    x, y, folds = _toy_selection_data(n_per_fold=5, n_folds=3)
    node = PLSRegression(
        input_channels=6,
        n_targets=1,
        max_components=2,
        holdout_folds=[99],
        refit_on_holdout=False,
    )
    with pytest.raises(ValueError, match="99"):
        _fit_with_folds(node, x, y, folds)


# ----- 4. uniform-scaling invariance ---------------------------------------


def test_uniform_scaling_invariance_fixed_components():
    x, y = _random_pls1_data(n=40, c=10, t=1, seed=7)
    scale = 3.7

    node_a = _fixed_node(c=10, t=1, n_components=4)
    _fit_fixed(node_a, x, y)

    node_b = _fixed_node(c=10, t=1, n_components=4)
    _fit_fixed(node_b, x * scale, y)

    cube_a = _frames_from_spectra(x)
    cube_b = _frames_from_spectra(x * scale)
    pred_a = node_a(cube=cube_a)["prediction"]
    pred_b = node_b(cube=cube_b)["prediction"]
    assert torch.allclose(pred_a, pred_b, atol=1e-4)


def test_uniform_scaling_invariance_selection():
    x, y, folds = _toy_selection_data(seed=11)
    scale = 0.42

    node_a = PLSRegression(input_channels=6, n_targets=1, max_components=6, holdout_folds="each")
    _fit_with_folds(node_a, x, y, folds)

    node_b = PLSRegression(input_channels=6, n_targets=1, max_components=6, holdout_folds="each")
    _fit_with_folds(node_b, x * scale, y, folds)

    assert int(node_a.n_selected.item()) == int(node_b.n_selected.item())

    cube_a = _frames_from_spectra(x)
    cube_b = _frames_from_spectra(x * scale)
    pred_a = node_a(cube=cube_a)["prediction"]
    pred_b = node_b(cube=cube_b)["prediction"]
    assert torch.allclose(pred_a, pred_b, atol=1e-3)


# ----- 5. statistical_initialization through multiple batches -------------


def test_statistical_initialization_multi_batch_stream():
    x, y, folds = _toy_selection_data(n_per_fold=10, seed=12)
    node = PLSRegression(input_channels=6, n_targets=1, max_components=5, holdout_folds="each")
    cube = _frames_from_spectra(x)
    target = torch.as_tensor(y, dtype=torch.float32)
    fold = torch.as_tensor(folds, dtype=torch.int32)

    # Split into several batches of varying size; the stream must be
    # consumed by iterating once, collecting everything.
    batches = []
    bounds = [0, 7, 23, 41, len(x)]
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        batches.append(
            {
                "cube": cube[start:end],
                "target": target[start:end],
                "fold": fold[start:end],
            }
        )
    node.statistical_initialization(iter(batches))
    assert node._statistically_initialized

    single = PLSRegression(input_channels=6, n_targets=1, max_components=5, holdout_folds="each")
    _fit_with_folds(single, x, y, folds)

    assert int(node.n_selected.item()) == int(single.n_selected.item())
    assert torch.allclose(node.coef, single.coef, atol=1e-5)


# ----- 6. reload from YAML + state_dict round-trip -------------------------


class _PLSReloadSource(_NodeBase):
    """Minimal source emitting a random cube and target, for the reload test."""

    INPUT_SPECS: dict = {}
    OUTPUT_SPECS = {
        "cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "target": PortSpec(dtype=torch.float32, shape=(-1, -1)),
    }

    def forward(self, **_):
        return {
            "cube": torch.rand(4, 1, 1, 6, dtype=torch.float32),
            "target": torch.rand(4, 1, dtype=torch.float32),
        }


def test_reload_from_yaml(tmp_path):
    from cuvis_ai_core.pipeline.factory import PipelineBuilder
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    x, y = _random_pls1_data(n=30, c=6, t=1, seed=8)
    node = PLSRegression(input_channels=6, n_targets=1, n_components=3, name="pls")
    _fit_fixed(node, x, y)

    pipeline = CuvisPipeline("pls_reload_test")
    src = _PLSReloadSource(name="src")
    pipeline.connect(src.outputs.cube, node.inputs.cube)
    pipeline.connect(src.outputs.target, node.inputs.target)

    out_path = tmp_path / "pls_pipeline.yaml"
    pipeline.save_to_file(str(out_path))

    rebuilt = PipelineBuilder().build_from_config(str(out_path))
    rebuilt_node = next(n for n in rebuilt.nodes if n.name == "pls")
    rebuilt_node.load_state_dict(node.state_dict())

    cube = _frames_from_spectra(x)
    out_before = node(cube=cube)["prediction"]
    out_after = rebuilt_node(cube=cube)["prediction"]
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_state_dict_round_trip():
    x, y = _random_pls1_data(n=30, c=6, t=1, seed=9)
    node = PLSRegression(input_channels=6, n_targets=1, n_components=3)
    _fit_fixed(node, x, y)

    state = node.state_dict()
    fresh = PLSRegression(input_channels=6, n_targets=1, n_components=3)
    fresh.load_state_dict(state)

    cube = _frames_from_spectra(x)
    assert torch.allclose(node(cube=cube)["prediction"], fresh(cube=cube)["prediction"], atol=1e-6)
    assert fresh._statistically_initialized


# ----- 7. export smoke -----------------------------------------------------


class _ExportProbe(torch.nn.Module):
    def __init__(self, node: PLSRegression):
        super().__init__()
        self.node = node

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        return self.node(cube=cube)["prediction"]


def test_export_smoke():
    x, y = _random_pls1_data(n=20, c=6, t=1, seed=13)
    node = _fixed_node(c=6, t=1, n_components=3).eval()
    _fit_fixed(node, x, y)

    example = _frames_from_spectra(x)
    program = torch.export.export(_ExportProbe(node), (example,))

    other = _frames_from_spectra(x) * 1.5
    traced = program.module()(other)
    assert torch.allclose(traced, node(cube=other)["prediction"], atol=1e-6)


def test_not_initialized_raises():
    node = PLSRegression(input_channels=5, n_targets=1, n_components=2)
    with pytest.raises(RuntimeError):
        node(cube=torch.rand(1, 1, 1, 5, dtype=torch.float32))


# ----- 8. constructor validation --------------------------------------------


def test_fixed_n_components_with_holdout_folds_warns():
    """A fixed n_components together with holdout_folds is likely a config typo."""
    with pytest.warns(UserWarning, match="holdout_folds"):
        PLSRegression(
            input_channels=4,
            n_targets=1,
            n_components=2,
            holdout_folds="each",
        )
