"""Partial least squares regression on hyperspectral spectra.

Ships :class:`PLSRegression`, a scikit-learn-free PLS1/PLS2 node that fits
Improved Kernel PLS Algorithm 1 (Dayal and MacGregor, 1997) during
``statistical_initialization`` and predicts with a single per-pixel linear
map in ``forward``.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

_HOLDOUT_EACH = "each"


def _ikpls_algorithm1(
    x: torch.Tensor, y: torch.Tensor, n_components: int, eps: float
) -> torch.Tensor:
    """Fit Improved Kernel PLS Algorithm 1 and return the coefficient tensor.

    Mirrors the inner loop of Dayal and MacGregor's Improved Kernel PLS
    Algorithm 1 (as implemented by ``ikpls.numpy``): deflates the
    cross-covariance ``X^T Y`` one component at a time and accumulates the
    regression coefficients ``B[a]`` for every component count ``1..A`` in
    one pass. ``x`` and ``y`` must already be centered (and, if desired,
    scaled); this function performs no centering of its own. All arithmetic
    runs in float64 for numerical parity with ``ikpls`` and scikit-learn.

    Parameters
    ----------
    x : torch.Tensor
        Centered predictor matrix, shape ``(N, K)``, float64.
    y : torch.Tensor
        Centered response matrix, shape ``(N, M)``, float64.
    n_components : int
        Number of components ``A`` to compute.
    eps : float
        Threshold below which a weight or score norm is treated as
        numerically exhausted; matches ``numpy.finfo(numpy.float64).eps``
        in the reference implementation when the caller passes it through.

    Returns
    -------
    torch.Tensor
        Regression coefficients, shape ``(A, K, M)``, float64. ``B[a]`` are
        the coefficients using ``a + 1`` components. Once a component
        underflows (the X-Y cross-covariance is exhausted or a score falls
        into the null space of ``X``), every later row repeats the last
        stable coefficient, matching ``ikpls``'s rank-deficiency behavior.
    """
    n, k = x.shape
    m = y.shape[1]
    device, dtype = x.device, x.dtype

    b = torch.zeros((n_components, k, m), dtype=dtype, device=device)
    w_all = torch.zeros((n_components, k), dtype=dtype, device=device)
    p_all = torch.zeros((n_components, k), dtype=dtype, device=device)
    r_all = torch.zeros((n_components, k), dtype=dtype, device=device)

    xty = x.T @ y  # (K, M)
    max_stable = n_components
    max_t_t = 0.0

    for i in range(n_components):
        if m == 1:
            norm = torch.linalg.norm(xty)
            if torch.isclose(norm, norm.new_zeros(()), atol=eps, rtol=0.0):
                max_stable = i
                break
            w = xty / norm
        elif m < k:
            xtyxty = xty.T @ xty
            eig_vals, eig_vecs = torch.linalg.eigh(xtyxty)
            q_vec = eig_vecs[:, -1:]
            w = xty @ q_vec
            norm = torch.linalg.norm(w)
            if torch.isclose(norm, norm.new_zeros(()), atol=eps, rtol=0.0):
                max_stable = i
                break
            w = w / norm
        else:
            xtyytx = xty @ xty.T
            eig_vals, eig_vecs = torch.linalg.eigh(xtyytx)
            norm = eig_vals[-1]
            if torch.isclose(norm, norm.new_zeros(()), atol=eps, rtol=0.0):
                max_stable = i
                break
            w = eig_vecs[:, -1:]

        if i == 0:
            r = w.clone()
        else:
            r = w - r_all[:i].T @ (p_all[:i] @ w)

        t = x @ r
        t_t = (t.T @ t).reshape(())
        max_t_t = max(max_t_t, t_t.item())
        if t_t.item() <= eps * max_t_t:
            max_stable = i
            break

        p = (t.T @ x).T / t_t
        if m == 1:
            # q is a (1, 1) tensor: norm / tTt, both scalars.
            q = (norm / t_t).reshape(1, 1)
        elif m < k:
            q = (torch.sqrt(eig_vals[-1]) / t_t) * q_vec
        else:
            q = (r.T @ xty).T / t_t

        w_all[i] = w.squeeze(-1)
        r_all[i] = r.squeeze(-1)
        p_all[i] = p.squeeze(-1)

        xty = xty - (p @ q.T) * t_t

        b[i] = (b[i - 1] if i > 0 else 0.0) + r @ q.T

    if 0 < max_stable < n_components:
        b[max_stable:] = b[max_stable - 1]

    return b


class PLSRegression(Node):
    """Predict continuous targets from a hyperspectral cube with fitted PLS.

    Fits an Improved Kernel PLS Algorithm 1 model (Dayal and MacGregor, 1997;
    also valid for several simultaneous targets, i.e. PLS1 generalized to
    PLS2) during ``statistical_initialization``, entirely in float64 torch
    with no scikit-learn dependency. ``forward`` then applies the fitted
    linear map per pixel in float32:
    ``prediction = (cube - x_mean) @ coef + y_mean``.

    The component count is either fixed (``n_components``) or chosen by a
    pooled sum-of-squared-errors criterion over held-out folds
    (``max_components`` with ``holdout_folds``). Three selection modes cover
    the course's three procedures:

    - ``holdout_folds="each"``, ``refit_on_holdout=True``: leave-one-fold-out
      cross-validation. For every fold present in the batch, fit on every
      other fold, predict the held-out fold, and sum the squared error over
      all folds and all component counts. The component count with the
      lowest pooled SSE is selected, and the model is refit on every frame
      (all folds together) at that component count.
    - ``holdout_folds=[<ids>]``, ``refit_on_holdout=False``: fit once on the
      frames whose fold is not in ``holdout_folds``, evaluate the pooled SSE
      on the held-out frames only, and keep that fit (no refit).
    - ``n_components`` fixed: fit once on every frame at that component
      count; ``holdout_folds`` and ``refit_on_holdout`` are ignored.

    The course's own preprocessing (SNV, autoscaling) centers and scales
    with ``ddof=0`` while cuvis-ai's ``UnitVarianceScaling`` uses ``ddof=1``.
    The two differ only by one scalar factor applied uniformly to every
    channel of ``X``, and PLS predictions (as well as the pooled-SSE
    component selection) are invariant to a uniform rescaling of ``X``: the
    fitted weight vector absorbs the scale, and the SSE ordering over
    component counts is unchanged. So this node needs no ddof variant; the
    uniform-scaling invariance is covered by a dedicated test.

    Non-exportable guards: component selection is a Python-side search over
    fold ids and component counts, so it lives only in
    ``statistical_initialization``, never inside ``forward``.

    Parameters
    ----------
    input_channels : int
        Number of spectral channels ``C`` in the incoming cube.
    n_targets : int, optional
        Number of regression targets ``T`` (default: ``1``).
    n_components : int or None, optional
        Fixed number of latent components. When given, ``max_components``,
        ``holdout_folds`` and ``refit_on_holdout`` are ignored and no
        held-out selection runs (default: ``None``).
    max_components : int or None, optional
        Upper bound on the number of components to try when
        ``n_components`` is not fixed; the component count with the lowest
        pooled held-out SSE is selected. Required (together with
        ``holdout_folds``) when ``n_components`` is ``None``
        (default: ``None``).
    holdout_folds : str or list[int] or None, optional
        ``"each"`` to leave every fold out in turn (summed pooled SSE over
        all folds), or a list of fold ids to hold out once. Required when
        ``n_components`` is ``None`` (default: ``None``).
    refit_on_holdout : bool, optional
        After selecting the component count, refit on every frame
        (``True``) or keep the fit computed while holding out the fold(s)
        (``False``). Ignored when ``n_components`` is fixed
        (default: ``True``).
    eps : float, optional
        Threshold for detecting a numerically exhausted weight or score
        norm during the PLS inner loop (default: ``1e-12``).

    Attributes
    ----------
    x_mean : torch.Tensor
        Per-channel training mean, shape ``(C,)``.
    coef : torch.Tensor
        Fitted regression coefficients, shape ``(C, T)``.
    y_mean : torch.Tensor
        Per-target training mean, shape ``(T,)``.
    cv_sse : torch.Tensor
        Pooled held-out sum of squared errors per component count, shape
        ``(max_components,)``. Left at zero when ``n_components`` is fixed.
    n_selected : torch.Tensor
        Int64 scalar: the selected number of components.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube, BHWC float32.",
        ),
        "target": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Per-frame regression target(s), needed for fitting.",
            optional=True,
        ),
        "fold": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame fold id; required when holdout_folds drives selection.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "prediction": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel prediction, (cube - x_mean) @ coef + y_mean.",
        ),
    }

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.REGRESSION, NodeTag.STATEFUL, NodeTag.TORCH})

    def __init__(
        self,
        input_channels: int,
        n_targets: int = 1,
        n_components: int | None = None,
        max_components: int | None = None,
        holdout_folds: str | list[int] | None = None,
        refit_on_holdout: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            n_targets=n_targets,
            n_components=n_components,
            max_components=max_components,
            holdout_folds=holdout_folds,
            refit_on_holdout=refit_on_holdout,
            eps=eps,
            **kwargs,
        )
        if n_components is None and max_components is None:
            raise ValueError(
                "PLSRegression requires either n_components (fixed) or "
                "max_components with holdout_folds."
            )
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if max_components is not None and max_components < 1:
            raise ValueError("max_components must be a positive integer.")
        if n_components is not None and (max_components is not None or holdout_folds is not None):
            warnings.warn(
                "PLSRegression received a fixed n_components together with "
                "max_components/holdout_folds; the fixed count wins and "
                "holdout_folds is ignored. Drop max_components/holdout_folds "
                "if this was intentional, or drop n_components if a fold-based "
                "selection was meant.",
                UserWarning,
                stacklevel=2,
            )

        self.input_channels = input_channels
        self.n_targets = n_targets
        self.n_components = n_components
        self.max_components = max_components
        self.holdout_folds = holdout_folds
        self.refit_on_holdout = refit_on_holdout
        self.eps = eps

        buffer_components = n_components if n_components is not None else max_components
        self._buffer_components = buffer_components

        self.register_buffer("x_mean", torch.zeros(input_channels, dtype=torch.float32))
        self.register_buffer("coef", torch.zeros(input_channels, n_targets, dtype=torch.float32))
        self.register_buffer("y_mean", torch.zeros(n_targets, dtype=torch.float32))
        self.register_buffer("cv_sse", torch.zeros(buffer_components, dtype=torch.float32))
        self.register_buffer("n_selected", torch.zeros((), dtype=torch.int64))

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit PLS from a stream of ``{cube, target, fold}`` batches.

        Collects every frame's spatial mean spectrum (so a 1x1 frame
        contributes its own spectrum unchanged) and its target into one
        ``(N, C)`` / ``(N, T)`` training table, then either fits once at a
        fixed component count or runs the fold-based selection described in
        the class docstring. Buffers are updated in place with ``.copy_()``.
        """
        spectra: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        folds: list[torch.Tensor] = []
        has_folds = True

        for batch in input_stream:
            cube = batch["cube"]
            target = batch.get("target")
            if target is None:
                raise RuntimeError(
                    "PLSRegression.statistical_initialization requires a 'target' in every batch."
                )
            fold = batch.get("fold")
            frame_mean = cube.reshape(cube.shape[0], -1, cube.shape[-1]).mean(dim=1)
            spectra.append(frame_mean.to(torch.float64))
            targets.append(target.reshape(target.shape[0], self.n_targets).to(torch.float64))
            if fold is None:
                has_folds = False
            else:
                folds.append(fold.to(torch.int64))

        x = torch.cat(spectra, dim=0)
        y = torch.cat(targets, dim=0)
        fold_ids = torch.cat(folds, dim=0) if (has_folds and folds) else None

        if self.n_components is not None:
            x_mean, y_mean, coef, _ = self._fit(x, y, self.n_components)
            selected = self.n_components
            cv_sse = torch.zeros(self._buffer_components, dtype=torch.float64)
        else:
            if fold_ids is None:
                raise RuntimeError(
                    "PLSRegression with max_components/holdout_folds requires a "
                    "'fold' input to drive component selection."
                )
            selected, cv_sse, x_mean, y_mean, coef = self._select_and_fit(x, y, fold_ids)

        self.x_mean.copy_(x_mean.to(torch.float32))
        self.y_mean.copy_(y_mean.to(torch.float32))
        self.coef.copy_(coef.to(torch.float32))
        self.cv_sse.copy_(cv_sse.to(torch.float32))
        self.n_selected.fill_(selected)
        self._statistically_initialized = True

    def _fit(
        self, x: torch.Tensor, y: torch.Tensor, n_components: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Center ``x``/``y`` and fit IKPLS Algorithm 1 at ``n_components``.

        Returns ``(x_mean, y_mean, coef, b_all)`` where ``coef`` is
        ``b_all[n_components - 1]`` and ``b_all`` holds every intermediate
        component count's coefficients, shape ``(n_components, C, T)``.
        """
        x_mean = x.mean(dim=0)
        y_mean = y.mean(dim=0)
        b_all = _ikpls_algorithm1(x - x_mean, y - y_mean, n_components, self.eps)
        coef = b_all[n_components - 1]
        return x_mean, y_mean, coef, b_all

    def _select_and_fit(
        self, x: torch.Tensor, y: torch.Tensor, fold_ids: torch.Tensor
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fold-based selection and return the chosen fit.

        Returns ``(selected_components, cv_sse, x_mean, y_mean, coef)``.
        """
        max_a = self.max_components
        unique_folds = sorted(torch.unique(fold_ids).tolist())

        if self.holdout_folds == _HOLDOUT_EACH:
            held_out_ids = unique_folds
        elif isinstance(self.holdout_folds, list):
            held_out_ids = list(self.holdout_folds)
        else:
            raise ValueError(
                "holdout_folds must be 'each' or a list of fold ids when max_components is set."
            )

        present_folds = set(unique_folds)
        for held_out in held_out_ids:
            if held_out not in present_folds:
                raise ValueError(
                    f"holdout_folds id {held_out!r} is not present in the batch's fold "
                    f"values {unique_folds}; the held-out split would be empty and the "
                    "pooled SSE criterion would never see a genuine sample."
                )
        if self.holdout_folds == _HOLDOUT_EACH and len(unique_folds) < 2:
            raise ValueError(
                "holdout_folds='each' requires at least two unique fold ids in the "
                f"batch to leave a non-empty training set; found {unique_folds}."
            )

        pooled_sse = torch.zeros(max_a, dtype=torch.float64)
        for held_out in held_out_ids:
            train_mask = fold_ids != held_out
            holdout_mask = fold_ids == held_out
            x_train, y_train = x[train_mask], y[train_mask]
            x_holdout, y_holdout = x[holdout_mask], y[holdout_mask]

            x_mean = x_train.mean(dim=0)
            y_mean = y_train.mean(dim=0)
            b_all = _ikpls_algorithm1(x_train - x_mean, y_train - y_mean, max_a, self.eps)

            xc = x_holdout - x_mean
            for a in range(max_a):
                pred = xc @ b_all[a] + y_mean
                pooled_sse[a] += ((pred - y_holdout) ** 2).sum()

        selected = int(torch.argmin(pooled_sse).item()) + 1

        if self.refit_on_holdout:
            x_mean, y_mean, coef, _ = self._fit(x, y, selected)
        else:
            keep_mask = ~torch.isin(fold_ids, torch.as_tensor(held_out_ids, dtype=fold_ids.dtype))
            x_mean, y_mean, coef, _ = self._fit(x[keep_mask], y[keep_mask], selected)

        return selected, pooled_sse, x_mean, y_mean, coef

    def forward(self, cube: torch.Tensor, target=None, fold=None, **_) -> dict[str, torch.Tensor]:
        """Apply the fitted linear map to every pixel of ``cube``.

        ``target`` and ``fold`` are accepted (they are needed only during
        ``statistical_initialization``) and otherwise unused here.
        """
        if not self._statistically_initialized:
            raise RuntimeError("PLSRegression requires statistical_initialization first")

        prediction = (cube - self.x_mean) @ self.coef + self.y_mean
        return {"prediction": prediction}
