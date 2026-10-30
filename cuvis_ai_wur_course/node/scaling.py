"""Global scalar standardization node for the WUR course pipelines."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai.utils.welford import WelfordAccumulator
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class GlobalScalarStandardizer(Node):
    """Subtract one scalar mean and divide by one scalar standard deviation.

    ``cuvis_ai.node.pretreatments.scaling.MeanCenter`` and ``UnitVarianceScaling`` fit
    one mean or standard deviation per channel. This node exists because the course
    (``end2end_spectral_vs_spatial_models``) standardizes with a single scalar over
    every value of the cube, pooling across wavelengths rather than keeping a
    per-channel statistic: ``SPECTRA_MEAN=0.031223`` and ``SPECTRA_STD=0.049720`` are
    one number each, computed over the whole training set. A per-channel scaler cannot
    reproduce that pooled statistic, so this node fits its own single-feature
    accumulator instead of wrapping or subclassing either pretreatment node.

    ``statistical_initialization`` streams every value of ``cube`` (flattened across
    batch, height, width and channel) through one single-feature
    ``WelfordAccumulator``; the fitted mean and standard deviation are copied into
    two scalar float32 buffers. ``forward`` applies ``(cube - mean) / std`` and keeps
    the input shape. There is no target-side standardization here: the model wrapper
    that consumes this node's output owns standardizing its own regression target.

    The course fits its preprocessing on the training split only (splits 2 to 5, 97
    bellies) and holds split 1 out (25 bellies) for validation. ``PLSRegression``
    already has this notion built in through its optional ``fold`` input and
    ``holdout_folds`` parameter, because it needs the fold ids to pick its own
    component count by held-out error. ``GlobalScalarStandardizer`` has no such
    need on its own; the only reason it also accepts ``exclude_folds`` and an
    optional ``fold`` input is that the statistical trainer feeds every node in a
    pipeline the same train loader, and that loader must carry every split
    (including the held-out one) so ``PLSRegression`` can see it. Without
    ``exclude_folds`` this node would then standardize using the held-out frames
    too, which is not what the course does. Setting ``exclude_folds`` lets this
    node opt out of exactly the frames the course never fits its scaling on,
    while the loader composition stays the same for every node in the pipeline.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom for the standard deviation: ``0`` for the population
        standard deviation (``numpy.std`` default, matching how the course computed
        ``SPECTRA_STD``) or ``1`` for the Bessel-corrected sample standard deviation
        (``WelfordAccumulator``'s own default). Default ``0``.
    eps : float, optional
        Floor applied to the fitted standard deviation before dividing, so a
        pathological near-constant input cannot divide by (near) zero (default 1e-6).
    exclude_folds : list[int] or None, optional
        Fold ids to leave out of the fit, matching the course's train/validation
        split. When set, every batch in ``statistical_initialization`` must carry a
        ``fold`` tensor; frames whose fold id is in this list do not enter the
        accumulator. When ``None`` (default), every frame is used and a ``fold``
        input, if present, is ignored.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.NORMALIZATION, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube, BHWC float32.",
        ),
        "fold": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame fold id; required when exclude_folds is set.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Standardized cube, same shape as the input.",
        ),
    }

    def __init__(
        self,
        ddof: int = 0,
        eps: float = 1e-6,
        exclude_folds: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {ddof}")
        self.ddof = int(ddof)
        self.eps = float(eps)
        self.exclude_folds = list(exclude_folds) if exclude_folds is not None else None
        super().__init__(ddof=self.ddof, eps=self.eps, exclude_folds=self.exclude_folds, **kwargs)
        self.register_buffer("mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("std", torch.ones((), dtype=torch.float32))

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit one scalar mean and standard deviation over every value of ``cube``.

        When ``exclude_folds`` is set, every batch must carry a ``fold`` tensor and
        only the frames whose fold id is not in ``exclude_folds`` enter the
        accumulator; a batch with no kept frame is skipped.

        Parameters
        ----------
        input_stream : InputStream
            Iterable of port-keyed batch dicts carrying a ``cube`` tensor, and a
            ``fold`` tensor when ``exclude_folds`` is set.
        """
        acc = WelfordAccumulator(n_features=1, track_covariance=False)
        for batch in input_stream:
            cube = batch["cube"]
            if self.exclude_folds is not None:
                fold = batch.get("fold")
                if fold is None:
                    raise ValueError(
                        f"{type(self).__name__} requires a 'fold' input in every batch "
                        "when 'exclude_folds' is set."
                    )
                keep = torch.isin(
                    fold, torch.as_tensor(self.exclude_folds, dtype=fold.dtype), invert=True
                )
                if not torch.any(keep):
                    continue
                cube = cube[keep]
            acc.update(cube.reshape(-1, 1).to(torch.float32))

        mean = acc.mean.reshape(())
        if self.ddof == 1:
            std = acc.std.reshape(())
        else:
            # WelfordAccumulator.var is Bessel-corrected (ddof=1); rescale to ddof=0
            # with the sample count it already tracked, rather than re-deriving a
            # second accumulator.
            n = acc.count
            var_ddof1 = acc.var.reshape(())
            std = (var_ddof1 * (n - 1) / n).clamp_min(0.0).sqrt()

        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(self.eps))
        self._statistically_initialized = True

    def forward(self, cube: Tensor, fold: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Apply ``(cube - mean) / std`` with the fitted scalars.

        Parameters
        ----------
        cube : Tensor
            Hyperspectral cube, BHWC float32.
        fold : Tensor or None, optional
            Accepted for port-wiring symmetry with ``statistical_initialization``,
            but unused here: ``exclude_folds`` only ever affects fitting.

        Returns
        -------
        dict[str, Tensor]
            ``{"cube": standardized}`` with the same shape as the input.
        """
        if not self._statistically_initialized:
            raise RuntimeError(f"{type(self).__name__} requires statistical_initialization first")
        return {"cube": (cube - self.mean) / self.std}


__all__ = ["GlobalScalarStandardizer"]
