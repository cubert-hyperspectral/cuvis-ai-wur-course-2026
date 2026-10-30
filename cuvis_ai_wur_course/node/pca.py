"""Dataset-level PCA projection node for the WUR course pipelines."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai.utils.welford import WelfordAccumulator
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class DatasetPCA(Node):
    """Project a cube onto principal components fitted once over a whole dataset pass.

    ``cuvis_ai.node.dimensionality_reduction.PCA`` fits independently on each frame it
    sees in ``forward`` (a fresh eigendecomposition per call, so its components are not
    stable from one frame to the next and it never looks past the current batch).
    ``cuvis_ai.node.dimensionality_reduction.TrainablePCA`` is close to what this node
    needs (a Welford-accumulated dataset covariance, eigendecomposed once in
    ``statistical_initialization``, buffers for mean/components/explained variance) but
    makes no sign-convention guarantee: ``torch.linalg.eigh`` returns whichever sign an
    eigenvector happens to have, which both ``TrainablePCA`` and its own
    ``FixedPCAProjection`` subclass note as a problem elsewhere in that module.
    ``DatasetPCA`` fixes the sign of every component after eigendecomposition (the
    largest-magnitude entry of each component is made positive), which
    ``sklearn.decomposition.PCA`` does not guarantee either, so the parity test below
    compares projections up to sign rather than exactly. ``TrainablePCA`` also declares
    ``TRAINABLE_BUFFERS`` for gradient fine-tuning; this node has none, since a course
    exploration projection is meant to stay the fixed dataset decomposition it was fit
    as, with an explicit ``input_channels``/``n_components`` constructor rather than
    ``TrainablePCA``'s ``num_channels``.

    ``statistical_initialization`` streams every pixel spectrum of the pass through one
    covariance-tracking ``WelfordAccumulator``, eigendecomposes the resulting
    ``[C, C]`` covariance in float64 for numerical stability, keeps the top
    ``n_components`` eigenvectors sorted by descending eigenvalue, and fixes their
    sign. ``forward`` centers the cube by the fitted mean and projects it onto the
    fitted components, and also emits ``explained_variance_ratio`` so
    ``cuvis_ai.node.metrics.ExplainedVarianceMetric`` (whose single input port is
    named ``explained_variance_ratio``) can read it directly.

    The course fits its preprocessing on the training split only (splits 2 to 5, 97
    bellies) and holds split 1 out (25 bellies) for validation. ``PLSRegression``
    already carries this notion through its optional ``fold`` input and
    ``holdout_folds`` parameter, because it needs the fold ids to pick its own
    component count by held-out error. ``DatasetPCA`` has no such need on its own;
    the only reason it also accepts ``exclude_folds`` and an optional ``fold``
    input is that the statistical trainer feeds every node in a pipeline the same
    train loader, and that loader must carry every split (including the held-out
    one) so ``PLSRegression`` can see it. Without ``exclude_folds`` this node would
    then fit its covariance using the held-out frames too, which is not what the
    course does. Setting ``exclude_folds`` lets this node opt out of exactly the
    frames the course never fits its decomposition on, while the loader
    composition stays the same for every node in the pipeline.

    Parameters
    ----------
    input_channels : int
        Number of channels of the incoming cube; sizes the ``mean`` and ``components``
        buffers eagerly.
    n_components : int
        Number of principal components to keep.
    eps : float, optional
        Floor added to the sum of retained eigenvalues before normalizing them into
        ``explained_variance_ratio``, guarding against division by zero on a
        degenerate (all-zero) fit (default 1e-6).
    exclude_folds : list[int] or None, optional
        Fold ids to leave out of the fit, matching the course's train/validation
        split. When set, every batch in ``statistical_initialization`` must carry a
        ``fold`` tensor; frames whose fold id is in this list do not enter the
        accumulator. When ``None`` (default), every frame is used and a ``fold``
        input, if present, is ignored.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.DIM_REDUCTION, NodeTag.PREPROCESSING, NodeTag.TORCH}
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
            shape=(-1, -1, -1, "n_components"),
            description="Cube projected onto the fitted principal components.",
        ),
        "explained_variance_ratio": PortSpec(
            dtype=torch.float32,
            shape=("n_components",),
            description="Fraction of total variance explained by each retained component.",
        ),
    }

    def __init__(
        self,
        input_channels: int,
        n_components: int,
        eps: float = 1e-6,
        exclude_folds: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if n_components > input_channels:
            raise ValueError(
                f"n_components ({n_components}) cannot exceed input_channels ({input_channels})"
            )
        self.input_channels = int(input_channels)
        self.n_components = int(n_components)
        self.eps = float(eps)
        self.exclude_folds = list(exclude_folds) if exclude_folds is not None else None
        super().__init__(
            input_channels=self.input_channels,
            n_components=self.n_components,
            eps=self.eps,
            exclude_folds=self.exclude_folds,
            **kwargs,
        )
        # Private buffer names: "cube" and "explained_variance_ratio" are already output
        # port names, and Node.__getattr__ resolves those to the port proxy, which makes
        # nn.Module.register_buffer see them as already-existing attributes.
        self.register_buffer("_mean", torch.zeros(self.input_channels, dtype=torch.float32))
        self.register_buffer(
            "_components",
            torch.zeros(self.input_channels, self.n_components, dtype=torch.float32),
        )
        self.register_buffer(
            "_explained_variance_ratio", torch.zeros(self.n_components, dtype=torch.float32)
        )

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit the mean, components and explained-variance ratio over the whole pass.

        When ``exclude_folds`` is set, every batch must carry a ``fold`` tensor and
        only the frames whose fold id is not in ``exclude_folds`` enter the
        accumulator; a batch with no kept frame is skipped.

        Parameters
        ----------
        input_stream : InputStream
            Iterable of port-keyed batch dicts carrying a ``cube`` tensor, and a
            ``fold`` tensor when ``exclude_folds`` is set.
        """
        acc = WelfordAccumulator(n_features=self.input_channels, track_covariance=True)
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
            acc.update(cube.reshape(-1, self.input_channels).to(torch.float32))

        mean = acc.mean.to(torch.float64)
        cov = acc.cov.to(torch.float64)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # eigh returns ascending eigenvalues; flip to descending and keep the top k.
        eigenvalues = eigenvalues.flip(0)[: self.n_components]
        eigenvectors = eigenvectors.flip(1)[:, : self.n_components]

        # Deterministic sign convention: make the largest-magnitude entry of each
        # component positive, so repeated fits (and comparisons against a library
        # reference) do not disagree only on an arbitrary overall sign.
        max_idx = eigenvectors.abs().argmax(dim=0)
        signs = torch.sign(eigenvectors[max_idx, torch.arange(self.n_components)])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        eigenvectors = eigenvectors * signs

        total_variance = torch.diagonal(cov).sum()
        explained_ratio = eigenvalues / (total_variance + self.eps)

        self._mean.copy_(mean.to(torch.float32))
        self._components.copy_(eigenvectors.to(torch.float32))
        self._explained_variance_ratio.copy_(explained_ratio.to(torch.float32))
        self._statistically_initialized = True

    def forward(self, cube: Tensor, fold: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Center the cube by the fitted mean and project it onto the fitted components.

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
            ``{"cube": projected, "explained_variance_ratio": ratio}``; ``projected``
            has shape ``[B, H, W, n_components]``.
        """
        if not self._statistically_initialized:
            raise RuntimeError(f"{type(self).__name__} requires statistical_initialization first")
        b, h, w, _ = cube.shape
        centered = cube - self._mean
        projected = centered @ self._components
        return {
            "cube": projected.reshape(b, h, w, self.n_components),
            "explained_variance_ratio": self._explained_variance_ratio,
        }


__all__ = ["DatasetPCA"]
