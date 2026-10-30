"""Spectral pretreatment nodes for the course chemometrics pipelines.

Both nodes reduce or filter the channel axis of a hyperspectral cube with a
fixed, data-independent linear operator (no statistical initialization): the
operator is built once from constructor hyperparameters and the forward pass
is pure ``torch`` so the nodes trace and export like any stateless transform.
"""

from __future__ import annotations

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from scipy.signal import savgol_filter


class SpectralBinning(Node):
    """Average adjacent channels of a hyperspectral cube into coarser bands.

    Groups the channel axis into consecutive, non-overlapping windows of
    ``bin_size`` channels and replaces each window with its mean, following
    the course's band-binning step (124 bands above 747.92 nm binned pairwise
    into 62 bands before the spectral front-end). A channel count that is not
    a multiple of ``bin_size`` drops the trailing remainder, the same
    truncation ``numpy.reshape`` performs on an even split; the course's own
    124-to-62 case is exact (124 / 2 = 62) so no remainder is ever dropped
    there, and ``input_channels`` must still be given up front so the output
    channel count and the optional wavelength binning are known at
    construction time.

    An optional ``wavelengths`` port carries the per-channel wavelengths (in
    nanometres, the convention used by the builtin channel selectors) through
    the same binning, so a downstream node that needs band centers keeps a
    consistent axis after the reduction.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Optional per-channel wavelengths [C] in nanometers, binned "
            "the same way as the cube.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Binned cube [B, H, W, C // bin_size].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Binned wavelengths [C // bin_size]; only present when the "
            "wavelengths input is connected.",
            optional=True,
        ),
    }

    def __init__(self, input_channels: int, bin_size: int = 2, **kwargs) -> None:
        """Create the node.

        Parameters
        ----------
        input_channels : int
            Number of channels the input cube carries; fixes the output
            channel count (``input_channels // bin_size``) up front.
        bin_size : int, optional
            Number of adjacent channels averaged into one output band
            (default: 2, the course's pairwise binning).
        """
        if input_channels < 1:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if bin_size < 1:
            raise ValueError(f"bin_size must be positive, got {bin_size}")
        super().__init__(input_channels=input_channels, bin_size=bin_size, **kwargs)
        self.input_channels = int(input_channels)
        self.bin_size = int(bin_size)
        self.n_bins = self.input_channels // self.bin_size
        self.n_used = self.n_bins * self.bin_size

    def forward(self, cube: torch.Tensor, wavelengths=None, **_) -> dict[str, torch.Tensor]:
        """Average every ``bin_size`` adjacent channels; drop the trailing remainder."""
        B, H, W, _ = cube.shape
        trimmed = cube[..., : self.n_used]
        binned = trimmed.reshape(B, H, W, self.n_bins, self.bin_size).mean(dim=-1)
        out: dict[str, torch.Tensor] = {"cube": binned}
        if wavelengths is not None:
            # Wavelengths are a np.int32 port (the builtin channel-selector
            # convention), not necessarily a torch tensor at runtime, so this
            # branch works in numpy; the cube path above stays torch-only.
            wl = np.asarray(wavelengths, dtype=np.float64)[: self.n_used]
            binned_wl = wl.reshape(self.n_bins, self.bin_size).mean(axis=-1)
            out["wavelengths"] = binned_wl.astype(np.int32)
        return out


class SavitzkyGolayInterp(Node):
    """Savitzky-Golay filter with SciPy's ``mode="interp"`` edge handling.

    Builds the exact linear operator of
    ``scipy.signal.savgol_filter(x, window_length, polyorder, deriv, delta,
    mode="interp")`` along the channel axis, by filtering the identity matrix
    once at construction time, and applies it as a single matrix multiply in
    ``forward``. ``cuvis_ai.node.pretreatments.savitzky_golay.SavitzkyGolay``
    was checked first and does not cover this case: its docstring states "The
    SciPy coefficients are convolution-oriented... This filter does not
    reproduce SciPy's ``mode="interp"`` boundary handling" (fixed convolution
    kernel, replicate/reflect/constant padding only). The course's PLS
    pretreatment chain filters all 62 bands with ``mode="interp"`` (the SciPy
    default), including the edge bands, so a fixed kernel with padding gives a
    different result there; this node exists to reproduce the edge behavior
    exactly, not just the interior.

    The ``[C, C]`` operator is built once in ``__init__`` with plain SciPy
    calls (constants, not part of the data path) and stored as a buffer;
    ``forward`` is one ``cube @ operator``, broadcasting over batch, height
    and width, so the node traces and exports like any linear transform.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH, NodeTag.DIFFERENTIABLE}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C], C == num_channels.",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Filtered cube [B, H, W, C], same shape as the input.",
        ),
    }

    def __init__(
        self,
        num_channels: int,
        window_length: int = 7,
        polyorder: int = 2,
        deriv: int = 2,
        delta: float = 1.0,
        **kwargs,
    ) -> None:
        """Create the node and build its ``[C, C]`` linear operator.

        Parameters
        ----------
        num_channels : int
            Number of spectral channels the operator is built for; the input
            cube's last dimension must equal this.
        window_length : int, optional
            SciPy ``window_length``, must be odd (default: 7).
        polyorder : int, optional
            SciPy ``polyorder``, must be less than ``window_length`` (default: 2).
        deriv : int, optional
            SciPy ``deriv``, the order of derivative to compute (default: 2).
        delta : float, optional
            SciPy ``delta``, the sample spacing used when ``deriv > 0`` (default: 1.0).
        """
        if num_channels < 1:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if window_length % 2 == 0:
            raise ValueError(f"window_length must be odd, got {window_length}")
        if polyorder >= window_length:
            raise ValueError(
                f"polyorder ({polyorder}) must be less than window_length ({window_length})"
            )
        super().__init__(
            num_channels=num_channels,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            **kwargs,
        )
        self.num_channels = int(num_channels)
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.delta = float(delta)

        identity = np.eye(self.num_channels, dtype=np.float64)
        # Filtering the identity matrix row by row gives, for each row i (the
        # unit spectrum e_i), the filtered output SG(e_i), which is column i
        # of the linear operator A satisfying SG(x) = A @ x for a single
        # spectrum x. So filtered_rows[i] == A[:, i], i.e. filtered_rows == A.T.
        # For a batch of row-vector spectra X [N, C], SG(X) = X @ A.T == X @ filtered_rows:
        # the operator applied on the right is filtered_rows itself, no transpose.
        filtered_rows = savgol_filter(
            identity,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=-1,
            mode="interp",
        )
        operator = torch.tensor(filtered_rows, dtype=torch.float32)
        self.register_buffer("operator", operator)

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Apply the Savitzky-Golay operator: ``cube @ operator`` along the channel axis."""
        filtered = cube @ self.operator.to(dtype=cube.dtype)
        return {"cube": filtered}
