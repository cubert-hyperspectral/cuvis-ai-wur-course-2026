"""Shared synthetic fixtures. No downloads, no course data: pure tensors."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def tiny_cube() -> torch.Tensor:
    """A small BHWC float32 cube [2, 4, 4, 5] with positive values."""
    return torch.rand(2, 4, 4, 5, dtype=torch.float32) + 0.1


@pytest.fixture
def spectra_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Mean spectra as 1 x 1 frames: x [N, 1, 1, C] and a one-column target y [N, 1]."""
    n, c = 40, 16
    x = torch.rand(n, 1, 1, c, dtype=torch.float32)
    coef = torch.linspace(-1.0, 1.0, c)
    y = (x.reshape(n, c) @ coef).unsqueeze(1) + 0.05 * torch.randn(n, 1)
    return x, y.to(torch.float32)
