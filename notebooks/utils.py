"""Thin matplotlib rendering and table helpers for the course notebooks.

Nothing here builds or wires a pipeline and nothing provisions data: the notebooks do
that in the open. These functions only draw what the nodes produced and lay the
reproduction rows out as a table.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def show_images(panels: Sequence[tuple[str, torch.Tensor | np.ndarray]], cmap: str | None = None):
    """Draw [H, W] or [H, W, 3] panels side by side with their titles."""
    _, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, image) in zip(np.atleast_1d(axes), panels, strict=True):
        array = _to_numpy(image)
        if array.ndim == 2 and cmap is None:
            ax.imshow(array, cmap="nipy_spectral", interpolation="nearest")
        else:
            ax.imshow(np.clip(array, 0, None) if array.ndim == 3 else array, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_map(
    chem_map: torch.Tensor | np.ndarray, title: str, vmin: float = 0.0, vmax: float = 100.0
):
    """Draw one chemical map [H, W] in percent with a colour bar."""
    array = _to_numpy(chem_map).squeeze()
    plt.figure(figsize=(5, 8))
    plt.imshow(array, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="fat (%)")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_spectra(
    wavelengths: torch.Tensor | np.ndarray,
    spectra: torch.Tensor | np.ndarray,
    title: str,
    reference: torch.Tensor | np.ndarray | None = None,
    labels: Sequence[str] | None = None,
):
    """Plot spectra [N, C] over a wavelength axis, with an optional reference spectrum."""
    x = _to_numpy(wavelengths).ravel()
    y = _to_numpy(spectra).reshape(-1, x.shape[0])
    plt.figure(figsize=(8, 4))
    for i, row in enumerate(y):
        plt.plot(x, row, lw=0.8, label=None if labels is None else labels[i])
    if reference is not None:
        plt.plot(x, _to_numpy(reference).ravel(), color="orange", lw=2.0, label="reference")
    plt.xlabel("wavelength (nm)")
    plt.title(title)
    if labels is not None or reference is not None:
        plt.legend(fontsize=8)
    plt.show()


def plot_component_selection(cv_sse: torch.Tensor | np.ndarray, selected: int, title: str):
    """Plot the pooled hold-out SSE per component count and mark the chosen count."""
    sse = _to_numpy(cv_sse).ravel()
    counts = np.arange(1, sse.shape[0] + 1)
    plt.figure(figsize=(6, 3.5))
    plt.plot(counts, sse, ".-")
    plt.axvline(selected, color="red", ls="--", label=f"selected: {selected}")
    plt.xlabel("components")
    plt.ylabel("pooled hold-out SSE")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_predictions(
    target: torch.Tensor | np.ndarray,
    predictions: dict[str, torch.Tensor | np.ndarray],
    title: str,
):
    """Scatter predicted against measured values for several models."""
    y = _to_numpy(target).ravel()
    plt.figure(figsize=(5, 5))
    for name, pred in predictions.items():
        plt.scatter(y, _to_numpy(pred).ravel(), s=14, label=name)
    lo, hi = float(y.min()), float(y.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    plt.xlabel("measured")
    plt.ylabel("predicted")
    plt.title(title)
    plt.legend()
    plt.show()


def reproduction_table(
    rows: Sequence[tuple[str, float | str, float | str, float | None]],
    synthetic: bool,
) -> pd.DataFrame:
    """Lay out (row, course value, ours, tolerance) as a table with a pass column.

    With ``synthetic`` the data is the stand-in and no row can pass or fail: the pass
    column says so instead of judging numbers that mean nothing.
    """
    records = []
    for name, course, ours, tolerance in rows:
        if synthetic:
            verdict = "synthetic data, not compared"
        elif isinstance(course, str) and isinstance(ours, str) and tolerance == 0:
            verdict = "pass" if course == ours else "open deviation"
        elif (
            tolerance is None
            or not isinstance(course, (int, float))
            or not isinstance(ours, (int, float))
        ):
            verdict = "reported"
        else:
            verdict = "pass" if abs(float(course) - float(ours)) <= tolerance else "open deviation"
        records.append(
            {"row": name, "course": course, "ours": ours, "tolerance": tolerance, "status": verdict}
        )
    return pd.DataFrame.from_records(records)
