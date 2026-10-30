"""Tests for ``cuvis_ai_wur_course.data.course_npz.CourseNpzDataModule``.

Pure synthetic frames written with ``write_course_frame``, no downloads.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

from cuvis_ai_wur_course.data.course_npz import CourseNpzDataModule, write_course_frame

pytestmark = pytest.mark.unit


# ----- helpers -----------------------------------------------------------


def _write_universe_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv_path = tmp_path / "universe.csv"
    fieldnames = ["source", "index", "materialized_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _build_mean_spectra_universe(tmp_path: Path, n: int = 6, c: int = 4) -> Path:
    """n 1x1xC frames, each with a target, a fold and no mask/group/quant."""
    rows = []
    for i in range(n):
        cube = np.full((1, 1, c), fill_value=float(i), dtype=np.float32)
        wl = np.linspace(747.92, 1015.78, c).astype(np.float32)
        fname = tmp_path / f"frame_{i}.npz"
        write_course_frame(
            fname,
            cube=cube,
            wavelengths=wl,
            target=np.array([10.0 * i], dtype=np.float32),
            fold=i % 3,
        )
        rows.append({"source": "mean_spectra", "index": str(i), "materialized_path": fname.name})
    return _write_universe_csv(tmp_path, rows)


def _build_image_universe(tmp_path: Path, n: int = 3, h: int = 5, w: int = 6, c: int = 4) -> Path:
    """n image frames with mask, group and quant scale/bias, no target."""
    rows = []
    for i in range(n):
        cube = np.random.rand(h, w, c).astype(np.float32)
        mask = (np.random.rand(h, w) > 0.5).astype(np.int32)
        wl = np.linspace(747.92, 1015.78, c).astype(np.float32)
        fname = tmp_path / f"slice_{i}.npz"
        write_course_frame(
            fname,
            cube=cube,
            wavelengths=wl,
            mask=mask,
            fold=i,
            group=100 + (i // 2),  # two slices share a belly id
            quant_scale=1000.0 + i,
            quant_bias=0.01 * i,
        )
        rows.append({"source": "image_slices", "index": str(i), "materialized_path": fname.name})
    return _write_universe_csv(tmp_path, rows)


def _dm_from_csv(csv_path: Path, n: int, batch_size: int = 1) -> CourseNpzDataModule:
    splits = DataSplitConfig(
        train=[
            Selector(
                kind=SelectorKind.FILE_INDICES, source=_source_of(csv_path), ids=list(range(n))
            )
        ]
    )
    return CourseNpzDataModule(universe_csv=str(csv_path), splits=splits, batch_size=batch_size)


def _source_of(csv_path: Path) -> str:
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader)["source"]


# ----- 1. registry name ----------------------------------------------------


def test_registry_name() -> None:
    assert CourseNpzDataModule.DATA_MODULE_NAME == "wur_course_npz"


# ----- 2. mean-spectra frames: target + fold, no mask/group/quant ---------


def test_mean_spectra_frames_round_trip(tmp_path: Path) -> None:
    csv_path = _build_mean_spectra_universe(tmp_path, n=6, c=4)
    dm = _dm_from_csv(csv_path, n=6, batch_size=2)
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))

    assert tuple(batch["cube"].shape) == (2, 1, 1, 4)
    assert batch["cube"].dtype.is_floating_point
    assert tuple(batch["target"].shape) == (2, 1)
    assert tuple(batch["fold"].shape) == (2,)
    assert "group" not in batch
    assert "quant_scale" not in batch
    assert "quant_bias" not in batch
    # mask is always emitted (zeros when absent), matching the base reader's contract.
    assert tuple(batch["mask"].shape) == (2, 1, 1)
    assert int(batch["mask"].abs().sum()) == 0


def test_wavelengths_stay_float32_not_rounded(tmp_path: Path) -> None:
    """The course's 124-channel window is a fractional np.linspace; the base
    class' int32 wavelength cast would round 747.92 -> 748, losing the exact
    axis. This module keeps the float array instead (documented deviation)."""
    csv_path = _build_mean_spectra_universe(tmp_path, n=2, c=4)
    dm = _dm_from_csv(csv_path, n=2, batch_size=1)
    dm.setup(stage="fit")
    ds = dm.train_ds
    assert ds.wavelengths_nm.dtype == np.float32
    assert not np.allclose(ds.wavelengths_nm, np.round(ds.wavelengths_nm))


# ----- 3. image frames: mask, group, quant scale/bias, no target ----------


def test_image_frames_round_trip(tmp_path: Path) -> None:
    csv_path = _build_image_universe(tmp_path, n=3, h=5, w=6, c=4)
    dm = _dm_from_csv(
        csv_path, n=3, batch_size=1
    )  # image frames: batch size 1 (per-frame mask size)
    dm.setup(stage="fit")
    batches = list(dm.train_dataloader())
    assert len(batches) == 3

    b0 = batches[0]
    assert tuple(b0["cube"].shape) == (1, 5, 6, 4)
    assert tuple(b0["mask"].shape) == (1, 5, 6)
    assert b0["mask"].dtype == __import__("torch").int32
    assert "target" not in b0
    assert tuple(b0["group"].shape) == (1,)
    assert tuple(b0["quant_scale"].shape) == (1,)
    assert tuple(b0["quant_bias"].shape) == (1,)
    assert float(b0["quant_scale"][0]) == pytest.approx(1000.0)
    assert float(b0["quant_bias"][0]) == pytest.approx(0.0)


def test_image_frames_group_shared_by_belly(tmp_path: Path) -> None:
    csv_path = _build_image_universe(tmp_path, n=4, h=3, w=3, c=2)
    dm = _dm_from_csv(csv_path, n=4, batch_size=1)
    dm.setup(stage="fit")
    # train_dataloader shuffles; sort for a deterministic assertion.
    groups = sorted(int(b["group"][0]) for b in dm.train_dataloader())
    assert groups == [100, 100, 101, 101]


# ----- 4. selector-driven subset selection ---------------------------------


def test_file_indices_selector_picks_subset(tmp_path: Path) -> None:
    csv_path = _build_mean_spectra_universe(tmp_path, n=6, c=3)
    source = _source_of(csv_path)
    splits = DataSplitConfig(
        train=[Selector(kind=SelectorKind.FILE_INDICES, source=source, ids=[0, 1, 2])],
        test=[Selector(kind=SelectorKind.FILE_INDICES, source=source, ids=[3, 4, 5])],
    )
    dm = CourseNpzDataModule(universe_csv=str(csv_path), splits=splits, batch_size=1)
    dm.setup()
    train_targets = sorted(float(b["target"][0, 0]) for b in dm.train_dataloader())
    test_targets = sorted(float(b["target"][0, 0]) for b in dm.test_dataloader())
    assert train_targets == [0.0, 10.0, 20.0]
    assert test_targets == [30.0, 40.0, 50.0]


# ----- 5. missing optional key yields no value, not a zero fill -----------


def test_missing_optional_key_absent_not_zero(tmp_path: Path) -> None:
    csv_path = _build_mean_spectra_universe(tmp_path, n=2, c=3)  # no group / quant keys written
    dm = _dm_from_csv(csv_path, n=2, batch_size=2)
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert "group" not in batch
    assert "quant_scale" not in batch
    assert "quant_bias" not in batch
