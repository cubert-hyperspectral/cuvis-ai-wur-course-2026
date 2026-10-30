"""Round-trip tests for ``cuvis_ai_wur_course.prepare_data``: prepare on a synthetic
course folder, then read back through ``CourseNpzDataModule``.

Every test builds its own small synthetic input (scipy.io.savemat, numpy,
pandas) in ``tmp_path``; no course data, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
from click.testing import CliRunner
from cuvis_ai_core.data.splits_io import load_splits
from sklearn.model_selection import KFold, train_test_split

from cuvis_ai_wur_course.data.course_npz import CourseNpzDataModule
from cuvis_ai_wur_course.prepare_data import main

pytestmark = pytest.mark.unit


def _run(args: list[str]) -> None:
    runner = CliRunner()
    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output


def _dm_for(out_dir: Path, batch_size: int = 1) -> CourseNpzDataModule:
    splits = load_splits(out_dir / "splits.json")
    return CourseNpzDataModule(
        universe_csv=str(out_dir / "universe.csv"), splits=splits, batch_size=batch_size
    )


# ----- (1) pork features: spatial_spectra_pig.mat -------------------------


def test_pork_features_round_trip(tmp_path: Path) -> None:
    n = 20
    rng = np.random.default_rng(0)
    spatial = rng.random((n, 6), dtype=np.float64).astype(np.float32)
    spectral = rng.random((n, 4), dtype=np.float64).astype(np.float32)
    ref = rng.random((n, 1), dtype=np.float64).astype(np.float32) * 50.0

    mat_path = tmp_path / "spatial_spectra_pig.mat"
    scipy.io.savemat(mat_path, {"spatial": spatial, "spectral": spectral, "ref": ref})
    out_dir = tmp_path / "out_pork"

    _run(["pork-features", "--mat-path", str(mat_path), "--out-dir", str(out_dir)])

    dm = _dm_for(out_dir)
    dm.setup()
    train_ds, test_ds = dm.train_ds, dm.test_ds

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    assert len(train_ds) == len(train_idx)
    assert len(test_ds) == len(test_idx)

    # Channel layout: spatial (6) then spectral (4) = 10, matching [spatial | spectral].
    sample0 = train_ds[0]
    assert tuple(sample0["cube"].shape) == (1, 1, 10)
    expected_cube = np.concatenate([spatial[train_idx[0]], spectral[train_idx[0]]])
    assert np.allclose(sample0["cube"][0, 0], expected_cube, atol=1e-6)
    assert float(sample0["target"][0]) == pytest.approx(float(ref[train_idx[0], 0]), abs=1e-5)

    # Fold assignment on the train part == KFold(5) over the train_test_split order.
    kf = KFold(n_splits=5, shuffle=False)
    expected_fold = np.full(len(train_idx), -1, dtype=np.int32)
    for fold_id, (_, val_pos) in enumerate(kf.split(train_idx)):
        expected_fold[val_pos] = fold_id
    actual_fold = [int(train_ds[i]["fold"]) for i in range(len(train_ds))]
    assert actual_fold == expected_fold.tolist()

    # Test frames carry no fold key.
    assert all("fold" not in test_ds[i] for i in range(len(test_ds)))


def test_pork_features_small_input_caps_fold_count(tmp_path: Path) -> None:
    """Fewer than 5 train bellies: the fold count caps at the train size instead of
    KFold raising (observed with n=6 total, which gives 4 train rows)."""
    n = 6
    rng = np.random.default_rng(4)
    spatial = rng.random((n, 6), dtype=np.float64).astype(np.float32)
    spectral = rng.random((n, 4), dtype=np.float64).astype(np.float32)
    ref = rng.random((n, 1), dtype=np.float64).astype(np.float32) * 50.0

    mat_path = tmp_path / "spatial_spectra_pig_small.mat"
    scipy.io.savemat(mat_path, {"spatial": spatial, "spectral": spectral, "ref": ref})
    out_dir = tmp_path / "out_pork_small"

    _run(["pork-features", "--mat-path", str(mat_path), "--out-dir", str(out_dir)])

    dm = _dm_for(out_dir)
    dm.setup()
    train_ds = dm.train_ds

    train_idx, _test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    n_splits = min(5, len(train_idx))
    kf = KFold(n_splits=n_splits, shuffle=False)
    expected_fold = np.full(len(train_idx), -1, dtype=np.int32)
    for fold_id, (_, val_pos) in enumerate(kf.split(train_idx)):
        expected_fold[val_pos] = fold_id
    actual_fold = [int(train_ds[i]["fold"]) for i in range(len(train_ds))]
    assert actual_fold == expected_fold.tolist()
    assert max(actual_fold) < 5


# ----- (2) notebook 03 mean spectra ----------------------------------------


def test_mean_spectra_03_round_trip(tmp_path: Path) -> None:
    n = 12
    rng = np.random.default_rng(1)
    meat_ids = list(range(100, 100 + n))
    splits_col = [0, 1, 2, 3, 4, 5] * 2
    fat = rng.random(n) * 40.0 + 10.0

    df = pd.DataFrame({"meat_id": meat_ids, "Fat_chem": fat, "split": splits_col})
    csv_path = tmp_path / "red_data_with_splits.csv"
    df.to_csv(csv_path, index=False)

    spectra_dir = tmp_path / "meat_id_mean_spectra" / "absorbance"
    spectra_dir.mkdir(parents=True)
    raw_spectra = {}
    for m in meat_ids:
        spec = rng.random(300).astype(np.float32)
        raw_spectra[m] = spec
        np.save(spectra_dir / f"{m}.npy", spec)

    out_dir = tmp_path / "out_03"
    _run(
        [
            "mean-spectra-03",
            "--csv-path",
            str(csv_path),
            "--spectra-dir",
            str(spectra_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    dm = _dm_for(out_dir)
    dm.setup()
    train_ds, val_ds, test_ds = dm.train_ds, dm.val_ds, dm.test_ds

    train_rows = [i for i, s in enumerate(splits_col) if s in (1, 2, 3, 4, 5)]
    val_rows = [i for i, s in enumerate(splits_col) if s == 1]
    test_rows = [i for i, s in enumerate(splits_col) if s == 0]
    assert len(train_ds) == len(train_rows)
    assert len(val_ds) == len(val_rows)
    assert len(test_ds) == len(test_rows)

    sample0 = train_ds[0]
    assert tuple(sample0["cube"].shape) == (1, 1, 124)
    row0 = train_rows[0]
    expected = raw_spectra[meat_ids[row0]][-124:]
    assert np.allclose(sample0["cube"][0, 0], expected, atol=1e-6)
    assert int(sample0["fold"]) == splits_col[row0]
    assert int(sample0["group"]) == meat_ids[row0]


def test_mean_spectra_03_writes_gradient_splits(tmp_path: Path) -> None:
    """``splits_gradient.json`` holds out split 1 from train (course gradient split),
    unlike ``splits.json`` which keeps split 1 in train for the statistical hold-out fits."""
    n = 12
    rng = np.random.default_rng(1)
    meat_ids = list(range(100, 100 + n))
    splits_col = [0, 1, 2, 3, 4, 5] * 2
    fat = rng.random(n) * 40.0 + 10.0

    df = pd.DataFrame({"meat_id": meat_ids, "Fat_chem": fat, "split": splits_col})
    csv_path = tmp_path / "red_data_with_splits.csv"
    df.to_csv(csv_path, index=False)

    spectra_dir = tmp_path / "meat_id_mean_spectra" / "absorbance"
    spectra_dir.mkdir(parents=True)
    for m in meat_ids:
        np.save(spectra_dir / f"{m}.npy", rng.random(300).astype(np.float32))

    out_dir = tmp_path / "out_03_gradient"
    _run(
        [
            "mean-spectra-03",
            "--csv-path",
            str(csv_path),
            "--spectra-dir",
            str(spectra_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    gradient_splits = load_splits(out_dir / "splits_gradient.json")
    train_ids = gradient_splits.train[0].ids
    val_ids = gradient_splits.val[0].ids
    test_ids = gradient_splits.test[0].ids

    expected_train = [i for i, s in enumerate(splits_col) if s in (2, 3, 4, 5)]
    expected_val = [i for i, s in enumerate(splits_col) if s == 1]
    expected_test = [i for i, s in enumerate(splits_col) if s == 0]
    assert train_ids == expected_train
    assert val_ids == expected_val
    assert test_ids == expected_test
    assert set(train_ids).isdisjoint(val_ids)


# ----- (3) notebook 02 image slices -----------------------------------------


def test_image_slices_round_trip(tmp_path: Path) -> None:
    meat_ids = [10, 70]
    n_slices_per_belly = 2
    h, w, c = 8, 10, 4  # small synthetic slice

    df = pd.DataFrame(
        {
            "meat_id": meat_ids,
            "Fat_chem": [25.48, 62.78],
            "split": [0, 1],
        }
    )
    csv_path = tmp_path / "red_data_with_splits.csv"
    df.to_csv(csv_path, index=False)

    images_dir = tmp_path / "HSI_data_binned_spectra"
    for sub in ("images", "masks", "quant_biases", "quant_scales"):
        (images_dir / sub).mkdir(parents=True)

    rng = np.random.default_rng(2)
    expected_cubes = {}
    for meat_id in meat_ids:
        for s in range(1, n_slices_per_belly + 1):
            stem = f"{meat_id}-{s}"
            q = rng.integers(0, 60000, size=(c, h, w)).astype(np.uint16)
            mask = rng.integers(0, 2, size=(1, h, w)).astype(np.uint8)
            scale = np.float32(1000.0 + meat_id)
            bias = np.float32(0.01 * s)
            np.save(images_dir / "images" / f"{stem}.npy", q)
            np.save(images_dir / "masks" / f"{stem}.npy", mask)
            np.save(images_dir / "quant_scales" / f"{stem}.npy", scale)
            np.save(images_dir / "quant_biases" / f"{stem}.npy", bias)
            expected_cubes[stem] = (q, mask, float(scale), float(bias))

    out_dir = tmp_path / "out_02_img"
    _run(
        [
            "image-slices",
            "--csv-path",
            str(csv_path),
            "--images-dir",
            str(images_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    dm = _dm_for(out_dir, batch_size=1)
    dm.setup()

    train_batches = list(dm.train_dataloader())  # belly 70, split 1
    test_batches = list(dm.test_dataloader())  # belly 10, split 0
    assert len(train_batches) == n_slices_per_belly
    assert len(test_batches) == n_slices_per_belly

    b0 = test_batches[0]
    assert tuple(b0["cube"].shape) == (1, h, w, c)
    assert tuple(b0["mask"].shape) == (1, h, w)
    assert "target" not in b0  # image frames carry no scalar regression target
    assert int(b0["group"][0]) == 10
    assert float(b0["quant_scale"][0]) == pytest.approx(1000.0 + 10, abs=1e-3)


def test_image_slices_all_bellies_in_the_test_split(tmp_path: Path) -> None:
    """The course ships only test bellies (10 and 70, split 0): no train selector is written."""
    meat_ids = [10, 70]
    h, w, c = 8, 10, 4
    df = pd.DataFrame({"meat_id": meat_ids, "Fat_chem": [25.48, 62.78], "split": [0, 0]})
    csv_path = tmp_path / "red_data_with_splits.csv"
    df.to_csv(csv_path, index=False)

    images_dir = tmp_path / "HSI_data_binned_spectra"
    for sub in ("images", "masks", "quant_biases", "quant_scales"):
        (images_dir / sub).mkdir(parents=True)
    rng = np.random.default_rng(3)
    for meat_id in meat_ids:
        for s in (1, 2):
            stem = f"{meat_id}-{s}"
            np.save(
                images_dir / "images" / f"{stem}.npy",
                rng.integers(0, 60000, (c, h, w)).astype(np.uint16),
            )
            np.save(
                images_dir / "masks" / f"{stem}.npy", rng.integers(0, 2, (1, h, w)).astype(np.uint8)
            )
            np.save(images_dir / "quant_scales" / f"{stem}.npy", np.float32(1000.0))
            np.save(images_dir / "quant_biases" / f"{stem}.npy", np.float32(0.0))

    out_dir = tmp_path / "out_02_img_test_only"
    _run(
        [
            "image-slices",
            "--csv-path",
            str(csv_path),
            "--images-dir",
            str(images_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    splits = load_splits(out_dir / "splits.json")
    assert splits.train == []
    assert [sel.ids for sel in splits.test] == [[0, 1, 2, 3]]
    assert [sel.ids for sel in splits.predict] == [[0, 1, 2, 3]]

    dm = _dm_for(out_dir, batch_size=1)
    dm.setup()
    assert len(list(dm.test_dataloader())) == 4


# ----- (4) kiwi cube -------------------------------------------------------


def test_kiwi_demo_round_trip(tmp_path: Path) -> None:
    h, w, c = 6, 7, 5
    rng = np.random.default_rng(3)
    img = rng.random((h, w, c)).astype(np.float32)
    wl = np.linspace(400.0, 1000.0, c).astype(np.float32).reshape(1, -1)

    mat_path = tmp_path / "1A.mat"
    scipy.io.savemat(mat_path, {"img": img, "Wavelength": wl})
    out_dir = tmp_path / "out_kiwi"

    spectrum_csv = tmp_path / "kiwi_spectrum.csv"
    np.savetxt(spectrum_csv, rng.random(c).astype(np.float32), delimiter=",")
    _run(
        [
            "kiwi-demo",
            "--mat-path",
            str(mat_path),
            "--out-dir",
            str(out_dir),
            "--spectrum-csv",
            str(spectrum_csv),
        ]
    )
    reference = np.load(out_dir / "kiwi_reference.npy")
    assert reference.shape == (1, 1, 1, c) and reference.dtype == np.float32
    assert np.allclose(reference.ravel(), np.loadtxt(spectrum_csv, delimiter=","), atol=1e-6)

    dm = _dm_for(out_dir)
    dm.setup(stage="predict")
    batches = list(dm.predict_dataloader())
    assert len(batches) == 1
    b0 = batches[0]
    assert tuple(b0["cube"].shape) == (1, h, w, c)
    assert np.allclose(b0["cube"][0].numpy(), img, atol=1e-6)
