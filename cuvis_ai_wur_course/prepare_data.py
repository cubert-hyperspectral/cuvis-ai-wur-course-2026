"""Converts the WUR course Drive folder into the npz layout ``CourseNpzDataModule`` reads.

Four subcommands, one per course dataset. Each writes, under its own output
directory: one ``.npz`` frame per sample (via
:func:`cuvis_ai_wur_course.data.course_npz.write_course_frame`), a
``universe.csv`` (``source, index, materialized_path``) and a ``splits.json``
(a serialized ``DataSplitConfig`` of ``file_indices`` selectors). The CLI and
the data module are proven consistent by the round-trip tests in
``tests/test_prepare_data.py``: prepare, then read back through
``CourseNpzDataModule``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import numpy as np
import pandas as pd
from cuvis_ai_core.data.splits_io import save_splits
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind
from sklearn.model_selection import KFold, train_test_split

from cuvis_ai_wur_course.data.course_npz import write_course_frame

#: The 124-channel window the course keeps from the 300-channel spectrum,
#: 747.92 to 1015.78 nm (an exact np.linspace, see the notebook extracts).
_SPECTRAL_WAVELENGTHS = np.linspace(747.92, 1015.78, 124).astype(np.float32)


def _write_universe_csv(path: Path, source: str, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "index", "materialized_path"])
        for i in range(n):
            writer.writerow([source, i, f"frames/{i:05d}.npz"])


def _file_indices(source: str, ids: list[int]) -> Selector:
    return Selector(kind=SelectorKind.FILE_INDICES, source=source, ids=ids)


# --------------------------------------------------------------------------
# (1) notebook 01 pork features: spatial_spectra_pig.mat
# --------------------------------------------------------------------------


def _build_pork_features(mat_path: Path, out_dir: Path, source: str) -> None:
    """One 1x1x812 frame per belly: channels [spatial 512 | spectral 300], target = ref.

    Frame order: ``train_test_split(test_size=0.2, random_state=42)`` train
    part first, then the test part (matching the notebook's one shared
    split). ``fold`` is a ``KFold(shuffle=False)`` block index over the
    train part only (consecutive blocks of the already-shuffled train
    order); test frames carry no ``fold`` key, and are assigned to the
    ``test`` split instead. The fold count is 5, capped at the train part's
    own size when fewer than 5 train bellies are available (``KFold`` raises
    otherwise on a pathologically small input); the course dataset has 182
    bellies, so the cap never engages in practice.
    """
    import scipy.io

    mat = scipy.io.loadmat(str(mat_path))
    spatial = np.asarray(mat["spatial"], dtype=np.float32)  # [N, 512]
    spectral = np.asarray(mat["spectral"], dtype=np.float32)  # [N, 300]
    ref = np.asarray(mat["ref"], dtype=np.float32).ravel()  # [N]
    n = spatial.shape[0]

    all_idx = np.arange(n)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    order = np.concatenate([train_idx, test_idx])  # train first, then test

    n_splits = min(5, len(train_idx))
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_of_train_pos = np.full(len(train_idx), -1, dtype=np.int32)
    for fold_id, (_, val_pos) in enumerate(kf.split(train_idx)):
        fold_of_train_pos[val_pos] = fold_id

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_pos, src_idx in enumerate(order):
        cube = np.concatenate([spatial[src_idx], spectral[src_idx]]).reshape(1, 1, -1)
        is_train = frame_pos < len(train_idx)
        fold = int(fold_of_train_pos[frame_pos]) if is_train else None
        write_course_frame(
            frames_dir / f"{frame_pos:05d}.npz",
            cube=cube,
            wavelengths=np.arange(cube.shape[-1], dtype=np.float32),
            target=np.array([ref[src_idx]], dtype=np.float32),
            fold=fold,
        )

    _write_universe_csv(out_dir / "universe.csv", source, n)

    train_ids = list(range(len(train_idx)))
    test_ids = list(range(len(train_idx), n))
    splits = DataSplitConfig(
        train=[_file_indices(source, train_ids)],
        test=[_file_indices(source, test_ids)],
    )
    save_splits(splits, out_dir / "splits.json")


@click.command("pork-features")
@click.option(
    "--mat-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to spatial_spectra_pig.mat (keys: spatial [N,512], spectral [N,300], ref [N,1]).",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for frames/, universe.csv and splits.json.",
)
@click.option("--source", default="pork_features", show_default=True, help="Universe source id.")
def pork_features(mat_path: Path, out_dir: Path, source: str) -> None:
    """Convert notebook 01's spatial_spectra_pig.mat into course npz frames."""
    _build_pork_features(mat_path, out_dir, source)
    click.echo(f"wrote pork feature frames to {out_dir}")


# --------------------------------------------------------------------------
# (2)/(3) mean spectra tables (notebook 02 and notebook 03), shared builder
# --------------------------------------------------------------------------


def _build_mean_spectra(
    csv_path: Path,
    spectra_dir: Path,
    out_dir: Path,
    source: str,
    *,
    target_column: str,
    num_channels: int = 124,
) -> pd.DataFrame:
    """Writes one 1x1xC frame per table row: the trailing ``num_channels`` channels
    of ``{spectra_dir}/{meat_id}.npy``, ``target`` = ``target_column``, ``fold`` = the
    table's own ``split`` column. Returns the table in the order written (frame
    index == row position) for the caller to build dataset-specific split selectors.
    """
    table = pd.read_csv(csv_path)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for pos, row in table.reset_index(drop=True).iterrows():
        spectrum_path = spectra_dir / f"{int(row['meat_id'])}.npy"
        spectrum = np.load(spectrum_path).astype(np.float32)[-num_channels:]
        cube = spectrum.reshape(1, 1, num_channels)
        write_course_frame(
            frames_dir / f"{pos:05d}.npz",
            cube=cube,
            wavelengths=_SPECTRAL_WAVELENGTHS[-num_channels:],
            target=np.array([float(row[target_column])], dtype=np.float32),
            fold=int(row["split"]),
            group=int(row["meat_id"]),
        )

    _write_universe_csv(out_dir / "universe.csv", source, len(table))
    return table.reset_index(drop=True)


@click.command("mean-spectra-03")
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="red_data_with_splits.csv (one row per belly: meat_id, Fat_chem, split, ...).",
)
@click.option(
    "--spectra-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="meat_id_mean_spectra/absorbance/ directory ({meat_id}.npy, 300 raw channels).",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--source", default="mean_spectra_03", show_default=True)
@click.option("--target-column", default="Fat_chem", show_default=True)
def mean_spectra_03(
    csv_path: Path, spectra_dir: Path, out_dir: Path, source: str, target_column: str
) -> None:
    """Convert notebook 03's mean-spectra table (course split 0-5) into course npz frames.

    Writes two split files side by side. ``splits.json`` has train = course
    splits 1 to 5, val = split 1, test = split 0: this is the file the
    statistical fits need, since the trainer feeds only the train loader and
    ``PLSRegression`` holds fold 1 out of that loader itself, so split 1
    stays inside train there. ``splits_gradient.json`` has train = splits 2
    to 5, val = split 1, test = split 0, matching how the course trains its
    MLP and 1D ResNet: val is a true hold-out from train, not a fold already
    present in it. Both files follow the table's own ``split`` column, which
    is also written into each frame as ``fold`` so a PLS hold-out-fold
    selector can read it without a second file.
    """
    table = _build_mean_spectra(
        csv_path, spectra_dir, out_dir, source, target_column=target_column, num_channels=124
    )
    split_col = table["split"].to_numpy()
    train_ids = [i for i, s in enumerate(split_col) if s in (1, 2, 3, 4, 5)]
    gradient_train_ids = [i for i, s in enumerate(split_col) if s in (2, 3, 4, 5)]
    val = [_file_indices(source, [i for i, s in enumerate(split_col) if s == 1])]
    test = [_file_indices(source, [i for i, s in enumerate(split_col) if s == 0])]
    splits = DataSplitConfig(train=[_file_indices(source, train_ids)], val=val, test=test)
    save_splits(splits, out_dir / "splits.json")

    gradient_splits = DataSplitConfig(
        train=[_file_indices(source, gradient_train_ids)], val=val, test=test
    )
    save_splits(gradient_splits, out_dir / "splits_gradient.json")

    click.echo(f"wrote {len(table)} mean-spectra frames to {out_dir}")


@click.command("mean-spectra-02")
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="red_data_with_splits.csv.",
)
@click.option(
    "--spectra-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="meat_id_mean_spectra/absorbance/ directory.",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--source", default="mean_spectra_02", show_default=True)
@click.option("--target-column", default="Fat_chem", show_default=True)
def mean_spectra_02(
    csv_path: Path, spectra_dir: Path, out_dir: Path, source: str, target_column: str
) -> None:
    """Convert notebook 02's mean-spectra table (used for the PLS baseline) into npz frames.

    One frame per belly, binned pairwise from 124 to 62 bands (the notebook's
    own ``load_spectrum(...).reshape(62, 2).mean(axis=1)``), ``fold`` = the
    table's ``split`` column (0 = test, 1 to 5 = the 5-fold CV set). Splits:
    train = CV splits 1 to 5 (fold selects the held-out member at fit time),
    test = split 0.
    """
    table = pd.read_csv(csv_path)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    binned_wavelengths = _SPECTRAL_WAVELENGTHS.reshape(62, 2).mean(axis=1).astype(np.float32)

    for pos, row in table.reset_index(drop=True).iterrows():
        spectrum_path = spectra_dir / f"{int(row['meat_id'])}.npy"
        spectrum = np.load(spectrum_path).astype(np.float32)[-124:]
        binned = spectrum.reshape(62, 2).mean(axis=1)
        cube = binned.reshape(1, 1, 62)
        write_course_frame(
            frames_dir / f"{pos:05d}.npz",
            cube=cube,
            wavelengths=binned_wavelengths,
            target=np.array([float(row[target_column])], dtype=np.float32),
            fold=int(row["split"]),
            group=int(row["meat_id"]),
        )

    _write_universe_csv(out_dir / "universe.csv", source, len(table))
    split_col = table["split"].to_numpy()
    train_ids = [i for i, s in enumerate(split_col) if s in (1, 2, 3, 4, 5)]
    test_ids = [i for i, s in enumerate(split_col) if s == 0]
    splits = DataSplitConfig(
        train=[_file_indices(source, train_ids)],
        test=[_file_indices(source, test_ids)],
    )
    save_splits(splits, out_dir / "splits.json")
    click.echo(f"wrote {len(table)} mean-spectra frames to {out_dir}")


# --------------------------------------------------------------------------
# (3) notebook 02 image slices
# --------------------------------------------------------------------------


@click.command("image-slices")
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="red_data_with_splits.csv.",
)
@click.option(
    "--images-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="HSI_data_binned_spectra/ (images/, masks/, quant_biases/, quant_scales/ subfolders).",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--source", default="image_slices_02", show_default=True)
def image_slices(csv_path: Path, images_dir: Path, out_dir: Path, source: str) -> None:
    """Convert notebook 02's quantized image slices into course npz frames.

    Keeps the course's uint16-quantized values as the stored cube and carries
    ``quant_scale`` / ``quant_bias`` so dequantization (``R = q/scale + bias``)
    happens in the pipeline, not here: the base dataset reader casts ``cube``
    to float32 on load (``CourseNpzDataModule`` / ``_CourseNpzDataset``), so
    the stored array is written as float32 holding the raw quantized integer
    values, not a real uint16 dtype (documented deviation; the exact integer
    values survive the cast, float32 has 24 bits of mantissa versus 16-bit
    source range). ``fold`` = the table's ``split`` column, ``group`` = belly
    id (``meat_id``), ``mask`` at its own (possibly smaller) resolution. The
    course folder ships test bellies only (10 and 70, split 0), so a stage
    without a slice is left out of ``splits.json``; ``predict`` lists every
    slice.
    """
    meta = pd.read_csv(csv_path)
    slice_files = sorted((images_dir / "images").glob("*.npy"))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    stems = [f.stem for f in slice_files]
    meat_ids = [int(s.split("-")[0]) for s in stems]
    meta_by_id = meta.set_index("meat_id")

    fold_by_pos: list[int] = []
    for pos, (stem, meat_id) in enumerate(zip(stems, meat_ids, strict=True)):
        q = np.load(images_dir / "images" / f"{stem}.npy")  # [C, H, W] uint16
        mask = np.load(images_dir / "masks" / f"{stem}.npy")  # [1, H, W]
        bias = float(np.load(images_dir / "quant_biases" / f"{stem}.npy"))
        scale = float(np.load(images_dir / "quant_scales" / f"{stem}.npy"))

        cube_hwc = np.moveaxis(q.astype(np.float32), 0, -1)  # [H, W, C]
        mask_hw = mask[0].astype(np.int32)
        split_value = int(meta_by_id.loc[meat_id, "split"])
        fold_by_pos.append(split_value)

        write_course_frame(
            frames_dir / f"{pos:05d}.npz",
            cube=cube_hwc,
            wavelengths=_SPECTRAL_WAVELENGTHS.reshape(62, 2).mean(axis=1).astype(np.float32),
            mask=mask_hw,
            fold=split_value,
            group=meat_id,
            quant_scale=scale,
            quant_bias=bias,
        )

    _write_universe_csv(out_dir / "universe.csv", source, len(stems))
    train_ids = [i for i, s in enumerate(fold_by_pos) if s in (1, 2, 3, 4, 5)]
    test_ids = [i for i, s in enumerate(fold_by_pos) if s == 0]
    stages = {"predict": [_file_indices(source, list(range(len(stems))))]}
    if train_ids:
        stages["train"] = [_file_indices(source, train_ids)]
    if test_ids:
        stages["test"] = [_file_indices(source, test_ids)]
    save_splits(DataSplitConfig(**stages), out_dir / "splits.json")
    click.echo(f"wrote {len(stems)} image-slice frames to {out_dir}")


# --------------------------------------------------------------------------
# (4) kiwi cube + RGB demo images
# --------------------------------------------------------------------------


@click.command("kiwi-demo")
@click.option(
    "--mat-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="1A.mat (keys: img [H,W,C], Wavelength [C]).",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--source", default="kiwi_cube", show_default=True)
@click.option(
    "--spectrum-csv",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="kiwi_spectrum.csv, the saved reference spectrum [C] the spectral-match filter uses.",
)
def kiwi_demo(mat_path: Path, out_dir: Path, source: str, spectrum_csv: Path | None) -> None:
    """Convert the kiwi demo cube (1A.mat) into a single course npz frame.

    One frame, index 0, no target/fold/group: the kiwi cube is a
    demonstration input for the SAM3 + spectral-match pipeline, not a
    chemometrics regression sample. With ``--spectrum-csv`` the course's saved
    kiwi reference spectrum is written next to it as ``kiwi_reference.npy``,
    shaped ``[1, 1, 1, C]`` so the builtin ``NpyReader`` hands it to
    ``SpectralAngleMapper`` as a one-pixel reference cube.
    """
    import scipy.io

    mat = scipy.io.loadmat(str(mat_path))
    cube = np.asarray(mat["img"], dtype=np.float32)  # [H, W, C]
    wavelengths = np.asarray(mat["Wavelength"], dtype=np.float32).ravel()

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    write_course_frame(frames_dir / "00000.npz", cube=cube, wavelengths=wavelengths)

    _write_universe_csv(out_dir / "universe.csv", source, 1)
    splits = DataSplitConfig(predict=[_file_indices(source, [0])])
    save_splits(splits, out_dir / "splits.json")
    if spectrum_csv is not None:
        reference = np.loadtxt(spectrum_csv, delimiter=",", dtype=np.float32).ravel()
        if reference.shape[0] != cube.shape[-1]:
            raise click.BadParameter(
                f"reference spectrum has {reference.shape[0]} values, "
                f"the cube {cube.shape[-1]} bands"
            )
        np.save(out_dir / "kiwi_reference.npy", reference.reshape(1, 1, 1, -1))
    click.echo(f"wrote the kiwi demo frame to {out_dir}")


@click.group()
def main() -> None:
    """Convert WUR course Drive folder contents into cuvis-ai course npz frames."""


main.add_command(pork_features)
main.add_command(mean_spectra_03)
main.add_command(mean_spectra_02)
main.add_command(image_slices)
main.add_command(kiwi_demo)


if __name__ == "__main__":
    main()
