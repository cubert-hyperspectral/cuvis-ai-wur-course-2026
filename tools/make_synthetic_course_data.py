"""Write a small synthetic stand-in for the course Drive folder and convert it.

Until the course data arrives, this script fabricates every input file the
``prepare-course-data`` commands read (the pork feature ``.mat``, the two mean-spectra
tables with their ``.npy`` spectra, the quantised image slices with masks, the kiwi cube
and its reference spectrum) with the course's shapes but random content, and runs the
converter on them. The result under ``data/course`` lets every pipeline, trainrun and
notebook execute end to end; the numbers it produces mean nothing.

Run from the repository root:

    uv run python tools/make_synthetic_course_data.py [--root data/synthetic_course] [--force]

The converted frames land in ``data/course/<set>`` and overwrite an earlier synthetic set;
``prepare-course-data`` on the real folder writes the same locations. When ``data/course``
holds converted sets but no ``--root`` folder exists, the sets came from the course data and
the script stops rather than overwrite them; ``--force`` overrides that check.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from click.testing import CliRunner

from cuvis_ai_wur_course.prepare_data import main as prepare

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "course"

N_BELLIES = 182
N_SPECTRA_BANDS = 300  # stored spectra; the converter keeps the trailing 124
IMAGE_CHANNELS = 62
IMAGE_HW = (2000, 912)  # course-sized slice: the U-Net needs hundreds of pixels per axis
MASK_HW = (804, 260)  # UNetChemicalMapEnsemble(weights=None) output shape for IMAGE_HW
KIWI_HW = (64, 80)
KIWI_BANDS = 125


def _run(args: list[str]) -> None:
    result = CliRunner().invoke(prepare, args, catch_exceptions=False)
    if result.exit_code != 0:
        raise SystemExit(result.output)
    print(result.output.strip())


def pork_features(root: Path, rng: np.random.Generator) -> None:
    mat = root / "spatial_spectra_pig.mat"
    spatial = rng.random((N_BELLIES, 512)).astype(np.float32)
    spectral = rng.random((N_BELLIES, 300)).astype(np.float32)
    ref = (rng.random((N_BELLIES, 1)) * 50.0 + 10.0).astype(np.float32)
    scipy.io.savemat(mat, {"spatial": spatial, "spectral": spectral, "ref": ref})
    _run(["pork-features", "--mat-path", str(mat), "--out-dir", str(OUT / "pork_features")])


def _mean_spectra_table(root: Path, rng: np.random.Generator) -> tuple[Path, Path]:
    meat_ids = list(range(100, 100 + N_BELLIES))
    splits = [i % 6 for i in range(N_BELLIES)]
    fat = rng.random(N_BELLIES) * 40.0 + 10.0
    csv_path = root / "red_data_with_splits.csv"
    pd.DataFrame({"meat_id": meat_ids, "Fat_chem": fat, "split": splits}).to_csv(
        csv_path, index=False
    )
    spectra_dir = root / "meat_id_mean_spectra" / "absorbance"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    base = np.linspace(0.2, 1.0, N_SPECTRA_BANDS, dtype=np.float32)
    for meat_id, fat_value in zip(meat_ids, fat, strict=True):
        spectrum = base * (1.0 + fat_value / 100.0) + 0.01 * rng.standard_normal(N_SPECTRA_BANDS)
        np.save(spectra_dir / f"{meat_id}.npy", spectrum.astype(np.float32))
    return csv_path, spectra_dir


def mean_spectra(root: Path, rng: np.random.Generator) -> Path:
    csv_path, spectra_dir = _mean_spectra_table(root, rng)
    for command, out_name in (
        ("mean-spectra-03", "mean_spectra_03"),
        ("mean-spectra-02", "mean_spectra_02"),
    ):
        _run(
            [
                command,
                "--csv-path",
                str(csv_path),
                "--spectra-dir",
                str(spectra_dir),
                "--out-dir",
                str(OUT / out_name),
            ]
        )
    return csv_path


def image_slices(root: Path, csv_path: Path, rng: np.random.Generator) -> None:
    """Write placeholder slices, scales, biases and masks.

    The mask ships at the U-Net output grid, not at the slice resolution: that
    is the course's own convention, and the 02 pipelines wire this mask
    straight into the map statistics with no resizing node in between. At the
    course's own slice size (``[62, 2360, 1272]``), the output grid works out
    to ``(input - 368) / 2`` per axis; the valid-convolution U-Net does not
    reduce every input size by that same affine rule, so ``MASK_HW`` is this
    stand-in's measured output shape for ``IMAGE_HW``, not a formula applied
    to it.
    """
    images_dir = root / "HSI_data_binned_spectra"
    for sub in ("images", "masks", "quant_biases", "quant_scales"):
        (images_dir / sub).mkdir(parents=True, exist_ok=True)
    h, w = IMAGE_HW
    mh, mw = MASK_HW
    for meat_id in (100, 101):  # present in the table: splits 4 and 5 (train)
        for s in (1, 2):
            stem = f"{meat_id}-{s}"
            q = rng.integers(0, 60000, size=(IMAGE_CHANNELS, h, w)).astype(np.uint16)
            mask = np.zeros((1, mh, mw), dtype=np.uint8)
            mask[0, mh // 8 : -mh // 8, mw // 8 : -mw // 8] = 1
            np.save(images_dir / "images" / f"{stem}.npy", q)
            np.save(images_dir / "masks" / f"{stem}.npy", mask)
            np.save(images_dir / "quant_scales" / f"{stem}.npy", np.float32(32552.7))
            np.save(images_dir / "quant_biases" / f"{stem}.npy", np.float32(0.0))
    _run(
        [
            "image-slices",
            "--csv-path",
            str(csv_path),
            "--images-dir",
            str(images_dir),
            "--out-dir",
            str(OUT / "image_slices"),
        ]
    )


def kiwi(root: Path, rng: np.random.Generator) -> None:
    h, w = KIWI_HW
    cube = rng.random((h, w, KIWI_BANDS)).astype(np.float32) * 0.3
    cube[16:40, 20:60, :] += 0.4  # one bright object
    wavelengths = np.linspace(400.0, 1000.0, KIWI_BANDS, dtype=np.float32).reshape(1, -1)
    mat = root / "1A.mat"
    scipy.io.savemat(mat, {"img": cube, "Wavelength": wavelengths})
    spectrum_csv = root / "kiwi_spectrum.csv"
    np.savetxt(spectrum_csv, cube[16:40, 20:60, :].mean(axis=(0, 1)), delimiter=",")
    _run(
        [
            "kiwi-demo",
            "--mat-path",
            str(mat),
            "--out-dir",
            str(OUT / "kiwi"),
            "--spectrum-csv",
            str(spectrum_csv),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=REPO / "data" / "synthetic_course")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite converted sets under data/course even when they came from the course data",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    converted = sorted(d.name for d in OUT.glob("*") if (d / "universe.csv").exists())
    if converted and not root.exists() and not args.force:
        raise SystemExit(
            f"{OUT} holds converted sets ({', '.join(converted)}) and {root} does not exist, so "
            "they came from the course data; pass --force to overwrite them with the stand-in"
        )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pork_features(root, rng)
    csv_path = mean_spectra(root, rng)
    image_slices(root, csv_path, rng)
    kiwi(root, rng)
    print(f"synthetic course inputs under {root}, converted frames under {OUT}")


if __name__ == "__main__":
    main()
