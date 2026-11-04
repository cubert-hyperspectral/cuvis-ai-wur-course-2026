"""Course data module: per-frame npz with optional chemometrics keys.

``CourseNpzDataModule`` (registry name ``wur_course_npz``) subclasses
``cuvis_ai_dataloader``'s ``MultiNpzDataModule`` and overrides
``build_dataset_from_refs`` with a frame reader that reads the standard
``cube`` / ``wavelengths`` / ``mask`` keys plus the optional per-frame keys
the WUR course data needs: a regression ``target``, a cross-validation
``fold``, a leakage ``group`` (e.g. a belly id), and the ``quant_scale`` /
``quant_bias`` pair used to dequantize a stored image slice. A frame whose
npz does not carry one of the optional keys yields no value for it; the
matching port on :class:`cuvis_ai_wur_course.node.source.CourseFrame` is
marked optional so the pipeline runs with it simply absent.

Image frames carry a mask at the mask's own resolution (not necessarily the
cube's), so this module reads them with batch size 1: the base class'
default collate cannot stack masks of different sizes across a batch. Mean
spectrum and feature frames (1x1 cubes) have no such restriction, but a
batch mixing image and non-image frames would still collate masks of
different shapes, so a universe is either all images or all 1x1 frames in
practice; callers using mixed universes keep ``batch_size=1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule
from torch.utils.data import Dataset


class _CourseNpzDataset(Dataset):
    """Reads one course npz frame, with the optional chemometrics keys.

    Holds the same ``rows`` shape as the base ``_MultiNpzDataset`` (a list of
    ``{frame_id, materialized_path, index}`` dicts built by
    ``build_dataset_from_refs``), but keeps ``wavelengths`` as float32 (the
    base class casts to int32 for node compatibility; the course's fractional
    nanometre axis, e.g. 747.92 to 1015.78 nm, would lose precision under
    that cast, so this reader keeps the exact float array instead).
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        if self._rows:
            with np.load(self._rows[0]["materialized_path"]) as z:
                wl = np.asarray(z["wavelengths"]).ravel()
            self.wavelengths_nm = wl.astype(np.float32, copy=False)
            self.num_channels = int(self.wavelengths_nm.shape[0])
        else:
            self.wavelengths_nm = np.array([], dtype=np.float32)
            self.num_channels = 0

    @property
    def rows(self) -> list[dict]:
        """Public read-only view of the per-frame rows."""
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        rec = self._rows[idx]
        with np.load(rec["materialized_path"]) as z:
            cube = np.asarray(z["cube"], dtype=np.float32)
            wavelengths = np.asarray(z["wavelengths"]).ravel().astype(np.float32, copy=False)
            mask = (
                np.asarray(z["mask"], dtype=np.int32)
                if "mask" in z.files
                else np.zeros((cube.shape[0], cube.shape[1]), dtype=np.int32)
            )
            sample: dict[str, np.ndarray | int] = {
                "cube": cube,
                "mask": mask,
                "wavelengths": wavelengths,
                "mesu_index": int(rec["index"]),
                "frame_id": int(rec["frame_id"]),
            }
            if "target" in z.files:
                sample["target"] = np.asarray(z["target"], dtype=np.float32).ravel()
            if "fold" in z.files:
                sample["fold"] = np.int32(np.asarray(z["fold"]).item())
            if "group" in z.files:
                sample["group"] = np.int32(np.asarray(z["group"]).item())
            if "quant_scale" in z.files:
                sample["quant_scale"] = np.float32(np.asarray(z["quant_scale"]).item())
            if "quant_bias" in z.files:
                sample["quant_bias"] = np.float32(np.asarray(z["quant_bias"]).item())
        return sample


class CourseNpzDataModule(MultiNpzDataModule):
    """Per-frame npz data module for the WUR course chemometrics datasets.

    Same universe/splits contract as ``MultiNpzDataModule`` (a
    ``universe_csv`` of ``source, index, materialized_path`` rows selected by
    a ``DataSplitConfig``), with a frame reader that additionally surfaces
    per-frame ``target``, ``fold``, ``group``, ``quant_scale`` and
    ``quant_bias`` when the npz carries them. Keeps the wavelength axis as
    float32 rather than the base class' int32 cast, since the course's
    124-channel window (747.92 to 1015.78 nm) is defined as an exact
    ``np.linspace`` and would be rounded to whole nanometres otherwise.
    """

    DATA_MODULE_NAME: ClassVar[str] = "wur_course_npz"

    def build_dataset_from_refs(self, refs: list[Any]) -> Dataset:
        """Build the frame dataset for a resolved subset of ``refs``.

        Mirrors ``MultiNpzDataModule.build_dataset_from_refs`` exactly (same
        identity lookup by ``(source, index)``) but returns
        ``_CourseNpzDataset`` instead of the base ``_MultiNpzDataset``, so the
        extra optional keys and the float32 wavelength axis are read.
        """
        assert self._universe is not None
        by_identity = {
            (rec["source"], int(rec["index"])): rec["materialized_path"] for rec in self._universe
        }
        rows = []
        for i, ref in enumerate(refs):
            key = (ref.source, int(ref.index if ref.index is not None else 0))
            path = by_identity.get(key)
            if path is None:
                raise ValueError(f"ref {key} has no matching npz in the universe")
            rows.append(
                {
                    "frame_id": i,
                    "materialized_path": path,
                    "index": int(ref.label_id if ref.label_id is not None else key[1]),
                }
            )
        return _CourseNpzDataset(rows)


def write_course_frame(
    path: str | Path,
    *,
    cube: np.ndarray,
    wavelengths: np.ndarray,
    mask: np.ndarray | None = None,
    target: np.ndarray | None = None,
    fold: int | None = None,
    group: int | None = None,
    quant_scale: float | None = None,
    quant_bias: float | None = None,
) -> None:
    """Write one course frame npz with the keys :class:`CourseNpzDataModule` reads.

    ``cube`` is ``[H, W, C]`` float32; ``wavelengths`` is ``[C]``. Every
    keyword past ``mask`` is optional and omitted from the file entirely when
    ``None``, matching the "a frame without a key yields no value" contract.
    Shared by :mod:`cuvis_ai_wur_course.prepare_data` and the data module's
    own tests, so the writer and the reader never drift apart.
    """
    payload: dict[str, np.ndarray] = {
        "cube": np.asarray(cube, dtype=np.float32),
        "wavelengths": np.asarray(wavelengths, dtype=np.float32).ravel(),
    }
    if mask is not None:
        payload["mask"] = np.asarray(mask, dtype=np.int32)
    if target is not None:
        payload["target"] = np.asarray(target, dtype=np.float32).ravel()
    if fold is not None:
        payload["fold"] = np.int32(fold)
    if group is not None:
        payload["group"] = np.int32(group)
    if quant_scale is not None:
        payload["quant_scale"] = np.float32(quant_scale)
    if quant_bias is not None:
        payload["quant_bias"] = np.float32(quant_bias)
    np.savez(path, **payload)


__all__ = ["CourseNpzDataModule", "write_course_frame"]
