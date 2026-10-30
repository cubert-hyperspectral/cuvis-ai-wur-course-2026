# Source

Vendored, unmodified, from the Hugging Face repository
[`Sm00thix/unet_chemical_map`](https://huggingface.co/Sm00thix/unet_chemical_map),
pinned at commit `9683a7ae6019c17e0e03f38a5e0d3c7dee9210e2` (2025-06-13), the `src/` tree.

The same tree ships in the course checkout at
`D:\code-repos\Chemometric-approaches-to-hyperspectral-processing\2026\helperfunctions\unet_end2end\`
and is bit-identical to the Hugging Face copy.

Accompanies Engstrom et al. (2025), "A pixel-is-not-a-sample chemical map": J. Chemometrics 39:e70041.

## Licences

- `LICENSE_CODE` (this directory, LGPL-3.0): the `src/` code (`callbacks/`, `dataloaders/`, `datasets/`,
  `losses/`, `models/`, `optimizers/`, `pipeline/`, `schedulers/`, `transforms/`) vendored here.
- `LICENSE_WEIGHTS` (this directory, CC BY-NC 4.0): the five pretrained U-Net checkpoints
  (`model_weights/1..5/best_weights.pt` on the Hugging Face repo). Not vendored in this repository;
  `UNetChemicalMapEnsemble` downloads them at construction time through `huggingface_hub.hf_hub_download`
  when `weights="hf"`, pinned to the same commit. The non-commercial restriction applies to the weights
  only, not to the architecture code.

## What was added on top of the vendored tree

Only `__init__.py` files, one per subdirectory, so the tree can be imported as a Python package
(`cuvis_ai_wur_course.upstream.unet_chemical_map.models.unet`, etc.). No source file was edited.
Every submodule here uses relative imports already (`from .unet import Permute`,
`from ..callbacks.early_stopping import EarlyStopping`), so no import shim or `sys.path` adjustment
was needed to make the tree importable from its new location.

The node wrapper (`cuvis_ai_wur_course/node/unet.py`) imports only from `models/`:
`compound_model.CompoundModel`, `conv3don2d.Conv3Don2D`, `ensemble_model.EnsembleModel`, `unet.UNet`.
The other subdirectories (`callbacks/`, `datasets/`, `losses/`, `pipeline/`, `schedulers/`,
`transforms/`, `dataloaders/`, `optimizers/`) are vendored for completeness and reference (the
training recipe, the study's own dataset and loss code) but are not imported by any cuvis-ai node in
this repository; `datasets/utils.py` additionally depends on the `spectral` package, which is not a
dependency of this plugin.
