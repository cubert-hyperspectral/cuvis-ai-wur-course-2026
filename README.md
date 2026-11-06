# cuvis-ai-wur-course

The 2026 notebooks of the WUR course "AI for spectral and hyperspectral data processing"
(Liza Genke, Puneet Mishra; repository
[LizaGenke/Chemometric-approaches-to-hyperspectral-processing](https://github.com/LizaGenke/Chemometric-approaches-to-hyperspectral-processing),
folder `2026`, commit `0d2c33e`) rebuilt with [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai).
This repository is a fork of the course repository: the course material stays under `2025/` and
`2026/` as published, with the course's own README as `README_course.md`, and the rebuild sits
next to it. The processing steps live in tested nodes of this plugin and are wired as pipeline YAMLs with
diagrams. The fits and training runs are trainrun YAMLs. Three teaching notebooks build the same
graphs in the open and end in a table that compares our numbers with the course's.

| Course notebook | Here | What it covers |
|---|---|---|
| `using_pretrained_ai.ipynb` | `notebooks/01_pretrained_ai.ipynb` | SAM3 on a kiwi tray with the course's size and spectral-angle filters, the course's yolov8n and the plugin's YOLO26 on the two photos, ResNet-18 features, PLS fusion for pork belly fat |
| `end2end_unet_chemical_maps.ipynb` | `notebooks/02_unet_chemical_maps.ipynb` | The study's U-Net ensemble as chemical maps, next to the pixel-wise PLS map, with the map statistics of the course |
| `end2end_spectral_vs_spatial_models.ipynb` | `notebooks/03_spectral_vs_spatial.ipynb` | Standardisation, PCA, PLS, an MLP and a 1D ResNet on mean spectra, with their trainruns |

Three parts of the course are deferred: the 2D ResNet and ViT on the slice images, the U-Net demo
training run with the study's losses, and the stacking fusion of notebook 01.

## Sources and licences

- Course notebooks and data: Liza Genke and Puneet Mishra, Wageningen University and Research. The
  course data (mean spectra, example belly slices, the kiwi cube and reference spectrum, the course's
  trained weights) is not part of this repository; `prepare-course-data` reads it from a local copy
  of the course folder. The course model code in `cuvis_ai_wur_course/upstream/course_models.py`
  and the use of the course weights follow the terms agreed with the course authors (pending).
- Study: O.-C. G. Engstrom, M. Albano-Gaglio, E. S. Dreier, Y. Bouzembrak, M. Font-i-Furnols,
  P. Mishra, K. S. Pedersen (2025). Transforming Hyperspectral Images Into Chemical Maps: An
  End-to-End Deep Learning Approach. Journal of Chemometrics 39:e70041,
  [doi:10.1002/cem.70041](https://doi.org/10.1002/cem.70041).
- U-Net code and weights: [Sm00thix/unet_chemical_map](https://huggingface.co/Sm00thix/unet_chemical_map).
  The code (LGPL-3.0) is vendored unmodified under `cuvis_ai_wur_course/upstream/unet_chemical_map/`
  with its licence file; the weights (CC BY-NC 4.0) are downloaded from Hugging Face at run time and
  never redistributed here.
- Pork belly data set: Albano-Gaglio et al. (2025).
- The code written here (`cuvis_ai_wur_course/` outside `upstream/`, `tests/`, `tools/`, `notebooks/`,
  `pipelines/`, `trainruns/`, `plugins/`) is Apache-2.0, see `LICENSE`. The licence does not cover the
  course material under `2025/` and `2026/`, which stays with the course authors.

## Layout

```
cuvis_ai_wur_course/      the plugin: node/ (one module per family), data/ (the data module),
                          prepare_data.py (the CLI), upstream/ (vendored course and study code)
plugins/                  the manifest cuvis_ai_wur_course.yaml (21 nodes and the data module)
pipelines/                one folder per notebook: pipeline YAMLs with rendered PNGs
trainruns/                three statistical fits and two gradient runs
notebooks/                the three notebooks and utils.py (plotting and tables only)
tools/                    build_pipelines.py (regenerates the YAMLs), make_synthetic_course_data.py
tests/                    pytest, synthetic tensors, no downloads
data/, outputs/           local, ignored by git
```

## Setup

```powershell
uv sync --all-extras
uv run pytest -q -m "not slow"
```

The `plugins` extra pins the sam3 and ultralytics plugins the notebook 01 pipelines use; cuvis-ai,
cuvis-ai-core and cuvis-ai-dataloader come from PyPI. The pipeline and trainrun commands below take
two plugin catalogs: this repository's `plugins/` and the catalog packaged with cuvis-ai
(`.venv/Lib/site-packages/cuvis_ai/configs/plugins` on Windows).

## Data

`prepare-course-data` converts the course folder into npz frames, a `universe.csv` and a
`splits.json` per set, under `data/course/<set>`:

```powershell
uv run prepare-course-data pork-features  --mat-path <course>/spatial_spectra_pig.mat --out-dir data/course/pork_features
uv run prepare-course-data mean-spectra-03 --csv-path <course>/red_data_with_splits.csv --spectra-dir <course>/meat_id_mean_spectra/absorbance --out-dir data/course/mean_spectra_03
uv run prepare-course-data image-slices   --csv-path <course>/red_data_with_splits.csv --images-dir <course>/HSI_data_binned_spectra --out-dir data/course/image_slices
uv run prepare-course-data kiwi-demo      --mat-path <course>/1A.mat --spectrum-csv <course>/kiwi_spectrum.csv --out-dir data/course/kiwi
```

`mean-spectra-03` also writes `splits_gradient.json` next to `splits.json`, holding split 1 out of train for the two gradient trainruns.

The course's trained weights go to `data/course/models/mlp/best_weights.pt` and
`data/course/models/resnet1d/best_weights.pt`, the two photos of notebook 01 to
`data/course/images/1A_rgb.png` and `data/course/images/carrots.jpg`.

Without the course data, `uv run python tools/make_synthetic_course_data.py` writes a random stand-in
with the course's shapes under `data/synthetic_course` and converts it into the same `data/course`
layout. Everything runs on it; the numbers mean nothing, and the reproduction tables say so.

## Pipelines and trainruns

```powershell
uv run restore-pipeline --pipeline-path pipelines/03_spectral_vs_spatial/spectral_models.yaml --plugins-dir plugins --plugins-dir .venv/Lib/site-packages/cuvis_ai/configs/plugins --pipeline-vis-ext png
uv run restore-trainrun --trainrun-path trainruns/spectral_models_fit.yaml --plugins-dir plugins --plugins-dir .venv/Lib/site-packages/cuvis_ai/configs/plugins --mode train
uv run restore-trainrun --trainrun-path trainruns/mlp_train.yaml --plugins-dir plugins --plugins-dir .venv/Lib/site-packages/cuvis_ai/configs/plugins --mode train
```

`pipeline` and `splits_path` in a trainrun are resolved relative to the trainrun file, `universe_csv`
relative to the working directory, so run from the repository root. `tools/build_pipelines.py`
regenerates every pipeline YAML from explicit node constructors and `connect` calls and validates
the ports as it goes.

## Notebooks

```powershell
uv run jupyter lab notebooks
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_spectral_vs_spatial.ipynb
```

Each notebook names its source notebook and commit, builds its pipelines cell by cell, reads the
metrics off the metric nodes, and ends with the reproduction table. The baked outputs in this
repository come from the course data. Every numeric row passes its tolerance except the best
validation MSE of our own 1D ResNet trainrun, listed as an open deviation with its cause; the kiwi
mask counts are reported next to the course's SAM v1 counts, the yolov8n label counts match the
course's, and the plugin's YOLO26 labels are reported next to them.

## Nodes

| Module | Nodes |
|---|---|
| `node/source.py` | `CourseFrame` |
| `node/regression.py` | `PLSRegression` (fixed or hold-out selected components, pooled SSE) |
| `node/metrics.py`, `node/losses.py` | `RegressionMetrics`, `RegressionMSELoss` |
| `node/pca.py` | `DatasetPCA` |
| `node/pretreatment.py` | `SpectralBinning`, `SavitzkyGolayInterp` |
| `node/fusion.py` | `ChannelSlice`, `PredictionMean` |
| `node/render.py` | `BandSweep` (one grayscale frame per band, for a video sweep through the cube) |
| `node/kiwi.py`, `node/features.py` | `LabelBBoxFilter`, `LabelCropPad`, `LabelKeep`, `ResNet18Features` |
| `node/models.py` | `SpectrumRegressor` (the course's MLP and 1D ResNet) |
| `node/unet.py`, `node/maps.py` | `UNetChemicalMapEnsemble`, `Dequantize`, `CenterCrop`, `MaskToMapResolution`, `GroupWeightedMean`, `ChemicalMapStatistics` |

`PLSRegression`, `RegressionMetrics`, `RegressionMSELoss`, `DatasetPCA` and `SavitzkyGolayInterp`
are candidates for cuvis-ai itself once the course session is over.
