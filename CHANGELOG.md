# Changelog

## [Unreleased]

### Added
- Repository scaffold: plugin package, manifest, tests, notebooks folder.
- The `wur_course_npz` data module, the `CourseFrame` source node and the `prepare-course-data` CLI.
- `PLSRegression` (fixed or hold-out selected components), `RegressionMetrics`, `RegressionMSELoss`, `GlobalScalarStandardizer` and `DatasetPCA`.
- `SpectralBinning`, `SavitzkyGolayInterp`, `ChannelSlice`, `PredictionMean` and `SpatialMean`.
- `LabelBBoxFilter`, `LabelCropPad` and `ResNet18Features`.
- `SpectrumRegressor` (the course's MLP and 1D ResNet) and `UNetChemicalMapEnsemble` with `Dequantize`, `CenterCrop`, `GroupWeightedMean` and `ChemicalMapStatistics`; the study code is vendored under `cuvis_ai_wur_course/upstream/`.
- The plugin manifest `plugins/cuvis_ai_wur_course.yaml` with palette metadata.
- `LabelKeep` (writes per-object decisions back onto a mask) and `MaskToMapResolution` (crop, resize, round and erode a slice mask to a chemical map's grid).
- The course pipelines under `pipelines/` with their diagrams: `kiwi_sam3_spectral_match`, `yolo_rgb`, `resnet18_features`, `pork_pls_fusion`, `unet_ensemble_maps`, `pls_mean_spectra_fit`, `pls_pixel_map`, `spectral_models`, `mlp_train`, `resnet1d_train`; `tools/build_pipelines.py` regenerates them.
- The trainruns under `trainruns/`: three statistical fits and the MLP and 1D ResNet gradient runs.
- The `plugins` extra pins the sam3 and ultralytics plugins the notebook 01 pipelines use.
- `tools/make_synthetic_course_data.py` fabricates a stand-in for the course Drive folder so every pipeline and trainrun runs before the data arrives.
- `prepare-course-data kiwi-demo --spectrum-csv` writes the kiwi reference spectrum for the spectral-match filter.
- The three teaching notebooks under `notebooks/` (`01_pretrained_ai`, `02_unet_chemical_maps`, `03_spectral_vs_spatial`) with `utils.py` for plotting and the reproduction tables, baked on the synthetic stand-in.
- `RegressionMetrics.compute()` returns the pooled RMSE, R2 and MAE as floats.

### Changed
- `CourseFrame` hands the wavelengths on as np.int32 `[C]`, like the builtin source node.
- `RegressionMetrics` exposes its metrics through the trainer's pooled-metric contract.
- `SpatialMean` takes no mask; the mask-weighted belly value is `GroupWeightedMean`'s job.
- `GlobalScalarStandardizer` and `DatasetPCA` take `exclude_folds` and an optional `fold` input, so the notebook 03 pipeline fits them on splits 2 to 5 like the course while the statistical trainer feeds splits 1 to 5.
- `prepare-course-data mean-spectra-03` also writes `splits_gradient.json` (splits 2 to 5 train, split 1 validates, split 0 tests); the MLP and 1D ResNet trainruns read it, so their validation bellies are no longer part of their training set.
- The notebook 02 pipelines feed the course's slice masks, which ship on the U-Net output grid, straight into the map statistics (`mask_scale` 2 for the full-resolution PLS map, as the course's `np.kron` does); `MaskToMapResolution` stays in the plugin for masks stored at the slice resolution. `CenterCrop` takes no mask any more.
- The synthetic stand-in writes its slice masks on the U-Net output grid, like the course data.
- `tools/make_synthetic_course_data.py` refuses to overwrite converted sets that came from the course data unless `--force` is passed.
- Notebook 01 runs the course's yolov8n and the plugin's YOLO26 on both course photos (the kiwi tray and the carrots) through the ultralytics nodes at `imgsz=640`; the yolov8n label counts match the course's and are judged in the reproduction table, the YOLO26 counts are reported next to them.
- The `plugins` extra pins cuvis-ai-sam3 v0.5.2 (autocast state restored after the model build, outputs on the input device) and cuvis-ai-ultralytics v0.1.5 (RGB kept before the model, `imgsz` letterbox); notebook 01 no longer builds the kiwi pipeline inside a disabled autocast.
- The three notebooks are baked on the course data; the reproduction tables compare against the course numbers.

### Fixed
- `LabelKeep` follows the mask device and `LabelCropPad` the cube device, so the kiwi pipeline runs on the GPU with the SAM3 node's cpu mask.
- `ResNet18Features` returns float32 whatever autocast state the caller runs under.
- `prepare-course-data image-slices` leaves a stage without a slice out of `splits.json` and writes a `predict` stage with every slice; the course folder ships test bellies only.
