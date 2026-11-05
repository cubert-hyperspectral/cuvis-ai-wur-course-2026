# Changelog

## [Unreleased]

### Changed
- `RegressionMetrics`, `RegressionMSELoss` and `PredictionMean` take the model's prediction map `[B, H, W, T]`; metrics and loss average it over H and W per sample, the fusion averages maps elementwise. `SpatialMean`, the shape adapter that stood between every model and its metrics, is removed; the numbers do not move (a mean spectrum gives a 1 x 1 map). `pork_pls_fusion` drops from 15 to 12 nodes, `spectral_models` from 13 to 10, `pls_mean_spectra_fit` from 7 to 6, `mlp_train` and `resnet1d_train` from 5 to 4.
- `CourseFrame(keys=...)` declares only `cube` and the named batch keys on one instance, so a diagram shows the frame ports its pipeline uses; every course pipeline names its keys. Without `keys` the frame keeps all eight ports.
- `kiwi_sam3_spectral_match` is split into three pipelines that match the notebook's three steps: `kiwi_segment` (pseudo-RGB, SAM3), `kiwi_filters` (the size and spectral-angle filters side by side on SAM3's mask) and `kiwi_crop_features` (one kiwi cropped, rendered and embedded). SAM3 runs once; the later stages read their inputs from the batch.
- Dependency floors `cuvis-ai>=0.22.1` and `cuvis-ai-core>=0.20.1`.
- The three notebooks are re-authored on these pipelines and re-baked: every pipeline cell draws its graph with the node names through `utils.show_pipeline` (the Standardize migration had dropped those calls, leaving whole module trees as cell output), notebook 01 teaches the kiwi lesson as the three stages with the hand-off built in the cell, and the prediction plots read the model outputs. All 37 reproduction rows keep their values; the 1D ResNet own trainrun, whose result moves from run to run, lands inside the 15 percent band in this bake (38.10 against the course's 34.30). `tests/test_notebooks.py` guards the notebook sources against the pipeline YAMLs and checks the baked tables.
- `pork_pls_fusion` centres and scales with one `Standardize(axis="channel")` node instead of `MeanCenter` followed by `UnitVarianceScaling`; `pls_mean_spectra_fit` and `pls_pixel_map` use `Standardize(axis="spectrum", ddof=0)` in place of `SNVCorrection`; `spectral_models` uses `Standardize(axis="global", ddof=0, eps=1e-6, exclude_folds=[1])` in place of the plugin's own `GlobalScalarStandardizer`, now removed with its tests. The SNV replacement moves from the old node's unbiased per-pixel standard deviation to `ddof=0`'s population one; the PLS rows are unaffected since the regression coefficient absorbs the constant per-pixel rescale.

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
- `BandSweep` renders one labelled grayscale frame per band, and the pipeline `kiwi_video` writes the kiwi cube as the course's pseudo-RGB, true colour and colour infrared stills and as a sweep through its 224 bands (four videos under `outputs/`; `tools/build_pipelines.py` builds the YAML).

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
- The pipeline diagrams in the notebooks (`utils.show_pipeline`) and the PNGs under `pipelines/` show each node's name under its class title, so the three `PLSRegression` cards of the pork fusion graph and the metric cards read apart.

### Fixed
- `LabelKeep` follows the mask device and `LabelCropPad` the cube device, so the kiwi pipeline runs on the GPU with the SAM3 node's cpu mask.
- `ResNet18Features` returns float32 whatever autocast state the caller runs under.
- `prepare-course-data image-slices` leaves a stage without a slice out of `splits.json` and writes a `predict` stage with every slice; the course folder ships test bellies only.
