# Final Service Manifest

This folder documents which files are used by the Flask service and which files are analysis-only.
It does not move runtime files out of `MOCA/`, because Flask imports them from that location.

## Runtime Files Used By Flask

- `MOCA/app.py`
  - `/gait/upload-csv` routes uploaded APK CSV files to the final axis-aligned PhysioNet-normal-corrected model when the final model artifact exists.
  - Falls back to the older gait pipeline only if the final artifact is missing.
- `MOCA/gait_axis_aligned_processor.py`
  - Thin Flask-facing predictor for APK CSV files.
  - Calls the shared final extractor and applies the deployed model threshold.
- `MOCA/gait_axis_aligned_core.py`
  - Shared final gait extractor used by both Flask and final analysis/training scripts.
  - Uses the analysis-script baseline: APK/public CSV to accelerometer array, estimated Hz, 100 Hz interpolation, gravity/anatomical axis handling, 0.6-3 Hz bandpass, and best-quality 10-second window.
- `MOCA/models/gait_axis_aligned_physionet_youden.joblib`
  - Final deployed model artifact.
- `MOCA/models/gait_axis_aligned_physionet_youden_metadata.json`
  - Final model metadata, features, thresholds, CV metrics, fold metrics, and dataset counts.

## Current Final Gait Feature Set

- `v_acf_stride_peak`
- `v_acf_stride_peak_width_sec`
- `ap_acf_stride_peak_width_sec`
- `ap_spec_entropy`

The final correction reference is `physionet_normal`: domains with available normal controls are median-shift corrected to PhysioNet LabWalks normal medians. The same domain correction is applied to both normal and impaired rows in that domain. OUR_SAMPLE is held out and is not used for model fitting or correction.

## Analysis-Only Files

- `analysis_scripts/build_axis_aligned_gait_dataset_and_model.py`
  - Builds the axis-aligned best-10-second subject table from public/raw datasets and local samples using `MOCA/gait_axis_aligned_core.py`.
- `analysis_scripts/model_axis_aligned_domain_corrected_gait.py`
  - Screens combinations after normal-reference domain correction.
- `analysis_scripts/train_final_axis_aligned_domain_corrected_gait_model.py`
  - Trains and exports the final PhysioNet-normal-corrected model artifact.
- `analysis_outputs/axis_aligned_gait_model/`
  - Intermediate axis-aligned subject table and model screens.
- `analysis_outputs/axis_aligned_domain_corrected_gait_model/`
  - Domain-corrected combination screen and sample predictions.
- `analysis_outputs/final_axis_aligned_physionet_normal_gait_model/`
  - Final OOF predictions, fold metrics, held-out SAMPLE predictions, and correction deltas.
- `analysis_archive/moca_nonruntime/`
  - Old MOCA document-generation scripts, local logs, temporary render outputs, and non-runtime documentation artifacts moved out of `MOCA/`.
- `analysis_archive/moca_training_and_cli/`
  - Training-only scripts, old CLI utilities, design screenshots, Roboflow/CDT training data, local Android/app artifacts, and non-runtime large files moved out of `MOCA/`.

## Deployment Layout

- `render.yaml` points to `rootDir: MOCA`, so Render deploys the Flask folder only.
- Root `.gitignore` excludes generated analysis outputs, external datasets, local gait samples, Android build outputs, and the analysis archive from upload.
- `MOCA/` is now the temporary-demo Flask folder. It keeps app code, templates/static/assets, final gait model artifacts, fallback gait model artifacts, scoring modules, and lightweight runtime metadata.
- `MOCA/data/` is ignored and recreated automatically by `database.py` when the demo starts.

## Exercise Sensor Prototype

- `MOCA/exercise_sensor_processor.py`
  - Prototype rule-based CSV processor for exercise counting.
  - Not part of the final gait model.
  - Needs exercise-specific CSV samples for threshold calibration before clinical/demo claims.
