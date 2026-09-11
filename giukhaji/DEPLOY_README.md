# Finalinzi Flask Demo Runtime

This folder is the Render/Flask deployment root.

Render uses the repository-level `render.yaml` with:

- `rootDir: giukhaji`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

Runtime gait files:

- `modeling/gait_axis_aligned_core.py`: shared final gait extractor
- `modeling/gait_axis_aligned_processor.py`: Flask-facing predictor
- `models/gait_daily_clinical_3feat.joblib`: final acc-only daily gait model
- `models/gait_daily_clinical_3feat_metadata.json`: final gait metadata

`app.py` calls `predict_daily_gait_csv()` in `modeling/gait_axis_aligned_processor.py`, which deploys
`gait_daily_clinical_3feat.joblib`. Pipeline: anatomical V/ML/AP axis alignment → 100 Hz resampling →
signal-level amplitude correction → 20-second segment / 10-second subwindow (2-second step) median·IQR
feature extraction.

Before feature extraction, `predict_daily_gait_csv()` applies the single fixed signal-amplitude
factor `alpha = 1.9705093832241642` stored in the model artifact. The same scalar is multiplied
across the V/ML/AP acceleration time series. It is the ratio of the PhysioNet normal vertical
gait-band RMS median (`0.193863`) to the smartphone normal-walk median (`0.098382`). This is
cross-domain amplitude harmonization, not Min-Max normalization and not a separate scale per axis.

Final gait label:

```text
TUG >= 12s
OR FSST >= 15s
OR BERG < 52
OR DGI <= 19
OR base_velocity < 1.0 m/s
OR s3_velocity < 1.0 m/s
```

Final input features:

- `v_jerk_rms_median` (movement impact, V Jerk RMS)
- `v_jerk_rms_iqr` (variability of movement impact)
- `v_harmonic_ratio_iqr` (variability of left-right gait symmetry, V ACF Symmetry)

Validation: subject-level 5-fold × 100 repeats, threshold chosen inside each training fold by 3-fold OOF
("sensitivity ≥ 0.80, then maximum specificity"): AUC `0.873 ± 0.007`, sensitivity `0.835`, specificity `0.731`.
The deployed service threshold is fixed at `0.50`.

Historical axis-wise harmonization experiment, not part of the deployed daily-model path:

- archived script: `analysis_archive/scripts/calibrate_waist_sensor_range_loss.py`
- reference: PhysioNet LabWalks normal lower-back/L5 IMU raw best-10-second windows
- calibration samples: OUR_SAMPLE normal files only
- loss: robust distance between bandpassed V/ML/AP raw signal RMS/P95 and the PhysioNet waist-normal reference, with conservative scale regularization and sensor-range saturation penalty when metadata is available
- fixed scale estimate: V `1.077`, ML `0.966`, AP `0.952`
- loss improved from `0.136` to `0.122`
- runtime status: retained as an analysis artifact; `predict_daily_gait_csv()` does not load or apply these axis-wise factors
- warning: without paired phone+IMU data this is sensor harmonization/QC, not full transfer calibration

Waist/back-only candidate, not deployed:

- training domains: PhysioNet LabWalks plus FoG-STAR back walking; OUR_SAMPLE held out
- features: `ap_acf_stride_peak_width_sec`, `ap_spec_entropy`
- threshold: Youden, `0.495`
- OOF: AUC `0.894`, sensitivity `0.657`, specificity `1.000`
- held-out OUR_SAMPLE: normal, impaired, normal all correctly classified
- reason not deployed: specificity `1.000` is suspicious because normal controls are only PhysioNet while FoG-STAR contributes only impaired rows, so label-domain confounding is likely.

Sensor range metadata, when present in uploaded CSV comments, is parsed by `gait_axis_aligned_core.py`.
Expected optional header keys:

- `# Accel_Maximum_Range_m_s2: ...`
- `# Accel_Resolution_m_s2: ...`
- `# Gyro_Maximum_Range_rad_s: ...`
- `# Gyro_Resolution_rad_s: ...`

Android `SensorEvent.values` are already physical units, so range metadata is used for unit validation,
saturation checks, clipping only when values exceed the declared physical range, and normalized-range
fallback detection. It is not blindly multiplied into every sample.

Analysis files, external datasets, local samples, and old training artifacts are outside this folder or ignored from upload.
