# MOCA Flask Demo Runtime

This folder is the Render/Flask deployment root.

Render uses the repository-level `render.yaml` with:

- `rootDir: MOCA`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

Runtime gait files:

- `gait_axis_aligned_core.py`: shared final gait extractor
- `gait_axis_aligned_processor.py`: Flask-facing predictor
- `models/gait_axis_aligned_physionet_youden.joblib`: final gait model
- `models/gait_axis_aligned_physionet_youden_metadata.json`: final gait metadata
- `models/waist_sensor_range_loss_calibration.json`: fixed waist-reference raw sensor calibration summary

`gait_axis_aligned_processor.py` deploys `gait_axis_aligned_physionet_youden.joblib`.
When `models/waist_sensor_range_loss_calibration.json` is present, its fixed V/ML/AP
axis scale is applied inside `gait_axis_aligned_core.py` after anatomical axis alignment
and before 100 Hz resampling / best-10-second feature extraction.

Waist-reference raw sensor calibration, not model retraining:

- script: `analysis_scripts/calibrate_waist_sensor_range_loss.py`
- reference: PhysioNet LabWalks normal lower-back/L5 IMU raw best-10-second windows
- calibration samples: OUR_SAMPLE normal files only
- loss: robust distance between bandpassed V/ML/AP raw signal RMS/P95 and the PhysioNet waist-normal reference, with conservative scale regularization and sensor-range saturation penalty when metadata is available
- fixed scale estimate: V `1.077`, ML `0.966`, AP `0.952`
- loss improved from `0.136` to `0.122`
- warning: do not fit this calibration on impaired or new test measurements; without paired phone+IMU data this is sensor harmonization/QC, not full transfer calibration

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
