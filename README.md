# Final Project Demo

## Current Applied App

The deployable app is in `MOCA/`.

```bash
cd MOCA
pip install -r requirements.txt
python app.py
```

Render uses the root `render.yaml`, with `rootDir: MOCA`.

## Current Applied Gait Model

The web app currently uses the axis-aligned PhysioNet-normal-corrected gait model.
The older NestedCV and 3-feature gait models remain in `MOCA/models/` only as legacy
fallback artifacts if the final axis-aligned artifact is removed.

- Deployed model: `MOCA/models/gait_axis_aligned_physionet_youden.joblib`
- Deployed metadata: `MOCA/models/gait_axis_aligned_physionet_youden_metadata.json`
- Fixed waist sensor calibration: `MOCA/models/waist_sensor_range_loss_calibration.json`
- Flask predictor: `MOCA/gait_axis_aligned_processor.py`
- Shared extractor: `MOCA/gait_axis_aligned_core.py`

Final 4-feature model summary:

- Reference/correction mode: `physionet_normal`
- Threshold mode: Youden
- Deploy threshold: `0.56`
- Training rows: `200`
- Held-out service samples: local `OUR_SAMPLE` CSV files are not used for model fitting or domain correction.

Training data used:

- Normal: PhysioNet LabWalks `38`, UCI_HAR `30`, GEOTEC_SP `10`
- Impaired: PhysioNet LabWalks `35`, Chapman PD OFF raw walking `20`, FoG-STAR back walking `67`

Features used:

- `v_acf_stride_peak`
- `v_acf_stride_peak_width_sec`
- `ap_acf_stride_peak_width_sec`
- `ap_spec_entropy`

5-fold GroupKFold OOF metrics:

- AUC: `0.851`
- Accuracy: `0.810`
- Sensitivity: `0.934`
- Specificity: `0.615`
- F1: `0.857`
- OOF threshold: `0.28`

Train-set metrics at the deployed threshold `0.56`:

- AUC: `0.870`
- Accuracy: `0.750`
- Sensitivity: `0.648`
- Specificity: `0.910`
- F1: `0.760`

## Current Sensor CSV Flow

CSV sensor ingestion is now supported for calibrated gait files:

- Endpoint: `POST /gait/upload-csv`
- Upload field: `file`
- Preferred columns from the current 7s wait -> 3s gravity calibration -> 20s walking app:
  `Timestamp_ns`, `Acc_Vertical_g`, `Acc_ML_g`, `Acc_AP_g`, `Gyro_Roll_deg_s`
- Fallback raw phone columns:
  `Timestamp_ns`, `Acc_X`, `Acc_Y`, `Acc_Z`
- Optional columns such as raw acceleration, raw gyro, cleaned gyro, and calibration metadata can remain in the file.
- Timestamped smartphone samples are interpolated onto a uniform `100 Hz` grid before filtering.
- If anatomical columns are present, `Acc_Vertical_g`, `Acc_ML_g`, and `Acc_AP_g` are used as V / ML / AP.
- If raw phone columns are present, the gravity-dominant axis is used as vertical and the remaining axes are assigned by horizontal motion power.
- Optional sensor metadata such as `Accel_Maximum_Range_m_s2` and `Gyro_Maximum_Range_rad_s` is used for range validation and saturation checks, not blindly multiplied into the samples.
- The fixed waist-reference V/ML/AP scale calibration is applied after anatomical axis alignment and before feature extraction.
- The server evaluates overlapping 10-second windows inside the 20-second recording and uses the best-quality 10-second window.

The intended service flow is:

```text
7-second preparation
-> 3-second gravity calibration
-> 20-second IMU collection
-> unit/range validation
-> V / ML / AP anatomical axis alignment
-> fixed waist-reference raw scale calibration
-> timestamp-based 100 Hz interpolation
-> 0.6-3.0 Hz Butterworth preprocessing
-> best-quality 10-second window selection
-> ACF / spectral entropy feature extraction
-> 4-feature axis-aligned Youden gait model inference
```

Feature extraction rule:

- `v_acf_stride_peak`: vertical-axis stride autocorrelation peak, measuring stride-level repetition.
- `v_acf_stride_peak_width_sec`: half-height width of the vertical stride autocorrelation peak.
- `ap_acf_stride_peak_width_sec`: half-height width of the anteroposterior stride autocorrelation peak.
- `ap_spec_entropy`: spectral entropy of AP-axis motion in the gait band.

## Care Type Logic

The final result page classifies users by cognitive and physical binary results:

```text
cognitive 0 / physical 0 -> A type, 유지형
cognitive 0 / physical 1 -> B type, 신체관리형
cognitive 1 / physical 0 -> C type, 통합관리형
cognitive 1 / physical 1 -> D type, 인지관리형
```
