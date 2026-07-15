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

The web app currently uses a hybrid gait-model rule:

- Primary model: `MOCA/models/gait_nested_youden.joblib`
- Primary metadata: `MOCA/models/gait_nested_youden_metadata.json`
- Fallback model: `MOCA/models/gait_three_feature_youden.joblib`
- Fallback metadata: `MOCA/models/gait_three_feature_youden_metadata.json`

Primary 4-feature model summary:

- Target: `DGI <= 19 OR TUG >= 12`
- Data: LabWalks 10-second walking windows
- Model: Logistic Regression, L2, `class_weight=balanced`, `C=0.5`
- Threshold strategy: `nested_inner_oof_youden`
- Threshold: `0.48978492988237443`
- Subjects: 67
- Excluded subjects: `CO024`, `FL020`

Features used:

- `v_amp_pool_median`
- `ml_amp_pool_iqr`
- `base_v_stride_regularity`
- `roll_amp_pool_iqr`

Primary nested CV metrics:

- AUC: `0.830`
- Sensitivity: `0.800`
- Specificity: `0.738`
- F1: `0.714`

Fallback 3-feature model:

- Used only when `base_v_stride_regularity` is missing.
- Required features: `v_amp_pool_median`, `ml_amp_pool_iqr`, `roll_amp_pool_iqr`
- Threshold strategy: `pooled_5fold_x100_oof_youden_three_feature_candidate`
- Threshold: `0.4605355091838269`

- AUC: `0.829`
- Accuracy: `0.730`
- Sensitivity: `0.789`
- Specificity: `0.695`
- F1: `0.686`

Missing-value rule:

- If all 4 features are present, use the primary 4-feature nested CV Youden model.
- If only `base_v_stride_regularity` is missing, use the validated 3-feature fallback model.
- If any of the other 3 required features are missing, ask the user to measure again.

## Current Demo Limitation

The UI currently shows a 20-second measurement flow, but real sensor ingestion is not connected yet. For now, model inference is confirmed through manual feature input.

CSV sensor ingestion is now supported for calibrated gait files:

- Endpoint: `POST /gait/upload-csv`
- Upload field: `file`
- Expected columns: `Timestamp_ns`, `Acc_X`, `Acc_Y`, `Gyro_Clean_X`
- Optional columns such as `Acc_Y`, raw gyro, and calibration metadata can remain in the file.
- Timestamped smartphone samples are interpolated onto a uniform `100 Hz` grid before filtering, matching the training IMU sampling rate.
- Acceleration values are converted from `m/s^2` to `g` scale before feature extraction.
- Gyroscope values are converted from `rad/s` to `deg/s` before `roll_amp_pool_iqr` extraction.
- For the current waist-mounted portrait phone protocol, `Acc_Y` is treated as vertical, `Acc_X` as medio-lateral, and `Gyro_Clean_X` as roll.
- The server extracts multiple overlapping 10-second windows from the 20-second recording, takes the median feature values, and sends them through the same gait model rule.

The intended service flow is:

```text
20-second IMU collection
→ extract multiple 10-second walking windows and aggregate median features
→ Butterworth preprocessing
→ feature extraction
→ 4-feature Youden gait model inference, or 3-feature fallback when stride regularity is missing
```

## Care Type Logic

The final result page classifies users by cognitive and physical binary results:

```text
cognitive 0 / physical 0 -> A type, 유지형
cognitive 0 / physical 1 -> B type, 신체관리형
cognitive 1 / physical 0 -> C type, 통합관리형
cognitive 1 / physical 1 -> D type, 인지관리형
```
