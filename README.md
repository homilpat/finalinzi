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

The web app currently loads:

- Model file: `MOCA/models/gait_nested_youden.joblib`
- Metadata: `MOCA/models/gait_nested_youden_metadata.json`
- Source validation script: `final__2026/03_code/03_validation/RUN_nested_domain4_oof_cv_final2026.py`

Applied model summary:

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

Nested CV metrics:

- AUC: `0.830`
- Sensitivity: `0.800`
- Specificity: `0.738`
- F1: `0.714`

## Current Demo Limitation

The UI currently shows a 20-second measurement flow, but real sensor ingestion is not connected yet. For now, model inference is confirmed through manual feature input.

The intended service flow is:

```text
20-second IMU collection
→ select a stable 10-second walking window
→ Butterworth preprocessing
→ feature extraction
→ Nested CV Youden gait model inference
```

## Care Type Logic

The final result page classifies users by cognitive and physical binary results:

```text
cognitive 0 / physical 0 -> A type, 유지형
cognitive 0 / physical 1 -> B type, 신체관리형
cognitive 1 / physical 0 -> C type, 통합관리형
cognitive 1 / physical 1 -> D type, 인지관리형
```

