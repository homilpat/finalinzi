# Final Gait Model Summary

## Final Model

- Model artifact: `MOCA/models/gait_axis_aligned_physionet_youden.joblib`
- Metadata: `MOCA/models/gait_axis_aligned_physionet_youden_metadata.json`
- Service processor: `MOCA/gait_axis_aligned_processor.py`
- Flask route: `/gait/upload-csv` in `MOCA/app.py`

## Pipeline

1. Read APK CSV from upload.
2. Parse `Timestamp_ns`.
3. Use anatomical columns `Acc_Vertical_g`, `Acc_ML_g`, `Acc_AP_g` when present.
4. If only raw `Acc_X`, `Acc_Y`, `Acc_Z` are present:
   - infer vertical axis from gravity,
   - use horizontal gait-band power to choose AP/ML candidates.
5. Resample to 100 Hz.
6. Apply 0.6-3 Hz gait-band filtering.
7. Extract candidate 10-second windows.
8. Select best 10-second window by vertical stride autocorrelation quality.
9. Extract final features.
10. Predict with the PhysioNet-normal-corrected logistic model.

## Final Features

| Feature | Axis | Definition |
|---|---|---|
| `v_acf_stride_peak` | Vertical | Autocorrelation peak height around stride lag, interpreted as vertical stride regularity. |
| `v_acf_stride_peak_width_sec` | Vertical | Half-height width of the vertical stride autocorrelation peak, interpreted as temporal spread/consistency of stride regularity. |
| `ap_acf_stride_peak_width_sec` | Anteroposterior | Half-height width of the AP stride autocorrelation peak, capturing forward-back stride timing spread. |
| `ap_spec_entropy` | Anteroposterior | Normalized spectral entropy in the 0.6-3 Hz gait band, capturing how concentrated or dispersed the AP walking rhythm is. |

## Data Used For Final Public Training

| Dataset | Label | N |
|---|---:|---:|
| PhysioNet LabWalks | Normal | 38 |
| PhysioNet LabWalks | Impaired | 35 |
| UCI HAR walking | Normal | 30 |
| GEOTEC smartphone walking/TUG segments | Normal | 10 |
| Chapman PD OFF raw walking | Impaired | 20 |
| FoG-STAR back walking | Impaired | 67 |

Total public training subjects/windows used at subject-level table: 200.
OUR_SAMPLE was held out for final sample prediction.

## Final Metrics

5-fold GroupKFold OOF validation on public data:

- AUC: 0.851
- Accuracy: 0.810
- Sensitivity: 0.934
- Specificity: 0.615
- F1: 0.857
- OOF threshold: 0.28

Deployed threshold:

- Threshold: 0.56
- Strategy: final public training set Youden threshold after model selection.
- Reason: the OOF threshold is reported for validation, while the deploy threshold is selected on the final public training set; held-out OUR_SAMPLE is not used for fitting.

Held-out APK SAMPLE predictions at deploy threshold:

| Sample | Target | Probability | Prediction | Correct |
|---|---:|---:|---:|---|
| `hazi_gait_anatomical_14cols_20260715_163129` | 0 | 0.378 | 0 | True |
| `hazi_gait_anatomical_14cols_20260716_081731_발다침_좌회전함` | 1 | 0.934 | 1 | True |
| `hazi_gait_calibrated_20s_20260715_155029` | 0 | 0.213 | 0 | True |
