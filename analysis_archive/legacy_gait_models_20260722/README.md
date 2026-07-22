# Legacy gait model artifacts

이 폴더는 2026-07-22 최종 모델 확정 과정에서 생성된 실험용/비채택 보행 모델 artifact 보관소다.

현재 서비스 기준 최종 모델은 다음 파일만 사용한다.

- `MOCA/models/gait_daily_clinical_3feat.joblib`
- `MOCA/models/gait_daily_clinical_3feat_metadata.json`

최종 기준:

```text
TUG >= 12s
OR FSST >= 15s
OR BERG < 52
OR DGI <= 19
OR base_velocity < 1.0 m/s
OR s3_velocity < 1.0 m/s
```

최종 입력 피처는 `v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr` 3개이며, threshold는 `0.50`으로 고정한다.
