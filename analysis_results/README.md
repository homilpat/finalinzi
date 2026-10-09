# 보행 모델 검증 결과 요약

README와 포트폴리오에 적은 보행 모델 수치의 출처입니다. 원본 실행 산출물(`analysis_outputs/`)은 용량 때문에 커밋하지 않고, 수치 확인에 필요한 요약 파일만 옮겼습니다. 모든 실험은 71명(정상 31 / 저하 40)의 피험자 단위 1인 1행 테이블과 같은 3개 피처(`v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr`)를 사용합니다. 스크립트의 `v_acf_step_stride_symmetry_iqr`는 `v_harmonic_ratio_iqr`의 이전 이름입니다.

재현 순서: 스크립트들은 저장소 루트 기준 상대경로를 쓰며, 입력 테이블 `analysis_outputs/final_training_conditions_nested_model_comparison_100rep/`는 `analysis_scripts/compare_final_training_conditions_nested_100rep.py`를 먼저 실행해 만듭니다. `validate_optuna_lr_no_s3_nested.py`가 기본으로 읽는 `final_training_conditions_no_s3_nested_model_comparison_100rep/`는 이 저장소의 스크립트로 생성되지 않습니다.

| 결과 폴더 | 생성 스크립트 | 내용 |
|---|---|---|
| `with_s3_lr_class_weight_default_youden_100rep/` | `analysis_scripts/compare_lr_class_weight_default_youden_100rep.py` | 기본 로지스틱 회귀, 5-fold × 100회. 고정 임계값 0.5와 학습 fold Youden 임계값 비교 |
| `with_s3_fixed_lr_none_vs_balanced_nested_100rep/` | `analysis_scripts/compare_fixed_lr_none_vs_balanced_nested_100rep.py` | class_weight 없음/balanced 비교. 임계값은 각 학습 fold 안 3-fold OOF에서 민감도 0.80 이상 중 특이도 최대 |
| `with_s3_lr_optuna_paired_nested_100rep/` | `analysis_scripts/validate_optuna_lr_no_s3_nested.py`의 입력(`final_training_conditions_nested_model_comparison_100rep`)과 출력 경로를 with_s3로 바꾸고 `--trials 15`로 실행 | 기본 설정 vs Optuna(C·penalty·class_weight, 목표 inner-CV AUC). 임계값 규칙은 위와 동일 |
| `with_s3_lr_optuna_maxiter_paired_nested_100rep/` | `analysis_scripts/validate_optuna_lr_with_s3_maxiter_nested.py` | 위 Optuna 비교를 max_iter 조정 후 재실행 |

## README에 쓰는 수치

| 조건 | AUC | 민감도 | 특이도 | 출처 |
|---|---|---|---|---|
| 고정 임계값 0.50 (배포 조건) | 0.873 ± 0.007 | 0.835 | 0.731 | `with_s3_lr_class_weight_default_youden_100rep`, class_weight 없음 · `fixed_0.5` 행 |
| fold 안 임계값 결정 (민감도 0.80 이상 중 특이도 최대) | 0.873 ± 0.008 | 0.811 | 0.746 | `with_s3_lr_optuna_paired_nested_100rep`, `Fixed LR` 행 |
| Optuna 튜닝 (같은 임계값 규칙) | 0.833 ± 0.034 | 0.816 | 0.702 | `with_s3_lr_optuna_paired_nested_100rep`, `Optuna LR` 행. 100회 중 90회에서 AUC 하락 |

배포 모델 메타데이터(`giukhaji/models/gait_daily_clinical_3feat_metadata.json`)의 `fixed_threshold_100rep_oof`(AUC 0.866 ± 0.009)는 `retrain_acconly_clean.py`의 이전 평가 구현(StratifiedGroupKFold)으로 계산한 값이며, 위 표의 수치와 직접 비교하지 않습니다.
