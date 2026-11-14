# No-s3 피처 조합 전수 비교 결과

## 평가 조건

- 피험자: 71명, 정상 37명 / 보행저하 34명
- 라벨: TUG, FSST, BERG, DGI, 일반 보행속도 OR 기준
- `s3_velocity` 단독 기준 제외
- 후보 피처: 10개
- 조합 수:
  - 2개 조합: 45개
  - 3개 조합: 120개
  - 4개 조합: 210개
  - 총 2~4개 조합: 375개
- 모델: Logistic Regression, `C=1`, L2
- 전처리: 각 training fold에서만 RobustScaler 학습
- 평가: subject-level Stratified 5-fold CV × 100회
- 상관 기준: 최대 절대 Spearman 상관계수 ≤ 0.85
- 다중공선성 기준: 최대 VIF ≤ 5

## 과거 0.7527의 의미

과거 문서의 AUC 0.7527은 다음 두 열만 사용한 개발 단계 LR 점수였다.

- `v_jerk_rms_median`
- `v_jerk_rms_iqr`

이는 전체 10개 피처 모델이 아니라, 하나의 원피처인 V jerk RMS를
median과 IQR로 요약한 2개 입력 피처 모델이다.

과거 0.7527은 이전 라벨·집계 조건의 메모이며 현재 no-s3 조건에서는
재현되지 않았다. 현재 동일 2피처 모델의 100회 평균 AUC는 **0.8125**다.

## 전체 원점수 상위 결과

| 전체 순위 | 피처 수 | 조합 | AUC | 표준편차 | 상관/VIF |
|---:|---:|---|---:|---:|---|
| 1 | 3 | AP ACF symmetry IQR + AP entropy IQR + Jerk median | 0.8220 | 0.0151 | 상관 탈락 |
| 2 | 2 | AP ACF symmetry IQR + Jerk median | **0.8202** | 0.0130 | 통과 |
| 3 | 1 | Jerk median | 0.8171 | 0.0065 | 통과 |
| 4 | 4 | AP ACF symmetry IQR + AP entropy IQR + Jerk median + Jerk IQR | 0.8155 | 0.0122 | 상관 탈락 |
| 5 | 3 | V ACF symmetry median + AP ACF symmetry IQR + Jerk median | 0.8127 | 0.0235 | 통과 |
| 6 | 2 | Jerk median + Jerk IQR | 0.8125 | 0.0085 | 통과 |
| 8 | 3 | V ACF symmetry median + Jerk median + Jerk IQR | 0.8117 | 0.0209 | 통과 |
| 11 | 4 | V ACF symmetry median + V stride frequency median + Jerk median + Jerk IQR | 0.8089 | 0.0241 | 통과 |
| 12 | 3 | AP entropy IQR + Jerk median + Jerk IQR | 0.8087 | 0.0119 | 통과 |
| 18 | 2 | V stride frequency IQR + Jerk median | 0.8073 | 0.0109 | 통과 |

원점수 1위 조합은 AP ACF symmetry IQR과 AP entropy IQR의 Spearman
상관이 약 0.888로 기준 0.85를 초과하여 탈락했다.

## 상관·VIF 통과 상위 조합

| 통과 순위 | 피처 수 | 조합 | AUC | 최대 \|Spearman\| | 최대 VIF |
|---:|---:|---|---:|---:|---:|
| 1 | 2 | AP ACF symmetry IQR + Jerk median | **0.8202** | 0.4170 | 1.24 |
| 2 | 1 | Jerk median | 0.8171 | 0.0000 | 1.00 |
| 3 | 3 | V ACF symmetry median + AP ACF symmetry IQR + Jerk median | 0.8127 | 0.4170 | 1.94 |
| 4 | 2 | Jerk median + Jerk IQR | 0.8125 | 0.6756 | 1.37 |
| 5 | 3 | V ACF symmetry median + Jerk median + Jerk IQR | 0.8117 | 0.6756 | 1.44 |
| 6 | 4 | V ACF symmetry median + V stride frequency median + Jerk median + Jerk IQR | 0.8089 | 0.6756 | 1.73 |
| 7 | 3 | AP entropy IQR + Jerk median + Jerk IQR | 0.8087 | 0.6756 | 3.22 |
| 8 | 4 | V ACF symmetry median + AP entropy median + Jerk median + Jerk IQR | 0.8080 | 0.6756 | 1.86 |
| 9 | 3 | AP ACF symmetry IQR + AP entropy median + Jerk median | 0.8074 | 0.6719 | 1.93 |
| 10 | 2 | V stride frequency IQR + Jerk median | 0.8073 | 0.4058 | 1.23 |

## 현재 서비스 조합

현재 서비스 입력:

- `v_jerk_rms_median`
- `v_jerk_rms_iqr`
- `v_acf_step_stride_symmetry_iqr`

| 항목 | 결과 |
|---|---:|
| 전체 1~4개 조합 순위 | 43위 |
| 피처 수 | 3개 |
| AUC 평균 | 0.8005 |
| AUC 표준편차 | 0.0116 |
| AUC 2.5~97.5 백분위 | 0.7720~0.8200 |

## 전체 10개 피처

| 입력 | AUC 평균 | 표준편차 | AUC 2.5~97.5 백분위 |
|---|---:|---:|---:|
| 전체 10개 피처 | 0.7620 | 0.0228 | 0.7201~0.7981 |

전체 피처를 모두 사용하면 소수 피처 조합보다 성능이 낮아졌다.

## 해석상 주의

이 결과는 각 고정 조합의 반복 CV 성능을 비교한 **개발 단계 전수
스크린**이다. 375개 결과를 확인한 뒤 최고 조합을 선택하면 선택 편향이
발생하므로, AUC 0.8202를 신규 최종 모델 성능으로 바로 보고하면 안 된다.

신규 2피처 후보를 채택하려면 다음 중 하나가 필요하다.

1. 피처 조합 선택 전체를 inner fold 안에서 수행하는 완전 nested CV
2. 현재 피처 선택에 사용하지 않은 독립 외부 검증 데이터

## 전체 결과 파일

- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/all_2_to_4_feature_combinations.csv`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/size_2_combinations.csv`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/size_3_combinations.csv`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/size_4_combinations.csv`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/repeat_auc_all_combinations.csv`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/full_10_feature_baseline.csv`
