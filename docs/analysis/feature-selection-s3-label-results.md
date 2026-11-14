# s3 포함 라벨 기준 피처 생성·선정 결과

## 1. 현재 라벨

다음 조건 중 하나라도 만족하면 보행저하군(`target=1`)으로 정의했다.

```text
TUG >= 12
OR FSST >= 15
OR BERG < 52
OR DGI <= 19
OR base_velocity < 1.0 m/s
OR s3_velocity < 1.0 m/s
```

| 구분 | no-s3 | s3 포함 |
|---|---:|---:|
| 정상군 | 37명 | 31명 |
| 보행저하군 | 34명 | 40명 |
| 전체 | 71명 | 71명 |

s3 기준을 추가하면서 `CO018`, `CO027`, `CO030`, `FL008`, `FL016`,
`FL033`의 6명이 정상군에서 보행저하군으로 이동했다.

## 2. 원시 신호에서 모델 입력까지

```text
가속도 원시 신호
→ 축 정렬 및 필터링
→ 보행으로 판정된 20초 segment 생성
→ segment 안에서 10초 window를 2초 간격으로 6개 생성
→ 각 10초 window에서 5개 원피처 계산
→ 6개 window 값의 median과 IQR 계산
→ 20초 segment당 10개 후보 피처 생성
→ 학습 데이터에서는 subject의 여러 segment 값을 다시 median으로 통합
→ subject당 1행으로 CV 수행
```

20초 안의 10초 window 시작점은 `0, 2, 4, 6, 8, 10초`이므로 총
6개다.

### 5개 원피처

| 원피처 | 의미 |
|---|---|
| V ACF step/stride symmetry | 수직축 step·stride 반복 관계 |
| AP ACF step/stride symmetry | 전후축 step·stride 반복 관계 |
| V stride frequency | 수직축에서 추정한 stride 빈도 |
| AP spectral entropy | 전후축 보행 신호의 주파수 복잡도 |
| V jerk RMS | 수직 가속도의 시간 변화량 |

각 원피처에 `median`과 `IQR(Q3-Q1)`을 적용해 총 10개 후보를 만든다.

예를 들어 한 20초 segment의 6개 window에서 수직 jerk RMS가
`J1, J2, ..., J6`으로 계산되면:

```text
v_jerk_rms_median = median(J1, J2, ..., J6)
v_jerk_rms_iqr    = Q3(J1, J2, ..., J6) - Q1(J1, J2, ..., J6)
```

라벨을 바꿔도 이 원피처와 median/IQR 값은 바뀌지 않는다. 바뀌는 것은
각 subject가 정상·저하 중 어느 그룹에 속하는지이며, 따라서
단일 피처 AUC, 조합 순위, 모델 계수와 임계값이 달라진다.

## 3. s3 포함 라벨의 단일 피처 판별력

평가는 subject당 1행, `StratifiedKFold(5) × 100회`, fold 내부
`RobustScaler`, 고정 LR(`C=1`, L2)로 수행했다.

| 순위 | 피처 | 평균 OOF AUC |
|---:|---|---:|
| 1 | `v_jerk_rms_median` | **0.8539** |
| 2 | `v_jerk_rms_iqr` | **0.8049** |
| 3 | `v_acf_step_stride_symmetry_iqr` | 0.6817 |
| 4 | `v_stride_freq_hz_median` | 0.6477 |
| 5 | `ap_spec_entropy_median` | 0.6107 |
| 6 | `v_stride_freq_hz_iqr` | 0.6037 |
| 7 | `ap_acf_step_stride_symmetry_iqr` | 0.5773 |
| 8 | `ap_spec_entropy_iqr` | 0.5718 |
| 9 | `ap_acf_step_stride_symmetry_median` | 0.5599 |
| 10 | `v_acf_step_stride_symmetry_median` | 0.5161 |

## 4. 1~4개 조합 전수 탐색

10개 후보에서 1~4개 조합 총 385개를 동일한 100회 분할로 비교했다.
이 중 2~4개 조합은 375개다.

| 구분 | 조합 | 평균 OOF AUC |
|---|---|---:|
| 전체 1위 | V ACF symmetry IQR + V stride frequency median + AP entropy IQR + V jerk IQR | **0.8785** |
| 3개 조합 1위 | V ACF symmetry IQR + AP entropy IQR + V jerk IQR | **0.8784** |
| 2개 조합 1위 | V ACF symmetry IQR + V jerk IQR | **0.8764** |
| 현재 서비스 3개 | V jerk median + V jerk IQR + V ACF symmetry IQR | **0.8734** |
| Jerk 2개 | V jerk median + V jerk IQR | 0.8714 |

현재 서비스 3개 조합의 반복 분포는 다음과 같다.

```text
AUC mean   = 0.8734
AUC SD     = 0.0078
95% 범위   = 0.8527–0.8843
Spearman 최대 절댓값 = 0.6756
최대 VIF   = 2.1742
```

## 5. 해석 시 구분

- `0.8734`: 피처를 현재 3개로 미리 고정한 개발용 반복 OOF 점수
- `0.8785`: 385개 결과를 확인한 뒤 고른 최고 조합의 개발용 점수
- `0.8794`: 별도 최종 재학습 스크립트에서 fold별 AUC를 평균한 값

`0.8734`와 `0.8794`는 라벨과 피처가 같지만 AUC 집계 방식이 다르다.
전자는 매 반복의 전체 OOF 예측을 합쳐 AUC를 계산하고, 후자는 각
fold AUC를 먼저 계산한 뒤 평균한다.

최고 조합 `0.8785`는 조합을 본 뒤 선택한 값이므로 최종 일반화
성능으로 보고하지 않는다. 피처 선택 과정 자체의 일반화 성능을
보고하려면 inner fold에서만 조합을 고르는 완전 nested CV를 별도로
수행해야 한다.

## 6. 저장 위치

- 전체 385개 결과:
  `analysis_outputs/s3_exhaustive_feature_combinations_100rep/all_1_to_4_feature_combinations.csv`
- 2~4개 조합 375개:
  `analysis_outputs/s3_exhaustive_feature_combinations_100rep/all_2_to_4_feature_combinations.csv`
- 단일 피처:
  `analysis_outputs/s3_exhaustive_feature_combinations_100rep/size_1_combinations.csv`
- 반복별 원자료:
  `analysis_outputs/s3_exhaustive_feature_combinations_100rep/repeat_auc_all_combinations.csv`
- 실행 조건:
  `analysis_outputs/s3_exhaustive_feature_combinations_100rep/metadata.json`

