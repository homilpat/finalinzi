# 보행저하 선별모델 피처 선정 발표자료

## 전체 흐름

> 결측률·계산 안정성 확인  
> → 단일 피처 판별력 확인  
> → 2~4개 조합 탐색  
> → 전체 10개와 소수 피처 모델 비교  
> → Spearman 상관분석  
> → VIF 다중공선성 확인  
> → 성능·복잡도·해석성을 종합하여 최종 3개 확정

---

## 슬라이드 1. 후보 피처 구성

### 화면 문구

**5개 원피처를 median과 IQR로 요약하여 10개 후보 생성**

| 보행 영역 | 원피처 | 20초 요약값 |
|---|---|---|
| 수직 주기 대칭성 | V ACF step/stride symmetry | median, IQR |
| 전후방 주기 대칭성 | AP ACF step/stride symmetry | median, IQR |
| 보행 리듬 | V stride frequency | median, IQR |
| 신호 복잡도 | AP spectral entropy | median, IQR |
| 움직임 변화·부드러움 | V jerk RMS | median, IQR |

### 발표 멘트

보행의 크기만 보는 것이 아니라 주기성, 대칭성, 리듬, 복잡도,
움직임의 급격한 변화라는 상호보완적인 영역을 포함하도록 5개
원피처를 사전 정의했습니다.

20초 측정 안에서 10초 window를 2초 간격으로 생성하고, 각 window에서
원피처를 계산했습니다. 이후 6개 window 값의 median과 IQR을 계산하여
20초 측정당 총 10개의 후보 피처를 구성했습니다.

- Median: 측정 중 대표적인 보행 수준
- IQR: 측정 중 window 간 변동성과 불안정성

---

## 슬라이드 2. 결측률 및 계산 안정성

### 화면 문구

**10개 후보 모두 71명에서 계산 가능**

| 검사 항목 | 결과 |
|---|---:|
| 분석 피험자 | 71명 |
| 후보 피처 | 10개 |
| 각 피처 유효값 | 71/71 |
| 각 피처 결측률 | 0% |

### 발표 멘트

피처 선택에 앞서 실제 서비스 조건에서 안정적으로 계산할 수 있는지를
확인했습니다. 10개 후보 모두 71명의 subject-level 데이터에서 유효값이
계산되었으며 결측률은 0%였습니다.

따라서 특정 피처의 계산 실패 때문에 피험자가 제외되거나, 결측치가
라벨과 연결되는 문제는 확인되지 않았습니다.

### 해석

이 단계는 피처의 판별 성능을 보는 단계가 아니라, 센서 신호에서 해당
피처를 반복적으로 산출할 수 있는지 확인하는 품질검사 단계입니다.

---

## 슬라이드 3. 단일 피처 판별력 확인

### 화면 문구

**Subject-level 단일 피처 OOF AUC**

| 순위 | 후보 피처 | 단독 OOF AUC |
|---:|---|---:|
| 1 | `v_jerk_rms_median` | **0.8171** |
| 2 | `v_jerk_rms_iqr` | **0.7292** |
| 3 | `v_stride_freq_hz_median` | 0.6034 |
| 4 | `v_acf_step_stride_symmetry_iqr` | 0.5723 |
| 5 | `ap_spec_entropy_iqr` | 0.5364 |
| 6 | `ap_spec_entropy_median` | 0.4835 |
| 7 | `ap_acf_step_stride_symmetry_median` | 0.4367 |
| 8 | `v_stride_freq_hz_iqr` | 0.4351 |
| 9 | `v_acf_step_stride_symmetry_median` | 0.4123 |
| 10 | `ap_acf_step_stride_symmetry_iqr` | 0.4109 |

### 발표 멘트

각 후보를 하나씩 사용한 subject-level 로지스틱 회귀의 OOF AUC를
확인했습니다. 수직 jerk의 대표 수준인 `v_jerk_rms_median`이 0.8171로
가장 높았고, jerk 변동성인 `v_jerk_rms_iqr`이 두 번째였습니다.

다만 단독 AUC가 낮다고 해서 바로 제거하지는 않았습니다. 단독 판별력은
낮더라도 다른 피처와 결합했을 때 상호보완적인 정보를 제공할 수 있기
때문입니다.

### 주의

이 수치는 현재 no-s3 라벨에서 각 고정 피처를 동일한 subject-level
5-fold CV로 100회 평가한 개발 스크린 점수입니다. 전체 결과를 보고
최고 피처를 사후 선택하면 낙관 편향이 생기므로 최종 일반화 성능으로
보고하지 않습니다.

---

## 슬라이드 4. 2~4개 피처 조합 비교

### 화면 문구

**10개 후보의 2~4개 조합 375개 전수 탐색**

| 구분 | 피처 조합 | AUC | 상관/VIF |
|---|---|---:|---|
| 원점수 1위 | AP ACF symmetry IQR + AP entropy IQR + Jerk median | 0.8220 | 상관 기준 탈락 |
| 통과 조합 1위 | AP ACF symmetry IQR + Jerk median | **0.8202** | 통과 |
| 4개 최고 | 위 원점수 1위 3개 + Jerk IQR | 0.8155 | 상관 기준 탈락 |
| Jerk 2개 | Jerk median + Jerk IQR | 0.8125 |
| 현재 서비스 3개 | V ACF symmetry IQR + Jerk median + Jerk IQR | 0.8005 |

### 발표 멘트

10개 후보에서 가능한 2개 조합 45개, 3개 조합 120개, 4개 조합
210개를 모두 비교했습니다.

현재 no-s3 조건에서 Jerk median과 Jerk IQR만 사용한 AUC는 0.8125입니다.
과거 문서의 0.7527은 이전 라벨·집계 조건의 개발 메모이며 현재 결과로
사용하지 않습니다.

가장 높은 원점수는 AP ACF symmetry IQR, AP entropy IQR, Jerk median의
3개 조합으로 AUC 0.8220이었습니다. 하지만 AP ACF symmetry IQR과
AP entropy IQR의 Spearman 상관이 약 0.888로 기준 0.85를 초과하여
탈락했습니다.

상관과 VIF를 모두 통과한 조합 중에서는 AP ACF symmetry IQR과
Jerk median의 2개 조합이 AUC 0.8202로 가장 높았습니다. 최대 절대
Spearman 상관은 0.417, 최대 VIF는 1.24였습니다.

### 핵심 해석

- Jerk median: 보행 변화량의 대표 수준
- Jerk IQR: 측정 중 변화량의 변동성
- ACF symmetry IQR: step/stride 대칭성의 시간적 변동

---

## 슬라이드 5. 전체 10개와 소수 피처 모델 비교

### 화면 문구

**피처가 많다고 성능이 높아지지는 않음**

| 모델 입력 | no-s3 100회 CV AUC |
|---|---:|
| 원점수 최고 3개 | 0.8220, 상관 기준 탈락 |
| 상관·VIF 통과 최고 2개 | **0.8202** |
| Jerk median 단독 | 0.8171 |
| Jerk median + Jerk IQR | 0.8125 |
| 현재 서비스 3개 | 0.8005 |
| 전체 10개 | 0.7620 |

### 발표 멘트

전체 10개 피처를 모두 사용한 모델의 AUC는 0.7620으로, 소수 피처
조합보다 낮았습니다. 상관과 VIF를 통과한 최고 2개 조합은
0.8202였습니다.

피험자가 71명인 상황에서 입력 피처를 불필요하게 늘리면 회귀계수의
불안정성과 과적합 위험이 증가합니다. 따라서 성능이 비슷하다면 피처가
적고 해석하기 쉬운 모델을 우선하는 원칙을 적용했습니다.

### 선택 원칙

> 성능 차이가 작거나 없으면 더 단순하고 설명 가능한 조합을 선택한다.

---

## 슬라이드 6. Spearman 상관분석

### 화면 문구

**최종 후보 3개의 순위 기반 중복성 확인**

|  | Jerk median | Jerk IQR | ACF symmetry IQR |
|---|---:|---:|---:|
| Jerk median | 1.0000 | 0.6756 | -0.6209 |
| Jerk IQR | 0.6756 | 1.0000 | -0.0421 |
| ACF symmetry IQR | -0.6209 | -0.0421 | 1.0000 |

### 발표 멘트

피처를 고른 뒤 Spearman 상관계수로 피처 간 순위 정보의 중복성을
확인했습니다.

Jerk median과 Jerk IQR의 상관은 0.6756으로 중간 정도였습니다. 두
피처가 관련되어 있지만, 하나는 대표 수준이고 다른 하나는 측정 중
변동성을 나타내므로 동일한 피처로 보지는 않았습니다.

ACF symmetry IQR은 Jerk IQR과의 상관이 -0.0421로 매우 낮아, jerk
변동성과 다른 정보를 제공했습니다.

### 정확한 표현

> 상관계수가 낮은 피처만 골랐다가 아니라, 성능과 보행 의미로 후보를
> 선택한 후 과도한 중복이 있는지를 상관분석으로 확인했다.

---

## 슬라이드 7. VIF 다중공선성 확인

### 화면 문구

**최종 후보 3개 모두 VIF 5 미만**

| 피처 | VIF |
|---|---:|
| `v_jerk_rms_median` | 2.17 |
| `v_jerk_rms_iqr` | 1.54 |
| `v_acf_step_stride_symmetry_iqr` | 1.59 |

### 발표 멘트

상관계수는 두 피처 사이의 관계만 확인하기 때문에, 한 피처가 나머지
피처의 선형결합으로 설명되는 정도를 VIF로 추가 확인했습니다.

전체 10개가 아니라 최종 후보 3개만으로 VIF를 다시 계산했습니다.
그 결과 VIF는 2.17, 1.54, 1.59로 모두 일반적인 다중공선성 주의
기준인 5보다 낮았습니다. 따라서 세 피처를 함께 사용해도 로지스틱
회귀계수가 심각하게 불안정해질 정도의 다중공선성은 없다고
판단했습니다.

---

## 슬라이드 8. 현재 서비스 3개 피처와 신규 후보

### 화면 문구

**현재 서비스 모델은 기존 3개 피처를 유지**

| 최종 피처 | 모델에서의 의미 |
|---|---|
| `v_jerk_rms_median` | 수직 움직임 변화량의 대표 수준 |
| `v_jerk_rms_iqr` | 수직 움직임 변화량의 측정 중 변동성 |
| `v_acf_step_stride_symmetry_iqr` | step/stride 대칭성의 시간적 변동 |

### 현재 서비스 3개 유지 근거

1. 71명 전원에서 계산 가능하여 결측률이 0%였다.
2. Jerk median과 Jerk IQR이 단일 분석에서 가장 높은 판별력을 보였다.
3. 현재 3개 조합은 no-s3 반복 평가에서 AUC 0.8005를 재현했다.
4. 전체 10개를 사용하면 AUC가 0.7620으로 낮아졌다.
5. 최종 3개의 VIF가 모두 5 미만이었다.
6. 각 피처가 대표 수준, 변동성, 주기 대칭성이라는 다른 보행 정보를
   제공하여 임상·서비스 설명이 가능했다.

### 발표 멘트

현재 서비스 모델은 계산 안정성, 피처 수, 상관관계, VIF, 보행
생체역학적 의미를 고려해 기존 3개 피처를 고정한 모델입니다.

다만 현재 no-s3 전체 조합 스크린에서는 상관과 VIF를 통과한
AP ACF symmetry IQR, Jerk median의 2개 조합이 AUC 0.8202로 더
높았습니다. 이 값은 375개 조합을 사후 비교한 개발 점수이므로,
별도의 완전 nested 피처 선택 또는 외부 검증 전에는 최종 모델을
교체하지 않습니다.

피처를 고정한 이후에는 subject 단위 반복 nested CV에서 모델과
하이퍼파라미터를 평가했습니다. 최종 no-s3 고정 LR은 100회 평가에서
AUC 0.8005, 민감도 0.8032, 특이도 0.6446을 보였습니다.

---

## 슬라이드 9. 검증상 주의점

### 화면 문구

**개발 단계 피처 탐색과 최종 성능 평가를 분리**

- 과거 0.7527/0.7550은 이전 조건의 개발 메모이므로 현재 결과에서 제외
- 현재 375개 조합 전수 스크린의 통과 조합 최고 AUC 0.8202도 최종
  성능은 아님
- 조합 전수 탐색 전체가 하나의 완전 nested 선택 절차였던 것은 아님
- 피처 확정 후 최종 모델 비교·Optuna 검증은 subject-level nested CV 수행
- 최종 성능은 no-s3 고정 LR의 outer test 결과로 보고

### 발표 멘트

과거 개발 단계의 0.7527과 0.7550은 현재 발표 수치로 사용하지 않습니다.
현재 no-s3 조건으로 전체 조합을 다시 계산했습니다. 원점수 최고
AUC는 0.8220이었지만 상관 기준에서 탈락했고, 상관·VIF 통과 조합
중 최고 AUC는 0.8202였습니다.

하지만 이 값 역시 375개 결과를 모두 확인한 뒤 고른 값이므로 최종
일반화 성능으로 제시하지 않습니다.

피처 확정 후에는 같은 피험자가 학습과 평가에 동시에 들어가지 않도록
subject 단위 outer 5-fold를 100회 반복했고, 전처리, 하이퍼파라미터,
확률 임계값은 inner fold에서만 결정했습니다.

따라서 발표에서는 피처 선정 결과와 최종 모델 성능을 서로 다른
단계의 결과로 구분하여 제시합니다.

---

## 한 장 요약

> 문헌과 보행 생체역학을 기반으로 5개 원피처를 사전 정의하고,
> 20초 측정의 대표성과 변동성을 반영하기 위해 median과 IQR로 총
> 10개 후보를 구성하였다. 모든 후보는 결측률 0%로 계산 가능했다.
> 단일 피처와 2~4개 조합의 subject-level 성능을 비교한 뒤, 전체
> 10개 모델과의 성능·복잡도 차이를 확인하였다. 현재 서비스 3개는
> AUC 0.8005를 보였으며, 전수 탐색에서는 상관·VIF를 통과한
> AUC 0.8202의 신규 2피처 후보가 확인되었다. 다만 신규 후보는
> 사후 선택 편향 가능성이 있으므로
> 완전 nested 피처 선택 또는 외부 검증 전까지 기존 모델을 유지한다.

## 근거 산출물

- `analysis_outputs/nested_feature_selection_final10_100rep/descriptive_missingness_all_candidates.csv`
- `analysis_outputs/nested_feature_selection_final10_100rep/descriptive_spearman_all_subjects.csv`
- `analysis_outputs/nested_feature_selection_final10_100rep/descriptive_vif_all_candidates.csv`
- `analysis_outputs/nested_feature_selection_final10_100rep/feature_selection_frequency.csv`
- `analysis_outputs/nested_feature_selection_final10_100rep/summary.json`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/all_2_to_4_feature_combinations.csv`
- `analysis_outputs/no_s3_exhaustive_feature_combinations_100rep/full_10_feature_baseline.csv`
- `analysis_outputs/final_training_conditions_no_s3_nested_model_comparison_100rep/metrics_summary.csv`
