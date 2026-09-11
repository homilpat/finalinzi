# FINAL__2026 정리 문서

작성일: 2026-07-10  
프로젝트: Labwalks 실험실 10초 보행 IMU 운동기능 평가 모델

## 1. 현재 최종 방향

본 프로젝트의 최종 방향은 낙상 여부를 직접 예측하는 모델이 아니라, 10초 보행 IMU 데이터로부터 임상 운동기능 저하 가능성을 선별하는 모델이다.

최종 서비스 구조는 다음과 같다.

```text
디지털 MoCA 기반 인지평가
+ 10초 보행 IMU 기반 운동평가
-> 인지 정상/저하 가능 x 운동 정상/저하 가능
-> 4개 사용자군 분류
-> 사용자군별 개인화 이중과제 솔루션 제공
-> RAG 기반 AI 펫 코치가 설명, 접근성, 운동 안내 보조
```

최종 4분류는 다음과 같이 정의한다.

| 군 | 인지평가 | 운동평가 | 솔루션 방향 |
|---|---|---|---|
| A | 인지 정상 | 운동 정상 | 유지/예방형 이중과제 |
| B | 인지 정상 | 운동저하 가능 | 운동기능 강화 중심 이중과제 |
| C | 인지저하 가능 | 운동 정상 | 인지자극 중심 이중과제 |
| D | 인지저하 가능 | 운동저하 가능 | 저강도 안전형 복합 이중과제 |

## 2. 최종 운동저하 라벨 정의

최종 운동평가 모델의 임상 기준 라벨은 다음과 같다.

```text
운동저하 가능군 = DGI <= 19 OR TUG >= 12
```

이 기준은 DGI와 TUG를 이용해 동적 보행능력 및 기능적 이동성 저하를 반영하는 조작적 정의이다. 의료 진단 라벨이 아니라, 10초 보행 IMU 기반 운동기능 저하 가능성 선별을 위한 임상 기준 라벨로 사용한다.

| 임상지표 | 의미 |
|---|---|
| DGI | 걷는 중 속도 변화, 고개 돌림, 장애물, 계단 등 동적 보행능력 |
| TUG | 일어나기, 걷기, 회전, 착석을 포함한 기능적 이동성 |

현재 매칭 대상자는 67명이며 (CO024·FL020 제외 — base_v_stride_regularity 전 구간 NaN), 위 기준에 따른 라벨 분포는 다음과 같다.

| 라벨 | 인원 |
|---|---:|
| 운동저하 가능군 | 25 |
| 운동 정상군 | 42 |
| 총 subject | 67 |

## 3. 최종 전처리 방식

최종 전처리 산출물은 `01_preprocessing/` 폴더에 정리했다.

주요 전처리 요약:

| 항목 | 값 |
|---|---:|
| 데이터 출처 | Labwalks 실험실 보행 기록 |
| 보행 window 길이 | 10초 |
| stride step | 2.5초 |
| turn 제외 margin | 4초 |
| acc Butterworth lowpass | 20 Hz |
| v·ml bandpass | 0.6–3.0 Hz |
| gyro(roll) bandpass | 0.5–5.0 Hz |
| matched subjects | 67명 |

전처리는 Labwalks 실험실 보행 IMU 데이터에서 10초 슬라이딩 윈도우를 추출하고, acc/gyro feature를 계산한 뒤 subject-level로 집계하는 방식이다.

중요한 점:

```text
window를 각각 독립 학습 샘플로 사용하지 않음
전체 window feature를 subject 단위로 먼저 집계 (median/IQR pooling)
subject 1명 = 모델 입력 1행
이후 subject-level에서 train/test 및 교차검증 수행
```

따라서 같은 사람의 window가 train/test에 섞이는 window-level leakage를 피했다.

## 4. 최종 모델

최종 모델 파일은 `02_model/final_motor_domain4_labwalks10_logistic_C0p5.joblib` 이다.

| 항목 | 값 |
|---|---|
| 모델 | Logistic Regression L2 |
| class imbalance 처리 | `class_weight=balanced` |
| feature 선택 방식 | domain4_fixed (4개 도메인 대표 feature 고정) |
| 선택 feature 수 | 4개 |
| C 값 | 0.5 |
| threshold 전략 | train fold에서 Youden index(sensitivity + specificity - 1) 최대 |
| 최종 threshold | 0.5243060158115224 |

최종 선택 feature:

| feature | 해석 | 도메인 |
|---|---|---|
| `v_amp_pool_median` | 수직 진폭 중앙값 | 보행 활력 |
| `ml_amp_pool_iqr` | 좌우 진폭 변동성 | 좌우 안정성 |
| `base_v_stride_regularity` | 수직축 stride 규칙성 | 리듬 규칙성 |
| `roll_amp_pool_iqr` | roll 진폭 변동성 | 몸통 회전 안정성 |

이 4개 지표는 10초 실험실 보행 IMU에서 임상 운동기능 저하 가능군을 구분하는 데 사용된 핵심 도메인 대표 지표이다. 운동능력 전체를 완전히 대표한다고 표현하지 않고, 10초 허리 IMU 기반 핵심 판정 지표로 표현한다.

## 5. 교차검증 성능 및 과적합 확인

검증 방식:

```text
A scheme: 5-fold cross-validation x 100 repeats (주요 검증)
B scheme: 3-fold cross-validation x 100 repeats
C scheme: 8:2 split x 100 repeats
E scheme: Leave-One-Subject-Out (LOSO)
feature selection 방식: domain4_fixed (train fold 외부에서 고정)
threshold 선택: train fold 안에서만 수행
test fold는 최종 평가에만 사용
```

검증 방식별 pooled subject-level 성능 (threshold strategy: youden):

| 검증방식 | AUC | Sensitivity | Specificity |
|---|---:|---:|---:|
| A (5-fold×100) | 0.830 | 0.800 | 0.738 |
| B (3-fold×100) | 0.827 | 0.760 | 0.738 |
| C (8:2×100) | 0.836 | 0.760 | 0.738 |
| E (LOSO) | 0.830 | 0.760 | 0.738 |

CV 최종 추천 모델 (A scheme, youden) fold-level 평균 성능:

| split | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Train CV mean | 0.870 | 0.815 | 0.877 | 0.778 | 0.780 |
| Test CV mean | 0.836 | 0.736 | 0.770 | 0.716 | 0.681 |
| Gap | 0.034 | 0.079 | 0.107 | 0.062 | 0.099 |

Train-test gap summary (youden):

| 검증방식 | AUC gap | Accuracy gap | Sensitivity gap | Specificity gap | F1 gap |
|---|---:|---:|---:|---:|---:|
| A (5-fold×100) | 0.034 | 0.079 | 0.107 | 0.062 | 0.099 |
| B (3-fold×100) | 0.045 | 0.104 | 0.137 | 0.084 | 0.132 |
| C (8:2×100) | 0.064 | 0.111 | 0.130 | 0.098 | 0.150 |

Apparent train 성능 (전체 데이터 1회):

| AUC | Accuracy | Sensitivity | Specificity | F1 |
|---:|---:|---:|---:|---:|
| 0.865 | 0.821 | 0.840 | 0.810 | 0.778 |

해석:

```text
AUC gap은 약 0.034(A scheme)로 크지 않지만, sensitivity/F1 gap은 약 0.10 수준이라 일반화 불확실성은 존재함
feature 4개 + Logistic Regression이라 모델 복잡도 낮음
4가지 검증 방식 모두에서 pooled subject-level AUC 0.827~0.836으로 안정적
LOSO fold-level sensitivity(0.290)는 test subject 1명이라 양성 없는 fold에서
0으로 처리되어 왜곡됨 → subject-level pooled sensitivity(0.760)가 올바른 값
외부 검증 데이터가 없기 때문에 최종 표현은 진단이 아니라 선별 모델로 제한해야 함
```

보고서에는 전체 데이터 apparent train 성능이 아니라 위의 반복 교차검증 test 평균 성능을 사용한다.

## 6. 서비스 리포트 구성

최종 사용자 리포트에는 모델이 실제 사용하는 핵심 지표와 4분류 결과를 중심으로 보여준다.

권장 리포트 구성:

| 섹션 | 내용 |
|---|---|
| 종합 결과 | 인지 정상/저하 가능, 운동 정상/저하 가능, 최종 4분류 |
| 디지털 MoCA 결과 | 총점, 정상/인지저하 가능 여부, 영역별 점수 |
| 운동평가 결과 | 운동저하 가능성 점수, 정상/주의/저하 가능 |
| 핵심 보행 지표 | 수직 진폭, 좌우 진폭 변동성, stride 규칙성, roll 변동성 |
| 군별 해석 | 4분류 중 해당 군의 의미 |
| 개인화 이중과제 | 해당 군에 맞는 훈련 솔루션 |
| AI 펫 코치 안내 | 쉬운 설명, TTS 속도, 글씨 크기, 다시 설명 기능 |
| 주의 문구 | 의료 진단이 아닌 선별/관리 보조 도구임을 명시 |

## 7. RAG 기반 AI 펫 코치 역할

AI 펫 코치는 최종 4분류를 새로 결정하는 모델이 아니라, 결과 설명과 실행 보조 역할을 한다.

주요 기능:

| 기능 | 설명 |
|---|---|
| 결과 설명 | MoCA와 운동평가 결과를 쉬운 말로 설명 |
| 군별 솔루션 안내 | 4분류 결과에 맞는 이중과제 훈련 안내 |
| 접근성 조정 | 글씨 크기 키우기, TTS 속도 느리게/빠르게 조절 |
| 쉬운 말 변환 | 어려운 용어를 고령 사용자에게 이해 쉬운 문장으로 재설명 |
| 보호자 설명 | 보호자가 이해할 수 있는 요약 문장 제공 |
| 훈련 실행 보조 | 같은 군 안에서 난이도 낮은 설명 또는 반복 안내 제공 |

RAG에 넣을 자료:

```text
운동 리포트 해석 가이드
DGI/TUG/MoCA 설명 문서
4분류별 이중과제 솔루션 콘텐츠
안전 수칙
접근성 설정 코드/명령 스키마
앱 UI 조작 가이드
```

## 8. 폴더 및 파일 설명

### `01_preprocessing/`

| 파일 | 설명 |
|---|---|
| `labwalks_service10_amp_spec_features.csv` | 최종 모델 입력에 사용한 10초 window amp+spec feature CSV (67명) |

### `02_model/`

| 파일 | 설명 |
|---|---|
| `final_motor_domain4_labwalks10_logistic_C0p5.joblib` | 최종 배포 후보 모델 |
| `final_motor_domain4_labwalks10_logistic_C0p5_metadata.json` | 라벨, threshold, 선택 feature, CV 성능 메타데이터 |
| `domain4_full_validation_metrics.csv` | A/B/C/LOSO 검증 방식별 fold-level 지표 |
| `domain4_oof_predictions.csv` | A/B/C/LOSO OOF 예측값 |
| `domain_binary_metrics_summary.csv` | CV 평균 성능 요약 |
| `domain_binary_metrics_by_fold.csv` | fold별 상세 성능 |
| `domain_feature_groups.csv` | 모델링에 사용한 feature group 정의 |
| `domain_selected_features_summary.csv` | 반복 CV에서 선택된 feature 요약 |
| `domain_selected_features_by_fold.csv` | fold별 선택 feature 상세 |

### `03_code/`

| 파일 | 설명 |
|---|---|
| `01_preprocessing/extract_labwalks_service20_features.py` | Labwalks IMU에서 슬라이딩 윈도우 feature 추출 (`--window-sec`으로 10/15/20초 모두 지원) |
| `01_preprocessing/run_extract_labwalks_service_windows.py` | 오케스트레이터 (기본 10,15,20초 순서 실행) |
| `02_modeling/RUN_service10_domain_representative_model_compare.py` | domain4 모델 학습 및 strategy 비교 |
| `03_validation/RUN_final_model_full_validation_suite.py` | A/B/C/LOSO 전체 검증 suite |
| `03_validation/RUN_final_model_stability_checks.py` | 안정성 확인 |
| `04_visualization/RUN_final_service10_visualizations.py` | 최종 모델 기준 시각화 |

### `04_clinical_data/`

| 파일 | 설명 |
|---|---|
| `ClinicalDemogData_COFL.xlsx` | DGI, TUG, FSST, BERG, velocity 등 임상/인구통계 데이터 |

## 9. 재현 명령어

**전처리 (10초 window feature 추출):**

```powershell
python 03_code/01_preprocessing/run_extract_labwalks_service_windows.py --windows 10
```

또는 단독 실행:

```powershell
python 03_code/01_preprocessing/extract_labwalks_service20_features.py `
  --window-sec 10 `
  --stride-sec 2.5 `
  --turn-exclude-margin-sec 4.0 `
  --out-dir 01_preprocessing/
```

**모델 학습 (domain4 비교):**

```powershell
python 03_code/02_modeling/RUN_service10_domain_representative_model_compare.py
```

**전체 검증 suite:**

```powershell
python 03_code/03_validation/RUN_final_model_full_validation_suite.py
```

**시각화:**

```powershell
python 03_code/04_visualization/RUN_final_service10_visualizations.py
```

## 10. Android 온디바이스 구현 계획

MOCA 앱에 보행(Gait) 탭 추가 — Android 네이티브 (Kotlin)  
허리밴드에 스마트폰 세로 고정 → 10초 걷기 → 즉석 분류  
서버 없이 완전 온디바이스 추론

**모델 가중치 (하드코딩용):**

```
intercept = -0.24046268336238408
threshold = 0.5243060158115224
```

StandardScaler mean/scale은 joblib에서 추출 필요 (`pipeline.named_steps['scale'].mean_`, `.scale_`)

**Android 축 → Labwalks 축 매핑 (허리밴드 세로 고정 기준):**

| Android 축 | 보행 축 |
|---|---|
| `acc.z` | `v` (수직) |
| `acc.x` | `ml` (좌우) |
| `acc.y` | `ap` (전후) |
| `gyro.z` | `roll` |

**Butterworth 계수 (Python에서 미리 계산 → Kotlin 하드코딩):**

```python
from scipy.signal import butter
sos_v_ml = butter(4, [0.6/50, 3.0/50], btype='bandpass', output='sos')   # v, ml
sos_roll = butter(4, [0.5/50, 5.0/50], btype='bandpass', output='sos')   # roll
```

**구현 단계:**

1. SensorManager로 IMU 10초 수집 (`SENSOR_DELAY_FASTEST` → ~100Hz)
2. Python에서 Butterworth 계수 추출 → Kotlin 하드코딩
3. 필터 적용 + amplitude_pooling (median, IQR) 구현
4. ACF + stride peak 탐지 → `base_v_stride_regularity`
5. joblib에서 scaler mean/scale + 모델 가중치 추출 → 하드코딩
6. 추론 + 결과 UI
7. MOCA 앱 탭에 통합

## 11. 최종 보고서 표현 권장 문장

```text
본 연구는 디지털 MoCA 기반 인지평가와 10초 보행 IMU 기반 운동평가를 결합하여, 인지기능 및 운동기능 저하 가능성을 선별하고 4개의 사용자군에 따른 개인화 이중과제 솔루션을 제공하는 서비스를 목표로 한다.
```

```text
운동평가 모델은 DGI 19점 이하 또는 TUG 12초 이상을 만족하는 경우를 임상 운동기능 저하 가능군으로 조작적으로 정의하였다. 최종 모델은 subject-level로 집계된 수직 진폭, 좌우 진폭 변동성, stride 규칙성, 몸통 roll 변동성의 4개 핵심 지표를 사용하였다.
```

```text
본 결과는 의료적 진단이 아니라, 디지털 인지평가와 보행 센서 기반 운동평가를 활용한 인지·운동기능 저하 가능성 선별 및 관리 보조 정보이다.
```

## 12. 현재 남은 주의점

| 항목 | 상태 |
|---|---|
| 외부 검증 | 아직 없음. 향후 독립 데이터 필요 |
| 대상자 수 | 67명으로 작음. 반복 CV + 4가지 검증 방식으로 보완했지만 한계 존재 |
| 최종 라벨 | DGI/TUG 기반 조작적 정의. 표준 진단명 아님 |
| 모델 용도 | 진단 모델이 아니라 선별/관리 보조 모델 |
| Android 구현 | 아직 시작 전 (Kotlin 포팅 필요) |
