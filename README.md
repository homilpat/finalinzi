# 파이널 프로젝트 데모

## 최종 보행 모델 재현 파일 색인

최종 확정 모델은 `MOCA/models/gait_daily_clinical_3feat.joblib`이며, 아래 파일들이 학습·전처리·보정·검증·시각화 재현에 사용되는 핵심 스크립트입니다.

| 목적 | 파일명 | 역할 |
|------|--------|------|
| 최종 모델링 / 재학습 | `analysis_scripts/retrain_acconly_clean.py` | 최종 acc-only 3피처 로지스틱 회귀 모델 재학습, 100회 반복 5-fold 검증, VIF 확인, `gait_daily_clinical_3feat.joblib` 및 metadata 저장 |
| 보행 구간 추출 / 20s→10s 서브윈도우 | `analysis_scripts/build_75h_subwindow_median_iqr.py` | PhysioNet 75h 일상보행에서 20초 구간을 만들고 내부 10초 슬라이딩 서브윈도우 피처를 median/IQR로 집계 |
| 런타임 전처리 / 피처 추출 | `MOCA/gait_axis_aligned_core.py` | CSV 파싱, 축 정렬(V/ML/AP), 100Hz 리샘플링, 밴드패스 필터, 서브윈도우 피처 추출 |
| 서버 예측 파이프라인 | `MOCA/gait_axis_aligned_processor.py` | 앱/APK CSV를 최종 모델 입력으로 변환하고 `predict_daily_gait_csv()`로 추론 |
| 스마트폰-허리센서 보정 | `analysis_scripts/calibrate_waist_sensor_range_loss.py` | 실제 스마트폰 샘플과 PhysioNet 기준 신호 범위 차이를 비교해 sensor-level 보정 계수 산출 |
| 최종 ML/DL 모델 비교 | `analysis_scripts/compare_final_speed_or6_models.py` | LR, RF, SVM, GBM, XGB, Voting, Stacking, CNN1D, LSTM 비교 평가 |
| 혼동행렬/ROC 시각화 출력 | `analysis_scripts/compare_final_speed_or6_models.py` | `analysis_outputs/final_model_comparison_speed_or6/`에 모델별 confusion matrix, ROC curve, 전체 ROC 비교 이미지 저장 |
| 후보 피처/도메인 민감도 시각 분석 | `analysis_scripts/summarize_comprehensive_gait_feature_analysis.py` | 서비스 후보 피처, 도메인 민감도, 조합 후보를 요약 CSV/Markdown으로 정리 |
| 정상/저하 보행 패턴 비교 시각화 | `analysis_scripts/compare_normal_impaired_gait_patterns.py` | 정상 보행 기준 데이터와 저하군 보행 패턴의 피처 차이 및 분리 가능성 분석 |

최종 서비스 라벨은 `TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19 OR base_velocity < 1.0 OR s3_velocity < 1.0`이며, 최종 입력 피처는 `v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr` 3개입니다.

## 개요

스마트폰 기반 인지·보행 이중 선별 시스템. 병원 방문 없이 태블릿/스마트폰만으로 MoCA-K 인지검사와 보행 운동기능 평가를 동시에 수행하고, 케어타입(A~D형)을 자동 분류한다.

배포 앱: `MOCA/` 디렉토리

```bash
cd MOCA
pip install -r requirements.txt
python app.py
```

Render 배포: 루트 `render.yaml`, 대시보드 Root Directory = `MOCA`

---

## MoCA-K 인지검사 모듈

### 개요

**MoCA-K (Montreal Cognitive Assessment - Korean)** 한국판 인지 선별 검사를 웹 기반으로 구현. 총 30점 만점이며 23점 미만 시 MCI(경도인지장애) 의심으로 판정한다.

- 저작권: © Z. Nasreddine MD, 한국판 JY. Lee / www.mocatest.org
- 교육 보정: 교육연수 6년 이하 → +1점 (최대 30점)

### 검사 버전 로테이션

반복 검사 시 문항 암기 효과를 방지하기 위해 **6개월 주기**로 버전을 자동 교체한다.

| 버전 | 기억 단어 | 유창성 과제 | 동물 (어휘력) | 추상력 쌍 |
|------|-----------|-------------|---------------|-----------|
| **MoCA-K** | 얼굴·비단·교회·진달래·빨강 | 시장에서 살 수 있는 것 11개↑ | 사자·코뿔소·낙타 | 기차-자전거 / 시계-자 |
| **K-MoCA** | 얼굴·비단·학교·피리·노랑 | ㄱ으로 시작하는 단어 6개↑ | 사자·박쥐·낙타 | 기차-비행기 / 시계-저울 |

→ 관련 파일: `MOCA/version_manager.py`

### 검사 항목 (총 30점)

| 항목 | 배점 | 입력 방식 | 채점 모듈 |
|------|------|-----------|-----------|
| **길만들기** | 1점 | 터치 드로잉 (숫자-한글 교차 연결) | `trail_making.py` |
| **드로잉** | 4점 | 손으로 그리기 (육면체 1점 + 시계 3점) | `cube.py`, `clock.py` |
| **어휘력** | 3점 | 동물 3마리 이름 말하기 (STT) | `naming.py` |
| **기억력** | 5점 | 단어 5개 즉각회상(×2) + 지연회상 (STT) | `memory.py` |
| **주의력** | 6점 | 숫자 따라하기·거꾸로·박수치기·연속빼기7 (STT) | `attention.py` |
| **언어** | 3점 | 문장 따라 말하기 + 유창성 (STT) | `language.py` |
| **추상력** | 2점 | 공통점 말하기 (STT) | `abstraction.py` |
| **지남력** | 6점 | 연·월·일·요일·장소·시군구 말하기 (STT) | `orientation.py` |

### 드로잉 채점 (규칙 기반 컴퓨터 비전)

손으로 그린 이미지를 OpenCV 기반 규칙 알고리즘으로 자동 채점한다.

**육면체 (`cube.py`) — 1점**

Hough Line Transform으로 선분을 추출한 뒤 4가지 기준을 모두 만족하면 1점.

1. 선분 개수: 6~20개 (육면체 12모서리 기준, 오차 허용)
2. 방향 다양성: 수평+수직+대각 또는 등각 투영(수평+좌대각+우대각) 구조
3. 평행선 쌍 존재 + 같은 방향 선분 간 길이 변동계수(CV) ≤ 0.5
4. 과도한 선분 없음 (30개 이하, 덧그리기 방지)

**시계 (`clock.py`) — 3점**

| 항목 | 채점 방법 | 기준 |
|------|-----------|------|
| 윤곽 (1점) | Hough Circle Transform | 반지름 80~250px 원이 이미지 중앙 30% 이내에 감지 |
| 숫자 (1점) | 컨투어 검출 → 30° 구역 매핑 | 12개 구역 중 10개↑ 점유, 구역당 3개 초과 없음, 총 8~16개 |
| 바늘 (1점) | Hough Line → 중심 기점 선분 필터링 | 11시 10분 방향(시침 240°·분침 330°, ±35°), 시침 < 분침 길이 |

### STT (음성인식)

모든 구술 응답은 `whisper_stt.py`를 통해 Whisper(OpenAI) 기반으로 전사한다. 브라우저에서 녹음 → 서버 전송 → 전사 → 채점 순으로 처리.

### MCI 판정 기준

```
최종점수 = 원점수 + 교육보정(교육연수 ≤ 6년이면 +1, 최대 30점)

최종점수 ≥ 23점 → 정상
최종점수 < 23점  → MCI 의심
```

→ 관련 파일: `MOCA/total_scorer.py`

---

## 최종 보행 모델 (2026-07-22 클린 재학습 확정)

### 모델: `gait_daily_clinical_3feat.joblib`

**일상보행 + 임상·보행속도 확장 OR 라벨 기반 acc-only 3피처 subwindow 집계 모델**

| 항목 | 값 |
|------|----|
| 모델 파일 | `MOCA/models/gait_daily_clinical_3feat.joblib` |
| 메타데이터 | `MOCA/models/gait_daily_clinical_3feat_metadata.json` |
| 피처 추출기 | `MOCA/gait_axis_aligned_core.py` → `extract_subwindow_daily_features()` |
| 예측 함수 | `MOCA/gait_axis_aligned_processor.py` → `predict_daily_gait_csv()` |

#### 피처 (3개)

| 피처명 | 설명 | 논문 근거 |
|--------|------|-----------|
| `v_jerk_rms_median` | 20s 내 10s 서브윈도우 수직 Jerk RMS의 중앙값 | Kavanagh & Menz 2008 |
| `v_jerk_rms_iqr` | 수직 Jerk RMS의 사분위범위 (변동성) | Kavanagh & Menz 2008 |
| `v_harmonic_ratio_iqr` | 수직 Harmonic Ratio IQR (보행 리듬 일관성) | Moe-Nilssen & Helbostad 2004 |

> Harmonic Ratio = ACF step peak / ACF stride peak (계단 대 보폭 비율, 보행 대칭성 지표)

#### 훈련 데이터

- **데이터셋**: PhysioNet 75h 일상보행 (47명 고령자, 도보 포함 전체 활동)
- **피험자 수**: 71명 (정상 31명, 저하 40명; 보행속도 포함 확장 라벨 기준)
- **라벨링**: CO/FL(낙상이력)이 아닌 **임상 운동평가 + 보행속도 컷오프 OR 조합** 재라벨링
  - 기준: `TUG ≥ 12s OR FSST ≥ 15s OR BERG < 52 OR DGI ≤ 19 OR base_velocity < 1.0 m/s OR s3_velocity < 1.0 m/s`
  - 지표별 양성 수: TUG 22명, FSST 16명, BERG 20명, DGI 17명, base velocity 21명, s3 velocity 28명
- **윈도우**: 20s 단위, 내부 10s 서브윈도우 슬라이딩(step=2s, ~6개/20s)

#### 임상 운동기능 저하 라벨 근거

최종 보행 라벨은 낙상이력(CO/FL)을 그대로 쓰지 않고, 임상 운동평가 지표의 기능저하 컷오프를 OR 조건으로 조합해 정의한다. 이 방식은 특정 낙상 사건 자체가 아니라, 낙상위험과 관련된 보행·균형·이동성 저하 상태를 선별하기 위한 조작적 정의다.

| 지표 | 컷오프 | 임상적 의미 | 근거 |
|------|--------|-------------|------|
| TUG | `≥ 12s` | 일어나기, 걷기, 회전, 착석을 포함한 기능적 이동성 저하 | fall risk functional measure systematic review에서 `≥12s`가 근거 있는 기능 측정치로 제시됨 |
| FSST | `≥ 15s` | 사방 stepping, 방향전환, 동적 균형 및 민첩성 저하 | Dite & Temple 계열 older adult 연구에서 `>15s`가 multiple fall risk 기준으로 제시됨 |
| BERG | `< 52` | 정적·동적 균형 저하 및 낙상위험 증가 신호 | BBS fall-risk 연구에서 약 45~51/52점대 컷오프가 사용됨. 단일검사만으로 낙상 예측 확정에는 한계가 있어 다요인 라벨의 한 축으로 사용 |
| DGI | `≤ 19` | 속도 변화, 고개 돌림, 장애물, 계단 등 동적 보행능력 저하 | Shumway-Cook / DGI 계열 연구와 RehabMeasures에서 19점 이하가 낙상위험 증가 기준으로 제시됨 |

보행속도는 임상적으로 기능적 이동성 저하를 나타내는 대표 지표이며, 1.0 m/s 미만은 노인 기능저하·이동성 제한·낙상위험 관련 연구에서 널리 쓰이는 기준이다. 따라서 최종 라벨에는 다음 기준을 포함한다.

```text
최종 운동기능 저하 라벨 =
TUG ≥ 12s
OR FSST ≥ 15s
OR BERG < 52
OR DGI ≤ 19
OR base_velocity < 1.0 m/s
OR s3_velocity < 1.0 m/s
```

`base_velocity`와 `s3_velocity`는 라벨 정의에만 사용하며, 최종 acc-only 모델 입력 피처에는 넣지 않는다. 모델 입력은 `v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr` 3개뿐이므로 보행속도 값 자체를 예측변수로 사용하는 직접 데이터누수는 없다. 단, 보행속도는 보행 IMU 피처와 의미적으로 가까운 기능 지표이므로, 발표와 문서에서는 "보행속도 저하까지 포함한 확장 운동기능 저하 라벨"로 명확히 설명한다.

**최종 라벨 검증 결과**: `TUG/FSST/BERG/DGI OR + base_velocity < 1.0 OR s3_velocity < 1.0` 기준에서는 71명 중 정상 31명, 저하 40명으로 재라벨링된다. 동일한 acc-only 3피처 모델을 subject-level `StratifiedGroupKFold` 5-fold × 100회로 검증했다. 최종 서비스 임계값은 선별 목적을 고려해 **fixed threshold = 0.50**으로 설정했으며, 100회 반복 OOF 기준 AUC는 **0.866 ± 0.009**, sensitivity **0.829 ± 0.021**, specificity **0.736 ± 0.027**이다.

#### 피처 산출 상세

**공통 전처리 파이프라인**

```
CSV (Timestamp_ns, Acc_Vertical_g, Acc_ML_g, Acc_AP_g)
  → 100Hz 리샘플 (선형 보간)
  → [도메인 보정] 수직 신호 × α(1.9705)
  → 0.6~3.0Hz Butterworth 4차 대역통과 필터 (보행 주파수 대역)
  → 20s 단위 분할 → 10s 서브윈도우 슬라이딩 (step=2s, ~6개/20s)
  → 서브윈도우별 피처 계산 → 집계
```

---

**피처 1: `v_jerk_rms_median`**

```
Jerk(t) = diff(v_bp(t)) × fs          # 수직 가속도의 1차 미분 × 샘플링 주파수
v_jerk_rms = sqrt( mean( Jerk(t)² ) ) # 10s 서브윈도우 RMS
v_jerk_rms_median = median( 서브윈도우별 jerk_rms )
```

- **물리적 의미**: 보행 중 수직 방향 충격의 평균 강도. 걸음이 부드러울수록 낮고, 불규칙하거나 발을 끌면 높음.
- **왜 선택했나**: Kavanagh & Menz (2008) — 노인 낙상 위험군에서 jerk RMS가 유의미하게 높음을 검증. 운동기능 저하군은 발이 지면에 더 세게 닿거나 중심이 불안정해 jerk 값이 커짐.
- **median 사용 이유**: 보행 중간에 잠깐 멈추거나 방향 전환 시 극단값이 튀는데, median은 이런 outlier에 강건함.

---

**피처 2: `v_jerk_rms_iqr`**

```
v_jerk_rms_iqr = Q75(jerk_rms) - Q25(jerk_rms)   # 서브윈도우 간 IQR
```

- **물리적 의미**: 보행 전반에 걸쳐 충격 강도가 얼마나 일정한가. IQR이 크면 어떤 걸음은 강하고 어떤 걸음은 약하다는 뜻 → 보행 리듬이 불안정.
- **왜 선택했나**: 평균(median)만으로는 "전반적으로 약한 충격"과 "들쭉날쭉한 충격"을 구분 못함. IQR은 보행 내 변동성을 직접 측정. 운동기능 저하군은 피로나 균형 문제로 걸음마다 힘 조절이 달라져 IQR이 높게 나타남.
- **주의**: 피처 레벨 덧셈 보정을 하지 않고 신호 레벨 α 보정을 쓰는 이유가 바로 이 피처 때문. 덧셈 보정 시 IQR에 delta를 더하면 음수가 되는 물리적 모순 발생.

---

**피처 3: `v_harmonic_ratio_iqr`**

```
# 자기상관함수(ACF)로 보행 주기 추출
c_v = ACF( v_bp )                         # 수직 신호 자기상관
stride_peak = ACF peak at lag 0.8~1.7s    # 보폭 주기(~1.14s = 0.88Hz)
step_peak   = ACF peak at lag stride/2    # 발걸음 주기 (stride의 절반)

v_harmonic_ratio = step_peak / stride_peak  # 10s 서브윈도우 HR
v_harmonic_ratio_iqr = Q75(HR) - Q25(HR)   # 서브윈도우 간 IQR
```

- **물리적 의미**: Harmonic Ratio는 좌우 발걸음의 대칭성 지표. 완벽히 대칭이면 ACF step peak ≈ stride peak → HR ≈ 1.0. 한쪽 발을 더 세게 딛거나 보폭이 다르면 HR이 낮아짐. IQR은 이 대칭성이 보행 중 얼마나 흔들리는지를 측정.
- **왜 선택했나**: Moe-Nilssen & Helbostad (2004) — Harmonic Ratio는 노인 낙상 예측에서 가장 강력한 보행 피처 중 하나. 운동기능 저하군은 한쪽 다리 약화나 통증으로 좌우 비대칭이 커짐.
- **median 대신 IQR 사용 이유**: HR median은 정상군과 저하군 간 분포 겹침이 많아 변별력이 낮음. IQR(시간적 변동성)이 AUC 기여도 높고 VIF도 낮아 최종 선택.

---

**왜 이 3개만 쓰나 (피처 선택 근거)**

| 검증 항목 | 결과 |
|-----------|------|
| VIF (분산팽창지수) | 1.54~2.17 — 3피처 간 다중공선성 낮음 |
| Train/Test AUC 갭 | +0.012 — 과적합 없음 |
| Feature importance | 이 3개가 subject-level AUC 0.881 달성 (4피처 이상 추가 시 오히려 하락) |
| 임상 논문 근거 | Kavanagh & Menz 2008, Moe-Nilssen & Helbostad 2004 |

→ 피처 수가 적을수록 소규모 데이터(71명)에서 과적합 위험이 낮고, 이 3개가 이미 충분한 임상 변별력을 가짐.

---

#### 성능 지표

**평가 방식**: StratifiedGroupKFold 5-fold, 100회 반복 (group = subject_id, fixed screening threshold = 0.50)

**100회 반복 CV 결과 (안정화 평균, n=100)**

| 지표 | Train | Test | Gap (Train − Test) |
|------|-------|------|---------------------|
| AUC | 0.892 | 0.866 ± 0.009 | 0.026 |
| Sensitivity | 0.845 | 0.829 | 0.017 |
| Specificity | 0.751 | 0.736 | 0.015 |

> Train/Test 모두 100회 반복 5-fold CV, concatenated OOF per seed 방식. Gap이 모두 0.03 이하로 과적합 없음.

- 100회 반복 AUC 95% CI: 약 0.849 ~ 0.878
- threshold=0.50 기준 sensitivity=0.829, specificity=0.736

**요약**

| 단위 | AUC | Sensitivity | Specificity |
|------|-----|-------------|-------------|
| 윈도우 (7,061개) — OOF | 0.751 | 0.806 | 0.610 |
| Subject (71명) — 100회 반복 CV 평균 | 0.866 ± 0.009 | 0.829 | 0.736 |
| **Threshold** | **0.50** | **선별 목적 고정 임계값** | **sensitivity ≥ 0.8 우선** |

- Threshold 전략: 선별 목적상 sensitivity 0.8 이상을 우선해 fixed threshold 0.50 사용
- VIF: 1.54~2.17 (다중공선성 낮음, 통상 주의 기준 5보다 충분히 낮음)
- Gap 모두 극소 → 과적합 없음

#### 도메인 보정 (신호 레벨, 고정값)

**방식**: 원시 수직 신호 진폭 기준 스케일 보정 (Signal-level amplitude correction)

PhysioNet 훈련 데이터와 우리 스마트폰 앱 측정값 사이의 신호 진폭 차이를 원시 신호 레벨에서 먼저 보정한 뒤 피처를 추출한다. 피처 레벨에서 덧셈/뺄셈으로 보정하면 `v_jerk_rms_iqr` 같은 피처가 음수로 내려가는 물리적 모순이 생기기 때문에 신호 레벨 보정이 원칙적으로 올바르다.

| 파라미터 | 값 | 의미 |
|----------|----|------|
| `alpha` | **1.9705** | 수직 신호 전체 진폭 배율 |
| `tau` | **1.0** | 시간축 배율 (보행주파수 동일 → 왜곡 없음) |

**산출 과정**:

1. **PhysioNet 기준값**: `analysis_outputs/waist_sensor_range_loss_calibration/physionet_waist_normal_raw_reference.csv`  
   → 정상 38명의 bandpass 수직신호 RMS 중앙값 = **0.1939 g**

2. **OUR_SAMPLE 측정값**: 정상 6명 (발다침·조심보행 샘플 제외) bandpass 수직신호 RMS 중앙값 = **0.0984 g**

3. **α 산출**: `α = 0.1939 / 0.0984 = 1.9705`  
   보행주파수 비교 결과 PhysioNet(0.858 Hz) ≈ OUR(0.874 Hz) → τ = 1.0

4. **적용**: 모든 입력 CSV → VMLAP 정렬 → `× alpha` → 피처 추출 → 모델 추론

**결과 (OUR_SAMPLE 10개 샘플: 정상 7개 + 저하 3개)**:

| 그룹 | 보정 후 확률 범위 | threshold=0.50 |
|------|------------------|-----------------|
| 정상 (7개 샘플) | 0.000 ~ 0.322 | 전원 정상 판정 ✓ |
| 저하 (3개 샘플) | 0.552 ~ 0.946 | 전원 저하 판정 ✓ |

→ 확장 라벨 재학습 모델의 fixed screening threshold(0.50) 사용

> **보정값 고정 근거**: PhysioNet 기준값(0.1939 g)은 38명 고정 데이터. OUR_SAMPLE도 동일 앱·측정 방식이면 진폭이 크게 변하지 않으므로 α=1.9705 고정 사용.  
> 앱 측정 방식(폰 위치, 수집 파이프라인)이 바뀔 때만 재산출 필요.

---

## CSV 업로드 흐름 (`POST /gait/upload-csv`)

```
스마트폰 보행 CSV 업로드
→ Acc_Vertical_g / Acc_ML_g / Acc_AP_g 축보정 컬럼 읽기
   (없으면 Acc_X/Y/Z로 fallback → 중력축 자동 정렬)
→ 100Hz 리샘플
→ [신호 레벨 도메인 보정] 수직/ML/AP 전체 × alpha(1.9705)
→ 0.6–3.0 Hz Butterworth 대역통과 필터 (window_features 내부)
→ 20s 단위 분할 → 내부 10s 서브윈도우 슬라이딩(step=2s, ~6개)
→ 각 서브윈도우: ACF 계산 → v_harmonic_ratio, v_jerk_rms
→ 서브윈도우 통계 집계: MEDIAN + IQR → 3피처
→ 3피처 LogisticRegression 추론 (RobustScaler + SimpleImputer)
→ threshold=0.50 기준 저하/정상 판정
```

CSV 권장 컬럼: `Timestamp_ns`, `Acc_Vertical_g`, `Acc_ML_g`, `Acc_AP_g`
최소 녹화 시간: **20초 이상 보행**

---

## 모델 비교 실험 (동일 3피처 기반)

### 최종 확장 라벨 기준 모델 비교

최종 확장 라벨(`TUG/FSST/BERG/DGI OR + base_velocity < 1.0 OR s3_velocity < 1.0`) 기준으로 전통 ML 모델과 CNN/LSTM을 동일한 subject-level 5-fold split에서 비교했다. 모든 전처리는 fold 내부에서만 fit했고, CNN/LSTM도 train fold에서 fit한 imputer/scaler를 test fold에 transform만 적용했다. 출력 임계값은 최종 서비스 설정과 동일하게 `threshold=0.50`으로 고정했다.

| 모델 | Test AUC | Sens | Spec | Gap | Threshold median |
|------|--------:|-----:|-----:|----:|-----------------:|
| **LR (최종 채택)** | **0.8819** | **0.8552** | **0.6935** | **+0.0098** | **0.3908** |
| XGB | 0.8547 | 0.8645 | 0.6439 | +0.0255 | 0.4703 |
| SVM | 0.8579 | 0.8525 | 0.6448 | +0.0401 | 0.3897 |
| Voting | 0.8763 | 0.8370 | 0.6965 | +0.0451 | 0.4371 |
| Stacking | 0.8727 | 0.8372 | 0.7110 | +0.0662 | 0.5246 |
| RF | 0.8754 | 0.8027 | 0.7474 | +0.0801 | 0.4519 |
| CNN1D | 0.8614 | 0.7965 | 0.7458 | +0.0960 | 0.4760 |
| GBM | 0.8748 | 0.7890 | 0.7965 | +0.1231 | 0.5663 |
| LSTM | 0.8155 | 0.8450 | 0.6506 | +0.0503 | 0.3156 |

LR이 Test AUC 0.8819로 전체 1위이고 gap +0.0098로 가장 작다. Voting/RF/GBM은 gap이 0.08~0.12로 커서 71명 소표본에서 과적합 위험이 크다. LR은 AUC 최고·gap 최소·설명 가능성까지 갖춰 최종 서비스 모델로 유지한다.

혼동행렬과 ROC 커브는 `analysis_outputs/final_model_comparison_speed_or6/`에 저장했다.

```text
roc_all_models.png
roc_curve_*.png
confusion_matrix_*.png
model_comparison_summary.csv
train_fold_metrics.csv
test_fold_metrics.csv
```

### 실험 설정

| 항목 | 내용 |
|------|------|
| 피처 | `v_jerk_rms_median`, `v_jerk_rms_iqr`, `v_harmonic_ratio_iqr` (3개 동일) |
| 훈련/평가 단위 | **Subject 71명** (subject 1행 = 자신의 window 피처 중앙값) |
| 분할 | 5-fold StratifiedKFold on subjects |
| 스케일링 | **fold 안에서만** (전통 ML: pipeline 내부, DL: fold-내 IMP+SC fit) |
| threshold | 모든 모델 비교와 최종 서비스 모델 모두 **fixed threshold=0.50** 사용 |
| LSTM/CNN | subject당 20s 윈도우 시퀀스 (max_len=100 × 3 features, GPU CUDA 12.4) |
| 스크립트 | `analysis_scripts/compare_final_speed_or6_models.py` |

> 이전 버전 버그: ① 글로벌 스케일러가 test 포함 71명 전체에 fit → 누수, ② threshold를 OOF 전체에서 Youden 최적화 후 같은 데이터에 적용 → Sen 부풀림. 두 버그 모두 수정.

## 케어타입 분류

```
인지 0 / 보행 0 → A형 (유지형)
인지 0 / 보행 1 → C형 (신체관리형)
인지 1 / 보행 0 → B형 (인지관리형)
인지 1 / 보행 1 → D형 (통합관리형)
```

---

## 앱 실행

```bash
cd MOCA
pip install -r requirements.txt
python app.py
```

Render 배포: 루트 `render.yaml`, `rootDir: MOCA`

---

## 분석 스크립트 파일 목록 (`analysis_scripts/`)

### 전처리 · 피처 추출

| 파일 | 역할 |
|------|------|
| `build_75h_subwindow_median_iqr.py` | PhysioNet 75h 일상보행 → 20s/10s 서브윈도우 슬라이딩 → v_jerk_rms / v_harmonic_ratio 집계 (최종 모델 입력 피처 산출) |

### 라벨링

| 파일 | 역할 |
|------|------|
| `retrain_acconly_clean.py` | 최종 라벨(`TUG/FSST/BERG/DGI OR + base_velocity < 1.0 OR s3_velocity < 1.0`) 생성, 3피처 모델 재학습, artifact/metadata 저장 |

### 모델링 · 검증

| 파일 | 역할 |
|------|------|
| `retrain_acconly_clean.py` | 최종 로지스틱 회귀 모델 확정, 100회 반복 subject-level CV, VIF/피처 분포 요약 저장 |
| `compare_final_speed_or6_models.py` | 최종 라벨 기준 LR/RF/SVM/XGB/Voting/Stacking/CNN/LSTM 비교, train-test gap, 혼동행렬, ROC 커브 생성 |

### 도메인 보정

| 파일 | 역할 |
|------|------|
| `calibrate_waist_sensor_range_loss.py` | PhysioNet 정상군 원시 수직신호 RMS 기준값 산출 → `analysis_outputs/waist_sensor_range_loss_calibration/physionet_waist_normal_raw_reference.csv` |
| `retrain_acconly_clean.py` | 최종 모델 artifact에 신호 레벨 보정값 `alpha=1.9705`, `tau=1.0` 저장 |

### 서비스 파이프라인 (`MOCA/`)

| 파일 | 역할 |
|------|------|
| `gait_axis_aligned_core.py` | 축정렬, 리샘플, 대역통과 필터, 서브윈도우 피처 추출, 신호 변환(`transform_signal`) |
| `gait_axis_aligned_processor.py` | CSV → 신호 레벨 보정(α) → 피처 추출 → 모델 추론 |
| `models/gait_daily_clinical_3feat.joblib` | 최종 로지스틱 회귀 모델 artifact (signal_correction, threshold 포함) |
| `models/gait_daily_clinical_3feat_metadata.json` | 모델 메타데이터 (AUC, 피처, 보정 파라미터) |

---

## 앱 데모 계획 (발표용)

### 아키텍처

```
폰 (Expo React Native 앱)
  ├── 보행 측정 (네이티브 센서)  ──POST /gait/upload-csv──┐
  └── MoCA 평가 (WebView)       ──http://노트북IP:5000──┐ │
                                                        ↓ ↓
                                          노트북 (Flask 로컬 서버)
                                            - 보행 모델 추론
                                            - Whisper STT
                                            - OpenCV 드로잉 채점
                                            - MoCA 채점 로직
                                            - DB (회원·기록)
```

### 역할 분담

| 앱 (React Native) | 서버 (노트북 로컬) |
|-------------------|--------------------|
| 센서 수집 (가속도 중심, 자이로 수집 가능) | acc-only 보행 모델 추론 |
| 보행 CSV 생성 및 전송 | Whisper STT |
| MoCA WebView 렌더링 | OpenCV 드로잉 채점 |
| 결과 화면 표시(WebView/앱) | MoCA 채점, 보행 판정, DB |

### 발표 시나리오

```
폰 앱 실행
  → 보행 측정 20초 (네이티브)
  → CSV 자동 서버 전송 → 보행 결과 표시
  → MoCA 평가 버튼 탭
  → WebView에서 인지검사 진행 (노트북 Flask)
  → 케어타입 (A~D형) 최종 결과
```

### 발표 환경 설정

- 노트북과 폰을 같은 WiFi 또는 핫스팟으로 연결
- `ipconfig`로 노트북 IPv4 확인 → 앱 `SERVER_URL` 설정
- `python app.py` → `0.0.0.0:5000` 실행
- 공용 WiFi는 AP isolation 가능 → 핫스팟으로 대체

### 이후 고도화 계획 (발표 후)

- AIHub 한국어 노인 음성 데이터로 Whisper 파인튜닝
- MoCA 화면 전체 React Native 네이티브 구현
- 파인튜닝 모델 서버 적용

---

## 고려사항 (향후 개선 검토)

### 피처 변경 / 나이·성별 추가 실험 (2026-07-23)

현행 3피처 모델이 최선임을 확인하기 위해 아래 실험을 수행했다. **모델은 현행 유지**.  
공통 subjects n=70 (정상 30 / 저하 40), 100-rep 5-fold StratifiedGroupKFold CV.

| 모델 | 피처 구성 | AUC | sens | spec |
|------|-----------|-----|------|------|
| ① 현행 **(배포 중)** | BPF jerk median/iqr + HR iqr (3피처) | **0.894 ±0.018** | 0.825 | 0.667 |
| ② 현행 + 나이/성별 | 현행 3피처 + age + gender (5피처) | **0.925 ±0.014** | 0.875 | **0.933** |
| ③ 논문기반 | LPF(20Hz) jerk mean/iqr + BPF HR median (3피처) | 0.792 ±0.027 | 0.875 | 0.600 |
| ④ 논문기반 + 나이/성별 | 논문 3피처 + age + gender (5피처) | 0.853 ±0.020 | 0.850 | 0.700 |
| ⑤ 나이/성별만 | age + gender (2피처) | 0.800 ±0.021 | 0.825 | 0.600 |

**실험 결론**

- **논문 표준 LPF jerk**는 실험실 환경에 최적화된 방식으로, 75h 일상보행에서는 3~20Hz 고주파 노이즈가 섞여 BPF jerk보다 AUC 약 0.10 낮음. 현행 BPF 방식 유지가 올바른 선택.
- **나이/성별 추가(②)** 시 AUC +0.031, spec +0.266으로 성능이 크게 향상되지만 인구통계 변수 의존도가 높아져 순수 보행 기능 선별 목적과 멀어짐. 나이 자체가 라벨과 높은 상관을 가지므로 과제의 의미가 희석됨.
- 관련 실험 스크립트: `analysis_scripts/experiment_paper_based_3feat.py`

### 펭트 AI 어시스턴트 로직 (2026-07-24)

펭트 관련 코드를 `MOCA/pengteu.py`로 분리하면서 확인한 동작상 짚을 점. **현행 유지하되 발표/개선 시 참고**.

**1. GPT가 핵심 도메인 질문에는 호출되지 않음** (`app.py` `assistant_chat_api`)

```python
if _pengteu_local_answer_ready(...):   # 보행/인지/운동 키워드 매칭되면
    reply = _basic_pengteu_reply(...)  # → 고정 템플릿
else:                                   # 키워드 없으면
    reply = _openai_pengteu_fallback() # → GPT 호출
```

- "보행 어때요", "내 점수 뭐야" 같은 핵심 도메인 질문은 키워드에 매칭돼 **전부 고정 템플릿 문장**으로 응답하고, GPT(OpenAI)는 키워드에 걸리지 않는 질문에만 호출됨.
- 비용/속도 절약 및 응답 안정성 측면에서는 합리적이나, "AI 펫 코치"의 개인화된 답변을 기대하는 질문일수록 정작 template만 나가는 구조.
- **발표 데모 시**: 시연 질문을 고를 때 이 동작을 인지하고, 개인화된 답변을 보여주려면 키워드에 안 걸리는 질문을 쓰거나 로컬 템플릿 자체를 풍부하게 다듬어야 함.

**2. `_clean_pengteu_reply` 필터가 과함** (`pengteu.py`)

- 응답 후처리의 blocked 토큰에 `"이 내용을 바탕으로"`, `"RAG"` 같은 흔한 표현이 포함돼, GPT가 자연스럽게 이 문구를 쓰면 **답변 전체가 버려지고 캔 문구로 대체**됨.
- `"이 내용을 바탕으로"`는 한국어에서 흔한 접속 표현이라 멀쩡한 GPT 답변도 날아갈 위험. 발표 데모에서 GPT 답변이 갑자기 이상한 고정 문구로 바뀌는 원인이 될 수 있음.
- **개선안**: blocked 토큰을 RAG 내부 구조 노출(`retrieved_knowledge`, `/static/audio`)로만 좁히고, 일반 접속 표현은 제외.

**사소한 크루프트 (버그 아님)**

- `_pengteu_local_answer_ready(message, knowledge=None)` — `knowledge` 인자를 받기만 하고 사용하지 않음.
- `_safe_int`가 `app.py`·`pengteu.py` 양쪽에 중복 정의 — 모듈 독립성을 위한 의도적 중복, 무해.
