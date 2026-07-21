# 파이널 프로젝트 데모

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

## 최종 보행 모델 (2026-07-21 확정)

### 모델: `gait_daily_clinical_3feat.joblib`

**일상보행 + 임상 OR 라벨 기반 3피처 subwindow 집계 모델**

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
- **피험자 수**: 71명 (정상 36명, 저하 35명)
- **라벨링**: CO/FL(낙상이력)이 아닌 **임상 운동평가 OR 조합** 재라벨링
  - 기준: `motor_impairment_score ≥ 0.5` (DGI, TUG, FSST, BERG 복합 점수)
  - DGI ≤ 19, TUG ≥ 12s, FSST ≥ 15s, BERG ≤ 45 등 기준 OR 조합
- **윈도우**: 20s 단위, 내부 10s 서브윈도우 슬라이딩(step=2s, ~6개/20s)

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
| VIF (분산팽창지수) | 1.26~1.75 — 3피처 간 다중공선성 없음 |
| Train/Test AUC 갭 | +0.012 — 과적합 없음 |
| Feature importance | 이 3개가 subject-level AUC 0.881 달성 (4피처 이상 추가 시 오히려 하락) |
| 임상 논문 근거 | Kavanagh & Menz 2008, Moe-Nilssen & Helbostad 2004 |

→ 피처 수가 적을수록 소규모 데이터(71명)에서 과적합 위험이 낮고, 이 3개가 이미 충분한 임상 변별력을 가짐.

---

#### 성능 지표

| 단위 | AUC | Sensitivity | Specificity | Threshold |
|------|-----|-------------|-------------|-----------|
| 윈도우 (7,061개) | 0.751 | 0.806 | 0.610 | 0.470 |
| **Subject (71명)** | **0.881** | **0.971** | **0.722** | 0.470 |

- GroupKFold 5-fold OOF (group = subject_id)
- Threshold 전략: sens ≥ 0.80 최적 spec
- VIF: 전 피처 1.26~1.75 (다중공선성 없음)
- Train/Test AUC Gap: +0.012 (과적합 없음)

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

**결과 (OUR_SAMPLE 8명: 정상 6명 + 저하 2명)**:

| 그룹 | 보정 후 확률 범위 | threshold=0.470 |
|------|------------------|-----------------|
| 정상 (6명) | 0.009 ~ 0.352 | 전원 정상 판정 ✓ |
| 저하 (2명) | 0.612 ~ 0.814 | 전원 저하 판정 ✓ |

→ 8/8 정답, 원래 PhysioNet Youden 임계값(0.470) 그대로 사용

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
→ threshold=0.470 기준 저하/정상 판정
```

CSV 권장 컬럼: `Timestamp_ns`, `Acc_Vertical_g`, `Acc_ML_g`, `Acc_AP_g`
최소 녹화 시간: **20초 이상 보행**

---

## 보조 모델 (Fallback)

`gait_daily_clinical_3feat.joblib`이 없을 경우 `gait_axis_aligned_physionet_youden.joblib`으로 자동 폴백.

### (이전) 축정렬 4피처 모델 (`gait_axis_aligned_physionet_youden.joblib`)

| 항목 | 값 |
|------|----|
| 피처 | `v_harmonic_ratio`, `ap_harmonic_ratio`, `v_stride_freq_hz`, `ap_spec_entropy` |
| 라벨 | 정상 vs 운동기능저하 (LabWalk + PhysioNet) |
| OOF AUC | 0.880 |
| Threshold | 0.440 (sens≥0.80) |
| 훈련 데이터 | PhysioNet LabWalks + UCI_HAR + GEOTEC_SP (correctable 3 domains) |

---

## 모델 개발 히스토리

| 버전 | 피처 | AUC (OOF) | 라벨 | 비고 |
|------|------|-----------|------|------|
| v1 (구) | ACF 원시 피크 4개 | 0.851 | CO/FL | axis-aligned |
| v2 | Harmonic Ratio 4개 | 0.880 | 정상/저하 (lab) | sens=0.811, spec=0.782 |
| **v3 (최종)** | **jerk+HR 3개** | **0.881 (subject)** | **임상 OR 라벨** | VIF<2, gap=0.012 |

---

## 케어타입 분류

```
인지 0 / 보행 0 → A형 (유지형)
인지 0 / 보행 1 → B형 (신체관리형)
인지 1 / 보행 0 → C형 (통합관리형)
인지 1 / 보행 1 → D형 (인지관리형)
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
| `build_75h_median_iqr_model.py` | 서브윈도우 median+IQR 피처로 로지스틱 회귀 학습 |
| `build_75h_daily_walk_model.py` | 75h 일상보행 전체 대상 초기 모델 학습 |
| `extract_physionet_labwalks_shape_features_all_or.py` | PhysioNet LabWalks 파형 형상 피처 추출 |

### 라벨링

| 파일 | 역할 |
|------|------|
| `reclabel_clinical_model.py` | CO/FL 이진 라벨 → 임상 운동평가 OR 조합(`motor_impairment_score ≥ 0.5`) 재라벨링 |
| `feature_selection_clinical.py` | 재라벨 데이터 기반 피처 중요도 스크리닝 |

### 모델링 · 검증

| 파일 | 역할 |
|------|------|
| `quick_physionet_only_8020_feature_screen.py` | PhysioNet 단독 80/20 분할 피처 후보 스크리닝 |
| `vif_and_width_test.py` | VIF 다중공선성 검사 + 신뢰구간 폭 확인 |
| `train_test_gap_check.py` | 훈련/테스트 AUC 갭 과적합 여부 확인 |
| `rebuild_harmonic_ratio_model.py` | Harmonic Ratio 기반 모델 재학습 |
| `retrain_filtered_domain_corrected.py` | 도메인 보정 후 필터링된 데이터로 재훈련 |
| `screen_gait_model_options_after_domain_audit.py` | 도메인 감사 후 모델 옵션 스크리닝 |

### 도메인 보정

| 파일 | 역할 |
|------|------|
| `calibrate_waist_sensor_range_loss.py` | PhysioNet 정상군 원시 수직신호 RMS 기준값 산출 → `analysis_outputs/waist_sensor_range_loss_calibration/physionet_waist_normal_raw_reference.csv` |
| `finalize_daily_model_with_correction.py` | 보정 파라미터를 모델 artifact에 저장하여 최종 확정 |
| `signal_level_mmd_correction.py` | 신호 레벨 amplitude(α) + time warp(τ) MMD 최적화 실험 스크립트 |
| `upgrade_affine_correction.py` | Affine 보정 실험 (현재 미사용, 참조용) |

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
| 센서 수집 (가속도·자이로) | 보행 모델 추론 |
| 보행 CSV 생성 및 전송 | Whisper STT |
| MoCA WebView 렌더링 | OpenCV 드로잉 채점 |
| 결과 화면 표시 | MoCA 채점 및 DB |

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
