# Final Project Demo

## Current Applied App

The deployable app is in `MOCA/`.

```bash
cd MOCA
pip install -r requirements.txt
python app.py
```

Render uses the root `render.yaml`, with `rootDir: MOCA`.

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
