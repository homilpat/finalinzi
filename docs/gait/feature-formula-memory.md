# 보행 모델 피처 계산 메모

## 모델 전제

- 검사 방식: 회전을 허용하는 20초 자유보행
- 명시적인 회전 구간 제거 필터는 사용하지 않는다.
- 자연스러운 회전과 보행 변화는 10초 하위창별 피처 변화 및 IQR에 반영된다.
- 원시 허리 가속도는 100 Hz의 V(수직), ML(좌우), AP(진행방향) 축으로 사용한다.
- 20초 신호(2,000 samples)를 10초 창(1,000 samples), 2초 간격으로 나누어 총 6개 창을 만든다.
- 5개 원피처 각각의 median과 IQR을 계산하여 10개 후보 피처를 만든다.

## 원시 센서값 변환

PhysioNet 원시 정수 센서값은 다음과 같이 실제 가속도로 변환한다.

```text
a_d[n] = (raw_d[n] - baseline_d) / gain_d
d ∈ {V, ML, AP}
```

각 10초 창의 V축과 AP축은 중앙값을 제거한 뒤 4차 Butterworth
0.6~3.0 Hz band-pass filter를 `sosfiltfilt` 방식으로 적용한다.

```text
x_centered[n] = x[n] - median(x)
x_filtered[n] = Butterworth_BPF_0.6_to_3.0Hz(x_centered[n])
```

## 원피처 1: V축 ACF step/stride 반복 대칭성

필터링된 수직 가속도에서 평균을 제거하고 정규화 자기상관함수를 계산한다.

```text
C_V[k] = (1 / (N-k)) × Σ(v[n] - mean(v)) × (v[n+k] - mean(v))
R_V[k] = C_V[k] / C_V[0]
```

0.8~1.7초 범위에서 stride peak `Ad2_V`를 찾는다. 검출된 stride
주기의 절반 부근인 `0.3×T_stride ~ 0.7×T_stride`에서 step peak
`Ad1_V`를 찾는다.

```text
V_ACF_step_stride_symmetry = Ad1_V / Ad2_V
```

현재 정식 이름은 `v_acf_step_stride_symmetry`이다. 과거
`v_harmonic_ratio`라는 이름을 사용했지만 Fourier harmonic ratio가
아니며 계산값은 동일하다.

## 원피처 2: AP축 ACF step/stride 반복 대칭성

V축과 동일한 계산을 AP축 가속도에 적용한다.

```text
C_AP[k] = (1 / (N-k)) × Σ(ap[n] - mean(ap)) × (ap[n+k] - mean(ap))
R_AP[k] = C_AP[k] / C_AP[0]
AP_ACF_step_stride_symmetry = Ad1_AP / Ad2_AP
```

현재 정식 이름은 `ap_acf_step_stride_symmetry`이다. 과거 이름은
`ap_harmonic_ratio`이다.

## 원피처 3: V축 stride frequency

V축 ACF에서 검출한 stride lag를 사용한다.

```text
T_stride_V = k_stride_V / fs
v_stride_freq_hz = 1 / T_stride_V = fs / k_stride_V
```

단위는 Hz이다.

## 원피처 4: AP축 spectral entropy

필터링된 AP 가속도에 4초 Hann window의 spectrogram을 계산하고
0.6~3.0 Hz의 power만 사용한다.

```text
p_i = S_i / ΣS_i
ap_spec_entropy = -Σ(p_i × log2(p_i)) / log2(M)
```

- 낮은 값: power가 특정 주파수에 집중된 비교적 규칙적인 신호
- 높은 값: power가 여러 주파수와 시간에 분산된 복잡한 신호

## 원피처 5: V축 jerk RMS

필터링된 수직 가속도의 1차 차분을 이용한다.

```text
j_V[n] = (v[n+1] - v[n]) × fs
v_jerk_rms = sqrt((1 / (N-1)) × Σ(j_V[n]^2))
```

가속도 단위가 g이면 jerk RMS의 단위는 g/s이다. 움직임의 급격성 또는
부드러움을 반영하지만 보행속도와 가속도 진폭의 영향도 받는다.

## 20초 요약 및 10개 후보 피처

각 원피처를 6개의 10초 창에서 계산한 뒤 다음과 같이 요약한다.

```text
feature_median = median(F_1, ..., F_6)
feature_iqr = percentile75(F_1, ..., F_6) - percentile25(F_1, ..., F_6)
```

10개 후보 피처:

```text
v_acf_step_stride_symmetry_median
v_acf_step_stride_symmetry_iqr
ap_acf_step_stride_symmetry_median
ap_acf_step_stride_symmetry_iqr
v_stride_freq_hz_median
v_stride_freq_hz_iqr
ap_spec_entropy_median
ap_spec_entropy_iqr
v_jerk_rms_median
v_jerk_rms_iqr
```

## 현재 최종 모델 입력 3개

```text
v_jerk_rms_median
v_jerk_rms_iqr
v_acf_step_stride_symmetry_iqr
```

최종 3개는 10개 후보의 단일 피처 및 2~4개 조합 성능, 모델 단순성,
Spearman 상관계수와 VIF를 함께 검토하여 고정했다. 이후 고정된 3개
피처의 no-s3 로지스틱 회귀 모델을 subject 단위 반복 nested CV로
평가했다.

## 표현상 주의점

- `보행의 크기`는 논문에서 `trunk acceleration magnitude`,
  `trunk acceleration amplitude` 또는 `gait intensity`로 표현할 수 있다.
- 허리 가속도만으로 지면 충격과 추진력을 직접 분리해 측정했다고
  표현하면 안 된다.
- ACF step/stride 피처는 왼발과 오른발을 직접 구분한 해부학적 대칭성이
  아니라 step과 stride 주기의 반복 관계이다.
- 현재 모델은 회전 능력을 별도 점수화하는 모델이 아니다.
