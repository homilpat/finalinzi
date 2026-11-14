# 보행 바이오마커 피처 계산 공식

## 1. 전체 계산 과정

```text
20초 원시 3축 가속도
        ↓
V·ML·AP축 정렬 및 100 Hz 리샘플링
        ↓
중앙값 제거 및 0.6~3.0 Hz band-pass filtering
        ↓
10초 window, 2초 step으로 6개 window 생성
        ↓
각 window에서 원피처 5개 계산
        ↓
원피처별 6개 값을 median과 IQR로 요약
        ↓
20초 측정당 10개 후보 피처 생성
```

20초 신호를 100 Hz로 측정하면 축마다 2,000개의 가속도 값이 생성된다.

```text
V  = [v1, v2, ..., v2000]
ML = [m1, m2, ..., m2000]
AP = [a1, a2, ..., a2000]
```

## 2. 전처리

축별 원신호에서 중앙값을 제거한다.

```text
x_centered[n] = x_raw[n] - median(x_raw)
```

이후 4차 Butterworth `0.6~3.0 Hz` band-pass filter를 전후방으로 적용한다.

```text
x_filtered = sosfiltfilt(BandPass_0.6~3.0Hz, x_centered)
```

- 중앙값 제거: 중력 및 DC 성분 감소
- Band-pass filter: 보행 주파수 이외의 저주파·고주파 성분 감소
- 전후방 필터링: 위상 지연 최소화

## 3. 10초 Window 생성

20초 신호를 10초 길이로 자르고 시작점을 2초씩 이동한다.

| Window | 시간 범위 | 100 Hz 샘플 범위 |
|---|---|---|
| W1 | 0~10초 | 1~1,000 |
| W2 | 2~12초 | 201~1,200 |
| W3 | 4~14초 | 401~1,400 |
| W4 | 6~16초 | 601~1,600 |
| W5 | 8~18초 | 801~1,800 |
| W6 | 10~20초 | 1,001~2,000 |

```text
window 개수 = (20-10)/2+1 = 6개
```

중첩된 6개 window는 서로 다른 학습 표본으로 사용하지 않고, 하나의 20초
측정을 안정적으로 요약하기 위한 반복 관찰값으로 사용한다.

## 4. Window별 원피처 5개

| 원피처 | 사용 축 | 핵심 계산 | 반영 특성 |
|---|---|---|---|
| `v_acf_step_stride_symmetry` | V | V축 `Ad1/Ad2` | 수직 step/stride 대칭성 |
| `ap_acf_step_stride_symmetry` | AP | AP축 `Ad1/Ad2` | 추진·감속 step/stride 대칭성 |
| `v_stride_freq_hz` | V | `1 / stride time` | stride 반복 속도 |
| `ap_spec_entropy` | AP | AP 주파수 파워의 Shannon entropy | 진행 방향 리듬의 복잡도 |
| `v_jerk_rms` | V | V축 jerk의 RMS | 수직 충격성과 움직임 활력 |

### 4.1 자기상관함수

한 window의 필터링된 축 신호를 다음과 같이 정의한다.

```text
x = [x1, x2, ..., xN], N = 1,000
```

평균을 제거한다.

```text
z[n] = x[n] - mean(x)
```

lag `k`에서 비편향 자기상관을 계산한다.

```text
R(k) = 1/(N-k) × Σ z[n]z[n+k]
```

lag 0의 자기상관으로 정규화한다.

```text
ACF(k) = R(k) / R(0)
```

수직축과 AP축에서 각각 stride peak와 step peak를 검출한다.

```text
k_stride = 0.8~1.7초 범위에서 ACF가 최대인 lag
Ad2 = ACF(k_stride)

k_step = k_stride의 절반 부근에서 ACF가 최대인 lag
Ad1 = ACF(k_step)
```

코드상 step peak 탐색 범위는 다음과 같다.

```text
0.3 × k_stride ≤ k_step ≤ 0.7 × k_stride
```

#### V ACF step/stride 대칭성

```text
v_acf_step_stride_symmetry = Ad1_V / Ad2_V
```

#### AP ACF step/stride 대칭성

```text
ap_acf_step_stride_symmetry = Ad1_AP / Ad2_AP
```

여기서 `Ad1`은 좌우 step 사이의 유사성을, `Ad2`는 동일한 발의 stride
간 반복성을 나타낸다. `Ad1/Ad2`는 stride 반복성에 대한 step 유사성의
상대적 크기다.

### 4.2 V Stride Frequency

V축 ACF에서 검출한 stride lag를 초로 변환한다.

```text
T_stride = k_stride / fs
```

`fs=100 Hz`이므로:

```text
v_stride_freq_hz = 1/T_stride = 100/k_stride
```

예를 들어 `k_stride=110 samples`이면:

```text
T_stride = 110/100 = 1.10초
v_stride_freq_hz = 1/1.10 = 0.909 Hz
```

### 4.3 AP Spectral Entropy

AP축 window 신호에 Hann window 기반 spectrogram을 적용한다.

```text
S(f,t) = Spectrogram(AP)
```

`0.6~3.0 Hz` 범위의 time-frequency power만 선택하여 하나의 배열로
펼친다.

```text
P = [P1, P2, ..., PM]
```

파워의 합이 1이 되도록 정규화한다.

```text
p_i = P_i / ΣP_i
```

정규화된 Shannon entropy를 계산한다.

```text
ap_spec_entropy
= -Σ(p_i × log2(p_i)) / log2(M)
```

- 낮은 값: 파워가 특정 보행 주파수에 집중됨
- 높은 값: 파워가 여러 주파수와 시간에 분산됨

### 4.4 V Jerk RMS

필터링된 V축 가속도의 인접 샘플 차이를 계산한다.

```text
Δv[n] = v[n+1] - v[n]
```

샘플링 주파수를 곱해 시간 미분을 근사한다.

```text
jerk[n] = fs × Δv[n]
        = 100 × (v[n+1] - v[n])
```

10초 window에는 1,000개의 가속도 값과 999개의 jerk 값이 생성된다.

```text
v_jerk_rms = sqrt(Σ jerk[n]^2 / 999)
```

가속도 단위가 `g`라면 jerk RMS 단위는 `g/s`다.

## 5. 6×5 Window-Feature 행렬

6개 window에서 5개 원피처를 계산하면 다음 행렬이 생성된다.

| Window | V ACF | AP ACF | V stride frequency | AP spectral entropy | V jerk RMS |
|---|---:|---:|---:|---:|---:|
| W1 | f11 | f12 | f13 | f14 | f15 |
| W2 | f21 | f22 | f23 | f24 | f25 |
| W3 | f31 | f32 | f33 | f34 | f35 |
| W4 | f41 | f42 | f43 | f44 | f45 |
| W5 | f51 | f52 | f53 | f54 | f55 |
| W6 | f61 | f62 | f63 | f64 | f65 |

각 열에는 동일한 원피처를 서로 다른 window에서 계산한 값 6개가 있다.

## 6. Median 계산

원피처 하나에서 계산된 6개 값을 다음과 같이 정의한다.

```text
F = [F1, F2, F3, F4, F5, F6]
```

크기순으로 정렬한다.

```text
F(1) ≤ F(2) ≤ F(3) ≤ F(4) ≤ F(5) ≤ F(6)
```

값이 6개이므로 중앙의 세 번째와 네 번째 값의 평균이 median이다.

```text
median(F) = [F(3) + F(4)] / 2
```

예를 들어 6개 window의 V jerk RMS가 다음과 같다면:

```text
[1.20, 1.40, 1.30, 1.80, 1.50, 1.60]
```

정렬 후:

```text
[1.20, 1.30, 1.40, 1.50, 1.60, 1.80]
```

Median은:

```text
v_jerk_rms_median = (1.40 + 1.50)/2 = 1.45
```

Median은 20초 동안 나타난 해당 보행 특성의 대표 수준이다.

## 7. IQR 계산

IQR은 75백분위수에서 25백분위수를 뺀 값이다.

```text
IQR(F) = Q75(F) - Q25(F)
```

현재 코드의 `pandas.Series.quantile()` 기본 방식과 동일하게 선형보간을
사용한다. 6개 값에서는 백분위수 위치를 다음과 같이 계산한다.

```text
Q25 위치 = (6-1) × 0.25 = 1.25
Q75 위치 = (6-1) × 0.75 = 3.75
```

0부터 위치를 세므로 정렬된 6개 값에서:

```text
Q25 = F(2) + 0.25 × [F(3)-F(2)]
Q75 = F(4) + 0.75 × [F(5)-F(4)]
```

앞의 V jerk RMS 예시에서는:

```text
Q25 = 1.30 + 0.25 × (1.40-1.30)
    = 1.325

Q75 = 1.50 + 0.75 × (1.60-1.50)
    = 1.575

v_jerk_rms_iqr = 1.575-1.325
               = 0.25
```

IQR은 20초 동안 window별 보행 특성이 얼마나 달라졌는지를 나타낸다.
극단적인 한두 window 값의 영향을 표준편차보다 적게 받는다.

## 8. 최종 후보 피처 10개

| 원피처 | 20초 대표 수준 | 20초 내 변동성 |
|---|---|---|
| V ACF 대칭성 | `v_acf_step_stride_symmetry_median` | `v_acf_step_stride_symmetry_iqr` |
| AP ACF 대칭성 | `ap_acf_step_stride_symmetry_median` | `ap_acf_step_stride_symmetry_iqr` |
| V stride frequency | `v_stride_freq_hz_median` | `v_stride_freq_hz_iqr` |
| AP spectral entropy | `ap_spec_entropy_median` | `ap_spec_entropy_iqr` |
| V jerk RMS | `v_jerk_rms_median` | `v_jerk_rms_iqr` |

```text
5개 원피처 × 2개 요약통계(median, IQR)
= 20초 측정당 10개 후보 피처
```

## 9. 표현 시 주의사항

Median과 IQR은 원시 가속도 2,000개를 직접 요약한 값이 아니다.

```text
잘못된 설명:
20초 원시 가속도의 median과 IQR을 계산했다.

정확한 설명:
20초 안의 6개 window에서 원피처를 각각 계산한 뒤,
원피처별 6개 값의 median과 IQR을 계산했다.
```

Median과 IQR 자체가 센서 노이즈를 제거하는 필터는 아니다. 센서 및
비보행 주파수 성분은 band-pass filtering으로 줄이고, median과 IQR은
일시적인 충격·회전·불안정한 window의 영향을 줄이면서 20초 측정의
대표 수준과 시간적 변동성을 요약하기 위해 사용한다.

## 10. 발표용 설명

> 20초 원시 가속도 신호를 보행축으로 정렬하고 100 Hz로 리샘플링한
> 뒤, 0.6~3.0 Hz band-pass filtering을 적용했습니다. 이후 10초
> window를 2초 간격으로 이동하여 총 6개 window를 생성하고, 각
> window에서 ACF 대칭성, stride frequency, spectral entropy 및 jerk
> RMS를 계산했습니다. 이렇게 생성된 원피처별 6개 값의 median과 IQR을
> 산출하여, 20초 측정의 대표 수준과 측정 중 변동성을 각각 표현했습니다.
