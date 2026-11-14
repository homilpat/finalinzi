# 보행저하 선별모델 피처 산출 및 선정 발표 대본

## 1. 발표 대본

저희 모델은 스마트폰 가속도 원시 신호에서 보행의 충격 특성과
주기적 대칭성을 정량화하고, 이를 이용해 보행저하 가능성을 선별하는
로지스틱 회귀 모델입니다.

먼저 입력 가속도는 수직축인 V, 좌우축인 ML, 진행 방향인 AP축으로
정렬했습니다. 이미 해부학적 축으로 저장된 데이터는 해당 축을 그대로
사용했고, 스마트폰의 X, Y, Z 원시축만 있는 경우에는 중력 성분이 가장
큰 축을 수직축으로 정했습니다. 나머지 두 축 중 보행 주파수 대역의
분산이 큰 축을 AP축, 다른 축을 ML축으로 정했습니다.

센서마다 샘플링 간격이 다를 수 있으므로 모든 신호를 100 Hz로
리샘플링했습니다. 이후 각 축에서 중앙값을 제거하고, 4차 Butterworth
필터를 이용해 0.6~3.0 Hz 보행 대역만 남겼습니다. 필터는
`sosfiltfilt` 방식으로 전후 방향으로 적용하여 위상 지연을 최소화했습니다.

한 번의 측정은 20초이며, 이를 10초 길이의 window로 나누었습니다.
window 시작점을 2초씩 이동했기 때문에 0, 2, 4, 6, 8, 10초에서 시작하는
총 6개 window가 생성됩니다. 중첩 window 6개를 서로 독립적인 학습
표본으로 사용한 것은 아니며, 하나의 20초 측정을 요약하기 위한
반복 관찰값으로 사용했습니다.

각 10초 window에서 처음 검토한 원피처는 5개입니다.

첫 번째는 수직축 ACF step/stride 대칭성입니다. 수직 가속도의
자기상관함수를 계산하고, 0.8~1.7초 범위에서 stride 주기의 peak인
Ad2를 검출했습니다. 그 절반 부근에서 step 주기의 peak인 Ad1을
검출한 뒤, Ad1을 Ad2로 나눈 비율을 계산했습니다. 이 값은 Fourier
harmonic ratio가 아니라, step과 stride의 반복성을 비교한
ACF 기반 대칭성 비율입니다.

두 번째는 AP축 ACF step/stride 대칭성입니다. 계산법은 수직축과
동일하지만, 진행 방향 가속도를 사용하여 좌우 step에서 앞뒤 추진과
감속 패턴이 얼마나 유사하게 반복되는지를 측정합니다.

세 번째는 수직 stride frequency입니다. 수직 ACF에서 검출한 stride
시간을 초 단위로 구하고, 그 역수인 `1 / stride time`으로 계산했습니다.
단위는 Hz이며 초당 stride 수를 의미합니다.

네 번째는 AP spectral entropy입니다. AP 가속도에 Hann window를
사용한 spectrogram을 계산하고, 0.6~3.0 Hz 대역의 power를 확률분포로
정규화한 뒤 Shannon entropy를 계산했습니다. 특정 주파수에 power가
집중되면 값이 낮고, 여러 주파수와 시간에 분산되면 값이 높습니다.

다섯 번째는 수직 jerk RMS입니다. jerk는 가속도의 시간 미분이므로,
연속한 수직 가속도 차이에 100 Hz를 곱해 계산했습니다. 이 jerk 값을
제곱하고 평균을 낸 뒤 제곱근을 취하여 RMS를 구했습니다. 수직 가속도
단위가 g이므로 jerk RMS의 단위는 g/s입니다.

각 원피처는 20초 측정 안에서 6개의 값을 갖습니다. 저희는 이 6개
값의 중앙값인 median과 75백분위수에서 25백분위수를 뺀 IQR을
계산했습니다. Median은 해당 측정의 대표적인 보행 상태를 나타내고,
IQR은 측정 중 보행 특성이 window마다 얼마나 변했는지를 나타냅니다.
평균과 표준편차 대신 median과 IQR을 사용한 이유는 회전, 일시적 충격,
센서 흔들림과 같은 극단값의 영향을 줄이기 위해서입니다.

따라서 5개 원피처에서 median과 IQR을 각각 생성하여 총 10개의
후보 피처를 구성했습니다.

피처 선택 단계에서는 먼저 각 후보를 하나씩 사용한 로지스틱 회귀의
subject-grouped 5-fold OOF AUC를 비교했습니다. 동일한 사람의 여러
segment가 train과 test에 동시에 들어가지 않도록 subject ID를 group으로
사용했습니다.

단일 피처 결과에서는 `v_jerk_rms_median`이 AUC 0.7217로 가장 높았고,
`v_jerk_rms_iqr`이 0.6227로 두 번째였습니다. 수직 stride frequency
median은 0.5995, AP spectral entropy median은 0.5979였습니다.

그다음 상위 후보에서 2개부터 4개까지의 조합을 비교했습니다.
`v_jerk_rms_median`, `v_jerk_rms_iqr`,
`v_acf_step_stride_symmetry_iqr` 조합이 AUC 0.7550으로 가장 높았습니다.
전체 10개 피처를 모두 사용한 결과는 0.7510이었고, stride frequency를
추가한 4피처 조합도 0.7531이었습니다. 따라서 성능이 더 낮거나 거의
같은 피처를 추가하지 않고, 가장 단순하면서 성능이 높았던 3개 조합을
선택했습니다.

첫 번째 최종 피처인 `v_jerk_rms_median`은 측정 중 수직 추진과 충격
변화량의 대표 수준을 나타냅니다. 두 번째인 `v_jerk_rms_iqr`은 같은
물리량의 단순 복제가 아니라, 20초 동안 추진과 충격 특성이 얼마나
변했는지를 나타냅니다. 세 번째인
`v_acf_step_stride_symmetry_iqr`은 step과 stride 대칭성 비율이
window마다 얼마나 흔들렸는지를 나타냅니다. 따라서 최종 3개는
대표 수준, 시간적 변동성, 주기적 대칭성 변동이라는 서로 다른 정보를
조합합니다.

피처 사이의 중복성을 확인하기 위해 상관분석과 VIF 검사를 수행했습니다.
Spearman 상관계수는 jerk median과 jerk IQR이 0.6756,
jerk median과 ACF 대칭성 IQR이 -0.6209,
jerk IQR과 ACF 대칭성 IQR이 -0.0421이었습니다.
일부 중간 정도의 상관은 있었지만 완전히 동일한 정보를 나타내는
수준은 아니었습니다.

VIF는 각 피처를 나머지 피처로 설명한 결정계수 R제곱을 이용하여
`1 / (1 - R제곱)`으로 계산했습니다. 최종 피처의 VIF는 각각
2.17, 1.54, 1.59였으며, 일반적인 다중공선성 주의 기준인 5보다
낮았습니다. 따라서 세 피처를 동시에 로지스틱 회귀에 넣어도 계수가
심하게 불안정해질 정도의 다중공선성은 없다고 판단했습니다.

모델 학습 시 결측값은 training fold의 중앙값으로 대체했고,
각 피처는 training fold의 median과 IQR을 사용하는 RobustScaler로
변환했습니다. 이후 로지스틱 회귀가 세 피처의 선형 결합을 sigmoid
함수에 통과시켜 보행저하 가능성 확률을 출력하도록 했습니다.
서비스 임계값은 0.50으로 고정했습니다.

정리하면 저희는 원시 가속도를 보행축으로 정렬하고, 보행 주파수
대역만 필터링한 뒤, 20초 안의 6개 10초 window에서 jerk와 ACF
대칭성을 계산했습니다. 이를 median과 IQR로 강건하게 요약하고,
subject 단위 교차검증의 조합 성능으로 최종 3개 피처를 선택한 뒤,
상관분석과 VIF 검사로 세 피처의 중복성과 다중공선성을 확인했습니다.

## 2. 산술식

### 전처리

```text
sampling frequency: fs = 100 Hz
band-pass: 4th-order Butterworth, 0.6~3.0 Hz
20-second measurement: 2,000 samples
10-second window: 1,000 samples
window starts: 0, 200, 400, 600, 800, 1,000 samples
```

### 정규화 자기상관함수

```text
R(k) = [1 / (N-k)] * Σ(x[t] - mean(x)) * (x[t+k] - mean(x))
ACF(k) = R(k) / R(0)
```

```text
Ad2 = stride-lag ACF peak, lag range 0.8~1.7 seconds
Ad1 = step-lag ACF peak, approximately half of stride lag
ACF step/stride symmetry = Ad1 / Ad2
```

### Stride frequency

```text
stride_frequency_hz = 1 / stride_lag_seconds
```

### Spectral entropy

```text
p[i] = PSD[i] / ΣPSD
entropy = -Σ p[i] * log2(p[i]) / log2(number_of_bins)
```

### Jerk RMS

```text
jerk[t] = (vertical_acc[t+1] - vertical_acc[t]) * fs
jerk_rms = sqrt(mean(jerk[t]^2))
```

### 20초 요약

```text
values = [f(window_1), ..., f(window_6)]
feature_median = median(values)
feature_iqr = percentile75(values) - percentile25(values)
```

## 3. 후보 피처와 당시 단독 AUC

| 후보 피처 | 단독 OOF AUC |
|---|---:|
| `v_jerk_rms_median` | 0.7217 |
| `v_jerk_rms_iqr` | 0.6227 |
| `v_stride_freq_hz_median` | 0.5995 |
| `ap_spec_entropy_median` | 0.5979 |
| `ap_acf_step_stride_symmetry_median` | 0.5536 |
| `v_acf_step_stride_symmetry_iqr` | 0.5280 |
| `v_stride_freq_hz_iqr` | 0.4996 |
| `ap_spec_entropy_iqr` | 0.4943 |
| `ap_acf_step_stride_symmetry_iqr` | 0.4648 |
| `v_acf_step_stride_symmetry_median` | 0.4530 |

## 4. 조합 비교

| 조합 | OOF AUC |
|---|---:|
| jerk median + jerk IQR + V ACF symmetry IQR | **0.7550** |
| 위 조합 + stride frequency median | 0.7531 |
| jerk median + jerk IQR | 0.7527 |
| 전체 10개 | 0.7510 |
| jerk median + jerk IQR + entropy median | 0.7474 |

## 5. 상관분석과 다중공선성

### 최종 3개 Spearman 상관계수

|  | Jerk median | Jerk IQR | ACF symmetry IQR |
|---|---:|---:|---:|
| Jerk median | 1.0000 | 0.6756 | -0.6209 |
| Jerk IQR | 0.6756 | 1.0000 | -0.0421 |
| ACF symmetry IQR | -0.6209 | -0.0421 | 1.0000 |

### 최종 3개 VIF

| 피처 | VIF |
|---|---:|
| `v_jerk_rms_median` | 2.17 |
| `v_jerk_rms_iqr` | 1.54 |
| `v_acf_step_stride_symmetry_iqr` | 1.59 |

상관분석은 두 피처가 거의 같은 순위 정보를 갖는지 확인하는 절차이고,
VIF는 한 피처가 나머지 피처들의 선형결합으로 얼마나 설명되는지
확인하는 절차다. 상관계수 하나만으로 피처를 제거하지 않았으며,
조합 AUC와 VIF를 함께 보고 최종 피처 수를 결정했다.

## 6. 발표에서 정확히 표현해야 할 부분

1. 피처의 생체역학적 개념은 선행연구에 기반하지만,
   `10초 window/2초 step/median·IQR` 조합은 서비스 조건에 맞춘
   프로젝트의 공학적 확장이다.
2. `v_acf_step_stride_symmetry_iqr`은 Fourier harmonic ratio가 아니다.
   ACF의 Ad1/Ad2 비율에 대한 IQR이다.
3. 최초 피처 선택은 subject-grouped CV를 이용했지만, 선택 과정 전체가
   완전 nested CV였던 것은 아니다. 따라서 당시 0.7550은 개발 단계의
   조합 선택 점수로 표현한다.
4. 후속 100회 완전 nested 피처선택 감사에서는 더 보수적인 AUC 0.780이
   나왔고 피처 선택의 불안정성도 확인됐다. 이는 라벨과 집계 조건이
   초기 선택 실험과 달라 직접적인 동일 수치 비교는 아니다.
5. 수정 라벨로 최종 3개를 고정해 반복 평가한 보수적 결과는
   AUC 약 0.80, 민감도 약 0.80 수준이다.
6. 현재 배포 artifact 자체는 과거 `s3 velocity < 1.0 m/s` 조건을 포함한
   라벨로 학습돼 있다. 수정 라벨 비교 결과와 배포 artifact를 동일한
   모델이라고 설명하면 안 된다.
7. 개발 데이터에서는 여러 20초 segment를 subject 수준으로 다시
   요약한 반면 서비스는 20초 한 번을 평가한다. 따라서 서비스 유효성은
   신규 20초 측정 데이터에서 별도로 검증해야 한다.

## 7. 예상 질문 답변

### 왜 jerk 피처를 두 개 사용했습니까?

Jerk median은 대표적인 움직임 크기를 보고, jerk IQR은 측정 중 변동성을
본다. 두 피처의 Spearman 상관은 0.6756으로 중간 정도였고 VIF는 각각
2.17과 1.54여서 완전히 중복되는 변수는 아니었다.

### ACF 대칭성 IQR은 단독 AUC가 낮은데 왜 선택했습니까?

단독 분류력보다 jerk 피처와 결합했을 때의 보완 정보가 중요했다.
Jerk 두 개만 사용한 AUC는 0.7527이었고, ACF 대칭성 IQR을 추가하면
0.7550으로 후보 조합 중 가장 높았다. 다만 증가폭이 작으므로 외부
검증이 필요한 후보라는 점을 함께 인정한다.

### 왜 전체 10개를 사용하지 않았습니까?

전체 10개의 AUC는 0.7510으로 3개 조합보다 낮았고, 피처 수가 증가하면
71명 규모에서 계수 불안정성과 과적합 위험이 커진다. 성능이 비슷하면
더 단순하고 해석 가능한 모델을 선택했다.

### 데이터 누수를 어떻게 방지했습니까?

동일 subject의 segment가 train과 test에 동시에 포함되지 않도록
subject 단위로 분할했고, 결측치 대체와 RobustScaler 학습은 training
fold 안에서 수행했다. 다만 최초 피처 조합 선택은 완전 nested가
아니었으므로, 이를 완전한 무편향 외부검증이라고 표현하지 않는다.

### 높은 jerk가 항상 보행저하를 의미합니까?

아니다. Jerk는 충격성과 움직임 활력을 함께 반영한다. 조심스럽고 느린
보행에서는 jerk가 낮아질 수도 있으므로 방향을 일반적인 임상 법칙으로
해석하지 않고, 본 개발 데이터에서 학습된 다변량 패턴으로 해석한다.
