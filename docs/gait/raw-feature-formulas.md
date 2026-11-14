# 보행 원피처 5개 계산식

## 공통 기호

| 기호 | 의미 |
|---|---|
| \(N\) | 분석 window의 전체 sample 수 |
| \(n\) | sample index |
| \(k\) | autocorrelation lag |
| \(f_s\) | sampling frequency, 현재 \(100\ \mathrm{Hz}\) |
| \(\Delta t\) | sampling interval, \(1/f_s=0.01\ \mathrm{s}\) |
| \(\tilde v[n]\) | 필터링된 수직축 가속도 |
| \(\widetilde{ap}[n]\) | 필터링된 AP축 가속도 |
| \(Ad1\) | step 주기 부근의 ACF peak |
| \(Ad2\) | stride 주기 부근의 ACF peak |

---

# 원피처 1. 수직축 ACF step/stride 반복 대칭성

기존 코드에서는 `v_harmonic_ratio`라고 명명했지만, 실제 계산값은
Fourier harmonic ratio가 아니다. 수직축 자기상관함수에서 검출한
step peak와 stride peak의 비율이다.

## 1. 수직축 자기상관함수

필터링된 수직 가속도에서 평균을 제거한다.

\[
v'[n]
=
\tilde v[n]
-
\overline{\tilde v}
\]

lag \(k\)에서 비편향 자기상관값을 계산한다.

\[
C_V[k]
=
\frac{1}{N-k}
\sum_{n=0}^{N-k-1}
v'[n]v'[n+k]
\]

자기상관값을 lag 0의 값으로 정규화한다.

\[
R_V[k]
=
\frac{C_V[k]}{C_V[0]}
\]

따라서 다음이 성립한다.

\[
R_V[0]=1
\]

## 2. Stride peak \(Ad2_V\)

stride 주기 탐색 범위는 \(0.8\sim1.7\ \mathrm{s}\)이다. \(f_s=100\)이면
lag 범위는 80~170 samples이다.

\[
k_{\mathrm{stride},V}
=
\underset{
0.8f_s\leq k\leq1.7f_s
}{
\operatorname{arg\,peak}
}
R_V[k]
\]

\[
T_{\mathrm{stride},V}
=
\frac{k_{\mathrm{stride},V}}{f_s}
\]

\[
Ad2_V
=
R_V[k_{\mathrm{stride},V}]
\]

코드에서는 prominence가 0.03 이상인 peak 중 prominence가 가장 큰
peak를 선택한다. 조건을 만족하는 peak가 없으면 탐색 범위 안의
최댓값 위치를 사용한다.

## 3. Step peak \(Ad1_V\)

step 주기는 stride 주기의 약 절반이므로 다음 범위에서 탐색한다.

\[
0.6
\left(
\frac{T_{\mathrm{stride},V}}{2}
\right)
\leq
T_{\mathrm{step},V}
\leq
1.4
\left(
\frac{T_{\mathrm{stride},V}}{2}
\right)
\]

이를 정리하면 다음과 같다.

\[
0.3T_{\mathrm{stride},V}
\leq
T_{\mathrm{step},V}
\leq
0.7T_{\mathrm{stride},V}
\]

해당 범위에서 검출한 ACF peak를 \(Ad1_V\)로 정의한다.

\[
Ad1_V
=
R_V[k_{\mathrm{step},V}]
\]

## 4. 최종 계산식

\[
\boxed{
F_1
=
\text{V ACF step/stride symmetry}
=
\frac{Ad1_V}{Ad2_V}
}
\]

| 항목 | 내용 |
|---|---|
| 현재 코드명 | `v_acf_step_stride_symmetry` |
| 구버전 코드명 | `v_harmonic_ratio` |
| 단위 | 무차원 비율 |
| 의미 | 수직축에서 step 반복성과 stride 반복성의 상대적 관계 |

이 값은 왼발과 오른발을 직접 검출하여 비교한 임상적 좌우 대칭성 지표가
아니다. 몸통 가속도에서 나타나는 step 주기와 stride 주기의 반복 관계를
나타내는 ACF 기반 비율이다.

---

# 원피처 2. AP축 ACF step/stride 반복 대칭성

원피처 1과 같은 계산을 AP축 가속도에 적용한다.

## 1. AP축 자기상관함수

\[
ap'[n]
=
\widetilde{ap}[n]
-
\overline{\widetilde{ap}}
\]

\[
C_{AP}[k]
=
\frac{1}{N-k}
\sum_{n=0}^{N-k-1}
ap'[n]ap'[n+k]
\]

\[
R_{AP}[k]
=
\frac{C_{AP}[k]}{C_{AP}[0]}
\]

\[
R_{AP}[0]=1
\]

## 2. Stride peak \(Ad2_{AP}\)

AP축에서도 \(0.8\sim1.7\ \mathrm{s}\) 범위에서 stride peak를 찾는다.

\[
k_{\mathrm{stride},AP}
=
\underset{
0.8f_s\leq k\leq1.7f_s
}{
\operatorname{arg\,peak}
}
R_{AP}[k]
\]

\[
T_{\mathrm{stride},AP}
=
\frac{k_{\mathrm{stride},AP}}{f_s}
\]

\[
Ad2_{AP}
=
R_{AP}[k_{\mathrm{stride},AP}]
\]

## 3. Step peak \(Ad1_{AP}\)

stride 주기의 절반 부근에서 step peak를 찾는다.

\[
0.3T_{\mathrm{stride},AP}
\leq
T_{\mathrm{step},AP}
\leq
0.7T_{\mathrm{stride},AP}
\]

\[
Ad1_{AP}
=
R_{AP}[k_{\mathrm{step},AP}]
\]

## 4. 최종 계산식

\[
\boxed{
F_2
=
\text{AP ACF step/stride symmetry}
=
\frac{Ad1_{AP}}{Ad2_{AP}}
}
\]

| 항목 | 내용 |
|---|---|
| 현재 코드명 | `ap_acf_step_stride_symmetry` |
| 구버전 코드명 | `ap_harmonic_ratio` |
| 단위 | 무차원 비율 |
| 의미 | AP축에서 step 반복성과 stride 반복성의 상대적 관계 |

수직축 피처와 마찬가지로 왼발과 오른발을 직접 검출한 값이 아니라,
step 주기와 stride 주기의 반복 관계를 나타낸다.

---

# 원피처 3. 수직축 stride frequency

원피처 1에서 검출한 수직축 stride lag를 사용한다.

## 1. Stride 주기

\[
T_{\mathrm{stride},V}
=
\frac{k_{\mathrm{stride},V}}{f_s}
\]

## 2. Stride frequency

Stride frequency는 stride 주기의 역수이다.

\[
\boxed{
F_3
=
f_{\mathrm{stride},V}
=
\frac{1}{T_{\mathrm{stride},V}}
=
\frac{f_s}{k_{\mathrm{stride},V}}
}
\]

## 3. 계산 예시

\(f_s=100\ \mathrm{Hz}\)이고 stride peak가 110번째 lag에서 검출된 경우:

\[
T_{\mathrm{stride},V}
=
\frac{110}{100}
=
1.1\ \mathrm{s}
\]

\[
f_{\mathrm{stride},V}
=
\frac{1}{1.1}
\approx
0.909\ \mathrm{Hz}
\]

| 항목 | 내용 |
|---|---|
| 코드명 | `v_stride_freq_hz` |
| 단위 | Hz |
| 의미 | 초당 stride 수 |

---

# 원피처 4. AP spectral entropy

필터링된 AP축 가속도에 Hann window를 적용한 spectrogram을 계산한다.

## 1. Spectrogram

\[
S(f,t)
=
\left|
\operatorname{STFT}
\left\{
\widetilde{ap}[n]
\right\}
\right|^2
\]

코드 설정은 다음과 같다.

- Window 함수: Hann
- Window 길이: 4초
- \(f_s=100\ \mathrm{Hz}\) 기준 400 samples
- 분석 주파수 대역: \(0.6\sim3.0\ \mathrm{Hz}\)

## 2. 분석 대역 추출

\[
\mathcal{B}
=
\left\{
S(f,t)
\mid
0.6\leq f\leq3.0
\right\}
\]

분석 대역에 포함된 모든 주파수·시간 power를 \(S_i\)로 나타낸다.

## 3. Power 정규화

\[
p_i
=
\frac{S_i}
{\displaystyle\sum_{j=1}^{M}S_j}
\]

여기서 \(M\)은 분석 대역에 포함된 전체 time-frequency bin의 수이다.

## 4. 정규화 Shannon entropy

\[
\boxed{
F_4
=
H_{AP}
=
-
\frac{
\displaystyle\sum_{i=1}^{M}p_i\log_2(p_i)
}{
\log_2(M)
}
}
\]

| 항목 | 내용 |
|---|---|
| 코드명 | `ap_spec_entropy` |
| 단위 | 무차원, 일반적으로 0~1 |
| 낮은 값 | Power가 일부 주파수·시간 bin에 집중된 신호 |
| 높은 값 | Power가 여러 주파수·시간 bin에 분산된 복잡한 신호 |

높은 entropy를 항상 보행저하로 해석해서는 안 된다. 보행속도, 회전,
환경 변화 등의 영향도 받을 수 있으므로 모델 안에서 다른 피처와 함께
해석해야 한다.

---

# 원피처 5. 수직축 jerk RMS

Jerk는 가속도의 시간 변화율이다. 필터링된 수직 가속도의 1차
차분으로 근사한다.

## 1. 수직축 jerk

\[
j_V[n]
=
\frac{
\tilde v[n+1]-\tilde v[n]
}{
\Delta t
}
\]

\[
\Delta t
=
\frac{1}{f_s}
\]

따라서 다음과 같이 표현할 수 있다.

\[
j_V[n]
=
f_s
\left(
\tilde v[n+1]-\tilde v[n]
\right)
\]

\(f_s=100\ \mathrm{Hz}\)이면:

\[
j_V[n]
=
100
\left(
\tilde v[n+1]-\tilde v[n]
\right)
\]

## 2. Jerk RMS

\[
\boxed{
F_5
=
J_{\mathrm{RMS},V}
=
\sqrt{
\frac{1}{N-1}
\sum_{n=0}^{N-2}
j_V[n]^2
}
}
\]

코드의 차분식으로 직접 나타내면 다음과 같다.

\[
\boxed{
F_5
=
f_s
\sqrt{
\frac{1}{N-1}
\sum_{n=0}^{N-2}
\left(
\tilde v[n+1]-\tilde v[n]
\right)^2
}
}
\]

| 항목 | 내용 |
|---|---|
| 코드명 | `v_jerk_rms` |
| 가속도 단위가 \(g\)일 때 | \(g/\mathrm{s}\) |
| 가속도 단위가 \(\mathrm{m/s^2}\)일 때 | \(\mathrm{m/s^3}\) |
| 의미 | 수직 가속도가 시간에 따라 얼마나 급격하게 변하는지 나타내는 RMS |

값이 클수록 수직 가속도가 시간에 따라 더 급격하게 변했다는 뜻이다.
다만 보행속도와 가속도 진폭의 영향도 받으므로, 값의 높고 낮음을
독립적으로 정상·저하로 단정해서는 안 된다.

---

# 5개 원피처 요약

| 번호 | 원피처 | 핵심 계산 | 단위 |
|---:|---|---|---|
| \(F_1\) | V ACF step/stride symmetry | \(Ad1_V/Ad2_V\) | 무차원 |
| \(F_2\) | AP ACF step/stride symmetry | \(Ad1_{AP}/Ad2_{AP}\) | 무차원 |
| \(F_3\) | V stride frequency | \(f_s/k_{\mathrm{stride},V}\) | Hz |
| \(F_4\) | AP spectral entropy | 정규화 Shannon entropy | 무차원 |
| \(F_5\) | V jerk RMS | 수직 가속도 시간차분의 RMS | \(g/\mathrm{s}\) |

각 10초 window에서 위 5개 원피처를 계산한다. 이후 20초 측정 안의
6개 overlapping window 값에 대해 median과 IQR을 각각 계산하여
최종 10개 후보 피처를 생성한다.
