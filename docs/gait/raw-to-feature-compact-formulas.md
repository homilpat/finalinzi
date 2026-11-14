# Raw 가속도에서 보행 피처를 계산하는 최종 공식

## 1. Raw 데이터와 축 신호

현재 입력 CSV의 해부학적 가속도 열은 다음과 같다.

| 축 | Raw CSV 열 |
|---|---|
| 수직축 | `Acc_Vertical_g` |
| 좌우축 | `Acc_ML_g` |
| 전후축 | `Acc_AP_g` |

Raw CSV가 `Acc_X`, `Acc_Y`, `Acc_Z`로 들어오면 먼저 V/ML/AP 축으로
정렬한다. 이후 \(0.6\sim3.0\ \mathrm{Hz}\) band-pass filter를 적용한다.

\[
v_{\mathrm{acc}}[n]
=
\operatorname{BPF}_{0.6-3.0}
\left(
\mathrm{Acc\_Vertical\_g}[n]
\right)
\]

\[
ap_{\mathrm{acc}}[n]
=
\operatorname{BPF}_{0.6-3.0}
\left(
\mathrm{Acc\_AP\_g}[n]
\right)
\]

Sampling frequency는 \(f_s=100\ \mathrm{Hz}\)이고, 한 window는 10초이므로:

\[
N
=
10\times100
=
1000\ \text{samples}
\]

---

# 2. ACF 계산

수직축과 AP축의 평균 제거 신호를 정의한다.

\[
v_c[n]
=
v_{\mathrm{acc}}[n]
-
\frac{1}{1000}
\sum_{m=0}^{999}v_{\mathrm{acc}}[m]
\]

\[
ap_c[n]
=
ap_{\mathrm{acc}}[n]
-
\frac{1}{1000}
\sum_{m=0}^{999}ap_{\mathrm{acc}}[m]
\]

수직축 정규화 ACF:

\[
\operatorname{ACF}_V(k)
=
\frac{
\displaystyle
\frac{1}{1000-k}
\sum_{n=0}^{999-k}v_c[n]v_c[n+k]
}{
\displaystyle
\frac{1}{1000}
\sum_{n=0}^{999}v_c[n]^2
}
\]

AP축 정규화 ACF:

\[
\operatorname{ACF}_{AP}(k)
=
\frac{
\displaystyle
\frac{1}{1000-k}
\sum_{n=0}^{999-k}ap_c[n]ap_c[n+k]
}{
\displaystyle
\frac{1}{1000}
\sum_{n=0}^{999}ap_c[n]^2
}
\]

Stride peak lag는 80~170 samples 범위에서 찾는다.

\[
k_{\mathrm{stride},V}
=
\underset{80\leq k\leq170}
{\operatorname{arg\,peak}}
\operatorname{ACF}_V(k)
\]

\[
k_{\mathrm{stride},AP}
=
\underset{80\leq k\leq170}
{\operatorname{arg\,peak}}
\operatorname{ACF}_{AP}(k)
\]

Step peak lag는 각 stride lag의 0.3~0.7배 범위에서 찾는다.

\[
k_{\mathrm{step},V}
=
\underset{
0.3k_{\mathrm{stride},V}
\leq k\leq
0.7k_{\mathrm{stride},V}
}
{\operatorname{arg\,peak}}
\operatorname{ACF}_V(k)
\]

\[
k_{\mathrm{step},AP}
=
\underset{
0.3k_{\mathrm{stride},AP}
\leq k\leq
0.7k_{\mathrm{stride},AP}
}
{\operatorname{arg\,peak}}
\operatorname{ACF}_{AP}(k)
\]

---

# 3. 원피처 5개 최종 공식

## 원피처 1. 수직축 ACF step/stride 반복 대칭성

\[
\boxed{
\mathrm{v\_acf\_step\_stride\_symmetry}
=
\frac{
\operatorname{ACF}_V
\left(k_{\mathrm{step},V}\right)
}{
\operatorname{ACF}_V
\left(k_{\mathrm{stride},V}\right)
}
}
\]

---

## 원피처 2. AP축 ACF step/stride 반복 대칭성

\[
\boxed{
\mathrm{ap\_acf\_step\_stride\_symmetry}
=
\frac{
\operatorname{ACF}_{AP}
\left(k_{\mathrm{step},AP}\right)
}{
\operatorname{ACF}_{AP}
\left(k_{\mathrm{stride},AP}\right)
}
}
\]

---

## 원피처 3. 수직축 stride frequency

\[
\boxed{
\mathrm{v\_stride\_freq\_hz}
=
\frac{f_s}{k_{\mathrm{stride},V}}
=
\frac{100}{k_{\mathrm{stride},V}}
}
\]

예를 들어 \(k_{\mathrm{stride},V}=110\)이면:

\[
\mathrm{v\_stride\_freq\_hz}
=
\frac{100}{110}
\approx
0.909\ \mathrm{Hz}
\]

---

## 원피처 4. AP spectral entropy

AP 가속도 \(ap_{\mathrm{acc}}[n]\)의 spectrogram에서
\(0.6\sim3.0\ \mathrm{Hz}\) 대역의 power를 \(S_1,\ldots,S_M\)이라고
정의한다.

\[
p_i
=
\frac{S_i}{
\displaystyle\sum_{j=1}^{M}S_j
}
\]

\[
\boxed{
\mathrm{ap\_spec\_entropy}
=
-
\frac{
\displaystyle\sum_{i=1}^{M}p_i\log_2(p_i)
}{
\log_2(M)
}
}
\]

---

## 원피처 5. 수직축 jerk RMS

수직축 jerk는 다음과 같이 계산한다.

\[
\mathrm{jerk}_V[n]
=
100
\left(
v_{\mathrm{acc}}[n+1]
-
v_{\mathrm{acc}}[n]
\right),
\qquad
n=0,\ldots,998
\]

10초 window에는 1000개 가속도 sample과 999개 jerk 값이 존재한다.

\[
\boxed{
\mathrm{v\_jerk\_rms}
=
\sqrt{
\frac{1}{999}
\sum_{n=0}^{998}
\mathrm{jerk}_V[n]^2
}
}
\]

jerk를 별도로 정의하지 않고 가속도에서 바로 나타내면:

\[
\boxed{
\mathrm{v\_jerk\_rms}
=
100
\sqrt{
\frac{1}{999}
\sum_{n=0}^{998}
\left(
v_{\mathrm{acc}}[n+1]
-
v_{\mathrm{acc}}[n]
\right)^2
}
}
\]

---

# 4. 20초 측정에서 median과 IQR 계산

20초 측정에서 10초 window를 2초 간격으로 생성한다.

\[
\text{window start}
=
\{0,2,4,6,8,10\}\ \mathrm{s}
\]

따라서 총 6개의 10초 window가 생성된다.

\[
W
=
\{W_1,W_2,W_3,W_4,W_5,W_6\}
\]

원피처 \(F\)를 각 window에서 계산한 값을 다음과 같이 정의한다.

\[
\mathbf{F}
=
\left[
F^{(1)},
F^{(2)},
F^{(3)},
F^{(4)},
F^{(5)},
F^{(6)}
\right]
\]

## Median

\[
\boxed{
F_{\mathrm{median}}
=
\operatorname{median}
\left(
F^{(1)},F^{(2)},F^{(3)},
F^{(4)},F^{(5)},F^{(6)}
\right)
}
\]

## IQR

\[
\boxed{
F_{\mathrm{IQR}}
=
Q_{0.75}(\mathbf{F})
-
Q_{0.25}(\mathbf{F})
}
\]

---

# 5. 최종 후보 피처 10개

| 원피처 | Median | IQR |
|---|---|---|
| V ACF step/stride symmetry | `v_acf_step_stride_symmetry_median` | `v_acf_step_stride_symmetry_iqr` |
| AP ACF step/stride symmetry | `ap_acf_step_stride_symmetry_median` | `ap_acf_step_stride_symmetry_iqr` |
| V stride frequency | `v_stride_freq_hz_median` | `v_stride_freq_hz_iqr` |
| AP spectral entropy | `ap_spec_entropy_median` | `ap_spec_entropy_iqr` |
| V jerk RMS | `v_jerk_rms_median` | `v_jerk_rms_iqr` |

즉, 5개 원피처를 각각 20초 측정 단위의 median과 IQR로 요약하여
총 10개의 모델 후보 피처를 생성한다.

---

# 6. 발표용 한 장 공식

\[
\boxed{
\mathrm{v\_acf\_step\_stride\_symmetry}
=
\frac{
\operatorname{ACF}_V(k_{\mathrm{step},V})
}{
\operatorname{ACF}_V(k_{\mathrm{stride},V})
}
}
\]

\[
\boxed{
\mathrm{ap\_acf\_step\_stride\_symmetry}
=
\frac{
\operatorname{ACF}_{AP}(k_{\mathrm{step},AP})
}{
\operatorname{ACF}_{AP}(k_{\mathrm{stride},AP})
}
}
\]

\[
\boxed{
\mathrm{v\_stride\_freq\_hz}
=
\frac{100}{k_{\mathrm{stride},V}}
}
\]

\[
\boxed{
\mathrm{ap\_spec\_entropy}
=
-
\frac{
\sum_{i=1}^{M}p_i\log_2(p_i)
}{
\log_2(M)
}
}
\]

\[
\boxed{
\mathrm{v\_jerk\_rms}
=
100
\sqrt{
\frac{1}{999}
\sum_{n=0}^{998}
\left(
v_{\mathrm{acc}}[n+1]
-
v_{\mathrm{acc}}[n]
\right)^2
}
}
\]

\[
\boxed{
F_{\mathrm{median}}
=
\operatorname{median}
\left(
F^{(1)},\ldots,F^{(6)}
\right)
}
\qquad
\boxed{
F_{\mathrm{IQR}}
=
Q_{0.75}(\mathbf F)
-
Q_{0.25}(\mathbf F)
}
\]
