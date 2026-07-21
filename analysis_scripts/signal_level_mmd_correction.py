"""
원시 신호 레벨 도메인 보정 최적화

피처 레벨 보정의 한계(jerk_iqr + delta < 0)를 극복하기 위해
신호 레벨에서 amplitude scale (α) + time warp (τ) 적용 후 피처 추출.

최적 (α, τ)를 MMD 최소화로 탐색 → 모델 artifact에 signal_correction 키로 저장.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
import json

from gait_axis_aligned_core import (
    load_sensor_csv_with_metadata,
    _acc_columns,
    align_to_vmlap,
    resample_array_to_100hz,
    window_features,
    TARGET_FS_HZ,
    _DAILY_WIN20,
    _DAILY_SUB_WIN,
    _DAILY_STEP,
)

# ─── 설정 ────────────────────────────────────────────────────────────────────
OUR_SAMPLE_DIR = ROOT / "보행SAMPLE"

# 정상 파일 목록 (발다침 제외)
NORMAL_FILES = [
    "hazi_gait_calibrated_20s_20260715_155029.csv",
    "hazi_gait_anatomical_14cols_20260715_163129.csv",
    "hazi_gait_anatomical_14cols_20260719_161127_80대_2.csv",
    # 제외: 회전 금지인 줄 알고 조심스럽게 걸어 jm이 인위적으로 낮음 (태스크 편향)
    # "hazi_gait_anatomical_14cols_20260716_083108_30대_직선.csv",
    # "hazi_gait_anatomical_14cols_20260721_124330.csv",
    "hazi_gait_anatomical_14cols_20260721_125400.csv",
    "hazi_gait_anatomical_14cols_20260721_125240.csv",
    "hazi_gait_anatomical_14cols_20260721_130032.csv",
]

PHYSIONET_TABLE = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr" / "subwindow_median_iqr_table.csv"
MODEL_PATH      = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat.joblib"
META_PATH       = ROOT / "MOCA" / "models" / "gait_daily_clinical_3feat_metadata.json"

FEAT_COLS = ["v_jerk_rms_median", "v_jerk_rms_iqr", "v_harmonic_ratio_iqr"]


# ─── CSV → VMLAP 100Hz 배열 ──────────────────────────────────────────────────
def load_vmlap(csv_path: Path) -> np.ndarray:
    df, metadata = load_sensor_csv_with_metadata(str(csv_path))
    acc, already_vmlap, _, _ = _acc_columns(df, metadata)
    t = df["Timestamp_ns"].to_numpy(float)
    duration = (np.nanmax(t) - np.nanmin(t)) / 1e9
    observed_fs = float(len(df) / duration) if duration > 0 else TARGET_FS_HZ
    aligned, _ = align_to_vmlap(acc, already_vmlap=already_vmlap, fs=observed_fs)
    return resample_array_to_100hz(aligned, observed_fs)


# ─── VMLAP 배열 → sub-window 피처 ────────────────────────────────────────────
def features_from_vmlap(vmlap: np.ndarray) -> dict | None:
    sub_feats = []
    n = len(vmlap)
    seg_starts = list(range(0, max(1, n - _DAILY_WIN20 + 1), _DAILY_WIN20 // 2))
    if not seg_starts:
        seg_starts = [0]
    for w0 in seg_starts:
        seg = vmlap[w0 : w0 + _DAILY_WIN20]
        if len(seg) < int(0.5 * _DAILY_WIN20):
            continue
        for s in range(0, max(1, len(seg) - _DAILY_SUB_WIN + 1), _DAILY_STEP):
            sub = seg[s : s + _DAILY_SUB_WIN]
            if len(sub) < int(0.8 * _DAILY_SUB_WIN):
                continue
            try:
                f = window_features(sub)
                sub_feats.append({
                    "v_harmonic_ratio": f.get("v_harmonic_ratio", np.nan),
                    "v_jerk_rms":       f.get("v_jerk_rms",       np.nan),
                })
            except Exception:
                continue
    if len(sub_feats) < 2:
        return None
    arr      = pd.DataFrame(sub_feats)
    hr_vals  = arr["v_harmonic_ratio"].dropna()
    jrk_vals = arr["v_jerk_rms"].dropna()
    if len(hr_vals) < 2 or len(jrk_vals) < 2:
        return None
    return {
        "v_jerk_rms_median":    float(jrk_vals.median()),
        "v_jerk_rms_iqr":       float(jrk_vals.quantile(0.75) - jrk_vals.quantile(0.25)),
        "v_harmonic_ratio_iqr": float(hr_vals.quantile(0.75)  - hr_vals.quantile(0.25)),
    }


# ─── 신호 변환: amplitude scale + time warp ──────────────────────────────────
def transform_signal(vmlap: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """
    alpha: 진폭 배율 (1.5이면 신호 1.5배 → jerk 1.5배)
    tau:   시간 배율 (0.9이면 신호 10% 압축 → 보행 주파수 1/0.9배로 빨라짐,
                      동시에 sample간 diff × 1/tau → jerk × 1/tau)
    결과는 100Hz 그리드에서의 배열 (길이 변함)
    """
    scaled = vmlap * alpha
    n = len(scaled)
    n_warped = max(10, int(round(n * tau)))
    if n_warped == n:
        return scaled
    t_orig = np.linspace(0.0, 1.0, n)
    t_warp = np.linspace(0.0, 1.0, n_warped)
    return np.column_stack([
        np.interp(t_warp, t_orig, scaled[:, i])
        for i in range(scaled.shape[1])
    ])


# ─── RBF 커널 MMD² (median heuristic bandwidth) ──────────────────────────────
def mmd_squared(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    # 표준화: 스케일 차이로 인한 커널 편향 방지
    std = np.std(np.vstack([X, Y]), axis=0)
    std[std < 1e-8] = 1.0
    X = X / std
    Y = Y / std
    Z = np.vstack([X, Y])
    dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
    pos = dists[dists > 0]
    sigma2 = float(np.median(pos)) if len(pos) else 1.0

    def K(A, B):
        d = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-d / (2.0 * sigma2))

    return float(K(X, X).mean() - 2.0 * K(X, Y).mean() + K(Y, Y).mean())


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    # 1. OUR_SAMPLE 정상 VMLAP 로드
    print("OUR_SAMPLE 정상 CSV 로드 (신호 레벨)...")
    vmaps, loaded_names = [], []
    for fn in NORMAL_FILES:
        p = OUR_SAMPLE_DIR / fn
        try:
            v = load_vmlap(p)
            vmaps.append(v)
            loaded_names.append(fn)
            dur = len(v) / TARGET_FS_HZ
            f0 = features_from_vmlap(v)
            if f0:
                print(f"  OK {fn[-20:]} ({dur:.1f}s) "
                      f"jm={f0['v_jerk_rms_median']:.3f} "
                      f"ji={f0['v_jerk_rms_iqr']:.3f} "
                      f"hr={f0['v_harmonic_ratio_iqr']:.3f}")
        except Exception as e:
            print(f"  NG {fn}: {e}")

    if len(vmaps) < 3:
        raise RuntimeError("정상 샘플이 3개 미만입니다.")

    # 2. PhysioNet 정상 피처 로드
    pn_df = pd.read_csv(PHYSIONET_TABLE)
    pn_feat = pn_df[pn_df["target"] == 0][FEAT_COLS].dropna().to_numpy(float)
    print(f"\nPhysioNet 정상 {len(pn_feat)}개")
    print(f"  평균: jm={pn_feat[:,0].mean():.3f}, "
          f"ji={pn_feat[:,1].mean():.3f}, "
          f"hr={pn_feat[:,2].mean():.3f}")

    # 3. 원본 OUR 피처 (α=1, τ=1)
    our_raw = np.array([
        [f[c] for c in FEAT_COLS]
        for v in vmaps
        if (f := features_from_vmlap(v)) is not None
    ])
    mmd_before = mmd_squared(our_raw, pn_feat)
    print(f"\nOUR 원본 평균: jm={our_raw[:,0].mean():.3f}, "
          f"ji={our_raw[:,1].mean():.3f}, "
          f"hr={our_raw[:,2].mean():.3f}")
    print(f"MMD² (보정 전): {mmd_before:.6f}")

    # 4. MMD 목적함수
    def objective(params):
        alpha, tau = float(params[0]), float(params[1])
        if alpha <= 0.1 or tau <= 0.1:
            return 1e9
        feats = []
        for v in vmaps:
            tv = transform_signal(v, alpha, tau)
            f  = features_from_vmlap(tv)
            if f:
                feats.append([f[c] for c in FEAT_COLS])
        if len(feats) < 3:
            return 1e9
        return mmd_squared(np.array(feats), pn_feat)

    # 5. Grid scan: α ∈ [0.8, 2.5], τ ∈ [0.7, 1.3]
    print("\n=== Grid scan (α × τ) ===")
    best_loss, best_params = float("inf"), (1.0, 1.0)
    alpha_grid = np.arange(0.8, 2.6, 0.15)
    tau_grid   = np.arange(0.70, 1.35, 0.05)
    grid_results = []
    for alpha in alpha_grid:
        for tau in tau_grid:
            loss = objective((alpha, tau))
            grid_results.append((alpha, tau, loss))
            if loss < best_loss:
                best_loss, best_params = loss, (float(alpha), float(tau))

    print(f"Grid 최적: α={best_params[0]:.2f}, τ={best_params[1]:.2f}, MMD²={best_loss:.6f}")

    # 상위 5개 출력
    grid_results.sort(key=lambda x: x[2])
    print("상위 5:")
    for a, t, l in grid_results[:5]:
        print(f"  α={a:.2f}, τ={t:.2f} → MMD²={l:.6f}")

    # 6. Nelder-Mead 정밀 최적화
    print("\n=== Nelder-Mead 정밀 최적화 ===")
    res = minimize(
        objective,
        best_params,
        method="Nelder-Mead",
        options={"xatol": 0.002, "fatol": 1e-7, "maxiter": 1000},
    )
    alpha_opt, tau_opt = float(res.x[0]), float(res.x[1])
    mmd_after = float(res.fun)
    print(f"최적 α={alpha_opt:.4f}, τ={tau_opt:.4f}")
    print(f"MMD² {mmd_before:.6f} → {mmd_after:.6f}")

    # 7. 최적 보정 후 피처 확인
    print("\n=== 최적 신호 보정 후 OUR 피처 ===")
    corrected_feats = []
    for i, v in enumerate(vmaps):
        tv = transform_signal(v, alpha_opt, tau_opt)
        f  = features_from_vmlap(tv)
        if f:
            corrected_feats.append([f[c] for c in FEAT_COLS])
            print(f"  {loaded_names[i][-20:]}: "
                  f"jm={f['v_jerk_rms_median']:.3f}, "
                  f"ji={f['v_jerk_rms_iqr']:.3f}, "
                  f"hr={f['v_harmonic_ratio_iqr']:.3f}")

    corr_arr = np.array(corrected_feats)
    print(f"보정 후 평균: jm={corr_arr[:,0].mean():.3f}, "
          f"ji={corr_arr[:,1].mean():.3f}, "
          f"hr={corr_arr[:,2].mean():.3f}")

    # 8. 모델 artifact 저장
    signal_correction = {
        "alpha":       alpha_opt,
        "tau":         tau_opt,
        "description": "signal-level: amplitude×alpha then time-warp×tau before feature extraction",
        "n_our_normals": len(vmaps),
        "mmd2_before": float(mmd_before),
        "mmd2_after":  mmd_after,
    }

    model_data = joblib.load(MODEL_PATH)
    model_data["signal_correction"] = signal_correction
    joblib.dump(model_data, MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["signal_correction"] = signal_correction
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"\n모델 artifact + 메타데이터 저장 완료")
    print(f"  alpha={alpha_opt:.4f}, tau={tau_opt:.4f}")
    print(f"  MMD² 개선: {mmd_before:.6f} → {mmd_after:.6f}")


if __name__ == "__main__":
    main()
