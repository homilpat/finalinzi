"""
PhysioNet LabWalk .dat 파일에서 pitch-velocity (deg/s) 채널을 읽어
acc와 동일한 sub-window 방식으로 pitch_band_rms 피처 추출.

출력: analysis_outputs/daily_subwindow_median_iqr/subwindow_pitch_table.csv
  subject_id, group, target, pitch_band_rms_median, pitch_band_rms_iqr
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ROOT       = Path(__file__).resolve().parents[1]
LABWALK_DIR = Path(ROOT.parent) / "파이널 보행 프로젝트" / "physionet_AWS" / "LabWalks"
CLINICAL_CSV = next(Path(ROOT.parent).glob("**/clinical_motor_label_modeling/subject_features_with_clinical.csv"))
OUT_DIR    = ROOT / "analysis_outputs" / "daily_subwindow_median_iqr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfilt

TARGET_FS   = 100          # Hz (PhysioNet LabWalks = 100Hz)
WIN20       = int(20 * TARGET_FS)
SUB_WIN     = int(10 * TARGET_FS)
STEP        = int(2  * TARGET_FS)
PITCH_CHAN  = 4            # 0-indexed: channels 0=v-acc 1=ml-acc 2=ap-acc 3=yaw 4=pitch 5=roll


def bandpass(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    sos = butter(4, [0.6, 3.0], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x - np.nanmean(x))


def extract_pitch_band_rms_iqr(pitch_dps: np.ndarray) -> dict | None:
    """pitch-velocity (deg/s) 시계열 → sub-window pitch_band_rms → IQR/median"""
    n = len(pitch_dps)
    seg_starts = list(range(0, max(1, n - WIN20 + 1), WIN20 // 2)) or [0]
    rms_list: list[float] = []
    for w0 in seg_starts:
        seg = pitch_dps[w0 : w0 + WIN20]
        if len(seg) < int(0.5 * WIN20):
            continue
        for s in range(0, max(1, len(seg) - SUB_WIN + 1), STEP):
            sub = seg[s : s + SUB_WIN]
            if len(sub) < int(0.8 * SUB_WIN):
                continue
            bp = bandpass(sub, TARGET_FS)
            rms_list.append(float(np.sqrt(np.nanmean(bp ** 2))))
    if len(rms_list) < 2:
        return None
    q75, q25 = np.percentile(rms_list, [75, 25])
    return {
        "pitch_band_rms_median": float(np.median(rms_list)),
        "pitch_band_rms_iqr":    float(q75 - q25),
        "n_sub_windows":         len(rms_list),
    }


# ── 임상 라벨 로드 ─────────────────────────────────────────────────
clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
clin["target"] = (clin["motor_impairment_score"] >= 0.5).astype(int)
clin_lbl = clin[["subject_id", "target"]].drop_duplicates("subject_id").set_index("subject_id")

# ── LabWalk .dat 파일 목록 ─────────────────────────────────────────
dat_files = sorted(LABWALK_DIR.glob("*.dat"))
print(f"LabWalk .dat 파일: {len(dat_files)}개")

rows = []
for dat_path in dat_files:
    record_name = dat_path.stem              # e.g. "co001_base"
    parts = record_name.split("_")
    subject_id  = parts[0].upper()            # e.g. "CO001"

    if subject_id not in clin_lbl.index:
        continue

    try:
        rec = wfdb.rdrecord(str(dat_path.with_suffix("")))
    except Exception as e:
        print(f"  [skip] {record_name}: {e}")
        continue

    if rec.p_signal is None or rec.p_signal.shape[1] <= PITCH_CHAN:
        print(f"  [skip] {record_name}: pitch channel 없음 (채널수={rec.p_signal.shape[1] if rec.p_signal is not None else 0})")
        continue

    pitch_dps = rec.p_signal[:, PITCH_CHAN].astype(float)  # degrees/s
    res = extract_pitch_band_rms_iqr(pitch_dps)
    if res is None:
        print(f"  [skip] {record_name}: sub-window 부족")
        continue

    row = {
        "subject_id":            subject_id,
        "group":                 clin.loc[clin["subject_id"] == subject_id, "group"].iloc[0] if subject_id in clin["subject_id"].values else "unknown",
        "target":                int(clin_lbl.loc[subject_id, "target"]),
        **res,
    }
    rows.append(row)
    print(f"  {subject_id}  {'impaired' if row['target'] else 'normal ':8s}  "
          f"pitch_rms_med={res['pitch_band_rms_median']:.2f} dps  "
          f"pitch_rms_iqr={res['pitch_band_rms_iqr']:.2f} dps  "
          f"n_win={res['n_sub_windows']}")

# ── 집계 및 저장 ──────────────────────────────────────────────────
df_out = pd.DataFrame(rows)
subj_df = df_out.groupby("subject_id").agg({
    "group":                 "first",
    "target":                "first",
    "pitch_band_rms_median": "median",
    "pitch_band_rms_iqr":    "median",
}).reset_index()

out_path = OUT_DIR / "subwindow_pitch_table.csv"
subj_df.to_csv(out_path, index=False)
print(f"\n저장: {out_path}  ({len(subj_df)}명)")

n0 = (subj_df["target"] == 0).sum(); n1 = (subj_df["target"] == 1).sum()
g0 = subj_df[subj_df["target"] == 0]["pitch_band_rms_median"]
g1 = subj_df[subj_df["target"] == 1]["pitch_band_rms_median"]
print(f"\n정상 {n0}명 / 저하 {n1}명")
print(f"pitch_band_rms_median  정상={g0.mean():.3f}  저하={g1.mean():.3f} deg/s")
g0i = subj_df[subj_df["target"] == 0]["pitch_band_rms_iqr"]
g1i = subj_df[subj_df["target"] == 1]["pitch_band_rms_iqr"]
print(f"pitch_band_rms_iqr     정상={g0i.mean():.3f}  저하={g1i.mean():.3f} deg/s")
