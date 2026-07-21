"""
Harmonic Ratio 피처셋으로 피처 테이블 재생성 + 모델 재학습
새 피처: v_harmonic_ratio, ap_harmonic_ratio, v_stride_freq_hz, ap_spec_entropy
참고: Moe-Nilssen & Helbostad 2004 (step/stride regularity + HR)
"""
from __future__ import annotations
import json, re, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, str(ROOT / "MOCA"))

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from gait_axis_aligned_core import (
    FEATURES, TARGET_FS_HZ, WINDOW_SEC,
    extract_axis_aligned_gait_features,
    extract_best10_from_acc_array,
)

print(f"피처셋: {FEATURES}")

GAIT_PROJECT = next(p for p in ROOT.parent.iterdir() if "보행" in p.name and p.is_dir())
OUT_DIR   = ROOT / "analysis_outputs" / "harmonic_ratio_model"
MODEL_DIR = ROOT / "MOCA" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_OUT = TARGET_FS_HZ


# ── 데이터셋 로더 ─────────────────────────────────────────
def best10(acc, fs, already_vmlap, meta) -> dict | None:
    try:
        ex = extract_best10_from_acc_array(acc, fs, already_vmlap=already_vmlap)
    except Exception as e:
        return None
    w = ex["window"]
    return {
        **meta,
        **ex["features"],
        **{k: v for k, v in ex.get("all_features", {}).items()},
        "quality_score": ex["features"].get("v_harmonic_ratio", np.nan),
        "best10_start_sec": w["start_sec"],
        "best10_end_sec":   w["end_sec"],
    }


def add_label(row, group):
    return {**row, "group": group, "target": 1 if group == "impaired" else 0,
            "group_id": f"{row['dataset']}::{row['subject_id']}"}


def iter_physionet():
    base = GAIT_PROJECT / "physionet_AWS" / "LabWalks"
    if not base.exists():
        print(f"  [skip] PhysioNet not found: {base}")
        return []
    stems = [p.stem for p in sorted(base.glob("*_base.hea"))]
    rows = []
    for stem in stems:
        lines = (base / f"{stem}.hea").read_text(encoding="utf-8").splitlines()
        parts = lines[0].split()
        fs, n, ch = float(parts[2]), int(parts[3]), int(parts[1])
        gains, baselines = [], []
        for line in lines[1:1+ch]:
            m = re.match(r"([0-9.]+)\((-?\d+)\)/", line.split()[2])
            gains.append(float(m.group(1))); baselines.append(float(m.group(2)))
        raw = np.memmap(base / f"{stem}.dat", dtype="<i2", mode="r", shape=(n, ch))
        data = (raw.astype(float) - np.array(baselines)) / np.array(gains)
        group = "normal" if stem.startswith("co") else "impaired"
        feat = best10(data[:, :3], fs, True, {"dataset": "PhysioNet_LabWalks", "subject_id": stem, "source_id": stem})
        if feat:
            rows.append(add_label(feat, group))
    print(f"  PhysioNet: {len(rows)} rows")
    return rows


def iter_uci():
    base = ROOT / "external_data" / "uci_har" / "dataset" / "UCI HAR Dataset"
    if not base.exists():
        print(f"  [skip] UCI HAR not found: {base}"); return []
    rows = []
    for split in ("train", "test"):
        y = np.loadtxt(base / split / f"y_{split}.txt", dtype=int)
        subjects = np.loadtxt(base / split / f"subject_{split}.txt", dtype=int)
        sig = base / split / "Inertial Signals"
        ax = np.loadtxt(sig / f"total_acc_x_{split}.txt")
        ay = np.loadtxt(sig / f"total_acc_y_{split}.txt")
        az = np.loadtxt(sig / f"total_acc_z_{split}.txt")
        for subj in sorted(set(subjects)):
            idx = np.flatnonzero((subjects == subj) & (y == 1))
            if len(idx) < 4: continue
            acc = np.vstack([np.column_stack([ax[i], ay[i], az[i]]) for i in idx[:8]])
            feat = best10(acc, 50.0, False, {"dataset": "UCI_HAR", "subject_id": str(subj), "source_id": f"{split}_{subj}"})
            if feat: rows.append(add_label(feat, "normal"))
    print(f"  UCI_HAR: {len(rows)} rows")
    return rows


def iter_geotec():
    base = ROOT / "external_data" / "geotec_tug_smartphone" / "extracted"
    if not base.exists():
        print(f"  [skip] GEOTEC not found: {base}"); return []
    rows = []
    for path in sorted(base.rglob("*_sp.csv")):
        df = pd.read_csv(path)
        if "label" not in df.columns: continue
        walk = df[df["label"].astype(str).eq("WALKING")]
        if len(walk) < 120 or not {"x_acc","y_acc","z_acc"}.issubset(walk.columns): continue
        t = pd.to_numeric(walk["timestamp"], errors="coerce").to_numpy(float)
        dur = (np.nanmax(t) - np.nanmin(t)) / 1000 if len(t) else np.nan
        fs = float(np.clip(len(walk) / dur if np.isfinite(dur) and dur > 0 else 25.0, 10, 100))
        acc = walk[["x_acc","y_acc","z_acc"]].to_numpy(float) / 9.80665
        m = re.search(r"s\d+", path.name)
        feat = best10(acc, fs, False, {"dataset": "GEOTEC_SP", "subject_id": m.group(0) if m else path.stem, "source_id": path.name})
        if feat: rows.append(add_label(feat, "normal"))
    print(f"  GEOTEC_SP: {len(rows)} rows")
    return rows


def iter_chapman():
    zpath = ROOT / "external_data" / "chapman_pd" / "RawWalkingDatabase.zip"
    if not zpath.exists():
        print(f"  [skip] Chapman not found: {zpath}"); return []
    rows = []
    with zipfile.ZipFile(zpath) as zf:
        for name in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
            df = pd.read_csv(zf.open(name), usecols=["accelerometer_x","accelerometer_y","accelerometer_z","class"])
            off = df[df["class"].astype(str).eq("C")]
            if len(off) < int(80 * WINDOW_SEC): continue
            acc = off[["accelerometer_x","accelerometer_y","accelerometer_z"]].to_numpy(float) / 9.80665
            feat = best10(acc, 80.0, False, {"dataset": "Chapman_PD_OFF_RAW", "subject_id": Path(name).stem, "source_id": name})
            if feat: rows.append(add_label(feat, "impaired"))
    print(f"  Chapman_PD: {len(rows)} rows")
    return rows


def iter_fog():
    path = ROOT / "external_data" / "fog_star" / "sensor_data.csv"
    if not path.exists():
        print(f"  [skip] FoG_STAR not found: {path}"); return []
    df = pd.read_csv(path, usecols=["timestamp","back_acc_x","back_acc_y","back_acc_z","activity","subjectID","sessionID","taskID"])
    df = df[df["activity"].eq(1)]
    rows = []
    for keys, part in df.groupby(["subjectID","sessionID","taskID"], sort=True):
        acc = part.sort_values("timestamp")[["back_acc_x","back_acc_y","back_acc_z"]].to_numpy(float)
        if len(acc) < int(60 * WINDOW_SEC): continue
        feat = best10(acc, 60.0, False, {"dataset": "FoG_STAR_BACK_WALK", "subject_id": str(keys[0]), "source_id": str(keys)})
        if feat: rows.append(add_label(feat, "impaired"))
    print(f"  FoG_STAR: {len(rows)} rows")
    return rows


def iter_our_samples():
    sample_dir = next((p for p in ROOT.iterdir() if p.is_dir() and "SAMPLE" in p.name), None)
    if not sample_dir:
        return []
    rows = []
    for path in sorted(sample_dir.glob("*.csv")):
        try:
            ex = extract_axis_aligned_gait_features(str(path))
        except Exception:
            continue
        w = ex["window"]
        group = "impaired" if "발다침" in path.stem else "normal"
        row = {"dataset": "OUR_SAMPLE", "subject_id": path.stem, "source_id": path.name,
               **ex["features"], **ex.get("all_features", {}),
               "best10_start_sec": w["start_sec"], "best10_end_sec": w["end_sec"]}
        rows.append(add_label(row, group))
    print(f"  OUR_SAMPLE: {len(rows)} rows")
    return rows


# ── 피처 테이블 빌드 ─────────────────────────────────────
print("\n[1] 피처 테이블 재생성 중...")
all_rows = []
for fn in [iter_physionet, iter_uci, iter_geotec, iter_chapman, iter_fog, iter_our_samples]:
    all_rows.extend(fn())

table = pd.DataFrame(all_rows)
table["target"] = pd.to_numeric(table["target"], errors="coerce")
table = table.dropna(subset=["target"])

table_path = OUT_DIR / "harmonic_ratio_subject_table.csv"
table.to_csv(table_path, index=False)
print(f"  저장: {table_path}  ({len(table)} rows)")
print(f"  정상={int((table['target']==0).sum())}  저하={int((table['target']==1).sum())}")
print(f"  도메인: {sorted(table['dataset'].unique())}")

# ── 피처값 분포 확인 ─────────────────────────────────────
print("\n[2] 피처 분포 (dataset × target 중앙값)")
for feat in FEATURES:
    if feat not in table.columns:
        print(f"  {feat}: 없음")
        continue
    grp = table.groupby(["dataset","target"])[feat].median().reset_index()
    print(f"\n  {feat}")
    for _, r in grp.iterrows():
        label = "정상" if r["target"] == 0 else "저하"
        print(f"    {r['dataset']:35s} {label}  {r[feat]:.4f}")


# ── 도메인 보정 ───────────────────────────────────────────
def apply_correction(df: pd.DataFrame, domains: set[str]) -> pd.DataFrame:
    df = df.copy()
    ref = df[df["dataset"].eq("PhysioNet_LabWalks") & df["target"].eq(0)][FEATURES].median()
    for ds, part in df.groupby("dataset"):
        if ds not in domains or ds == "PhysioNet_LabWalks": continue
        normals = part[part["target"].eq(0)]
        if len(normals) < 2: continue
        delta = ref - normals[FEATURES].median()
        df.loc[df["dataset"].eq(ds), FEATURES] += delta
    return df


# ── OOF + Youden ─────────────────────────────────────────
def youden_thr(y, p):
    best_j, best_t = -np.inf, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        s  = tp/(tp+fn) if tp+fn else 0.0
        sp = tn/(tn+fp) if tn+fp else 0.0
        if s + sp - 1 > best_j:
            best_j, best_t = s + sp - 1, t
    return float(best_t)


def oof_metrics(train: pd.DataFrame) -> dict:
    y      = train["target"].astype(int).to_numpy()
    groups = (train["dataset"] + "::" + train["subject_id"]).to_numpy()
    oof    = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(train[FEATURES], y, groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        pipe.fit(train.iloc[tr][FEATURES].to_numpy(), y[tr])
        oof[te] = pipe.predict_proba(train.iloc[te][FEATURES].to_numpy())[:, 1]
    thr  = youden_thr(y, oof)
    pred = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {"auc": round(float(roc_auc_score(y, oof)), 4),
            "sens": round(float(tp/(tp+fn)) if tp+fn else 0.0, 4),
            "spec": round(float(tn/(tn+fp)) if tn+fp else 0.0, 4),
            "f1":   round(float(f1_score(y, pred)), 4),
            "thr":  round(float(thr), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def final_train(train: pd.DataFrame):
    y = train["target"].astype(int).to_numpy()
    X = train[FEATURES].to_numpy()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    pipe.fit(X, y)
    prob = pipe.predict_proba(train[FEATURES].to_numpy())[:, 1]
    thr  = youden_thr(y, prob)
    return pipe, float(thr)


# ── 실험 A(전체) / B(보정가능만) ────────────────────────────
CORRECTABLE = {"PhysioNet_LabWalks", "UCI_HAR", "GEOTEC_SP"}
ALL_DOMAINS = set(table["dataset"].unique()) - {"OUR_SAMPLE"}

print("\n[3] 모델 학습")
results = []
for name, domains in [("A_전체", ALL_DOMAINS), ("B_보정가능만", CORRECTABLE)]:
    subset = table[table["dataset"].isin(domains | {"OUR_SAMPLE"})].copy()
    corrected = apply_correction(subset, domains)
    train = corrected[corrected["dataset"].ne("OUR_SAMPLE")].copy()
    train = train.dropna(subset=FEATURES)
    n0, n1 = int((train["target"]==0).sum()), int((train["target"]==1).sum())
    if n0 < 5 or n1 < 5:
        print(f"  [{name}] 데이터 부족 skip"); continue
    m = oof_metrics(train)
    print(f"\n  [{name}] 정상={n0} 저하={n1}  도메인={sorted(domains)}")
    print(f"    OOF  AUC={m['auc']}  sens={m['sens']}  spec={m['spec']}  f1={m['f1']}  thr={m['thr']}")
    print(f"    TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}")

    # 우리 샘플 예측
    pipe, thr = final_train(train)
    our = corrected[corrected["dataset"].eq("OUR_SAMPLE")].dropna(subset=FEATURES)
    if not our.empty:
        prob = pipe.predict_proba(our[FEATURES])[:, 1]
        print(f"    [우리샘플 thr={thr:.3f}]")
        for i, (_, r) in enumerate(our.iterrows()):
            ok = "O" if (prob[i] >= thr) == bool(r["target"]) else "X"
            print(f"      {ok} {str(r['subject_id'])[:50]:50s}  prob={prob[i]:.3f}  pred={int(prob[i]>=thr)}  label={int(r['target'])}")

    results.append({"실험": name, **m, "n_normal": n0, "n_impaired": n1})

    # B 모델 저장
    if name == "B_보정가능만":
        artifact = {
            "pipeline":  pipe,
            "features":  FEATURES,
            "threshold": thr,
            "model_mode": "harmonic_ratio_correctable_domains",
            "excluded_domains": list(ALL_DOMAINS - CORRECTABLE),
            "reference": "Moe-Nilssen & Helbostad 2004 (Harmonic Ratio)",
        }
        model_path = MODEL_DIR / "gait_harmonic_ratio_youden.joblib"
        joblib.dump(artifact, model_path)
        meta_out = {"features": FEATURES, "threshold": thr,
                    "oof": m, "excluded": list(ALL_DOMAINS - CORRECTABLE)}
        (MODEL_DIR / "gait_harmonic_ratio_youden_metadata.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  [저장] {model_path}")

print("\n[완료]")
if results:
    cols = ["실험","n_normal","n_impaired","auc","sens","spec","f1","thr"]
    print(pd.DataFrame(results)[cols].to_string(index=False))
