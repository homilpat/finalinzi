"""Audit whether gyro-detected turning confounds the 40s free-walk ACC top3 model."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts"))

from compare_single_20s_segment_100rep import RAW_DIR
from smoke_compare_freewalk_20s_30s_40s import FS, SEED, evaluate, top3_features


SOURCE_DIR = ROOT / "analysis_outputs" / "smoke_freewalk_20s_30s_40s"
OUT_DIR = ROOT / "analysis_outputs" / "gyro_turn_audit_freewalk40"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TURN_PEAK_DPS = 15.0
TURN_EDGE_DPS = 5.0
MIN_TURN_SEC = 0.50
MAX_TURN_SEC = 10.0
MIN_TURN_ANGLE_DEG = 45.0
TURN_FEATURES = [
    "turn_event_count",
    "turn_ratio",
    "turn_yaw_abs_angle_deg",
    "turn_yaw_net_angle_deg",
    "turn_yaw_peak_rate_dps",
]


def parse_header(subject_id: str) -> dict:
    lines = (RAW_DIR / f"{subject_id}.hea").read_text(encoding="utf-8").splitlines()
    first = lines[0].split()
    n_channels, fs, n_samples = int(first[1]), float(first[2]), int(first[3])
    gains, baselines, units = [], [], []
    for line in lines[1 : 1 + n_channels]:
        parts = line.split()
        match = re.match(r"([-\d.]+)(?:\(([-\d]+)\))?/([^ ]+)", parts[2])
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2) or 0))
        units.append(match.group(3))
    return {
        "n_channels": n_channels,
        "fs": fs,
        "n_samples": n_samples,
        "gains": np.asarray(gains),
        "baselines": np.asarray(baselines),
        "units": units,
    }


def read_interval(subject_id: str, start_sec: float, duration_sec: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    header = parse_header(subject_id)
    start = int(round(start_sec * header["fs"]))
    end = start + int(round(duration_sec * header["fs"]))
    raw = np.memmap(
        RAW_DIR / f"{subject_id}.dat",
        dtype="<i2",
        mode="r",
        shape=(header["n_samples"], header["n_channels"]),
    )
    physical = (
        raw[start:end, :6].astype(float)
        - header["baselines"][:6]
    ) / header["gains"][:6]
    acc = physical[:, :3]
    gyro = physical[:, 3:6]
    sos = butter(4, 20.0 / (0.5 * header["fs"]), btype="low", output="sos")
    return acc, sosfiltfilt(sos, gyro, axis=0)


def event_mask(yaw: np.ndarray) -> tuple[np.ndarray, int]:
    sos = butter(4, 1.5 / (0.5 * FS), btype="low", output="sos")
    filtered = sosfiltfilt(sos, yaw)
    active = np.abs(filtered) >= TURN_EDGE_DPS
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return np.zeros(len(active), dtype=bool), 0

    groups = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > 1:
            groups.append((start, previous + 1))
            start = index
        previous = index
    groups.append((start, previous + 1))

    mask = np.zeros(len(active), dtype=bool)
    count = 0
    for start, end in groups:
        duration = (end - start) / FS
        peak = float(np.max(np.abs(filtered[start:end])))
        angle = float(np.abs(np.sum(filtered[start:end]) / FS))
        if (
            MIN_TURN_SEC <= duration <= MAX_TURN_SEC
            and peak >= TURN_PEAK_DPS
            and angle >= MIN_TURN_ANGLE_DEG
        ):
            mask[start:end] = True
            count += 1
    return mask, count


def turn_metrics(gyro: np.ndarray) -> dict:
    yaw = gyro[:, 0] - np.median(gyro[:, 0])
    mask, count = event_mask(yaw)
    turn_yaw = yaw[mask]
    abs_angle = float(np.sum(np.abs(turn_yaw)) / FS) if mask.any() else 0.0
    net_angle = float(np.abs(np.sum(turn_yaw) / FS)) if mask.any() else 0.0
    peak = float(np.max(np.abs(turn_yaw))) if mask.any() else 0.0

    contaminated = 0
    for start in range(0, len(yaw) - 10 * FS + 1, 2 * FS):
        if mask[start : start + 10 * FS].any():
            contaminated += 1
    n_windows = (len(yaw) - 10 * FS) // (2 * FS) + 1
    return {
        "turn_event_count": count,
        "turn_ratio": float(mask.mean()),
        "turn_yaw_abs_angle_deg": abs_angle,
        "turn_yaw_net_angle_deg": net_angle,
        "turn_yaw_peak_rate_dps": peak,
        "turn_contaminated_10s_windows": contaminated,
        "turn_clean_10s_windows": n_windows - contaminated,
        "turn_contaminated_window_ratio": contaminated / n_windows,
    }


def rank_turn_score(table: pd.DataFrame) -> pd.Series:
    parts = [
        table[name].rank(method="average", pct=True)
        for name in ["turn_ratio", "turn_yaw_abs_angle_deg", "turn_yaw_peak_rate_dps", "turn_event_count"]
    ]
    return pd.concat(parts, axis=1).mean(axis=1)


def group_audit(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [*TURN_FEATURES, "turn_contaminated_window_ratio", "turn_score"]:
        normal = table.loc[table["target"].eq(0), feature].to_numpy(float)
        impaired = table.loc[table["target"].eq(1), feature].to_numpy(float)
        test = mannwhitneyu(impaired, normal, alternative="two-sided")
        rank_biserial = 2.0 * float(test.statistic) / (len(impaired) * len(normal)) - 1.0
        rows.append({
            "feature": feature,
            "normal_median": np.median(normal),
            "impaired_median": np.median(impaired),
            "mannwhitney_p": float(test.pvalue),
            "rank_biserial_impaired_vs_normal": rank_biserial,
        })
    return pd.DataFrame(rows)


def correlation_audit(table: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "v_jerk_rms_median",
        "v_jerk_rms_iqr",
        "ap_spec_entropy_median",
        "probability",
    ]
    rows = []
    for turn_feature in [*TURN_FEATURES, "turn_contaminated_window_ratio", "turn_score"]:
        for outcome in outcomes:
            result = spearmanr(table[turn_feature], table[outcome], nan_policy="omit")
            rows.append({
                "turn_feature": turn_feature,
                "outcome": outcome,
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
            })
    return pd.DataFrame(rows)


def stratified_auc(table: pd.DataFrame) -> pd.DataFrame:
    work = table.copy()
    ordered_score = work["turn_score"].rank(method="first")
    work["turn_tertile"] = pd.qcut(ordered_score, 3, labels=["low", "medium", "high"])
    rows = []
    for level, part in work.groupby("turn_tertile", observed=True):
        rows.append({
            "turn_tertile": str(level),
            "n": len(part),
            "normal_n": int(part["target"].eq(0).sum()),
            "impaired_n": int(part["target"].eq(1).sum()),
            "top3_auc": roc_auc_score(part["target"], part["probability"]),
            "turn_score_median": part["turn_score"].median(),
        })
    return pd.DataFrame(rows)


def presence_auc(table: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("no_validated_turn", table["turn_event_count"].eq(0)),
        ("one_or_more_turns", table["turn_event_count"].gt(0)),
        ("turn_affected_windows_le_50pct", table["turn_contaminated_window_ratio"].le(0.5)),
        ("turn_affected_windows_gt_50pct", table["turn_contaminated_window_ratio"].gt(0.5)),
    ]
    rows = []
    for name, mask in groups:
        part = table.loc[mask]
        rows.append({
            "group": name,
            "n": len(part),
            "normal_n": int(part["target"].eq(0).sum()),
            "impaired_n": int(part["target"].eq(1).sum()),
            "top3_auc": (
                roc_auc_score(part["target"], part["probability"])
                if part["target"].nunique() == 2 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    intervals = pd.read_csv(SOURCE_DIR / "chosen_bouts.csv")
    intervals = intervals[intervals["duration_sec"].eq(40)].copy()
    predictions = pd.read_csv(SOURCE_DIR / "smoke_predictions.csv")
    predictions = predictions[predictions["representation"].eq("freewalk_40s_top3")][
        ["subject_id", "probability", "prediction", "threshold"]
    ]

    rows = []
    for _, interval in intervals.iterrows():
        subject_id = str(interval["subject_id"])
        acc, gyro = read_interval(subject_id, float(interval["interval_start_sec"]))
        top3, _ = top3_features(acc)
        rows.append({
            **interval.to_dict(),
            "v_jerk_rms_median": float(top3[0]),
            "v_jerk_rms_iqr": float(top3[1]),
            "ap_spec_entropy_median": float(top3[2]),
            **turn_metrics(gyro),
        })
    table = pd.DataFrame(rows).merge(predictions, on="subject_id", how="left", validate="one_to_one")
    table["turn_score"] = rank_turn_score(table)
    table.to_csv(OUT_DIR / "turn_metrics_by_subject.csv", index=False, encoding="utf-8-sig")

    group = group_audit(table)
    corr = correlation_audit(table)
    strata = stratified_auc(table)
    presence = presence_auc(table)
    group.to_csv(OUT_DIR / "turn_group_audit.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT_DIR / "turn_correlations.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(OUT_DIR / "turn_stratified_auc.csv", index=False, encoding="utf-8-sig")
    presence.to_csv(OUT_DIR / "turn_presence_auc.csv", index=False, encoding="utf-8-sig")

    y = table["target"].to_numpy(int)
    top3_x = table[["v_jerk_rms_median", "v_jerk_rms_iqr", "ap_spec_entropy_median"]].to_numpy(float)
    turn_x = table[TURN_FEATURES].to_numpy(float)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(top3_x, y))
    model_rows = []
    for name, x in [
        ("acc_top3", top3_x),
        ("gyro_turn_only", turn_x),
        ("acc_top3_plus_turn", np.column_stack([top3_x, turn_x])),
    ]:
        metric, _ = evaluate(name, x, y, splits)
        model_rows.append(metric)
    models = pd.DataFrame(model_rows)
    models.to_csv(OUT_DIR / "turn_model_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot(
        [
            table.loc[table["target"].eq(0), "turn_score"],
            table.loc[table["target"].eq(1), "turn_score"],
        ],
        labels=["Normal", "Impaired"],
    )
    axes[0].set_title("Gyro turn burden by target")
    axes[0].set_ylabel("Rank-based turn score")
    colors = np.where(table["target"].eq(1), "#d95f02", "#1b9e77")
    axes[1].scatter(table["turn_score"], table["probability"], c=colors, alpha=0.8)
    axes[1].set_xlabel("Rank-based turn score")
    axes[1].set_ylabel("ACC top3 probability")
    rho = spearmanr(table["turn_score"], table["probability"]).statistic
    axes[1].set_title(f"Prediction vs turn burden (rho={rho:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "turn_audit.png", dpi=180)
    plt.close(fig)

    print("\n=== Group audit ===")
    print(group.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== Turn-stratified AUC ===")
    print(strata.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== Model comparison ===")
    print(models.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWritten: {OUT_DIR}")


if __name__ == "__main__":
    main()
