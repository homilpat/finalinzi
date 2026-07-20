from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "analysis_outputs" / "service_reference_pattern_feature_screen" / "service_reference_feature_screen.csv"
COMBOS = ROOT / "analysis_outputs" / "service_reference_pattern_feature_screen" / "service_reference_combo_cv.csv"
OUT_DIR = ROOT / "analysis_outputs" / "comprehensive_gait_feature_analysis"


DOMAIN_SENSITIVE_KEYWORDS = [
    "sig_",
    "bandpower",
    "harmonic_ratio",
    "spectral_flatness",
    "jerk_rms",
    "spec_peak_freq",
]


def classify_feature(row: pd.Series) -> str:
    feature = str(row["feature"])
    sample_dist = float(row["sample_to_normal_robust_dist"])
    auc = float(row["separation_auc_directional"])
    if any(k in feature for k in DOMAIN_SENSITIVE_KEYWORDS):
        if sample_dist > 0.75 or auc > 0.80:
            return "domain_sensitive_hold"
    if sample_dist <= 0.50 and auc >= 0.62:
        return "service_candidate"
    if sample_dist <= 1.00 and auc >= 0.60:
        return "secondary_candidate"
    return "hold"


def combo_risk(features: str) -> str:
    parts = [p.strip() for p in features.split("+")]
    sensitive = [p for p in parts if any(k in p for k in DOMAIN_SENSITIVE_KEYWORDS)]
    if "harmonic_ratio" in parts:
        return "high_domain_leakage_risk"
    if sensitive:
        return "domain_sensitive_check_required"
    return "lower_domain_risk"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    screen = pd.read_csv(SCREEN)
    combos = pd.read_csv(COMBOS)

    screen["recommendation"] = screen.apply(classify_feature, axis=1)
    combos["risk_flag"] = combos["features"].apply(combo_risk)

    service = screen[screen["recommendation"].isin(["service_candidate", "secondary_candidate"])].copy()
    service = service.sort_values(["recommendation", "service_usable_score"], ascending=[True, False])

    lower_risk_combos = combos[combos["risk_flag"].eq("lower_domain_risk")].copy()
    lower_risk_combos = lower_risk_combos.sort_values(["auc", "spec", "sens"], ascending=[False, False, False])

    screen.to_csv(OUT_DIR / "all_feature_screen_classified.csv", index=False, encoding="utf-8-sig")
    service.to_csv(OUT_DIR / "recommended_service_feature_candidates.csv", index=False, encoding="utf-8-sig")
    lower_risk_combos.to_csv(OUT_DIR / "lower_domain_risk_combo_candidates.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(OUT_DIR / "all_combo_candidates_with_risk.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# Comprehensive gait feature analysis\n")
    lines.append("## Data actually used\n")
    lines.append("- Normal raw domains: UCI_HAR, GEOTEC_SP")
    lines.append("- Impaired raw domains: Chapman_PD_OFF_RAW, FoG_STAR_BACK_WALK")
    lines.append("- PD_TURNING_IMU excluded from straight-walk model screening")
    lines.append("- Existing fixed_best10 table excluded from raw-only leakage-sensitive screening")
    lines.append("- Unit for model screening: subject-level best-quality 10 s window\n")

    lines.append("## Feature categories\n")
    for rec in ["service_candidate", "secondary_candidate", "domain_sensitive_hold", "hold"]:
        part = screen[screen["recommendation"].eq(rec)].head(12)
        lines.append(f"### {rec}\n")
        if part.empty:
            lines.append("- none\n")
            continue
        for _, row in part.iterrows():
            lines.append(
                f"- {row['feature']}: sample={row['sample_median']:.4g}, "
                f"normal={row['public_normal_median']:.4g}, impaired={row['impaired_median']:.4g}, "
                f"sample-normal dist={row['sample_to_normal_robust_dist']:.3f}, "
                f"AUC={row['separation_auc_directional']:.3f}, direction={row['direction']}"
            )
        lines.append("")

    lines.append("## Lower domain-risk model candidates\n")
    for _, row in lower_risk_combos.head(12).iterrows():
        lines.append(
            f"- {row['features']}: n={int(row['n'])}, AUC={row['auc']:.3f}, "
            f"Acc={row['acc']:.3f}, Sens={row['sens']:.3f}, Spec={row['spec']:.3f}, "
            f"max_corr={row['max_spearman_abs_corr']:.3f}"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "- Very high scores from harmonic_ratio/bandpower/amplitude-like features are suspicious because they can encode sensor/protocol/domain differences."
    )
    lines.append(
        "- The most defensible service features are those close to OUR_SAMPLE normals while still separating impaired rows: ACF stride regularity, ACF stride width, sample entropy, spectral peak ratio/prominence, and step timing."
    )
    lines.append(
        "- Final reporting should use subject-level GroupKFold; random 8:2 is only exploratory because label is strongly coupled to dataset source."
    )
    lines.append(
        "- The current public-domain impaired labels do not perfectly match the service normal samples; app-collected normal and impaired samples are still needed for a final deployable correction."
    )

    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("written", OUT_DIR)
    print("\nrecommended features")
    print(service[["feature", "recommendation", "sample_to_normal_robust_dist", "separation_auc_directional", "direction", "service_usable_score"]].to_string(index=False))
    print("\nlower-risk combos")
    print(lower_risk_combos[["features", "n", "auc", "acc", "sens", "spec", "max_spearman_abs_corr"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
