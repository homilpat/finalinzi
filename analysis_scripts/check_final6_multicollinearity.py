from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_outputs" / "final6_multicollinearity_check"
if str(ROOT / "analysis_scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_scripts"))

from model_all_domains_to_service_reference import (  # noqa: E402
    FEATURES,
    SUBJECT_TABLE,
    align_to_service,
    fit_service_reference,
)


def vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    x_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(x),
        columns=features,
        index=df.index,
    )
    rows = []
    for feature in features:
        others = [f for f in features if f != feature]
        y = x_imp[feature].to_numpy(float)
        x_other = x_imp[others].to_numpy(float)
        reg = LinearRegression().fit(x_other, y)
        r2 = float(reg.score(x_other, y))
        vif = 1.0 / max(1.0 - r2, 1e-12)
        rows.append({"feature": feature, "r2_explained_by_other_features": r2, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def pairwise_corr(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    corr = df[features].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            rows.append({"feature_a": a, "feature_b": b, "spearman": corr.loc[a, b], "abs_spearman": abs(corr.loc[a, b])})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SUBJECT_TABLE)
    table = table[table["target"].notna()].copy()
    table["target"] = table["target"].astype(int)
    table = table.dropna(subset=FEATURES, how="all").reset_index(drop=True)

    service_med, service_scale, refs = fit_service_reference(table)
    aligned = align_to_service(table, service_med, service_scale, refs)

    for name, df in [("raw", table), ("service_aligned", aligned)]:
        vif = vif_table(df, FEATURES)
        corr = pairwise_corr(df, FEATURES)
        vif.to_csv(OUT_DIR / f"{name}_final6_vif.csv", index=False, encoding="utf-8-sig")
        corr.to_csv(OUT_DIR / f"{name}_final6_spearman_pairs.csv", index=False, encoding="utf-8-sig")
        print(f"\n{name} VIF")
        print(vif.to_string(index=False))
        print(f"\n{name} top Spearman pairs")
        print(corr.head(15).to_string(index=False))

    print("\nwritten", OUT_DIR)


if __name__ == "__main__":
    main()
