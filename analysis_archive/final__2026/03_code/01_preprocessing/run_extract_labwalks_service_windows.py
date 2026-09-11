from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = PROJECT_ROOT / "75h_processing_butterworth" / "extract_labwalks_service20_features.py"
BASE_OUT = PROJECT_ROOT / "physionet_AWS" / "strict_preprocessing_runs" / "labwalks_service_window_features"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def summarize(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    return {
        "csv": str(csv_path),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "controls": int(df[df["subject_id"].astype(str).str.startswith("CO")]["subject_id"].nunique())
        if "subject_id" in df.columns
        else 0,
        "fallers": int(df[df["subject_id"].astype(str).str.startswith("FL")]["subject_id"].nunique())
        if "subject_id" in df.columns
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=str, default="10,15,20")
    parser.add_argument("--turn-exclude-margin-sec", type=float, default=4.0)
    parser.add_argument("--out-base", type=Path, default=BASE_OUT)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")

    args.out_base.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for window_sec in [float(x.strip()) for x in args.windows.split(",") if x.strip()]:
        stride_sec = 2.5 if window_sec <= 15 else 5.0
        tag = f"service{int(window_sec)}"
        out_dir = args.out_base / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        run(
            [
                sys.executable,
                str(EXTRACT_SCRIPT),
                "--out-dir",
                str(out_dir),
                "--window-sec",
                str(window_sec),
                "--stride-sec",
                str(stride_sec),
                "--turn-exclude-margin-sec",
                str(args.turn_exclude_margin_sec),
            ],
            env,
        )

        legacy_csv = out_dir / "labwalks_service20_amp_spec_features.csv"
        generic_csv = out_dir / f"labwalks_{tag}_amp_spec_features.csv"
        if legacy_csv.exists():
            summary_source = legacy_csv
            if legacy_csv.resolve() != generic_csv.resolve():
                shutil.copyfile(legacy_csv, generic_csv)
                if generic_csv.exists() and generic_csv.stat().st_size > 0:
                    summary_source = generic_csv
            summaries.append({"window_sec": window_sec, "stride_sec": stride_sec, **summarize(summary_source)})

    summary_df = pd.DataFrame(summaries)
    summary_path = args.out_base / "labwalks_service_window_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\nFeature extraction summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
