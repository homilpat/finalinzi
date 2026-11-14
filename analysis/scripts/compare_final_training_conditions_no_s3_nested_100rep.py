"""Run the final-condition comparison without an absolute s3-speed cutoff.

The deployed model and the original comparison outputs are left unchanged.
This experiment uses the revised clinical target:
TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19
OR usual/base gait velocity < 1.0 m/s.
"""
from __future__ import annotations

import json

import pandas as pd
from joblib import parallel_backend

import compare_final_training_conditions_nested_100rep as comparison


OUT_DIR = (
    comparison.ROOT
    / "analysis_outputs"
    / "final_training_conditions_no_s3_nested_model_comparison_100rep"
)


def clinical_target_without_s3(clinical: pd.DataFrame) -> pd.Series:
    columns = ["TUG", "FSST", "BERG", "DGI", "base(velocity)"]
    for column in columns:
        clinical[column] = pd.to_numeric(clinical[column], errors="coerce")
    return (
        (clinical["TUG"] >= 12)
        | (clinical["FSST"] >= 15)
        | (clinical["BERG"] < 52)
        | (clinical["DGI"] <= 19)
        | (clinical["base(velocity)"] < 1.0)
    ).astype(int)


original_finalize = comparison.finalize


def finalize_with_label_metadata() -> pd.DataFrame:
    summary = original_finalize()
    metadata_path = OUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "label_rule": (
                "TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19 "
                "OR base(velocity) < 1.0 m/s"
            ),
            "excluded_label_rule": (
                "s3(velocity) < 1.0 m/s is not used as a standalone criterion"
            ),
            "label_revision_reason": (
                "s3 is a serial-subtraction dual-task gait speed and does not "
                "have a validated universal absolute 1.0 m/s impairment cutoff"
            ),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    comparison.OUT_DIR = OUT_DIR
    comparison.CACHE_PATH = OUT_DIR / "subject_all_segments_final3_cache.npz"
    comparison.ASSIGNMENT_PATH = OUT_DIR / "subject_repeat_assignments.csv"
    comparison.clinical_target = clinical_target_without_s3
    comparison.finalize = finalize_with_label_metadata
    with parallel_backend("threading"):
        comparison.main()


if __name__ == "__main__":
    main()
