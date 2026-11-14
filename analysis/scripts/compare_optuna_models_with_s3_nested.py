"""With-s3 counterpart of the 4-model nested Optuna comparison.

Identical pipeline to ``compare_optuna_models_no_s3_nested`` but the subject
labels include the s3 (serial-subtraction dual-task) gait-speed < 1.0 m/s
criterion again, i.e. the original clinical target:

    TUG >= 12 OR FSST >= 15 OR BERG < 52 OR DGI <= 19
    OR base(velocity) < 1.0 OR s3(velocity) < 1.0

The with-s3 subject feature table (same 3 features, s3-included target) is read
from the original ``final_training_conditions_nested_model_comparison_100rep``
run, and every artefact (confusion matrices, ROC curves, summary/paired CSVs)
is written to a separate ``with_s3_optuna_4model_nested_100rep`` directory so the
no-s3 outputs are never overwritten.
"""
from __future__ import annotations

import compare_optuna_models_no_s3_nested as base

# With-s3 subject table: same 3 features, target uses the s3 velocity criterion.
base.TABLE_PATH = (
    base.ROOT
    / "analysis_outputs"
    / "final_training_conditions_nested_model_comparison_100rep"
    / "subject_feature_table.csv"
)
base.OUT_DIR = base.ROOT / "analysis_outputs" / "with_s3_optuna_4model_nested_100rep"


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    arguments = base.parse_args()
    base.run(arguments.repeats, arguments.trials)
