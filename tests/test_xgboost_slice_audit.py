"""Phase 17 slice-definition and diagnostic safeguards."""

import unittest

import numpy as np
import pandas as pd

from src.available_data_audit import (
    SLICE_FIELDS,
    TARGET,
    add_error_slice_groups,
    slice_sample_status,
)
from src.xgboost_slice_audit import (
    MODEL_NAMES,
    high_confidence_summary,
    pooled_error_types,
    robustness_classification,
)


class XGBoostSliceAuditTest(unittest.TestCase):
    def test_exact_phase15_slice_fields(self):
        self.assertEqual(SLICE_FIELDS, (
            "prediction_month", "progress_range", "roads_group", "sector",
            "ministry", "agency", "age_range", "cost_ratio_range",
        ))

    def test_exact_phase15_boundaries(self):
        frame = pd.DataFrame({
            "progress_current": [25, 50, 75, 76],
            "sector": ["Roads & Highways", "A", "A", "A"],
            "project_age_months": [36, 84, 144, 145],
            "expenditure_to_original_cost": [.25, .5, .75, 1.01],
        })
        grouped = add_error_slice_groups(frame)
        self.assertEqual(grouped.progress_range.astype(str).tolist(),
                         ["<=25", "25-50", "50-75", ">75"])
        self.assertEqual(grouped.age_range.astype(str).tolist(),
                         ["<=3y", "3-7y", "7-12y", ">12y"])

    def test_exact_phase15_support_rule(self):
        adequate = pd.DataFrame({
            TARGET: [1] * 20 + [0] * 30,
            "identity_key": [f"P:{i}" for i in range(50)],
        })
        self.assertEqual(slice_sample_status(adequate), "ADEQUATE")
        self.assertEqual(
            slice_sample_status(adequate.iloc[:49]), "INSUFFICIENT_SAMPLE"
        )

    def test_pooled_errors_use_only_fixed_thresholds(self):
        rows = []
        for model in MODEL_NAMES:
            for actual, probability in zip([0, 0, 1, 1], [.2, .6, .4, .8]):
                rows.append({"model": model, TARGET: actual, "probability": probability})
        result = pooled_error_types(pd.DataFrame(rows))
        self.assertEqual(sorted(result.threshold.unique().tolist()), [.4, .5])
        at_50 = result[result.threshold.eq(.5)]
        self.assertTrue((at_50.false_positives == 1).all())
        self.assertTrue((at_50.false_negatives == 1).all())

    def test_classification_requires_important_and_error_robustness(self):
        slices = pd.DataFrame({
            "slice_feature": ["sector"] * 5,
            "slice_value": [
                "Electricity Generation", "Education", "Railways",
                "Roads & Highways", "Other",
            ],
            "delta_pr_auc": [.01, .01, .01, .01, -.001],
        })
        errors = pd.DataFrame({
            "threshold": [.4, .4, .5, .5],
            "model": [MODEL_NAMES[0], MODEL_NAMES[1], MODEL_NAMES[0], MODEL_NAMES[1]],
            "false_negatives": [100, 80, 120, 90],
        })
        self.assertEqual(
            robustness_classification(slices, errors),
            "ROBUST ACROSS SUPPORTED SLICES",
        )


if __name__ == "__main__":
    unittest.main()
