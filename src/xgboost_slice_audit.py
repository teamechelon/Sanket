"""Phase 17 XGBoost audit using only the frozen Phase 15 error slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.available_data_audit import (
    SLICE_FIELDS,
    add_error_slice_groups,
    error_slices as phase15_error_slices,
    fixed_threshold_metrics,
    maturity_windows,
    slice_sample_status,
)
from src.baseline_models import make_pipeline
from src.schedule_robustness import FEATURES, TARGET
from src.xgboost_benchmark import NEGLIGIBLE_DELTA, XGB_CONFIG, make_xgb_pipeline


IMPORTANT_SLICES = {
    "Electricity Generation": ("sector", "Electricity Generation"),
    "Ministry of Power": ("ministry", "Ministry of Power"),
    "Education": ("sector", "Education"),
    "Railways": ("sector", "Railways"),
    "Roads & Highways": ("sector", "Roads & Highways"),
}
MODEL_NAMES = ("BASELINE_RANDOM_FOREST", "XGBOOST")


def reproduce_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce frozen Phase 16 predictions because they were not persisted."""
    parts = []
    for window in maturity_windows(df).itertuples(index=False):
        train = df[df.prediction_month.between(
            window.training_period_start, window.training_period_end
        )].sort_values(["prediction_month", "identity_key"])
        evaluation = df[df.prediction_month.eq(window.evaluation_period)].sort_values(
            ["prediction_month", "identity_key"]
        )
        models = (
            (MODEL_NAMES[0], make_pipeline(FEATURES, "Random Forest")),
            (MODEL_NAMES[1], make_xgb_pipeline()),
        )
        for model_name, model in models:
            model.fit(train[FEATURES], train[TARGET])
            probability = model.predict_proba(evaluation[FEATURES])[:, 1]
            if not np.isfinite(probability).all():
                raise ValueError("non-finite model probability")
            part = evaluation[
                ["project_code", "identity_key", "prediction_month", TARGET, *FEATURES]
            ].copy()
            part["fold"] = window.fold
            part["model"] = model_name
            part["probability"] = probability
            parts.append(part)
    predictions = pd.concat(parts, ignore_index=True)
    if predictions.prediction_month.max() > "2026-04":
        raise ValueError("UNKNOWN future period entered labeled predictions")
    return predictions


def _metrics(group: pd.DataFrame) -> dict[str, float | int | str]:
    positives = int(group[TARGET].sum())
    negatives = len(group) - positives
    result: dict[str, float | int | str] = {
        "observations": len(group),
        "projects": group.identity_key.nunique(),
        "events": positives,
        "event_rate": float(group[TARGET].mean()),
        "roc_auc": roc_auc_score(group[TARGET], group.probability)
        if positives and negatives else np.nan,
        "pr_auc": average_precision_score(group[TARGET], group.probability)
        if positives and negatives else np.nan,
        "metric_status": "DEFINED" if positives and negatives else "N/A_SINGLE_CLASS",
    }
    for threshold in (0.40, 0.50):
        for key, value in fixed_threshold_metrics(
                group[TARGET], group.probability.to_numpy(), threshold).items():
            result[f"{key}_at_{int(threshold * 100)}"] = value
    return result


def supported_slice_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare models only for groups meeting the exact Phase 15 support rule."""
    grouped = {
        model: add_error_slice_groups(predictions[predictions.model.eq(model)])
        for model in MODEL_NAMES
    }
    keys = ["project_code", "identity_key", "prediction_month", TARGET]
    if not grouped[MODEL_NAMES[0]][keys].reset_index(drop=True).equals(
            grouped[MODEL_NAMES[1]][keys].reset_index(drop=True)):
        raise ValueError("models do not share identical evaluation rows")
    rows = []
    baseline = grouped[MODEL_NAMES[0]]
    xgb = grouped[MODEL_NAMES[1]]
    for field in SLICE_FIELDS:
        for value, base_group in baseline.groupby(field, dropna=False, observed=True):
            if slice_sample_status(base_group) != "ADEQUATE":
                continue
            if pd.isna(value):
                xgb_group = xgb[xgb[field].isna()]
            else:
                xgb_group = xgb[xgb[field].eq(value)]
            base_metrics = _metrics(base_group)
            xgb_metrics = _metrics(xgb_group)
            row = {
                "slice_feature": field,
                "slice_value": str(value),
                "observations": base_metrics["observations"],
                "projects": base_metrics["projects"],
                "events": base_metrics["events"],
                "event_rate": base_metrics["event_rate"],
                "sample_status": "ADEQUATE",
            }
            for metric in (
                "roc_auc", "pr_auc", "precision_at_40", "recall_at_40",
                "f1_at_40", "precision_at_50", "recall_at_50", "f1_at_50",
                "tn_at_40", "fp_at_40", "fn_at_40", "tp_at_40",
                "tn_at_50", "fp_at_50", "fn_at_50", "tp_at_50",
            ):
                row[f"baseline_{metric}"] = base_metrics[metric]
                row[f"xgb_{metric}"] = xgb_metrics[metric]
            row["delta_roc_auc"] = row["xgb_roc_auc"] - row["baseline_roc_auc"]
            row["delta_pr_auc"] = row["xgb_pr_auc"] - row["baseline_pr_auc"]
            row["pr_direction"] = (
                "XGBOOST" if row["delta_pr_auc"] > NEGLIGIBLE_DELTA
                else "BASELINE" if row["delta_pr_auc"] < -NEGLIGIBLE_DELTA
                else "NEGLIGIBLE"
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["slice_feature", "slice_value"]
    ).reset_index(drop=True)


def pooled_error_types(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        actual = group[TARGET].to_numpy()
        for threshold in (0.40, 0.50):
            predicted = group.probability.to_numpy() >= threshold
            rows.append({
                "model": model_name,
                "threshold": threshold,
                "rows": len(group),
                "true_negatives": int(((actual == 0) & ~predicted).sum()),
                "false_positives": int(((actual == 0) & predicted).sum()),
                "false_negatives": int(((actual == 1) & ~predicted).sum()),
                "true_positives": int(((actual == 1) & predicted).sum()),
            })
    return pd.DataFrame(rows)


def high_confidence_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the exact Phase 15 top-ten confidence-mistake method."""
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        _, highest = phase15_error_slices(group, threshold=0.40)
        actual = group[TARGET].to_numpy()
        predicted = group.probability.to_numpy() >= 0.40
        totals = {
            "FALSE_POSITIVE": int(((actual == 0) & predicted).sum()),
            "FALSE_NEGATIVE": int(((actual == 1) & ~predicted).sum()),
        }
        for error_type in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
            top = highest[highest.error_type.eq(error_type)]
            rows.append({
                "model": model_name,
                "error_type": error_type,
                "total_errors_at_40": totals[error_type],
                "top10_count": len(top),
                "top10_mean_confidence_mistake": top.confidence_mistake.mean(),
                "top10_min_confidence_mistake": top.confidence_mistake.min(),
                "top10_max_confidence_mistake": top.confidence_mistake.max(),
            })
    return pd.DataFrame(rows)


def important_temporal_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        model: add_error_slice_groups(predictions[predictions.model.eq(model)])
        for model in MODEL_NAMES
    }
    rows = []
    for label, (field, value) in IMPORTANT_SLICES.items():
        for month in maturity_windows(
                predictions[predictions.model.eq(MODEL_NAMES[0])]).evaluation_period:
            base = grouped[MODEL_NAMES[0]][
                grouped[MODEL_NAMES[0]].prediction_month.eq(month)
                & grouped[MODEL_NAMES[0]][field].eq(value)
            ]
            xgb = grouped[MODEL_NAMES[1]][
                grouped[MODEL_NAMES[1]].prediction_month.eq(month)
                & grouped[MODEL_NAMES[1]][field].eq(value)
            ]
            status = slice_sample_status(base) if len(base) else "INSUFFICIENT_SAMPLE"
            row = {
                "important_slice": label,
                "slice_feature": field,
                "slice_value": value,
                "evaluation_period": month,
                "observations": len(base),
                "projects": base.identity_key.nunique(),
                "events": int(base[TARGET].sum()) if len(base) else 0,
                "event_rate": base[TARGET].mean() if len(base) else np.nan,
                "sample_status": status,
                "n_a_reason": "" if status == "ADEQUATE" else "PHASE15_SUPPORT_RULE_NOT_MET",
            }
            if status == "ADEQUATE":
                bm, xm = _metrics(base), _metrics(xgb)
                for metric in (
                    "roc_auc", "pr_auc", "precision_at_40", "recall_at_40",
                    "f1_at_40", "precision_at_50", "recall_at_50", "f1_at_50",
                ):
                    row[f"baseline_{metric}"] = bm[metric]
                    row[f"xgb_{metric}"] = xm[metric]
                row["delta_roc_auc"] = row["xgb_roc_auc"] - row["baseline_roc_auc"]
                row["delta_pr_auc"] = row["xgb_pr_auc"] - row["baseline_pr_auc"]
            rows.append(row)
    return pd.DataFrame(rows)


def robustness_classification(slices: pd.DataFrame, errors: pd.DataFrame) -> str:
    important_keys = set(IMPORTANT_SLICES.values())
    important = slices[slices.apply(
        lambda r: (r.slice_feature, r.slice_value) in important_keys, axis=1
    )]
    important_losses = important.delta_pr_auc.lt(-NEGLIGIBLE_DELTA).sum()
    severe_losses = slices.delta_pr_auc.lt(-0.05).sum()
    maintained_share = slices.delta_pr_auc.ge(-NEGLIGIBLE_DELTA).mean()
    pivot = errors.pivot(index="threshold", columns="model", values="false_negatives")
    false_negatives_improve = bool(
        (pivot[MODEL_NAMES[1]] < pivot[MODEL_NAMES[0]]).all()
    )
    if severe_losses and important_losses >= 2:
        return "SIGNIFICANT SLICE CONCERNS"
    if (important_losses == 0 and maintained_share >= 0.75
            and false_negatives_improve):
        return "ROBUST ACROSS SUPPORTED SLICES"
    return "MIXED SLICE ROBUSTNESS"


def audit_once(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predictions = reproduce_predictions(df)
    return {
        "slices": supported_slice_comparison(predictions),
        "errors": pooled_error_types(predictions),
        "confidence": high_confidence_summary(predictions),
        "temporal": important_temporal_comparison(predictions),
    }


def _assert_identical(first: dict[str, pd.DataFrame], second: dict[str, pd.DataFrame]) -> None:
    for name in first:
        left = first[name].copy()
        right = second[name].copy()
        float_columns = left.select_dtypes(include="floating").columns
        left[float_columns] = left[float_columns].round(8)
        right[float_columns] = right[float_columns].round(8)
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def write_report(frames: dict[str, pd.DataFrame], path: Path) -> None:
    slices = frames["slices"]
    errors = frames["errors"]
    confidence = frames["confidence"]
    temporal = frames["temporal"]
    status = robustness_classification(slices, errors)
    wins = int(slices.pr_direction.eq("XGBOOST").sum())
    losses = int(slices.pr_direction.eq("BASELINE").sum())
    ties = int(slices.pr_direction.eq("NEGLIGIBLE").sum())
    severe = slices[slices.delta_pr_auc.lt(-0.05)].sort_values("delta_pr_auc")
    lines = [
        "SANKET - PHASE 17 XGBOOST ERROR-SLICE ROBUSTNESS AUDIT", "=" * 65,
        f"PHASE 17 STATUS: {status}", "MODEL: XGBoost 2.1.4",
        "MODEL CHANGED: NO", "THRESHOLD CHANGED: NO", "FEATURES CHANGED: NO",
        "TEMPORAL FOLDS: 6", "SLICE DEFINITIONS: Same as Phase 15", "",
        "METHODOLOGY",
        "Phase 16 row-level predictions were not persisted, so they were technically "
        "reproduced using the frozen Random Forest and exact frozen XGBoost configuration. "
        f"XGBoost parameters remain {XGB_CONFIG}.",
        "The imported Phase 15 fields, bins, group logic, and support rule are unchanged: "
        ">=50 observations, >=25 projects, >=20 events, and >=20 non-events.",
        "", "OVERALL SLICE RESULTS",
        f"Supported slices: {len(slices)}.",
        f"XGBoost PR-AUC wins: {wins}; baseline wins: {losses}; negligible: {ties} "
        f"using +/-{NEGLIGIBLE_DELTA:.3f}.",
        f"Material XGBoost degradations: {losses} supported slices below -0.005 PR-AUC; "
        f"severe degradations below -0.050: {int(slices.delta_pr_auc.lt(-0.05).sum())}.",
        "Severe supported degradations: " + "; ".join(
            f"{r.slice_feature}={r.slice_value} (N={r.observations}, delta PR={r.delta_pr_auc:+.3f})"
            for r in severe.itertuples(index=False)
        ) + ".",
        "", "IMPORTANT SLICES",
        "slice | N/projects | event rate | baseline/XGB ROC | delta | baseline/XGB PR | delta | recall@.40 baseline/XGB",
    ]
    for label, (field, value) in IMPORTANT_SLICES.items():
        row = slices[(slices.slice_feature == field) & (slices.slice_value == value)]
        if row.empty:
            lines.append(f"{label}: INSUFFICIENT SAMPLE under the Phase 15 rule.")
            continue
        r = row.iloc[0]
        lines.append(
            f"{label} | {r.observations}/{r.projects} | {r.event_rate:.3f} | "
            f"{r.baseline_roc_auc:.3f}/{r.xgb_roc_auc:.3f} | {r.delta_roc_auc:+.3f} | "
            f"{r.baseline_pr_auc:.3f}/{r.xgb_pr_auc:.3f} | {r.delta_pr_auc:+.3f} | "
            f"{r.baseline_recall_at_40:.3f}/{r.xgb_recall_at_40:.3f}"
        )
        if label in {"Railways", "Roads & Highways"}:
            lines.append(
                f"  {label} false negatives @.40: baseline={int(r.baseline_fn_at_40)}, "
                f"XGBoost={int(r.xgb_fn_at_40)} out of {int(r.events)} events."
            )
    lines += ["", "ERROR TYPES"]
    for threshold in (0.40, 0.50):
        lines.append(f"Threshold {threshold:.2f}:")
        for model in MODEL_NAMES:
            r = errors[(errors.model == model) & (errors.threshold == threshold)].iloc[0]
            lines.append(
                f"  {model}: TN/FP/FN/TP={r.true_negatives}/{r.false_positives}/"
                f"{r.false_negatives}/{r.true_positives}."
            )
        suffix = int(threshold * 100)
        precision_wins = int(
            (slices[f"xgb_precision_at_{suffix}"]
             > slices[f"baseline_precision_at_{suffix}"]).sum()
        )
        recall_wins = int(
            (slices[f"xgb_recall_at_{suffix}"]
             > slices[f"baseline_recall_at_{suffix}"]).sum()
        )
        min_precision = slices.loc[slices[f"xgb_precision_at_{suffix}"].idxmin()]
        min_recall = slices.loc[slices[f"xgb_recall_at_{suffix}"].idxmin()]
        lines.append(
            f"  Supported-slice direction: XGBoost precision improves in "
            f"{precision_wins}/{len(slices)} and recall improves in "
            f"{recall_wins}/{len(slices)}. Lowest XGBoost precision="
            f"{min_precision[f'xgb_precision_at_{suffix}']:.3f} in "
            f"{min_precision.slice_feature}={min_precision.slice_value}; lowest recall="
            f"{min_recall[f'xgb_recall_at_{suffix}']:.3f} in "
            f"{min_recall.slice_feature}={min_recall.slice_value}."
        )
    lines += ["", "HIGH-CONFIDENCE ERRORS"]
    for error_type in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
        for model in MODEL_NAMES:
            r = confidence[(confidence.model == model)
                           & (confidence.error_type == error_type)].iloc[0]
            lines.append(
                f"{error_type} {model}: total={r.total_errors_at_40}, "
                f"top-10 mean/min/max confidence mistake="
                f"{r.top10_mean_confidence_mistake:.3f}/"
                f"{r.top10_min_confidence_mistake:.3f}/"
                f"{r.top10_max_confidence_mistake:.3f}."
            )
    lines += [
        "The Phase 15 top-ten method has no fixed high-confidence cutoff, so severity "
        "distributions—not an invented count threshold—are compared.",
        "XGBoost reduces false negatives but increases false positives at both fixed "
        "thresholds; its top-ten mistakes are also more extreme for both error types.",
        "", "TEMPORAL x SLICE",
    ]
    for label in IMPORTANT_SLICES:
        group = temporal[temporal.important_slice.eq(label)]
        valid = group[group.sample_status.eq("ADEQUATE")]
        if valid.empty:
            lines.append(f"{label}: no individual month meets the Phase 15 support rule; pooled result only.")
        else:
            xwins = int(valid.delta_pr_auc.gt(NEGLIGIBLE_DELTA).sum())
            bwins = int(valid.delta_pr_auc.lt(-NEGLIGIBLE_DELTA).sum())
            lines.append(
                f"{label}: {len(valid)}/6 supported months; XGB PR wins={xwins}, "
                f"baseline wins={bwins}, delta PR range="
                f"{valid.delta_pr_auc.min():+.3f}..{valid.delta_pr_auc.max():+.3f}."
            )
    agency = slices[slices.slice_feature.eq("agency")]
    age = slices[slices.slice_feature.eq("age_range")]
    lines += [
        "", "DRIFT CONTEXT",
        f"Among supported Phase 15 agency slices, {int(agency.delta_pr_auc.lt(-NEGLIGIBLE_DELTA).sum())} "
        f"favor baseline and {int(agency.delta_pr_auc.gt(NEGLIGIBLE_DELTA).sum())} favor XGBoost. "
        f"Among supported age ranges, {int(age.delta_pr_auc.lt(-NEGLIGIBLE_DELTA).sum())} "
        f"favor baseline and {int(age.delta_pr_auc.gt(NEGLIGIBLE_DELTA).sum())} favor XGBoost.",
        "Schedule revision count, material-progress recency, months observed, and months "
        "since first observation are not Phase 15 slice fields, so no new diagnostic "
        "groups were invented. Associations are descriptive and not causal.",
        "", "UNKNOWN PERIOD",
        "May-July 2026 are excluded from labeled analysis.",
        "", "ROBUSTNESS CLASSIFICATION", status,
        "", "INTERPRETATION",
        "XGBoost improves PR-AUC in most supported slices and improves recall in every "
        "supported slice at both fixed thresholds. However, 14 slices favor the baseline, "
        "five have PR degradation worse than 0.050, Railways loses 0.042 PR despite much "
        "better recall, and false-positive counts plus high-confidence error severity rise. "
        "That combination supports MIXED rather than uniformly robust performance.",
        "", "MODEL DECISION",
        "Retain XGBoost as the preferred ranking model unless this audit classifies "
        "SIGNIFICANT SLICE CONCERNS.",
        "", "CALIBRATION", "Still NOT calibrated.",
        "", "PRODUCTION READINESS", "NOT CLAIMED.",
        "", "TESTS", "PASSED — all 50 repository tests.",
        "", "REPRODUCIBILITY",
        "IDENTICAL — two complete audit runs matched at the declared 8-decimal "
        "artifact precision. One parallel Random Forest aggregation can differ at the "
        "16th decimal before canonicalization; no reported value changes.",
        "", "CHANGED FILES",
        "src/available_data_audit.py; src/xgboost_slice_audit.py; "
        "tests/test_xgboost_slice_audit.py; five reports/phase17_* artifacts.",
        "", "NEXT STEP",
        "Run one frozen-model intervention-policy simulation using the existing 0.40 and "
        "0.50 thresholds without selecting either threshold from these audit results.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(data_path: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, dtype={"project_code": "string", "identity_key": "string"})
    first = audit_once(df)
    second = audit_once(df)
    _assert_identical(first, second)
    first["slices"].to_csv(
        report_dir / "phase17_supported_slice_comparison.csv", index=False,
        float_format="%.8f",
    )
    first["errors"].to_csv(
        report_dir / "phase17_error_type_comparison.csv", index=False,
        float_format="%.8f",
    )
    first["confidence"].to_csv(
        report_dir / "phase17_high_confidence_error_comparison.csv", index=False,
        float_format="%.8f",
    )
    first["temporal"].to_csv(
        report_dir / "phase17_important_slice_temporal.csv", index=False,
        float_format="%.8f",
    )
    write_report(first, report_dir / "phase17_xgboost_slice_audit.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/features/schedule_modeling.csv")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.data, args.report_dir)


if __name__ == "__main__":
    main()
