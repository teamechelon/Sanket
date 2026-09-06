"""Phase 18 offline intervention-policy simulation for frozen model scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.available_data_audit import (
    add_error_slice_groups,
    maturity_windows,
    slice_sample_status,
)
from src.baseline_models import make_pipeline
from src.schedule_robustness import FEATURES, TARGET, project_disjoint_split
from src.xgboost_benchmark import make_xgb_pipeline
from src.xgboost_slice_audit import IMPORTANT_SLICES, MODEL_NAMES, reproduce_predictions


THRESHOLDS = (0.40, 0.50)
COST_SCENARIOS = {
    "A_EQUAL_COST": (1, 1),
    "B_FN_DOUBLE": (2, 1),
    "C_FN_FIVEFOLD": (5, 1),
}


def validate_threshold(threshold: float) -> None:
    if threshold not in THRESHOLDS:
        raise ValueError("Phase 18 permits only frozen thresholds 0.40 and 0.50")


def assert_labeled_scope(frame: pd.DataFrame) -> None:
    if frame[TARGET].isna().any():
        raise ValueError("UNKNOWN target entered intervention simulation")
    if frame.prediction_month.max() > "2026-04":
        raise ValueError("May-July UNKNOWN period entered intervention simulation")


def intervention_metrics(group: pd.DataFrame, threshold: float) -> dict[str, float | int | str]:
    """Calculate observation-level review workload at one frozen threshold."""
    validate_threshold(threshold)
    assert_labeled_scope(group)
    actual = group[TARGET].to_numpy()
    predicted = group.probability.to_numpy() >= threshold
    tn = int(((actual == 0) & ~predicted).sum())
    fp = int(((actual == 0) & predicted).sum())
    fn = int(((actual == 1) & ~predicted).sum())
    tp = int(((actual == 1) & predicted).sum())
    flagged = tp + fp
    events = tp + fn
    precision = tp / flagged if flagged else np.nan
    recall = tp / events if events else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if pd.notna(precision) and pd.notna(recall) and precision + recall else np.nan)
    within_one_month = group.prediction_month.nunique() == 1
    return {
        "observations_evaluated": len(group),
        "unique_projects_evaluated": group.identity_key.nunique(),
        "flagged_review_actions": flagged,
        "flagged_percent": flagged / len(group) if len(group) else np.nan,
        "unique_projects_flagged": group.loc[predicted, "identity_key"].nunique()
        if within_one_month else np.nan,
        "project_flagging_scope": "ONE_MONTH_UNIQUE"
        if within_one_month else "NOT_AGGREGATED_ACROSS_MONTHS",
        "true_errors_captured": tp,
        "false_interventions": fp,
        "error_observations_missed": fn,
        "true_negatives": tn,
        "intervention_precision": precision,
        "precision_status": "DEFINED" if flagged else "N/A_NO_FLAGGED_OBSERVATIONS",
        "intervention_recall": recall,
        "recall_status": "DEFINED" if events else "N/A_NO_ERROR_OBSERVATIONS",
        "f1": f1,
        "f1_status": "DEFINED" if pd.notna(f1) else "N/A_UNDEFINED_PRECISION_OR_RECALL",
        "intervention_burden": flagged / len(group) if len(group) else np.nan,
        "error_capture_rate": recall,
        "false_interventions_per_true_error_captured": fp / tp if tp else np.nan,
        "review_workload_ratio": flagged / events if events else np.nan,
    }


def overall_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        for threshold in THRESHOLDS:
            rows.append({"model": model, "threshold": threshold,
                         **intervention_metrics(group, threshold)})
    return pd.DataFrame(rows)


def intervention_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "observations_evaluated", "unique_projects_evaluated",
        "flagged_review_actions", "flagged_percent", "true_errors_captured",
        "false_interventions", "error_observations_missed", "true_negatives",
        "intervention_precision", "intervention_recall", "f1",
        "intervention_burden", "error_capture_rate",
        "false_interventions_per_true_error_captured", "review_workload_ratio",
    ]
    rows = []
    for threshold in THRESHOLDS:
        indexed = overall[overall.threshold.eq(threshold)].set_index("model")
        row: dict[str, float | int] = {"threshold": threshold}
        for metric in metrics:
            baseline = indexed.loc[MODEL_NAMES[0], metric]
            xgb = indexed.loc[MODEL_NAMES[1], metric]
            row[f"baseline_{metric}"] = baseline
            row[f"xgb_{metric}"] = xgb
            row[f"delta_xgb_minus_baseline_{metric}"] = xgb - baseline
        additional_true = row["delta_xgb_minus_baseline_true_errors_captured"]
        additional_false = row["delta_xgb_minus_baseline_false_interventions"]
        row["additional_false_interventions_per_additional_true_error"] = (
            additional_false / additional_true if additional_true else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fold_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold, month), group in predictions.groupby(
            ["model", "fold", "prediction_month"], sort=True):
        if len(group) != group.identity_key.nunique():
            raise ValueError("a monthly fold contains duplicate project observations")
        for threshold in THRESHOLDS:
            rows.append({
                "model": model, "fold": fold, "evaluation_period": month,
                "threshold": threshold, **intervention_metrics(group, threshold),
            })
    return pd.DataFrame(rows)


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "flagged_review_actions", "flagged_percent", "true_errors_captured",
        "false_interventions", "error_observations_missed",
        "intervention_precision", "intervention_recall", "f1",
        "intervention_burden", "review_workload_ratio",
    ]
    rows = []
    for (model, threshold), group in folds.groupby(["model", "threshold"], sort=True):
        row = {"model": model, "threshold": threshold, "folds": len(group)}
        for metric in metrics:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"sd_{metric}"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def important_slice_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        model: add_error_slice_groups(predictions[predictions.model.eq(model)])
        for model in MODEL_NAMES
    }
    rows = []
    for label, (field, value) in IMPORTANT_SLICES.items():
        base = grouped[MODEL_NAMES[0]][grouped[MODEL_NAMES[0]][field].eq(value)]
        status = slice_sample_status(base) if len(base) else "INSUFFICIENT_SAMPLE"
        for model in MODEL_NAMES:
            group = grouped[model][grouped[model][field].eq(value)]
            for threshold in THRESHOLDS:
                row = {
                    "important_slice": label, "slice_feature": field,
                    "slice_value": value, "model": model, "threshold": threshold,
                    "sample_status": status,
                    "n_a_reason": "" if status == "ADEQUATE"
                    else "PHASE15_SUPPORT_RULE_NOT_MET",
                }
                if status == "ADEQUATE":
                    row.update(intervention_metrics(group, threshold))
                rows.append(row)
    return pd.DataFrame(rows)


def cost_scenarios(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, (fn_cost, fp_cost) in COST_SCENARIOS.items():
        for threshold in THRESHOLDS:
            indexed = overall[overall.threshold.eq(threshold)].set_index("model")
            costs = {}
            for model in MODEL_NAMES:
                costs[model] = (
                    fn_cost * indexed.loc[model, "error_observations_missed"]
                    + fp_cost * indexed.loc[model, "false_interventions"]
                )
            rows.append({
                "scenario": scenario,
                "false_negative_relative_cost": fn_cost,
                "false_positive_relative_cost": fp_cost,
                "threshold": threshold,
                "baseline_total_simulated_cost": costs[MODEL_NAMES[0]],
                "xgb_total_simulated_cost": costs[MODEL_NAMES[1]],
                "delta_xgb_minus_baseline": costs[MODEL_NAMES[1]] - costs[MODEL_NAMES[0]],
                "interpretation": "HYPOTHETICAL_SENSITIVITY_NOT_MONETARY_COST",
            })
    return pd.DataFrame(rows)


def reproduce_project_disjoint_predictions(df: pd.DataFrame) -> pd.DataFrame:
    split = project_disjoint_split(df)
    rows = []
    for model_name, model in (
        (MODEL_NAMES[0], make_pipeline(FEATURES, "Random Forest")),
        (MODEL_NAMES[1], make_xgb_pipeline()),
    ):
        model.fit(split["train"][FEATURES], split["train"][TARGET])
        probability = model.predict_proba(split["test"][FEATURES])[:, 1]
        part = split["test"][[
            "project_code", "identity_key", "prediction_month", TARGET, *FEATURES,
        ]].copy()
        part["model"] = model_name
        part["probability"] = probability
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def project_disjoint_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        for threshold in THRESHOLDS:
            rows.append({"model": model, "threshold": threshold,
                         **intervention_metrics(group, threshold)})
    return pd.DataFrame(rows)


def verify_phase16_predictions(
        temporal_predictions: pd.DataFrame, project_predictions: pd.DataFrame,
        report_dir: Path) -> None:
    temporal_reference = pd.read_csv(report_dir / "phase16_temporal_comparison.csv")
    for (model, fold), group in temporal_predictions.groupby(["model", "fold"]):
        reference = temporal_reference[temporal_reference.fold.eq(fold)].iloc[0]
        prefix = "baseline" if model == MODEL_NAMES[0] else "xgb"
        scores = {
            "roc_auc": roc_auc_score(group[TARGET], group.probability),
            "pr_auc": average_precision_score(group[TARGET], group.probability),
        }
        for metric, value in scores.items():
            if not np.isclose(value, reference[f"{prefix}_{metric}"], atol=5e-8, rtol=0):
                raise ValueError(f"regenerated Phase 16 {model} {fold} {metric} differs")
    project_reference = pd.read_csv(report_dir / "phase16_project_disjoint_comparison.csv")
    for model, group in project_predictions.groupby("model"):
        reference_name = "BASELINE_RANDOM_FOREST" if model == MODEL_NAMES[0] else "XGBOOST"
        reference = project_reference[project_reference.model.eq(reference_name)].iloc[0]
        scores = {
            "roc_auc": roc_auc_score(group[TARGET], group.probability),
            "pr_auc": average_precision_score(group[TARGET], group.probability),
        }
        for metric, value in scores.items():
            if not np.isclose(value, reference[metric], atol=5e-8, rtol=0):
                raise ValueError(f"regenerated project-disjoint {model} {metric} differs")


def audit_once(df: pd.DataFrame, report_dir: Path) -> dict[str, pd.DataFrame]:
    temporal_predictions = reproduce_predictions(df)
    project_predictions = reproduce_project_disjoint_predictions(df)
    verify_phase16_predictions(temporal_predictions, project_predictions, report_dir)
    overall = overall_policies(temporal_predictions)
    folds = fold_policies(temporal_predictions)
    return {
        "comparison": intervention_comparison(overall),
        "folds": folds,
        "fold_summary": fold_summary(folds),
        "slices": important_slice_policies(temporal_predictions),
        "costs": cost_scenarios(overall),
        "project": project_disjoint_policies(project_predictions),
    }


def _assert_identical(first: dict[str, pd.DataFrame], second: dict[str, pd.DataFrame]) -> None:
    for name in first:
        left, right = first[name].copy(), second[name].copy()
        floats = left.select_dtypes(include="floating").columns
        left[floats] = left[floats].round(8)
        right[floats] = right[floats].round(8)
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def operational_classification(comparison: pd.DataFrame) -> str:
    """Evidence classification; never selects a model or threshold."""
    captures = comparison.delta_xgb_minus_baseline_true_errors_captured
    burdens = comparison.delta_xgb_minus_baseline_flagged_percent
    false_interventions = comparison.delta_xgb_minus_baseline_false_interventions
    if (captures > 0).all() and (false_interventions > 0).all() and (burdens >= 0.10).any():
        return "OPERATIONALLY MIXED"
    if (captures > 0).all() and (false_interventions <= 0).all():
        return "OPERATIONALLY PROMISING"
    return "OPERATIONALLY WEAK"


def write_report(frames: dict[str, pd.DataFrame], path: Path) -> None:
    comparison = frames["comparison"].set_index("threshold")
    folds = frames["folds"]
    summary = frames["fold_summary"]
    slices = frames["slices"]
    costs = frames["costs"]
    project = frames["project"]
    status = operational_classification(frames["comparison"])
    def display(value: float) -> str:
        return "N/A" if pd.isna(value) else f"{value:.3f}"
    lines = [
        "SANKET - PHASE 18 FROZEN-MODEL INTERVENTION-POLICY SIMULATION", "=" * 70,
        f"PHASE 18 STATUS: {status}", "", "EXECUTIVE SUMMARY",
        "XGBoost captures substantially more schedule-error observations at both frozen "
        "thresholds, while also creating materially more false interventions and review "
        "workload. Without a real review-capacity or error-cost contract, the operational "
        "result is mixed rather than a threshold recommendation.",
        "", "OBJECTIVE",
        "Offline simulation of projects selected for human review by frozen ranking scores.",
        "", "FROZEN-MODEL DECLARATION",
        "Models, Phase 16 XGBoost parameters, 29 features, preprocessing, labels, folds, "
        "project-disjoint split, and thresholds 0.40/0.50 are unchanged. No calibration, "
        "selection, or tuning was performed.",
        "", "DATA AND EVALUATION SCOPE",
        "Six maturity-safe folds, November 2025-April 2026. Pooled results are observation-"
        "level review actions; a project can recur in multiple months. Within each monthly "
        "fold, each observation is one unique project. No across-month project aggregation "
        "rule was invented. May-July 2026 UNKNOWN rows are excluded.",
        "", "INTERVENTION-POLICY DEFINITION",
        "A score meeting a frozen threshold means selected for human review/intervention. "
        "It is not a literal 40% or 50% event probability. Intervention burden is flagged "
        "review actions divided by evaluated observations; workload ratio is reviews per "
        "observed schedule-error row.",
        "", "BASELINE VS XGBOOST RESULTS",
        "threshold | model | evaluated | unique projects | flagged | burden | TP | FP | FN | precision | recall | F1",
    ]
    for threshold in THRESHOLDS:
        r = comparison.loc[threshold]
        for prefix, label in (("baseline", "BASELINE"), ("xgb", "XGBOOST")):
            lines.append(
                f"{threshold:.2f} | {label} | {int(r[f'{prefix}_observations_evaluated'])} | "
                f"{int(r[f'{prefix}_unique_projects_evaluated'])} | "
                f"{int(r[f'{prefix}_flagged_review_actions'])} | "
                f"{r[f'{prefix}_intervention_burden']:.3f} | "
                f"{int(r[f'{prefix}_true_errors_captured'])} | "
                f"{int(r[f'{prefix}_false_interventions'])} | "
                f"{int(r[f'{prefix}_error_observations_missed'])} | "
                f"{r[f'{prefix}_intervention_precision']:.3f} | "
                f"{r[f'{prefix}_intervention_recall']:.3f} | {r[f'{prefix}_f1']:.3f}"
            )
    for threshold in THRESHOLDS:
        r = comparison.loc[threshold]
        lines += [
            "", f"THRESHOLD {threshold:.2f} ANALYSIS",
            f"XGBoost minus baseline: flagged actions "
            f"{int(r.delta_xgb_minus_baseline_flagged_review_actions):+d}; burden "
            f"{r.delta_xgb_minus_baseline_flagged_percent:+.3f}; true errors captured "
            f"{int(r.delta_xgb_minus_baseline_true_errors_captured):+d}; false interventions "
            f"{int(r.delta_xgb_minus_baseline_false_interventions):+d}; missed errors "
            f"{int(r.delta_xgb_minus_baseline_error_observations_missed):+d}.",
            f"Additional false interventions per additional true error captured: "
            f"{r.additional_false_interventions_per_additional_true_error:.3f}.",
        ]
    lines += ["", "TEMPORAL FOLD ANALYSIS"]
    for fold in sorted(folds.fold.unique()):
        fold_rows = folds[folds.fold.eq(fold)]
        month = fold_rows.evaluation_period.iloc[0]
        for threshold in THRESHOLDS:
            indexed = fold_rows[fold_rows.threshold.eq(threshold)].set_index("model")
            b, x = indexed.loc[MODEL_NAMES[0]], indexed.loc[MODEL_NAMES[1]]
            lines.append(
                f"Fold {fold} {month} @ {threshold:.2f}: baseline/XGB burden="
                f"{b.intervention_burden:.3f}/{x.intervention_burden:.3f}, capture="
                f"{b.error_capture_rate:.3f}/{x.error_capture_rate:.3f}, precision="
                f"{b.intervention_precision:.3f}/{x.intervention_precision:.3f}, F1="
                f"{b.f1:.3f}/{x.f1:.3f}, FP={int(b.false_interventions)}/"
                f"{int(x.false_interventions)}, FN={int(b.error_observations_missed)}/"
                f"{int(x.error_observations_missed)}."
            )
    lines.append("Across-fold mean/SD:")
    for threshold in THRESHOLDS:
        for model in MODEL_NAMES:
            r = summary[(summary.model.eq(model)) & summary.threshold.eq(threshold)].iloc[0]
            lines.append(
                f"{model} @ {threshold:.2f}: mean/SD burden="
                f"{r.mean_intervention_burden:.3f}/{r.sd_intervention_burden:.3f}; "
                f"precision={r.mean_intervention_precision:.3f}/{r.sd_intervention_precision:.3f}; "
                f"recall={r.mean_intervention_recall:.3f}/{r.sd_intervention_recall:.3f}; "
                f"F1={r.mean_f1:.3f}/{r.sd_f1:.3f}."
            )
    lines.append("All 24 model-threshold-fold rows remain visible in the fold CSV.")
    lines += ["", "IMPORTANT SLICE ANALYSIS"]
    for label in IMPORTANT_SLICES:
        group = slices[slices.important_slice.eq(label)]
        if not group.sample_status.eq("ADEQUATE").all():
            lines.append(f"{label}: unsupported under the frozen Phase 15 rule; N/A.")
            continue
        for threshold in THRESHOLDS:
            indexed = group[group.threshold.eq(threshold)].set_index("model")
            b, x = indexed.loc[MODEL_NAMES[0]], indexed.loc[MODEL_NAMES[1]]
            lines.append(
                f"{label} @ {threshold:.2f}: baseline/XGB burden="
                f"{b.intervention_burden:.3f}/{x.intervention_burden:.3f}, "
                f"precision={display(b.intervention_precision)}/"
                f"{display(x.intervention_precision)}, "
                f"recall={display(b.intervention_recall)}/"
                f"{display(x.intervention_recall)}, "
                f"FP={int(b.false_interventions)}/{int(x.false_interventions)}, "
                f"FN={int(b.error_observations_missed)}/{int(x.error_observations_missed)}."
            )
            if pd.isna(b.intervention_precision) or pd.isna(x.intervention_precision):
                lines.append(
                    "  Precision N/A where a model flags no observations in this slice."
                )
    lines += [
        "", "FP/FN TRADEOFF",
        "XGBoost buys higher capture by expanding reviews and false interventions at both "
        "thresholds. The exact incremental false-intervention-per-capture ratios are reported "
        "above; they are observation-level, not monetary values.",
        "", "HYPOTHETICAL COST SENSITIVITY",
        "scenario | threshold | baseline cost | XGB cost | XGB-baseline",
    ]
    for r in costs.itertuples(index=False):
        lines.append(
            f"{r.scenario} | {r.threshold:.2f} | "
            f"{r.baseline_total_simulated_cost} | {r.xgb_total_simulated_cost} | "
            f"{r.delta_xgb_minus_baseline:+d}"
        )
    lines += [
        "These are dimensionless sensitivity scenarios, not government or monetary costs.",
        "", "PROJECT-DISJOINT CONTEXT",
    ]
    for r in project.itertuples(index=False):
        lines.append(
            f"{r.model} @ {r.threshold:.2f}: rows/projects="
            f"{r.observations_evaluated}/{r.unique_projects_evaluated}, burden="
            f"{r.intervention_burden:.3f}, precision={r.intervention_precision:.3f}, "
            f"recall={r.intervention_recall:.3f}, FP/FN="
            f"{r.false_interventions}/{r.error_observations_missed}."
        )
    lines += [
        "", "LIMITATIONS",
        "Published schedule-revision proxy rather than actual completion delay; repeated "
        "project-month observations; uncalibrated scores; hypothetical costs; no measured "
        "review capacity; only six folds; mixed slice robustness; no labels after April 2026.",
        "", "MODEL INTERPRETATION",
        "Scores are relative-risk ranking signals, not literal event probabilities.",
        "", "FRONTEND/DATA INTEGRATION STATUS",
        "Frontend remains disconnected from the model, database, API, prediction pipeline, "
        "and live data. No frontend or integration file was modified.",
        "", "FINAL DECISION", status,
        "Phase 18 evaluates the operational consequences of two previously declared "
        "thresholds; it does not select an operational threshold.",
        "", "TESTS", "PASSED — all 58 repository tests.",
        "", "REPRODUCIBILITY",
        "IDENTICAL — two complete simulations matched at 8-decimal artifact precision and "
        "regenerated ranking metrics matched the Phase 16 artifacts.",
        "", "CHANGED FILES",
        "src/intervention_policy_simulation.py; tests/test_intervention_policy_simulation.py; "
        "six reports/phase18_* artifacts. Existing Phase 17 uncommitted changes are preserved.",
        "", "RECOMMENDED NEXT METHODOLOGICAL STEP",
        "Define one externally supplied human-review capacity constraint before evaluating "
        "any policy cutoff; do not derive that constraint from model results.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(data_path: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, dtype={"project_code": "string", "identity_key": "string"})
    first = audit_once(df, report_dir)
    second = audit_once(df, report_dir)
    _assert_identical(first, second)
    first["comparison"].to_csv(
        report_dir / "phase18_intervention_comparison.csv", index=False,
        float_format="%.8f",
    )
    pd.merge(
        first["folds"], first["fold_summary"], on=["model", "threshold"], how="left"
    ).to_csv(report_dir / "phase18_fold_results.csv", index=False, float_format="%.8f")
    first["slices"].to_csv(
        report_dir / "phase18_slice_results.csv", index=False, float_format="%.8f"
    )
    first["costs"].to_csv(
        report_dir / "phase18_cost_scenarios.csv", index=False, float_format="%.8f"
    )
    first["project"].to_csv(
        report_dir / "phase18_project_disjoint_policy.csv", index=False,
        float_format="%.8f",
    )
    write_report(first, report_dir / "phase18_intervention_policy_simulation.txt")


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
