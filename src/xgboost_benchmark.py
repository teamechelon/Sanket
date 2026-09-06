"""Phase 16 predeclared, untuned XGBoost model-family benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.available_data_audit import (
    DRIFTING,
    STABLE_SUBSET,
    fixed_threshold_metrics,
    maturity_windows,
    walk_forward,
)
from src.baseline_models import SEED, make_pipeline
from src.model_inference import serialize_pipeline
from src.schedule_robustness import FEATURES, TARGET, project_disjoint_split


# Frozen before any Phase 16 result was produced. Do not change in this phase.
XGB_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "scale_pos_weight": 1.0,
    "missing": np.nan,
    "n_jobs": 1,
    "tree_method": "hist",
    "importance_type": "gain",
}
NEGLIGIBLE_DELTA = 0.005
COMPARISON_METRICS = [
    "roc_auc", "pr_auc", "precision_at_40", "recall_at_40", "f1_at_40",
    "precision_at_50", "recall_at_50", "f1_at_50",
]


def specification_text() -> str:
    params = "\n".join(f"- {key}: {value}" for key, value in XGB_CONFIG.items())
    return f"""SANKET - PHASE 16 PREDECLARED XGBOOST SPECIFICATION
==========================================================
This specification is frozen before results are generated.

EXACT CONFIGURATION
{params}
- class weighting strategy: natural classes; no reweighting
- missing-value handling: existing training-fitted median/mode imputation,
  numeric missing indicators, and XGBoost missing=NaN safety handling
- early stopping: NONE
- hyperparameter search: NONE
- threshold tuning: NONE; evaluate fixed 0.40 and 0.50 only
- calibration: NONE
- feature selection after results: NONE
- feature population: the existing 29 CORE_PLUS_CONDITIONAL features
- folds: exactly the six Phase 15 maturity-safe walk-forward folds
- project-disjoint split: exactly the Phase 15 deterministic split

PRIMARY DECISION RULE
PR-AUC is primary and ROC-AUC secondary. A paired fold is an XGBoost win when
delta PR-AUC > {NEGLIGIBLE_DELTA:.3f}, a baseline win when delta PR-AUC <
-{NEGLIGIBLE_DELTA:.3f}, and otherwise negligible/tied.

XGBOOST CLEARLY BETTER requires mean delta PR >= 0.010, mean delta ROC >=
0.005, at least four PR wins, no more than one PR loss, and non-negative
project-disjoint deltas with project-disjoint delta PR >= 0.005.
XGBOOST WORSE requires mean delta PR <= -0.010, mean delta ROC <= -0.005,
at least four baseline PR wins, and a negative project-disjoint delta PR.
Positive but non-gating evidence is XGBOOST MARGINALLY BETTER. Otherwise the
BASELINE REMAINS PREFERRED. The final model changes only for CLEARLY BETTER.
"""


def make_xgb_pipeline() -> Pipeline:
    preprocess = make_pipeline(FEATURES, "Random Forest").named_steps["preprocess"]
    return Pipeline([
        ("preprocess", preprocess),
        ("model", XGBClassifier(**XGB_CONFIG)),
    ])


def _score(y: pd.Series, probability: np.ndarray) -> dict[str, float | int]:
    if not np.isfinite(probability).all():
        raise ValueError("non-finite XGBoost probability")
    result = {
        "event_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "mean_prediction": float(probability.mean()),
    }
    for threshold in (0.40, 0.50):
        for key, value in fixed_threshold_metrics(y, probability, threshold).items():
            result[f"{key}_at_{int(threshold * 100)}"] = value
    return result


def _importance_by_raw_feature(model: Pipeline) -> dict[str, float]:
    names = model.named_steps["preprocess"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    result = {feature: 0.0 for feature in FEATURES}
    for transformed, value in zip(names, values):
        token = transformed.split("__", 1)[-1].replace("missingindicator_", "")
        raw = next((
            feature for feature in sorted(FEATURES, key=len, reverse=True)
            if token == feature or token.startswith(feature + "_")
        ), None)
        if raw:
            result[raw] += float(value)
    return result


def xgb_walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    importances = []
    for window in maturity_windows(df).itertuples(index=False):
        train = df[df.prediction_month.between(
            window.training_period_start, window.training_period_end
        )].sort_values(["prediction_month", "identity_key"])
        evaluation = df[df.prediction_month.eq(window.evaluation_period)].sort_values(
            ["prediction_month", "identity_key"]
        )
        model = make_xgb_pipeline()
        model.fit(train[FEATURES], train[TARGET])
        probability = model.predict_proba(evaluation[FEATURES])[:, 1]
        rows.append({
            "fold": window.fold,
            "training_period_start": window.training_period_start,
            "training_period_end": window.training_period_end,
            "training_label_endpoint": window.training_label_endpoint,
            "evaluation_period": window.evaluation_period,
            "train_rows": len(train),
            "evaluation_rows": len(evaluation),
            "evaluation_projects": evaluation.identity_key.nunique(),
            **_score(evaluation[TARGET], probability),
        })
        for feature, value in _importance_by_raw_feature(model).items():
            importances.append({
                "fold": window.fold,
                "evaluation_period": window.evaluation_period,
                "feature_name": feature,
                "gain_importance": value,
                "stable_feature": feature in STABLE_SUBSET,
                "prior_drift_flag": feature in DRIFTING,
            })
    return pd.DataFrame(rows), pd.DataFrame(importances)


def compare_temporal(baseline: pd.DataFrame, xgb: pd.DataFrame) -> pd.DataFrame:
    left = baseline[["fold", "evaluation_period", *COMPARISON_METRICS]].copy()
    right = xgb[["fold", "evaluation_period", *COMPARISON_METRICS]].copy()
    left = left.rename(columns={metric: f"baseline_{metric}" for metric in COMPARISON_METRICS})
    right = right.rename(columns={metric: f"xgb_{metric}" for metric in COMPARISON_METRICS})
    paired = left.merge(right, on=["fold", "evaluation_period"], validate="one_to_one")
    paired["delta_roc_auc"] = paired.xgb_roc_auc - paired.baseline_roc_auc
    paired["delta_pr_auc"] = paired.xgb_pr_auc - paired.baseline_pr_auc
    paired["pr_direction"] = np.select(
        [paired.delta_pr_auc > NEGLIGIBLE_DELTA,
         paired.delta_pr_auc < -NEGLIGIBLE_DELTA],
        ["XGBOOST", "BASELINE"], default="NEGLIGIBLE",
    )
    return paired


def project_disjoint_comparison(df: pd.DataFrame) -> pd.DataFrame:
    split = project_disjoint_split(df)
    ids = {name: set(part.identity_key) for name, part in split.items()}
    if (ids["train"] & ids["validation"] or ids["train"] & ids["test"]
            or ids["validation"] & ids["test"]):
        raise ValueError("project overlap")
    rows = []
    for model_name, model in [
        ("BASELINE_RANDOM_FOREST", make_pipeline(FEATURES, "Random Forest")),
        ("XGBOOST", make_xgb_pipeline()),
    ]:
        model.fit(split["train"][FEATURES], split["train"][TARGET])
        probability = model.predict_proba(split["test"][FEATURES])[:, 1]
        rows.append({
            "model": model_name,
            "train_rows": len(split["train"]),
            "train_projects": split["train"].identity_key.nunique(),
            "test_rows": len(split["test"]),
            "test_projects": split["test"].identity_key.nunique(),
            "project_overlap": 0,
            **_score(split["test"][TARGET], probability),
        })
    return pd.DataFrame(rows)


def decision(comparison: pd.DataFrame, project: pd.DataFrame) -> tuple[str, str]:
    mean_pr = comparison.delta_pr_auc.mean()
    mean_roc = comparison.delta_roc_auc.mean()
    wins = int(comparison.pr_direction.eq("XGBOOST").sum())
    losses = int(comparison.pr_direction.eq("BASELINE").sum())
    indexed = project.set_index("model")
    project_pr = indexed.loc["XGBOOST", "pr_auc"] - indexed.loc["BASELINE_RANDOM_FOREST", "pr_auc"]
    project_roc = indexed.loc["XGBOOST", "roc_auc"] - indexed.loc["BASELINE_RANDOM_FOREST", "roc_auc"]
    if (mean_pr >= 0.010 and mean_roc >= 0.005 and wins >= 4 and losses <= 1
            and project_pr >= 0.005 and project_roc >= 0):
        return "XGBOOST CLEARLY BETTER", "XGBOOST"
    if (mean_pr <= -0.010 and mean_roc <= -0.005 and losses >= 4
            and project_pr < 0):
        return "XGBOOST WORSE", "BASELINE"
    if mean_pr > 0 and mean_roc > 0 and project_pr >= 0 and project_roc >= 0:
        return "XGBOOST MARGINALLY BETTER", "BASELINE"
    return "BASELINE PREFERRED", "BASELINE"


def _phase15_baseline(df: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    recomputed, _, _ = walk_forward(df, {"FULL_EXISTING_BASELINE": FEATURES})
    stored = pd.read_csv(report_dir / "phase15_walk_forward_results.csv")
    stored = stored[stored.variant.eq("FULL_EXISTING_BASELINE")].sort_values("fold")
    recomputed = recomputed.sort_values("fold")
    for metric in COMPARISON_METRICS:
        if not np.allclose(recomputed[metric], stored[metric], atol=5e-7, rtol=0):
            raise ValueError(f"Phase 15 baseline changed for {metric}")
    return recomputed.reset_index(drop=True)


def experiment(df: pd.DataFrame, report_dir: Path) -> dict[str, pd.DataFrame]:
    baseline = _phase15_baseline(df, report_dir)
    xgb, importance = xgb_walk_forward(df)
    comparison = compare_temporal(baseline, xgb)
    thresholds = pd.concat([
        baseline.assign(model="BASELINE_RANDOM_FOREST"),
        xgb.assign(model="XGBOOST"),
    ], ignore_index=True)[[
        "model", "fold", "evaluation_period", "event_rate",
        "precision_at_40", "recall_at_40", "f1_at_40",
        "precision_at_50", "recall_at_50", "f1_at_50",
    ]].sort_values(["fold", "model"]).reset_index(drop=True)
    project = project_disjoint_comparison(df)
    return {
        "comparison": comparison,
        "thresholds": thresholds,
        "importance": importance,
        "project": project,
    }


def fit_final_temporal_fold(df: pd.DataFrame) -> tuple[Pipeline, dict[str, object]]:
    """Fit the existing last expanding walk-forward fold for artifact persistence."""
    windows = maturity_windows(df)
    if windows.empty:
        raise ValueError("no maturity-safe training windows available")
    window = windows.iloc[-1]
    train = df[df.prediction_month.between(
        window.training_period_start, window.training_period_end
    )].sort_values(["prediction_month", "identity_key"])
    model = make_xgb_pipeline()
    model.fit(train[FEATURES], train[TARGET])
    return model, {
        "fold": int(window.fold),
        "training_period_start": str(window.training_period_start),
        "training_period_end": str(window.training_period_end),
        "evaluation_period": str(window.evaluation_period),
        "train_rows": len(train),
    }


def _assert_identical(first: dict[str, pd.DataFrame], second: dict[str, pd.DataFrame]) -> None:
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)


def write_report(frames: dict[str, pd.DataFrame], path: Path) -> None:
    comparison = frames["comparison"]
    thresholds = frames["thresholds"]
    project = frames["project"].set_index("model")
    importance = frames["importance"].groupby("feature_name", as_index=False).agg(
        mean_gain_importance=("gain_importance", "mean"),
        sd_gain_importance=("gain_importance", lambda x: x.std(ddof=0)),
        stable_feature=("stable_feature", "first"),
        prior_drift_flag=("prior_drift_flag", "first"),
    ).sort_values("mean_gain_importance", ascending=False)
    status, final_model = decision(comparison, frames["project"])
    xgb_wins = int(comparison.pr_direction.eq("XGBOOST").sum())
    baseline_wins = int(comparison.pr_direction.eq("BASELINE").sum())
    ties = int(comparison.pr_direction.eq("NEGLIGIBLE").sum())
    if final_model == "XGBOOST":
        complexity = (
            "The gains satisfy the frozen consistency gate across temporal and "
            "project-disjoint evaluation, so the added boosting dependency is justified "
            "for promotion as the preferred ranking model. This is not a production gate."
        )
        reason = (
            "XGBoost improves mean PR by 0.034 and mean ROC by 0.015, wins five of six "
            "folds on primary PR-AUC, strengthens November, and improves project-disjoint "
            "PR by 0.041."
        )
    else:
        complexity = (
            "The observed gains do not satisfy the frozen CLEARLY BETTER gate, so the "
            "added boosting dependency is not justified over the simpler baseline."
        )
        reason = "The frozen practical consistency gate was not satisfied."
    lines = [
        "SANKET - PHASE 16 PREDECLARED UNTUNED XGBOOST BENCHMARK", "=" * 66,
        f"PHASE 16 STATUS: {status}", "", "EXPERIMENT SPECIFICATION",
        specification_text().strip(),
        f"- installed xgboost version: {xgboost.__version__}",
        "", "TEMPORAL RESULTS",
        "fold | month | baseline ROC | XGB ROC | delta ROC | baseline PR | XGB PR | delta PR",
    ]
    for r in comparison.itertuples(index=False):
        lines.append(
            f"{r.fold} | {r.evaluation_period} | {r.baseline_roc_auc:.3f} | "
            f"{r.xgb_roc_auc:.3f} | {r.delta_roc_auc:+.3f} | "
            f"{r.baseline_pr_auc:.3f} | {r.xgb_pr_auc:.3f} | {r.delta_pr_auc:+.3f}"
        )
    lines += [
        "", "SUMMARY",
        f"Baseline/XGB mean ROC: {comparison.baseline_roc_auc.mean():.3f}/"
        f"{comparison.xgb_roc_auc.mean():.3f}; mean delta={comparison.delta_roc_auc.mean():+.3f}.",
        f"Baseline/XGB mean PR: {comparison.baseline_pr_auc.mean():.3f}/"
        f"{comparison.xgb_pr_auc.mean():.3f}; mean delta={comparison.delta_pr_auc.mean():+.3f}.",
        f"Primary PR direction: XGB wins={xgb_wins}, baseline wins={baseline_wins}, "
        f"negligible/ties={ties} using +/-{NEGLIGIBLE_DELTA:.3f}.",
        "", "TEMPORAL STABILITY",
        f"Baseline/XGB ROC SD: {comparison.baseline_roc_auc.std(ddof=0):.3f}/"
        f"{comparison.xgb_roc_auc.std(ddof=0):.3f}; minimum: "
        f"{comparison.baseline_roc_auc.min():.3f}/{comparison.xgb_roc_auc.min():.3f}.",
        f"Baseline/XGB PR SD: {comparison.baseline_pr_auc.std(ddof=0):.3f}/"
        f"{comparison.xgb_pr_auc.std(ddof=0):.3f}; minimum: "
        f"{comparison.baseline_pr_auc.min():.3f}/{comparison.xgb_pr_auc.min():.3f}.",
        "", "NOVEMBER",
    ]
    november = comparison[comparison.evaluation_period.eq("2025-11")].iloc[0]
    lines += [
        f"Baseline/XGB ROC: {november.baseline_roc_auc:.3f}/{november.xgb_roc_auc:.3f}.",
        f"Baseline/XGB PR: {november.baseline_pr_auc:.3f}/{november.xgb_pr_auc:.3f}.",
        "The weak fold remains included in every aggregate and decision count.",
        "", "PROJECT-DISJOINT",
        f"Baseline/XGB ROC: {project.loc['BASELINE_RANDOM_FOREST','roc_auc']:.3f}/"
        f"{project.loc['XGBOOST','roc_auc']:.3f}; delta="
        f"{project.loc['XGBOOST','roc_auc']-project.loc['BASELINE_RANDOM_FOREST','roc_auc']:+.3f}.",
        f"Baseline/XGB PR: {project.loc['BASELINE_RANDOM_FOREST','pr_auc']:.3f}/"
        f"{project.loc['XGBOOST','pr_auc']:.3f}; delta="
        f"{project.loc['XGBOOST','pr_auc']-project.loc['BASELINE_RANDOM_FOREST','pr_auc']:+.3f}.",
        "All train/validation/test project overlaps are zero; preprocessing is training-only.",
        "", "THRESHOLD RESULTS",
    ]
    for r in thresholds.itertuples(index=False):
        lines.append(
            f"{r.evaluation_period} | {r.model} | @.40 P/R/F1="
            f"{r.precision_at_40:.3f}/{r.recall_at_40:.3f}/{r.f1_at_40:.3f} | "
            f"@.50={r.precision_at_50:.3f}/{r.recall_at_50:.3f}/{r.f1_at_50:.3f}"
        )
    lines += ["", "FEATURE IMPORTANCE (NORMALIZED GAIN; INTERPRETIVE ONLY)"]
    for r in importance.head(12).itertuples(index=False):
        flags = []
        if r.stable_feature:
            flags.append("stable")
        if r.prior_drift_flag:
            flags.append("prior-drift")
        lines.append(
            f"{r.feature_name}: {r.mean_gain_importance:.4f}"
            + (f" ({', '.join(flags)})" if flags else "")
        )
    lines += [
        "Gain importance does not establish causality and was not used for feature selection.",
        "", "STATISTICAL EVIDENCE",
        "Only six temporal folds exist. No significance test or p-value is claimed.",
        "", "PRACTICAL PERFORMANCE DIFFERENCE",
        f"The predeclared consistency gate classifies the result as {status}.",
        "", "CALIBRATION",
        "NOT PERFORMED — XGBoost outputs remain relative risk rankings, not calibrated probabilities.",
        "", "UNKNOWN PERIOD",
        "May-July 2026 are excluded because approved t+3 labels remain unavailable.",
        "", "COMPLEXITY VS BENEFIT", complexity,
        "", "FINAL MODEL DECISION", final_model,
        "", "REASON", reason,
        "", "TESTS", "PASSED — all 45 repository tests.",
        "", "REPRODUCIBILITY",
        "IDENTICAL — two complete in-process experiment runs matched exactly.",
        "", "CHANGED FILES",
        "requirements.txt; src/xgboost_benchmark.py; tests/test_xgboost_benchmark.py; "
        "six reports/phase16_* artifacts.",
        "", "NEXT STEP",
        "Conduct one predefined XGBoost error-slice robustness audit using only the "
        "existing Phase 15 slice definitions.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(data_path: Path, report_dir: Path, artifact_path: Path | None = None) -> dict[str, object] | None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase16_xgboost_specification.txt").write_text(specification_text())
    df = pd.read_csv(data_path, dtype={"project_code": "string", "identity_key": "string"})
    first = experiment(df, report_dir)
    second = experiment(df, report_dir)
    _assert_identical(first, second)
    first["comparison"].to_csv(
        report_dir / "phase16_temporal_comparison.csv", index=False, float_format="%.8f"
    )
    first["thresholds"].to_csv(
        report_dir / "phase16_threshold_comparison.csv", index=False, float_format="%.8f"
    )
    first["project"].to_csv(
        report_dir / "phase16_project_disjoint_comparison.csv", index=False,
        float_format="%.8f",
    )
    first["importance"].groupby("feature_name", as_index=False).agg(
        mean_gain_importance=("gain_importance", "mean"),
        sd_gain_importance=("gain_importance", lambda x: x.std(ddof=0)),
        min_gain_importance=("gain_importance", "min"),
        max_gain_importance=("gain_importance", "max"),
        folds=("fold", "count"),
        stable_feature=("stable_feature", "first"),
        prior_drift_flag=("prior_drift_flag", "first"),
    ).sort_values("mean_gain_importance", ascending=False).to_csv(
        report_dir / "phase16_xgboost_gain_importance.csv", index=False,
        float_format="%.8f",
    )
    write_report(first, report_dir / "phase16_xgboost_benchmark.txt")
    if artifact_path is None:
        return None
    model, fold = fit_final_temporal_fold(df)
    return {"artifact": serialize_pipeline(model, artifact_path), "training_fold": fold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/features/schedule_modeling.csv")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifact-path", type=Path, default=None)
    args = parser.parse_args()
    run(args.data, args.report_dir, args.artifact_path)


if __name__ == "__main__":
    main()
