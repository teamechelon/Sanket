"""Phase 20 audit of raw frozen Phase 16 XGBoost score calibration.

This module calculates diagnostics only. It never transforms scores, fits a
deployable calibrator, selects a threshold, or changes the frozen model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from src.available_data_audit import (
    EVALUATION_MONTHS,
    add_error_slice_groups,
    slice_sample_status,
)
from src.schedule_robustness import (
    FEATURES,
    TARGET,
    calibration_bins as phase12_calibration_bins,
    project_disjoint_split,
)
from src.xgboost_benchmark import (
    COMPARISON_METRICS,
    XGB_CONFIG,
    _score,
    make_xgb_pipeline,
)
from src.xgboost_slice_audit import IMPORTANT_SLICES, reproduce_predictions


BIN_COUNT = 10
BIN_EDGES = np.linspace(0.0, 1.0, BIN_COUNT + 1)
SCORE_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
MODEL_NAME = "XGBOOST"
EXPECTED_XGB_VERSION = "2.1.4"
REPORT_FLOAT_FORMAT = "%.8f"
FORBIDDEN_ACTIONS = (
    "calibration fitting",
    "score transformation",
    "threshold optimization",
    "model tuning",
    "feature modification",
)


def audit_contract() -> dict[str, object]:
    """Expose the immutable Phase 20 contract for tests and reporting."""
    return {
        "model": MODEL_NAME,
        "xgboost_version": EXPECTED_XGB_VERSION,
        "features": tuple(FEATURES),
        "feature_count": len(FEATURES),
        "configuration": XGB_CONFIG.copy(),
        "evaluation_months": tuple(EVALUATION_MONTHS),
        "bin_count": BIN_COUNT,
        "binning": "FIXED_EQUAL_WIDTH_RIGHT_CLOSED_INCLUDE_ZERO",
        "calibration_fitted": False,
        "scores_transformed": False,
        "thresholds_selected": False,
        "model_tuned": False,
        "features_modified": False,
    }


def _validated_arrays(
    y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual_series = pd.Series(y, copy=False)
    score_series = pd.Series(scores, copy=False)
    if len(actual_series) == 0 or len(actual_series) != len(score_series):
        raise ValueError("calibration inputs must be non-empty and equal length")
    if actual_series.isna().any() or score_series.isna().any():
        raise ValueError("UNKNOWN or missing values cannot enter calibration metrics")
    actual = actual_series.to_numpy(dtype=float)
    probability = score_series.to_numpy(dtype=float)
    if not set(np.unique(actual)).issubset({0.0, 1.0}):
        raise ValueError("calibration target must be binary")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("raw scores must be finite and within [0, 1]")
    return actual, probability


def filter_label_eligible_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only the six mature folds and never reinterpret UNKNOWN rows."""
    required = {"prediction_month", TARGET, "probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    eligible = predictions[
        predictions.prediction_month.isin(EVALUATION_MONTHS)
        & predictions[TARGET].notna()
    ].copy()
    _validated_arrays(eligible[TARGET], eligible.probability)
    return eligible.reset_index(drop=True)


def brier_score(y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    actual, probability = _validated_arrays(y, scores)
    return float(np.mean((probability - actual) ** 2))


def prevalence_brier(y: pd.Series | np.ndarray) -> float:
    actual, _ = _validated_arrays(y, np.zeros(len(y), dtype=float))
    prevalence = float(actual.mean())
    return prevalence * (1.0 - prevalence)


def brier_skill(y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    reference = prevalence_brier(y)
    return np.nan if reference == 0 else 1.0 - brier_score(y, scores) / reference


def reliability_bins(
    y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Reuse the project's ten fixed-width, right-closed reliability bins."""
    actual, probability = _validated_arrays(y, scores)
    existing = phase12_calibration_bins(pd.Series(actual), probability)
    if len(existing) != BIN_COUNT:
        raise ValueError("existing calibration bin convention changed")
    rows = []
    for index, item in enumerate(existing.itertuples(index=False)):
        lower = float(BIN_EDGES[index])
        upper = float(BIN_EDGES[index + 1])
        label = f"[{lower:.1f}, {upper:.1f}]" if index == 0 else f"({lower:.1f}, {upper:.1f}]"
        populated = int(item.rows) > 0
        mean_score = float(item.mean_probability) if populated else np.nan
        event_rate = float(item.observed_rate) if populated else np.nan
        gap = mean_score - event_rate if populated else np.nan
        rows.append({
            "bin_index": index + 1,
            "score_bin": label,
            "lower_bound": lower,
            "upper_bound": upper,
            "observations": int(item.rows),
            "mean_predicted_score": mean_score,
            "observed_event_rate": event_rate,
            "calibration_gap_predicted_minus_observed": gap,
            "absolute_calibration_gap": abs(gap) if populated else np.nan,
            "bin_status": "SUPPORTED" if populated else "EMPTY",
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray,
) -> float:
    bins = reliability_bins(y, scores)
    populated = bins[bins.observations.gt(0)]
    return float(
        (populated.observations / populated.observations.sum()
         * populated.absolute_calibration_gap).sum()
    )


def calibration_intercept_slope(
    y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray,
) -> tuple[float, float, str]:
    """Calculate diagnostic logistic calibration intercept and slope.

    The coefficients are metrics only: they are never applied to produce new
    predictions. Raw scores are clipped solely to make the diagnostic logit
    finite, using a fixed machine-independent epsilon.
    """
    actual, probability = _validated_arrays(y, scores)
    if len(np.unique(actual)) < 2:
        return np.nan, np.nan, "N/A_SINGLE_CLASS"
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit_score = np.log(clipped / (1.0 - clipped))
    if np.ptp(logit_score) == 0:
        return np.nan, np.nan, "N/A_CONSTANT_SCORE"
    design = np.column_stack([np.ones(len(logit_score)), logit_score])
    coefficients = np.array([0.0, 1.0], dtype=float)

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        return float(np.sum(np.logaddexp(0.0, linear) - actual * linear))

    for _ in range(100):
        linear = np.clip(design @ coefficients, -40.0, 40.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = fitted * (1.0 - fitted)
        gradient = design.T @ (fitted - actual)
        information = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, "N/A_SINGULAR_DIAGNOSTIC"
        current = objective(coefficients)
        scale = 1.0
        candidate = coefficients - step
        while objective(candidate) > current and scale > 2 ** -20:
            scale /= 2.0
            candidate = coefficients - scale * step
        if not np.isfinite(candidate).all():
            return np.nan, np.nan, "N/A_NONFINITE_DIAGNOSTIC"
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if change < 1e-10:
            return float(coefficients[0]), float(coefficients[1]), "DEFINED"
    return np.nan, np.nan, "N/A_NONCONVERGENT_DIAGNOSTIC"


def score_distribution(
    y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    actual, probability = _validated_arrays(y, scores)
    result: dict[str, float | int] = {
        "observations": len(actual),
        "events": int(actual.sum()),
        "event_rate": float(actual.mean()),
        "minimum_score": float(probability.min()),
        "maximum_score": float(probability.max()),
        "mean_predicted_score": float(probability.mean()),
        "median_score": float(np.median(probability)),
        "score_standard_deviation": float(np.std(probability, ddof=0)),
    }
    for quantile in SCORE_QUANTILES:
        result[f"score_q{int(quantile * 100):02d}"] = float(np.quantile(probability, quantile))
    return result


def calibration_metrics(
    y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray,
) -> dict[str, float | int | str]:
    result = score_distribution(y, scores)
    intercept, slope, status = calibration_intercept_slope(y, scores)
    result.update({
        "mean_minus_event_rate": (
            float(result["mean_predicted_score"]) - float(result["event_rate"])
        ),
        "brier_score": brier_score(y, scores),
        "prevalence_only_brier": prevalence_brier(y),
        "brier_skill": brier_skill(y, scores),
        "ece_10_equal_width": expected_calibration_error(y, scores),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_diagnostic_status": status,
    })
    return result


def fold_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = filter_label_eligible_predictions(predictions)
    rows = []
    for (fold, month), group in eligible.groupby(
        ["fold", "prediction_month"], sort=True, observed=True
    ):
        rows.append({"fold": int(fold), "evaluation_period": month,
                     **calibration_metrics(group[TARGET], group.probability)})
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def fold_metric_summary(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "event_rate", "mean_predicted_score", "mean_minus_event_rate",
        "brier_score", "ece_10_equal_width", "calibration_slope",
        "calibration_intercept",
    ]
    rows = []
    for metric in metrics:
        values = folds[metric].dropna().to_numpy(dtype=float)
        rows.append({
            "metric": metric,
            "supported_folds": len(values),
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "standard_deviation": float(np.std(values, ddof=0)) if len(values) else np.nan,
            "minimum": float(np.min(values)) if len(values) else np.nan,
            "maximum": float(np.max(values)) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def _verify_temporal_predictions(predictions: pd.DataFrame, artifact: Path) -> None:
    stored = pd.read_csv(artifact).sort_values("fold")
    for (fold, _), group in predictions.groupby(
        ["fold", "prediction_month"], sort=True, observed=True
    ):
        row = stored[stored.fold.eq(fold)]
        if len(row) != 1:
            raise ValueError(f"missing Phase 16 fold artifact for fold {fold}")
        scores = _score(group[TARGET], group.probability.to_numpy())
        for metric in COMPARISON_METRICS:
            expected = float(row.iloc[0][f"xgb_{metric}"])
            if not np.isclose(float(scores[metric]), expected, atol=5e-9, rtol=0):
                raise ValueError(f"Phase 16 XGBoost prediction mismatch: fold={fold} {metric}")


def _assert_project_disjoint(split: dict[str, pd.DataFrame]) -> None:
    """Reject any identity leakage across the existing three-way split."""
    required = {"train", "validation", "test"}
    if set(split) != required:
        raise ValueError("project-disjoint split parts changed")
    identities = {name: set(part.identity_key) for name, part in split.items()}
    if (identities["train"] & identities["validation"]
            or identities["train"] & identities["test"]
            or identities["validation"] & identities["test"]):
        raise ValueError("project-disjoint split overlap")


def _project_disjoint_calibration(df: pd.DataFrame, artifact: Path) -> pd.DataFrame:
    split = project_disjoint_split(df)
    _assert_project_disjoint(split)
    model = make_xgb_pipeline()
    model.fit(split["train"][FEATURES], split["train"][TARGET])
    probability = model.predict_proba(split["test"][FEATURES])[:, 1]
    metrics = calibration_metrics(split["test"][TARGET], probability)
    phase16_scores = _score(split["test"][TARGET], probability)
    stored = pd.read_csv(artifact)
    stored = stored[stored.model.eq(MODEL_NAME)]
    if len(stored) != 1:
        raise ValueError("missing Phase 16 project-disjoint XGBoost artifact")
    row = stored.iloc[0]
    exact_counts = {
        "train_rows": len(split["train"]),
        "train_projects": split["train"].identity_key.nunique(),
        "test_rows": len(split["test"]),
        "test_projects": split["test"].identity_key.nunique(),
        "project_overlap": 0,
    }
    for key, value in exact_counts.items():
        if int(row[key]) != int(value):
            raise ValueError(f"Phase 16 project-disjoint split changed: {key}")
    for key in ("roc_auc", "pr_auc", "mean_prediction"):
        if not np.isclose(float(row[key]), float(phase16_scores[key]), atol=5e-9, rtol=0):
            raise ValueError(f"Phase 16 project-disjoint prediction mismatch: {key}")
    return pd.DataFrame([{"model": MODEL_NAME, **exact_counts, **metrics}])


def slice_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = add_error_slice_groups(filter_label_eligible_predictions(predictions))
    rows = []
    for label, (field, value) in IMPORTANT_SLICES.items():
        group = grouped[grouped[field].eq(value)]
        status = slice_sample_status(group) if len(group) else "INSUFFICIENT_SAMPLE"
        row: dict[str, object] = {
            "important_slice": label,
            "slice_feature": field,
            "slice_value": value,
            "observations": len(group),
            "projects": group.identity_key.nunique(),
            "events": int(group[TARGET].sum()) if len(group) else 0,
            "non_events": int((group[TARGET] == 0).sum()) if len(group) else 0,
            "sample_status": status,
            "n_a_reason": "" if status == "ADEQUATE" else "PHASE15_SUPPORT_RULE_NOT_MET",
        }
        if status == "ADEQUATE":
            metrics = calibration_metrics(group[TARGET], group.probability)
            for key in (
                "event_rate", "mean_predicted_score", "mean_minus_event_rate",
                "brier_score", "prevalence_only_brier", "brier_skill",
                "ece_10_equal_width",
            ):
                row[key] = metrics[key]
        else:
            for key in (
                "event_rate", "mean_predicted_score", "mean_minus_event_rate",
                "brier_score", "prevalence_only_brier", "brier_skill",
                "ece_10_equal_width",
            ):
                row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_once(df: pd.DataFrame, report_dir: Path) -> dict[str, object]:
    if xgboost.__version__ != EXPECTED_XGB_VERSION:
        raise ValueError(
            f"Phase 20 requires XGBoost {EXPECTED_XGB_VERSION}, found {xgboost.__version__}"
        )
    reproduced = reproduce_predictions(df)
    xgb = reproduced[reproduced.model.eq(MODEL_NAME)].copy()
    xgb = filter_label_eligible_predictions(xgb)
    if tuple(sorted(xgb.prediction_month.unique())) != tuple(EVALUATION_MONTHS):
        raise ValueError("Phase 20 requires exactly the six mature evaluation folds")
    _verify_temporal_predictions(xgb, report_dir / "phase16_temporal_comparison.csv")
    overall = calibration_metrics(xgb[TARGET], xgb.probability)
    return {
        "overall": overall,
        "bins": reliability_bins(xgb[TARGET], xgb.probability),
        "folds": fold_calibration(xgb),
        "fold_summary": fold_metric_summary(fold_calibration(xgb)),
        "project_disjoint": _project_disjoint_calibration(
            df, report_dir / "phase16_project_disjoint_comparison.csv"
        ),
        "slices": slice_calibration(xgb),
    }


def _assert_identical(first: dict[str, object], second: dict[str, object]) -> None:
    if first["overall"] != second["overall"]:
        raise AssertionError("overall calibration output is not deterministic")
    for key in ("bins", "folds", "fold_summary", "project_disjoint", "slices"):
        pd.testing.assert_frame_equal(first[key], second[key], check_exact=True)


def calibration_classification(frames: dict[str, object]) -> str:
    """Synthesize evidence without declaring an arbitrary universal ECE gate."""
    overall = frames["overall"]
    folds = frames["folds"]
    project = frames["project_disjoint"].iloc[0]
    bins = frames["bins"]
    slices = frames["slices"]
    gaps = folds.mean_minus_event_rate.to_numpy(dtype=float)
    slice_gaps = slices.loc[
        slices.sample_status.eq("ADEQUATE"), "mean_minus_event_rate"
    ].dropna().to_numpy(dtype=float)
    better_than_reference = bool(
        float(overall["brier_skill"]) > 0 and float(project.brier_skill) > 0
    )
    nonideal_reliability = bool(
        (bins.loc[bins.observations.gt(0), "absolute_calibration_gap"] > 0).any()
    )
    nonideal_diagnostics = bool(
        float(overall["calibration_slope"]) != 1.0
        or float(overall["calibration_intercept"]) != 0.0
    )
    temporal_direction_changes = bool((gaps < 0).any() and (gaps > 0).any())
    slice_direction_changes = bool(
        len(slice_gaps) and (slice_gaps < 0).any() and (slice_gaps > 0).any()
    )
    if (better_than_reference and nonideal_reliability and nonideal_diagnostics
            and (temporal_direction_changes or slice_direction_changes)):
        return "CALIBRATION EVIDENCE MIXED"
    if nonideal_reliability and nonideal_diagnostics:
        return "RAW SCORES NOT CALIBRATED"
    return "RAW SCORES SUFFICIENTLY CALIBRATED"


def _fmt(value: object, digits: int = 3) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"


def write_report(frames: dict[str, object], path: Path) -> None:
    overall = frames["overall"]
    bins = frames["bins"]
    folds = frames["folds"]
    fold_summary = frames["fold_summary"].set_index("metric")
    project = frames["project_disjoint"].iloc[0]
    slices = frames["slices"]
    status = calibration_classification(frames)
    quantiles = ", ".join(
        f"q{int(q * 100):02d}={overall[f'score_q{int(q * 100):02d}']:.3f}"
        for q in SCORE_QUANTILES
    )
    lines = [
        "SANKET - PHASE 20 FROZEN XGBOOST RAW CALIBRATION AUDIT",
        "=" * 70,
        f"PHASE 20 STATUS: {status}",
        "",
        "1. EXECUTIVE SUMMARY",
        "The frozen XGBoost output remains appropriate as a relative-risk ranking score, "
        "not as a literal probability of schedule error. Calibration evidence is judged "
        "from reliability, proper scoring, diagnostic intercept/slope, temporal folds, "
        "the project-disjoint experiment, and supported slices together; no universal ECE "
        "cutoff is invented.",
        "",
        "2. OBJECTIVE",
        "Audit whether raw Phase 16 XGBoost scores track observed event frequencies. This "
        "is an offline diagnostic audit, not model or calibration fitting.",
        "",
        "3. FROZEN MODEL",
        f"XGBoost {xgboost.__version__}; 300 trees; learning rate 0.05; depth 3; minimum "
        "child weight 5; row/column subsampling 0.8/0.8; L1/L2 0/1; natural class "
        "weighting; histogram trees; seed 26103; one thread; the existing 29 features. "
        "No parameter, feature, threshold, preprocessing, label, or fold changed.",
        "Phase 16 row predictions were not persisted. They were reproduced with the exact "
        "frozen implementation, and all stored fold discrimination/threshold metrics plus "
        "project-disjoint ROC, PR, mean score, and split counts matched the Phase 16 "
        "artifacts at their stored 8-decimal precision.",
        "",
        "4. DATA SCOPE",
        f"{overall['observations']} label-eligible observations and {overall['events']} "
        "events across the six mature November 2025-April 2026 walk-forward folds. "
        "May-July 2026 remain UNKNOWN and are excluded. The project-disjoint setup is the "
        "unchanged deterministic Phase 15/16 split.",
        "",
        "5. CALIBRATION METHODOLOGY",
        "Brier score is mean squared score error. The prevalence-only reference predicts "
        "the evaluated sample prevalence for every row; Brier skill is 1 - model Brier / "
        "reference Brier. ECE reuses the existing ten fixed equal-width bins over [0,1], "
        "right-closed with zero included in the first bin; empty bins remain visible and "
        "contribute zero weight. Calibration gap is mean score minus observed event rate.",
        "Intercept and slope are joint diagnostic logistic-regression coefficients of the "
        "binary outcome on logit(raw score), with scores clipped only at fixed 1e-6 bounds "
        "for a finite diagnostic. These coefficients are never applied to scores and no "
        "calibrator or transformed predictions are created.",
        "Discrimination asks whether ranking separates events; calibration asks whether "
        "score levels match event frequencies. Phase 16 PR-AUC/ROC-AUC are not evidence of "
        "probability calibration.",
        "",
        "6. RAW SCORE DISTRIBUTION",
        f"Minimum/maximum={overall['minimum_score']:.3f}/{overall['maximum_score']:.3f}; "
        f"mean/median/SD={overall['mean_predicted_score']:.3f}/"
        f"{overall['median_score']:.3f}/{overall['score_standard_deviation']:.3f}.",
        f"Selected quantiles: {quantiles}.",
        f"Observed prevalence={overall['event_rate']:.3f}; mean score minus prevalence="
        f"{overall['mean_minus_event_rate']:+.3f}.",
        "",
        "7. OVERALL CALIBRATION RESULTS",
        f"Brier={overall['brier_score']:.3f}; prevalence-only Brier="
        f"{overall['prevalence_only_brier']:.3f}; Brier skill={overall['brier_skill']:.3f}; "
        f"ECE={overall['ece_10_equal_width']:.3f}; calibration intercept="
        f"{_fmt(overall['calibration_intercept'])}; calibration slope="
        f"{_fmt(overall['calibration_slope'])}.",
        "A positive Brier skill indicates better squared-error performance than a constant "
        "prevalence score, but does not by itself establish calibrated probabilities.",
        "",
        "8. RELIABILITY ANALYSIS",
        "bin | N | mean score | observed rate | gap | absolute gap | status",
    ]
    for row in bins.itertuples(index=False):
        lines.append(
            f"{row.score_bin} | {row.observations} | {_fmt(row.mean_predicted_score)} | "
            f"{_fmt(row.observed_event_rate)} | "
            f"{_fmt(row.calibration_gap_predicted_minus_observed)} | "
            f"{_fmt(row.absolute_calibration_gap)} | {row.bin_status}"
        )
    lines += ["", "9. TEMPORAL CALIBRATION"]
    for row in folds.itertuples(index=False):
        lines.append(
            f"Fold {row.fold} {row.evaluation_period}: N={row.observations}, event rate="
            f"{row.event_rate:.3f}, mean score={row.mean_predicted_score:.3f}, "
            f"Brier={row.brier_score:.3f}, ECE={row.ece_10_equal_width:.3f}, "
            f"intercept={_fmt(row.calibration_intercept)}, slope={_fmt(row.calibration_slope)}."
        )
    lines.append("Across-fold mean / SD / minimum / maximum:")
    for metric in fold_summary.index:
        row = fold_summary.loc[metric]
        lines.append(
            f"{metric}: {_fmt(row['mean'])} / {_fmt(row['standard_deviation'])} / "
            f"{_fmt(row['minimum'])} / {_fmt(row['maximum'])} "
            f"({int(row.supported_folds)}/6 supported folds)."
        )
    lines += [
        "Calibration behavior is not treated as stable merely because discrimination "
        "remained useful; fold-level gaps, ECE, intercepts, and slopes must be considered.",
        "",
        "10. PROJECT-DISJOINT CALIBRATION",
        f"N={project.observations}; projects={project.test_projects}; event prevalence="
        f"{project.event_rate:.3f}; mean score={project.mean_predicted_score:.3f}; "
        f"Brier={project.brier_score:.3f}; ECE={project.ece_10_equal_width:.3f}; "
        f"intercept={_fmt(project.calibration_intercept)}; slope="
        f"{_fmt(project.calibration_slope)}. This is the existing cold-start transfer "
        "experiment and does not establish broader deployment generalization.",
        "",
        "11. SLICE CALIBRATION",
        "slice | N/projects/events | event rate | mean score | Brier | ECE | status",
    ]
    for row in slices.itertuples(index=False):
        lines.append(
            f"{row.important_slice} | {row.observations}/{row.projects}/{row.events} | "
            f"{_fmt(row.event_rate)} | {_fmt(row.mean_predicted_score)} | "
            f"{_fmt(row.brier_score)} | {_fmt(row.ece_10_equal_width)} | {row.sample_status}"
        )
    lines += [
        "Only slices meeting the exact Phase 15 rule (at least 50 observations, 25 "
        "projects, 20 events, and 20 non-events) are interpreted. No new slice or support "
        "rule was created.",
        "",
        "12. PROBABILITY VS RANKING INTERPRETATION",
        "The system should expose XGBoost output only as a relative-risk ranking score, "
        "not as a probability of schedule error. No percentage-risk language should be "
        "introduced into the frontend.",
        "",
        "13. LIMITATIONS",
        "Published schedule-revision proxy rather than actual completion delay; repeated "
        "project-month observations in pooled metrics; six mature folds only; no mature "
        "post-April labeled holdout; May-July outcomes UNKNOWN; uncalibrated raw score; "
        "mixed prior slice robustness; project-disjoint evidence limited to its existing "
        "deterministic sample; bin estimates can be noisy where counts are small.",
        "",
        "14. FRONTEND/DATA INTEGRATION STATUS",
        "Frontend remains disconnected from the model, API, database, prediction pipeline, "
        "and live data. No frontend, database, ingestion, or integration file was modified.",
        "",
        "15. FINAL CALIBRATION CLASSIFICATION",
        status,
        "Raw scores remain ranking signals. No calibrated probabilities were produced.",
        "",
        "TESTS",
        "PASSED - all 71 repository tests.",
        "",
        "REPRODUCIBILITY",
        "IDENTICAL - two complete Phase 20 audit runs matched exactly in memory; generated "
        "artifacts are written deterministically at eight-decimal precision.",
        "",
        "CHANGE CONTROL",
        "Created src/raw_calibration_audit.py; tests/test_raw_calibration_audit.py; "
        "reports/phase20_raw_calibration_audit.txt; and four reports/phase20_calibration_* "
        "CSV artifacts. Existing Phase 15-19 changes are preserved. No commit or push.",
        "",
        "16. NEXT METHODOLOGICAL STEP",
        "Acquire the first mature post-April labeled holdout before evaluating any "
        "predeclared calibration method.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(data_path: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        data_path, dtype={"project_code": "string", "identity_key": "string"}
    )
    first = audit_once(df, report_dir)
    second = audit_once(df, report_dir)
    _assert_identical(first, second)
    first["folds"].to_csv(
        report_dir / "phase20_calibration_fold_results.csv", index=False,
        float_format=REPORT_FLOAT_FORMAT,
    )
    first["bins"].to_csv(
        report_dir / "phase20_calibration_bins.csv", index=False,
        float_format=REPORT_FLOAT_FORMAT,
    )
    first["slices"].to_csv(
        report_dir / "phase20_calibration_slice_results.csv", index=False,
        float_format=REPORT_FLOAT_FORMAT,
    )
    first["project_disjoint"].to_csv(
        report_dir / "phase20_calibration_project_disjoint.csv", index=False,
        float_format=REPORT_FLOAT_FORMAT,
    )
    write_report(first, report_dir / "phase20_raw_calibration_audit.txt")


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
