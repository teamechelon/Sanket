"""Leakage-safe point-in-time features for approved SANKET proxy targets.

Each row is computed from the current or earlier reports for one source-backed
identity. Targets are generated separately and joined only after features exist.
This module does not impute, scale, split, or train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.label_feasibility import CANDIDATES, _candidate_labels, load_data
from src.utils import PROCESSED_DIR, REPORTS_DIR


FEATURE_SPECS = [
    ("state", "static", "State printed at cutoff", "state", True, False, "CONDITIONAL", "Categorical value can be missing or corrected in a later publication."),
    ("sector", "static", "Sector printed at cutoff", "sector", True, False, "SAFE", "Snapshot category."),
    ("ministry", "static", "Ministry printed at cutoff", "ministry", True, False, "CONDITIONAL", "Missing in some source generations."),
    ("agency", "static", "Agency printed at cutoff", "agency", True, False, "CONDITIONAL", "Missing in some source generations."),
    ("project_age_months", "static", "Months from cutoff-published start date, else approval date, to report month", "start_date,date_of_approval,report_month", True, False, "CONDITIONAL", "Preserved missing; later corrections are not backfilled."),
    ("progress_current", "current", "Physical progress printed at cutoff", "physical_progress", True, False, "SAFE", "Missing preserved."),
    ("expenditure_current", "current", "Cumulative expenditure printed at cutoff", "expenditure", True, False, "SAFE", "Missing preserved."),
    ("original_cost_current", "current", "Original cost printed at cutoff", "original_cost", True, False, "SAFE", "Current publication only."),
    ("revised_cost_current", "current", "Revised cost already printed at cutoff", "revised_cost", True, False, "CONDITIONAL", "Never backfilled from a future report."),
    ("effective_target_months_from_cutoff", "current", "Cutoff-known effective target minus report month", "original_doc,revised_doc,report_month", True, False, "CONDITIONAL", "Uses revised DoC only when published by cutoff."),
    ("progress_change_1m", "historical", "Current progress minus exact t-1 progress", "physical_progress,report_month", True, False, "SAFE", "Requires exact prior month."),
    ("progress_change_3m", "historical", "Current progress minus exact t-3 progress", "physical_progress,report_month", True, False, "SAFE", "Requires exact three-month lag."),
    ("progress_velocity_3m", "historical", "Progress change over exact three months divided by 3", "physical_progress,report_month", True, False, "SAFE", "Backward-looking."),
    ("progress_acceleration_3m", "historical", "Current one-month progress change minus prior one-month change", "physical_progress,report_month", True, False, "SAFE", "Uses t, t-1 and t-2 only."),
    ("expenditure_change_1m", "historical", "Current expenditure minus exact t-1 expenditure", "expenditure,report_month", True, False, "SAFE", "Requires exact prior month."),
    ("expenditure_change_3m", "historical", "Current expenditure minus exact t-3 expenditure", "expenditure,report_month", True, False, "SAFE", "Requires exact three-month lag."),
    ("expenditure_velocity_3m", "historical", "Expenditure change over exact three months divided by 3", "expenditure,report_month", True, False, "SAFE", "Backward-looking."),
    ("cost_revision_count_to_date", "historical", "Count of effective-cost changes through cutoff", "original_cost,revised_cost", True, False, "SAFE", "Expanding count stops at cutoff."),
    ("last_cost_revision_pct", "historical", "Latest effective-cost percentage change observed by cutoff", "original_cost,revised_cost", True, False, "CONDITIONAL", "Missing until a valid prior cost exists."),
    ("months_since_cost_revision", "recency", "Months since latest effective-cost change known at cutoff", "original_cost,revised_cost,report_month", True, False, "SAFE", "Missing before first revision."),
    ("schedule_revision_count_to_date", "historical", "Count of effective-target changes through cutoff", "original_doc,revised_doc", True, False, "SAFE", "Expanding count stops at cutoff."),
    ("months_since_schedule_revision", "recency", "Months since latest effective-target change known at cutoff", "original_doc,revised_doc,report_month", True, False, "SAFE", "Missing before first revision."),
    ("months_since_material_progress_change", "recency", "Months since latest absolute progress change of at least one point", "physical_progress,report_month", True, False, "SAFE", "Missing before first material change."),
    ("expenditure_to_original_cost", "ratio", "Cutoff expenditure divided by cutoff original cost", "expenditure,original_cost", True, False, "SAFE", "Missing for zero/missing denominator."),
    ("expenditure_to_revised_cost", "ratio", "Cutoff expenditure divided by cutoff revised cost", "expenditure,revised_cost", True, False, "CONDITIONAL", "Available only when revised cost is already published."),
    ("cost_revision_pct_current", "ratio", "Cutoff effective cost versus cutoff original cost", "original_cost,revised_cost", True, False, "CONDITIONAL", "Snapshot revision magnitude, not a formal approval event."),
    ("months_observed", "recency", "Number of observations through cutoff", "identity_key,report_month", True, False, "SAFE", "Expanding count only."),
    ("months_since_first_observation", "recency", "Calendar months since first observed report through cutoff", "identity_key,report_month", True, False, "SAFE", "No final-history knowledge."),
    ("revised_cost_missing", "missingness", "One when revised cost is absent at cutoff", "revised_cost", True, False, "SAFE", "Missingness is retained rather than imputed."),
    ("progress_missing", "missingness", "One when physical progress is absent at cutoff", "physical_progress", True, False, "SAFE", "Missingness is retained rather than imputed."),
    ("future_revised_cost", "rejected", "Revised cost from after cutoff", "future revised_cost", False, True, "UNSAFE", "Target-window information; excluded."),
    ("future_revised_completion", "rejected", "Revised DoC from after cutoff", "future revised_doc", False, True, "UNSAFE", "Target-window information; excluded."),
    ("full_history_revision_count", "rejected", "Revision count across complete dataset", "all project months", False, True, "UNSAFE", "Leaks observations after cutoff; excluded."),
    ("last_observed_value", "rejected", "Final observed project value", "all project months", False, True, "UNSAFE", "Leaks dataset availability; excluded."),
]


def feature_inventory() -> pd.DataFrame:
    columns = ["feature_name", "category", "definition", "source_fields", "available_at_cutoff", "uses_future_information", "status", "notes"]
    return pd.DataFrame(FEATURE_SPECS, columns=columns)


def _months_from_date(values: pd.Series, report_number: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, format="%m/%Y", errors="coerce").dt.to_period("M")
    nums = pd.Series(dates.dt.year * 12 + dates.dt.month, index=values.index, dtype="float64")
    return report_number - nums


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["traceable"]].copy().sort_values(["identity_key", "month_number"])
    if d.duplicated(["identity_key", "report_month"]).any():
        raise ValueError("project-month uniqueness violated before feature construction")
    for field in ("physical_progress", "expenditure", "effective_cost"):
        for lag in (1, 2, 3):
            lagged = d[["identity_key", "month_number", field]].copy()
            lagged["month_number"] += lag
            lagged = lagged.rename(columns={field: f"_{field}_lag{lag}"})
            d = d.merge(lagged, on=["identity_key", "month_number"], how="left", validate="one_to_one")
    d["progress_change_1m"] = d["physical_progress"] - d["_physical_progress_lag1"]
    d["progress_change_3m"] = d["physical_progress"] - d["_physical_progress_lag3"]
    d["progress_velocity_3m"] = d["progress_change_3m"] / 3
    d["progress_acceleration_3m"] = d["progress_change_1m"] - (d["_physical_progress_lag1"] - d["_physical_progress_lag2"])
    d["expenditure_change_1m"] = d["expenditure"] - d["_expenditure_lag1"]
    d["expenditure_change_3m"] = d["expenditure"] - d["_expenditure_lag3"]
    d["expenditure_velocity_3m"] = d["expenditure_change_3m"] / 3
    start = d["start_date"].where(d["start_date"].notna(), d["date_of_approval"])
    d["project_age_months"] = _months_from_date(start, d["month_number"])
    d["effective_target_months_from_cutoff"] = d["effective_end"].map(lambda x: np.nan if pd.isna(x) else x.ordinal + 1970 * 12 + 1) - d["month_number"]
    d["expenditure_to_original_cost"] = d["expenditure"] / d["original_cost"].where(d["original_cost"] > 0)
    d["expenditure_to_revised_cost"] = d["expenditure"] / d["revised_cost"].where(d["revised_cost"] > 0)
    d["cost_revision_pct_current"] = d["effective_cost"] / d["original_cost"].where(d["original_cost"] > 0) - 1
    d["revised_cost_missing"] = d["revised_cost"].isna().astype(int)
    d["progress_missing"] = d["physical_progress"].isna().astype(int)
    d["months_observed"] = d.groupby("identity_key").cumcount() + 1
    d["months_since_first_observation"] = d["month_number"] - d.groupby("identity_key")["month_number"].transform("first")

    # Revision history compares with the latest observation available by T.
    # Unlike fixed-lag trends, it remains meaningful when an intermediate
    # report is absent, and still never looks beyond the cutoff.
    cost_prev = d.groupby("identity_key", sort=False)["effective_cost"].shift(1)
    target_prev = d.groupby("identity_key", sort=False)["effective_end"].shift(1)
    cost_change = d["effective_cost"].notna() & cost_prev.notna() & ~np.isclose(d["effective_cost"], cost_prev, atol=.01, rtol=0)
    schedule_change = d["effective_end"].notna() & target_prev.notna() & d["effective_end"].ne(target_prev)
    material_progress = d["progress_change_1m"].abs().ge(1)
    d["cost_revision_count_to_date"] = cost_change.groupby(d["identity_key"]).cumsum().astype(int)
    d["schedule_revision_count_to_date"] = schedule_change.groupby(d["identity_key"]).cumsum().astype(int)
    d["last_cost_revision_pct"] = ((d["effective_cost"] / cost_prev - 1).where(cost_change)).groupby(d["identity_key"]).ffill()
    for flag, name in ((cost_change, "months_since_cost_revision"), (schedule_change, "months_since_schedule_revision"), (material_progress, "months_since_material_progress_change")):
        last = d["month_number"].where(flag).groupby(d["identity_key"]).ffill()
        d[name] = d["month_number"] - last

    rename = {"physical_progress": "progress_current", "expenditure": "expenditure_current", "original_cost": "original_cost_current", "revised_cost": "revised_cost_current"}
    d = d.rename(columns=rename)
    safe_names = feature_inventory().query("status in ['SAFE','CONDITIONAL']")["feature_name"].tolist()
    return d[["project_code", "identity_key", "report_month", *safe_names]].rename(columns={"report_month": "prediction_month"}).reset_index(drop=True)


def _target(name: str):
    return next(c for c in CANDIDATES if c.target_name == name)


def modeling_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(df)
    schedule, _ = _candidate_labels(df, _target("future_schedule_later_3m"))
    cost, _ = _candidate_labels(df, _target("future_cost_increase_5pct_6m"))
    schedule = schedule.rename(columns={"label": "future_schedule_later_3m"})
    cost = cost.rename(columns={"label": "future_cost_increase_5pct_6m"})
    keys = ["identity_key", "prediction_month"]
    return features, features.merge(schedule, on=keys, how="inner", validate="one_to_one"), features.merge(cost, on=keys, how="inner", validate="one_to_one")


def write_quality(features: pd.DataFrame, schedule: pd.DataFrame, cost: pd.DataFrame, inventory: pd.DataFrame, source_rows: int, path: Path) -> None:
    model_features = inventory.query("status in ['SAFE','CONDITIONAL']")["feature_name"].tolist()
    missing = features[model_features].isna().mean().mul(100).sort_values(ascending=False)
    constants = [c for c in model_features if features[c].nunique(dropna=True) <= 1]
    lines = ["SANKET - POINT-IN-TIME FEATURE QUALITY", "=" * 52, f"Total point-in-time feature rows: {len(features)}",
             f"Schedule eligible rows: {len(schedule)}", f"Cost eligible rows: {len(cost)}",
             f"Unique projects: {features.identity_key.nunique()}", f"Features created: {len(model_features)}",
             f"Identifierless observations excluded: {source_rows-len(features)}", "Target-unknown observations are excluded from each supervised table, never converted to zero.", "",
             "Missingness (%):", *[f"  {k}: {v:.2f}" for k,v in missing.items()], "", f"Constant features: {', '.join(constants) if constants else 'none'}",
             "Suspicious features: negative expenditure/progress changes can represent source corrections; values are retained for later train-only treatment.",
             "Unsafe features excluded: " + ", ".join(inventory.query("status == 'UNSAFE'").feature_name),
             "Conditional features: " + ", ".join(inventory.query("status == 'CONDITIONAL'").feature_name),
             "No imputation, scaling, model fitting, or accuracy calculation was performed."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(input_path: Path, feature_dir: Path, report_dir: Path) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    inventory = feature_inventory(); features, schedule, cost = modeling_tables(df)
    inventory.to_csv(report_dir / "feature_inventory.csv", index=False)
    schedule.to_csv(feature_dir / "schedule_modeling.csv", index=False, float_format="%.6f")
    cost.to_csv(feature_dir / "cost_modeling.csv", index=False, float_format="%.6f")
    write_quality(features, schedule, cost, inventory, len(df), report_dir / "feature_quality_report.txt")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input", type=Path, default=PROCESSED_DIR/"project_monthly.csv"); p.add_argument("--feature-dir", type=Path, default=Path("data/features")); p.add_argument("--report-dir", type=Path, default=REPORTS_DIR); a=p.parse_args(); run(a.input,a.feature_dir,a.report_dir)


if __name__ == "__main__": main()
