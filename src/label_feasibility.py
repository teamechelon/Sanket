"""Evaluate defensible future labels without training a model.

All candidate outcomes are constructed at identity_key + prediction_month.
Only source-backed primary (P:) and alternate (L:) identifiers are eligible.
An observation is negative only when the same project is observed at the exact
lookahead endpoint; disappearance is never interpreted as completion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR, REPORTS_DIR


@dataclass(frozen=True)
class Candidate:
    target_name: str
    definition: str
    horizon: int
    kind: str
    recommendation: str
    leakage_risk: str
    data_quality: str


CANDIDATES = (
    Candidate(
        "future_schedule_later_3m",
        "Effective completion target becomes later than its cutoff value within months t+1 through t+3; project must be observed at t+3.",
        3, "schedule_later", "POSSIBLE", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Observable publication outcome, but 447 later revised-target transitions occur in March 2026 and require source-semantic audit.",
    ),
    Candidate(
        "future_schedule_later_6m",
        "Effective completion target becomes later than its cutoff value within months t+1 through t+6; project must be observed at t+6.",
        6, "schedule_later", "POSSIBLE", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Longer horizon reduces eligible rows because the dataset spans only 16 months.",
    ),
    Candidate(
        "future_schedule_revision_3m",
        "Effective completion target changes in either direction within months t+1 through t+3; project must be observed at t+3.",
        3, "schedule_change", "WEAK", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Includes accelerations and corrections, so it is not a pure delay outcome.",
    ),
    Candidate(
        "future_overdue_active_3m",
        "At t+3 the project remains in the ongoing-project list and the target known at cutoff is earlier than the t+3 report month.",
        3, "overdue_active", "POSSIBLE", "MEDIUM",
        "Measures continued listing after a known target, not authoritative actual completion delay.",
    ),
    Candidate(
        "future_cost_increase_3m",
        "Effective project cost increases above its cutoff value within months t+1 through t+3; project must be observed at t+3.",
        3, "cost_increase", "POSSIBLE", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Sensitive to corrections and revised-cost fields appearing after prior blanks.",
    ),
    Candidate(
        "future_cost_increase_6m",
        "Effective project cost increases above its cutoff value within months t+1 through t+6; project must be observed at t+6.",
        6, "cost_increase", "POSSIBLE", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Longer horizon reduces eligible rows and includes small administrative changes.",
    ),
    Candidate(
        "future_cost_increase_5pct_6m",
        "Effective project cost increases by at least 5% over its cutoff value within months t+1 through t+6; project must be observed at t+6.",
        6, "cost_increase_5pct", "POSSIBLE", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Meaningful threshold, but 77 of 145 observed >5% monthly transitions occur in July 2026 and require source-semantic audit.",
    ),
    Candidate(
        "future_cost_overrun_onset_10pct_6m",
        "Among observations at or below 1.10 revised/original cost ratio at cutoff, ratio first exceeds 1.10 within t+1 through t+6; project observed at t+6.",
        6, "cost_overrun_onset", "POSSIBLE", "LOW_IF_FUTURE_FIELDS_EXCLUDED",
        "Excludes projects already over 10% at cutoff; depends on stable original cost reporting.",
    ),
)


def _month_number(values: pd.Series) -> pd.Series:
    periods = pd.PeriodIndex(values.astype(str), freq="M")
    return pd.Series(periods.year * 12 + periods.month, index=values.index, dtype="int64")


def _parse_month_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%m/%Y", errors="coerce").dt.to_period("M")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"project_code": "string", "legacy_ocms_code": "string", "identity_key": "string"},
    )
    df["month_number"] = _month_number(df["report_month"])
    df["original_end"] = _parse_month_date(df["original_doc"])
    df["revised_end"] = _parse_month_date(df["revised_doc"])
    df["effective_end"] = df["revised_end"].fillna(df["original_end"])
    for field in ("original_cost", "revised_cost", "expenditure", "physical_progress"):
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df["effective_cost"] = df["revised_cost"].fillna(df["original_cost"])
    df["traceable"] = df["identity_key"].str.match(r"^[PL]:", na=False)
    return df


def _longest_run(months: list[int]) -> int:
    if not months:
        return 0
    longest = current = 1
    for previous, value in zip(months, months[1:]):
        current = current + 1 if value == previous + 1 else 1
        longest = max(longest, current)
    return longest


def build_history_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    traceable = df[df["traceable"]].sort_values(["identity_key", "month_number"])
    for identity, group in traceable.groupby("identity_key", sort=True):
        group = group.drop_duplicates("month_number").sort_values("month_number")
        months = group["month_number"].astype(int).tolist()
        schedule_changed = group["effective_end"].ne(group["effective_end"].shift()) & group["effective_end"].notna() & group["effective_end"].shift().notna()
        cost_changed = ~np.isclose(group["effective_cost"], group["effective_cost"].shift(), rtol=0, atol=0.01, equal_nan=True)
        cost_changed &= group["effective_cost"].notna() & group["effective_cost"].shift().notna()
        revision_positions = group.loc[schedule_changed | cost_changed, "month_number"].astype(int).tolist()
        project_codes = group["project_code"].dropna()
        alternate_codes = group["legacy_ocms_code"].dropna()
        gaps = (months[-1] - months[0] + 1) - len(months)
        rows.append({
            "identity_key": identity,
            "project_code": project_codes.iloc[0] if len(project_codes) else pd.NA,
            "alternate_identifier": alternate_codes.iloc[0] if len(alternate_codes) else pd.NA,
            "first_observed_month": group["report_month"].iloc[0],
            "last_observed_month": group["report_month"].iloc[-1],
            "number_of_months": len(months),
            "consecutive_months": _longest_run(months),
            "gaps": gaps,
            "months_before_first_revision": revision_positions[0] - months[0] if revision_positions else pd.NA,
            "months_before_last_revision": revision_positions[-1] - months[0] if revision_positions else pd.NA,
            "eligible_cutoffs_3m": sum((m + 3) in months for m in months),
            "eligible_cutoffs_6m": sum((m + 6) in months for m in months),
        })
    return pd.DataFrame(rows)


def _candidate_labels(df: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, int]:
    traceable = df[df["traceable"]].copy()
    lookup = {(row.identity_key, int(row.month_number)): row for row in traceable.itertuples(index=False)}
    results: list[dict[str, object]] = []
    unknown = 0
    for cutoff in traceable.itertuples(index=False):
        current_month = int(cutoff.month_number)
        endpoint = lookup.get((cutoff.identity_key, current_month + candidate.horizon))
        if endpoint is None:
            unknown += 1
            continue
        future = [lookup.get((cutoff.identity_key, current_month + offset)) for offset in range(1, candidate.horizon + 1)]
        future = [row for row in future if row is not None]
        label: bool | None = None
        if candidate.kind in {"schedule_later", "schedule_change"}:
            if pd.notna(cutoff.effective_end) and pd.notna(endpoint.effective_end):
                dates = [row.effective_end for row in future if pd.notna(row.effective_end)]
                label = any(value > cutoff.effective_end for value in dates) if candidate.kind == "schedule_later" else any(value != cutoff.effective_end for value in dates)
        elif candidate.kind == "overdue_active":
            if pd.notna(cutoff.effective_end):
                endpoint_period = pd.Period(endpoint.report_month, freq="M")
                label = cutoff.effective_end < endpoint_period
        elif candidate.kind in {"cost_increase", "cost_increase_5pct"}:
            if pd.notna(cutoff.effective_cost) and cutoff.effective_cost > 0 and pd.notna(endpoint.effective_cost):
                threshold = cutoff.effective_cost * (1.05 if candidate.kind == "cost_increase_5pct" else 1.0)
                label = any(pd.notna(row.effective_cost) and row.effective_cost > threshold + 0.01 for row in future)
        elif candidate.kind == "cost_overrun_onset":
            if pd.notna(cutoff.original_cost) and cutoff.original_cost > 0 and pd.notna(cutoff.effective_cost):
                cutoff_ratio = cutoff.effective_cost / cutoff.original_cost
                if cutoff_ratio <= 1.10:
                    ratios = [row.effective_cost / row.original_cost for row in future if pd.notna(row.original_cost) and row.original_cost > 0 and pd.notna(row.effective_cost)]
                    label = any(value > 1.10 for value in ratios)
        if label is None:
            unknown += 1
            continue
        results.append({"identity_key": cutoff.identity_key, "prediction_month": cutoff.report_month, "label": int(label)})
    return pd.DataFrame(results), unknown


def build_candidate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    traceable_rows = int(df["traceable"].sum())
    for candidate in CANDIDATES:
        labels, unknown = _candidate_labels(df, candidate)
        positive = int(labels["label"].sum()) if len(labels) else 0
        eligible = len(labels)
        rows.append({
            "target_name": candidate.target_name,
            "definition": candidate.definition,
            "prediction_horizon": f"{candidate.horizon} months",
            "eligible_rows": eligible,
            "positive_rows": positive,
            "negative_rows": eligible - positive,
            "unknown_rows": unknown + int((~df["traceable"]).sum()),
            "positive_rate": positive / eligible if eligible else np.nan,
            "projects_covered": labels["identity_key"].nunique() if len(labels) else 0,
            "leakage_risk": candidate.leakage_risk,
            "data_quality": candidate.data_quality,
            "recommendation": candidate.recommendation,
        })
    return pd.DataFrame(rows)


def build_leakage_audit() -> pd.DataFrame:
    rows = [
        ("original_cost", "SAFE", "Value printed in the cutoff report; retain cutoff value only."),
        ("revised_cost", "CONDITIONAL", "Safe only as known at cutoff; unsafe for a target based on later cost revision if sourced from future rows."),
        ("expenditure", "CONDITIONAL", "Cutoff value and lagged history are safe; future expenditure is unsafe."),
        ("physical_progress", "CONDITIONAL", "Cutoff value and lagged history are safe; future progress is unsafe."),
        ("original_end_date", "SAFE", "Original target printed by cutoff is available at prediction time."),
        ("revised_end_date", "CONDITIONAL", "Latest value known at cutoff is safe; any later published revision is target information."),
        ("report_month", "SAFE", "Defines the prediction cutoff and temporal split."),
        ("project_age", "CONDITIONAL", "Safe when calculated only from approval/start information known by cutoff; invalid or future-corrected dates require exclusion."),
        ("months_since_start", "CONDITIONAL", "Safe from a cutoff-known start date; do not backfill with future corrections."),
        ("historical_expenditure_growth", "SAFE", "Use only observations at or before cutoff; never centered or future-filled windows."),
        ("historical_progress_growth", "SAFE", "Use only observations at or before cutoff; do not use future progress."),
        ("historical_cost_revisions", "SAFE", "Count/value changes observed no later than cutoff."),
        ("future_expenditure", "UNSAFE", "Occurs inside the target window."),
        ("future_progress", "UNSAFE", "Occurs inside the target window."),
        ("future_revised_cost", "UNSAFE", "Directly constructs future cost targets."),
        ("future_revised_end_date", "UNSAFE", "Directly constructs future schedule targets."),
        ("last_observed_month", "UNSAFE", "Uses knowledge of future dataset availability and project disappearance."),
        ("full_history_aggregates", "UNSAFE", "Aggregates after the cutoff leak future observations."),
    ]
    return pd.DataFrame(rows, columns=["feature", "classification", "reason"])


def write_history_summary(profile: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    counts = {threshold: int((profile["number_of_months"] >= threshold).sum()) for threshold in (1, 2, 3, 6, 12)}
    lines = [
        "SANKET — LABEL HISTORY PROFILE",
        "=" * 50,
        f"Input project-month rows: {len(df)}",
        f"Traceable project-month rows: {int(df['traceable'].sum())}",
        f"Identifierless rows excluded from longitudinal labels: {int((~df['traceable']).sum())}",
        f"Traceable projects: {len(profile)}",
        "",
        "Observation depth:",
        f"  Projects with exactly 1 observation: {int((profile['number_of_months'] == 1).sum())}",
        f"  Projects with 2+ observations: {counts[2]}",
        f"  Projects with 3+ observations: {counts[3]}",
        f"  Projects with 6+ observations: {counts[6]}",
        f"  Projects with 12+ observations: {counts[12]}",
        f"  Projects with at least one exact 3-month lookahead: {int((profile['eligible_cutoffs_3m'] > 0).sum())}",
        f"  Projects with at least one exact 6-month lookahead: {int((profile['eligible_cutoffs_6m'] > 0).sum())}",
        f"  Total exact 3-month cutoff observations: {int(profile['eligible_cutoffs_3m'].sum())}",
        f"  Total exact 6-month cutoff observations: {int(profile['eligible_cutoffs_6m'].sum())}",
        "",
        "Definitions:",
        "  consecutive_months is the longest uninterrupted observed run.",
        "  gaps counts missing calendar months between first and last observation.",
        "  months_before_first/last_revision measure months from first observation to the first/last detected change in effective completion target or effective cost.",
        "  Projects are linked only by source-backed P: primary or L: alternate identity keys.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readiness_report(profile: pd.DataFrame, candidates: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    def row(name: str) -> pd.Series:
        return candidates.set_index("target_name").loc[name]
    schedule = row("future_schedule_later_3m")
    cost = row("future_cost_increase_5pct_6m")
    lines = [
        "SANKET — ML LABEL READINESS REPORT",
        "=" * 50,
        "",
        "Decision",
        "--------",
        "SCHEDULE DELAY: PARTIALLY READY",
        "COST OVERRUN: PARTIALLY READY",
        "OVERALL: NOT LABEL-READY",
        "",
        "First label to validate",
        "-----------------------",
        "future_schedule_later_3m is the strongest first candidate for source audit: a later completion target first published during the next three months relative to the target known at cutoff.",
        f"It provides {int(schedule.eligible_rows)} eligible project-month cutoffs, {int(schedule.positive_rows)} positives, {int(schedule.negative_rows)} negatives, and a {schedule.positive_rate:.2%} positive rate.",
        "Its class balance is usable, but 447 later revised-target transitions occur in March 2026. That concentration must be checked against source semantics before label approval.",
        "This predicts a future published schedule deterioration, not actual completion delay. It must be described that way and is not approved for training yet.",
        "",
        "Cost candidate",
        "--------------",
        "future_cost_increase_5pct_6m is the most meaningful cost candidate because it requires material escalation rather than any small correction.",
        f"It provides {int(cost.eligible_rows)} eligible project-month cutoffs, {int(cost.positive_rows)} positives, {int(cost.negative_rows)} negatives, and a {cost.positive_rate:.2%} positive rate.",
        "Its 3.83% positive rate is materially imbalanced. Also, 77 of 145 observed month-to-month increases above 5% occur in July 2026, so that reporting boundary requires source audit.",
        "",
        "Prediction-time boundary",
        "------------------------",
        "AVAILABLE_AT_CUTOFF: the current report row, original cost/date, revised cost/date already published, current expenditure/progress, report month, and lagged values no later than the cutoff.",
        "ONLY_KNOWN_IN_FUTURE: later reports, future revised costs/dates, future expenditure/progress, whether a project remains listed, and the final observed month.",
        "",
        "Why approval is still required",
        "------------------------------",
        f"The dataset contains only 16 report months and {int((~df['traceable']).sum())} identifierless rows excluded from longitudinal construction.",
        "A published target revision is observable and temporally defensible, but it is not the same as authoritative actual delay or formally approved cost overrun.",
        "Disappearance cannot be labeled completion because projects can disappear and reappear.",
        "Source anomalies include malformed dates, expenditure decreases, original-cost changes, and one progress value over 100; candidate rows need quality filtering before training.",
        "Class balance and project-level temporal splits must be reviewed after the label definition is approved. Multiple rows from one project must never cross train/test partitions.",
        "No candidate receives RECOMMENDED status in label_candidate_analysis.csv until the synchronized March schedule and July cost changes are source-verified.",
        "",
        "External data needed for outcome-grade labels",
        "---------------------------------------------",
        "Authoritative completion/status events with event dates; formal schedule-revision approval history; formal cost-revision approval history; and a stable project identifier across all reports.",
        "",
        "No model was trained in this phase.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    profile = build_history_profile(df)
    candidates = build_candidate_analysis(df)
    leakage = build_leakage_audit()
    profile.to_csv(output_dir / "label_history_profile.csv", index=False)
    candidates.to_csv(output_dir / "label_candidate_analysis.csv", index=False, float_format="%.6f")
    leakage.to_csv(output_dir / "label_feature_leakage_audit.csv", index=False)
    write_history_summary(profile, df, output_dir / "label_history_profile.txt")
    write_readiness_report(profile, candidates, df, output_dir / "ml_label_readiness_report.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess future outcome-label feasibility")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "project_monthly.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
