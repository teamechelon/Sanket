# Baseline feature selection

## CORE_SAFE

- `sector`
- `progress_current`
- `expenditure_current`
- `original_cost_current`
- `progress_change_1m`
- `progress_change_3m`
- `progress_velocity_3m`
- `progress_acceleration_3m`
- `expenditure_change_1m`
- `expenditure_change_3m`
- `expenditure_velocity_3m`
- `cost_revision_count_to_date`
- `months_since_cost_revision`
- `schedule_revision_count_to_date`
- `months_since_schedule_revision`
- `months_since_material_progress_change`
- `expenditure_to_original_cost`
- `months_observed`
- `months_since_first_observation`
- `revised_cost_missing`
- `progress_missing`

## CONDITIONAL_CANDIDATE included

- `state`
- `ministry`
- `agency`
- `project_age_months`
- `revised_cost_current`
- `effective_target_months_from_cutoff`
- `expenditure_to_revised_cost`
- `cost_revision_pct_current`

## EXCLUDE_FROM_BASELINE

- `last_cost_revision_pct`: deferred because 95%+ values are missing.
- Future revised values, full-history aggregates, identifiers, project name, and last-observed values are excluded.

All preprocessing is fitted on training rows only. Feature sets are evaluated separately.
