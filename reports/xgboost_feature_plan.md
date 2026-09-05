# Pre-XGBoost feature plan

## CORE_FEATURE_SET

Use the 21 `CORE_SAFE` features from Phase 11. They are cutoff-known and form
the comparison contract for any later model. Keep exact-lag missingness intact;
fit imputation and encoding on training data only.

## OPTIONAL_FEATURES

Evaluate the eight reviewed cutoff-known conditional features only as a
separate ablation: state, ministry, agency, project age, current revised cost,
effective target distance, expenditure/revised-cost ratio, and current cost
revision percentage. Retain only if robustness outside the March boundary and
project-disjoint behavior do not materially deteriorate.

## DEFERRED_FEATURES

Defer `last_cost_revision_pct` because more than 95% is missing. Exclude future
revised values, complete-history aggregates, identity/name fields, final
observations, and target-window fields. Treat perfectly scaled trend variants
as redundant candidates and compare one representative per pair before any
tree boosting experiment.

Do not start XGBoost or SHAP until another forward schedule window validates
the baseline and the conditional-feature ablation is approved.
