# Point-in-time model split design

## Evaluation boundary

Future experiments must use forward calendar splits. A random row split is
prohibited because adjacent observations from the same project share history
and because later reports must never inform an earlier prediction.

The split unit is a prediction month, with all projects from a month assigned
to the same partition. Fit preprocessing, missing-value treatment, category
encoding, feature selection, thresholds, and calibration on the training
partition only. Apply the fitted transformations unchanged to validation and
test data.

## Proposed experiment

- Train: prediction months through December 2025.
- Validation: January through February 2026.
- Test: March 2026 onward where an exact target horizon exists.
- Keep a gap equal to the target horizon between the last training cutoff and
  the first evaluated cutoff if labels are generated operationally rather than
  from this frozen historical dataset.
- Report project-grouped sensitivity results that remove identities previously
  seen in training. This measures transfer to new projects separately from
  forecasting later observations of known projects.

Schedule and cost tables require separate ranges because their exact horizons
are three and six months. The final periods must be recomputed from available
prediction months before training. Do not force identical row eligibility.

March schedule and July cost concentrations require month-level performance
reporting and a sensitivity analysis excluding the concentrated publication
boundary. They must not be redistributed across folds.

No split or model is executed in Phase 10.
