# Schedule label semantic definition

## Source fields

The February and March 2026 Table 6 headers call the first completion field
`Orignal/Target DoC` and the parenthesised field `Revised DoC`. In the record
layout, the first row carries approval date, original/target DoC, and original
cost. The parenthesised continuation carries start date, revised DoC, and
revised cost. `DoC` means date of commissioning in these reports.

`original_doc` is therefore the source's original/target commissioning month.
`revised_doc` is the revised commissioning month printed in that report.
The effective completion target is `revised_doc` when printed, otherwise
`original_doc`.

These fields are planned/revised targets. They are not actual completion dates,
formal approval-event timestamps, or proof that work finished late.

## Proposed target

At prediction month `t`, freeze the effective completion target printed at or
before `t`. `future_schedule_later_3m = 1` when an effective target printed in
`t+1` through `t+3` is later. Assign zero only when the same source-backed
project identity is observed at exact month `t+3` and no later target appears.
Otherwise the label is unknown.

Features may use only cutoff-or-earlier reports. Future revised dates, future
progress, future expenditure, and final observation status are excluded.

## Interpretation boundary

The target means **future published schedule-target deterioration**. It must
not be described as true actual delay without authoritative completion and
formal revision-event data.
