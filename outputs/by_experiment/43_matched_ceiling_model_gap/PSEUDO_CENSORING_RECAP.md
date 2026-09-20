# Existing pseudo-censoring results for the meeting

No new calibration was fitted for this comparison.

Experiment38 fitted a 2.263548 SD multiplier using known pseudo-hidden
measurements on three development trajectories, then froze it. Inflation
was applied only at observed-censored pixels; reconstruction means remained
unchanged. It improved broad/censored-region CRPS but worsened top1 CRPS
in 170/180 held-out trajectory-ceiling cases. The richer feature rule worsened
top1 CRPS in 179/180 cases. These counts were recomputed from results.csv.

| Global inflation comparison | Reference ceiling | 20% stress ceiling |
|---|---:|---:|
| Overall CRPS, before -> after (K) | 0.241 -> 0.235 | 0.661 -> 0.612 |
| Observed-censored CRPS, before -> after (K) | 1.846 -> 1.649 | 1.791 -> 1.547 |
| Top1 CRPS, before -> after (K) | 1.738 -> 1.792 | 1.896 -> 2.020 |
| Top1 coverage, before -> after | 84.5% -> 96.9% | 81.2% -> 96.9% |
| Top1 width, before -> after (K) | 7.37 -> 16.68 | 8.00 -> 18.12 |

The apparent transfer is therefore metric/region-specific. Increased coverage
does not offset all the extra width in hottest-tail CRPS. This does not prove
that any pseudo-censoring method must fail above the hardware ceiling.

Experiment39b strengthened the nested-response test to ten ceilings, five
anchors, and 2633 known examples spanning 0--2.628 K pseudo-exceedance. On
larger unseen development pseudo-exceedances, smooth-response features reduced
RMSE from 0.979 to 0.966 K (1.3%) but reduced Spearman correlation by 0.007.
The predeclared incremental-information gate failed. Neither the 30 held-out
paths nor the genuine saturated tail was tested with this failed-gate model.

Meeting wording: pseudo-censoring supplied transferable broad calibration
information in these tests, but the correction did not improve hottest-1%
distribution quality; richer severity/features and smooth nested-response
features did not resolve that limitation. Do not generalize this to an
impossibility of learning hidden magnitude from all alternative models.
