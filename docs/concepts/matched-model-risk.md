# Matched model and risk comparisons

These research helpers separate estimator comparisons from portfolio accounting.
They do not implement a trading strategy or certify point-in-time source data.

## Chronological fitting

`alpha_research.matched_rank_models.fit_matched_rank_models` accepts training and
inference DataFrames, feature names, a decision date and registry configurations.
Training keys are `formation_date`, `symbol`, `label_end_date`, and `target`.
Inference rows require `formation_date`, `symbol`, and the same feature columns.
All training formation dates and label maturity dates must precede the decision
strictly. Every inference row must belong to that decision date.

Missing features use training-column medians, with zero for entirely missing
training columns. Targets must be finite. The default target transform is
within-formation percentile rank with average ties; `identity` preserves numeric
risk targets. Supported estimators are Ridge, standardized Ridge, random forest,
pointwise XGBoost, and pairwise XGBoost. Pairwise requires rank targets and the
`rank:pairwise` objective; it groups training observations by formation date.

The return value contains predictions and a receipt with imputation medians,
resolved configurations, maturity bounds and a hash of the actual training rows.
Matching hashes support a same-input comparison; they do not prove the source
features were historically available. Scores are neither calibrated expected
returns nor probabilities. Callers own PIT provenance, OOS split construction,
frozen experiment specifications, weight conversion and execution-cost evaluation.

## Risk targets and gating

`alpha_research.downside_target.next_close_downside_target` measures daily downside
RMS over a fixed number of supplied exchange sessions after next-close entry.
Positive-return sessions remain in the denominator. Missing or nonpositive marks
produce an unavailable target, not a longer horizon or an assumed zero return.
`trailing_downside_rms` provides the corresponding observed-history control.

`alpha_research.forecast_skill_gate.forecast_skill_gate` compares previously
generated OOF model and control forecasts with realized targets. Entire formations
must be mature strictly before the decision. The fixed heuristic requires eight
formations, twenty matched finite observations per formation, positive average
formation-level MSE improvement and a winning-formation fraction of at least 60%.
It cannot establish that caller-supplied forecasts are genuinely out of fold.
Its boolean output is not a confidence probability or a significance test.

Always compare learned risk weights with simple exposure controls. Lower
volatility alone does not demonstrate forecasting skill, and an ex-post
volatility-matched comparison is a hindsight diagnostic rather than an executable
allocation rule. Drawdown depth, unrecovered episodes, recovery duration, turnover
and net returns remain portfolio-layer evaluations.
