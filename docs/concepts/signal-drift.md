# Signal distribution drift

`alpha_research.signal_drift` provides small, framework-neutral distribution diagnostics for comparing a frozen research/reference signal population with a later paper/live population.

The first report includes:

- PSI using reference quantile bins;
- empirical two-sample KS statistic;
- mean shift in reference-standard-deviation units;
- current/reference standard-deviation ratio;
- reference/current finite observation counts;
- explicit constant-reference status.

The module deliberately does not convert these metrics into a strategy lifecycle verdict. Thresholds, invalidation conditions, claims, and stop/continue decisions remain in the workspace evidence/decision-governance layer.

Evidently can later be evaluated as an optional differential/monitoring backend. Any adapter should normalize results back to the platform drift contract rather than making Evidently's project model a cross-repository dependency.
