# Factor risk model boundary

`alpha-research` now owns a small, framework-neutral factor-risk primitive that turns already-computed factor return history and specific return history into a PIT-bounded risk estimate.

## Model

For exposure matrix `X`, factor covariance `F`, and specific variance diagonal `D`, the projected asset covariance is:

```text
Sigma_asset = X F X' + D
```

The first implementation intentionally does not estimate factor returns from stock returns. Factor construction is a research assumption and stays explicit in the caller. This function only validates and summarizes histories that have already been produced by the research layer.

## Inputs

- current/as-of asset × factor exposures;
- historical factor returns;
- historical asset-specific returns;
- explicit `as_of` timestamp;
- covariance shrinkage toward the factor-variance diagonal;
- minimum common observations.

All return observations must be at or before `as_of`. Future rows fail closed.

## Outputs

`FactorRiskModelEstimate` contains:

- factor covariance;
- per-asset specific risk;
- exposures;
- history start/end and observation count;
- estimator configuration;
- `asset_covariance()` projection;
- versioned receipt metadata.

## Future work

- evaluate industry/style factor sets and estimation windows;
- add estimator comparison and stability diagnostics;
- publish risk-model evidence through research artifacts;
- connect the canonical estimate to `portfolio-backtester` optimizer requests;
- compare against licensed RQData risk-model outputs when an authorized dataset is available.

Third-party risk-model objects must stay inside adapters. Promotion evidence must distinguish estimator quality from optimizer quality.
