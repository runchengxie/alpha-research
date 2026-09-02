# Factor Risk Model Design

## Goal

Add a PIT-safe, framework-neutral factor-risk estimate that can later feed portfolio optimization and attribution without coupling alpha research to the portfolio owner.

## Design

The caller supplies as-of exposures, historical factor returns, and historical specific returns. The risk model computes factor covariance with optional diagonal shrinkage, asset-specific risk, and the projected asset covariance `X F X' + D`.

The implementation does not infer factor returns, optimize portfolios, or import `portfolio_backtester`. All history must be at or before the explicit `as_of` date.

## Non-goals

- no proprietary RQData data retrieval;
- no optimizer integration in this PR;
- no factor-selection policy;
- no claim that this simple covariance estimator is production-optimal;
- no external risk-model object in public artifacts.
