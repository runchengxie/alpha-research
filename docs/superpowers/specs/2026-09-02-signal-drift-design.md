# Signal Drift Design

## Goal

Add a minimal, auditable research-vs-paper/live population drift report without giving a monitoring library authority over strategy lifecycle decisions.

## Design

The alpha owner computes PSI, an empirical two-sample KS statistic, reference-standardized mean shift, and volatility ratio from finite reference/current samples. Constant reference distributions are explicit. The result is a small platform-owned receipt.

Evidently remains a possible optional monitoring/differential backend. Any future adapter must normalize to the platform result and must not replace workspace claims, invalidation conditions, evidence gates, or human review.
