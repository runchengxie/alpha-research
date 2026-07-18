# Ownership migration

`alpha_research` is the canonical owner of DailyWatch20 model lifecycle, rolling out-of-sample scoring, signal diagnostics and small-sample inference.

The legacy `strategy_pipeline.daily_watch20_*` imports remain compatibility facades. New alpha logic must be added here. The package exposes only ordinary Python objects and canonical artifacts; strategy-pipeline and broker runtimes are forbidden dependencies.
