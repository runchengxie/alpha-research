# alpha 研究产物契约

本页记录 alpha 研究层产出的报告文件与字段契约。运行编排和 `summary.json` 顶层结构见
[`strategy-pipeline` 输出摘要](https://github.com/runchengxie/strategy-pipeline/blob/main/docs/output-summary.md)。跨仓库字段约定见
[`research-workspace` 产物契约](https://github.com/runchengxie/research-workspace/blob/main/docs/contracts.md)。

## CPCV 稳健性审计报告

`strategy alpha cpcv` 默认写入 `artifacts/reports/cpcv_<tag>/`，也可以通过 `--out` 指定目录。

输出目录结构：

```text
artifacts/reports/cpcv_<tag>/
  cpcv_splits.csv
  cpcv_path_returns.csv
  cpcv_path_metrics.csv
  cpcv_summary.json
```

`cpcv_splits.csv` 常用列：

```text
split_id,test_groups,train_groups,train_start,train_end,test_start,test_end,train_dates_raw,train_dates,test_dates,purged_train_dates,embargoed_train_dates,purge_mode,status
```

`cpcv_path_metrics.csv` 常用列：

```text
path_id,split_ids,test_start,test_end,observation_count,sharpe,total_return,ann_return,ann_vol,max_drawdown,ic_mean,ic_ir,long_short,avg_turnover,avg_cost_drag,active_total_return,information_ratio,tracking_error
```

`cpcv_summary.json` 顶层常用字段：

- `n_groups`、`test_groups`、`split_count`、`valid_split_count`
- `path_count`、`valid_path_count`
- `include_final_oos`、`excluded_final_oos_dates`
- `purge_mode`
- `sharpe_mean`、`sharpe_median`、`sharpe_p25`、`sharpe_p10`、`sharpe_min`、`positive_sharpe_ratio`
- `ic_median`、`long_short_median`、`max_drawdown_p10`、`turnover_median`、`cost_drag_median`

## CSCV / PBO / DSR 报告

`strategy alpha pbo` 默认写入 `artifacts/reports/pbo_<tag>/`。

输出目录结构：

```text
artifacts/reports/pbo_<tag>/
  pbo_splits.csv
  pbo_summary.json
```

`pbo_splits.csv` 常用列：

```text
split_id,train_groups,test_groups,selected_candidate,selected_train_sharpe,selected_oos_sharpe,selected_oos_relative_rank,logit_oos_rank,is_overfit
```

`pbo_summary.json` 顶层常用字段：

- `n_groups`、`test_groups`、`split_count`、`candidate_count`、`n_trials`
- `pbo`、`logit_oos_rank_mean`、`selected_oos_sharpe_mean`、`selected_oos_sharpe_p25`
- `selected_candidate`、`selected_sharpe`、`selected_max_drawdown`
- `dsr`、`dsr_z`、`dsr_n_trials`、`dsr_n_obs`、`dsr_expected_max_sharpe`

## 动态多信号组合报告

`strategy alpha dynamic-signal-ensemble` 输出到配置里的 `output_dir` 或 `--output-dir`。

输出文件：

```text
dynamic_scores.parquet
stock_weights.parquet
factor_weights.parquet
factor_monitor.csv
portfolio_monitor.csv
direction_calibration.csv
dynamic_signal_ensemble_summary.json
```

`dynamic_signal_ensemble_summary.json` 常用字段：

```text
schema_version,artifact_type,no_level2,rolling_metrics_shifted,date_count,signal_count,stock_score_dates,risk_penalty_enabled,correlation_threshold,avg_active_factor_count,avg_factor_turnover,avg_stock_turnover,files
```

约束：

1. 所有 rolling RankIC / ICIR / long-short / coverage / dispersion 诊断在用于当期选择前都会整体
   `shift(1)`。
2. 方向校准使用历史 RankIC 和 inertia。反向证据不足时保留上一期方向。
3. 本命令不读取 Level2、分钟线或执行系统 `targets.json`。

## 特征证据报告

`strategy alpha feature-evidence` 输出特征证据报告，包含：

- `generate-ablation`：生成配置和 `jobs.csv`
- `summarize-ablation`：汇总相对 baseline 的指标变化
- `factor-ic`：单因子独立 IC 证据

字段与协议细节见 [特征研究协议](../concepts/feature-research-protocol.md)。

## 防过拟合 sidecar 报告

`strategy alpha overfitting-diagnostics` 输出防过拟合 sidecar 报告。关键字段：

- `flag_zero_feature_importance`
- `flag_constant_prediction`
- `feature_importance_nonzero`

## 试验台账

`strategy trial-registry` 不写文件，除非传 `--output` 或 `--output-json`。来源是递归扫描
`--runs-dir` 下的 `summary.json` 与 `config.used.yml`，提取 run、数据、特征、标签、模型、组合构造
和主要结果字段，并计算 `config_hash`、`summary_hash`、`features_hash` 和 `model_params_hash`。

## 模型调参与线性搜索

`strategy alpha tune` 与 `strategy alpha sweep-linear` 输出模型调参搜索产物，默认写入
`artifacts/sweeps/<tag>/`。字段契约与 pipeline 的[输出产物说明](https://github.com/runchengxie/strategy-pipeline/blob/main/docs/output-artifacts.md)
一致。

## 信号产物

`signals.parquet` 与 `signals.meta.json` 是 alpha 向回测与编排层交付的标准信号文件，契约名称
`alpha_research.signals`，当前版本 1。字段与读写入口见 `docs/reference/signal-artifacts.md`。
