# 特征研究协议

本页定义同一研究单元里的特征研究协议。它只用于下面这种场景：

- `universe`、`label`、`eval`、`backtest`、`market/data` 已经固定
- 你还在同一个研究矩阵单元里
- 这次要回答当前特征空间是否够用，或下一步该加哪一组特征

它的作用是统一研究口径，避免后续配置一会儿靠直觉加列、一会儿随手删列，最后没人能解释为什么
结果变化。

`benchmark-protocol.md` 的其他内容（市场 benchmark 阶梯、晋升门槛、固定分数组合层协议）分别由
`portfolio-backtester` 和 `strategy-pipeline` 维护。

## 6.1 先按特征簇组织

默认先把当前候选特征拆成几个可解释的 family，再决定删哪组、加哪组。

HK selected 当前常见 family 可以按下面理解：

- 动量：`ret_*`、`sma_*`、`sma_*_diff`、`rsi_*`、`macd_hist`
- 波动 / 流动性：`rv_*`、`volume_sma*_ratio`、`log_vol`、`vol`
- 成长：`growth_sales`、`growth_basic_earnings_per_share`、`growth_net_profit`、`growth_cash_flow_from_operating_activities`
- 质量 / 盈利：`profit_margin`、`operating_margin`、`cfo_margin`、`cfo_to_profit`
- 估值 / 规模：`market_cap`、`log_mcap`、`pe_ttm`、`pb`
- 新鲜度 / 时效性：`days_since_report`、`valuation_age_days`
- 行业 / 状态：当前更适合先做 diagnostics、`bucket_ic` 或组合约束。要升成训练特征时，单独记录假设

如果当前研究线刻意不包含某一类，例如 monthly `no_ret` 候选不再直接使用 trailing-return 动量，
这种留白本身就是研究假设。

## 6.2 默认顺序一：先做特征族消融

默认先做特征族（feature family）级消融。单个 `feature_importance` 排名容易受到模型和窗口影响，
适合放在后面辅助解释。

推荐顺序：

1. 固定研究块，只改 `features`
2. 先跑 `baseline`
3. 再按 family 做 `minus_<family>` 对照
4. 比较 `summary.json`、`feature_importance.csv`、回测表现和换手

最少建议覆盖：

- `minus_vol_liq`
- `minus_growth`
- `minus_quality`
- `minus_valuation_size`
- `minus_freshness`

如果某条线本来就没有某个 family，就不要为了凑表而硬加一个空组。

现有工具入口：

```bash
strategy alpha feature-evidence generate-ablation \
  --config strategy-pipeline/configs/experiments/sweeps/a_share__research_protocol_feature_evidence.yml

strategy alpha feature-evidence summarize-ablation \
  --config strategy-pipeline/configs/experiments/sweeps/a_share__research_protocol_feature_evidence.yml
```

`generate-ablation` 只生成配置和 `jobs.csv`。跑完这些配置后，再用 `summarize-ablation` 汇总相对
baseline 的指标变化。

## 6.3 默认顺序二：对单调变换对先做 raw/log dedup

如果当前配置在建模前已经做每期横截面 `rank` 或 `zscore`，默认优先检查这类单调变换对：

- `market_cap` / `log_mcap`
- `vol` / `log_vol`

推荐做法：

- `raw-scale dedup`：保留 raw 列，删 log 列
- `log-scale dedup`：保留 log 列，删 raw 列

先判断这两列是否本质重复，再决定是否继续动 `pb`、`cfo_to_profit`、`ret_120` 这类更可能带独立
信息的列。

## 6.4 默认顺序三：新增稀疏 PIT 因子前先做 coverage probe

对资产负债表风险、杠杆、营运资本这类 PIT 因子的推荐处理顺序：

1. 先加 coverage-safe 的小探针，例如 `operating_margin`
2. 再加 debt / structure block，例如 `debt`、`debt_to_assets`、`debt_to_equity`、`net_debt_to_assets`
3. 每一步都检查可用样本有没有明显塌缩

至少检查这些信号：

- `run.log` 里是否出现 `Feature availability collapse`
- 过滤后模型日期是否明显变少
- 是否出现 `flag_zero_feature_importance=true`
- 是否出现 `flag_constant_prediction=true`

如果新特征把历史压缩到很短窗口，就先回退到 coverage-safe 版本。

## 6.5 默认顺序四：默认不开启的 missing indicators

`features.missing.add_indicators` 目前应视为专项假设。

只有在下面两种情况才建议打开：

- 你明确在研究缺失本身是否携带信息
- 你要验证 report staleness / sparse PIT 覆盖是否应该显式入模

否则默认保持关闭，并先用：

- `cross_sectional_median`
- `days_since_report`
- `valuation_age_days`

这些更容易解释的方式处理时效和覆盖问题。

## 6.6 跑完先看什么

做完这套特征实验后，优先检查：

- `config.used.yml`
- `summary.json`
- `feature_importance.csv`
- `run.log`

重点字段：

- `feature_importance_nonzero`
- `flag_zero_feature_importance`
- `flag_constant_prediction`
- OOS `IC`
- long-only / long-short 表现
- turnover / cost drag

如果新增特征后只得到指标略升但样本覆盖更差、重要度更集中、换手更高，默认不升主线。

## 6.7 什么时候可以不按这套顺序来

下面这些场景可以跳过部分步骤，但要在配置名或研究笔记里写清楚：

- `price-only` / `PIT-only` floor benchmark
- 明确的 placebo / diagnostic sidecar
- 目标只是验证数据路线，特征增量留到后续实验
- 明确只回答单一问题，例如只看 `operating_margin` 有没有增量

简单说：

- 主线 / benchmark 配置，默认按这套顺序走
- probe / sidecar 配置，可以只做和当前问题直接相关的那一步
