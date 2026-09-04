# 结果解读：预测质量与稳健性

本页说明 alpha 研究侧的结果如何解读。回测收益、成本、换手、容量和暴露的解读见
[`portfolio-backtester/docs/concepts/backtest-interpretation.md`](https://github.com/runchengxie/portfolio-backtester/blob/main/docs/concepts/backtest-interpretation.md)。运行编排、产物目录和
`summary.json` 顶层结构见
[`research-workspace` 产物契约](https://github.com/runchengxie/research-workspace/blob/main/docs/contracts.md) 与
[`strategy-pipeline` 输出摘要](https://github.com/runchengxie/strategy-pipeline/blob/main/docs/output-summary.md)。

## 标签与分数

标签定义回答模型到底在预测什么。相关配置主要在
[`strategy-research` 实验配置](https://github.com/runchengxie/strategy-research/blob/main/research/experiments/configs/README.md) 的
`label` 下。

常看字段：

- `target_col`：通常是 `future_return`。
- `horizon_days`：固定持仓期长度。
- `horizon_mode`：`fixed` 表示固定天数，`next_rebalance` 表示持有到下一次调仓。
- `rebalance_frequency`：使用 `next_rebalance` 时，未来收益对齐到哪个调仓频率。
- `shift_days`：信号日到实际入场日的间隔。
- `train_target_transform`：训练前是否对标签做横截面 `zscore`、`rank` 等变换。
- `train_target_group_cols`：训练标签变换的截面分组。默认按交易日，也可以按交易日加行业分组。

看任何指标前，先确认标签口径。否则看到的 IC 或回测收益，可能对应你以为之外的持仓周期。

### 分数字段

如果启用了 `eval.save_scored_artifact=true`，`eval_scored.parquet` 通常包含三类分数：

- `pred`：模型原始输出。
- `signal_eval`：评估侧使用的方向校正分数。
- `signal_backtest`：回测侧使用的最终分数。

日常排序、分位收益、Top-K 和回测解读，优先使用 `signal_eval` 或 `signal_backtest`。直接看
`pred` 容易忽略方向翻转。

### Ranker 与 Regressor

`xgb_regressor` 或线性回归的输出更像连续预测值。它能否解释成预期收益率，取决于训练标签是否
仍是原始收益率。

`xgb_ranker` 的输出是同一交易日截面内的相对排序分数。它回答的是这些股票谁更靠前，通常不适合
解释成未来涨幅。

简单规则：

- 训练标签是原始 `future_return`，且没有做尺度变换时，模型分数才有预期收益率的参考意义。
- 训练标签做过 `zscore`、`rank` 或其他标准化后，分数更适合叫 alpha score 或排序分数。
- `quantile_mean`、`long_short` 和回测结果负责把抽象分数连接到真实未来收益。
- `label.train_target_transform != none` 时，`mae`、`rmse`、`r2` 更适合作为排障指标。

## 预测质量

预测质量指标通常在 `summary.json -> eval`。

### IC

系统默认输出两类 IC：

- `ic`：Spearman Rank IC，衡量预测排序和真实收益排序是否一致。
- `pearson_ic`：Pearson IC，衡量预测幅度和真实收益是否有线性关系。

- `ic` 回答模型给出的排名是否接近未来真实收益排名。
- `pearson_ic` 回答模型分数强弱是否接近未来收益强弱。

怎么看：

- 先看 `mean`。它代表平均方向和平均强度。
- 再看 `ir`。它代表这项表现是否稳定。
- `t_stat` 和 `p_value` 只作参考。金融时间序列很难满足严格的独立同分布假设。

常见字段：

- `n`：有效交易日或截面数量。越小越容易被偶然波动影响。
- `mean`：平均值。
- `std`：时间序列标准差。越高说明表现越不稳定。
- `ir`：`mean / std`，用于粗看稳定性。
- `t_stat`：平均值相对标准误的偏离倍数。
- `p_value`：和 `t_stat` 相关的显著性参考值。

文件：

- `ic_test.csv`
- `ic_pearson_test.csv`

### 训练期对照与方向校正

训练与评估分离后，指标通常按阶段分开统计，避免用训练期表现冒充样本外表现。方向校正确认模型
分数符号与真实收益方向是否一致，防止局部样本把符号学反。

### 误差指标

`mae`、`rmse`、`r2` 更适合作为排障指标，不适合直接判断策略好坏。模型分数不是未来涨幅。

### 分位数收益

`quantile_mean`、`long_short` 把抽象分数连接到真实未来收益。它们回答分位组合在未来区间的
实际收益，是 alpha 信号质量的直接证据。

### 换手率与缓冲区

评估侧换手（`turnover_mean`、`turnover_eval.csv`）衡量信号在相邻时点的一致性，属于 alpha 信号
稳定性。组合构造的入场出场缓冲区属于 `portfolio-backtester` 的组合参数。

### 命中率与 Top-K 正收益占比

`hit_rate` 和 `topk_positive_ratio` 衡量预测方向命中率，属于 alpha 预测方向指标。

### 可选切面

`bucket_ic`、`rolling_ic`、`recency_diagnostics` 和 `permutation_test` 都是 alpha 诊断。
分桶 IC 看信号在横截面分组的单调性，滚动 IC 看时间稳定性，近期表现诊断看最近样本是否恶化。

## 最终留出期

`final_oos` 是 alpha 验证阶段，冻结样本不参与训练。内嵌的回测子块指向 portfolio 的回测解读。

## Walk-Forward 与特征重要度

Walk-Forward、CPCV、PBO、DSR 和特征重要度都属于 alpha 稳健性诊断。

### Walk-Forward

滚动前向验证固定训练窗口并在后续窗口回放，检查表现是否随时间衰减。

### CPCV

组合式带清理交叉验证在时间上做块切分，模拟实际滚动部署，避免未来信息泄漏。

### Feature Importance

特征重要度回答模型依赖哪些输入。它不等同于单因子 IC，也不等同于置换检验的重要度。

### 单因子 IC

`strategy alpha feature-evidence factor-ic` 产出单因子的独立 IC 证据，用于特征簇消融决策。

## 常见误读

- 模型分数不能直接当成未来涨幅。
- Ranker 和 Regressor 可以共用指标，但语义不同。
- 模型分数、特征重要度、单因子有效性是三件事。
- 默认 permutation test 不是特征置换重要度。
- `hit_rate` 是辅助指标。
- 分桶 IC、暴露分析和容量压力测试各看一件事。

## 推荐阅读顺序

按 pipeline 的 metrics 文档阅读顺序，先看预测质量，再回测，最后做稳健性诊断。
