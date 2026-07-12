# StyleReplica 信号与组合构造

`cstree.alpha.style_replica` 实现 StyleReplica-A80B20-v0 的信号计算和目标持仓构造。该策略采用日频规则打分，把候选证券分为 A、B 两个组合腿。

本页说明模型输入、信号产物、组合规则和当前限制。运行编排、数据目录、日报和 `targets.json` 导出由 `strategy-pipeline` 负责。通用持仓回放、成本估算和容量分析由 `portfolio-backtester` 负责。

## 模块入口

可以从 `cstree.alpha.style_replica` 导入：

```python
from cstree.alpha.style_replica import (
    StyleReplicaConfig,
    StyleReplicaPortfolioConfig,
    StyleReplicaSignalGenerator,
    build_style_replica_positions,
    compute_daily_changes,
    compute_daily_exposure,
)
```

## 信号层

`StyleReplicaSignalGenerator` 接收价格、换手率、市值、行业和证券基础信息，计算 A、B 两套分数。

A 组合腿主要使用：

- 残差波动率
- 流动性
- 市值
- 20 日和 120 日动量
- 市场 beta
- 行业动量
- 可选的分钟级成交活跃度

B 组合腿主要使用：

- 波动率收敛
- 低残差波动率
- 流动性
- 20 日和 120 日动量
- 可选的 Hermite 稳定性指标

信号输出采用长表格式，主要字段包括：

| 字段 | 含义 |
| --- | --- |
| `signal_date` | 信号日期 |
| `symbol` | 证券代码 |
| `score_a` | A 组合腿分数 |
| `score_b` | B 组合腿分数 |
| `raw_pred` | 用于统一排序的分数 |
| `leg` | 初步组合腿分类 |
| `theme` | 主题分类 |
| `industry` | 行业分类 |
| `model_version` | 模型版本 |
| `feature_set_id` | 特征集合标识 |

标准信号文件名为 `signals_style_replica.parquet`，元数据文件名为 `signals_style_replica.meta.json`。

## 配置关系

`StyleReplicaConfig` 管理信号和策略共享参数，例如槽位数量、主题配额、行业上限、重叠处理和模型版本。

`StyleReplicaPortfolioConfig` 管理持仓构造阶段使用的参数，包括缓冲区和每日替换限制。可以从信号配置生成组合配置：

```python
signal_config = StyleReplicaConfig()
portfolio_config = StyleReplicaPortfolioConfig.from_signal_config(signal_config)
```

这样可以让信号层和组合层共享槽位、权重和主题配额，同时保留组合构造特有的缓冲参数。

## A 组合腿

A 组合腿按主题分别选择证券：

1. 在每个主题内按 `score_a` 从高到低排序
2. 新持仓需要进入主题配额范围
3. 原有持仓可以在退出缓冲区内继续保留
4. 每个主题先按 `theme_quotas` 分配名额
5. 初始选择不足 `a_slots` 时，从仍未入选的主题证券中按分数补足

`a_buffer_exit_multiplier` 控制原有持仓的退出范围。例如主题配额为 10，倍率为 1.3 时，原有持仓通常在主题内排名降到约 13 名之后退出初始保留范围。

## B 组合腿

B 组合腿按 `score_b` 从高到低排序，初始选择阶段使用以下约束：

- `b_slots`：目标持仓数量
- `b_industry_cap`：单一行业数量上限
- `b_buffer_entry_rank`：新持仓进入范围
- `b_buffer_exit_rank`：原有持仓保留范围
- `b_max_daily_replacements`：每日新增数量上限

初始选择不足时，当前实现会放宽排名范围并继续补足。补足阶段不会重新应用行业上限和每日新增上限，因此调用方需要检查最终持仓是否仍满足运行要求。

## 重叠持仓和权重

`overlap_policy` 支持：

- `aggregate`：A、B 同时选中的证券合并权重，最高不超过 `max_name_weight`
- `deduplicate`：从 B 组合腿移除已经进入 A 组合腿的证券

普通槽位使用 `normal_slot_weight`。当前构造器不会自动把总权重归一化到 1。总权重取决于槽位数量、重叠数量和权重设置。

## 持仓输出

`build_style_replica_positions` 返回满足 `positions_by_rebalance.csv` 基础契约的持仓表，主要字段包括：

- `rebalance_date`
- `entry_date`
- `symbol`
- `weight`
- `side`
- `leg`
- `signal`
- `score_a`
- `score_b`
- `theme`
- `industry`
- `rank`

生成后的持仓可以交给 `portfolio-backtester` 的 `run_position_backtest` 进行通用回放。

## 辅助分析

`compute_daily_changes` 比较相邻日期持仓，输出 `new`、`exit`、`weight_change` 和 `stay`。

`compute_style_exposure_summary` 汇总单日组合腿数量、总权重、主题分布和行业分布。

`compute_daily_exposure` 对全部调仓日期逐日生成暴露摘要。

## 当前限制

- `a_capital_weight` 和 `b_capital_weight` 目前用于表达设计目标，实际权重仍由槽位权重和重叠规则决定
- `max_daily_replacements` 目前没有统一约束 A、B 两个组合腿的合计替换数量
- B 组合腿补足阶段可能突破行业上限和新增数量上限
- 组合构造属于策略专用逻辑，修改默认参数可能改变历史持仓
- 当前实现适合研究和日频目标持仓生成，不能替代逐笔撮合、真实订单状态和券商侧风控

修改组合规则时，应同步更新本页、专用行为测试和 `strategy-pipeline` 的运行配置。
