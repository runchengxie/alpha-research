# StyleReplica Alpha 信号研究

`alpha_research.style_replica` 只维护 StyleReplica 的因子、研究分类、A/B 分数和标准信号产物。最终槽位分配、主题配额、持仓缓冲、替换限制、重叠处理和持仓权重已经移出 alpha owner。

运行编排和 `targets.json` 导出由 `strategy-pipeline` 负责，StyleReplica 冻结策略政策由 `strategy-app` 负责，通用目标持仓构造、回测、成本和容量由 `portfolio-backtester` 负责。

## 模块入口

```python
from alpha_research.style_replica import (
    StyleReplicaConfig,
    StyleReplicaSignalGenerator,
    generate_daily_signals,
    map_stock_to_theme,
)
```

本包不再公开 `StyleReplicaPortfolioConfig`、`build_style_replica_positions`、持仓变化或组合暴露构造器。

## 信号层

`StyleReplicaSignalGenerator` 接收价格、换手率、市值、行业和证券基础信息，计算 A、B 两套研究分数。

A 分数主要使用残差波动率、流动性、市值、20 日和 120 日动量、市场 beta、行业动量和可选分钟成交活跃信息。B 分数主要使用波动率收敛、低残差波动率、流动性、20 日和 120 日动量，以及可选 Hermite 稳定性指标。

信号输出主要字段包括：

| 字段 | 含义 |
| --- | --- |
| `signal_date` | 信号日期 |
| `symbol` | 证券代码 |
| `score_a` | A 研究分数 |
| `score_b` | B 研究分数 |
| `raw_pred` | 统一研究排序分数 |
| `signal_eval` | 评估侧分数 |
| `signal_backtest` | 下游组合回测使用的最终分数 |
| `leg` | 基于研究分类得到的候选腿标签 |
| `theme` | 研究主题分类，不包含主题配额 |
| `industry` | 行业分类 |
| `model_version` | Alpha 模型版本 |
| `feature_set_id` | 特征集合标识 |

标准信号文件名为 `signals_style_replica.parquet`，元数据文件名为 `signals_style_replica.meta.json`。

`signal_backtest` 仍然只是分数，不代表投资组合收益、成交结果或账户净值。

## 配置边界

`StyleReplicaConfig` 仅包含 alpha 研究参数和模型/特征身份。槽位数量、主题配额、行业持仓上限、buffer、replacement、overlap 和最终权重不属于该配置。

策略侧参数由 `strategy_app.style_replica.StyleReplicaPolicy` 冻结，再转换成 `portfolio-backtester` 的通用 sleeve portfolio spec。

```text
alpha_research.style_replica
    signals
      ↓
strategy_app.style_replica.StyleReplicaPolicy
      ↓
portfolio_backtester.sleeve_portfolio
    positions_by_rebalance
```

## 主题映射

`theme_map` 负责回答证券属于哪个研究主题，保留主题 key、展示标签、行业映射和概念关键词映射。

它不再定义每个主题买几只。主题 quota 是策略政策，因此归 `strategy-app`。

## 信号稳定性

Alpha 侧可以检查 Top-K 排名集合的变化，但该指标使用 `topk_membership_churn` 语义，只衡量研究信号集合变化。

它不应用组合 buffer、权重、行业持仓上限、交易可行性或交易成本。正式组合换手率由 `portfolio-backtester` 在目标持仓生成之后计算。

## 跨层边界

修改新因子、score、signal artifact、IC、recency 或其他研究诊断时在本仓处理。

修改槽位、buffer、replacement、overlap、最终权重、组合成本或容量时在 `portfolio-backtester` 处理。修改 StyleReplica A/B 身份、主题配额和冻结策略版本时在 `strategy-app` 处理。

Alpha 研究代码不得通过运行时导入调用 `portfolio_backtester` 或 `quant_execution_engine`。
