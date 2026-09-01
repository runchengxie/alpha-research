# 基本面状态预测

`alpha_research.fundamental_state` 支持一条研究专用路线：先预测未来企业经营状态，再把预测结果和当前估值组合成横截面研究分数。

这条路线和 DailyWatch20 的基本面 shadow 不共享标签语义。DailyWatch20 仍以未来股票收益排名为目标；本模块把未来财务状态本身作为监督学习 target。

## 研究链路

```text
canonical PIT annual fundamentals
        ↓
build_annual_fundamental_target_panel
        ↓
persistence baseline / walk-forward Ridge or XGB
        ↓
evaluate_fundamental_forecast
        ↓
build_fundamental_forecast_score
        ↓
portfolio-backtester
```

第一版只研究一年期 annual targets。长期投资 thesis 可以持续多年，但模型不需要一次预测五年或十年。

## 标签契约

输入必须已经由数据平台整理成每个 `(symbol, report_period)` 一条 canonical annual observation，并且包含合法的 `available_date`。修订版本选择、原始披露证明和 PIT provenance 继续由 `market-data-platform` 负责。

示例：

```python
from alpha_research.fundamental_state import (
    FundamentalTargetSpec,
    build_annual_fundamental_target_panel,
)

specs = (
    FundamentalTargetSpec("delta_roa_1y", "roa", "delta"),
    FundamentalTargetSpec("revenue_growth_1y", "revenue", "pct_change"),
    FundamentalTargetSpec("future_gross_margin_1y", "gross_margin", "level"),
)

targets = build_annual_fundamental_target_panel(annual_frame, specs)
```

输出保留：

- `feature_as_of_date`：当前报表合法可见的日期
- `target_report_period`：目标年度报表期
- `target_available_date`：目标报表合法可见的日期
- `fundamental_label_end_date`：与 `target_available_date` 相同，用于泄漏控制

`pct_change` 只在当前基值为正且有限时计算。零值或负值分母返回缺失值，避免把穿越零点的会计量伪装成稳定增长率。对于净利润由亏损转盈利等场景，应单独设计 level、delta 或缩放后的变化 target。

## Persistence baseline

基本面通常具有较强持续性，所以复杂模型必须和一个很难看但很重要的 baseline 比较。

```python
from alpha_research.fundamental_state import build_persistence_baseline

baseline = build_persistence_baseline(frame, specs[0])
```

- level target：预测下一期等于当前值
- delta target：预测变化为零
- pct-change target：预测增长为零

如果 ML 在 OOS 上不能稳定超过 persistence，研究不应因为模型名字更贵就继续升级复杂度。

## Walk-forward baseline runner

`run_walk_forward_fundamental_forecast` 用 expanding-window 形成真正的 OOS 预测表。每个形成期只能使用更早形成期、并且 `fundamental_label_end_date` 严格早于该测试形成期最早 `feature_as_of_date` 的训练行。

```python
from alpha_research.fundamental_state import run_walk_forward_fundamental_forecast

run = run_walk_forward_fundamental_forecast(
    targets.frame,
    target_spec=specs[0],
    feature_cols=("roa", "gross_margin", "revenue_growth_history"),
    model_configs={
        "ridge": {"type": "ridge", "params": {"alpha": 1.0}},
        "xgb": {"type": "xgb_regressor", "params": {}},
    },
    min_train_rows=500,
    min_train_periods=5,
)
```

返回表会包含 `pred_persistence`、`pred_ridge`、`pred_xgb` 等 OOS 列。模型实现继续复用 `alpha_research.modeling` 的注册表，因此不会为本研究另建训练框架。

一个形成期内公司的财报可用日可能不同。runner 使用该形成期最早的 `feature_as_of_date` 作为训练信息截止点，因此语义偏保守：宁可少用几条刚刚披露的历史标签，也不让较晚披露公司的信息污染同一形成期较早可交易的样本。

该 runner 假定输入特征已经完成研究侧缺失值、缩尾和截面标准化等预处理；它不会静默替调用方拟合 preprocessing。

## 与 ResearchModel 配合

现有 `ResearchModel` 仍可用于单目标模型训练和现有后端实验。第一版保持一个 target 一个模型，不新增 multi-task deep learning 基础设施。

```python
from alpha_research.research_model import ResearchModel

model = ResearchModel.from_config(
    {"type": "xgb_regressor", "params": {}},
    features=feature_columns,
    target_col="delta_roa_1y",
    date_col="feature_as_of_date",
)
```

## OOS 评价

```python
from alpha_research.fundamental_state import evaluate_fundamental_forecast

metrics = evaluate_fundamental_forecast(
    run.frame,
    "delta_roa_1y",
    "pred_xgb",
    directional=True,
    date_col="report_period",
)
```

返回：

- `count`
- `mae`
- `rmse`
- `rank_ic`
- 可选 `direction_accuracy`

提供 `date_col` 时，`rank_ic` 会先在每个形成期或报表期截面内计算，再对有效截面取平均。没有 `date_col` 时才计算全样本 rank correlation。不要把不同年份的数值尺度混成一个看似漂亮的截面 IC。

第一层先验证企业状态能不能预测。股票收益、Sharpe 和组合成本属于下一层问题。

## Purge 与 embargo

未来基本面标签的有效区间从 `feature_as_of_date` 延伸到 `fundamental_label_end_date`。如果这段区间与测试区间重叠，该训练 observation 必须 purge。

```python
from alpha_research.fundamental_state import purge_and_embargo_fundamental_rows

clean = purge_and_embargo_fundamental_rows(
    training_candidates,
    test_start="2020-01-01",
    test_end="2020-12-31",
    embargo_days=31,
)
```

该函数也会删除测试区间结束后 embargo buffer 内启动的 observation。纯 expanding walk-forward 通常不会把测试期之后的样本放回训练集，因此 embargo 在这种流程里往往不起作用；CPCV 或非单向时间切片更需要显式 embargo。

## 预测与估值桥接

第一版采用透明的同日横截面百分位加权，不实现完整 DCF：

```python
from alpha_research.fundamental_state import (
    FundamentalScoreSpec,
    build_fundamental_forecast_score,
)

scored = build_fundamental_forecast_score(
    forecasts,
    (
        FundamentalScoreSpec("pred_quality", weight=2.0),
        FundamentalScoreSpec("pred_growth"),
        FundamentalScoreSpec("earnings_yield"),
    ),
)
```

每个输入先在 `signal_date` 截面内转成百分位，再按权重组合。低值更优的指标使用 `higher_is_better=False`。

研究应至少比较：

```text
current fundamentals only
forecast fundamentals only
current fundamentals + valuation
forecast fundamentals + valuation
```

这样可以直接检验预测未来基本面是否真的提供了当前财务数据之外的信息。

## 文献关系

研究结构主要参考：

- Ou and Penman (1989), Financial statement analysis and the prediction of stock returns, Journal of Accounting and Economics, DOI `10.1016/0165-4101(89)90017-7`
- Abarbanell and Bushee (1997), Fundamental analysis, future earnings, and stock prices, Journal of Accounting Research, DOI `10.2307/2491464`
- Chen, Cho, Dou and Lev (2022), Predicting Future Earnings Changes Using Machine Learning and Detailed Financial Data, Journal of Accounting Research, DOI `10.1111/1475-679X.12429`
- Fama and French (2006), Profitability, investment and average returns, Journal of Financial Economics, DOI `10.1016/j.jfineco.2005.09.009`
- Novy-Marx (2013), The other side of value: The gross profitability premium, Journal of Financial Economics, DOI `10.1016/j.jfineco.2013.01.003`
- Richardson and Stock (1989), Drawing inferences from statistics based on multiyear asset returns, Journal of Financial Economics, DOI `10.1016/0304-405X(89)90086-X`

这些文献提供的是问题拆分和统计动机。当前实现不声称复现任何论文结果。
