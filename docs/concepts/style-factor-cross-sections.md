# 风格因子形成日截面

`alpha_research.style_factors` 提供纯 DataFrame 的风格因子计算。调用方可以在不改变历史窗口基准的前提下，用形成日股票池重新计算截面型因子和最终标准分。

## `formation_universe`

`compute_factors(..., formation_universe=...)` 接受只包含以下唯一键的表：

```text
trade_date
symbol
```

这个参数只表达形成日成员关系。ST、停牌、市值、上市时间、指数成分或其他经济含义由调用方负责，本包不会推断股票池规则。

执行顺序保持为：

```text
完整历史日线
→ momentum / volatility / beta 等滚动特征
→ 限定形成日
→ 对齐估值与 PIT fundamentals
→ formation_universe key filter
→ quality 等形成日截面复合因子
→ 形成日辅助因子
→ PIT 行业去均值
→ 最终截面 z-score
```

因此，过滤形成日股票池不会截断保留证券的历史 momentum、volatility 或 beta 窗口。quality 子指标的缩尾和标准化、行业去均值以及最终 z-score 会在过滤后的股票池上重新计算。

`formation_universe` 出现重复 `trade_date + symbol` 键时失败关闭。

## `standardize_factor_panel`

`standardize_factor_panel` 是独立的形成日截面变换：

```python
from alpha_research.style_factors import standardize_factor_panel

standardized = standardize_factor_panel(
    raw_factor_panel,
    factor_columns=("factor_size", "factor_value"),
)
```

默认处理顺序：

1. 每个 `trade_date` 内按 1% / 99% 缩尾。
2. 有有效 `industry_l1` 时，在形成日行业内去均值。
3. 每个 `trade_date` 内计算 z-score。

函数不负责选择股票池，不读取文件，也不访问数据平台。零方差截面输出 NaN。调用方可以显式调整日期列、行业列和缩尾分位。

未提供 `formation_universe` 时，`compute_factors` 保持原有全截面行为。
