# AFML 方法组件

本页记录从金融机器学习方法中落地到 `alpha-research` 的研究组件。它们保持 owner-native API，不导入编排层、组合回测层或执行层实现。

## 组件

| 模块 | 用途 | 主要产物 |
| --- | --- | --- |
| `alpha_research.event_labeling` | 波动率目标、triple barrier、meta-label | `label_events.parquet` |
| `alpha_research.sample_weighting` | label concurrency、average uniqueness、return attribution、time decay、sequential bootstrap | `sample_weights.parquet`、`sample_weights.receipt.json` |
| `alpha_research.probability_calibration` | 只使用历史窗口的 isotonic / Platt 校准、概率到 bet size 映射 | OOS calibration report |
| `alpha_research.fracdiff` | fixed-width fractional differentiation 和训练窗口内的 d 选择 | feature evidence |
| `alpha_research.structural_breaks` | CUSUM、recursive residual CUSUM、SADF | regime / break evidence |

## 事件表契约

`label_triple_barrier` 的输出至少包含：

- `event_id`
- `symbol`
- `signal_date`
- `label_start`
- `label_end`
- `first_touch`
- `barrier`
- `target`
- `side`
- `realized_return`
- `side_adjusted_return`
- `label`
- `meta_label`

同一张事件表应同时用于标签、purging、embargo 和 sample weighting。不要在不同模块中各自推断一个近似的 label horizon。

## 样本权重

推荐正式候选使用：

```python
from alpha_research.sample_weighting import SampleWeightConfig, build_event_sample_weights

weights, receipt = build_event_sample_weights(
    label_events,
    bar_index=trade_calendar,
    config=SampleWeightConfig(
        mode="uniqueness_time_decay",
        uniqueness_power=1.0,
        time_decay_halflife=252,
        min_weight=0.05,
    ),
)
```

权重输出按均值 1 归一化。receipt 会记录事件 hash、有效样本量、权重集中度和 average uniqueness。

正式研究协议应在 event-window 构建不完整时拒绝晋升。探索运行可以降级，但必须显式记录 fallback 和覆盖率。

## 概率校准和 meta-label

概率校准只允许使用严格早于当前打分日的已实现结果。`expanding_probability_calibration` 不会使用同日 outcome。校准后的概率可以转成 bet size，但最终组合约束、风险目标和换手限制仍由 `portfolio-backtester` 拥有。

## Fracdiff

`d` 的选择必须位于训练窗口内，或在独立开发期冻结。不得在全样本上扫描 `d` 后按最终回测表现挑选。每个候选应同时报告：

- ADF t-stat
- 与原序列的相关性
- 有效观测数
- feature-family ablation

## Structural break

CUSUM 和 SADF 输出属于研究特征或诊断证据。它们不能直接触发实盘下单或自动换模型。对于较高频输入，应先在数据层聚合；`sadf_series` 面向日频和预聚合序列，不适合 raw order-book 消息。
