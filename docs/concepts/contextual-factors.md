# Contextual Alpha 特征

本页说明 `alpha_research.contextual` 的研究边界和公开接口。该模块把已经满足时间点语义的数据表转换为宏观情境特征、公司暴露和 `context × exposure` 交互特征。

数据下载、raw snapshot、发布时间、vintage 和 current contract 由 `market-data-platform` 维护。这里不导入 TuShare、国家统计局、国家能源局、AKShare 或其他 provider SDK。

## 数据流

```text
PIT context observations
        ↓
ContextTransformSpec
        ↓
context state

PIT industry + fundamentals
        ↓
ExposureSpec
        ↓
company exposure

context state + company exposure
        ↓
ContextInteractionSpec
        ↓
stock-date contextual features
```

现有 ranker 通过 `DailyWatch20Config.features` 或其他研究配置显式选择新列。Contextual Alpha 不创建新的模型类型，也不修改既有默认 feature set。

## Context transform

`ContextTransformSpec` 定义一个稳定情境特征：

```python
from alpha_research import ContextTransformSpec, build_context_features

spec = ContextTransformSpec(
    series_id="rates.shibor_3m",
    transform="change_np",
    window=20,
    minimum_history=21,
    feature_name="ctx__shibor_3m_change20",
    staleness_limit_days=10,
)

features = build_context_features(context_pit, [spec])
```

第一版 transform：

- `level`
- `change_1p`
- `change_np`
- `yoy`
- `rolling_zscore`
- `acceleration`
- `rolling_percentile`

输入必须已经完成 revision selection，同一个 `series_id + period_end` 只能出现一个可见 vintage。该约束把数据修订选择留在 PIT 数据层，避免 transform 函数自行猜测哪次修订应该可见。

`yoy` 定义为同一自然周期上一年的算术差。它不会自动把 level 转成百分比增长。对于 M2、工业增加值、用电量等已经有官方同比系列的变量，优先直接使用来源明确的 YoY series。这样可以避免零值、负值和指数类数据出现隐式单位变化。

缺少最小历史窗口时输出缺失值，不补零。

## Company exposure

`ExposureSpec` 使用 PIT 行业 prior 和可选 PIT 财务 modifier 构造有界公司敏感度：

```python
from alpha_research import ExposureSpec, FundamentalModifier, build_company_exposures

spec = ExposureSpec(
    name="rate_sensitivity",
    industry_prior_map={
        "银行": 0.35,
        "房地产": -0.80,
        "计算机": -0.50,
    },
    fundamental_modifiers=(
        FundamentalModifier(
            field="leverage",
            direction=-1.0,
            weight=0.25,
            normalization="rank_pct",
            missing="ignore_modifier",
        ),
    ),
    unknown_industry="zero_prior",
    version="rate.v1",
)

exposures = build_company_exposures(stock_pit_frame, [spec])
```

modifier 只在同一 `trade_date` 横截面内标准化，不读取其他日期的数据。支持：

- `rank_pct`
- `zscore_clip`

缺失行为只能显式选择：

- `ignore_modifier`
- `missing_exposure`

未知行业只能显式选择：

- `zero_prior`
- `missing_exposure`

最终 exposure 按 spec 的 `clip_min`、`clip_max` 截断。默认范围为 `[-1, 1]`。

`default_context_exposure_specs()` 提供第一版粗粒度行业先验：

- `rate_sensitivity`
- `credit_sensitivity`
- `industrial_activity_sensitivity`
- `energy_input_sensitivity`
- `energy_output_sensitivity`

这些 prior 是版本化研究假设，不是客观公司属性。它们只到行业层，不维护逐股票人工权重。更细暴露需要 PIT 合法的业务构成或其他证据后再扩展。

## As-of join

`attach_context_as_of` 使用向后 as-of join。某条 context state 的最早可见时间定义为：

```text
max(available_at, source_retrieved_at)
```

因此一个后来补抓的旧网页不会因为原始发布日期很早而穿越回历史股票行。

```python
from alpha_research import attach_context_as_of

stock_with_context = attach_context_as_of(
    stock_frame,
    context_features,
    feature_names=["ctx__shibor_3m_change20"],
    series_age_limits={"ctx__shibor_3m_change20": 10},
)
```

输出同时增加：

```text
<feature>__period_end
<feature>__available_at
<feature>__source_retrieved_at
<feature>__age_days
```

超出 age limit 时只把对应特征置为缺失，股票行仍保留。

## Interaction

公司 exposure 必须按 `(trade_date, symbol, exposure_name)` 精确匹配，禁止把当前 exposure 回填到过去。

```python
from alpha_research import ContextInteractionSpec, build_context_interactions

interaction = ContextInteractionSpec(
    context_feature="ctx__shibor_3m_change20",
    exposure_name="rate_sensitivity",
    output_name="ctx__shibor_3m_change20__x__rate_sensitivity",
)

feature_frame = build_context_interactions(
    stock_frame,
    context_features,
    exposures,
    [interaction],
)
```

交互值为：

```text
context state × company exposure
```

任何一侧缺失时，interaction 保持缺失。

## Feature-set identity

所有 transform、exposure 和 interaction spec 可以生成稳定 SHA-256：

```python
from alpha_research import contextual_feature_set_id

feature_set_id = contextual_feature_set_id(
    transform_specs,
    exposure_specs,
    interaction_specs,
)
```

Mapping 的插入顺序不会改变 identity。transform window、exposure version、industry prior 或 interaction 语义变化都会改变 identity。

行业 prior 或 modifier 调整时必须升级对应 `ExposureSpec.version`。版本号用于研究审计，不能用相同版本名覆盖不同假设。

## 研究使用约束

Contextual Alpha 特征进入正式证据前仍应满足既有研究协议：

- PIT universe
- revision-safe 或明确 reconstructed 状态
- walk-forward
- CPCV / PBO
- feature ablation
- final OOS
- 交易成本和换手
- regime stability

宏观 context 对同一天所有股票相同，单独加入横截面模型只能表达 regime。`context × exposure` 才把宏观状态映射成股票之间的差异，因此研究报告应把纯 context 增量和 interaction 增量分开比较。
