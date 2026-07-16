# 分钟因子边界

本仓库保留两组分钟因子实现。正式研究应先确认所用模块、输入时钟和股票日契约。

## 当前权威入口

`alpha_research.minute_friend_factors` 是朋友因子挑战组的当前权威定义。它只生成 DuckDB SQL，不读取文件，也不选择数据供应商。

调用方需要先准备规范化分钟表：

- `trade_date` 和 `symbol` 标识股票日
- `bar_index` 在早盘和午盘之间连续递增，并在股票日内保持唯一
- `session_id` 标识早盘和午盘
- `open`、`close` 和 `volume` 已转换为统一单位

`friend_minute_feature_query` 输出五个量能活动因子、三个强化波动因子、两个波动中间量和分钟完整性诊断。重复 `bar_index` 会进入诊断，并使该股票日的正式因子值为空。

```python
from alpha_research.minute_friend_factors import friend_minute_feature_query

query = friend_minute_feature_query(
    relation="canonical_minute_bars",
    expected_bar_count=240,
)
```

`expected_bar_count` 只用于覆盖率诊断。它不会补齐缺失分钟，也不会统一 09:30 和 09:31 起始差异。数据生产层仍需保存供应商、数据层级、有效时间、原始分区哈希和生成凭据。

## 旧探索模块

`alpha_research.minute_factors` 是早期探索辅助模块。`compute_volume_perc` 使用简化的连续时钟分桶，没有按午休拆分交易会话。当前实现会把部分下午分钟压入最后一个桶。

该模块暂不作为正式股票日分钟资产的事实来源。修正会话映射、供应商时钟和覆盖率契约前，只用于局部探索。正式朋友因子挑战组使用 `minute_friend_factors`。

## 使用要求

进入正式发现或回测前，应至少检查：

- 股票日键是否唯一
- `bar_index` 是否连续且无重复
- 早盘和午盘映射是否符合数据源实际时间
- 缺失、无效价格和非正成交量诊断
- 供应商切换和 240 或 241 根分钟差异
- 因子可用时间是否早于信号生成时间

分钟因子只生成研究特征。样本选择、标签、模型训练、组合构造和执行成本分别由对应模块负责。
