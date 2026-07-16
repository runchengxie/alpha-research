# 信号产物契约

`signals.parquet` 是 alpha 研究层向回测和编排层交付的标准信号文件。对应元数据文件为 `signals.meta.json`，契约名称为 `alpha_research.signals`，当前版本为 1。

## 必需字段

| 字段 | 要求 |
| --- | --- |
| `signal_date` | `YYYYMMDD` 字符串 |
| `symbol` | 非空证券代码 |
| `raw_pred` | 数值型原始分数 |
| `signal_eval` | 数值型评估分数 |
| `signal_backtest` | 数值型回测分数 |
| `signal_direction` | 数值型方向 |
| `rank` | 整数排序 |
| `model_version` | 模型版本 |
| `feature_set_id` | 特征集合标识 |
| `eligible_for_backtest` | 布尔型回测资格 |
| `eligible_for_live` | 布尔型实时资格 |

## 读写入口

```python
from pathlib import Path

from alpha_research.signal_artifact import (
    read_signal_artifact,
    write_signal_artifact,
)

signal_path = Path("artifacts/run/signals.parquet")
signals, summary = write_signal_artifact(scored, signal_path)
restored = read_signal_artifact(signal_path)
```

`write_signal_artifact` 会写出契约名称、版本、摘要和调用方元数据。`read_signal_artifact` 默认执行字段校验。校验器会检查必需字段、日期格式、证券代码、分数字段、排序类型和资格字段类型。

`eligible_for_live` 只记录研究产物资格。执行审批、账户约束和下单授权仍由编排层与执行层负责。

修改字段名称、类型或含义时，应同步更新 `alpha_research.signal_artifact`、契约测试、调用方适配和本页。
