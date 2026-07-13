# Qlib 研究后端

Qlib 在本仓库中是可选的叶子后端。它不负责市场数据发布、PIT 过滤、泄漏检查、
promotion 决策或 `signals.parquet` 契约。

需要使用时显式安装：

```bash
uv sync --extra qlib --extra dev
```

native backend 仍是默认路径。调用方需要显式选择 Qlib：

```python
from cstree.alpha.backends import QlibDatasetBackend, QlibTrainerBackend

dataset_backend = QlibDatasetBackend(source_metadata=published_adapter.dataset_metadata)
trainer_backend = QlibTrainerBackend()
```

`published_adapter` 可以来自 `market-data-platform` 的只读适配器。Alpha 只接收其中
可写入 JSON 的 provenance，不导入另一个仓库的 Qlib 适配器，也不在两个适配器之间
传递运行时对象。

## 数据生命周期映射

数据集首先经过 alpha-research 自有的构建流程，包括 PIT、缺失值、截面处理、标签处理
和样本筛选。适配器再把已经治理过的模型数据映射到 Qlib 的 `DataHandlerLP` 语义：

| Alpha 生命周期 | Qlib key | 含义 |
| --- | --- | --- |
| `raw_feature_label` | `DK_R` | 模型 processor 之前的特征与标签输入 |
| `infer_frame` | `DK_I` | 可用于推理的数据 |
| `learn_frame` | `DK_L` | 包含仅训练期标签处理的数据 |

更早的 `raw_panel` 仍只作为来源 lineage，因为它位于模型 handler 边界之前。
`QlibDatasetBackend.build()` 在返回 canonical `ResearchDataset` 前逐帧做精确 parity
检查。Qlib handler 和 dataset 不会进入 metadata。

## 模型支持范围

`QlibTrainerBackend` 将 canonical `ridge` 映射到 Qlib
`LinearModel(estimator="ridge")`，将 `xgb_regressor` 映射到 Qlib `XGBModel`。
XGBoost 的验证权仍属于 alpha-research 外层 CV。Qlib API 要求 valid segment，因此适配器
仅为评估日志复用 train segment，并关闭 early stopping。

预测结果会重新对齐调用方原始行，再交给既有 signal artifact writer。跨仓库输出仍只使用
现有信号契约。未显式映射的模型会直接报错，不会悄悄回退到 native trainer。

模型句柄只序列化 backend id、model id、model type、特征名和运行时 provenance。
进程内 Qlib 模型保存在 `QlibTrainerBackend` 的私有 registry 中；句柄只携带不落盘的字符串
lookup token，`FittedModelHandle.to_metadata()` 不包含 token 或 Qlib 对象。

native / Qlib 的一致性晋升流程、显式阈值和 artifact 重放方式见
[研究后端治理与晋升证据](../concepts/backend-governance.md)。

## 实验记录

`QlibExperimentRecorder` 把 Qlib 的 MLflow experiment manager 适配为框架无关的
`ExperimentReceipt`。调用方必须传入本地 tracking 目录或显式 tracking URI。每个
recorder 实例同一时间只允许一个 active run。

## 兼容和回滚

- 不安装 `qlib` extra 时，native 路径行为不变。
- 导入 `cstree.alpha` 或 `cstree.alpha.backends.qlib` 不会加载 Qlib。
- 回滚只需取消显式 Qlib backend 选择，artifact 和 signal schema 无需迁移。
- Qlib 对象不得写入 signal metadata、promotion evidence 或其他跨仓库契约。
