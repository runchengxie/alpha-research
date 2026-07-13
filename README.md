# alpha-research

量化研究中的 alpha 层包。

本仓库负责 `cstree.alpha.*`。它承载特征数据集、特征证据、模型训练辅助、walk-forward、CPCV、PBO、过拟合诊断、信号产物和动态信号组合工具。它也包含研究特征派生、近期表现诊断、滚动窗口诊断、候选晋升证据评分，以及与具体 alpha 模型绑定的目标持仓构造规则。

当前状态：本仓库已经从原研究仓库中拆出，并作为 `research-workspace` 的子模块锁定版本。完整研究运行仍由 `strategy-pipeline` 编排。本仓库负责 alpha 研究层和模型专用规则，要求能够在不导入 `cstree.pipeline` 和 `cstree.backtesting` 内部实现的情况下完成训练、诊断、信号产物和模型专用目标持仓输出。

## 研究后端边界

`cstree.alpha.backends` 定义框架无关的 `DatasetBackend`、`TrainerBackend` 和
`ExperimentRecorder` 端口。现有实现由 `NativeDatasetBackend`、
`NativeTrainerBackend` 和 `NullExperimentRecorder` 包装，默认行为不变。

后续 Qlib 等可选实现只能存在于 adapter 内部：跨模块结果只记录 backend id、模型 id
和普通 metadata，不得序列化第三方框架对象。`ResearchDataset` 仍是 raw / infer /
learn 数据生命周期的 canonical 内部边界，signal artifact 和公共 CLI 契约保持不变。

## 负责的文档

后续新增或迁移文档时，以下主题应优先放在本仓库：

- 特征工程、特征窗口、特征证据和单因子证据
- 模型训练、模型评估、Rank IC、final OOS、walk-forward、CPCV、PBO 和过拟合诊断
- `signals.parquet`、`signals.meta.json` 和 signal artifact 相关契约
- 与具体 alpha 模型绑定的信号打分、组合腿和目标持仓规则
- promotion gate 中属于 alpha 证据的部分

`strategy-pipeline` 文档中仍保留的研究编排页可以链接到这里，具体 alpha 方法说明应逐步迁入本仓库。通用组合回测、成本和执行模拟文档仍由 `portfolio-backtester` 维护。

## 本地检查

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev ty check
uv run --extra dev pytest
```

发布前或需要诊断类型债时，再运行：

```bash
uv run --extra dev basedpyright
```
