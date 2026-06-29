# alpha-research

量化研究中的 alpha 层包。

本仓库负责 `cstree.alpha.*`。它承载特征数据集、特征证据、模型训练辅助、walk-forward、
CPCV、PBO、过拟合诊断、信号产物和动态信号组合工具。它也包含研究特征派生、近期表现诊断、
滚动窗口诊断和候选晋升证据评分等能力。

当前状态：本仓库已经从原研究仓库中拆出，并作为 `research-workspace` 的子模块锁定版本。
完整研究运行仍由 `strategy-pipeline` 编排；本仓库只负责 alpha 研究层，要求能够在不导入
`cstree.pipeline` 和 `cstree.backtesting` 内部实现的情况下完成训练、诊断和信号产物输出。

## 负责的文档

后续新增或迁移文档时，以下主题应优先放在本仓库：

- 特征工程、特征窗口、特征证据和单因子证据。
- 模型训练、模型评估、Rank IC、final OOS、walk-forward、CPCV、PBO 和过拟合诊断。
- `signals.parquet`、`signals.meta.json` 和 signal artifact 相关契约。
- promotion gate 中属于 alpha 证据的部分。

`strategy-pipeline` 文档中仍保留的研究编排页可以链接到这里，具体 alpha 方法说明应逐步迁入本仓库。

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
