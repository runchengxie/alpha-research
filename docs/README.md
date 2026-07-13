# alpha-research 文档入口

本目录用于承接 alpha 研究层和模型专用组合规则的文档。当前仓库的详细说明仍以根目录 [README.md](../README.md) 为入口。

## 文档归属

适合放在本仓库的主题：

- 特征工程、特征窗口、特征证据、单因子 IC 和特征相关性
- 模型训练、模型诊断、walk-forward、CPCV、PBO、DSR 和过拟合诊断
- `signals.parquet`、`signals.meta.json`、signal artifact 摘要和信号稳定性
- 与具体 alpha 模型绑定的信号到目标持仓规则
- promotion gate 中和 alpha 证据直接相关的规则

通用组合回测、交易成本和容量分析由 `portfolio-backtester` 维护。运行编排、CLI、配置合成、运行目录和执行目标导出由 `strategy-pipeline` 维护。

## 模型专用文档

- [StyleReplica 信号与组合构造](concepts/style-replica.md)

## 可选集成

- [Qlib 研究后端](integrations/qlib.md)

## 已承接页面

这些页面已经从 `strategy-pipeline` 迁入，并由本仓库维护：

- [concepts/model-selection.md](concepts/model-selection.md)
- [concepts/model-landscape.md](concepts/model-landscape.md)
- [concepts/overfitting-controls.md](concepts/overfitting-controls.md)
- [playbooks/research-template-design.md](playbooks/research-template-design.md)

## 后续优先承接内容

后续从 `strategy-pipeline` 拆分文档时，优先迁入：

- `strategy-pipeline/docs/metrics.md` 中的 IC、CV、CPCV、PBO、DSR、feature evidence 和 signal artifact 内容
- `strategy-pipeline/docs/reference/outputs/summary.md` 中的 alpha 报告字段
