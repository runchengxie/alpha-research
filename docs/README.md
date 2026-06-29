# alpha-research 文档入口

本目录用于承接 alpha 研究层文档。当前仓库的详细说明仍以根目录 [README.md](../README.md) 为入口。

## 文档归属

适合放在本仓库的主题：

- 特征工程、特征窗口、特征证据、单因子 IC 和特征相关性。
- 模型训练、模型诊断、walk-forward、CPCV、PBO、DSR 和过拟合诊断。
- `signals.parquet`、`signals.meta.json`、signal artifact 摘要和信号稳定性。
- promotion gate 中和 alpha 证据直接相关的规则。

仍留在 `strategy-pipeline` 的文档应聚焦编排、CLI、配置合成、运行目录和执行目标导出。后续从 `strategy-pipeline/docs/` 迁移 alpha 主题时，先在原位置保留跳转说明，再更新相对链接和测试。

## 优先承接页面

后续优先从 `strategy-pipeline` 迁入：

- `docs/concepts/model-selection.md`
- `docs/concepts/model-landscape.md`
- `docs/concepts/overfitting-controls.md`
- `docs/playbooks/research-template-design.md`
- `docs/metrics.md` 中的 IC、CV、CPCV、PBO、DSR、feature evidence 和 signal artifact 内容
- `docs/reference/outputs/summary.md` 中的 alpha 报告字段
