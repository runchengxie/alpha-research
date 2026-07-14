# alpha-research

`alpha-research` 是量化研究工作区的 alpha 研究包，权威 Python 包是 `alpha_research`。

本仓库维护：

- 特征工程、特征证据和单因子诊断
- 模型训练辅助与稳健性评估
- walk-forward、CPCV、PBO 和过拟合诊断
- 信号产物与稳定性分析
- 模型专用目标持仓规则
- 候选晋升中的 alpha 证据

完整研究流程由 `strategy-pipeline` 编排。通用组合构造和回测由 `portfolio-backtester` 维护。

## 边界

本仓库可以读取外部数据资产和研究配置，不在运行时依赖：

- `strategy_pipeline.pipeline`
- `portfolio_backtester` 内部实现
- 券商执行代码

跨仓库交接使用公开 API 和稳定文件契约。

历史 `cstree.alpha` 路径由 `strategy-pipeline` 在工作区 1.x 期间提供兼容入口。新代码只使用 `alpha_research`。

## 安装和测试

```bash
uv sync --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名。BasedPyright 用于发布前诊断：

```bash
scripts/dev/run_tests.sh basedpyright
```

详细范围见 [docs/testing.md](docs/testing.md)。

## 主要产物

```text
signals.parquet
signals.meta.json
feature evidence
model diagnostics
promotion evidence
```

修改信号产物契约时，应同步更新代码、测试和文档。

## 文档入口

- [文档首页](docs/README.md)
- [模型选择](docs/concepts/model-selection.md)
- [模型版图](docs/concepts/model-landscape.md)
- [过拟合控制](docs/concepts/overfitting-controls.md)
- [StyleReplica](docs/concepts/style-replica.md)
- [研究模板设计](docs/playbooks/research-template-design.md)
- [测试和质量检查](docs/testing.md)
