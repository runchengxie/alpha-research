# alpha-research

`alpha-research` 是量化研究工作区的 alpha 研究包，权威 Python 包是 `alpha_research`。

本仓库维护：

- 特征工程和研究数据集
- 特征证据和单因子诊断
- 模型训练辅助
- walk-forward、CPCV、PBO 和过拟合诊断
- 信号产物与稳定性分析
- 与具体 alpha 模型绑定的目标持仓规则
- 候选晋升证据评分

完整研究流程由 `strategy-pipeline` 编排。通用组合构造和回测由 `portfolio-backtester` 维护。

## 仓库边界

本仓库可以读取外部数据资产和研究配置，不应在运行时依赖：

- `strategy_pipeline.pipeline`
- `portfolio_backtester` 内部实现
- 券商执行代码

跨仓库交接使用稳定文件契约和公开 API。

`alpha_research.backends` 提供框架无关的 `DatasetBackend`、`TrainerBackend` 和 `ExperimentRecorder` 接口。可选框架实现留在 adapter 内部，跨模块结果只保存普通元数据和本工作区定义的产物。

工作区 2.0 已删除旧共享 namespace 和 facade。新代码只使用 `alpha_research`。

## 安装和测试

```bash
uv sync --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名，都会运行完整测试集。

BasedPyright 用于发布前诊断：

```bash
scripts/dev/run_tests.sh basedpyright
```

详细范围见 [docs/testing.md](docs/testing.md)。

## Python namespace

The canonical package is `alpha_research`. Workspace 2.0 has removed the 1.x
compatibility namespace and facade; all imports, contracts, artifact types,
logger names, and environment variables are now owner-native.

## 产物

本仓库主要维护以下研究产物和约定：

```text
signals.parquet
signals.meta.json
feature evidence
model diagnostics
promotion evidence
```

修改信号产物契约时，应同步更新代码、测试、README 和相关文档。

## 文档入口

- [文档首页](docs/README.md)
- [模型选择](docs/concepts/model-selection.md)
- [模型版图](docs/concepts/model-landscape.md)
- [过拟合控制](docs/concepts/overfitting-controls.md)
- [StyleReplica 信号与组合构造](docs/concepts/style-replica.md)
- [研究模板设计](docs/playbooks/research-template-design.md)
- [测试和质量检查](docs/testing.md)

后续从 `strategy-pipeline` 迁入的 alpha 方法说明应放在本仓库。编排、配置合成、运行目录和 `targets.json` 导出继续由 `strategy-pipeline` 维护。
