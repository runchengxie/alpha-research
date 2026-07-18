# alpha-research

`alpha-research` 是量化研究工作区的 alpha 研究包，权威 Python 包是 `alpha_research`。

本仓库维护：

- 特征工程、特征证据和单因子诊断
- 分钟因子 SQL 定义与股票日输出契约，不负责文件读写
- 模型训练与稳健性评估
- walk-forward、CPCV、PBO 和过拟合诊断
- 信号产物与模型专用目标持仓规则
- 候选晋升中的 alpha 证据

研究编排由 `strategy-pipeline` 负责，通用组合回测由 `portfolio-backtester` 负责。

## 边界

本仓库读取外部数据资产和研究配置，不在运行时依赖 `strategy_pipeline.pipeline`、`portfolio_backtester` 内部实现或券商执行代码。

跨仓库交接使用公开 API 和稳定文件契约。工作区 2.0 已删除旧共享 namespace 和 facade。新代码只使用 `alpha_research`。

`alpha_research.backends` 提供框架无关的 `DatasetBackend`、`TrainerBackend` 和 `ExperimentRecorder` 接口。可选框架实现留在 adapter 内部，跨模块结果只保存普通元数据和本工作区定义的产物。

## 安装和测试

```bash
uv sync --locked --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
scripts/dev/run_tests.sh basedpyright
```

`market-data-platform` 固定到 `pyproject.toml` 和 `uv.lock` 记录的 Git commit。
独立检出时需要对应私有仓库的 GitHub 读取权限。

`fast` 和 `unit` 是 `all` 的兼容别名。详细范围见 [docs/testing.md](docs/testing.md)。

## 主要产物

```text
signals.parquet
signals.meta.json
feature evidence
model diagnostics
promotion evidence
```

信号字段和元数据约定见 [docs/reference/signal-artifacts.md](docs/reference/signal-artifacts.md)。修改契约时，应同步更新代码、测试和文档。

文档从 [docs/README.md](docs/README.md) 进入。
