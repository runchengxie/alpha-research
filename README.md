# alpha-research

`alpha-research` 是量化研究工作区的 alpha 研究包，权威 Python 包是 `alpha_research`。

本仓库维护：

- 特征工程、特征证据和单因子诊断
- 分钟因子 SQL 定义与股票日输出契约，不负责文件读写
- 模型训练与稳健性评估
- 滚动前向（walk-forward）、组合式带清理交叉验证（CPCV）、过拟合概率（PBO）和过拟合诊断
- 信号产物与模型专用目标持仓规则
- 候选晋升中的 alpha 证据

研究编排由 `strategy-pipeline` 负责，通用组合回测由 `portfolio-backtester` 负责。

## 边界

本仓库读取外部数据资产和研究配置，不在运行时依赖 `strategy_pipeline.pipeline`、`portfolio_backtester` 内部实现或券商执行代码。

跨仓库交接使用公开 API 和稳定文件契约。工作区 2.0 已删除旧共享命名空间和兼容入口。
新代码只使用 `alpha_research`。

风格因子形成日股票池需要显式过滤并重新计算截面时，使用 `compute_factors(..., formation_universe=...)`；独立缩尾、行业去均值和 z-score 变换使用 `standardize_factor_panel`。完整语义见[风格因子形成日截面](docs/concepts/style-factor-cross-sections.md)。

## DailyWatch20 排序目标

`DailyWatch20Ranker` 通过 `DailyWatch20Config.model_params["objective"]` 选择横截面训练思路，输出契约始终保持为按交易日计算的相对百分位和排名：

- `reg:*`（推荐基线为 `reg:squarederror`）：pointwise。逐股票拟合连续横截面标签，再按预测值排序。
- `rank:pairwise`：pairwise，也是当前默认行为。XGBoost 按交易日 query group 训练两两排序偏好。
- `rank:ndcg`：listwise。训练前把 `[0, 1]` 百分位标签离散到 `0..31` relevance grade，并按交易日 query group 优化 NDCG。

三种 objective 会生成不同的 `model_version`，因此持久化模型不会在不同训练语义之间误恢复，`feature_set_id` 保持只描述特征与标签定义。下游 `signals.parquet` 仍接收统一的相对分数，可在相同组合构造和回测假设下比较三种训练目标。

## 研究后端状态

`alpha_research.backends` 提供框架中立的 `DatasetBackend`、`TrainerBackend` 和
`ExperimentRecorder` 接口。原生实现为 `NativeDatasetBackend`、`NativeTrainerBackend`
和 `NullExperimentRecorder`。

Qlib 后端通过可选依赖接入（见 ADR-0005）。`QlibTrainerBackend` 用 Qlib 的 XGBModel
训练与预测，`QlibDatasetBackend` 复用 Qlib 预处理管线的横截面标准化。Qlib 通过
`pyqlib` extra 安装。未安装时 native 路径保持可导入、可测试、可运行。跨模块结果仍只
保存普通元数据和本工作区定义的产物，Qlib 对象不进入跨仓库产物。完整边界见
[研究后端与 Qlib 状态](docs/concepts/framework-backends.md)。

## 安装和测试

```bash
uv sync --locked --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh typecheck-release
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
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
