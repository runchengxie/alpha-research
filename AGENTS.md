# AGENTS.md

本文件说明 `alpha-research` 的协作边界。工作区级规则见顶层 `research-workspace/AGENTS.md`。

## 仓库职责

本仓库维护 `alpha_research` 包，负责：

- 特征工程、特征窗口和特征证据
- 模型训练与模型诊断
- walk-forward、CPCV、PBO 和过拟合控制
- signal artifact 和信号稳定性
- 与具体 alpha 模型绑定的目标持仓规则
- 候选晋升中的 alpha 证据

完整研究编排由 `strategy-pipeline` 负责。通用组合回测由 `portfolio-backtester` 负责。

## 依赖边界

- 保持仓库可以独立安装和测试。
- 不通过同级仓库源码路径补齐导入。
- 不在运行时导入 `strategy_pipeline.pipeline`。
- 不依赖 `portfolio_backtester` 内部实现。
- 跨仓库交接使用公开 API 或稳定产物契约。
- 修改 signal artifact 契约时，同步检查顶层工作区文件约定。

## 常用命令

```bash
uv sync --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名。

发布前类型诊断：

```bash
scripts/dev/run_tests.sh basedpyright
```

定点测试可以直接运行：

```bash
uv run --extra dev python -m pytest tests/test_signal_artifact.py -q
```

## 文档分工

本仓库文档优先覆盖：

- 特征工程和特征证据
- 模型训练、评估和稳健性诊断
- walk-forward、CPCV、PBO、DSR 和过拟合控制
- `signals.parquet` 与 `signals.meta.json`
- 模型专用目标持仓规则
- alpha 证据和候选晋升规则

以下内容由其他仓库维护：

- 通用回测、成本和容量分析：`portfolio-backtester`
- 编排、CLI、配置和运行目录：`strategy-pipeline`
- 数据生产和当前数据契约：`market-data-platform`
- 券商执行和审计：`quant-execution-engine`

## 编辑规则

- 中文说明使用自然、直接的表达。
- 中文正文使用中文标点。
- 保留必要的命令、路径、配置键和 API 名称。
- 用户指南聚焦当前能力和使用方式。
- 历史迁移材料放在归档或 PR 中。
- 修改公开入口时补充导入测试。
- 修改模型规则时补充正常路径和边界路径测试。
- 不提交 `artifacts/`、`outputs/`、缓存、凭证或本地环境文件。

## Git

大范围文档、契约或跨仓库调整使用短期分支和 PR。修改本仓后，如需让工作区采用新版本，再更新顶层子模块指针。
