# AGENTS.md

本文件给维护者、外部贡献者和代码代理使用。它描述 `alpha-research` 的本仓协作规则，工作区层面的规则仍以顶层 `research-workspace/AGENTS.md` 为准。

## 仓库范围

本仓库负责 alpha 研究与特征工程模块（`cstree.alpha.*`），维护特征、模型、walk-forward、CPCV、PBO、过拟合诊断、feature evidence、signal artifact、alpha 诊断工具，以及与具体 alpha 模型绑定的目标持仓构造规则。

本仓库可以消费外部数据资产和研究配置，但不应在运行时导入策略编排（`cstree.pipeline`）、通用组合回测（`cstree.backtesting`）或交易执行实现。完整研究编排仍由 `strategy-pipeline` 负责。

## 常用命令

日常阻塞检查：

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev ty check
uv run --extra dev pytest
```

统一脚本入口：

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
```

发布前或诊断类型债时运行 BasedPyright：

```bash
scripts/dev/run_tests.sh basedpyright
```

GitHub Actions 中 Ruff、format、`ty check` 和维护性 ratchet 是阻塞检查，BasedPyright 是非阻塞建议项。

## GitHub 发布偏好

- 用户明确要求 commit、push 或发布本仓改动时，默认直接在 `main` 上提交并推送到 `origin/main`
- 不要默认新建 `codex/*` 分支或 draft PR，只有用户明确要求 PR、远端规则阻止直接推送、工作区存在难以拆分的混杂改动，或改动风险需要人工 review 时才走分支和 PR
- 本仓作为 `research-workspace` 子模块使用时，推送本仓后还要回到顶层仓库提交更新后的 submodule gitlink

## 文档归属

新增 alpha 研究说明时，优先放在本仓 `docs/`：

- 特征工程、特征窗口、特征证据、单因子 IC 和特征相关性
- 模型训练、模型诊断、walk-forward、CPCV、PBO、DSR 和过拟合诊断
- `signals.parquet`、`signals.meta.json`、signal artifact 摘要和信号稳定性
- 与具体 alpha 模型绑定的组合腿、缓冲、配额和目标持仓规则
- promotion gate 中和 alpha 证据直接相关的规则

留在 `strategy-pipeline` 的说明应聚焦编排、CLI、配置合成、运行目录和执行目标导出。通用组合回测、交易成本和容量分析说明由 `portfolio-backtester` 维护。

## 编辑规则

- 保持本仓可独立安装和测试，不通过 sibling source path 补齐 import
- 不提交 `.pytest_cache/`、`__pycache__/`、`artifacts/`、`outputs/`、provider 凭证或本地 `.env*`
- 修改 signal artifact 契约时，同步更新 README、docs 和对应测试
- 修改模型专用持仓规则时，同步更新模型文档、行为测试和调用方配置
- 涉及跨仓库文件约定时，同步检查顶层 `research-workspace` 的 contract 文档和 submodule gitlink
