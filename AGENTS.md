# AGENTS.md

本文件说明 `alpha-research` 的协作边界。工作区级规则见顶层 `research-workspace/AGENTS.md`。

## 仓库职责

本仓库维护 `alpha_research` 包，覆盖特征、模型、稳健性诊断、信号产物和模型专用目标持仓规则。

研究编排由 `strategy-pipeline` 负责。通用组合回测由 `portfolio-backtester` 负责。

## 依赖边界

- 保持仓库可以独立安装和测试。
- 不通过同级仓库源码路径补齐导入。
- 不在运行时导入 `strategy_pipeline.pipeline`。
- 不依赖 `portfolio_backtester` 内部实现。
- 跨仓库交接使用公开 API 或稳定产物契约。
- 修改 signal artifact 契约时同步检查顶层文件约定。

## 外部框架边界

`alpha_research.backends` 公开框架中立接口、原生数据集后端、原生训练后端、空实验
记录器，以及 Qlib 可选适配器（`QlibTrainerBackend` / `QlibDatasetBackend`，见 ADR-0005）。
Qlib 通过 `qlib` extra 作为独立可选依赖安装。未安装时原生路径保持可导入、可测试。

新增外部研究框架时，应把实现放在归属仓库内的适配器中，并使用独立可选依赖。原生路径
必须在未安装外部框架时保持可导入、可测试。合并前还要补齐确定性数据集等价、PIT 和泄漏
边界、训练预测、实验记录以及产物序列化测试。

## 常用命令

```bash
uv sync --locked --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh typecheck-release
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名。

## 文档范围

本仓库文档维护：

- 特征工程和特征证据
- 模型训练、评估和稳健性诊断
- walk-forward、CPCV、PBO 和 DSR
- `signals.parquet` 与 `signals.meta.json`
- 模型专用目标持仓规则
- alpha 证据和候选晋升规则

回测、编排、数据生产和券商执行说明留在对应仓库。

## 编辑规则

- 中文说明使用自然、直接的表达和中文标点。
- 保留必要的命令、路径、配置键和 API 名称。
- 用户指南聚焦当前能力和使用方式。
- 历史迁移材料放在归档或 PR 中。
- 修改公开入口时补充导入测试。
- 修改模型规则时补充正常路径和边界路径测试。
- 不提交 `artifacts/`、`outputs/`、缓存、凭证或本地环境文件。

## Git

本仓可能由多个 agent 并行开发。每个改动都必须使用独立 worktree 与功能分支，避免
多个 agent 在同一检出目录竞争同一组文件。

远端常驻分支只有 `main`。功能分支（`feat/*`、`fix/*`、`hotfix/*`、`release/*`）
只用于拉取请求流程、临时存在。每个改动遵循以下顺序：

1. 从 `origin/main` 新建 worktree 与功能分支：

   ```bash
   git fetch origin
   git worktree add <path> -b feat/<主题> origin/main
   ```

2. 在独立 worktree 内完成改动，提交前运行本地门禁。
3. 提交并推送功能分支：

   ```bash
   git push -u origin feat/<主题>
   ```

4. 用 `gh pr create` 开拉取请求，合并到 `main`。
5. 合并完成后删除功能分支并移除 worktree：

   ```bash
   git push origin --delete feat/<主题>
   git branch -d feat/<主题>
   git worktree remove <path>
   ```

作为 `research-workspace` 子模块时，本仓提交推送合并完成后，再回到顶层更新 gitlink。
同一仓库的多个 worktree 共享主工作树的 `core.hooksPath` 配置，不要在独立 worktree
内重装或改写 hook。新的并行任务必须新建 worktree，不要直接在主检出目录的 `main`
上提交改动。本仓无运行中的远端 CI，质量事实来源是本地门禁与共享 pre-push。
