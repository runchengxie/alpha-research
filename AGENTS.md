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

## 常用命令

```bash
uv sync --extra dev
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
scripts/dev/run_tests.sh basedpyright
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

大范围文档、契约或跨仓库调整使用短期分支和 PR。合并后再按需更新顶层子模块指针。
