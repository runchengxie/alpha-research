# 测试和质量检查

本页说明 `alpha-research` 的本地测试入口和检查范围。

## 安装开发依赖

```bash
uv sync --extra dev
```

## 统一入口

```bash
scripts/dev/run_tests.sh <mode> [args...]
```

| 模式 | 实际范围 |
| --- | --- |
| `all` | 完整 `pytest` 测试集 |
| `fast` | `all` 的兼容别名 |
| `unit` | `all` 的兼容别名 |
| `lint` | Ruff 代码检查 |
| `format` | Ruff 格式检查 |
| `format-all` | `format` 的兼容别名 |
| `typecheck` | `ty` 配置范围 |
| `basedpyright` | BasedPyright 配置范围 |
| `typecheck-release` | `basedpyright` 的兼容别名 |
| `maintainability` | 维护性指标和当前预算 |

`fast` 和 `unit` 都会运行完整测试集。

## 常用命令

```bash
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh maintainability
scripts/dev/run_tests.sh basedpyright
```

定点测试示例：

```bash
uv run --extra dev python -m pytest tests/test_signal_artifact.py -q
uv run --extra dev python -m pytest tests/test_cpcv.py -q
```

## 推送前检查

在 `research-workspace` 受管检出中，顶层共享 `pre-push` 会按照工作区清单运行本仓库的导入检查、Ruff、格式检查、`ty` 和完整测试集。

单独克隆本仓库时不会继承共享钩子。推送前应手动运行上方列出的 `lint`、`format`、`typecheck`、`all`、`maintainability` 和 `basedpyright`。

## 测试重点

当前测试应重点保护：

- 特征数据集和特征证据
- 模型训练与评估
- walk-forward、CPCV 和 PBO
- 信号产物字段与元数据
- 模型专用目标持仓规则
- 包导入和跨仓库依赖隔离
- 维护性指标

修复缺陷时先添加复现测试。新增能力至少覆盖正常路径和一个异常或边界路径。

## 自动化状态

当前仓库没有启用 GitHub Actions 远端测试。本地脚本、`pyproject.toml` 和工作区验证记录共同构成本地质量门禁。
