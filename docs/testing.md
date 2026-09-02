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
| `coverage` | 完整测试集加覆盖率报告 |
| `lint` | Ruff 代码检查 |
| `format` | Ruff 格式检查 |
| `format-all` | `format` 的兼容别名 |
| `typecheck` | `ty` 配置范围 |
| `typecheck-release` | `typecheck` 的兼容别名 |
| `maintainability` | 维护性指标和当前预算 |

`fast` 和 `unit` 都会运行完整测试集。

## 常用命令

```bash
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh coverage
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh typecheck-release
scripts/dev/run_tests.sh maintainability
```

## 依赖与安全

依赖审计和静态安全扫描按仓库运行：

```bash
uv run --extra dev pip-audit
uvx deptry .
uvx bandit -q -r src -lll
```

coverage 按高风险模块逐步提高，不设置统一阈值。

定点测试示例：

```bash
uv run --extra dev python -m pytest tests/test_signal_artifact.py -q
uv run --extra dev python -m pytest tests/test_cpcv.py -q
```

## 研究后端测试范围

`tests/test_research_backends.py` 当前覆盖框架中立接口、`NativeDatasetBackend`、
`NativeTrainerBackend`、`NullExperimentRecorder`，以及产物元数据不携带运行时模型对象的
约束。

标准 `dev` 依赖没有安装 Qlib。Qlib 后端通过 `qlib` extra 安装后可运行
`QlibTrainerBackend` 的确定性训练与预测测试（见 `tests/test_backends_qlib.py`）。
接入条件和验证要求见
[研究后端与 Qlib 状态](concepts/framework-backends.md)。

## 推送前检查

在 `research-workspace` 受管检出中，顶层共享 `pre-push` 会按照工作区清单运行本仓库的导入检查、Ruff、格式检查、`ty` 和完整测试集。

单独克隆本仓库时不会继承共享钩子。推送前应手动运行上方列出的 `lint`、`format`、`typecheck`、`all` 和 `maintainability`。

`typecheck-release` 与 `typecheck` 使用相同的 `ty` 配置。`[tool.ty.src]` 已合并原发布检查范围，迁移后没有缩小类型检查覆盖。

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

本仓库是 public，GitHub Actions 在拉取请求时运行不依赖私有数据平台的 Ruff 检查、格式检查和离线测试。本地脚本、`pyproject.toml` 和工作区验证记录继续构成本地完整质量门禁，`ty` 当前仍需在本地环境中单独处理可选 Qlib 后端和既有类型债务。需要真实数据平台的分钟源目录测试通过 `market-data` extra 单独运行。
