# 研究后端与 Qlib 状态

`alpha_research.backends` 用稳定接口隔离数据集构建、模型训练和实验记录。调用方通过接口
组织研究流程，产物只记录普通 Python 元数据和工作区定义的文件契约。

## 当前实现

| 接口 | 当前实现 | 用途 |
| --- | --- | --- |
| `DatasetBackend` | `NativeDatasetBackend` / `QlibDatasetBackend` | 使用现有建模状态构建 `ResearchDataset`。Qlib 版本复用横截面标准化预处理 |
| `TrainerBackend` | `NativeTrainerBackend` / `QlibTrainerBackend` | 训练、预测和特征重要性计算。Qlib 版本用 XGBModel |
| `ExperimentRecorder` | `NullExperimentRecorder` | 返回可序列化回执，不连接外部实验服务 |

Qlib 适配器（ADR-0005）位于 `alpha_research.backends.qlib`。`pyqlib` 通过 `qlib` extra
作为独立可选依赖安装。未安装时原生路径仍可导入并通过标准门禁。Qlib 对象不写入跨仓库产物。

## Qlib 接入条件

Qlib 适配器已实现首期训练与预处理管线。完整升级（DataHandler 三个数据视图确定性语义、
PIT 股票池、处理器拟合窗口、与原生基线可复验差异报告）仍需满足：

- 未安装 Qlib 时，原生路径仍可导入并通过全部标准门禁
- DataHandler 和 Dataset 映射保留原始、推理、学习三个数据视图的确定性语义
- 测试覆盖 PIT 股票池、时间边界、处理器拟合窗口和泄漏保护
- 训练、预测、特征重要性和实验记录与原生基线形成可复验差异报告
- 标准产物不序列化 Qlib 运行时对象

完成这些条件后，文档才能把 Qlib 列为完整受支持后端。

## 当前验证入口

```bash
uv run --locked --extra dev python -m pytest tests/test_research_backends.py -q
```

该测试验证当前接口和原生实现。标准 `dev` 门禁不包含 Qlib 运行时验证。
