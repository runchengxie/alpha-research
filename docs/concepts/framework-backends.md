# 研究后端与 Qlib 状态

`alpha_research.backends` 用稳定接口隔离数据集构建、模型训练和实验记录。调用方通过接口
组织研究流程，产物只记录普通 Python 元数据和工作区定义的文件契约。

## 当前实现

| 接口 | 当前实现 | 用途 |
| --- | --- | --- |
| `DatasetBackend` | `NativeDatasetBackend` | 使用现有建模状态构建 `ResearchDataset` |
| `TrainerBackend` | `NativeTrainerBackend` | 调用本包模型注册表完成训练、预测和特征重要性计算 |
| `ExperimentRecorder` | `NullExperimentRecorder` | 返回可序列化回执，不连接外部实验服务 |

当前 `main` 没有 `pyqlib` 依赖、`qlib` extra 或 `alpha_research.backends.qlib` 模块。
历史分支曾有基于旧命名空间的 Qlib 候选实现，该代码没有进入当前主线，也没有形成当前版本
可复验的等价证据。

## Qlib 接入条件

未来接入 Qlib 时需要同时满足以下条件：

- 实现位于 `alpha_research.backends.qlib` 等归属仓库内的模块
- `pyqlib` 通过独立的可选依赖组安装
- 未安装 Qlib 时，原生路径仍可导入并通过全部标准门禁
- DataHandler 和 Dataset 映射保留原始、推理、学习三个数据视图的确定性语义
- 测试覆盖 PIT 股票池、时间边界、处理器拟合窗口和泄漏保护
- 训练、预测、特征重要性和实验记录与原生基线形成可复验差异报告
- 标准产物不序列化 Qlib 运行时对象

完成这些条件后，文档才能把 Qlib 列为受支持后端。

## 当前验证入口

```bash
uv run --locked --extra dev python -m pytest tests/test_research_backends.py -q
```

该测试验证当前接口和原生实现。标准 `dev` 门禁不包含 Qlib 运行时验证。
