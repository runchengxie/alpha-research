# alpha-research 文档入口

本目录记录 alpha 研究层和模型专用规则。

## 推荐阅读

| 主题 | 文档 |
| --- | --- |
| 项目定位和安装 | [根目录 README](../README.md) |
| 模型选择 | [concepts/model-selection.md](concepts/model-selection.md) |
| 模型版图 | [concepts/model-landscape.md](concepts/model-landscape.md) |
| 过拟合控制 | [concepts/overfitting-controls.md](concepts/overfitting-controls.md) |
| AFML 方法组件 | [concepts/afml-methodology.md](concepts/afml-methodology.md) |
| 研究后端与 Qlib 状态 | [concepts/framework-backends.md](concepts/framework-backends.md) |
| 分钟因子边界 | [concepts/minute-factors.md](concepts/minute-factors.md) |
| StyleReplica | [concepts/style-replica.md](concepts/style-replica.md) |
| 研究模板设计 | [playbooks/research-template-design.md](playbooks/research-template-design.md) |
| 信号产物契约 | [reference/signal-artifacts.md](reference/signal-artifacts.md) |
| DailyWatch20 alpha 归属 | [ownership-migration.md](ownership-migration.md) |
| 测试和质量检查 | [testing.md](testing.md) |

## 文档边界

适合放在本仓库的主题：

- 特征工程、特征窗口和特征证据
- 单因子 IC、特征相关性和信号稳定性
- 模型训练、滚动前向（walk-forward）、组合式带清理交叉验证（CPCV）、过拟合概率（PBO）和修正夏普比（DSR）
- `signals.parquet`、`signals.meta.json` 和信号产物
- 模型专用目标持仓规则
- 候选晋升中的 alpha 证据

通用组合回测、交易成本和容量分析由 `portfolio-backtester` 维护。运行编排、CLI、配置合成、运行目录和目标文件导出由 `strategy-pipeline` 维护。

从其他仓库迁入文档时，应同时更新旧页面的跳转说明，避免形成两份活跃版本。
