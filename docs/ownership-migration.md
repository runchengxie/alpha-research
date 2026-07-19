# DailyWatch20 alpha 归属

`alpha_research` 是 DailyWatch20 alpha 逻辑的权威实现，负责：

- 特征定义、标签和分钟因子变换
- 模型训练、恢复和生命周期判断
- 滚动样本外打分和信号诊断
- IC、小样本推断和稳健性证据
- 标准信号产物和模型专用目标持仓规则

工作区 2.0 已完成归属迁移，旧 `strategy_pipeline.daily_watch20_*` 兼容入口已经删除。
新增 alpha 逻辑应直接加入 `alpha_research` 的公开模块。

本包只向下游提供普通 Python 对象和标准产物。运行编排由 `strategy-pipeline` 负责，组合构造
由 `portfolio-backtester` 负责，策略级研究组合由 `research-apps` 负责。券商运行时不属于
本仓库依赖。
