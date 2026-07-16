# 防过拟合机制总览

> status: active
> owner: alpha-research
> last_verified: 2026-07-16
> source_of_truth: yes
> superseded_by: n/a

本页汇总研究流程中的数据防泄漏、时间切分、特征证据、多重试验控制和候选晋升要求。具体命令见 `strategy-pipeline/docs/cli.md`，配置键见 `strategy-pipeline/docs/config.md`，输出字段见 `strategy-pipeline/docs/outputs.md`。

`alpha-research` 维护模型验证、特征证据、CPCV、PBO 和信号诊断。数据、回测、流程编排和执行边界见根目录 `docs/platform-workflow.md`。

## 主要风险

金融样本存在时间顺序、标签窗口重叠和市场状态依赖。随机切分、单条回测曲线和单个高夏普值都不足以支持晋升结论。

| 风险 | 常见表现 | 主要防线 |
| --- | --- | --- |
| 未来函数和幸存者偏差 | 用最新成分、财报或行业标签回填历史 | PIT 股票池、平台资产契约、资产健康检查 |
| 标签重叠 | 多期收益标签共享未来收益窗口 | 事件窗口样本清理、禁入期、CPCV |
| 时间依赖 | 相邻日期和同一市场状态高度相关 | 时间序列切分、前推验证、滚动训练窗口 |
| 多重试验 | 搜索大量参数后只保留赢家 | 完整试验台账、DSR、PBO、晋升门 |
| 路径依赖 | 模型只适应某一段行情 | 多路径验证、场景回测、冻结样本外观察 |
| 特征替代 | 高相关特征互相稀释重要度 | 相关性审计、消融、置换重要度、SFI |

常用缩写如下：

- PIT：按当时可见的数据还原历史。
- CPCV：组合式带清理的交叉验证，用多条样本外路径观察结果分布。
- DSR：修正后的夏普，用于降低多次试验后选择赢家产生的乐观偏差。
- PBO：过拟合概率，用于估计样本内赢家在样本外变差的风险。
- CSCV：组合式对称交叉验证，是计算 PBO 的常见方法。
- 最终样本外留出段：配置中写作 `eval.final_oos`，正式复核前不参与训练、调参和候选选择。

## 已有机制

下表中的 `strategy` 命令由 `strategy-pipeline` 提供，研究实现位于 `alpha-research`。

| 层级 | 机制 | 当前入口 | 使用方式 |
| --- | --- | --- | --- |
| 数据 | PIT 股票池和资产契约 | `strategy-pipeline/configs/presets/a_share.yml` | 固定历史可见数据和资产版本 |
| 切分 | 日期间隔和事件窗口样本清理 | `src/alpha_research/split.py`、`eval.cv_purge_mode` | 防止训练标签窗口与测试区间重叠 |
| 训练窗口 | 滚动或扩展窗口 | `model.train_window` | 检查历史长度和市场状态混杂 |
| 前推验证 | Walk-forward | `eval.walk_forward` | 检查信号能否跨时间窗口延续 |
| 最终留出 | 冻结样本外区间 | `eval.final_oos` | 为最后复核保留未参与选择的数据 |
| 多路径验证 | CPCV | `strategy alpha cpcv`、`promotion_gate.cpcv` | 观察多条样本外路径的结果分布 |
| 特征证据 | 消融、单因子 IC 和置换重要度 | `strategy alpha feature-evidence` | 检查特征及特征族的边际贡献 |
| 特征关系 | 相关性审计、SFI 和 drop-column | `strategy alpha feature-evidence` | 检查相关特征簇和替代效应 |
| 负控制 | 标签置换、错位标签、随机特征和哨兵特征 | `eval.permutation_test`、`negative-controls` | 检查无意义任务是否也产生异常结果 |
| 模型复杂度 | 浅树、抽样、正则和线性基线 | `model.params`、`model.type` | 判断复杂模型是否提供稳定增量 |
| 多重试验 | 试验台账、DSR、PBO 和 CSCV | `strategy trial-registry`、`strategy alpha pbo` | 保留完整试验集合并校正选择偏差 |
| 样本重叠 | 唯一性和顺序自助采样编号 | `strategy alpha overfitting-diagnostics uniqueness` | 衡量事件并发度和样本冗余 |
| 路径压力 | 分块重采样场景回测 | `scenario-backtest` | 检查收益路径是否脆弱 |
| 生命周期 | 候选冻结清单 | `candidate-freeze` | 固定候选运行、目标和晋升证据 |
| 晋升治理 | 晋升门 | `strategy promotion-gate` | 统一检查证据完整性和硬性拒绝项 |

PBO、负控制、样本唯一性、场景回测和候选冻结属于专项复核工具，不会随每次普通训练自动运行。研究协议负责决定哪些证据是晋升必需项。

## 正式研究流程

### 固定数据和比较口径

先记录数据资产、股票池、标签、特征集合、交易成本和组合构造。候选与基线必须共享这些口径。多日标签还要设置与事件窗口一致的样本清理和禁入期。

### 使用时间切分

普通交叉验证使用时间序列切分。将 `eval.cv_purge_mode` 设为 `event_window` 后，调参和线性搜索会继承事件窗口样本清理。候选短名单再运行 CPCV，检查结果在多条样本外路径上的分布。

最终样本外留出段要保持冻结。查看留出结果后继续改特征、参数或构造规则，会削弱这段数据的证据价值。

### 检查特征证据

正式候选至少要有特征族消融、单因子 IC、置换重要度和前推稳定性证据。高相关特征应结合相关性审计、SFI 和 drop-column 结果解释。

负控制用于识别异常流程。`strategy alpha overfitting-diagnostics negative-controls` 支持错位标签、未来特征哨兵、随机特征和随机股票池检查。

### 记录完整试验台账

`strategy trial-registry` 可以从历史运行构建试验台账。台账应记录以下内容：

- 配置和数据资产 hash
- 特征集合与标签定义
- 模型类型和超参数
- `top_k`、权重、成本和再平衡设置
- 运行状态和净收益序列
- 研究假设、失败结果和候选状态

DSR 和 PBO 依赖完整、可比的试验集合。遗漏失败试验会使统计结果过于乐观。

### 使用 DSR 和 PBO

DSR 可以作为晋升门证据。下面的 `0.95` 是研究协议示例，正式阈值要结合样本频率、持仓周期和台账完整性确定。

```yaml
promotion_gate:
  required_evidence:
    - dsr
  hard_rejections:
    min_dsr_n_trials: 2
  soft_thresholds:
    min_dsr: 0.95
```

`strategy alpha pbo` 读取可比试验的同步收益矩阵，输出 PBO、`logit lambda` 分布、样本内最佳试验的样本外表现和衰减情况。

### 检查样本重复和路径压力

`strategy alpha overfitting-diagnostics uniqueness` 根据标签事件窗口计算平均唯一性、唯一性权重和顺序自助采样编号。它提供诊断产物，训练流程是否采用这些权重要单独验证。

`strategy alpha overfitting-diagnostics scenario-backtest` 使用分块重采样重新组合收益路径，检查收益、夏普和回撤对历史顺序的敏感度。市场状态分层和流动性压力可以在专项复核中补充。

## 候选晋升证据清单

| 顺序 | 证据 | 入口或产物 |
| --- | --- | --- |
| 1 | PIT 和资产契约检查 | A 股预设、平台资产指针和验证回执 |
| 2 | 可复现基线 | `summary.json`、`config.used.yml` |
| 3 | 特征证据 | `strategy alpha feature-evidence ...` |
| 4 | 线性基础校验 | `model.type=ridge` 或 `model.type=elasticnet` |
| 5 | 主模型结果 | IC、多空收益、回测和换手指标 |
| 6 | 前推验证 | `eval.walk_forward` |
| 7 | 最终样本外留出段 | `eval.final_oos` |
| 8 | CPCV | `strategy alpha cpcv`、`promotion_gate.cpcv` |
| 9 | DSR 和试验数量 | `strategy summarize --sort-by dsr`、`promotion_gate.dsr` |
| 10 | PBO | `strategy alpha pbo` |
| 11 | 基准阶梯和暴露筛查 | `benchmark-ladder`、`exposure-screen` |
| 12 | 成本、换手和容量压力 | 回测摘要和晋升门 |
| 13 | 候选冻结与影子观察 | 候选冻结清单和执行审计 |

## 结果解释规则

- 查看 `backtest_sharpe` 时，同时检查 IC、多空收益、回撤、换手、成本、主动收益和暴露。
- 单个运行只提供研究线索。正式结论至少需要可比基线、特征证据和前推验证。
- CPCV 适合候选短名单的压力复核，无需成为每次训练的默认阶段。
- DSR 依赖完整的可比试验集合，不能单独支持晋升结论。
- 特征重要度不代表因果关系，高相关和替代效应会改变重要度分配。
- 候选进入执行流程前，需要冻结研究产物并完成模拟或影子观察。
