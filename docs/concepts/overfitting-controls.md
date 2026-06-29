# 防过拟合机制总览

> status: active
> owner: alpha-research
> last_verified: 2026-06-29
> source_of_truth: yes
> superseded_by: n/a

本页说明：把项目里分散的数据防泄漏、验证、特征证据、回测治理和候选晋升机制整理成一张防过拟合地图。\
范围：解释每条防线解决什么问题、现在怎么使用、还有哪些边界条件；具体命令参数见 `strategy-pipeline/docs/cli.md`，配置键见 `strategy-pipeline/docs/config.md`，输出字段见 `strategy-pipeline/docs/outputs.md`。\
适合对象：准备做正式研究、复核候选策略，或者想判断一条高夏普结果是否可靠的人。\
读完后可以了解：哪些防线已经落地，哪些属于辅助诊断，以及候选策略晋升前应该准备哪些证据。\
相关页面：`strategy-pipeline/docs/concepts/benchmark-protocol.md`、`docs/concepts/model-selection.md`、`strategy-pipeline/docs/metrics.md`、`strategy-pipeline/docs/config.md`、`strategy-pipeline/docs/cli.md`、`strategy-pipeline/docs/playbooks/a-share-baseline.md`

迁移说明：本页从 `strategy-pipeline/docs/concepts/overfitting-controls.md` 迁入。`strategy-pipeline` 保留 CLI 接线和 benchmark workflow 说明。

## 基本判断

项目已经有一套比较完整的防过拟合机制，原本分散在配置、命令行工具、指标解释、基准验证协议和 A 股操作手册里。当前最强的研究防线集中在 `alpha-research` 和 `strategy-pipeline` 的组合里：

- 数据层：PIT 股票池、平台资产 current contract 资产契约，避免用当前成分或最新快照回填历史。
- 训练验证层：时间序列交叉验证、滚动或扩展训练窗口、前推验证、最终样本外留出段。
- 金融机器学习验证层：CPCV、事件窗口样本清理、禁入期。
- 特征证据层：特征族消融、单因子 IC、置换重要度、前推验证特征稳定性。
- 模型复杂度层：浅树、行抽样、列抽样、L2 正则，以及 ridge、elasticnet 弱模型基线。
- 研究治理层：实验台账、晋升门、常数预测和零重要度硬拒绝、CPCV 与 DSR 门槛、成本、换手和暴露复核、PBO、CSCV 诊断和候选冻结。

剩余边界主要在流程执行层面：PBO、负控制、场景回测、样本唯一性和候选冻结都有辅助命令。这些命令用于正式复核和晋升材料，不会在每次普通训练时自动运行。是否把某项证据设为晋升必需项，由具体研究协议配置决定。

跨仓库分工如下：`market-data-platform` 负责 current contract、PIT 股票池、PIT 基本面、历史行业和数据验证证据；`alpha-research` 负责模型验证、特征证据、CPCV、PBO 和 alpha 诊断；`portfolio-backtester` 负责回测、成本、容量、暴露和 benchmark 证据；`strategy-pipeline` 负责把这些证据编排到 run 和晋升流程里；`quant-execution-engine` 负责执行层隔离、dry-run、paper、live 分层和订单审计。

本项目的防过拟合思路采纳了多种金融机器学习研究方法，其中包括 Marcos López de Prado 的 `Advances in Financial Machine Learning`。这本书面向金融数据建模和量化研究，重点讨论金融时间序列不独立、收益标签重叠、反复试验带来的选择偏差、样本外验证和回测过拟合等问题。本文不会要求读者熟悉这本书，只把相关思想对应到项目里已经落地的机制。

## 为什么金融机器学习特别容易过拟合

这个项目面对的主要风险不同于普通监督学习：

| 风险 | 在本项目里的表现 | 需要的防线 |
| --- | --- | --- |
| 未来函数和幸存者偏差 | 用当前股票池、最新财报、当前行业标签回填历史 | PIT 股票池、平台资产契约、资产健康检查 |
| 标签重叠 | 多期收益标签共享未来收益窗口 | 样本清理、禁入期、CPCV、最终样本外留出段 |
| 样本相互依赖 | 相邻日期、同一市场状态和同一持仓周期高度相关 | TimeSeriesSplit、walk-forward、滚动训练窗口 |
| 多重测试和选择偏差 | 反复搜索参数、特征、top-k、成本后只展示赢家 | 实验台账、DSR、PBO、晋升门 |
| 历史路径依赖 | 模型只贴合某一段行情路径 | CPCV、场景回测、模拟交易 |
| 解释失真 | 高相关特征互相替代，特征重要度被稀释 | 特征消融、分组置换、相关性审计、SFI |

简单说，金融样本通常有时间顺序、市场状态和持仓周期的牵连。普通随机切分、单条回测曲线和单个高夏普值都容易让研究者过早相信结果。这个项目把数据口径、验证切分、特征证据、实验记录和晋升门串起来，目标是降低误把运气当作规律的概率。

下面几个缩写会反复出现：

- PIT：按当时可见数据还原历史，避免用后来的信息改写过去。
- CPCV：组合式带清理的交叉验证，把多段样本组合成多条样本外路径，观察结果分布。
- DSR：修正后的夏普，用来降低多次试验后只挑赢家造成的乐观偏差。
- PBO：过拟合概率，用来估计样本内赢家在样本外变差的风险。
- CSCV：组合式对称交叉验证，是计算 PBO 的一种常见做法。
- 最终样本外留出段：代码和输出里有时写作 final OOS，正式复核前不参与训练、调参和候选选择。

## 已有机制

| 层级 | 机制 | 当前状态 | 主要入口 | 作用 |
| --- | --- | --- | --- | --- |
| 数据 | PIT 股票池、current contract 资产契约 | 已有 | `configs/presets/a_share.yml`、`strategy-pipeline/docs/playbooks/a-share-baseline.md` | 降低未来成分、幸存者偏差和资产口径漂移 |
| 切分 | TimeSeriesSplit、日期间隔、事件窗口样本清理 | 已有，可配置 | `../alpha-research/src/cstree/alpha/split.py`、`eval.cv_purge_mode`、`eval.purge_days`、`eval.embargo_days` | 避免随机切分和重叠标签窗口造成泄漏 |
| 训练窗口 | 滚动或扩展训练窗口 | 已有 | `model.train_window` | 降低过久历史和行情状态混杂对训练的影响 |
| 验证 | Walk-forward 前推验证 | 已有 | `eval.walk_forward` | 检查信号是否跨时间窗口稳定 |
| 留出 | 最终样本外留出段 | 已有，需显式启用或提供替代证据 | `eval.final_oos` | 保留最后一段样本不参与训练、调参和候选选择 |
| 样本清理 | 事件窗口样本清理、禁入期 | CPCV 和普通交叉验证均可用 | `cstree alpha cpcv`、`eval.cv_purge_mode=event_window` | 移除与测试标签窗口重叠或靠得太近的训练样本 |
| 多路径验证 | CPCV | 已有，强防线 | `cstree alpha cpcv`、`promotion_gate.cpcv` | 用多条样本外路径分布替代单条前推验证结果 |
| 特征证据 | 特征族消融 | 已有 | `cstree alpha feature-evidence generate-ablation`、`summarize-ablation` | 判断某组特征是否有边际贡献 |
| 特征证据 | 单因子 IC | 已有 | `cstree alpha feature-evidence factor-ic` | 在单因子层面检查 Rank IC、Pearson IC、覆盖率和分位收益 |
| 特征证据 | 置换重要度 | 已有 | `cstree alpha feature-evidence permutation-importance` | 检查特征或特征族被打乱后的收益代理变化 |
| 特征证据 | 相关性审计、SFI、drop-column importance | 已有辅助命令 | `cstree alpha feature-evidence correlation-audit`、`sfi`、`drop-column-importance` | 检查高相关特征簇、单特征证据和移除特征族后的边际损失 |
| 稳定性 | 前推验证特征稳定性 | 已有 | walk-forward 产物、晋升门 | 检查重要度是否只在少数窗口出现 |
| 负控制 | 标签置换检验 | 已有 | `eval.permutation_test` | 验证模型是否能在打乱标签后仍产生异常表现 |
| 负控制 | 错位标签 / 随机特征 / 随机股票池 / 哨兵特征 | 已有辅助命令 | `cstree alpha overfitting-diagnostics negative-controls` | 检查无意义任务或未来哨兵特征是否也产生异常 IC |
| 模型复杂度 | 浅树、行抽样、列抽样、正则 | 已有 | `model.params` | 限制 XGBoost 复杂度和特征共适应 |
| 弱模型基线 | Ridge、elasticnet | 已有 | `model.type` | 提供低复杂度基础校验 |
| 晋升治理 | 晋升门 | 已有 | `cstree promotion-gate` | 防止候选只凭单个 summary 指标替换 baseline |
| 夏普校正 | DSR 汇总与晋升门证据 | 已有 | `cstree summarize --sort-by dsr`、`cstree alpha pbo`、`promotion_gate.dsr` | 在可比试验组内修正夏普膨胀 |
| 多重测试 | 实验台账、PBO、CSCV | 已有辅助命令 | `cstree trial-registry`、`cstree alpha pbo` | 记录完整试验集合并检查样本内赢家、样本外输家的概率 |
| 样本重叠 | 样本唯一性、sequential bootstrap ids | 已有辅助命令 | `cstree alpha overfitting-diagnostics uniqueness` | 估计标签事件并发度、average uniqueness 和可选 bootstrap event ids |
| 路径压力 | 场景回测 | 已有辅助命令 | `cstree alpha overfitting-diagnostics scenario-backtest` | 用 block bootstrap 检查收益路径脆弱性 |
| 生命周期 | Candidate freeze / paper trading manifest | 已有辅助命令 | `cstree alpha overfitting-diagnostics candidate-freeze` | 冻结候选 run、targets 和 gate report，用于 paper / shadow 期 |
| 风险复核 | 成本、换手、暴露筛查 | 已有 | `cstree backtest exposure-screen`、晋升门 | 检查收益是否来自不可执行换手或未解释暴露 |

## 金融机器学习思想与项目落点

| 思路 | 项目已有落点 | 使用边界 |
| --- | --- | --- |
| 移除会泄漏测试标签的训练样本，并设置禁入期 | `cstree alpha cpcv` 的事件窗口样本清理和禁入期；普通交叉验证可设 `eval.cv_purge_mode=event_window` | 不同标签模式下会有覆盖率差异，报告里需要查看有效路径数量 |
| 用多条样本外路径观察稳定性 | `cstree alpha cpcv`、CPCV 摘要、晋升门的 CPCV 证据 | 适合候选短名单的压力复核，不需要每次普通训练都运行 |
| 用修正后的夏普降低多次试验偏差 | `cstree summarize --sort-by dsr`、`cstree alpha pbo`、`promotion_gate.dsr` | DSR 依赖可比试验组，实验台账越完整，解释力越强 |
| 多角度检查特征是否真的有贡献 | 特征证据、特征族消融、单因子 IC、SFI、置换重要度、相关性审计、drop-column importance | 高相关特征会互相替代，读结果时要结合相关性审计和消融结果 |
| 检查样本内赢家在样本外失效的概率 | `cstree trial-registry`、`cstree alpha pbo`、晋升门和 DSR | 需要研究侧提供同步收益矩阵和完整试验集合 |
| 检查重叠标签造成的样本重复 | `date_equal`、`time_decay`、`cstree alpha overfitting-diagnostics uniqueness` | 当前用于生成诊断权重和 bootstrap event ids，训练采样是否采用需单独评估 |
| 用改造后的历史路径做压力复核 | `cstree alpha overfitting-diagnostics scenario-backtest` | 当前提供 block bootstrap 分块重采样版本，适合先做路径脆弱性检查 |
| 把研究候选安全交给执行流程 | 晋升门、snapshot、export-targets、candidate freeze manifest | paper trading 和 shadow period 的审批规则由研究协议和执行侧治理决定 |

## 正式研究流程怎么使用这些机制

### 候选进入晋升前

正式替换 baseline 前，候选策略要经过同一套复核流程。进入晋升门的候选应准备 CPCV、特征证据、成本、换手、暴露筛查，以及最终样本外留出段或正式替代说明。run 摘要和晋升门输出会区分已通过、缺证据、不可比和仅诊断，方便判断问题出在结果本身还是证据不完整。

### 记录完整实验台账

`cstree trial-registry` 可以从历史 run 构建实验台账。它至少记录：

- 配置 hash
- 数据资产 hash 或 current asset pointer
- 特征集合
- 标签定义
- 模型类型和超参数
- top-k、weighting、cost、rebalance 设置
- run 状态
- 净收益序列路径
- 研究假设或备注
- 是否进入候选短名单

完整实验台账的作用是把成功和失败的试验都纳入统计口径。这样 DSR、PBO、FDR 这类控制方法才会面对完整试验集合，避免只在幸存 run 上做复核。

### 用 DSR 修正多次试验后的夏普

DSR 已可作为证据接入晋升门。研究协议可以把它设为软门槛：

```yaml
promotion_gate:
  required_evidence:
    - dsr
  hard_rejections:
    min_dsr_n_trials: 2
  soft_thresholds:
    min_dsr: 0.95
```

`0.95` 可以作为金融机器学习风格的初始研究门槛。最终阈值应结合样本频率、持仓周期、实验台账完整性和策略用途确认。A 股研究协议示例已展示 `min_dsr` 和 `min_dsr_n_trials`。

### 用 PBO 检查样本内赢家能否延续

`cstree alpha pbo` 已能读取一组可比试验的同步收益矩阵，输出：

- PBO
- logit lambda 分布
- 样本内最佳与样本外表现对照
- 试验数量
- 入选试验的表现衰减

它回答的问题很直接：当前研究选择流程有多大概率把样本内赢家推成候选，但这个候选到了样本外表现变差。

### 让调参阶段使用更严格的标签窗口过滤

普通 `time_series_cv_ic()` 默认使用日期 gap。设为 `eval.cv_purge_mode=event_window` 后，会改用标签事件窗口样本清理。`cstree alpha tune` 和 `sweep-linear` 复用主流程，因此会继承该配置。

目标是避免调参阶段使用比最终 CPCV 更宽松的验证口径。

### 检查特征是否互相替代

`cstree alpha feature-evidence correlation-audit` 已经能输出高相关特征对和特征簇；`sfi` 与 `drop-column-importance` 用于补充单特征证据，以及移除特征族后的边际证据。当前主线证据覆盖以下问题：

- 哪些特征高度相似。
- 单个特征是否有稳定证据。
- 移除某个特征族后，候选表现是否明显变差。
- 打乱某个特征或特征族后，收益代理是否下降。

这主要处理特征替代效应：高度相关特征会互相稀释重要性，让特征重要度看起来比实际更稳定。主成分分析或正交化重要度属于更重的专项分析，当前没有列为基础晋升证据。

### 检查重叠标签带来的样本重复

`cstree alpha overfitting-diagnostics uniqueness` 已能基于标签事件窗口输出：

- average uniqueness
- uniqueness 加权样本权重
- sequential bootstrap event ids

当前 `date_equal` 和 `time_decay` 已能缓解一部分样本权重问题；uniqueness 报告会直接刻画重叠标签导致的样本冗余。它的定位是晋升前诊断证据，训练采样器是否采用这类权重需要单独评估。

### 做负控制和场景压力测试

现有标签置换检验会继续保留。`cstree alpha overfitting-diagnostics negative-controls` 已经覆盖：

- 错位标签检验。
- 未来特征哨兵检验。
- 随机特征检验。
- 随机股票池检验。

`cstree alpha overfitting-diagnostics scenario-backtest` 已提供 block bootstrap 分块重采样版本。它会重新拼接收益区块，观察候选在不同历史路径组合下的收益、夏普和回撤。更复杂的季度分块、市场状态分层、因子场景和流动性压力通常放在专项复核里，不作为基础晋升证据。

## 候选晋升证据清单

正式替换 baseline 前，建议按下面顺序核对。每一项都已经有对应入口、配置位置或产物位置。

| 顺序 | 证据 | 入口或产物 | 说明 |
| --- | --- | --- | --- |
| 1 | PIT 和 data contract 检查 | `configs/presets/a_share.yml`、平台 current contract | 先确认历史数据口径可靠 |
| 2 | Baseline run 可复现 | `summary.json`、`config.used.yml` | 候选必须和可复现 baseline 比较 |
| 3 | 特征证据 | `cstree alpha feature-evidence ...` | 覆盖特征族消融、单因子 IC、置换重要度、相关性审计 |
| 4 | 弱模型基础校验 | `model.type=ridge` 或 `model.type=elasticnet` | 检查复杂模型是否真的提供额外价值 |
| 5 | 主模型结果 | XGBoost 回归器或排序器 | 读取 IC、多空收益、回测和换手等指标 |
| 6 | Walk-forward 前推验证 | `eval.walk_forward` | 检查不同时间窗口的稳定性 |
| 7 | 最终样本外留出段 | `eval.final_oos` | 保留一段干净样本，用于最后复核 |
| 8 | CPCV | `cstree alpha cpcv`、`promotion_gate.cpcv` | 用多条样本外路径看结果分布 |
| 9 | DSR 和试验数量 | `cstree summarize --sort-by dsr`、`promotion_gate.dsr` | 校正多次试验后的夏普乐观偏差 |
| 10 | PBO | `cstree alpha pbo` | 需要足够多可比试验的同步收益矩阵 |
| 11 | 基准阶梯 | `cstree backtest benchmark-ladder` | 检查候选是否真正优于逐层基准 |
| 12 | 暴露筛查 | `cstree backtest exposure-screen` | 检查收益是否来自未解释暴露 |
| 13 | 换手、成本和容量压力 | backtest 摘要、晋升门 | 检查收益是否被交易成本或容量约束吃掉 |
| 14 | 模拟交易或实盘影子观察 | candidate freeze manifest、执行侧审计 | 研究候选进入执行前的隔离观察 |

## 读结果时的保守规则

- 读取 `backtest_sharpe` 时，同时查看 IC、多空收益、回撤、换手、成本拖累、基准主动收益和暴露。
- 单个 run 的好结果只能作为线索。形成研究结论前，至少需要前推验证、特征证据和可比 baseline。
- CPCV 适合候选短名单的压力审计，无需作为每次 run 的默认阶段。
- 最终样本外留出段应保持干净。反复查看后再改配置，会削弱它的留出意义。
- DSR 依赖完整可比试验集合，不能单独当作结论。
- 特征重要度不等于因果解释；高相关特征和替代效应会扭曲重要度。

## 当前结论

这篇文档建议长期保留。项目已经有较多防过拟合工具，集中说明后，读者可以从一张总览里看到：

- 已有防线是什么。
- 每条防线主要防什么风险。
- 哪些机制已经强制接入晋升流程。
- 哪些机制作为辅助诊断使用。
- 每类金融机器学习防过拟合思想在项目里的入口和边界。

集中说明这些机制，可以减少在 `metrics.md`、`config.md`、`benchmark-protocol.md` 和操作手册之间来回查找时产生的误用。
