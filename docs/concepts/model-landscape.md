# 模型版图与当前选择

> status: reference
> owner: alpha-research
> last_verified: 2026-06-29
> source_of_truth: yes
> superseded_by: n/a

本页解决什么：从更广的算法空间理解当前四个模型的保留依据，以及下一步什么时候值得扩模型。\
范围：四个已维护模型之间的快速选择、具体参数、CLI 用法和单次实验结果明细见相关页面。\
适合谁：想判断某类算法是否值得纳入本项目，或想解释当前四模型选择原因的读者。\
读完你会得到什么：一套按任务结构组织的模型版图、当前四模型的角色分工、以及后续扩展的优先级。\
相关页面：`docs/concepts/model-selection.md`、`strategy-pipeline/docs/concepts/benchmark-protocol.md`、`strategy-pipeline/docs/playbooks/hk-selected.md`、`strategy-pipeline/docs/archive/research/hk/notes/hk-quarterly-target-design-and-direction-20260324.md`、`strategy-pipeline/docs/archive/research/hk/notes/hk-quarterly-pit-regime-shift-202603.md`\

迁移说明：本页从 `strategy-pipeline/docs/concepts/model-landscape.md` 迁入。模型家族和算法取舍由 `alpha-research` 维护。

这个文档讨论广义模型空间和当前仓库实际维护的模型集合之间的关系。\
如果你的问题已经缩小成现有四个模型这次该选哪个，直接回到 [model-selection.md](model-selection.md)；本页只讨论模型边界、角色分工和扩展优先级。后续模型方法细节应迁到 `alpha-research`。

## 页面边界

| 问题 | 入口 |
| --- | --- |
| 在 `xgb_regressor`、`xgb_ranker`、`ridge`、`elasticnet` 中选一个 | [model-selection.md](model-selection.md) |
| 判断 Logistic、KNN、PCA、Autoencoder 等是否值得纳入主线 | 本页 |
| 配置模型参数 | `strategy-pipeline/docs/config.md` |
| 设计 benchmark | `strategy-pipeline/docs/concepts/benchmark-protocol.md` |

先说结论：

* `strategy-pipeline/docs/concepts/universe-modes.md` 讲的是股票池模式。
* 当前仓库代码真正支持的模型类型只有 `xgb_regressor`、`xgb_ranker`、`ridge`、`elasticnet`，见 `alpha_research.modeling`。
* 当前任务结构决定了最值得先回答的是五个独立问题，对应五层诊断金字塔：市场 beta（无特征）→ 线性 alpha（Ridge）→ 非线性增量（Random Forest）→ Boosting 增量（XGBoost）→ 排序增量（XGBRanker）。
* ElasticNet 在 30-100 特征规模下和 Ridge 高度重叠，诊断价值有限；Random Forest 填补了当前缺失的「非线性本身」对照。
* 当前 HK quarterly PIT + overlay 主线里，最稳的基线仍然是 `xgb_ranker + h12_w16`；最值得继续追的 challenger 是 `xgb_regressor + zscore target`。
* 现阶段更高优先级的改进，是补全诊断金字塔（加底层 baseline 和 RF 对照），其次继续收口 `target`、`signal_direction` 和 anti-drift 骨架。

## 1. 先把任务说对

这个项目的核心任务是低频截面选股：

* 数据是表格型、按时间滚动、带明显时间漂移
* 主标签通常是连续型 `future_return`
* 评估关注 `IC`、`top_k`、回测、稳定性，
* 最终动作通常是按分数做横截面排序，再取前若干只股票构造组合

以当前 HK selected 入口为例，默认口径通常是：

* `label.target_col: future_return`
* `eval.top_k: 20`
* `backtest.top_k: 20`

这会直接影响模型选择。

## 2. 我们可以考虑的主流的机器学习算法

最有用的分法，按它和当前任务的关系来分：

| 类别 | 代表算法 | 在本项目里的典型角色 |
| --- | --- | --- |
| 线性回归 | 线性回归、Ridge、Lasso、ElasticNet | 做线性基线、收缩、稀疏化 |
| 线性分类 | Logistic、Softmax Regression | 只有把任务改成二分类或多分类时才自然 |
| 邻近样本 / 原型方法 | KNN、K-means clustering | 更像教学模型、聚类辅助工具或 regime 探索工具 |
| 概率生成式分类 | Naive Bayes | 假设很强，通常不贴当前高相关表格特征 |
| 间隔最大化 | SVM、SVR、RankSVM | 理论可用，工程优先级低 |
| 树模型 | 决策树、随机森林、Bagging | 可做非线性对照，当前通常排在更靠后的位置 |
| Boosting / Ranking | XGBoost、LightGBM、CatBoost、GBDT Ranker | 最贴表格数据与截面排序任务 |
| 概率增强 Boosting | NGBoost、Quantile Regression GBDT | 自带概率输出或分位数预测，适合风控证据 |
| 集成方法 | Stacking、Blending | 合成多模型信号，提升稳健性 |
| 生存分析 | Cox PH、Random Survival Forest | 建模「barrier 触及时间」，天然贴合 Triple Barrier |
| 可解释非线性 | MARS、GAM | 特征诊断和关系可视化 |
| 图神经网络 | GCN、GAT、GraphSAGE | 利用股票间关联关系（行业、供应链等） |
| 降维 / 表征学习 | PCA、Kernel PCA、Autoencoder、VAE、t-SNE、UMAP | 更像特征工程、诊断或可视化工具，当前适合作为辅助工具 |

PS：`K-means clustering`更适合做无监督分群或 regime 辅助分析。

## 3. 各类算法在本项目里的适配度

### 3.1 线性回归家族

| 算法 | 擅长什么 | 为什么合适 / 不合适这个项目 |
| --- | --- | --- |
| 线性回归 | 最朴素的连续值预测 baseline | 可做教学级 baseline，但对共线性脆弱，通常被 `ridge` 替代 |
| Ridge | 稳定的线性回归、抗共线性 | 很合适，适合作为线性 sanity benchmark |
| Lasso | 稀疏化、变量选择 | 有一定价值，但大部分思路已被 `elasticnet` 覆盖 |
| ElasticNet | 同时做收缩和稀疏化 | 合适，适合作为稀疏线性 challenger，但稳定性通常不如 `ridge` |

线性家族的价值，在于它们能回答最基础的问题：这套特征在较弱归纳偏好下，是否已经有可重复的线性信号。

### 3.2 线性分类家族

| 算法 | 擅长什么 | 为什么当前不在主线 |
| --- | --- | --- |
| Logistic Regression | 预测是否属于正类 | 当前主标签是连续型 `future_return`，不存在明确二分类标签 |
| Softmax Regression | 预测离散桶或多类别 | 当前任务缺少天然多分类标签；强行分桶会丢失收益幅度和相对强弱信息 |

什么时候它们会变得合理：

* 任务被正式改写成是否进入未来收益前 20%
* 或者研究目标变成是否正收益、是否超过阈值

在那之前，回归或排序目标通常更自然。

### 3.3 邻近样本、聚类和简单概率模型

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| KNN | 低维、局部结构稳定、距离定义清晰的问题 | 金融表格特征一多，距离很容易失真；再叠加时间漂移，邻居关系不稳 |
| K-means clustering | 无监督分群、寻找相似样本簇 | 可用于 regime 探索、股票分组或特征压缩前分析，直接预测价值有限 |
| Naive Bayes | 便宜的分类 baseline | 条件独立假设太强，不贴当前高相关因子与财务特征结构 |

这类方法可以做辅助研究，但通常不该承担主线 alpha 预测任务。

### 3.4 SVM、决策树和树的袋装

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| SVM / SVR | 中小样本、边界较清晰的问题 | 对缩放、参数和样本规模更敏感；在这类滚动表格研究里，通常不比 GBDT 更自然 |
| 单棵决策树 | 规则探索、可解释的分裂 | 太不稳，容易把噪声学成规则 |
| Bagging / Random Forest | 非线性、交互、稳健性比单树更好 | 可以做非 boosting 对照，但本质仍是 pointwise 打分，通常不如 boosting / ranking 对题 |

随机森林的优先级在 `xgb_ranker`、`xgb_regressor` 和现有 anti-drift 探索之后。

### 3.5 Boosting 与排序模型

| 算法 | 擅长什么 | 为什么适合这个项目 |
| --- | --- | --- |
| XGBoost / GBDT Regressor | 表格数据、非线性、特征交互 | 很合适，是当前强非线性 benchmark |
| XGBRanker | 同日分组排序学习 | 非常合适，最贴近横截面排序后取 top-k 的最终动作 |
| LightGBM / CatBoost Ranker | 更大的 boosting / ranking 家族 | 理论上值得考虑，当前工程优先级低于现有主线收口 |

为什么这类模型天然更贴题：

* 当前数据是标准表格型因子 / 财务 / 量价特征
* 非线性和条件交互在这类任务里很常见
* `xgb_ranker` 能直接按 `trade_date` 分组学习同日排序，避免先回归再间接排序

当前研究页也已经给出更细的状态判断：

* 在 HK quarterly PIT + overlay 这条主线上，`xgb_ranker + h12_w16` 仍是当前最稳的主基线
* `xgb_regressor + zscore target` 是更值得继续追的 challenger
* 这说明当前最缺的是更稳的目标设计和抗漂移机制

### 3.5b 概率增强 Boosting

| 算法 | 擅长什么 | 为什么值得考虑 |
| --- | --- | --- |
| NGBoost（Natural Gradient Boosting） | 自带概率输出，预测完整分布参数 | 不需要额外的 CalibratedClassifierCV 就能给出校准概率；对不对称损失函数（如更重视极端收益）有天然支持 |
| Quantile Regression GBDT | 预测收益分布的各个分位数 | 适合风险管理场景，既能预测均值，也能预测 VaR、CVaR 等尾部指标 |

为什么当前不优先：

- NGBoost 的概率输出优势在二分类任务中最明显。当前项目主标签是连续型 `future_return`，回归 + 分位数回归更贴题
- Quantile Regression 更适合做风控辅助证据（如「策略在第 5 百分位收益下是否会崩溃」），不适合直接替代主预测模型
- 两者都可以在 protocol 稳定后作为证据补充工具引入，定位是辅助证据，不替换主线

### 3.5c 集成 Stacking / Blending

| 方法 | 擅长什么 | 为什么值得考虑 |
| --- | --- | --- |
| Stacking | 多个异构模型输出作为 meta-model 输入 | 把 Ridge、XGBoost、LightGBM 的分数合成一个更稳健的信号 |
| Blending | 类似 Stacking 但用 hold-out 集训练 meta-model | 更简单的 Stacking 变体，减少过拟合风险 |

和当前 benchmark protocol 的关系：

- 当前做法是通过 `promotion-gate` 手动选择最佳单一模型
- Stacking 可以自动合成多模型信号，可能提升稳健性
- 但 Stacking 会模糊「哪个模型在起作用」的解释，和当前 protocol 的设计理念有冲突

什么时候推进：当 `xgb_regressor`、`xgb_ranker`、`ridge` 各自都有稳定 IC 且方向一致时，Stacking 有正期望。否则可能是在合成噪音。

### 3.5d 生存分析（Survival Analysis）

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| Cox Proportional Hazards | 建模「事件发生时间」和协变量关系 | 可以把持有期建模为「barrier 被触及的时间」：是什么特征让止盈/止损更快发生 |
| Random Survival Forest | 非线性的生存分析 | 比 Cox 更灵活，但工程更重 |

和 Triple Barrier 的关系：

- Triple Barrier 本质是生存分析问题：在多长时间内、以哪种 barrier 结束
- 生存分析能直接输出「未来 20 天触及止盈的概率」这样的曲线
- 但目前项目没有 Triple Barrier Labeling，且当前的 `future_return` 是单点标签，不适合生存分析的 `(时间, 事件类型)` 输入格式

推进时机：如果未来引入 Triple Barrier 标签体系（time-series-ml 已有），生存分析会变得非常贴题。当前项目仍然建议先稳住 `future_return` 主线。

### 3.5e 可解释非线性模型

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| MARS（Multivariate Adaptive Regression Splines） | 自动发现非线性 hinge 函数 | 比 GBDT 更可解释，可能揭示特征的非线性关系 |
| GAM（Generalized Additive Model） | 每个特征的独立非线性贡献可视化 | 适合做特征诊断，观察某个特征是否在某个区间突然失效 |

定位：

- 这两个模型更适合放在「特征诊断」工具箱里，定位是诊断工具
- 例如：用 GAM 看 `log_mcap` 对 `future_return` 的贡献曲线，发现市值 > 1000 亿后信号消失
- 当前 `feature-evidence` 和 `factor_diagnostics` 已经在做类似的事
- 不必急于引入，但如果你哪天想精细化理解某个特征，它们是合适工具

### 3.5f 图神经网络（GNN）

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| GCN / GAT（Graph Convolutional / Attention Networks） | 利用股票之间的关系（行业、供应链、相关性） | 可以把「同行业股票往往一起涨跌」的关系编码进模型 |
| GraphSAGE | 归纳式图学习，支持新节点 | 适合动态变化的股票池 |

和现有数据的关系：

- 当前项目已经有行业分类（`industry`）和股票池（`universe`）
- 但缺少供应链、产业链上下游等更丰富的图结构数据
- 构建「A 股行业关联图」或「沪深 300 相关性图」需要单独的数据工程
- 一旦有了图数据，GNN 可以自然地处理「这只股票涨，同行业其他股票也会涨」的信号

推进时机：当数据层支持图结构输入后再考虑。目前更优先的是继续收口当前表格型特征路线。

### 3.6 降维、表征学习和可视化

| 方法 | 擅长什么 | 在本项目里的更合理定位 |
| --- | --- | --- |
| PCA | 线性降维、去共线性 | 可作为线性分支的辅助特征工程 |
| Kernel PCA | 非线性降维 | 可能更灵活，但计算、解释和稳定性成本更高 |
| Autoencoder | 非线性表征压缩 | 只有在特征规模、样本量和基础设施都明显升级时才值得认真引入 |
| VAE | 带生成假设的表征学习 | 研究性更强，当前项目阶段过重 |
| t-SNE / UMAP | 可视化、结构探索、regime 观察 | 更适合做研究诊断图，不适合直接塞进稳定生产训练管线 |

尤其是 `t-SNE / UMAP`，它们最自然的用途是：

* 看特征空间是否有明显分群
* 看某段时期是否发生了分布漂移
* 辅助理解 regime shift

## 4. 当前四个模型的问题和建议核心集

### 4.1 当前官方四个模型存在的问题

当前仓库官方维护的四个模型是 `ridge`、`elasticnet`、`xgb_regressor`、`xgb_ranker`。这个选择有两个结构性问题：

问题一：ElasticNet 提供的信息和 Ridge 高度重叠。

金融特征通常只有 30-100 个，L1 稀疏化的边际价值很小。在绝大多数 run 中，ElasticNet 和 Ridge 的 IC 差异在 0.01 以内，这意味着它提供的额外诊断信息很少。如果目标是特征选择，XGBoost 的 feature importance + `feature-evidence ablation` 已经做得更好。

换句话说，`ridge → elasticnet` 这一步花了两个模型的成本，但得到的信息增量几乎为零。

问题二：线性直接跳到 Boosting，跳过了「非线性本身」这个中间问题。

当前的比较链条是：

```
ridge → elasticnet（几乎无增量）→ xgb_regressor（跳过了中间环节）
```

如果 `xgb_regressor` 的 IC 远高于 `ridge`，你无法判断增益来自「树模型能捕捉非线性」还是「boosting 机制能更好利用特征」。这两个问题被混在了一起。

缺少的是非 boosting 的非线性对照。它能隔离「放弃直线改用曲线」这个决策的效果，避免把 boosting 的功劳也算进去。Random Forest 就是为此设计的。

### 4.2 推荐的五层诊断金字塔

如果让我重新设计，应该是五个诊断层，每个回答一个独立问题：

```
Layer 0: 等权持有 / 市值加权      → 市场本身有没有 beta？（无特征 baseline）
Layer 1: Ridge                   → 线性特征有没有 alpha？
Layer 2: Random Forest           → 非线性本身有没有额外增益？
Layer 3: XGBoost Regressor       → Boosting 机制有没有额外增益？
Layer 4: XGBoost Ranker          → 直接学排序目标有没有额外增益？
```

每一层的增量都是可解释的：

| 比较 | 回答的问题 |
| --- | --- |
| Layer 0 vs Layer 1 | 这套特征在线性假设下，有没有超越市场本身的信号 |
| Layer 1 vs Layer 2 | 放弃直线、改用曲线（树模型），有没有额外收益 |
| Layer 2 vs Layer 3 | 在树模型基础上加 boosting，有没有额外收益 |
| Layer 3 vs Layer 4 | 从「预测收益率」换成「直接学排序」，有没有额外收益 |

这个金字塔的好处：

- 如果 Layer 1 就打不过 Layer 0（Ridge 弱于等权持有），说明特征集本身有问题，不需要再往上层走
- 如果 Layer 2 ≈ Layer 1（RF ≈ Ridge），说明非线性没有增量，特征关系可能是线性的，或者特征本身不够好
- 如果 Layer 3 ≈ Layer 2（XGBoost ≈ RF），说明 boosting 在这个数据集上没额外帮助，可以考虑停在 RF
- 如果 Layer 4 ≈ Layer 3（Ranker ≈ Regressor），说明排序目标没有增量，回归信号已经足够支撑排序

### 4.3 四模型核心集（Layer 1-4）

从五层金字塔中，Layer 1-4 是依赖特征的核心诊断集：

| 模型 | 当前角色 | 它主要回答什么 |
| --- | --- | --- |
| `ridge` | 线性 sanity benchmark | 如果只允许线性关系，这套特征有没有基本信号 |
| `random_forest` | 非线性 sanity benchmark | 树模型的非线性本身有没有增益，用于排除 boosting 因素 |
| `xgb_regressor` | Boosting 非线性 | Boosting + 非线性 + 特征交互，是否显著超越 RF |
| `xgb_ranker` | 任务最贴题的排序主线 | 既然最终动作是排序选股，直接学同日排序会不会更稳 |

这套组合的优点是：

- 覆盖了线性、非线性、boosting、排序四个独立诊断维度
- 每个模型的角色都清楚，每两个相邻模型之间的增量恰好回答一个问题
- 不会把「非线性增益」和「boosting 增益」混在一起

### 4.4 ElasticNet 的去留

ElasticNet 在这里的定位变为可选的历史参考：

- 当特征数量膨胀到 200+ 且有大量高度相关特征时，它的稀疏化能力才有独立诊断价值
- 当前的 30-100 个特征规模下，保留 ElasticNet 的边际信息增量几乎为零
- 如果仓库已有的 history run 大量依赖 ElasticNet 配置，可以保留但不再作为核心诊断层的默认推荐

### 4.5 和当前研究结论的关系

新框架和当前结论并不冲突，只是诊断精度更高了：

- `xgb_ranker h12_w16` 仍是当前主基线，对应 Layer 4
- `xgb_regressor + zscore target` 仍然值得继续追，对应 Layer 3
- `ridge` 仍然是必要的 sanity check，对应 Layer 1
- 新增的 `random_forest` 填补了 Layer 1 到 Layer 3 之间的诊断空白
- `elasticnet` 仍然保留在代码中，但不再作为默认推荐的核心诊断模型

## 5. 哪些方向值得继续推进，什么时候推进

| 方向 | 为什么值得做 | 什么时候推进最合适 |
| --- | --- | --- |
| `target` 设计 | 当前研究已支持 `zscore > rank > raw` 的 regressor 路线排序 | 现在就该持续推进，它已经是正式副线 |
| `signal_direction` 与 anti-drift | 研究里已观察到明显阶段性方向切换和 regime shift | 现在就该推进，优先级高于扩充模型集合 |
| 增加一个额外的 GBDT challenger | 可验证效果来自 XGBoost 特有实现，还是 boosting / ranking 家族普遍有效 | 只有当当前 protocol 已稳定、维护成本可接受时再做，首选可考虑 LightGBM |
| 增加随机森林对照 | 可帮助拆分树模型本身与 boosting / ranking objective 带来的效果，填补 Layer 1 到 Layer 3 之间的诊断空白 | 现在就可以推进；它只增加一个对照维度，不替代现有模型 |
| 增加最底层 baseline（等权持有 / 市值加权） | 如果 Ridge 打不过无特征 baseline，说明特征集本身有问题 | 现在就可以推进，优先级高于扩充模型集合 |
| NGBoost / Quantile Regression | 自带概率输出，适合做风控辅助证据 | protocol 稳定后，作为证据补充工具引入 |
| Stacking / Blending | 合成多模型信号，可能提升稳健性 | 当多个模型各自有稳定 IC 且方向一致时 |
| 生存分析（Cox PH） | 天然贴合 Triple Barrier 标签体系 | 如果引入 Triple Barrier 标签后 |
| MARS / GAM | 特征非线性关系诊断 | 想精细化理解某个特征时，作为诊断工具 |
| GNN | 利用行业/供应链图关系 | 当数据层支持图结构输入后 |
| 分类分支 | 如果决策问题改成门槛式入选，Logistic 一类会变得自然 | 只有当标签和决策本身正式改写成分类问题时 |
| PCA / 降维支线 | 当特征数膨胀、线性分支共线性上升时，可帮助降噪 | 只在线性分支真的被特征规模拖垮时推进 |
| Autoencoder / VAE | 可能提供更强表征能力 | 只有当样本量、特征复杂度、研究基础设施都明显上台阶时 |

当前更合理的优先级可以直接写成：

1. 继续收口 `target`、方向规则和 anti-drift。
2. 增加最底层 baseline（等权持有 / 市值加权），回答「特征集本身有效吗」这个前置问题。
3. 增加 Random Forest 作为非线性对照，填补 Layer 1 到 Layer 3 的诊断空白。
4. 在当前 protocol 稳住后，再考虑新增一个 boosting challenger（LightGBM）。
5. NGBoost / Quantile Regression 作为风控证据补充，协议稳定后接入。
6. Stacking 只在多模型各自稳定后才有意义。
7. 生存分析（Cox PH）等 Triple Barrier 标签引入后再考虑。当前 time-series-ml 已有，`alpha-research` 尚无。
8. MARS / GAM 作为特征诊断工具箱，按需使用。
9. GNN 需要先建设图数据层。
10. 更远的分类分支、复杂降维或深度表征学习，放到后面。

## 6. 一句话收口

这个项目当前选择 `ridge + elasticnet + xgb_regressor + xgb_ranker`，但这四个模型覆盖的诊断维度不完整：`elasticnet` 和 `ridge` 高度重叠，`ridge` 到 `xgb_regressor` 之间缺少一个隔离「非线性本身」的对照。推荐的五层诊断金字塔是 `等权持有 → Ridge → Random Forest → XGBoost → XGBRanker`，其中 RF 填补了当前最关键的诊断空白。在 `target`、方向和抗漂移问题还没收口之前，先把诊断金字塔补全，通常比继续增加更多 boosting 变体更有价值。
