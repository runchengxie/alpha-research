# 模型选择指南

> status: active
> owner: alpha-research
> last_verified: 2026-06-29
> source_of_truth: yes
> superseded_by: n/a

本页解决什么：四个模型之间的选择与取舍。\
范围：参数定义、配置细节和更多模型家族的边界见相关页面。\
适合谁：需要在模型之间做选择的人。\
读完你会得到什么：模型选择的快速决策与对比视角。\
相关页面：`strategy-pipeline/docs/config.md`、`strategy-pipeline/docs/cookbook.md`、`strategy-pipeline/docs/concepts/benchmark-protocol.md`、`docs/concepts/model-landscape.md`、`docs/concepts/overfitting-controls.md`

迁移说明：本页从 `strategy-pipeline/docs/concepts/model-selection.md` 迁入。`strategy-pipeline` 只保留跳转页和配置/CLI 编排说明。

这个文档帮助你在 `xgb_regressor`、`xgb_ranker`、`ridge`、`elasticnet` 这四个模型之间做选择。\
本页只回答已维护四个模型里这次该选谁；模型保留范围和扩展时机见 [model-landscape.md](model-landscape.md)。\
更广的算法范围，以及 `Logistic`、`KNN`、`PCA`、`Autoencoder` 等方法的取舍，也放在 [model-landscape.md](model-landscape.md)。

## 页面边界

| 问题 | 入口 |
| --- | --- |
| 已维护四个模型这次选谁 | 本页 |
| 为什么当前只维护这四个模型 | [model-landscape.md](model-landscape.md) |
| 模型参数怎样写进配置 | `strategy-pipeline/docs/config.md` |
| 怎样做可比 benchmark | `strategy-pipeline/docs/concepts/benchmark-protocol.md` |
| 怎样判断候选是否过拟合风险可控 | [overfitting-controls.md](overfitting-controls.md) |

## 快速决策

| 你的情况 | 推荐模型 |
|---------|---------|
| 第一次跑，想先拿一个强基线 | `xgb_regressor` |
| 关心同日相对排序，想比较回归和排序 | `xgb_ranker` |
| 先做线性基线，做 sanity check | `ridge` |
| 想在线性模型里做特征压缩 | `elasticnet` |

## 模型对比

| 模型 | 训练目标 | 优点 | 局限 | 适合场景 |
|------|---------|------|------|---------|
| `xgb_regressor` | 直接拟合数值型 label | 能吃非线性和特征交互，通常是最强的通用起点 | 参数更多，训练更慢，可解释性更弱 | 已经有一套稳定研究单元，想先拿一个强非线性基线 |
| `xgb_ranker` | 按 `trade_date` 分组做排序学习 | 和截面选股的排序目标更接近 | 训练口径更特殊，调参和结果解释都更挑数据 | 关心同日相对排序，想比较直接回归与直接排序 |
| `ridge` | 带 L2 正则的线性回归 | 训练快，稳定，参数少，系数容易看 | 只能表达线性关系，吃不到复杂交互 | 要先做线性基线、做 sanity check、或快速比较很多研究单元 |
| `elasticnet` | 带 L1 + L2 正则的线性回归 | 比 `ridge` 更容易压缩无效特征 | 超参数更多，稳定性通常不如 `ridge`，更容易出现退化 run | 想在线性模型里同时做收缩和稀疏化 |

## 为什么默认推荐 xgb_regressor

仓库里的 `default`、`hk` legacy reference、以及多数研究候选都使用 `xgb_regressor`，原因是：

1. 通用性强 - 不需要对数据做什么特殊假设
2. 非线性能力 - 能自动捕获特征交互，这在因子研究中很常见
3. 实战验证多 - 仓库里积累的 baseline 和对比实验大多基于这个模型

## 什么时候考虑换模型

### 用 ridge 做基线

如果你想快速验证这套特征和标签有没有稳定关系，先用 `ridge`。
如果你想按仓库默认顺序做完整 benchmark，对应入口见 `strategy-pipeline/docs/concepts/benchmark-protocol.md`。

最小配置示意：

```yaml
model:
  type: ridge
  params:
    alpha: 1.0
  sample_weight_mode: date_equal
```

`ridge` 跑得快，结果容易解释。如果 `ridge` 的 IC 接近 0，说明这个研究单元本身可能没什么信号。

### 用 xgb_ranker 做专项对照

如果你关心同一天到底该选 A 还是选 B 这种相对排序问题，`xgb_ranker` 更直接：

```yaml
model:
  type: xgb_ranker
  params:
    objective: rank:pairwise
```

但它训练更慢，调参更复杂，适合你已经有一个稳定基线后做对比实验，不适合作为默认起点。

### 用 elasticnet 做特征压缩

如果你在线性模型里还想要稀疏性，可以试试 `elasticnet`：

```yaml
model:
  type: elasticnet
  params:
    alpha: 1.0
    l1_ratio: 0.5
  sample_weight_mode: date_equal
```

注意：`elasticnet` 更容易出现退化 run（常数预测、全零特征重要性），跑完后记得检查 `summary.json` 里的 `flag_constant_prediction` 和 `flag_zero_feature_importance`。

## 线性模型搜索（sweep-linear）

下面的 `strategy` 命令由 `strategy-pipeline` 提供；`alpha-research` 提供命令调用的研究实现。

如果你想做 `ridge` 或 `elasticnet` 的超参数搜索，用 `strategy alpha sweep-linear`：

```bash
strategy alpha sweep-linear --config default --tag a_share_linear_probe --dry-run
```

这个命令会：
1. 批量生成不同 `alpha`（对 ridge）或 `alpha` + `l1_ratio`（对 elasticnet）的配置
2. 逐个执行 `strategy run`
3. 自动汇总结果

注意：这里的线性模型搜索只覆盖 `ridge` 和 `elasticnet`，不包括普通的最小二乘回归。

## XGB / 训练结构调参（tune）

如果你想对 `xgb_regressor`、`xgb_ranker`，或者它们外层的训练结构参数做自动化搜索，用 `strategy alpha tune`：

```bash
strategy alpha tune --tune-config path/to/tune.yml
```

这个命令会：

1. 从 `base_config` 出发，按 `search_space` 生成 trial config
2. 逐个执行 `strategy run`
3. 读取每个 trial 的 `summary.json` 算 objective score
4. 写出 `best_trial.json` 和 `best_config.yml`

如果你在 monthly / 小样本时序问题上想避免账面很强但 `cv_ic` 根本不可判分的 winner，可以在 tune spec 的 `objective` 段加 `min_cv_ic_valid_folds`，把 `eval.cv_ic.scores` 里有效折数不足的 trial 直接排除出 best trial 选择。

更推荐的边界是：

1. 用 `strategy alpha tune` 扫 `model.params`、`sample_weight`、`train_window` 这类训练侧参数
2. 再用 `strategy backtest grid` 在 best signal 上扫 `top_k / cost / buffer / weighting`

不要一开始就把模型参数和 construction 参数混在同一锅里做大网格；这两层在仓库里本来就是分开的。

## 跑完后要检查什么

无论选哪个模型，跑完后都建议检查 `summary.json`：

1. `flag_constant_prediction=true` - 模型输出是常数，特征没用上
2. `flag_zero_feature_importance=true` - 所有特征重要性为 0
3. `train_ic` vs `test_ic` - 过拟合严重吗
4. `backtest_sharpe` - 回测收益是否稳定

如果出现退化 run，汇总时记得排除：

```bash
strategy summarize \
  --runs-dir artifacts/runs \
  --exclude-flag-constant-prediction \
  --exclude-flag-zero-feature-importance
```

## 相关文档

- 配置参数：`strategy-pipeline/docs/config.md`
- CLI 命令：`strategy-pipeline/docs/cli.md`
- 输出字段：`strategy-pipeline/docs/outputs.md`
