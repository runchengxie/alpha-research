# 模型选择指南

> status: active
> owner: alpha-research
> last_verified: 2026-07-16
> source_of_truth: yes
> superseded_by: n/a

本页帮助研究者在四个可训练模型中选择起点。模型注册范围和扩展条件见 [model-landscape.md](model-landscape.md)。参数定义和命令说明见 `strategy-pipeline/docs/configuration.md` 与 `strategy-pipeline/docs/cli-helpers.md`。

模型注册表还包含 `fixed_score_artifact`。它读取外部冻结分数，不训练预测模型，因此单独放在产物回放流程中使用。

## 快速选择

| 研究目的 | 建议模型 |
| --- | --- |
| 建立默认非线性基线 | `xgb_regressor` |
| 检查同日排序目标是否有增量 | `xgb_ranker` |
| 快速检查线性信号 | `ridge_scaled` |
| 已同尺度特征的兼容性对照 | `ridge` |
| 检查稀疏线性约束 | `elasticnet` |
| 回放已经冻结的外部分数 | `fixed_score_artifact` |

## 模型角色

| 模型 | 训练方式 | 优点 | 使用边界 |
| --- | --- | --- | --- |
| `xgb_regressor` | 拟合连续标签 | 能表达非线性关系和特征交互 | 参数较多，需要稳定的时间切分和退化检查 |
| `xgb_ranker` | 按 `trade_date` 分组学习排序 | 训练目标贴近横截面选股 | 依赖有效的日内分组，训练和调参成本较高 |
| `ridge` | 带 L2 正则的线性回归 | 训练快，系数容易解释，对共线性较稳 | 无法表达复杂的非线性关系 |
| `ridge_scaled` | 折内标准化后带 L2 正则的线性回归 | 适合比例、金额和比率混合的财务特征，避免尺度主导正则化 | 仍无法表达复杂的非线性关系，标准化参数必须只在训练折拟合 |
| `random_forest_regressor` | 随机森林回归 | 通过随机树平均建立非线性对照 | 仍需时间切分和标签成熟检查，袋外分数不能替代时间外验证 |
| `elasticnet` | 带 L1 和 L2 正则的线性回归 | 可以同时收缩系数和压缩特征 | 需要同时选择 `alpha` 与 `l1_ratio`，更容易出现全零输出 |

A 股预设当前使用 `xgb_regressor`。财务研究应优先使用 `ridge_scaled` 作为线性对照，再用 `ridge` 做兼容性检查。两者都不能替代严格的时间切分。

## 最小配置

### Ridge

```yaml
model:
  type: ridge
  params:
    alpha: 1.0
  sample_weight_mode: date_equal
```

`ridge` 适合快速检查特征和连续标签之间是否存在稳定的线性关系。线性结果弱时，应先复核数据、标签和特征证据，再增加模型复杂度。

### 标准化 Ridge

```yaml
model:
  type: ridge_scaled
  params:
    alpha: 1.0
```

`ridge_scaled` 在每个训练折内拟合 `StandardScaler`，再拟合 Ridge。它适合财务特征同时包含百分比、金额和比率的研究。标准化器不得使用测试折或未来日期拟合。

### XGBoost Ranker

```yaml
model:
  type: xgb_ranker
  params:
    objective: rank:pairwise
```

`xgb_ranker` 会按 `trade_date` 组织训练分组。它适合在已有可复现回归基线后，检验直接学习排序是否改善横截面结果。

### ElasticNet

```yaml
model:
  type: elasticnet
  params:
    alpha: 1.0
    l1_ratio: 0.5
  sample_weight_mode: date_equal
```

`elasticnet` 用于稀疏线性专项对照。运行后需要检查 `summary.json` 中的 `flag_constant_prediction` 和 `flag_zero_feature_importance`。

### 固定分数产物

```yaml
model:
  type: fixed_score_artifact
  params:
    score_col: pred
```

输入数据必须包含 `score_col` 指定的列。该入口用于原样回放冻结分数，适合验证外部策略或跨仓库信号交接。

## 参数搜索

下列 `strategy` 命令由 `strategy-pipeline` 提供，研究实现位于 `alpha-research`。

线性模型搜索覆盖 `ridge` 和 `elasticnet`：

```bash
strategy alpha sweep-linear --config default --tag a_share_linear_probe --dry-run
```

XGBoost 参数和训练结构搜索使用：

```bash
strategy alpha tune --tune-config path/to/tune.yml
```

调参时建议遵守以下边界：

1. 先固定数据、标签、特征和时间切分。
2. 训练阶段搜索 `model.params`、`sample_weight` 和 `train_window`。
3. 在选定信号后，再用 `strategy backtest grid` 检查 `top_k`、成本、缓冲和权重。
4. 小样本任务可通过 `min_cv_ic_valid_folds` 排除有效折数不足的试验。
5. 保留完整试验台账，避免只记录表现最好的配置。

## 运行后检查

至少检查 `summary.json` 中的以下内容：

1. `flag_constant_prediction`，识别常数预测。
2. `flag_zero_feature_importance`，识别全零重要度。
3. `train_ic` 与 `test_ic`，观察训练和测试差距。
4. `backtest_sharpe`、换手和成本，判断收益是否具备执行价值。
5. 前推验证和最终样本外证据，判断结果是否跨时间稳定。

汇总时可以排除已确认的退化运行：

```bash
strategy summarize --runs-dir artifacts/runs --exclude-flag-constant-prediction --exclude-flag-zero-feature-importance
```

正式比较模型时，应统一数据资产、标签、时间切分、成本和组合构造。研究治理要求见 [overfitting-controls.md](overfitting-controls.md)。
