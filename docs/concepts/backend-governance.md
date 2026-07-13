# 研究后端治理与晋升证据

## 所有权边界

alpha-research 拥有模型选择政策、调参目标函数、CPCV / PBO、特征证据、后端一致性证据和
promotion gate。strategy-pipeline 只负责解析公共 CLI、准备跨仓库依赖并调用这些服务；它
不应再复制搜索空间展开、trial 评分或最佳 trial 选择逻辑。

native 是默认研究后端。Qlib 只有在调用方显式安装可选依赖并注入
`QlibDatasetBackend` / `QlibTrainerBackend` 时才会启用。CPCV 和时序 CV 都沿用同一个
`TrainerBackend`，因此不会绕过调用方选择的后端。

边界上的对象只有四类：

- canonical `ResearchDataset` / DataFrame；
- 只含字符串和 JSON metadata 的 `FittedModelHandle`、`ExperimentReceipt`；
- 带绝对路径和 SHA-256 的 `ArtifactHandle`；
- 从持久化 artifact 重放得到的比较或晋升报告。

拟合后的 sklearn、XGBoost 和 Qlib 对象仅保存在对应 backend 实例的进程内 registry 中。
`runtime_ref` 只是进程内查找 token，不会进入 `to_metadata()`。反序列化后的模型 metadata
不能恢复模型，也不能用于预测；需要预测时必须重新训练或使用另行定义的安全模型发布格式。
任何 governance JSON 都采用严格 JSON 写入。既有科学计算报告会先把明确识别的 NumPy
标量投影为普通数值、把 NaN / Infinity 投影为 JSON `null`；handle、receipt 和 comparison
等新契约要求调用方预先提供有限 JSON 值。未知对象始终直接失败，禁止 `default=str`、
pickle、joblib 等隐式序列化逃生口。

## 可重放的 native / Qlib 比较

先把两个后端在同一 canonical 样本上的预测和指标分别冻结：

```python
from cstree.alpha.backend_comparison import (
    BackendPromotionThresholds,
    compare_backend_evaluations,
    write_backend_evaluation,
    write_backend_comparison_replay_receipt,
)

native = write_backend_evaluation(
    "artifacts/backend-check/native",
    backend_id="native",
    run_id="ridge-native",
    predictions=native_predictions,
    metrics={"rank_ic": 0.031, "sharpe": 1.08},
    model_handle=native_handle,
)
qlib = write_backend_evaluation(
    "artifacts/backend-check/qlib",
    backend_id="qlib",
    run_id="ridge-qlib",
    predictions=qlib_predictions,
    metrics={"rank_ic": 0.031, "sharpe": 1.08},
    model_handle=qlib_handle,
)

report = compare_backend_evaluations(
    native,
    qlib,
    thresholds=BackendPromotionThresholds(
        min_overlap_rows=1000,
        min_overlap_ratio=1.0,
        min_prediction_pearson=0.999,
        min_prediction_spearman=0.999,
        max_prediction_mae=1e-8,
        max_prediction_abs_error=1e-7,
        metric_abs_tolerances={"rank_ic": 1e-8, "sharpe": 1e-8},
    ),
    output_path="artifacts/backend-check/comparison.json",
)

receipt = write_backend_comparison_replay_receipt(
    report,
    output_path="artifacts/backend-check/replay-receipt.json",
)
```

`backend_evaluation.v1` 保存排序后的 canonical prediction CSV、指标、纯 metadata 和 CSV
摘要哈希。`backend_comparison.v1` 再保存两个 evaluation manifest 的路径及哈希、全部阈值、
实测差异和决定。`replay_backend_comparison()` 会重新校验两层 manifest 与 prediction 哈希，
并重新计算决定；源文件被改动或报告决定被手工修改都会失败。
`write_backend_comparison_replay_receipt()` 会在成功重放后写出严格 JSON 的
`backend_comparison_replay_receipt.v1`，其中固定包含源报告 SHA-256、native / Qlib 身份、
完整比较摘要、阈值、决定和确定性的验证方式，不写入墙上时钟时间。上层仓库因此只需按
schema 读取 receipt，不必反向导入 alpha 或 Qlib。

promotion gate 可直接要求这份证据：

```yaml
promotion_gate:
  required_evidence:
    - main_eval
    - backtest
    - backend_comparison
  backend_comparison:
    candidate_report: artifacts/backend-check/comparison.json
    require_pass: true
```

报告不可重放时会产生 `backend_comparison_unverified`；比较超过阈值时会产生
`backend_comparison_rejected`。阈值属于 comparison artifact 本身，promotion gate 不会用
运行时默认值悄悄改写历史决定。

## 调参应用服务

`TuningApplicationService` 是 alpha-owned 的稳定委托点。它负责：

- 校验并展开 grid / seeded random 搜索空间；
- 应用点路径 override，生成 `configs/trial_NNN.yml`；
- 读取 runner 返回且通过 SHA-256 校验的 summary artifact；
- 用既有目标函数评分，写 `jobs.csv`、`trial_results.csv`、`best_trial.json`、
  `best_config.yml` 和 `tuning_receipt.json`；
- 通过框架无关 `ExperimentRecorder` 记录每个 trial。

strategy-pipeline 只需实现 `TuningTrialRunner.run(job)`：调用 pipeline 后，把落盘的
`summary.json` 包装为 `ArtifactHandle` 并返回 `TuningTrialOutcome`。runner 不能把 pipeline
context、模型对象或 Qlib recorder 传回 alpha。现有 CLI 可以继续保留，跨 run 的
`runs_summary.csv` 也仍可由 strategy-pipeline 的通用 summarize 步骤生成。

dry-run 不需要 runner，仍会生成与现有 CLI 相同的 trial config 和 jobs manifest。执行路径
只消费持久化 summary handle，因此后续 SP-02 可以把命令层缩成参数解析、runner adapter 和
最终 summarize，而不再维护第二套调参政策。

## 现有证据链

- artifact CPCV 只读取冻结 CSV / Parquet，pipeline-backed CPCV 也只通过 trainer handle
  预测；
- PBO 只读取 candidate-return matrix，并写严格 JSON summary 与拆分 CSV；
- feature evidence 只读取 YAML、run summary 和冻结 frame，JSON 输出先显式归一化 NumPy
  标量与非有限值；
- promotion gate 只读取 run/artifact 报告，并可校验 backend comparison 的完整重放链。

这些约束让 native 与 Qlib 共用一套研究治理，不让可选后端演变成第二套证据框架。
