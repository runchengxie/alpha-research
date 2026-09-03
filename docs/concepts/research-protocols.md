# 分级研究协议

本页说明 `alpha_research.research_protocols` 如何把已有防过拟合工具组织成可检查的探索、候选和发布协议。命令行编排仍由 pipeline 负责，协议规则和证据校验由本仓库维护。

## 三个级别

### exploratory

探索运行必须记录：

- 已发布数据契约
- 可复现 run 信息
- trial registry

允许使用普通日期 gap 和显式 fallback。探索结果不得直接作为实盘发布依据。

### candidate

候选晋升增加：

- feature evidence
- 弱模型 baseline
- walk-forward
- final OOS
- CPCV
- 成本、换手和容量证据
- exposure screen
- negative controls

候选协议要求 event-window purge 覆盖率至少为 95%，且不允许 fallback。

### release

发布交接在 candidate 基础上增加：

- DSR
- PBO，或明确的 `insufficient_evidence`
- scenario backtest
- candidate freeze
- paper/shadow evidence
- sizing receipt
- strategy risk report
- operator approval

发布协议要求 event-window purge 覆盖率至少为 99%，且不允许 fallback。

## 初始化清单

```bash
strategy research-protocol \
  --level candidate \
  --init-manifest artifacts/reports/candidate_protocol.yml
```

清单中的每个文件证据项包含：

```yaml
status: missing
path: null
sha256: null
notes: Evidence description
```

`status: pass` 本身不构成证据。candidate 和 release 检查会确认文件存在，并核对 SHA-256。路径相对于 manifest 所在目录解析。人工审批必须填写：

```yaml
operator_approval:
  status: pass
  approved_by: operator-id
  approved_at: 2026-07-14T12:00:00-07:00
```

PBO 在可比较试验不足时可使用 `insufficient_evidence`，但必须填写原因，避免为了让状态灯变绿而对两个毫无可比性的 run 做统计杂技。

## 从真实 run 生成 AFML sidecar

运行目录已经包含 `backtest_net.csv`、`backtest_gross.csv`、`backtest_turnover.csv` 和当前持仓时，可以生成机器计算的 sizing 与 strategy-risk 证据：

```bash
strategy afml-evidence \
  --run-dir artifacts/runs/<run> \
  --target-sharpe 1.0 \
  --evaluation-years 2 \
  --bootstrap-samples 2000 \
  --manifest artifacts/reports/release_protocol.yml
```

命令生成：

```text
sizing_receipt.json
strategy_risk_report.json
afml_evidence_fragment.json
```

并把 `sizing_receipt`、`strategy_risk` 的路径和 SHA-256 合并到指定 protocol manifest。权重方法从 `config.used.yml` 的 `strategy.weighting` / `backtest.weighting` 读取，receipt 描述最终持仓，不重新构造组合。

可选传入多模型或 sleeve 收益矩阵：

```bash
strategy afml-evidence \
  --run-dir artifacts/runs/<run> \
  --hrp-returns artifacts/reports/sleeve_returns.csv
```

这会额外生成 `hrp_weights.csv` 和 `hrp_receipt.json`。HRP 输入第一列为日期，后续至少两列为同步收益序列。

## 在正式 run 后自动生成

正式候选配置可以在回测和持仓产物写完后自动生成 sidecar：

```yaml
research_protocol:
  generate_afml_evidence: true
  target_sharpe: 1.0
  evaluation_years: 2.0
  bootstrap_samples: 2000
  random_state: 20260714
  require_release_report: true
```

该开关默认关闭，参数 sweep 和普通探索运行不会自动执行 bootstrap。若需要 HRP，可额外配置 `hrp_returns` 指向同步的模型或 sleeve 收益矩阵。

## 执行协议

```bash
strategy research-protocol \
  --level release \
  --manifest artifacts/reports/release_protocol.yml \
  --output artifacts/runs/<run>/research_protocol_report.json
```

默认 strict 模式下，缺失、哈希不匹配或失败证据会返回非零退出码。使用 `--no-strict` 只适合生成诊断报告。

## 执行交接

`research_protocol_report.json` 放在 run 目录后，现有 liveops/export quality gate 会在目标导出前检查。报告只决定研究候选是否允许交接，不参与标的、权重或订单计算。

研究协议报告应进入 `targets.json.lineage.json`，而标准 `targets.json` 继续只包含执行目标。

