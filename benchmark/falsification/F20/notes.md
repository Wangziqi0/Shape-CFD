# F20: strong_baseline_saturation

**实验时期**: V11
**Failure family**: numerical_design_pathology
**测试数据集**: scifact
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/scifact_*.json

---

## Hypothesis

PQ-Chamfer + Laplacian improves over any cosine baseline including strong ones.

## Protocol

SciFact (cosine baseline 0.4483, strongest in our set).

## Result

Token-level PQ-Chamfer -3.1% on SciFact.

## Root cause (geometric / statistical / numerical mechanism)

When cosine baseline near empirical ceiling, geometric reranking has little headroom: residual ranking error dominated by intrinsic label noise not concentration noise. PQ-Chamfer gain scales inversely with baseline quality.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F20
- **Raw experiment logs**: `benchmark/data/results/scifact_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV11 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
