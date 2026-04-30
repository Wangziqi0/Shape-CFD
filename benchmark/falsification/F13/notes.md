# F13: turing_patterns

**实验时期**: V5.2
**Failure family**: numerical_design_pathology
**测试数据集**: (N/A — category error)
**Provenance**: `reconstructed`
**Raw data source**: walkthrough.md V5.2 Turing attempt

---

## Hypothesis

Turing instabilities create spatial patterns, self-organized cluster boundaries.

## Protocol

Single-species Allen-Cahn with periodic boundary on document graph.

## Result

Mathematically impossible to instantiate.

## Root cause (geometric / statistical / numerical mechanism)

Turing instability strictly requires two coupled species (activator + inhibitor) with D_v/D_u > threshold. Single scalar field C cannot exhibit Turing patterns. Hypothesis was a category error.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F13
- **Raw experiment logs**: `walkthrough.md V5.2 Turing attempt`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV5.2 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
