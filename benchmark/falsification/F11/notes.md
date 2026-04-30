# F11: l3_subspace_cross_visit

**实验时期**: V11
**Failure family**: numerical_design_pathology
**测试数据集**: nfcorpus
**Provenance**: `reconstructed`
**Raw data source**: V11 LID measurement notes

---

## Hypothesis

Cross-subspace token matching exploits unequal info content across Qwen3 dims.

## Protocol

Cross-subspace matching with permutation-cost penalty; vs fixed within-subspace.

## Result

No measurable gain; latency +3x.

## Root cause (geometric / statistical / numerical mechanism)

Local intrinsic dimensionality (LID) shows Qwen3 dims approximately uniform info content. No asymmetry to exploit by cross-subspace matching; combinatorial freedom inflates cost without benefit.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F11
- **Raw experiment logs**: `V11 LID measurement notes`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV11 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
