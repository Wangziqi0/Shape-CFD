# F17: procrustes_alignment

**实验时期**: V8
**Failure family**: numerical_design_pathology
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/procrustes_*.json

---

## Hypothesis

Procrustes alignment of token clouds before Chamfer corrects rotational nuisance.

## Protocol

Iterative orthogonal Procrustes with regularization; per-pair convergence check.

## Result

0% convergence; latency 85.8 ms per document pair.

## Root cause (geometric / statistical / numerical mechanism)

Token clouds in R^4096 have no canonical reference frame: a global rotation aligning one query-document pair misaligns others. Procrustes well-defined only when underlying isometry exists; for token clouds it does not, so iteration cycles indefinitely.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F17
- **Raw experiment logs**: `benchmark/data/results/procrustes_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV8 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
