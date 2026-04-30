# F15: mixed_initial_field

**实验时期**: V5.2
**Failure family**: numerical_design_pathology
**测试数据集**: nfcorpus, fiqa, scidocs
**Provenance**: `complete`
**Raw data source**: walkthrough.md §3 V5.2 mixed initial

---

## Hypothesis

Mixing query-aligned mass with uniform prior stabilizes diffusion.

## Protocol

C(x,0) = (1-beta)*cos(q,x) + beta*1/n; beta in {0.1,0.3,0.5}.

## Result

Inconsistent: NFCorpus +3.2%, FiQA -1.5%, SCIDOCS +0.3%. No stable improvement.

## Root cause (geometric / statistical / numerical mechanism)

Optimal beta depends on query-prior mass ratio (varies with dataset). Per-dataset hyperparameter masquerading as method; any apparent gain = hyperparameter overfitting on dev set.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F15
- **Raw experiment logs**: `walkthrough.md §3 V5.2 mixed initial`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV5.2 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
