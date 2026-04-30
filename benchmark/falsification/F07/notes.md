# F07: density_weighted_chamfer

**实验时期**: V8
**Failure family**: aggregation_bias
**测试数据集**: nfcorpus, fiqa, scidocs
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/density_ablation_*.json

---

## Hypothesis

Density-weighted Chamfer emphasizes thematic tokens, improves discrimination.

## Protocol

w_i ~ density(t_i) via k-NN density (k=10); rerank by sum_i w_i min_j d(t_i,t_j).

## Result

NDCG@10 -0.9% to -1.3% across NFCorpus, FiQA, SCIDOCS.

## Root cause (geometric / statistical / numerical mechanism)

High density in token-embedding space = synonymous redundancy (function words, repeated content), not semantic importance. Up-weighting density amplifies redundancy, dilutes rare topical tokens. Intuition reversed.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F07
- **Raw experiment logs**: `benchmark/data/results/density_ablation_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV8 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **aggregation_bias** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
