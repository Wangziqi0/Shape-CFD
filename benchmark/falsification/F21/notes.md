# F21: multi_emission_coarse

**实验时期**: V11
**Failure family**: aggregation_bias
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/multi_emission_*.json

---

## Hypothesis

Multi-candidate coarse filter expands recall pool, gives reranker headroom.

## Protocol

Query -> top-M (M=5) emissions -> union recall set -> rerank.

## Result

Recall@1000 +22%, but NDCG@10 unchanged.

## Root cause (geometric / statistical / numerical mechanism)

Additional recalled docs are marginal (low-similarity); reranker discriminative power on them at noise floor, cannot reliably promote marginal-but-relevant past previously-ranked. Recall and NDCG@10 measure different things.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F21
- **Raw experiment logs**: `benchmark/data/results/multi_emission_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV11 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **aggregation_bias** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
