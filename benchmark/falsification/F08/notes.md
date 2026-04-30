# F08: pq_code_reconstruction

**实验时期**: V8
**Failure family**: aggregation_bias
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/pq_recon_*.json

---

## Hypothesis

PQ-quantized reconstruction accelerates Chamfer with preserved ranking.

## Protocol

64 subspaces x 256 centroids; x_hat = concat(c_{i_s}); rerank on x_hat instead of x.

## Result

NDCG@10 -34% on NFCorpus.

## Root cause (geometric / statistical / numerical mechanism)

Chamfer takes min_j d(t_i,t_j), dominated by smallest-error pairs. Quantization error unbiased on average but min-aggregation extreme-value statistics biases it downward. Rank order = quantization artifacts not semantic similarity. Sum/mean would tolerate but min does not.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F08
- **Raw experiment logs**: `benchmark/data/results/pq_recon_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV8 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **aggregation_bias** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
