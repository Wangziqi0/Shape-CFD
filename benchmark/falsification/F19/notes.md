# F19: local_pca_projection

**实验时期**: V8
**Failure family**: aggregation_bias
**测试数据集**: nfcorpus
**Provenance**: `reconstructed`
**Raw data source**: V8 PCA experiments

---

## Hypothesis

Per-document PCA to top-k components reduces noise.

## Protocol

Per-document PCA, retain top-k components (k in {16,32,64}).

## Result

NDCG@10 within +/-0.3% of baseline; latency +33 ms per document.

## Root cause (geometric / statistical / numerical mechanism)

Per-document PCA bases not aligned across documents (each PCA is data-dependent). Cross-document distances in projected space are not comparable. Marginal accuracy reflects this incoherence not robust signal. Global PCA would align bases but lose per-document structure.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F19
- **Raw experiment logs**: `V8 PCA experiments`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV8 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **aggregation_bias** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
