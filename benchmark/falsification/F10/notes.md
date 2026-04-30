# F10: global_graph_diffusion

**实验时期**: V8
**Failure family**: aggregation_bias
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/global_diffusion_*.json

---

## Hypothesis

Diffusing over full document graph propagates relevance to multi-hop neighbors.

## Protocol

Full graph G with n^2 edges weighted by cosine; T steps of C <- (1-a)C + a P C.

## Result

NDCG@10 -7.6% on NFCorpus.

## Root cause (geometric / statistical / numerical mechanism)

Spectral gap collapses with too many edges. Small eigenvalues of L=I-P approach 0; diffusion mixes toward stationary distribution (uniform on largest connected component). Over-smoothing dominates after O(log n) steps.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F10
- **Raw experiment logs**: `benchmark/data/results/global_diffusion_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV8 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **aggregation_bias** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
