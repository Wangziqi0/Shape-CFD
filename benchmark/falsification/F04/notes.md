# F04: jl_random_projection

**实验时期**: V3
**Failure family**: curse_of_orthogonality
**测试数据集**: nfcorpus, fiqa, scidocs
**Provenance**: `reconstructed`
**Raw data source**: cfd_problems_briefing.md problem 1

---

## Hypothesis

JL random projection 4096->128 preserves distances within 1+/-eps with 32x speedup.

## Protocol

W ~ N(0,1/d) in R^(128x4096); cosine reranking on projected representations.

## Result

NDCG@10 -7.6% across NFCorpus, FiQA, SCIDOCS.

## Root cause (geometric / statistical / numerical mechanism)

JL bound eps ~ sqrt(log n / k); n~10^4 k=128 gives eps~0.27. Cosine std sigma~0.016, distortion is ~17 sigma, vastly exceeds discriminative scale. Information loss dominates noise reduction.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F04
- **Raw experiment logs**: `cfd_problems_briefing.md problem 1`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV3 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
