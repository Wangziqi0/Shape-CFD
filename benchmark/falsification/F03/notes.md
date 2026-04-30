# F03: allen_cahn_normalized

**实验时期**: V5.1
**Failure family**: numerical_design_pathology
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: walkthrough.md §2.2 V5.1 normalized failure

---

## Hypothesis

Min-max normalization of C lifts F2's reaction-term dormancy.

## Protocol

Same as F2; pre-normalize C to [0,1] before reaction.

## Result

NDCG@10 -12.9% on NFCorpus.

## Root cause (geometric / statistical / numerical mechanism)

Normalization rescales effective gamma to gamma/0.05=500 (50x amplification). Reaction now dominates diffusion, pushes scores to basin minima {0,a,1}, collapses rank distinctions, numerical divergence.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F03
- **Raw experiment logs**: `walkthrough.md §2.2 V5.1 normalized failure`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV5.1 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
