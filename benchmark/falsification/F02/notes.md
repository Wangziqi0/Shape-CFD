# F02: allen_cahn_dormant

**实验时期**: V5.2
**Failure family**: numerical_design_pathology
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: walkthrough.md §2.3 V5.2 Allen-Cahn

---

## Hypothesis

Bistable reaction f(C)=gamma*C(1-C)(C-a) sharpens relevance/irrelevance boundary.

## Protocol

NFCorpus, gamma in {0.1,1,5}, a in {0.3,0.5}, dt=0.01, 50 steps.

## Result

dNDCG@10 <= 5e-4 across all settings. Dormant.

## Root cause (geometric / statistical / numerical mechanism)

For short queries score range narrow ~0.05. Reaction f(C) ~ O(Delta^3) when C-a~Delta is small; per-step contribution ~10^-9, dormant relative to diffusion's O(Delta).

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F02
- **Raw experiment logs**: `walkthrough.md §2.3 V5.2 Allen-Cahn`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV5.2 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
