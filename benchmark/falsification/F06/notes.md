# F06: darcy_scalar_potential

**实验时期**: V3
**Failure family**: curse_of_orthogonality
**测试数据集**: nfcorpus, scifact, fiqa, scidocs, arguana, quora
**Provenance**: `reconstructed`
**Raw data source**: V3 internal experiments

---

## Hypothesis

u = -grad(phi) for scalar potential phi gives curl-free advection, resolves F1.

## Protocol

phi via query similarity; u = -grad(phi) on KNN graph; otherwise like F1.

## Result

NDCG@10 strictly below pure Laplacian on 6/6 datasets.

## Root cause (geometric / statistical / numerical mechanism)

On finite undirected graph, -nabla.(grad(phi).) is a weighted Laplacian. Darcy variant reduces algebraically to isotropic label propagation modulated by ||grad phi||. No genuinely directional component; advection framing is illusory.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F06
- **Raw experiment logs**: `V3 internal experiments`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV3 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
