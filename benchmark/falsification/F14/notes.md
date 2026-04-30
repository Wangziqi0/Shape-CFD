# F14: tensor_diffusion

**实验时期**: V3
**Failure family**: curse_of_orthogonality
**测试数据集**: fiqa
**Provenance**: `reconstructed`
**Raw data source**: V3 design notes

---

## Hypothesis

Per-edge anisotropic tensor D_ij captures direction-dependent flow.

## Protocol

Per-edge D_ij in R^(4096x4096).

## Result

Memory infeasible.

## Root cause (geometric / statistical / numerical mechanism)

4096x4096 float32 tensor = ~64MB; FiQA graph ~10^5 edges => ~6.4TB edge state. Sparsified to 100 nonzeros per edge still impractical; underlying directional signal still bounded by O(1/sqrt(d)) (F1).

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F14
- **Raw experiment logs**: `V3 design notes`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV3 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
