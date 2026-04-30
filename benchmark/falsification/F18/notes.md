# F18: block_aligned_attention

**实验时期**: V11
**Failure family**: curse_of_orthogonality
**测试数据集**: fiqa, scidocs
**Provenance**: `reconstructed`
**Raw data source**: V11 attention experiments

---

## Hypothesis

Within-block (PQ subspace) attention combines learned and geometric matching.

## Protocol

64-block soft attention; Pe_block = 0.226 analog.

## Result

20% convergence; unstable on FiQA, SCIDOCS.

## Root cause (geometric / statistical / numerical mechanism)

Block attention reintroduces directional signal that F1 proved O(1/sqrt(d))-noisy at block level (subspace dim 64). Block SNR 1/sqrt(64)=0.125 better than full-space 0.016, but attention amplifies still-noisy signal nonlinearly => cascading instability.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F18
- **Raw experiment logs**: `V11 attention experiments`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV11 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
