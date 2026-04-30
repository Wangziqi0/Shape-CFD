# F12: binary_hdc_popcount

**实验时期**: V8
**Failure family**: numerical_design_pathology
**测试数据集**: nfcorpus
**Provenance**: `reconstructed`
**Raw data source**: V8 profiling notes

---

## Hypothesis

Binarize embeddings (sign per dim) + Hamming distance gives 32x acceleration.

## Protocol

sign(x) in {-1,+1}^4096; bitpack and Hamming-rerank.

## Result

No system-level speedup.

## Root cause (geometric / statistical / numerical mechanism)

Profiling: pipeline bottleneck is token-cloud I/O and graph construction, not inner distance kernel. Distance kernel already at SIMD throughput. Hypothesis correct (HDC faster per op) but irrelevant (op cost not bottleneck).

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F12
- **Raw experiment logs**: `V8 profiling notes`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV8 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **numerical_design_pathology** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
