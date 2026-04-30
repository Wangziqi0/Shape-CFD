# F05: flow_field_annealing

**实验时期**: V3
**Failure family**: curse_of_orthogonality
**测试数据集**: nfcorpus, scifact, fiqa
**Provenance**: `reconstructed`
**Raw data source**: V3 internal experiments

---

## Hypothesis

Stochastic per-step random advection direction escapes local optima.

## Protocol

Per-step random unit vector replacing u; otherwise identical to F1.

## Result

Indistinguishable from JL noise (F4) and pure diffusion. No annealing signal.

## Root cause (geometric / statistical / numerical mechanism)

Random direction in R^4096 = JL-style isotropic variance amplification. Injects O(1/sqrt(d)) noise without exploiting landscape. 'Annealing' is misnomer because no biased signal to anneal toward.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F05
- **Raw experiment logs**: `V3 internal experiments`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV3 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
