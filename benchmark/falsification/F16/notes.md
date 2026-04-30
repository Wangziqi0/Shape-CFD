# F16: temperature_scaled_advection

**实验时期**: V3
**Failure family**: curse_of_orthogonality
**测试数据集**: nfcorpus
**Provenance**: `reconstructed`
**Raw data source**: V3 stability tests

---

## Hypothesis

Adaptive temperature T(x) softens advection in high-density regions.

## Protocol

T(x) ~ 1/density(x); u_eff = T(x)*u.

## Result

Numerical divergence within 10 steps.

## Root cause (geometric / statistical / numerical mechanism)

Graph Peclet Pe = ||u_eff|| h / D = 0.604 exceeds explicit-Euler bound Pe_max ~ 0.5. Adaptive scaling violates CFL non-uniformly across graph; harder to fix than baseline advection.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F16
- **Raw experiment logs**: `V3 stability tests`
- **Provenance level**: `reconstructed`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV3 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
