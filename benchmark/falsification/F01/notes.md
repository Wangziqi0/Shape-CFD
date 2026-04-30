# F01: advection_diffusion

**实验时期**: V1-V3
**Failure family**: curse_of_orthogonality
**测试数据集**: nfcorpus, scifact, fiqa, scidocs, arguana, quora
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/, walkthrough.md V1-V3 logs

---

## Hypothesis

Convection-diffusion PDE with query-aligned advection u outperforms pure diffusion.

## Protocol

72-config grid over (D, |u|, dt); 6 BEIR datasets; NDCG@10. Compare PDE vs Lap on identical KNN graphs.

## Result

Advection harmful/neutral on 5/5 validations. NFCorpus PDE_55=0.2852 vs Lap_55=0.2900.

## Root cause (geometric / statistical / numerical mechanism)

In R^4096, displacement vectors are nearly orthogonal to query direction (sigma=1/sqrt(d)=0.016). Advection SNR is O(1/sqrt(d)). Diffusion isotropic so escapes; advection cannot. Curse of orthogonality applied to vector fields.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F01
- **Raw experiment logs**: `benchmark/data/results/, walkthrough.md V1-V3 logs`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV1-V3 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **curse_of_orthogonality** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
