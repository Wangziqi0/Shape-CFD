# F09: pq_chamfer_first_stage

**实验时期**: V12
**Failure family**: aggregation_bias
**测试数据集**: nfcorpus
**Provenance**: `complete`
**Raw data source**: benchmark/data/results/pq_first_stage_*.json

---

## Hypothesis

PQ-Chamfer as first-stage retrieval directly captures token-level matches.

## Protocol

Inverted index over PQ codes; query = single pooled embedding; rank by PQ-Chamfer.

## Result

Recall@1000 -4.2% on NFCorpus.

## Root cause (geometric / statistical / numerical mechanism)

Chamfer asymmetric, degenerates when one side has single point: Chamfer({q},D) = nearest-neighbor distance to closest token, dominated by document length and noise. Chamfer's value is in cloud-to-cloud aggregation; single-point queries waste it.

---

## Reproduce

如需复现此实验，请参考:

- **完整 hypothesis / protocol / root-cause 分析**: paper `main.tex` Appendix C §F09
- **Raw experiment logs**: `benchmark/data/results/pq_first_stage_*.json`
- **Provenance level**: `complete`
  - `complete` = config + per-query results + notes 全有
  - `partial` = aggregate result + notes，per-query 失踪
  - `reconstructed` = 仅从 paper Appendix C 重建，原始 raw data 在 VV12 实验未落盘

## Notes

此 falsification 属于 paper §C synthesis 中的 **aggregation_bias** 家族。Synthesis 段说明 PQ-Chamfer + Laplacian 主管线为什么避开此类失败模式。
