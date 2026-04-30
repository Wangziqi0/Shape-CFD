# Shape-CFD 21 Falsifications — Raw Data Index

paper main.tex Appendix C §C.2 commitment 的 reproducibility companion. 每个 F## 子目录有 config.yaml / results.jsonl / notes.md 三件套。


## Status table

| F## | Short name | Period | Family | Provenance | Datasets | Result summary |
|-----|-----------|--------|--------|------------|----------|----------------|
| F01 | advection_diffusion | V1-V3 | curse_of_orthogonality | complete | nfcorpus, scifact, fiqa, scidocs, arguana, quora | Advection harmful/neutral on 5/5 validations. NFCorpus PDE_55=0.2852 vs Lap_55=0... |
| F02 | allen_cahn_dormant | V5.2 | numerical_design_pathology | complete | nfcorpus | dNDCG@10 <= 5e-4 across all settings. Dormant. |
| F03 | allen_cahn_normalized | V5.1 | numerical_design_pathology | complete | nfcorpus | NDCG@10 -12.9% on NFCorpus. |
| F04 | jl_random_projection | V3 | curse_of_orthogonality | reconstructed | nfcorpus, fiqa, scidocs | NDCG@10 -7.6% across NFCorpus, FiQA, SCIDOCS. |
| F05 | flow_field_annealing | V3 | curse_of_orthogonality | reconstructed | nfcorpus, scifact, fiqa | Indistinguishable from JL noise (F4) and pure diffusion. No annealing signal. |
| F06 | darcy_scalar_potential | V3 | curse_of_orthogonality | reconstructed | nfcorpus, scifact, fiqa, scidocs, arguana, quora | NDCG@10 strictly below pure Laplacian on 6/6 datasets. |
| F07 | density_weighted_chamfer | V8 | aggregation_bias | complete | nfcorpus, fiqa, scidocs | NDCG@10 -0.9% to -1.3% across NFCorpus, FiQA, SCIDOCS. |
| F08 | pq_code_reconstruction | V8 | aggregation_bias | complete | nfcorpus | NDCG@10 -34% on NFCorpus. |
| F09 | pq_chamfer_first_stage | V12 | aggregation_bias | complete | nfcorpus | Recall@1000 -4.2% on NFCorpus. |
| F10 | global_graph_diffusion | V8 | aggregation_bias | complete | nfcorpus | NDCG@10 -7.6% on NFCorpus. |
| F11 | l3_subspace_cross_visit | V11 | numerical_design_pathology | reconstructed | nfcorpus | No measurable gain; latency +3x. |
| F12 | binary_hdc_popcount | V8 | numerical_design_pathology | reconstructed | nfcorpus | No system-level speedup. |
| F13 | turing_patterns | V5.2 | numerical_design_pathology | reconstructed | — | Mathematically impossible to instantiate. |
| F14 | tensor_diffusion | V3 | curse_of_orthogonality | reconstructed | fiqa | Memory infeasible. |
| F15 | mixed_initial_field | V5.2 | numerical_design_pathology | complete | nfcorpus, fiqa, scidocs | Inconsistent: NFCorpus +3.2%, FiQA -1.5%, SCIDOCS +0.3%. No stable improvement. |
| F16 | temperature_scaled_advection | V3 | curse_of_orthogonality | reconstructed | nfcorpus | Numerical divergence within 10 steps. |
| F17 | procrustes_alignment | V8 | numerical_design_pathology | complete | nfcorpus | 0% convergence; latency 85.8 ms per document pair. |
| F18 | block_aligned_attention | V11 | curse_of_orthogonality | reconstructed | fiqa, scidocs | 20% convergence; unstable on FiQA, SCIDOCS. |
| F19 | local_pca_projection | V8 | aggregation_bias | reconstructed | nfcorpus | NDCG@10 within +/-0.3% of baseline; latency +33 ms per document. |
| F20 | strong_baseline_saturation | V11 | numerical_design_pathology | complete | scifact | Token-level PQ-Chamfer -3.1% on SciFact. |
| F21 | multi_emission_coarse | V11 | aggregation_bias | complete | nfcorpus | Recall@1000 +22%, but NDCG@10 unchanged. |

## Provenance distribution

- `complete`: 11 / 21
- `partial`: 0 / 21
- `reconstructed`: 10 / 21

## Family distribution

- `numerical_design_pathology`: 8 / 21
- `curse_of_orthogonality`: 7 / 21
- `aggregation_bias`: 6 / 21

## Provenance levels

- **complete**: config + per-query results + notes 全有
- **partial**: aggregate result + notes，per-query 失踪
- **reconstructed**: 仅从 paper Appendix C 重建，原始 raw data 在该 V 期实验未落盘
