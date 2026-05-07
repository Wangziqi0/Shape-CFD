# Shape-CFD

**Geometric Post-Processing for Weak-Baseline LLM Embeddings: PQ-Chamfer Distance, Graph Regularization, and a Supervised Per-Query Fusion Predictor.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19553941.svg)](https://doi.org/10.5281/zenodo.19553941)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--8344--1149-a6ce39)](https://orcid.org/0009-0008-8344-1149)

A hybrid pipeline composing parameter-free geometric operators (PQ-Chamfer distance + graph Laplacian smoothing) with a supervised per-query fusion-weight predictor (GBM-11) for retrieval over frozen general-purpose LLM token clouds.

The work documents three substantive contributions plus a 21-entry falsification record:

- **PQ-without-quantization (Theorem 1)**: closed-form discrimination bound on anisotropic embedding spaces, validated empirically across $M \in \{16, 32, 64, 128\}$ within 6-14% relative error.
- **Graph Laplacian on a point-cloud-distance KNN graph**: edges from PQ-Chamfer distance between document token clouds (rather than centroid cosine), with spectral-gap convergence (Proposition 3, Shuman et al. 2013).
- **Curse-of-orthogonality + falsification taxonomy**: $O(1/\sqrt{d})$ SNR upper bound for directional-transport mechanisms in $\mathbb{R}^d$, organising 21 systematically-failed approaches into three classes (curse-of-orthogonality, aggregation pathology, saturation).

## Headline Results (six BEIR corpora)

| Dataset | #Docs | Cosine | Ours (GBM-11) | Rel. Gain | 6-baseline rank |
|---|---|---|---|---|---|
| NFCorpus | 2473 | 0.2195 | **0.3297** | +50.2% | 4/6 |
| SciFact | 3752 | 0.4483 | **0.5418** | +20.9% | 6/6 (last) |
| ArguAna | 8674 | 0.3047 | **0.4862** | +59.6% | 2/6 |
| SCIDOCS | 25337 | 0.1110 | **0.2451** | +120.8% | 1/6 (first) |
| FiQA | 56391 | 0.1683 | **0.4331** | +157.4% | 4/6 |
| Quora | 522931 | 0.6370 | **0.6749** | +6.0% | (no token-level) |

Against PLAID specifically (state-of-the-art high-throughput late-interaction baseline), Ours dominates on three of five corpora (ArguAna +36.0%, SCIDOCS +37.7%, FiQA +11.7%); approximately matches on NFCorpus; trails on SciFact.

The pipeline is positioned as a niche tool for weak-baseline LLM embeddings (cosine NDCG@10 < 0.45), not a general-purpose replacement for retrieval-trained encoders.

## Pipeline

```
Query tokens (frozen LLM hidden states)
      |
      v
Stage 0: Centroid Coarse Filter         (~23 ms, top-100)
      |
      v
Stage 1: PQ-Chamfer Reranking            (~23 ms, top-55)
   subspace decomposition R^4096 = + R^64 (M=64)
   exact per-subspace cosine, averaged
      |
      v
Stage 2: Graph Laplacian Smoothing        (~26 ms)
   k=3 KNN graph from PQ-Chamfer distances
   T=5 iterations of (I - alpha L) C
      |
      v
GBM-11 fusion predictor (supervised, 11 features)
      |
      v
Top-10 (total ~72 ms / query on commodity CPU; total experimental cost \$47)
```

## Repository Structure

```
benchmark/                   - Python + Node.js benchmark scripts
  adaptive_fusion_lambda_v2.py    GBM-11 supervised per-query lambda
  gbm_cross_corpus_loco.py        leave-one-corpus-out cross-corpus audit
  gbm_loo_feature_ablation.py     per-feature leave-one-out
  pq_m_sweep.py                   Theorem 1 multi-M empirical validation
  hartree_lambda_toy.py           Hartree self-consistent lambda (negative result)
  cross_model_unified_5corpora.py 5x2 cross-model grid (BGE-M3 + BGE-large)
  cross_model_per_model_tuned_5corpora.py
  paired_bootstrap_5corpora.py    51-seed bootstrap robustness check
  llm_rerank_bench_bge.py         RankGPT-style listwise on Qwen3-8B-Q4
  llm_rerank_pairwise_setwise.py
  colbertv2_lap_eval_5060.py      ColBERTv2 + Laplacian on consumer GPU
  measure_sigma_subspace.py       sigma_sub / sigma_full cross-corpus measurement
  encode_beir_robust.py           BGE-M3 audited rebuild encode
  data/results/                   per-corpus JSONL + JSON summaries

paper/                        - LaTeX manuscript + figures + bib
  main_blind.tex                  anonymous main (49 pages)
  supplementary_blind.tex         supplementary A: 21-entry falsification record
  references_blind.bib
  figures/                        fig1 pipeline + fig2 scatter + fig4 PQ decomp

rust-engine/                  - Rust core (NAPI bindings)
  src/lib.rs                      NAPI exports
  src/pq_chamfer.rs               PQ subspace distance
  src/token_chamfer.rs            token-level two-stage retrieval
  src/cloud_store.rs              point cloud SQLite storage
  src/pde.rs                      graph Laplacian smoothing
  standalone_tools/train_pq_codebook/  PQ codebook training (NUMA-aware)

scripts/                      - encode-side scripts (E5-Mistral ROCm + 5060 CUDA)
scripts_9070xt/               - 9070XT-specific scripts
```

## Quick Start

### Prerequisites

- Rust 1.75+ with NAPI-RS (for the core distance kernel)
- Python 3.10+ (for benchmark scripts; `numpy`, `scikit-learn`, `transformers`)
- An embedding model accessible via llama.cpp or transformers (Qwen3-8B / BGE-M3 / BGE-large)
- Approx 16-32 GB RAM, optional GPU (CPU-only inference works; ~72 ms / query)

### Build the Rust core

```bash
cd rust-engine
cargo build --release
```

### Reproduce the headline results

```bash
# 1. Download BEIR data (place at benchmark/data/beir_data/<corpus>/)
# 2. Encode token-level point clouds
node benchmark/build_clouds.js benchmark/data/beir_data/scifact

# 3. Run the two-stage pipeline + Laplacian smoothing
python3 benchmark/adaptive_fusion_lambda_v2.py --datasets nfcorpus,scifact,arguana,scidocs,fiqa

# 4. Reproduce Theorem 1 multi-M empirical validation
python3 benchmark/pq_m_sweep.py

# 5. Reproduce cross-corpus LOCO oracle-leakage audit
python3 benchmark/gbm_cross_corpus_loco.py
```

## Manuscript

The manuscript is currently **under review at Information Sciences** (Elsevier).

The main paper (49 pages) and supplementary (13 pages) are in `paper/main_blind.pdf` and `paper/supplementary_blind.pdf`.

## Citation

```bibtex
@software{chen2026shapecfd,
  author    = {Chen, Yifan},
  title     = {Shape-CFD: Geometric Post-Processing for Weak-Baseline LLM Embeddings},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19553941},
  url       = {https://github.com/Wangziqi0/Shape-CFD},
  orcid     = {0009-0008-8344-1149}
}
```

When the journal version is published, please prefer the journal citation.

## License

[Apache License 2.0](LICENSE) -- fully open source, free for any use including production / commercial.

## Contact

Yifan Chen -- TheMexicancjz@net-shopping.com

Issues, pull requests, and reproductions are welcome.
