# Shape-CFD: Point Cloud Retrieval with PQ-Chamfer Distance and Graph Smoothing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19553941.svg)](https://doi.org/10.5281/zenodo.19553941)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)
[![Status: v0.1.0](https://img.shields.io/badge/status-v0.1.0-orange)](https://github.com/Wangziqi0/Shape-CFD/releases)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--8344--1149-a6ce39)](https://orcid.org/0009-0008-8344-1149)
[![Manuscript](https://img.shields.io/badge/manuscript-under_review_at_IPM-yellow)](#manuscript)

**Documents Are Shapes, Not Points.**

A training-free geometric post-processing pipeline that improves dense retrieval by treating LLM hidden states as point clouds.

**Keywords**: PQ-Chamfer distance, graph Laplacian smoothing, training-free retrieval, dense retrieval, BEIR, point cloud, Chamfer matching, reranking, LLM hidden states.

## Results

| Dataset | #Docs | Cosine | Ours Best | Gain |
|---------|-------|--------|-----------|------|
| NFCorpus | 2,473 | 0.2195 | **0.3271** | +49.0% |
| SciFact | 3,752 | 0.4483 | **0.4827** | +7.7% |
| ArguAna | 8,674 | 0.3047 | **0.4417** | +45.0% |
| SCIDOCS | 25,337 | 0.1110 | **0.2147** | +93.5% |
| FiQA | 56,391 | 0.1683 | **0.3977** | +136.2% |
| Quora | 522,931 | 0.6370 | **0.6749** | +6.0% |

Surpasses BGE-large-en-v1.5 on FiQA and SCIDOCS without any retrieval-specific training.

## Pipeline

```
Query tokens -> Centroid Coarse Filter -> PQ-Chamfer Rerank -> Graph Laplacian Smoothing -> Score Fusion -> Top-10
```

## Key Ideas

1. **PQ-Chamfer Distance**: Split 4096d vectors into 64x64d subspaces, compute cosine independently, aggregate via Chamfer matching
2. **Graph Laplacian Smoothing**: Build KNN graph on candidates, propagate scores through neighborhood
3. **Training-Free**: Works on any LLM's hidden states (verified on Qwen3-8B, BGE-M3, BGE-large)

## Repository Structure

```
rust-engine/          -- Rust core (NAPI bindings for Node.js)
  src/
    lib.rs            -- NAPI exports
    pq_chamfer.rs     -- PQ subspace distance
    token_chamfer.rs  -- Token-level two-stage retrieval
    cloud_store.rs    -- Point cloud SQLite storage
    vt_distance.rs    -- VT-Aligned distance + cosine ranking
    pde.rs            -- Graph Laplacian smoothing (+ legacy PDE)
    bin/extract_tokens.rs -- High-performance token extraction tool
benchmark/            -- BEIR benchmark scripts (Node.js + Python)
paper/                -- LaTeX source + PDF
```

## Quick Start

### Prerequisites
- Rust 1.75+ with NAPI-RS
- Node.js 18+
- An embedding model (llama.cpp recommended)

### Build
```bash
cd rust-engine
npm install
npm run build
```

### Run Benchmark
```bash
# Prepare data (requires embedding server)
python3 benchmark/beir_encode_turbo.py scifact

# Build point clouds
node benchmark/build_clouds.js beir_data/scifact

# Run benchmark
node benchmark/beir_multi_bench.js scifact 55,100,200
```

<a id="manuscript"></a>
## Manuscript

The manuscript describing this work is currently **under review at Information Processing & Management** (submission ID: `IPM-D-26-02154`, submitted 2026-04-04, required reviews in progress).

## Citation

If you use Shape-CFD in academic work:

```bibtex
@software{chen2026shapecfd,
  author       = {Chen, Yifan},
  title        = {Shape-CFD: Training-Free Retrieval via PQ-Chamfer Distance and Graph Smoothing},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.19553941},
  url          = {https://doi.org/10.5281/zenodo.19553941},
  orcid        = {0009-0008-8344-1149}
}
```

**Note on DOIs**:
- Concept DOI: [`10.5281/zenodo.19553941`](https://doi.org/10.5281/zenodo.19553941) — use for general academic citation (always points to the latest version)
- Version DOI: [`10.5281/zenodo.19553942`](https://doi.org/10.5281/zenodo.19553942) — use when reproducibility to v0.1.0 matters

Structured citation metadata (supported by GitHub, Zenodo, Zotero) is in [`CITATION.cff`](CITATION.cff). When the IPM manuscript is published, the journal citation should be preferred.

## License

[Business Source License 1.1](LICENSE) -- free for non-production use. Converts to Apache 2.0 on 2030-03-31.

For commercial licensing inquiries, please open an issue.
