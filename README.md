# Shape-CFD: Physics-Inspired Reranking via Token-Level Point Clouds

> **+47.2% NDCG@10 on NFCorpus. Zero training. Zero GPU reranking. 26ms CPU latency.**

Shape-CFD is a physics-inspired post-retrieval reranking framework that replaces the entire cosine-based dense retrieval paradigm with token-level point cloud matching and PDE-based score fusion.

## Key Results

| Method | NFCorpus NDCG@10 | vs Cosine | Latency |
|:-------|:----------------:|:---------:|:-------:|
| Cosine baseline | 0.2195 | — | 20ms |
| BM25 | 0.2573 | +17.2% | — |
| Shape-CFD v10 (sentence PDE) | 0.2852 | +29.9% | 26ms |
| Token Chamfer (full scan) | 0.3214 | +46.4% | 305ms |
| Token 2-Stage (centroid coarse) | 0.3220 | +46.7% | 23ms |
| **Fusion (token + PDE, lambda=0.7)** | **0.3232** | **+47.2%** | **26ms** |

Complete 10-method ablation in the [paper](paper/Shape-CFD_v5_Chen_Yifan.pdf).

## What's New

- **Documents are point clouds, not vectors.** Each document is represented as ~356 token-level 4096d points extracted from the embedding model's hidden states. Queries are also expanded to ~6 token points.

- **Chamfer distance replaces cosine everywhere.** PQ-Chamfer computes symmetric nearest-neighbor matching across 64 independent subspaces — equivalent to zero-parameter hard cross-attention under the Unbalanced Optimal Transport (UOT) framework.

- **PDE convection-diffusion as orthogonal signal fusion.** The sentence-level PDE captures global semantic propagation structure on the document graph, while token Chamfer captures local matching quality. Two orthogonal views combined via score interpolation.

- **Zero cosine, zero cross-encoder, zero training.** The entire pipeline from coarse retrieval to final ranking uses no cosine similarity, no GPU-based reranker, and no task-specific training.

## Architecture

```
                    Offline (one-time)
    ┌─────────────────────────────────────────────┐
    │  Documents → Embedding Model (pooling=none)  │
    │  → per-token hidden states (4096d)           │
    │  → SQLite storage (f32 BLOB, 14GB)           │
    └─────────────────────────────────────────────┘

                    Online (per query)
    ┌─────────────────────────────────────────────┐
    │  Query → token point cloud (~6 points)       │
    │                                              │
    │  Stage 1: Token centroid coarse filter        │
    │           (top-100, <1ms)                    │
    │                                              │
    │  Stage 2: Full Token PQ-Chamfer re-ranking   │
    │           (top-100 → top-55, ~22ms)          │
    │                                              │
    │  Fusion: 0.7 × token_score + 0.3 × PDE_score│
    │          (26ms total, parallel execution)    │
    └─────────────────────────────────────────────┘
```

## Core Innovations

| Innovation | Description | Impact |
|:-----------|:------------|:-------|
| V1: Document = Point Cloud | Multi-sentence vectors form a "shape" | Framework foundation |
| V2: PDE Convection-Diffusion | Conservative upwind scheme on KNN graph | Core reranking |
| V4: Chamfer Distance | Symmetric point cloud matching | +14.3% vs cosine |
| V8: PQ-Chamfer 64x64 | Subspace decomposition breaks concentration | +24.3% |
| V11: Token-Level Point Clouds | Per-token hidden states as dense clouds | **+46.4%** |
| V13: VT-Aligned Virtual Tokens | PQ subspaces as semantic dimensions | +27.4% |
| Fusion: PDE as Orthogonal Signal | Token Chamfer + sentence PDE integration | **+47.2%** |

## Quick Start

### Prerequisites

- Rust (nightly or stable 1.75+)
- Node.js 18+
- An embedding model server with `--pooling none` support (e.g., llama.cpp)

### Build

```bash
cd rust-engine
cargo build --release
```

### Extract Token Embeddings

```bash
# Start embedding server with per-token output
llama-server --model your-model.gguf --embedding --pooling none --port 8081

# Extract token embeddings for your corpus
cargo run --release --bin extract_tokens -- \
  --mode corpus \
  --input corpus.jsonl \
  --id-map id_map.json \
  --output token_clouds.sqlite \
  --api-url http://localhost:8081/embedding
```

### Run Benchmark (NFCorpus)

```bash
cd benchmark
node beir_token_bench.js
```

## Project Structure

```
shape-cfd/
├── paper/                          # Paper PDF, DOCX, source
│   ├── Shape-CFD_v5_Chen_Yifan.pdf
│   ├── Shape-CFD_v5_Chen_Yifan.docx
│   └── paper.md                    # Paper source (Markdown + LaTeX math)
├── rust-engine/                    # Core Rust engine
│   ├── src/
│   │   ├── lib.rs                  # NAPI entry point + pipeline orchestration
│   │   ├── token_chamfer.rs        # Token-level PQ-Chamfer distance
│   │   ├── cloud_store.rs          # Document point cloud storage (SQLite)
│   │   ├── vt_distance.rs          # VT-Aligned distance computation
│   │   ├── pde.rs                  # PDE solver (KNN graph + advection-diffusion)
│   │   ├── pq_chamfer.rs           # PQ-Chamfer subspace distance
│   │   └── bin/
│   │       └── extract_tokens.rs   # Token embedding extraction tool
│   └── Cargo.toml
├── benchmark/                      # BEIR benchmark scripts
│   ├── beir_token_bench.js         # V11 token-level benchmark
│   └── beir_rust_parallel.js       # Sentence-level parallel benchmark
├── LICENSE                         # BSL 1.1 (converts to Apache 2.0 in 2030)
└── README.md
```

## Paper

**AD-Rank & Shape-CFD: Physics-Inspired Reranking via Conservative Advection-Diffusion on Token-Level Point Clouds for Dense Retrieval**

Chen, Yifan. March 2026. v5: Token-Level Point Clouds & PDE Orthogonal Fusion.

- [PDF](paper/Shape-CFD_v5_Chen_Yifan.pdf)
- [Zenodo DOI: 10.5281/zenodo.19347363](https://doi.org/10.5281/zenodo.19347363)

## Citation

```bibtex
@misc{chen2026shapecfd,
  title   = {AD-Rank \& Shape-CFD: Physics-Inspired Reranking via Conservative
             Advection-Diffusion on Token-Level Point Clouds for Dense Retrieval},
  author  = {Chen, Yifan},
  year    = {2026},
  doi     = {10.5281/zenodo.19347363},
  url     = {https://doi.org/10.5281/zenodo.19347363}
}
```

## License

[Business Source License 1.1](LICENSE) — free for non-production use. Converts to Apache 2.0 on 2030-03-31.

For commercial licensing inquiries, please open an issue.

## Author

**Chen, Yifan** — Independent researcher. This work was conducted entirely independently without institutional affiliation.

Contact: Open an issue on this repository.
