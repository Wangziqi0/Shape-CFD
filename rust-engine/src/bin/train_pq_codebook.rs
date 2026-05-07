//! train_pq_codebook — Standalone binary to train PQ codebook from token-level sqlite.
//!
//! 利用 law-vexus 已有 PQ codebook 训练 building blocks:
//!   - CloudStore::load_from_sqlite (读 4096-d f32 token vectors from token_clouds.sqlite)
//!   - TokenInvertedIndex::build (64 subspaces × 256 clusters K-means × 20 iter, parallel via rayon)
//!   - PqCodebook::from_flat (export codebook to 64 × 256 × 64 = 4 MB flat layout)
//!   - PqStore::encode_from_cloud_store (encode all tokens to PQ codes, 256× compression)
//!
//! Usage:
//!   cargo run --release --bin train_pq_codebook -- \
//!     --corpus nfcorpus \
//!     --token-sqlite /home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus/token_clouds.sqlite \
//!     --out-codebook /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/pq_codebook/nfcorpus.bin \
//!     --out-pq-store /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/pq_store_qwen3/nfcorpus.sqlite

use anyhow::{Context, Result};
use clap::Parser;
use law_vexus::cloud_store::CloudStore;
use law_vexus::inverted_index::{TokenInvertedIndex, NUM_CENTROIDS};
use law_vexus::pq_chamfer::{NUM_SUBSPACES, SUB_DIM};
use law_vexus::pq_store::{PqCodebook, PqStore};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "train_pq_codebook", about = "Train PQ codebook on Qwen3 4096-d token-level data")]
struct Args {
    #[arg(long, help = "corpus name (just for logging)")]
    corpus: String,
    #[arg(long, help = "input: token_clouds.sqlite (chunks: id, file_id, chunk_text, vector(BLOB f32 4096-d))")]
    token_sqlite: PathBuf,
    #[arg(long, help = "output: codebook.bin (64 * 256 * 64 = 1M floats = 4 MB binary file)")]
    out_codebook: PathBuf,
    #[arg(long, help = "(optional) output: pq_store.sqlite (encoded all tokens with PQ codes)")]
    out_pq_store: Option<PathBuf>,
    #[arg(long, default_value_t = 0, help = "(optional) NUMA node id for thread pinning hint (0..7 for 7B13 NPS4)")]
    numa_node: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_total = Instant::now();

    eprintln!("==========================================");
    eprintln!("train_pq_codebook (corpus={}, NUMA hint={})", args.corpus, args.numa_node);
    eprintln!("  subspaces={} sub_dim={} centroids={}",
             NUM_SUBSPACES, SUB_DIM, NUM_CENTROIDS);
    eprintln!("  expected codebook size: {} bytes ({:.1} MB)",
             NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM * 4,
             NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM * 4 / 1024 / 1024);
    eprintln!("==========================================");

    // ---------- Phase 1: load token cloud from sqlite ----------
    let t = Instant::now();
    eprintln!("[Phase 1/4] Loading token cloud sqlite: {}", args.token_sqlite.display());
    let store = CloudStore::load_from_sqlite(&args.token_sqlite.display().to_string())
        .map_err(|e| anyhow::anyhow!("CloudStore load error: {}", e))
        .context("Phase 1: failed to load token sqlite")?;
    let n_docs = store.doc_count();
    eprintln!(
        "[Phase 1/4] Loaded {} documents in {:.1}s",
        n_docs,
        t.elapsed().as_secs_f64()
    );

    // ---------- Phase 2: train inverted index (PQ codebook) ----------
    let t = Instant::now();
    eprintln!("[Phase 2/4] Training inverted index (PQ codebook K-means)...");
    eprintln!("  64 subspaces × {} clusters × 20 iters, parallel via rayon", NUM_CENTROIDS);
    let inv = TokenInvertedIndex::build(&store);
    eprintln!(
        "[Phase 2/4] Trained in {:.1}s ({:.1} min)",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() / 60.0
    );

    // ---------- Phase 3: export codebook to flat binary ----------
    let t = Instant::now();
    eprintln!("[Phase 3/4] Exporting flat codebook...");
    let codebook_flat = inv.export_codebook_flat();
    let expected = NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM;
    if codebook_flat.len() != expected {
        anyhow::bail!(
            "codebook size mismatch: got {} floats, expected {}",
            codebook_flat.len(),
            expected
        );
    }
    let codebook_bytes: Vec<u8> = codebook_flat
        .iter()
        .flat_map(|f: &f32| f.to_le_bytes())
        .collect();
    if let Some(parent) = args.out_codebook.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_codebook, &codebook_bytes)
        .with_context(|| format!("write codebook to {}", args.out_codebook.display()))?;
    eprintln!(
        "[Phase 3/4] Saved codebook ({} bytes = {:.2} MB) in {:.1}s -> {}",
        codebook_bytes.len(),
        codebook_bytes.len() as f64 / 1024.0 / 1024.0,
        t.elapsed().as_secs_f64(),
        args.out_codebook.display()
    );

    // ---------- Phase 4 (optional): encode + save PQ store ----------
    if let Some(out_pq) = args.out_pq_store {
        let t = Instant::now();
        eprintln!("[Phase 4/4] Encoding PQ store (256× compression)...");
        let codebook = PqCodebook::from_flat(codebook_flat);
        let pq = PqStore::encode_from_cloud_store(&store, &codebook);
        eprintln!(
            "[Phase 4/4] Encoded in {:.1}s, memory usage {} bytes ({:.2} MB)",
            t.elapsed().as_secs_f64(),
            pq.memory_usage(),
            pq.memory_usage() as f64 / 1024.0 / 1024.0
        );

        let t = Instant::now();
        eprintln!("[Phase 4/4] Saving pq_store sqlite -> {}", out_pq.display());
        if let Some(parent) = out_pq.parent() {
            std::fs::create_dir_all(parent)?;
        }
        pq.save_to_sqlite(&out_pq.display().to_string())
            .map_err(|e| anyhow::anyhow!("save_to_sqlite: {}", e))?;
        let pq_size_mb = std::fs::metadata(&out_pq)?.len() as f64 / 1024.0 / 1024.0;
        eprintln!(
            "[Phase 4/4] Saved pq_store sqlite in {:.1}s ({:.1} MB)",
            t.elapsed().as_secs_f64(),
            pq_size_mb
        );
    } else {
        eprintln!("[Phase 4/4] Skipped (--out-pq-store not given)");
    }

    eprintln!("==========================================");
    eprintln!(
        "DONE corpus={} total {:.1}s ({:.1} min)",
        args.corpus,
        t_total.elapsed().as_secs_f64(),
        t_total.elapsed().as_secs_f64() / 60.0
    );
    eprintln!("==========================================");
    Ok(())
}
