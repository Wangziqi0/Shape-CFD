//! train_pq_codebook (standalone, no law-vexus dependency).
//!
//! Reads token_clouds.sqlite (chunks: id, file_id, chunk_text, vector(BLOB f32 4096-d)),
//! trains 64 subspace × 256 cluster × 20 iter K-means in parallel via rayon,
//! exports flat codebook binary (64 * 256 * 64 = 1M floats = 4 MB).
//!
//! Usage:
//!   ./train_pq_codebook --corpus nfcorpus \
//!     --token-sqlite /path/to/token_clouds.sqlite \
//!     --out-codebook /path/to/codebook_nfcorpus.bin

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use rusqlite::Connection;
use std::path::PathBuf;
use std::time::Instant;

const NUM_SUBSPACES: usize = 64;
const SUB_DIM: usize = 64;
const FULL_DIM: usize = NUM_SUBSPACES * SUB_DIM; // 4096
const NUM_CENTROIDS: usize = 256;
const KMEANS_MAX_ITER: usize = 20;

#[derive(Parser)]
#[command(name = "train_pq_codebook", about = "PQ codebook training on Qwen3 4096-d token sqlite")]
struct Args {
    #[arg(long)] corpus: String,
    #[arg(long)] token_sqlite: PathBuf,
    #[arg(long)] out_codebook: PathBuf,
    #[arg(long, default_value_t = 0)] numa_node: usize,
    #[arg(long, help = "Cap number of token vectors loaded (for sanity test on small subset)")]
    max_tokens: Option<usize>,
}

#[inline(always)]
fn l2_sq_64d(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

/// K-means on 64-d vectors, K=256, max_iter=20.
/// data: slice of &[f32] each of length SUB_DIM.
/// Returns Vec<Vec<f32>> shape (k, SUB_DIM) — k centroid vectors.
fn kmeans_64d(data: &[&[f32]], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if n < k {
        // not enough data, replicate
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            out.push(data[i % n].to_vec());
        }
        return out;
    }

    // init: stride sample
    let step = n / k;
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| data[i * step].to_vec()).collect();
    let mut assignments = vec![0u32; n];

    for _it in 0..max_iter {
        // E-step: assign
        let new_assignments: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = data[i];
                let mut best = 0u32;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let d = l2_sq_64d(v, &centroids[c]);
                    if d < best_d {
                        best_d = d;
                        best = c as u32;
                    }
                }
                best
            })
            .collect();

        let changed = assignments
            .iter()
            .zip(&new_assignments)
            .filter(|(a, b)| a != b)
            .count();
        assignments = new_assignments;

        if changed == 0 {
            break;
        }

        // M-step: update centroids
        let mut new_centroids: Vec<Vec<f64>> = vec![vec![0.0f64; SUB_DIM]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            let v = data[i];
            let target = &mut new_centroids[a as usize];
            for d in 0..SUB_DIM {
                target[d] += v[d] as f64;
            }
            counts[a as usize] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for d in 0..SUB_DIM {
                    centroids[c][d] = (new_centroids[c][d] * inv) as f32;
                }
            }
        }
    }
    centroids
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_total = Instant::now();
    eprintln!("==========================================");
    eprintln!("train_pq_codebook (standalone)");
    eprintln!("  corpus={} numa_hint={}", args.corpus, args.numa_node);
    eprintln!("  subspaces={} sub_dim={} centroids={} max_iter={}",
        NUM_SUBSPACES, SUB_DIM, NUM_CENTROIDS, KMEANS_MAX_ITER);
    eprintln!("==========================================");

    // ----- Phase 1: load all token vectors from sqlite -----
    let t = Instant::now();
    eprintln!("[Phase 1/3] Loading {}", args.token_sqlite.display());
    let conn = Connection::open(&args.token_sqlite).context("open sqlite")?;
    let mut stmt = conn
        .prepare("SELECT id, vector FROM chunks WHERE vector IS NOT NULL ORDER BY file_id, id")?;

    // Load into one big Vec<f32> of shape (n_tokens, FULL_DIM)
    let mut all_vectors: Vec<f32> = Vec::new();
    let mut n_tokens = 0usize;
    let max = args.max_tokens.unwrap_or(usize::MAX);

    let rows = stmt.query_map([], |row| {
        let _id: i64 = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        Ok(blob)
    })?;
    for row in rows {
        let blob = row?;
        if blob.len() != FULL_DIM * 4 {
            anyhow::bail!(
                "Token {} blob size {} != expected {} (4096 f32)",
                n_tokens,
                blob.len(),
                FULL_DIM * 4
            );
        }
        all_vectors.reserve(FULL_DIM);
        for chunk in blob.chunks_exact(4) {
            all_vectors.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        n_tokens += 1;
        if n_tokens >= max {
            eprintln!("  capped at max_tokens={}", max);
            break;
        }
        if n_tokens % 100_000 == 0 {
            eprintln!("  loaded {} tokens ({:.1}s)", n_tokens, t.elapsed().as_secs_f64());
        }
    }
    eprintln!(
        "[Phase 1/3] Loaded {} tokens × {}-d (= {} f32, {:.1} GB) in {:.1}s",
        n_tokens,
        FULL_DIM,
        all_vectors.len(),
        all_vectors.len() as f64 * 4.0 / 1024.0 / 1024.0 / 1024.0,
        t.elapsed().as_secs_f64()
    );

    if n_tokens == 0 {
        anyhow::bail!("no tokens loaded");
    }

    // ----- Phase 2: train 64 subspaces in parallel via rayon -----
    let t = Instant::now();
    eprintln!("[Phase 2/3] Training 64 subspace K-means × K=256 × {} iter (parallel rayon)...", KMEANS_MAX_ITER);

    // For each subspace s, we need slices into all_vectors at offsets [s*SUB_DIM..(s+1)*SUB_DIM] per token.
    // Strategy: build per-subspace owned buffers (memory: 64 × n_tokens × 64 × 4 bytes = same as input).
    // For 880k tokens × 64-d × 4 bytes per subspace = 224 MB per subspace × 64 = 14 GB. Too much.
    //
    // Better: process each subspace serially in the OUTER loop, parallelize K-means INNER (E-step).
    let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(NUM_SUBSPACES);

    for s in 0..NUM_SUBSPACES {
        let t_s = Instant::now();
        // Build sub-vector slice references in this subspace
        let sub_refs: Vec<&[f32]> = (0..n_tokens)
            .map(|i| {
                let off = i * FULL_DIM + s * SUB_DIM;
                &all_vectors[off..off + SUB_DIM]
            })
            .collect();

        let centroids = kmeans_64d(&sub_refs, NUM_CENTROIDS, KMEANS_MAX_ITER);
        codebooks.push(centroids);

        if s % 8 == 0 || s == NUM_SUBSPACES - 1 {
            eprintln!(
                "  subspace {}/{} done in {:.1}s (running total {:.1}s)",
                s + 1,
                NUM_SUBSPACES,
                t_s.elapsed().as_secs_f64(),
                t.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "[Phase 2/3] Training done in {:.1}s ({:.1} min)",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() / 60.0
    );

    // ----- Phase 3: save flat codebook binary -----
    let t = Instant::now();
    let mut flat: Vec<f32> = Vec::with_capacity(NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM);
    for s in 0..NUM_SUBSPACES {
        for c in 0..NUM_CENTROIDS {
            flat.extend_from_slice(&codebooks[s][c]);
        }
    }
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    if let Some(parent) = args.out_codebook.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out_codebook, &bytes)
        .with_context(|| format!("write {}", args.out_codebook.display()))?;
    eprintln!(
        "[Phase 3/3] Saved {} bytes ({:.1} MB) in {:.1}s -> {}",
        bytes.len(),
        bytes.len() as f64 / 1024.0 / 1024.0,
        t.elapsed().as_secs_f64(),
        args.out_codebook.display()
    );

    eprintln!("==========================================");
    eprintln!(
        "DONE corpus={} n_tokens={} total {:.1}s ({:.1} min)",
        args.corpus,
        n_tokens,
        t_total.elapsed().as_secs_f64(),
        t_total.elapsed().as_secs_f64() / 60.0
    );
    eprintln!("==========================================");
    Ok(())
}
