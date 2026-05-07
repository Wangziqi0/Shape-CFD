"""
pq_codebook_quality_report.py — 7B13 NUMA 4-7 reconstructed retrieval quality benchmark.

For each corpus codebook (4 MB binary):
  - Load 10000 random tokens from token_clouds.sqlite
  - PQ-encode each token (find nearest cluster per subspace) -> 64-byte code
  - Reconstruct from code -> 4096-d approximated vector
  - Report: mean L2 reconstruction error, cosine similarity to original
  - Report: pairwise Chamfer distance preservation (10000 random pairs)
  - Report: storage compression ratio (16384 -> 64 bytes = 256x)

Output: paper §6 storage future-work supplementary numbers
"""

import json
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np

NUM_SUBSPACES = 64
SUB_DIM = 64
FULL_DIM = NUM_SUBSPACES * SUB_DIM  # 4096
NUM_CENTROIDS = 256
SAMPLE_SIZE = 10000
N_PAIRS = 10000

CORPORA = ["nfcorpus", "scifact", "arguana", "scidocs"]  # FiQA pending
DATA_ROOT = Path("/home/amd/HEZIMENG/legal-assistant/beir_data")
CODEBOOK_DIR = Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/pq_codebook")
OUT_PATH = Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/pq_codebook_quality_report_20260504.json")


def load_codebook(path):
    """Load flat binary codebook -> array (NUM_SUBSPACES, NUM_CENTROIDS, SUB_DIM)."""
    with open(path, "rb") as f:
        data = f.read()
    expected = NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM * 4
    assert len(data) == expected, f"codebook size {len(data)} != expected {expected}"
    flat = np.frombuffer(data, dtype=np.float32)
    return flat.reshape(NUM_SUBSPACES, NUM_CENTROIDS, SUB_DIM)


def sample_tokens(sqlite_path, n_sample, seed=42):
    """Sample n_sample tokens from sqlite chunks table, return (n_sample, FULL_DIM) f32."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks WHERE vector IS NOT NULL")
    total = cur.fetchone()[0]
    sample_ids = sorted(rng.choice(total, size=min(n_sample, total), replace=False).tolist())
    out = np.zeros((len(sample_ids), FULL_DIM), dtype=np.float32)

    placeholders = ",".join("?" * len(sample_ids))
    cur.execute(f"SELECT id, vector FROM chunks WHERE id IN ({placeholders}) ORDER BY id", sample_ids)
    for i, (rid, blob) in enumerate(cur.fetchall()):
        if len(blob) != FULL_DIM * 4:
            continue
        out[i] = np.frombuffer(blob, dtype=np.float32)
    conn.close()
    return out


def pq_encode(tokens, codebook):
    """Encode (n, FULL_DIM) -> (n, NUM_SUBSPACES) uint8 codes."""
    n = tokens.shape[0]
    codes = np.zeros((n, NUM_SUBSPACES), dtype=np.uint8)
    for s in range(NUM_SUBSPACES):
        sub = tokens[:, s * SUB_DIM:(s + 1) * SUB_DIM]  # (n, 64)
        cb = codebook[s]  # (256, 64)
        # find nearest centroid: argmin ||sub - cb||²
        dists = ((sub[:, None, :] - cb[None, :, :]) ** 2).sum(axis=-1)
        codes[:, s] = np.argmin(dists, axis=1)
    return codes


def pq_reconstruct(codes, codebook):
    """Reconstruct (n, NUM_SUBSPACES) codes -> (n, FULL_DIM) f32."""
    n = codes.shape[0]
    out = np.zeros((n, FULL_DIM), dtype=np.float32)
    for s in range(NUM_SUBSPACES):
        out[:, s * SUB_DIM:(s + 1) * SUB_DIM] = codebook[s][codes[:, s]]
    return out


def report_corpus(corpus, codebook_path, sqlite_path):
    print(f"\n=== {corpus} ===", file=sys.stderr)
    t = time.time()
    cb = load_codebook(codebook_path)
    print(f"  codebook loaded: {cb.shape} ({cb.nbytes/1024/1024:.2f} MB)", file=sys.stderr)

    print(f"  sampling {SAMPLE_SIZE} tokens from {sqlite_path}...", file=sys.stderr)
    tokens = sample_tokens(sqlite_path, SAMPLE_SIZE)
    n = tokens.shape[0]
    print(f"  sampled {n} tokens in {time.time()-t:.1f}s", file=sys.stderr)

    print(f"  PQ encode + reconstruct...", file=sys.stderr)
    codes = pq_encode(tokens, cb)
    rec = pq_reconstruct(codes, cb)

    # reconstruction error per token
    err = tokens - rec
    l2_err = np.linalg.norm(err, axis=1)
    orig_norm = np.linalg.norm(tokens, axis=1) + 1e-9
    rel_err = l2_err / orig_norm

    # cosine similarity orig vs reconstructed
    cos_sim = np.sum(tokens * rec, axis=1) / (orig_norm * (np.linalg.norm(rec, axis=1) + 1e-9))

    # pairwise Chamfer-style distance preservation: random pairs
    rng = np.random.default_rng(42)
    pair_idx = rng.choice(n, size=(N_PAIRS, 2), replace=True)
    a_orig, b_orig = tokens[pair_idx[:, 0]], tokens[pair_idx[:, 1]]
    a_rec, b_rec = rec[pair_idx[:, 0]], rec[pair_idx[:, 1]]
    # cosine distance = 1 - cos
    def cos_dist(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-9
        nb = np.linalg.norm(b, axis=1) + 1e-9
        return 1.0 - np.sum(a * b, axis=1) / (na * nb)
    d_orig = cos_dist(a_orig, b_orig)
    d_rec = cos_dist(a_rec, b_rec)
    # Pearson correlation
    pearson = float(np.corrcoef(d_orig, d_rec)[0, 1])

    # storage compression
    raw_bytes = FULL_DIM * 4  # 16384
    pq_bytes = NUM_SUBSPACES * 1  # 64
    compression = raw_bytes / pq_bytes  # 256x

    return {
        "corpus": corpus,
        "n_sampled_tokens": int(n),
        "reconstruction_error": {
            "l2_mean": float(l2_err.mean()),
            "l2_std": float(l2_err.std()),
            "l2_p50": float(np.median(l2_err)),
            "l2_p90": float(np.quantile(l2_err, 0.9)),
            "rel_l2_mean": float(rel_err.mean()),
            "rel_l2_std": float(rel_err.std()),
            "cos_sim_mean": float(cos_sim.mean()),
            "cos_sim_std": float(cos_sim.std()),
            "cos_sim_p10": float(np.quantile(cos_sim, 0.1)),
        },
        "distance_preservation": {
            "n_pairs": N_PAIRS,
            "orig_cos_dist_mean": float(d_orig.mean()),
            "rec_cos_dist_mean": float(d_rec.mean()),
            "abs_diff_mean": float(np.abs(d_orig - d_rec).mean()),
            "pearson_r": pearson,
        },
        "storage": {
            "raw_bytes_per_token": raw_bytes,
            "pq_bytes_per_token": pq_bytes,
            "compression_ratio": compression,
            "raw_corpus_GB_estimate": None,  # filled per corpus
            "pq_corpus_MB_estimate": None,
        },
        "wall_time_sec": time.time() - t,
    }


def main():
    out = {
        "config": {
            "num_subspaces": NUM_SUBSPACES,
            "sub_dim": SUB_DIM,
            "num_centroids": NUM_CENTROIDS,
            "sample_size": SAMPLE_SIZE,
            "n_pairs": N_PAIRS,
            "compression_ratio": (FULL_DIM * 4) / NUM_SUBSPACES,
        },
        "per_corpus": {},
    }

    for corpus in CORPORA:
        cb_path = CODEBOOK_DIR / f"{corpus}.bin"
        sqlite_path = DATA_ROOT / corpus / "token_clouds.sqlite"
        if not cb_path.exists():
            print(f"SKIP {corpus}: codebook missing", file=sys.stderr)
            continue
        if not sqlite_path.exists():
            print(f"SKIP {corpus}: sqlite missing", file=sys.stderr)
            continue
        try:
            out["per_corpus"][corpus] = report_corpus(corpus, cb_path, sqlite_path)
        except Exception as e:
            print(f"ERROR {corpus}: {e}", file=sys.stderr)
            out["per_corpus"][corpus] = {"error": str(e)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n=== Quality summary ===", file=sys.stderr)
    for c, r in out["per_corpus"].items():
        if "error" in r:
            print(f"  {c}: ERROR {r['error']}", file=sys.stderr)
            continue
        re = r["reconstruction_error"]
        dp = r["distance_preservation"]
        print(f"  {c}: cos_sim={re['cos_sim_mean']:.4f}±{re['cos_sim_std']:.4f}  "
              f"rel_l2={re['rel_l2_mean']:.3f}  "
              f"pair_r={dp['pearson_r']:.4f}  "
              f"compress={r['storage']['compression_ratio']:.0f}x", file=sys.stderr)
    print(f"  output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
