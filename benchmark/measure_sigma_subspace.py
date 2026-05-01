#!/usr/bin/env python3
"""
measure_sigma_subspace.py — Senior Reviewer Issue 1.2 fix
====================================================================

Empirical measurement of pairwise-cosine standard deviation $\sigma$
on Qwen3-8B sentence-level centroid embeddings from NFCorpus, broken
down by:
  (a) full 4096-d cosine
  (b) per-subspace 64-d cosine (64 contiguous subspaces of 64-d each)

Replaces the hand-wave $\sigma \approx 1/\sqrt{d}$ claim in §3.2 with
measured values; tests whether Qwen3-8B's contextual embeddings are
"approximately uniform in information content" across subspaces (the
claim made in the original paper).

Output: benchmark/data/results/sigma_subspace_measurement.json
"""
import json
import argparse
from pathlib import Path
import numpy as np


def load_vectors(jsonl_path, limit=2000):
    vecs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            v = rec.get("vector") or rec.get("emb")
            if v is None:
                continue
            vecs.append(v)
    return np.asarray(vecs, dtype=np.float32)


def pairwise_cosine_stats(vecs, n_pairs=10000, seed=42):
    """Sample n_pairs random pairs and compute cosine, return mean+std."""
    n = len(vecs)
    rng = np.random.default_rng(seed)
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.maximum(norms, 1e-12)
    # sample pairs
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    cos = np.einsum("kd,kd->k", vecs_n[i], vecs_n[j])
    return float(cos.mean()), float(cos.std()), int(len(cos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_vectors",
                   default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data/nfcorpus/corpus_vectors.jsonl")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--n_pairs", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--M", type=int, default=64, help="number of subspaces")
    p.add_argument("--out",
                   default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/sigma_subspace_measurement.json")
    args = p.parse_args()

    print(f"Loading {args.limit} vectors from {args.corpus_vectors}...")
    V = load_vectors(args.corpus_vectors, limit=args.limit)
    if V.shape[0] < 100:
        raise SystemExit(f"too few vectors: {V.shape}")
    n, d = V.shape
    d_s = d // args.M
    print(f"  {n} vectors of dim {d}, partitioned into {args.M} subspaces of dim {d_s}")

    # full cosine
    mean_full, std_full, n_used = pairwise_cosine_stats(V, n_pairs=args.n_pairs, seed=args.seed)
    print(f"\nFull {d}-d cosine: mean={mean_full:.4f}, std (sigma_full) = {std_full:.4f}")
    print(f"  Theoretical 1/sqrt(d) = {1.0/np.sqrt(d):.4f}; ratio measured/theoretical = {std_full/(1.0/np.sqrt(d)):.2f}x")

    # per-subspace
    sub_stds = []
    sub_means = []
    for s in range(args.M):
        Vs = V[:, s * d_s:(s + 1) * d_s]
        m, sd, _ = pairwise_cosine_stats(Vs, n_pairs=args.n_pairs, seed=args.seed + s)
        sub_stds.append(sd)
        sub_means.append(m)

    sub_stds = np.asarray(sub_stds)
    sub_means = np.asarray(sub_means)
    print(f"\nPer-subspace ({args.M} subspaces of {d_s}-d) cosine std:")
    print(f"  mean across subspaces:   {sub_stds.mean():.4f}")
    print(f"  std across subspaces:    {sub_stds.std():.4f}")
    print(f"  min subspace sigma:      {sub_stds.min():.4f}")
    print(f"  max subspace sigma:      {sub_stds.max():.4f}")
    print(f"  Theoretical 1/sqrt(d_s) = {1.0/np.sqrt(d_s):.4f}")
    print(f"  ratio measured/theoretical (mean) = {sub_stds.mean()/(1.0/np.sqrt(d_s)):.2f}x")
    print(f"\nSubspace sigma / full sigma ratio: {sub_stds.mean()/std_full:.2f}x  "
          f"(paper claims 7.8x; theoretical sqrt(d/d_s) = sqrt({args.M}) = {np.sqrt(args.M):.2f}x)")

    out = {
        "n_vectors": int(n),
        "dim_total": int(d),
        "M_subspaces": int(args.M),
        "d_subspace": int(d_s),
        "n_pairs_per_test": int(args.n_pairs),
        "seed": int(args.seed),
        "full_cosine": {"mean": mean_full, "std": std_full, "theoretical_1_over_sqrt_d": 1.0 / np.sqrt(d)},
        "subspace_cosine": {
            "mean_of_means": float(sub_means.mean()),
            "std_of_means": float(sub_means.std()),
            "mean_of_stds": float(sub_stds.mean()),
            "std_of_stds": float(sub_stds.std()),
            "min_std": float(sub_stds.min()),
            "max_std": float(sub_stds.max()),
            "theoretical_1_over_sqrt_d_s": 1.0 / np.sqrt(d_s),
            "per_subspace_stds": [float(x) for x in sub_stds],
            "per_subspace_means": [float(x) for x in sub_means],
        },
        "ratio_subspace_to_full": float(sub_stds.mean() / std_full),
        "theoretical_ratio_sqrt_M": float(np.sqrt(args.M)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
