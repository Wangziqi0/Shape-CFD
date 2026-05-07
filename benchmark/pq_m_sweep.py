#!/usr/bin/env python3
"""
PQ subspace M sweep: measure sigma_sub/sigma_full ratio at M in {16, 32, 64, 128}
on Qwen3-8B NFCorpus sentence-level centroid embeddings. Validates Theorem 1
closed-form sigma_sub/sigma_full = sqrt(M / (1 + (M-1)*rho)) under different
subspace decompositions.

Output: benchmark/data/results/pq_m_sweep_results.json
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/amd/HEZIMENG/Shape-CFD/benchmark')

VEC_PATH = Path("/home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus/corpus_vectors.jsonl")
OUT_PATH = Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/pq_m_sweep_results.json")
M_VALUES = [16, 32, 64, 128]
N_PAIRS = 10000
SEED = 42


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


def measure(vecs, M, n_pairs=N_PAIRS, seed=SEED):
    n, d = vecs.shape
    if d % M != 0:
        return None
    d_s = d // M

    # normalize once
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.maximum(norms, 1e-9)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_pairs, 2))
    a = vecs_n[idx[:, 0]]
    b = vecs_n[idx[:, 1]]

    # full-dim cosine
    cos_full = np.sum(a * b, axis=1)
    sigma_full = float(np.std(cos_full))
    mean_full = float(np.mean(cos_full))

    # per-subspace cosines
    sub_sigmas = []
    sub_means = []
    sub_cos_all = []  # for correlation
    for s in range(M):
        a_s = a[:, s*d_s:(s+1)*d_s]
        b_s = b[:, s*d_s:(s+1)*d_s]
        # per-subspace L2 normalize
        an_s = a_s / np.maximum(np.linalg.norm(a_s, axis=1, keepdims=True), 1e-9)
        bn_s = b_s / np.maximum(np.linalg.norm(b_s, axis=1, keepdims=True), 1e-9)
        c_s = np.sum(an_s * bn_s, axis=1)
        sub_sigmas.append(float(np.std(c_s)))
        sub_means.append(float(np.mean(c_s)))
        sub_cos_all.append(c_s)

    sigma_sub_mean = float(np.mean(sub_sigmas))
    ratio = sigma_sub_mean / sigma_full if sigma_full > 0 else float('inf')

    # Average pairwise correlation between subspace cosines (rho in Theorem 1)
    sub_cos_arr = np.array(sub_cos_all)  # (M, n_pairs)
    corr_matrix = np.corrcoef(sub_cos_arr)  # (M, M)
    # Average off-diagonal correlation
    rho = float((corr_matrix.sum() - M) / (M * (M - 1))) if M > 1 else 0.0

    # Theorem 1 prediction: sigma_sub/sigma_full = sqrt(M / (1 + (M-1)*rho))
    theorem_pred = (M / (1 + (M - 1) * rho)) ** 0.5 if rho > -1/(M-1) else float('inf')

    return {
        "M": M,
        "d_s": d_s,
        "sigma_full": sigma_full,
        "mean_full": mean_full,
        "sigma_sub_mean": sigma_sub_mean,
        "sigma_sub_min": float(min(sub_sigmas)),
        "sigma_sub_max": float(max(sub_sigmas)),
        "ratio_measured": ratio,
        "rho_avg": rho,
        "theorem_pred_ratio": theorem_pred,
        "abs_error": abs(ratio - theorem_pred),
        "rel_error_pct": 100 * abs(ratio - theorem_pred) / theorem_pred if theorem_pred > 0 else 0,
    }


def main():
    print(f"[pq-m-sweep] Loading vectors from {VEC_PATH}")
    vecs = load_vectors(str(VEC_PATH), limit=2000)
    print(f"[pq-m-sweep] Loaded {vecs.shape[0]} vectors of dim {vecs.shape[1]}")

    results = {"corpus": "nfcorpus", "model": "Qwen3-8B", "n_vectors": int(vecs.shape[0]),
               "n_pairs": N_PAIRS, "seed": SEED, "M_sweep": []}

    print(f"\n{'M':<6} {'d_s':<6} {'σ_full':<10} {'σ_sub':<10} {'ratio':<10} {'ρ_avg':<10} {'Thm1 pred':<12} {'rel err%':<10}")
    print("-" * 80)
    for M in M_VALUES:
        r = measure(vecs, M)
        if r is None:
            print(f"  M={M} skipped (d not divisible)")
            continue
        results["M_sweep"].append(r)
        print(f"{r['M']:<6} {r['d_s']:<6} {r['sigma_full']:<10.4f} {r['sigma_sub_mean']:<10.4f} "
              f"{r['ratio_measured']:<10.3f}× {r['rho_avg']:<10.4f} {r['theorem_pred_ratio']:<12.3f}× "
              f"{r['rel_error_pct']:<10.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[pq-m-sweep] Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
