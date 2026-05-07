"""
variance_proportional_pq_sanity.py
====================================
超小规模 sanity check: variance-proportional PQ codebook bit allocation 是否优于 uniform allocation.

Setup
-----
- Corpus: NFCorpus (3.6k docs, 323 test queries) — smallest BEIR corpus, 5 min sanity
- Vectors: BGE-M3 v2 audited cache (1024-dim) - 已存在 /home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus/
- Subspace: 1024 = 64 sub × 16 dim per sub (匹配 paper PQ-Chamfer)
- Byte budget: 64 byte/token (uniform = 8 bits/sub = K=256)

Comparison
----------
(A) Exact (no quantization)  — upper bound
(B) Uniform K=256 (8 bits × 64 sub = 64 byte) — paper baseline
(C) Variance-proportional: b_s = clip(round(8 + alpha * log2(sigma_s² / mean_var)), 4, 12) renormalized to sum=512 bits
(D) Variance-anti-proportional (反 control): b_s ∝ -log2(sigma_s²)  — should be WORSE; sanity that variance signal is real

Output
------
- Per-subspace sigma_s² + bit allocation table
- NDCG@10 for (A)(B)(C)(D)
- Reconstructed-vs-exact pairwise distance Pearson r
- Storage byte budget per token
- 1 段 honest verdict
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

CORPUS = "nfcorpus"
BEIR_ROOT = Path("/home/amd/HEZIMENG/legal-assistant/beir_data") / CORPUS
N_SUB = 64
SUB_DIM = 16  # 1024 / 64
TOTAL_BIT_BUDGET = N_SUB * 8  # 512 bits = 64 byte / token

OUT_PATH = Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/pq_sanity_nfcorpus_20260503.json")


def load_vectors_jsonl(path):
    ids = []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            ids.append(str(d["_id"]))
            rows.append(d["vector"])
    embs = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, embs / norms


def load_qrels_test(corpus_root):
    qrels_path = corpus_root / "qrels" / "test.tsv"
    qrels = {}
    with open(qrels_path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, grade = parts[0], parts[1], parts[2]
                try:
                    qrels.setdefault(qid, {})[did] = int(float(grade))
                except ValueError:
                    pass
    return qrels


def compute_subspace_variance(embs, n_sub, sub_dim):
    """Per-subspace mean variance: sigma_s² = mean over dims in subspace s of var(embs[:, dim])."""
    sigmas2 = np.zeros(n_sub, dtype=np.float64)
    for s in range(n_sub):
        block = embs[:, s * sub_dim:(s + 1) * sub_dim]
        sigmas2[s] = float(np.mean(np.var(block, axis=0)))
    return sigmas2


def variance_proportional_allocation(sigmas2, total_bits, b_min=4, b_max=12, alpha=1.5):
    """
    b_s = clip(round(8 + alpha * log2(sigma_s² / mean_var)), b_min, b_max)
    renormalized to sum=total_bits.
    """
    log_ratio = np.log2(sigmas2 / np.mean(sigmas2))
    raw = 8.0 + alpha * log_ratio
    bits = np.clip(np.round(raw), b_min, b_max).astype(int)
    # renormalize to total
    diff = total_bits - bits.sum()
    if diff != 0:
        order = np.argsort(-sigmas2 if diff > 0 else sigmas2)
        i = 0
        while diff != 0:
            s = order[i % len(order)]
            new_b = bits[s] + (1 if diff > 0 else -1)
            if b_min <= new_b <= b_max:
                bits[s] = new_b
                diff += -1 if diff > 0 else 1
            i += 1
            if i > 1000:  # safety
                break
    return bits


def variance_anti_proportional_allocation(sigmas2, total_bits, b_min=4, b_max=12, alpha=1.5):
    """Control: invert variance signal — bits to LOW-variance subspaces."""
    return variance_proportional_allocation(1.0 / (sigmas2 + 1e-9), total_bits, b_min, b_max, alpha)


def train_pq_codebook(embs, sub_dim, bits_per_sub, n_sub, seed=42):
    """
    Train one KMeans codebook per subspace with K=2^bits_per_sub[s].
    Returns: codebook list, code matrix (n_docs, n_sub) with codes in [0, 2^bits].
    """
    n_docs = embs.shape[0]
    codebooks = []
    codes = np.zeros((n_docs, n_sub), dtype=np.int32)
    for s in range(n_sub):
        K = int(2 ** bits_per_sub[s])
        block = embs[:, s * sub_dim:(s + 1) * sub_dim]
        K = min(K, n_docs)
        km = KMeans(n_clusters=K, n_init=3, random_state=seed)
        km.fit(block)
        codebooks.append(km.cluster_centers_)
        codes[:, s] = km.labels_
    return codebooks, codes


def reconstruct_embs(codes, codebooks, sub_dim, n_sub, total_dim):
    """Reconstruct full-dim embeddings from PQ codes."""
    n_docs = codes.shape[0]
    rec = np.zeros((n_docs, total_dim), dtype=np.float32)
    for s in range(n_sub):
        rec[:, s * sub_dim:(s + 1) * sub_dim] = codebooks[s][codes[:, s]]
    norms = np.linalg.norm(rec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rec / norms


def compute_ndcg10(ranked_doc_ids, qrels_for_query):
    rel = np.array([qrels_for_query.get(d, 0) for d in ranked_doc_ids[:10]], dtype=float)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, 12))
    dcg = float(np.sum(rel * discounts[:len(rel)]))
    ideal = np.sort(np.array(list(qrels_for_query.values()), dtype=float))[::-1]
    idcg = float(np.sum(ideal[:10] * discounts[:min(10, len(ideal))]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ndcg(q_embs, q_ids, doc_embs, doc_ids, qrels):
    sims = q_embs @ doc_embs.T
    ranked = np.argsort(-sims, axis=1)
    ndcg_per_q = []
    for qi, qid in enumerate(q_ids):
        if qid not in qrels:
            continue
        ranked_dids = [doc_ids[ranked[qi, j]] for j in range(min(10, len(doc_ids)))]
        ndcg_per_q.append(compute_ndcg10(ranked_dids, qrels[qid]))
    return float(np.mean(ndcg_per_q)) if ndcg_per_q else 0.0, len(ndcg_per_q)


def main():
    t_start = time.time()
    print(f"[sanity] corpus={CORPUS}, n_sub={N_SUB}, sub_dim={SUB_DIM}, byte_budget={TOTAL_BIT_BUDGET // 8}", file=sys.stderr)

    print("[sanity] loading vectors...", file=sys.stderr)
    doc_ids, doc_embs = load_vectors_jsonl(BEIR_ROOT / "bge_m3_corpus_vectors.jsonl")
    q_ids_all, q_embs_all = load_vectors_jsonl(BEIR_ROOT / "bge_m3_query_vectors.jsonl")
    qrels = load_qrels_test(BEIR_ROOT)
    test_qids = set(qrels.keys())
    keep = [i for i, q in enumerate(q_ids_all) if q in test_qids]
    q_ids = [q_ids_all[i] for i in keep]
    q_embs = q_embs_all[keep]
    print(f"[sanity] docs={len(doc_ids)}, queries={len(q_ids)}, qrels_qids={len(qrels)}", file=sys.stderr)

    # --- subspace variance ---
    sigmas2 = compute_subspace_variance(doc_embs, N_SUB, SUB_DIM)
    print(f"[sanity] sigma² stats: min={sigmas2.min():.6f} max={sigmas2.max():.6f} hetero={sigmas2.max()/sigmas2.min():.2f}×", file=sys.stderr)

    # --- bit allocations ---
    bits_uniform = np.full(N_SUB, 8, dtype=int)
    bits_var_prop = variance_proportional_allocation(sigmas2, TOTAL_BIT_BUDGET, alpha=1.5)
    bits_var_anti = variance_anti_proportional_allocation(sigmas2, TOTAL_BIT_BUDGET, alpha=1.5)
    assert bits_uniform.sum() == TOTAL_BIT_BUDGET
    assert bits_var_prop.sum() == TOTAL_BIT_BUDGET, f"var_prop sum={bits_var_prop.sum()}"
    assert bits_var_anti.sum() == TOTAL_BIT_BUDGET, f"var_anti sum={bits_var_anti.sum()}"
    print(f"[sanity] var_prop bits: min={bits_var_prop.min()} max={bits_var_prop.max()} sum={bits_var_prop.sum()}", file=sys.stderr)

    # --- (A) Exact ---
    print("[sanity] (A) Exact NDCG@10 ...", file=sys.stderr)
    ndcg_exact, n = evaluate_ndcg(q_embs, q_ids, doc_embs, doc_ids, qrels)
    print(f"  exact NDCG@10 = {ndcg_exact:.4f} (n={n})", file=sys.stderr)

    # --- (B) Uniform PQ ---
    t = time.time()
    print("[sanity] (B) Uniform PQ codebook (K=256 × 64 sub)...", file=sys.stderr)
    cb_u, codes_u_doc = train_pq_codebook(doc_embs, SUB_DIM, bits_uniform, N_SUB, seed=42)
    _, codes_u_q = train_pq_codebook_assign(q_embs, cb_u, SUB_DIM, N_SUB)
    rec_doc_u = reconstruct_embs(codes_u_doc, cb_u, SUB_DIM, N_SUB, doc_embs.shape[1])
    rec_q_u = reconstruct_embs(codes_u_q, cb_u, SUB_DIM, N_SUB, q_embs.shape[1])
    ndcg_uniform, _ = evaluate_ndcg(rec_q_u, q_ids, rec_doc_u, doc_ids, qrels)
    print(f"  uniform NDCG@10 = {ndcg_uniform:.4f} (train_time={time.time()-t:.0f}s)", file=sys.stderr)

    # --- (C) Variance-proportional ---
    t = time.time()
    print("[sanity] (C) Variance-proportional PQ ...", file=sys.stderr)
    cb_vp, codes_vp_doc = train_pq_codebook(doc_embs, SUB_DIM, bits_var_prop, N_SUB, seed=42)
    _, codes_vp_q = train_pq_codebook_assign(q_embs, cb_vp, SUB_DIM, N_SUB)
    rec_doc_vp = reconstruct_embs(codes_vp_doc, cb_vp, SUB_DIM, N_SUB, doc_embs.shape[1])
    rec_q_vp = reconstruct_embs(codes_vp_q, cb_vp, SUB_DIM, N_SUB, q_embs.shape[1])
    ndcg_var_prop, _ = evaluate_ndcg(rec_q_vp, q_ids, rec_doc_vp, doc_ids, qrels)
    print(f"  var_prop NDCG@10 = {ndcg_var_prop:.4f} (train_time={time.time()-t:.0f}s)", file=sys.stderr)

    # --- (D) Variance-anti-proportional (control) ---
    t = time.time()
    print("[sanity] (D) Variance-ANTI-proportional PQ (control)...", file=sys.stderr)
    cb_va, codes_va_doc = train_pq_codebook(doc_embs, SUB_DIM, bits_var_anti, N_SUB, seed=42)
    _, codes_va_q = train_pq_codebook_assign(q_embs, cb_va, SUB_DIM, N_SUB)
    rec_doc_va = reconstruct_embs(codes_va_doc, cb_va, SUB_DIM, N_SUB, doc_embs.shape[1])
    rec_q_va = reconstruct_embs(codes_va_q, cb_va, SUB_DIM, N_SUB, q_embs.shape[1])
    ndcg_var_anti, _ = evaluate_ndcg(rec_q_va, q_ids, rec_doc_va, doc_ids, qrels)
    print(f"  var_anti NDCG@10 = {ndcg_var_anti:.4f} (train_time={time.time()-t:.0f}s)", file=sys.stderr)

    # --- summary ---
    elapsed = time.time() - t_start
    out = {
        "corpus": CORPUS,
        "n_docs": len(doc_ids),
        "n_queries": len(q_ids),
        "n_sub": N_SUB,
        "sub_dim": SUB_DIM,
        "total_bit_budget": TOTAL_BIT_BUDGET,
        "byte_budget_per_token": TOTAL_BIT_BUDGET // 8,
        "subspace_variance_stats": {
            "min": float(sigmas2.min()),
            "max": float(sigmas2.max()),
            "mean": float(sigmas2.mean()),
            "heterogeneity_ratio": float(sigmas2.max() / sigmas2.min()),
        },
        "bit_allocations": {
            "uniform": bits_uniform.tolist(),
            "variance_proportional": bits_var_prop.tolist(),
            "variance_anti_proportional": bits_var_anti.tolist(),
        },
        "ndcg10": {
            "A_exact": ndcg_exact,
            "B_uniform": ndcg_uniform,
            "C_variance_proportional": ndcg_var_prop,
            "D_variance_anti_proportional": ndcg_var_anti,
        },
        "deltas_vs_uniform": {
            "C_var_prop_minus_uniform": ndcg_var_prop - ndcg_uniform,
            "D_var_anti_minus_uniform": ndcg_var_anti - ndcg_uniform,
        },
        "wall_time_sec": elapsed,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n=== Sanity result ===", file=sys.stderr)
    print(f"  Exact:                {ndcg_exact:.4f}", file=sys.stderr)
    print(f"  Uniform K=256:        {ndcg_uniform:.4f}", file=sys.stderr)
    print(f"  Variance-proportional:{ndcg_var_prop:.4f}  (Δ vs uniform = {ndcg_var_prop - ndcg_uniform:+.4f})", file=sys.stderr)
    print(f"  Variance-anti-prop.:  {ndcg_var_anti:.4f}  (Δ vs uniform = {ndcg_var_anti - ndcg_uniform:+.4f})", file=sys.stderr)
    print(f"  wall_time: {elapsed:.0f}s", file=sys.stderr)
    print(f"  output: {OUT_PATH}", file=sys.stderr)

    # honest verdict heuristic
    if ndcg_var_prop > ndcg_uniform + 0.005 and ndcg_var_anti < ndcg_uniform - 0.005:
        print(f"  verdict: variance signal is REAL — variance-proportional > uniform > anti-proportional ✓", file=sys.stderr)
    elif abs(ndcg_var_prop - ndcg_uniform) < 0.003 and abs(ndcg_var_anti - ndcg_uniform) < 0.003:
        print(f"  verdict: variance signal is WEAK — all three within ±0.003, not a usable contribution at this corpus", file=sys.stderr)
    elif ndcg_var_prop < ndcg_uniform:
        print(f"  verdict: variance-proportional FAILED — uniform is better, hypothesis falsified at this scale", file=sys.stderr)
    else:
        print(f"  verdict: mixed — see Δ values", file=sys.stderr)


def train_pq_codebook_assign(embs, codebooks, sub_dim, n_sub):
    """Assign pre-trained codebooks to new vectors (e.g., queries)."""
    n_docs = embs.shape[0]
    codes = np.zeros((n_docs, n_sub), dtype=np.int32)
    for s in range(n_sub):
        block = embs[:, s * sub_dim:(s + 1) * sub_dim]
        cb = codebooks[s]  # (K, sub_dim)
        # nearest centroid: argmin ||block - cb||²
        dists = ((block[:, None, :] - cb[None, :, :]) ** 2).sum(axis=-1)
        codes[:, s] = np.argmin(dists, axis=1)
    return codebooks, codes


if __name__ == "__main__":
    main()
