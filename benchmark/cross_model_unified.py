#!/usr/bin/env python3
"""
cross_model_unified.py — multi-venue agent W5 fix
====================================================
paper Table 9 reports cross-model gain with each model using its OWN tuned (α, T):
  Qwen3-8B    (α=0.15, T=5)
  BGE-M3      (α=0.10, T=10)
  BGE-large   (α=0.02, T=20)
This is "best-case practitioner deployment" per paper's own admission and a
soft target for the "model-agnostic" claim.

This script computes graph Laplacian smoothing under the UNIFIED default
(α=0.15, T=5, K=3, β=2) on BGE-large and BGE-M3 NFCorpus vectors as a
controlled comparison. Output goes to paper §4.7.

Pipeline:
  1. Load BGE-{large,m3} corpus + query vectors (already encoded on 7b13)
  2. Cosine top-100 per query
  3. KNN graph (K=3) on top-100 with W_ij = exp(-2*d_ij)
  4. Laplacian smoothing: C^{t+1} = (I - α L) C^t for T=5, α=0.15
  5. Re-rank by smoothed scores; NDCG@10

Pure CPU numpy, runs on 7b13.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
import time


def load_vectors_jsonl(path):
    """Load {_id, vector} jsonl -> (ids, vec_array)."""
    ids, vecs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            v = r.get("vector") or r.get("emb")
            if v is None:
                continue
            ids.append(r.get("_id"))
            vecs.append(v)
    return ids, np.asarray(vecs, dtype=np.float32)


def load_qrels(path):
    qrels = {}
    with open(path, "r") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.startswith("query"):
                    continue
            p = line.split("\t")
            if len(p) >= 3:
                qrels.setdefault(p[0], {})[p[1]] = int(p[2])
    return qrels


def cosine_topk(q_vec, doc_vecs, doc_ids, k):
    qn = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    dn = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    sims = dn @ qn
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(doc_ids[i], float(sims[i]), int(i)) for i in idx]


def laplacian_smooth(scores, doc_vecs_top, K=3, beta=2.0, alpha=0.15, T=5):
    """C^{t+1} = (I - α L) C^t with KNN graph weights."""
    N = len(scores)
    if N <= 1:
        return scores
    # cosine sim matrix
    dn = doc_vecs_top / (np.linalg.norm(doc_vecs_top, axis=1, keepdims=True) + 1e-8)
    sim_mat = dn @ dn.T
    # KNN graph (bidirectional)
    W = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nbrs = np.argsort(-sim_mat[i])
        cnt = 0
        for j in nbrs:
            if j == i:
                continue
            d = 1.0 - float(sim_mat[i, j])
            W[i, j] = np.exp(-beta * d)
            cnt += 1
            if cnt >= K:
                break
    W = np.maximum(W, W.T)
    D = W.sum(axis=1)
    L = np.diag(D) - W
    C = scores.astype(np.float32).copy()
    for _ in range(T):
        C = C - alpha * (L @ C)
    return C


def ndcg_at_k(ranked_doc_ids, qrel, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        rel = qrel.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / np.log2(i + 2)
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal):
        if rel > 0:
            idcg += (2 ** rel - 1) / np.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate(model_name, corpus_path, query_path, qrels_path, K_pool=100, alpha=0.15, T=5):
    print(f"[{model_name}] loading...")
    d_ids, d_arr = load_vectors_jsonl(corpus_path)
    q_ids, q_arr = load_vectors_jsonl(query_path)
    qrels = load_qrels(qrels_path)
    print(f"  {len(d_ids)} docs, {len(q_ids)} queries, {len(qrels)} test qrels")

    q_id_to_idx = {q: i for i, q in enumerate(q_ids)}
    cos_ndcgs, lap_ndcgs = [], []
    t0 = time.time()
    for query_id, rels in qrels.items():
        if query_id not in q_id_to_idx:
            continue
        qv = q_arr[q_id_to_idx[query_id]]
        topn = cosine_topk(qv, d_arr, d_ids, K_pool)
        # Cosine baseline (top-10 from top-100)
        cos_top10 = [t[0] for t in topn[:10]]
        cos_ndcgs.append(ndcg_at_k(cos_top10, rels, k=10))
        # Laplacian smooth on top-K_pool
        scores = np.array([t[1] for t in topn], dtype=np.float32)
        d_top = d_arr[[t[2] for t in topn]]
        smoothed = laplacian_smooth(scores, d_top, K=3, beta=2.0, alpha=alpha, T=T)
        # Re-rank
        order = np.argsort(-smoothed)
        lap_top10 = [topn[i][0] for i in order[:10]]
        lap_ndcgs.append(ndcg_at_k(lap_top10, rels, k=10))
    elapsed = time.time() - t0
    cos_mean = float(np.mean(cos_ndcgs)) if cos_ndcgs else 0
    lap_mean = float(np.mean(lap_ndcgs)) if lap_ndcgs else 0
    gain = (lap_mean - cos_mean) / cos_mean * 100 if cos_mean > 0 else 0
    print(f"  {model_name} cos={cos_mean:.4f}  lap_unified={lap_mean:.4f}  gain={gain:+.2f}%  ({elapsed:.0f}s)")
    return {
        "model": model_name,
        "alpha": alpha, "T": T, "K_pool": K_pool,
        "n_test_queries": len(cos_ndcgs),
        "cosine_ndcg10": cos_mean,
        "laplacian_unified_ndcg10": lap_mean,
        "relative_gain_pct": gain,
        "elapsed_seconds": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data/nfcorpus")
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--K_pool", type=int, default=100)
    p.add_argument("--out", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/cross_model_unified_results.json")
    args = p.parse_args()

    qrels_path = Path(args.data) / "qrels.tsv"
    if not qrels_path.exists():
        qrels_path = Path(args.data) / "qrels" / "test.tsv"

    results = {"alpha": args.alpha, "T": args.T, "K_pool": args.K_pool, "models": {}}

    print(f"=== Cross-model unified config (α={args.alpha}, T={args.T}) on NFCorpus ===\n")
    for model_name, c, q in [
        ("BGE-M3", "bge_m3_corpus_vectors.jsonl", "bge_m3_query_vectors.jsonl"),
        ("BGE-large-en-v1.5", "bge_large_corpus_vectors.jsonl", "bge_large_query_vectors.jsonl"),
    ]:
        c_path = Path(args.data) / c
        q_path = Path(args.data) / q
        if not c_path.exists() or not q_path.exists():
            print(f"[{model_name}] SKIP missing {c_path} or {q_path}")
            continue
        results["models"][model_name] = evaluate(model_name, c_path, q_path, qrels_path,
                                                  K_pool=args.K_pool, alpha=args.alpha, T=args.T)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
