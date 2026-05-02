#!/usr/bin/env python3
"""BGE-M3 cosine NDCG@10 on a BEIR dataset. Adapter from E5-Mistral eval (1024-dim)."""
import json, sys
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import time

DATASET = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/bge_m3" / DATASET
DATA = ROOT / "beir_data" / DATASET
DIM = 1024

t0 = time.time()

def load_parquet_dir(d):
    out_ids, out_embs = [], []
    files = sorted(d.glob("*.parquet"))
    for f in files:
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()
        emb_blobs = t.column("emb_fp16").to_pylist()
        for did, blob in zip(ids, emb_blobs):
            arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            if arr.shape[0] != DIM:
                continue
            out_ids.append(did)
            out_embs.append(arr)
    return out_ids, np.array(out_embs, dtype=np.float32)

q_ids, q_arr = load_parquet_dir(EMB / "queries")
d_ids, d_arr = load_parquet_dir(EMB / "corpus")
print(f"[bge-m3-eval] {DATASET}: {len(q_ids)} queries, {len(d_ids)} docs, dim={DIM}")

q_arr = q_arr / np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-12)
d_arr = d_arr / np.maximum(np.linalg.norm(d_arr, axis=1, keepdims=True), 1e-12)

qrels = {}
with open(DATA / "qrels/test.tsv") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        qid, did, score = parts[0], parts[1], int(parts[2])
        qrels.setdefault(qid, {})[did] = score

q_id_to_idx = {qid: i for i, qid in enumerate(q_ids)}
test_qids = [qid for qid in qrels if qid in q_id_to_idx]

def ndcg_at_k(ranked_dids, qrel_scores, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked_dids[:k]):
        rel = qrel_scores.get(did, 0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / np.log2(i + 2)
    ideal = sorted(qrel_scores.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        if rel > 0:
            idcg += (2.0 ** rel - 1.0) / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

test_q_idx = np.array([q_id_to_idx[qid] for qid in test_qids])
q_subset = q_arr[test_q_idx]
sims = q_subset @ d_arr.T

ndcgs = []
ndcg_per_q = {}
for i, qid in enumerate(test_qids):
    top_idx = np.argsort(-sims[i])[:100]
    ranked = [d_ids[di] for di in top_idx]
    ndcg = ndcg_at_k(ranked, qrels[qid], k=10)
    ndcgs.append(ndcg)
    ndcg_per_q[qid] = ndcg

mean_ndcg = float(np.mean(ndcgs))
total = time.time() - t0
print(f"\n=== {DATASET} BGE-M3 cosine NDCG@10 = {mean_ndcg:.4f} ({len(test_qids)} queries, {total:.0f}s) ===")

out = {
    "model": "bge-m3-f16.gguf",
    "dataset": DATASET,
    "method": "cosine",
    "n_test_queries": len(test_qids),
    "ndcg_at_10": mean_ndcg,
    "ndcg_per_query": ndcg_per_q,
    "elapsed_seconds": total,
}
out_path = ROOT / f"outputs/bge_m3_{DATASET}_cosine_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"saved: {out_path}")
