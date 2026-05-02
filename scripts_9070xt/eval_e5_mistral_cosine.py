#!/usr/bin/env python3
"""E5-Mistral cosine NDCG@10 evaluation on a BEIR dataset.
E5-Mistral parquet schema (per row): id (str), embedding (bytes float32 of shape (4096,)).
Single-vector cosine retrieval. Pure CPU numpy.
Usage: python eval_e5_mistral_cosine.py <dataset>
"""
import json, sys
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import time

DATASET = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/e5_mistral" / DATASET
DATA = ROOT / "beir_data" / DATASET
DIM = 4096

t0 = time.time()

def load_parquet_dir(d):
    out_ids, out_embs = [], []
    files = sorted(d.glob("*.parquet"))
    for f in files:
        t = pq.read_table(f)
        cols = t.column_names
        ids = t.column("id").to_pylist()
        # Find embedding column (could be 'embedding' / 'emb' / etc.)
        emb_col = None
        for c in ("embedding", "emb", "vec", "vector", "mean_emb", "doc_emb"):
            if c in cols:
                emb_col = c
                break
        if emb_col is None:
            # take first non-id column with binary type
            for c in cols:
                if c != "id":
                    emb_col = c
                    break
        emb_blobs = t.column(emb_col).to_pylist()
        for did, blob in zip(ids, emb_blobs):
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.shape[0] != DIM:
                # try float16 or other dtype
                arr16 = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
                if arr16.shape[0] == DIM:
                    arr = arr16
            out_ids.append(did)
            out_embs.append(arr)
    return out_ids, np.array(out_embs, dtype=np.float32)

# Look for queries / corpus subdir
q_dir = EMB / "queries"
c_dir = EMB / "corpus"
print(f"[e5-cosine-eval] {DATASET} schema check")
for sub_q in (q_dir / "queries.parquet", q_dir):
    if sub_q.exists():
        if sub_q.is_file():
            t = pq.read_table(sub_q)
            print(f"  queries.parquet schema: {t.schema}")
            break
        else:
            files = list(sub_q.glob("*.parquet"))
            if files:
                t = pq.read_table(files[0])
                print(f"  queries dir first parquet: {t.schema}")
                break
print(f"[e5-cosine-eval] loading queries from {q_dir}...")
q_ids, q_arr = load_parquet_dir(q_dir)
print(f"  {len(q_ids)} queries; emb shape: {q_arr.shape}")

print(f"[e5-cosine-eval] loading corpus from {c_dir}...")
d_ids, d_arr = load_parquet_dir(c_dir)
print(f"  {len(d_ids)} docs; emb shape: {d_arr.shape}")

# L2 normalize (cosine)
q_arr = q_arr / np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-12)
d_arr = d_arr / np.maximum(np.linalg.norm(d_arr, axis=1, keepdims=True), 1e-12)

# Load qrels test
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
print(f"  {len(test_qids)} test queries with qrels")

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

# Cosine = q_arr @ d_arr.T (since both normalized)
print(f"[e5-cosine-eval] computing cosine NDCG@10...")
ndcgs = []
ndcg_per_q = {}
test_q_idx = np.array([q_id_to_idx[qid] for qid in test_qids])
q_subset = q_arr[test_q_idx]  # (n_test, DIM)

# Batch matmul: (n_test, n_docs)
sims = q_subset @ d_arr.T  # (n_test, n_docs)
print(f"  sim matrix: {sims.shape}, mem: {sims.nbytes / 1e9:.2f} GB")

# Per query top-100 + NDCG@10
for i, qid in enumerate(test_qids):
    top_idx = np.argsort(-sims[i])[:100]
    ranked = [d_ids[di] for di in top_idx]
    ndcg = ndcg_at_k(ranked, qrels[qid], k=10)
    ndcgs.append(ndcg)
    ndcg_per_q[qid] = ndcg

mean_ndcg = float(np.mean(ndcgs))
total = time.time() - t0
print(f"\n=== RESULTS ===")
print(f"  E5-Mistral {DATASET} cosine NDCG@10 = {mean_ndcg:.4f}")
print(f"  n_test_queries = {len(test_qids)}")
print(f"  total time: {total:.0f}s")

out = {
    "model": "intfloat/e5-mistral-7b-instruct",
    "dataset": DATASET,
    "method": "cosine",
    "n_test_queries": len(test_qids),
    "ndcg_at_10": mean_ndcg,
    "ndcg_per_query": ndcg_per_q,
    "elapsed_seconds": total,
}
out_path = ROOT / f"outputs/e5_mistral_{DATASET}_cosine_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"  saved: {out_path}")
