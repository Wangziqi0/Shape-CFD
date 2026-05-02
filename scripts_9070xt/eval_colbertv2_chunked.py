#!/usr/bin/env python3
"""ColBERTv2 NDCG@10 with chunked corpus to avoid OOM on FiQA / large datasets.
Processes corpus in chunks; per-query maxsim aggregated across chunks.
Usage: python eval_colbertv2_chunked.py <dataset> [chunk_size]
"""
import json, sys
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import time

DATASET = sys.argv[1] if len(sys.argv) > 1 else "fiqa"
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/colbertv2" / DATASET
DATA = ROOT / "beir_data" / DATASET
DIM = 768

t0 = time.time()
def load_parquet(path):
    t = pq.read_table(path)
    d = t.to_pydict()
    out = []
    for i in range(t.num_rows):
        n = d["n_tokens"][i]
        arr = np.frombuffer(d["token_emb"][i], dtype=np.float32).reshape(n, DIM)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
        out.append((d["id"][i], arr))
    return out

print(f"[chunked-eval] {DATASET} loading queries...")
qs = load_parquet(EMB / "queries/queries.parquet")
print(f"  {len(qs)} queries")

q_id_to_idx = {q[0]: i for i, q in enumerate(qs)}
qrels = {}
with open(DATA / "qrels/test.tsv") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        qid, did, score = parts[0], parts[1], int(parts[2])
        qrels.setdefault(qid, {})[did] = score
test_qids = [qid for qid in qrels if qid in q_id_to_idx]
test_q_idx = [q_id_to_idx[qid] for qid in test_qids]
print(f"  {len(test_qids)} test queries")

# Per-query running top-100: list of (doc_id, score) heap-style sorted desc
per_query_topk = {qid: [] for qid in test_qids}
TOPK = 100

def merge_topk(existing, new_pairs):
    merged = existing + new_pairs
    merged.sort(key=lambda x: -x[1])
    return merged[:TOPK]

corpus_files = sorted((EMB / "corpus").glob("corpus_*.parquet"))
print(f"  {len(corpus_files)} corpus shards")

total_docs = 0
for fi, f in enumerate(corpus_files):
    ds_chunk = load_parquet(f)
    n = len(ds_chunk)
    total_docs += n
    # Process in inner chunks of CHUNK to bound memory
    for start in range(0, n, CHUNK):
        sub = ds_chunk[start:start+CHUNK]
        max_d_len = max(d[1].shape[0] for d in sub)
        N_d = len(sub)
        d_block = np.zeros((N_d, max_d_len, DIM), dtype=np.float32)
        d_lens = np.zeros(N_d, dtype=np.int32)
        d_id_list = []
        for i, (did, arr) in enumerate(sub):
            d_block[i, :arr.shape[0]] = arr
            d_lens[i] = arr.shape[0]
            d_id_list.append(did)
        d_mask_2d = (np.arange(max_d_len)[None, :] < d_lens[:, None])

        # For each test query compute scores against this chunk
        for cnt, qid in enumerate(test_qids):
            q_arr = qs[q_id_to_idx[qid]][1]
            sims = np.einsum('id,nkd->ink', q_arr, d_block)
            sims = np.where(d_mask_2d[None, :, :], sims, -np.inf)
            maxsim = sims.max(axis=2)
            score_per_doc = maxsim.sum(axis=0)
            top_idx = np.argsort(-score_per_doc)[:TOPK]
            new_pairs = [(d_id_list[i], float(score_per_doc[i])) for i in top_idx]
            per_query_topk[qid] = merge_topk(per_query_topk[qid], new_pairs)
        elapsed = time.time() - t0
        print(f"  shard {fi+1}/{len(corpus_files)} chunk {start}/{n} time={elapsed:.0f}s", flush=True)
    # free chunk
    ds_chunk = None
    d_block = None

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

ndcgs = []
ndcg_per_q = {}
for qid in test_qids:
    ranked = [d for d, s in per_query_topk[qid][:10]]
    ndcg = ndcg_at_k(ranked, qrels[qid], k=10)
    ndcgs.append(ndcg)
    ndcg_per_q[qid] = ndcg

mean_ndcg = float(np.mean(ndcgs))
total = time.time() - t0
print(f"\n=== {DATASET} ColBERTv2 (chunked) NDCG@10 = {mean_ndcg:.4f} ({len(test_qids)} queries, {total:.0f}s) ===")

out = {
    "model": "colbert-ir/colbertv2.0",
    "dataset": DATASET,
    "chunked": True,
    "chunk_size": CHUNK,
    "n_test_queries": len(test_qids),
    "ndcg_at_10": mean_ndcg,
    "ndcg_per_query": ndcg_per_q,
    "elapsed_seconds": total,
}
out_path = ROOT / f"outputs/colbertv2_{DATASET}_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"saved: {out_path}")
