#!/usr/bin/env python3
"""BGE-reranker-v2-m3 NDCG@10 evaluation on a BEIR dataset.
Pipeline: BGE-M3 cosine top-100 candidates -> BGE-reranker rerank -> NDCG@10.
Usage: python eval_bge_reranker.py <dataset>
"""
import json, sys, time
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

DATASET = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/bge_m3" / DATASET
DATA = ROOT / "beir_data" / DATASET
RERANK_URL = "http://localhost:8081/rerank"
DIM = 1024
TOP_K_CANDIDATES = 100
MAX_DOC_CHARS = 1500  # truncate doc text to avoid 8192 ctx overflow when 100 docs concat

def load_emb_dir(d):
    out_ids, out_embs = [], []
    for f in sorted(d.glob("*.parquet")):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()
        blobs = t.column("emb_fp16").to_pylist()
        for did, blob in zip(ids, blobs):
            arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            if arr.shape[0] != DIM:
                continue
            out_ids.append(did)
            out_embs.append(arr)
    return out_ids, np.array(out_embs, dtype=np.float32)

def read_jsonl_dict(path, key="_id"):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec[key]] = rec
    return out

print(f"[bge-rerank] {DATASET} loading embeddings + texts...")
q_ids, q_arr = load_emb_dir(EMB / "queries")
d_ids, d_arr = load_emb_dir(EMB / "corpus")
q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
d_norms = np.linalg.norm(d_arr, axis=1, keepdims=True)
q_arr = q_arr / np.maximum(q_norms, 1e-12)
d_arr = d_arr / np.maximum(d_norms, 1e-12)

q_texts = read_jsonl_dict(DATA / "queries.jsonl")
d_texts = read_jsonl_dict(DATA / "corpus.jsonl")
print(f"  {len(q_ids)} queries, {len(d_ids)} docs")

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
print(f"  {len(test_qids)} test queries")

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

def truncate(text, n=MAX_DOC_CHARS):
    if not text:
        return ""
    return text[:n]

def rerank_call(query, docs):
    payload = {"query": query, "documents": docs}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(RERANK_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
        # results is list of {index, relevance_score} sorted desc by score (or unsorted)
        return [(r["index"], r["relevance_score"]) for r in result["results"]]
    except Exception as e:
        print(f"  rerank error: {e!r}", flush=True)
        return None

# Cosine top-K candidate pool per query
print(f"[bge-rerank] computing cosine top-{TOP_K_CANDIDATES}...")
test_q_idx = np.array([q_id_to_idx[qid] for qid in test_qids])
q_subset = q_arr[test_q_idx]
sims = q_subset @ d_arr.T

# Now rerank each query
print(f"[bge-rerank] reranking {len(test_qids)} queries via {RERANK_URL}...")
ndcgs = []
ndcg_per_q = {}
t0 = time.time()
errors = 0
for cnt, qid in enumerate(test_qids):
    top_idx = np.argsort(-sims[cnt])[:TOP_K_CANDIDATES]
    cand_dids = [d_ids[i] for i in top_idx]
    cand_texts = []
    for did in cand_dids:
        rec = d_texts.get(did, {})
        text = (rec.get("title", "") + " " + rec.get("text", "")).strip()
        cand_texts.append(truncate(text))
    qtext = q_texts.get(qid, {}).get("text", "")
    if not qtext:
        continue
    res = rerank_call(qtext, cand_texts)
    if res is None:
        errors += 1
        # fallback to cosine ranking
        ranked = cand_dids[:10]
    else:
        # rerank by score
        res_sorted = sorted(res, key=lambda x: -x[1])
        ranked = [cand_dids[idx] for idx, _ in res_sorted[:10]]
    ndcg = ndcg_at_k(ranked, qrels[qid], k=10)
    ndcgs.append(ndcg)
    ndcg_per_q[qid] = ndcg
    if (cnt + 1) % 25 == 0:
        elapsed = time.time() - t0
        avg = elapsed / (cnt + 1)
        eta = (len(test_qids) - cnt - 1) * avg
        print(f"  [{cnt+1}/{len(test_qids)}] mean={np.mean(ndcgs):.4f} ETA {eta:.0f}s")

mean_ndcg = float(np.mean(ndcgs))
total = time.time() - t0
print(f"\n=== {DATASET} BGE-reranker NDCG@10 = {mean_ndcg:.4f} ({len(test_qids)} queries, {total:.0f}s, {errors} errors) ===")

out = {
    "model": "BAAI/bge-reranker-v2-m3 (Q8_0 gguf via llama-server)",
    "first_stage": "bge-m3 cosine top-100",
    "dataset": DATASET,
    "n_test_queries": len(test_qids),
    "ndcg_at_10": mean_ndcg,
    "ndcg_per_query": ndcg_per_q,
    "errors": errors,
    "elapsed_seconds": total,
}
out_path = ROOT / f"outputs/bge_reranker_{DATASET}_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"saved: {out_path}")
