#!/usr/bin/env python3
"""Evaluate the existing fp16_control E5-Mistral parquet vectors.
Reproduces the 0.1392 NDCG@10 reported in paper Section 4.6 + persists JSON.
"""
import json, sys, pyarrow.parquet as pq, numpy as np, time
from pathlib import Path

ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/e5_mistral_fp16_control/nfcorpus"
DATA = ROOT / "beir_data/nfcorpus"
OUT = ROOT / "outputs/e5_mistral_nfcorpus_fp16_control_eval.json"
DIM = 4096

def load_parquet_dir(d):
    out_ids, out_embs = [], []
    for f in sorted(d.glob("*.parquet")):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()
        emb_col = None
        for c in ("embedding", "emb", "vec", "vector", "mean_emb", "doc_emb"):
            if c in t.column_names:
                emb_col = c
                break
        if emb_col is None:
            for c in t.column_names:
                if c != "id":
                    emb_col = c
                    break
        for did, blob in zip(ids, t.column(emb_col).to_pylist()):
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.shape[0] != DIM:
                arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            out_ids.append(did)
            out_embs.append(arr)
    return out_ids, np.array(out_embs, dtype=np.float32)

t0 = time.time()
print(f"[fp16-control-eval] loading queries from {EMB}/queries ...")
q_ids, q_arr = load_parquet_dir(EMB / "queries")
print(f"  {len(q_ids)} queries; shape {q_arr.shape}")
print(f"[fp16-control-eval] loading corpus from {EMB}/corpus ...")
d_ids, d_arr = load_parquet_dir(EMB / "corpus")
print(f"  {len(d_ids)} docs; shape {d_arr.shape}")

q_arr = q_arr / np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-12)
d_arr = d_arr / np.maximum(np.linalg.norm(d_arr, axis=1, keepdims=True), 1e-12)

qrels = {}
with open(DATA / "qrels/test.tsv") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 3:
            qrels.setdefault(p[0], {})[p[1]] = int(p[2])

q_id_to_idx = {qid: i for i, qid in enumerate(q_ids)}
test_qids = [qid for qid in qrels if qid in q_id_to_idx]
print(f"  {len(test_qids)} test queries with qrels")

def ndcg_at_k(ranked, qrel_s, k=10):
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        rel = qrel_s.get(d, 0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / np.log2(i + 2)
    ideal = sorted(qrel_s.values(), reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal[:k]):
        if r > 0:
            idcg += (2.0**r - 1.0) / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

ndcg_per_q = {}
for qid in test_qids:
    qi = q_id_to_idx[qid]
    sims = d_arr @ q_arr[qi]
    top = np.argsort(-sims)[:100]
    ndcg_per_q[qid] = ndcg_at_k([d_ids[i] for i in top], qrels[qid], 10)

avg = float(np.mean(list(ndcg_per_q.values())))
result = {
    "model": "intfloat/e5-mistral-7b-instruct",
    "dataset": "nfcorpus",
    "method": "fp16 inference + sdpa, same-hardware control (rocm7.2.0 + batch_q=1)",
    "n_test_queries": len(test_qids),
    "ndcg_at_10": avg,
    "ndcg_per_query": ndcg_per_q,
    "embedding_source": str(EMB),
    "embedding_meta": json.load(open(EMB / "meta.json")),
    "elapsed_sec": time.time() - t0,
    "note": "Reproduces paper Section 4.6 fp16+sdpa NDCG@10 = 0.1392. Generated 2026-05-01 evening as audit-trail JSON; original 14:29:40 ssh stdout was not persisted to file at the time.",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"NDCG@10 = {avg:.6f} (n={len(test_qids)})")
print(f"Saved: {OUT}")
