#!/usr/bin/env python3
"""ColBERTv2 NDCG@10 on NFCorpus.
Schema (per row): id (str), token_emb (bytes float32), mean_emb (bytes float32, dim=768), n_tokens (int).
token_emb has length = n_tokens * 768 * 4 bytes (no padding).
"""
import json
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import time

ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/colbertv2/nfcorpus"
DATA = ROOT / "beir_data/nfcorpus"
DIM = 768

t0 = time.time()
def load_parquet(path):
    t = pq.read_table(path)
    d = t.to_pydict()
    out = []
    for i in range(t.num_rows):
        n = d["n_tokens"][i]
        arr = np.frombuffer(d["token_emb"][i], dtype=np.float32).reshape(n, DIM)
        # L2 normalize per token (ColBERT convention)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
        out.append((d["id"][i], arr))
    return out

print(f"[eval] loading queries...")
qs = load_parquet(EMB / "queries/queries.parquet")
print(f"  {len(qs)} queries; sample tokens: {qs[0][1].shape}")

print(f"[eval] loading corpus...")
ds = []
for f in sorted((EMB / "corpus").glob("corpus_*.parquet")):
    ds.extend(load_parquet(f))
print(f"  {len(ds)} docs; sample tokens: {ds[0][1].shape}")

q_id_to_idx = {q[0]: i for i, q in enumerate(qs)}
d_id_to_idx = {d[0]: i for i, d in enumerate(ds)}

# Load qrels test
print(f"[eval] loading qrels...")
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
print(f"  {len(test_qids)} test queries")

# Pre-stack corpus tensors with per-doc tokens (varying len). Use a single big block + offsets.
# Actually for efficient computation, batch: pad all docs to max_len.
max_d_len = max(d[1].shape[0] for d in ds)
print(f"  max doc len: {max_d_len}")
N_d = len(ds)
d_block = np.zeros((N_d, max_d_len, DIM), dtype=np.float32)
d_lens = np.zeros(N_d, dtype=np.int32)
d_id_list = []
for i, (did, arr) in enumerate(ds):
    d_block[i, :arr.shape[0]] = arr
    d_lens[i] = arr.shape[0]
    d_id_list.append(did)
d_mask_2d = (np.arange(max_d_len)[None, :] < d_lens[:, None])  # (N_d, max_d_len)
print(f"  d_block: {d_block.shape}, mem: {d_block.nbytes / 1e9:.2f} GB")

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

print(f"[eval] computing MaxSim NDCG@10...")
ndcgs = []
ndcg_per_q = {}
t1 = time.time()
for cnt, qid in enumerate(test_qids):
    q_arr = qs[q_id_to_idx[qid]][1]  # (q_n, 768)
    # MaxSim: for each q_token find max sim across all d_tokens of all docs, then sum over q_tokens.
    # sims shape: (q_n, N_d, max_d_len)
    sims = np.einsum('id,nkd->ink', q_arr, d_block)
    sims = np.where(d_mask_2d[None, :, :], sims, -np.inf)  # mask invalid d_tokens
    maxsim = sims.max(axis=2)  # (q_n, N_d)
    score_per_doc = maxsim.sum(axis=0)  # (N_d,)
    top_idx = np.argsort(-score_per_doc)[:100]
    ranked = [d_id_list[di] for di in top_idx]
    ndcg = ndcg_at_k(ranked, qrels[qid], k=10)
    ndcgs.append(ndcg)
    ndcg_per_q[qid] = ndcg
    if (cnt + 1) % 50 == 0:
        elapsed = time.time() - t1
        avg = elapsed / (cnt + 1)
        eta = (len(test_qids) - cnt - 1) * avg
        print(f"  [{cnt+1}/{len(test_qids)}] mean NDCG@10={np.mean(ndcgs):.4f} ({avg*1000:.0f} ms/q ETA {eta:.0f}s)")

mean_ndcg = float(np.mean(ndcgs))
total = time.time() - t0
print(f"\n=== RESULTS ===\n  ColBERTv2 NFCorpus NDCG@10 = {mean_ndcg:.4f}")
print(f"  n_test_queries = {len(test_qids)}\n  total time: {total:.0f}s")

out = {
    "model": "colbert-ir/colbertv2.0",
    "dataset": "nfcorpus",
    "n_test_queries": len(test_qids),
    "ndcg_at_10": mean_ndcg,
    "ndcg_per_query": ndcg_per_q,
    "elapsed_seconds": total,
}
out_path = ROOT / "outputs/colbertv2_nfcorpus_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"  saved: {out_path}")
