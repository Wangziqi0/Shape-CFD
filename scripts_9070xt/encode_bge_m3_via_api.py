#!/usr/bin/env python3
"""Encode a BEIR dataset queries+corpus via local llama-server BGE-M3 (port 8080).
Output: parquet with (id, emb_fp16) column matching E5-Mistral schema for eval reuse.
"""
import json, sys, time, os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

DATASET = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
DATA_ROOT = Path("/home/amd/Shape-CFD-9070XT/beir_data")
OUT_ROOT = Path("/home/amd/Shape-CFD-9070XT/embeddings/bge_m3")
URL = "http://localhost:8080/embedding"
DIM = 1024  # BGE-M3
BATCH = 4  # parallel HTTP requests (reduced for long-text datasets)
SHARD_SIZE = 2000
MAX_CHARS = 6000  # truncate to fit BGE-M3 ctx 8192 ~= 6000 chars at ~0.75 tok/char

def emb_one(text):
    text = (text or "").strip()
    if not text:
        return np.zeros(DIM, dtype=np.float32)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    data = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
        arr = np.array(result[0]["embedding"][0], dtype=np.float32)
        if arr.shape[0] != DIM:
            return np.zeros(DIM, dtype=np.float32)
        return arr
    except Exception as e:
        print(f"  emb_one error: {e!r} text_len={len(text)}", flush=True)
        return np.zeros(DIM, dtype=np.float32)

def emb_batch(texts):
    """Use ThreadPoolExecutor for parallel HTTP. emb_one swallows errors -> zero vec."""
    with ThreadPoolExecutor(max_workers=BATCH) as ex:
        results = list(ex.map(emb_one, texts))
    return results

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

ds_dir = DATA_ROOT / DATASET
out_dir = OUT_ROOT / DATASET
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "queries").mkdir(exist_ok=True)
(out_dir / "corpus").mkdir(exist_ok=True)

# Queries
print(f"[bge-m3] {DATASET} encoding queries...")
qs = read_jsonl(ds_dir / "queries.jsonl")
print(f"  {len(qs)} queries")
t0 = time.time()
q_ids, q_embs_bytes = [], []
for i in range(0, len(qs), BATCH):
    chunk = qs[i:i+BATCH]
    texts = [r["text"] for r in chunk]
    embs = emb_batch(texts)
    for rec, e in zip(chunk, embs):
        q_ids.append(rec["_id"])
        q_embs_bytes.append(e.astype(np.float16).tobytes())
    if (i // BATCH) % 20 == 0:
        elapsed = time.time() - t0
        n_done = i + len(chunk)
        eta = (len(qs) - n_done) * elapsed / max(n_done, 1)
        print(f"  [{n_done}/{len(qs)}] {elapsed:.0f}s ETA {eta:.0f}s")
pq.write_table(
    pa.table({"id": q_ids, "emb_fp16": q_embs_bytes}),
    out_dir / "queries/queries.parquet",
    compression="zstd",
)
print(f"[bge-m3] queries done {time.time()-t0:.1f}s")

# Corpus
print(f"[bge-m3] {DATASET} encoding corpus...")
cs = read_jsonl(ds_dir / "corpus.jsonl")
print(f"  {len(cs)} docs")
t0 = time.time()
shard_idx = 0
shard_buf = {"id": [], "emb_fp16": []}

def flush_shard():
    global shard_idx, shard_buf
    if not shard_buf["id"]:
        return
    pq.write_table(
        pa.table(shard_buf),
        out_dir / f"corpus/corpus_{shard_idx:05d}.parquet",
        compression="zstd",
    )
    shard_idx += 1
    shard_buf = {"id": [], "emb_fp16": []}

for i in range(0, len(cs), BATCH):
    chunk = cs[i:i+BATCH]
    texts = [(r.get("title", "") + " " + r.get("text", "")).strip() for r in chunk]
    embs = emb_batch(texts)  # emb_one swallows errors -> zero vec
    for rec, e in zip(chunk, embs):
        shard_buf["id"].append(rec["_id"])
        shard_buf["emb_fp16"].append(e.astype(np.float16).tobytes())
    if len(shard_buf["id"]) >= SHARD_SIZE:
        flush_shard()
    if (i // BATCH) % 50 == 0:
        elapsed = time.time() - t0
        n_done = i + len(chunk)
        eta = (len(cs) - n_done) * elapsed / max(n_done, 1)
        print(f"  [{n_done}/{len(cs)}] {elapsed:.0f}s ETA {eta:.0f}s")

flush_shard()
print(f"[bge-m3] corpus done {time.time()-t0:.1f}s, shards={shard_idx}")

meta = {
    "model": "bge-m3-f16.gguf via llama-server",
    "dataset": DATASET,
    "n_queries": len(qs),
    "n_corpus": len(cs),
    "embedding_dim": DIM,
    "n_shards": shard_idx,
    "via": "http://localhost:8080/embedding",
}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"[bge-m3] DONE: {out_dir}/meta.json")
