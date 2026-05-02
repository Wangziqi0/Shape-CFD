"""
ColBERTv2 dense encoding on BEIR/NFCorpus, ROCm/9070XT.
- Encodes queries + corpus with colbert-ir/colbertv2.0
- For ColBERTv2 baseline: produces token-level embeddings (T x 768)
  AND mean-pooled doc/query embeddings for downstream PQ-Chamfer.
- Output: parquet shards (one row per doc/query) into outputs/colbertv2/<dataset>/
- Logs latency, VRAM peak per shard.
"""

import os, sys, time, json, argparse
from pathlib import Path

import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def encode_batch(model, tok, texts, max_len, device, dtype=torch.float32):
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (B, T, D)
    # mask out pad tokens
    mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
    out = out * mask
    # mean pooled (over non-pad tokens)
    lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
    mean_pooled = out.sum(dim=1) / lengths  # (B, D)
    return out.cpu().to(torch.float32).numpy(), mean_pooled.cpu().to(torch.float32).numpy(), enc["attention_mask"].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--data-root", default="/home/amd/Shape-CFD-9070XT/beir_data")
    ap.add_argument("--out-root", default="/home/amd/Shape-CFD-9070XT/embeddings/colbertv2")
    ap.add_argument("--model", default="colbert-ir/colbertv2.0")
    ap.add_argument("--q-max-len", type=int, default=32)
    ap.add_argument("--d-max-len", type=int, default=180)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--limit-corpus", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ds_dir = Path(args.data_root) / args.dataset
    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queries").mkdir(exist_ok=True)
    (out_dir / "corpus").mkdir(exist_ok=True)

    device = "cuda"
    print(f"[colbertv2] device={device} ROCm/HIP available={torch.cuda.is_available()}")
    print(f"[colbertv2] device name={torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    print(f"[colbertv2] model loaded: {args.model}")

    # ---------- queries ----------
    print(f"[colbertv2] encoding queries from {ds_dir/'queries.jsonl'}")
    q_recs = list(read_jsonl(ds_dir / "queries.jsonl"))
    print(f"[colbertv2] N_queries = {len(q_recs)}")

    q_t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    q_ids, q_token_embs, q_mean_embs, q_masks = [], [], [], []
    for i in tqdm(range(0, len(q_recs), args.batch), desc="q-batch"):
        chunk = q_recs[i:i+args.batch]
        texts = [r["text"] for r in chunk]
        token_emb, mean_emb, mask = encode_batch(model, tok, texts, args.q_max_len, device)
        for j, rec in enumerate(chunk):
            valid_len = int(mask[j].sum())
            q_ids.append(rec["_id"])
            q_token_embs.append(token_emb[j, :valid_len].astype(np.float32).tobytes())
            q_mean_embs.append(mean_emb[j].astype(np.float32).tobytes())
            q_masks.append(valid_len)

    q_table = pa.table({
        "id": q_ids,
        "token_emb": q_token_embs,
        "mean_emb": q_mean_embs,
        "n_tokens": q_masks,
    })
    pq.write_table(q_table, out_dir / "queries" / "queries.parquet", compression="zstd")
    print(f"[colbertv2] queries done in {time.time()-q_t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

    # ---------- corpus ----------
    print(f"[colbertv2] encoding corpus from {ds_dir/'corpus.jsonl'}")
    c_recs = list(read_jsonl(ds_dir / "corpus.jsonl"))
    if args.limit_corpus > 0:
        c_recs = c_recs[:args.limit_corpus]
    print(f"[colbertv2] N_corpus = {len(c_recs)}")

    c_t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    shard_idx = 0
    shard_buf = {"id": [], "token_emb": [], "mean_emb": [], "n_tokens": []}

    def flush_shard():
        nonlocal shard_idx, shard_buf
        if not shard_buf["id"]:
            return
        t = pa.table(shard_buf)
        out_path = out_dir / "corpus" / f"corpus_{shard_idx:05d}.parquet"
        pq.write_table(t, out_path, compression="zstd")
        shard_idx += 1
        shard_buf = {"id": [], "token_emb": [], "mean_emb": [], "n_tokens": []}

    pbar = tqdm(range(0, len(c_recs), args.batch), desc="c-batch")
    for i in pbar:
        chunk = c_recs[i:i+args.batch]
        texts = [(r.get("title", "") + " " + r.get("text", "")).strip() for r in chunk]
        token_emb, mean_emb, mask = encode_batch(model, tok, texts, args.d_max_len, device)
        for j, rec in enumerate(chunk):
            valid_len = int(mask[j].sum())
            shard_buf["id"].append(rec["_id"])
            shard_buf["token_emb"].append(token_emb[j, :valid_len].astype(np.float32).tobytes())
            shard_buf["mean_emb"].append(mean_emb[j].astype(np.float32).tobytes())
            shard_buf["n_tokens"].append(valid_len)
        if len(shard_buf["id"]) >= args.shard_size:
            flush_shard()
            pbar.set_postfix(shards=shard_idx, vram=f"{torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

    flush_shard()
    print(f"[colbertv2] corpus done in {time.time()-c_t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1024**2:.1f} MB, shards={shard_idx}")

    # write meta
    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "q_max_len": args.q_max_len,
        "d_max_len": args.d_max_len,
        "n_queries": len(q_recs),
        "n_corpus": len(c_recs),
        "n_shards": shard_idx,
        "dtype": "float32",
        "embedding_dim": 768,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[colbertv2] DONE: {out_dir}/meta.json")


if __name__ == "__main__":
    main()
