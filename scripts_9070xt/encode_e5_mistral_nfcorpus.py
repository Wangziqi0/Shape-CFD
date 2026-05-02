"""
E5-Mistral-7B-Instruct dense encoding on BEIR/NFCorpus, ROCm/9070XT (bf16).
- intfloat/e5-mistral-7b-instruct
- For E5-Mistral baseline: produces (4096-d) sentence embedding per doc/query
  using last-token pooling + Instruct prompt for queries.
- Output: parquet shards with float16 storage (halve disk).
- Tracks latency + VRAM peak.

Note: E5-Mistral uses last-token pooling (decoder LLM), not mean-pool.
"""

import os, sys, time, json, argparse
from pathlib import Path

import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


QUERY_INSTRUCT = (
    "Instruct: Given a search query, retrieve relevant passages that answer the query.\n"
    "Query: "
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def last_token_pool(last_hidden_states, attention_mask):
    # E5-Mistral pooling per official model card
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def encode_batch(model, tok, texts, max_len, device):
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (B, T, 4096)
    pooled = last_token_pool(out, enc["attention_mask"])  # (B, 4096)
    # CPU normalize to bypass HIP vector_norm kernel missing on gfx1201 (ROCm 7.0)
    pooled = pooled.cpu().to(torch.float32)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--data-root", default="/home/amd/Shape-CFD-9070XT/beir_data")
    ap.add_argument("--out-root", default="/home/amd/Shape-CFD-9070XT/embeddings/e5_mistral")
    ap.add_argument("--model", default="intfloat/e5-mistral-7b-instruct")
    ap.add_argument("--q-max-len", type=int, default=128)
    ap.add_argument("--d-max-len", type=int, default=512)
    ap.add_argument("--batch-q", type=int, default=4)
    ap.add_argument("--batch-d", type=int, default=2)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--limit-corpus", type=int, default=0)
    ap.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    args = ap.parse_args()

    ds_dir = Path(args.data_root) / args.dataset
    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queries").mkdir(exist_ok=True)
    (out_dir / "corpus").mkdir(exist_ok=True)

    device = "cuda"
    print(f"[e5-mistral] device={device} dev={torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[e5-mistral] loading model {args.dtype} attn={args.attn_impl} ...")
    t0 = time.time()
    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModel.from_pretrained(
        args.model,
        dtype=target_dtype,
        attn_implementation=args.attn_impl,
    ).to(device).eval()
    print(f"[e5-mistral] model loaded in {time.time()-t0:.1f}s")
    torch.cuda.synchronize()
    print(f"[e5-mistral] VRAM after load: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    # ---------- queries ----------
    q_recs = list(read_jsonl(ds_dir / "queries.jsonl"))
    print(f"[e5-mistral] N_queries = {len(q_recs)}")
    q_t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    q_ids, q_embs = [], []
    for i in tqdm(range(0, len(q_recs), args.batch_q), desc="q-batch"):
        chunk = q_recs[i:i+args.batch_q]
        texts = [QUERY_INSTRUCT + r["text"] for r in chunk]
        embs = encode_batch(model, tok, texts, args.q_max_len, device)
        for j, rec in enumerate(chunk):
            q_ids.append(rec["_id"])
            q_embs.append(embs[j].astype(np.float16).tobytes())

    q_table = pa.table({"id": q_ids, "emb_fp16": q_embs})
    pq.write_table(q_table, out_dir / "queries" / "queries.parquet", compression="zstd")
    print(f"[e5-mistral] queries done in {time.time()-q_t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

    # ---------- corpus ----------
    c_recs = list(read_jsonl(ds_dir / "corpus.jsonl"))
    if args.limit_corpus > 0:
        c_recs = c_recs[:args.limit_corpus]
    print(f"[e5-mistral] N_corpus = {len(c_recs)}")
    c_t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    shard_idx = 0
    shard_buf = {"id": [], "emb_fp16": []}

    def flush_shard():
        nonlocal shard_idx, shard_buf
        if not shard_buf["id"]:
            return
        t = pa.table(shard_buf)
        pq.write_table(t, out_dir / "corpus" / f"corpus_{shard_idx:05d}.parquet", compression="zstd")
        shard_idx += 1
        shard_buf = {"id": [], "emb_fp16": []}

    pbar = tqdm(range(0, len(c_recs), args.batch_d), desc="c-batch")
    for i in pbar:
        chunk = c_recs[i:i+args.batch_d]
        texts = [(r.get("title", "") + " " + r.get("text", "")).strip() for r in chunk]
        embs = encode_batch(model, tok, texts, args.d_max_len, device)
        for j, rec in enumerate(chunk):
            shard_buf["id"].append(rec["_id"])
            shard_buf["emb_fp16"].append(embs[j].astype(np.float16).tobytes())
        if len(shard_buf["id"]) >= args.shard_size:
            flush_shard()
            pbar.set_postfix(shards=shard_idx, vram=f"{torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

    flush_shard()
    print(f"[e5-mistral] corpus done in {time.time()-c_t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1024**2:.1f} MB, shards={shard_idx}")

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "pooling": "last-token",
        "normalized": True,
        "q_max_len": args.q_max_len,
        "d_max_len": args.d_max_len,
        "batch_q": args.batch_q,
        "batch_d": args.batch_d,
        "n_queries": len(q_recs),
        "n_corpus": len(c_recs),
        "n_shards": shard_idx,
        "dtype": f"{args.dtype} (native inference, vectors stored as fp16)",
        "embedding_dim": 4096,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "query_instruct": QUERY_INSTRUCT,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[e5-mistral] DONE: {out_dir}/meta.json")


if __name__ == "__main__":
    main()
