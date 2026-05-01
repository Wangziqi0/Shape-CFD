"""
e5_cuda_control.py — E5-Mistral CUDA cross-platform control on RTX 5060 (8GB)

Closes Senior Reviewer Issue 4.1 + multi-venue agent W6 part: bf16 vs fp16
control on ROCm shows the issue is NOT specific to bf16 (both anomalous);
the remaining hypothesis is "ROCm-wide pipeline bug vs platform-specific".
This script runs E5-Mistral on NVIDIA CUDA (RTX 5060) using 4-bit
quantization (bitsandbytes nf4) since fp16 7B = 14 GB > 8 GB VRAM.

Caveat: 4-bit quantization vs ROCm bf16 is not a perfectly clean comparison
(quantization can degrade NDCG by 1-3%), but a sanity check: if 4-bit on
CUDA gives NDCG@10 close to published reference (e.g. NFCorpus ~0.30+,
ArguAna ~0.45+), this rules out the model itself; if CUDA 4-bit also
gives ~0.14 / ~0.08 like ROCm, the issue is in our inference pipeline.

Pipeline: load E5-Mistral nf4 → encode queries (with Instruct prompt) and
corpus (with title+text) → cosine top-10 → NDCG@10 vs qrels.

Hardware: NVIDIA RTX 5060 8 GB CUDA 13.1, torch 2.11+cu130, bitsandbytes.
"""
import os, sys, time, json, gc
from pathlib import Path

# Suppress HF warnings + force local-cache mode (Win has no symlink)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from tqdm import tqdm


QUERY_INSTRUCT = (
    "Instruct: Given a search query, retrieve relevant passages that answer the query.\n"
    "Query: "
)


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


def last_token_pool(last_hidden, attn_mask):
    left = (attn_mask[:, -1].sum() == attn_mask.shape[0])
    if left:
        return last_hidden[:, -1]
    seq_len = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_len]


def encode_texts(model, tok, texts, max_len, batch, device):
    embs = []
    for i in tqdm(range(0, len(texts), batch), desc="enc"):
        chunk = texts[i:i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        pooled = last_token_pool(out, enc["attention_mask"])
        # CPU normalize (avoid GPU-side fp issues)
        pooled = pooled.float().cpu()
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embs.append(pooled.numpy())
    return np.concatenate(embs, axis=0)


def ndcg_at_k(ranked, qrel, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked[:k]):
        rel = qrel.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / np.log2(i + 2)
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal):
        if rel > 0:
            idcg += (2 ** rel - 1) / np.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    base = Path(r"C:\Users\amd\Desktop\HEZIMENG\e5_cuda_control") / ds
    qrels = {}
    qpath = base / "qrels.tsv"
    with open(qpath) as f:
        next(f)
        for ln in f:
            p = ln.strip().split("\t")
            if len(p) >= 3:
                qrels.setdefault(p[0], {})[p[1]] = int(p[2])

    print(f"[e5-cuda] dataset={ds}  test qrels={len(qrels)}")
    print(f"[e5-cuda] CUDA={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}")

    print("[e5-cuda] loading E5-Mistral nf4 (bitsandbytes 4-bit)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        "intfloat/e5-mistral-7b-instruct",
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Encode queries
    q_recs = list(read_jsonl(base / "queries.jsonl"))
    q_test_ids = [r["_id"] for r in q_recs if r["_id"] in qrels]
    q_test_text = [QUERY_INSTRUCT + next(r["text"] for r in q_recs if r["_id"] == qid) for qid in q_test_ids]
    print(f"[e5-cuda] encoding {len(q_test_ids)} test queries (max_len=128, batch=2)...")
    t_q = time.time()
    q_emb = encode_texts(model, tok, q_test_text, max_len=128, batch=2, device="cuda")
    print(f"  queries done {time.time()-t_q:.1f}s, q_emb shape {q_emb.shape}")

    # Encode corpus
    c_recs = list(read_jsonl(base / "corpus.jsonl"))
    c_ids = [r["_id"] for r in c_recs]
    c_text = [(r.get("title", "") + " " + r.get("text", "")).strip() for r in c_recs]
    print(f"[e5-cuda] encoding {len(c_ids)} corpus docs (max_len=512, batch=1)...")
    t_c = time.time()
    c_emb = encode_texts(model, tok, c_text, max_len=512, batch=1, device="cuda")
    print(f"  corpus done {time.time()-t_c:.1f}s")

    # Cosine + NDCG
    print("[e5-cuda] computing cosine NDCG@10...")
    ndcgs = []
    for i, qid in enumerate(q_test_ids):
        sims = c_emb @ q_emb[i]
        top = np.argsort(-sims)[:10]
        ranked = [c_ids[j] for j in top]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], k=10))
    mean_ndcg = float(np.mean(ndcgs))
    print(f"\n=== {ds} CUDA 4-bit nf4 NDCG@10 = {mean_ndcg:.4f} (n={len(ndcgs)}) ===")

    refs = {"nfcorpus": (0.140, 0.39), "arguana": (0.083, 0.55)}
    if ds in refs:
        rocm, pub = refs[ds]
        print(f"  ROCm bf16 (paper):     {rocm:.3f}")
        print(f"  Published reference:    ~{pub:.2f}")
        if mean_ndcg > 0.25:
            print(f"  → CUDA fixes ROCm artifact? YES (CUDA gives reasonable NDCG)")
        else:
            print(f"  → CUDA still anomalous? Pipeline issue more likely.")

    out_path = base / f"e5_cuda_4bit_{ds}_eval.json"
    out_path.write_text(json.dumps({
        "dataset": ds, "method": "E5-Mistral nf4 (bitsandbytes 4-bit)",
        "n_test_queries": len(ndcgs), "ndcg_at_10": mean_ndcg,
        "hardware": "NVIDIA RTX 5060 Laptop 8GB, CUDA 13.1, torch 2.11+cu130",
        "compute_dtype": "fp16", "quant_type": "nf4 (double quant)",
    }, indent=2))
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
