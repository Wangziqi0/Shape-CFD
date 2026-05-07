"""
encode_beir_robust.py
======================
ROBUST encoding of BEIR corpus + queries via BGE-M3 server (port 8080).

CRITICAL FIX vs encode_beir_corpus_via_bge_m3.py (which produced zero-vectors):
  - Per-document tokenize-then-chunk (use llama.cpp /tokenize endpoint)
  - HTTP 500 retry with exponential backoff (3 retries)
  - Skip doc on persistent failure (DO NOT write zero vector — this was the v1 RCA root cause)
  - Per-batch size adapt: start 16 (not 32), reduce to 8 / 4 / 2 on error
  - Verbose logging: every batch report success/skip
  - Output schema: {_id, vector (list[float]|null), sentences: []}
    - vector=null for skipped docs, downstream filter

Usage:
    python benchmark/encode_beir_robust.py --corpus trec-covid
    python benchmark/encode_beir_robust.py --corpus webis-touche2020
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--beir-root", default="/home/amd/HEZIMENG/legal-assistant/beir_data")
    p.add_argument("--llama-url", default="http://192.168.31.22:8080")
    p.add_argument("--initial-batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=480,
                   help="truncate to ≤ batch_size 512 token boundary - safety margin")
    p.add_argument("--max-chars-per-doc", type=int, default=1500,
                   help="character-level safety cap before tokenization")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-base-wait", type=float, default=2.0)
    return p.parse_args()


def tokenize_count(text, llama_url, max_retries=2):
    """Get token count via /tokenize endpoint."""
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{llama_url}/tokenize", json={"content": text}, timeout=30)
            r.raise_for_status()
            tokens = r.json().get("tokens", [])
            return len(tokens)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue
            print(f"    tokenize fail: {e}", file=sys.stderr)
            return None


def truncate_to_token_budget(text, llama_url, max_tokens, max_chars=1500):
    """Conservative truncation: cap chars first, then verify tokens, halve until fit."""
    text = text[:max_chars] if text else "[EMPTY]"
    tok = tokenize_count(text, llama_url)
    if tok is None:
        return text[:1000]  # fallback char-based cap
    if tok <= max_tokens:
        return text
    # binary trim
    lo, hi = 100, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = text[:mid]
        tc = tokenize_count(cand, llama_url)
        if tc is None or tc > max_tokens:
            hi = mid - 1
        else:
            lo = mid
    return text[:lo]


def embed_batch(texts, llama_url, max_retries=3, retry_base=2.0):
    """Embed batch with retry on HTTP 500. Returns list[Optional[List[float]]]."""
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{llama_url}/v1/embeddings",
                json={"input": texts, "model": "bge-m3"},
                timeout=180,
            )
            r.raise_for_status()
            payload = r.json()
            return [d["embedding"] for d in payload["data"]]
        except requests.HTTPError as e:
            if attempt < max_retries - 1:
                wait = retry_base ** attempt
                print(f"    HTTP {e.response.status_code if e.response else '?'}, retry {attempt+1}/{max_retries} after {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            return [None] * len(texts)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_base ** attempt)
                continue
            print(f"    embed fail: {e}", file=sys.stderr)
            return [None] * len(texts)
    return [None] * len(texts)


def adaptive_embed(texts, llama_url, initial_bs=16, min_bs=2, max_retries=3, retry_base=2.0):
    """
    Adaptive batch size: start at initial_bs, halve on error to min_bs.
    Returns list[Optional[List[float]]] same length as texts.
    """
    n = len(texts)
    out = [None] * n
    bs = initial_bs
    i = 0
    while i < n:
        batch = texts[i:i + bs]
        embs = embed_batch(batch, llama_url, max_retries=max_retries, retry_base=retry_base)
        if all(e is not None for e in embs):
            for j, e in enumerate(embs):
                out[i + j] = e
            i += bs
        else:
            # batch failed
            if bs > min_bs:
                bs = max(min_bs, bs // 2)
                print(f"    batch[{i}:{i+len(batch)}] failed, reducing batch size to {bs}", file=sys.stderr)
                continue
            else:
                # min_bs also failed: write None for these docs, advance
                for j, e in enumerate(embs):
                    out[i + j] = e  # may still be None
                skipped = sum(1 for e in embs if e is None)
                print(f"    SKIP {skipped} docs at min_bs={min_bs} after batch[{i}:{i+len(batch)}]", file=sys.stderr)
                i += len(batch)
    return out


def encode_jsonl_robust(in_path, out_path, llama_url, args, label, id_key, text_extractor):
    if out_path.exists() and out_path.stat().st_size > 1000:
        try:
            with open(out_path) as fh:
                first = json.loads(fh.readline())
                if first.get("vector") is not None:
                    print(f"[{label}] cached ({out_path.stat().st_size/1024/1024:.1f} MB), skip", file=sys.stderr)
                    return
        except Exception:
            pass

    items = []
    with open(in_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            items.append({"id": str(d[id_key]), "text": text_extractor(d)})
    n = len(items)
    print(f"[{label}] {n} items, max_tokens={args.max_tokens}, initial_bs={args.initial_batch_size}", file=sys.stderr)

    # truncate per doc to token budget (sample every 200 to estimate tokenization speed first)
    print(f"[{label}] truncating to token budget...", file=sys.stderr)
    t0 = time.time()
    truncated = []
    for j, item in enumerate(items):
        truncated.append(truncate_to_token_budget(item["text"], llama_url, args.max_tokens, args.max_chars_per_doc))
        if (j + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (j + 1) / max(1.0, elapsed)
            print(f"  truncate {j+1}/{n} ({rate:.0f} docs/s)", file=sys.stderr)
    print(f"[{label}] truncate done in {time.time()-t0:.0f}s", file=sys.stderr)

    print(f"[{label}] encoding...", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_skipped = 0
    n_written = 0
    BLOCK = 256  # write in blocks of 256 to allow incremental saving
    with open(out_path, "w", encoding="utf-8") as fh_out:
        for block_start in range(0, n, BLOCK):
            block_end = min(block_start + BLOCK, n)
            block_texts = truncated[block_start:block_end]
            block_ids = [items[k]["id"] for k in range(block_start, block_end)]
            embs = adaptive_embed(block_texts, llama_url,
                                  initial_bs=args.initial_batch_size,
                                  max_retries=args.max_retries,
                                  retry_base=args.retry_base_wait)
            for did, emb in zip(block_ids, embs):
                if emb is None:
                    n_skipped += 1
                fh_out.write(json.dumps({"_id": did, "vector": emb, "sentences": []}, ensure_ascii=False) + "\n")
                n_written += 1
            elapsed = time.time() - t0
            rate = n_written / max(1.0, elapsed)
            eta_min = (n - n_written) / max(1.0, rate) / 60
            print(f"  encode {n_written}/{n} (skip {n_skipped}, {rate:.1f} docs/s, ETA {eta_min:.1f}min)", file=sys.stderr)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    skip_pct = 100 * n_skipped / max(1, n)
    print(f"[{label}] done: {n_written}/{n} ({n_skipped} skipped = {skip_pct:.2f}%) in {elapsed/60:.1f}min, {size_mb:.1f}MB", file=sys.stderr)


def main():
    args = parse_args()
    corpus_root = Path(args.beir_root) / args.corpus
    corpus_jsonl = corpus_root / "corpus.jsonl"
    queries_jsonl = corpus_root / "queries.jsonl"
    if not corpus_jsonl.exists() or not queries_jsonl.exists():
        print(f"ERROR: missing {corpus_jsonl} or {queries_jsonl}", file=sys.stderr)
        sys.exit(2)

    out_corpus = corpus_root / "bge_m3_corpus_vectors.jsonl"
    out_queries = corpus_root / "bge_m3_query_vectors.jsonl"

    encode_jsonl_robust(
        corpus_jsonl, out_corpus, args.llama_url, args,
        label=f"{args.corpus}/corpus",
        id_key="_id",
        text_extractor=lambda d: (d.get("title", "") + " " + d.get("text", "")).strip(),
    )

    encode_jsonl_robust(
        queries_jsonl, out_queries, args.llama_url, args,
        label=f"{args.corpus}/queries",
        id_key="_id",
        text_extractor=lambda d: d["text"],
    )


if __name__ == "__main__":
    main()
