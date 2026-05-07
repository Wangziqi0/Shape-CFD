"""
build_first_stage_for_rerank.py
================================
Generate first-stage retrieval candidates (BGE-M3 v2 cosine top-K with doc text)
for downstream pairwise/setwise LLM rerank. Output schema is consumed by
benchmark/llm_rerank_pairwise_setwise.py.

For each BEIR corpus in {nfcorpus, scifact, arguana, scidocs, fiqa}:
  1. Load corpus.jsonl + queries.jsonl + qrels/test.tsv
  2. Embed corpus docs via BGE-M3 llama-server 8080 (9070XT, batched)
  3. Embed queries via same server
  4. Cosine similarity → top-K candidates per query (K=20 default)
  5. Dump JSON: {qid: {query_text, candidates: [{doc_id, doc_text}]}}

Output path:
  benchmark/data/results/first_stage_bge_m3_v2/<corpus>_top<K>.json

Caching:
  - per-corpus doc embeddings cached as numpy .npy in /tmp/embed_cache_<corpus>.npy
  - if cache exists and matches doc count, skipped re-encode
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
    import numpy as np
except ImportError:
    print("error: pip install requests numpy", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beir-root", default="/home/amd/HEZIMENG/legal-assistant/beir_data")
    p.add_argument("--corpus", required=True, choices=[
        "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
    ])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--llama-url", default="http://192.168.31.22:8080",
                   help="BGE-M3 llama-server (port 8080 on 9070XT)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-doc-chars", type=int, default=2000,
                   help="truncate doc text to avoid HTTP 500 (BGE-M3 batch boundary)")
    p.add_argument("--embed-cache", type=Path, default=None)
    p.add_argument("--out-dir", type=Path,
                   default=Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/first_stage_bge_m3_v2"))
    return p.parse_args()


def embed_batch(texts, llama_url, max_chars=2000):
    """Embed a batch of strings via /v1/embeddings endpoint (OpenAI-compatible)."""
    truncated = [t[:max_chars] if t else "[EMPTY]" for t in texts]
    r = requests.post(
        f"{llama_url}/v1/embeddings",
        json={"input": truncated, "model": "bge-m3"},
        timeout=180,
    )
    r.raise_for_status()
    payload = r.json()
    embs = np.array([d["embedding"] for d in payload["data"]], dtype=np.float32)
    return embs


def encode_corpus_or_queries(items, llama_url, batch_size=32, max_chars=2000, label=""):
    """Encode list of {id, text} via BGE-M3 server, returns (ids, embeddings_normalized)."""
    ids = [item["id"] for item in items]
    texts = [item["text"] for item in items]
    n = len(texts)
    embs = np.zeros((n, 1024), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        try:
            be = embed_batch(batch, llama_url, max_chars)
            embs[i:i+len(batch)] = be
        except Exception as e:
            print(f"  WARN: batch {i} failed ({e}), using zeros", file=sys.stderr)
        if (i // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + batch_size) / max(1.0, elapsed)
            eta = (n - i - batch_size) / max(1.0, rate)
            print(f"  {label}: {i+batch_size}/{n} ({rate:.1f} docs/s, ETA {eta:.0f}s)", file=sys.stderr)
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embs / norms
    print(f"  {label}: encoded {n} items in {time.time()-t0:.0f}s", file=sys.stderr)
    return ids, embs_norm


def load_beir_corpus(corpus_root):
    items = []
    with open(corpus_root / "corpus.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            text = d.get("title", "") + " " + d.get("text", "")
            items.append({"id": d["_id"], "text": text.strip(), "raw_doc_text": text.strip()})
    return items


def load_beir_queries(corpus_root):
    items = []
    qfile = corpus_root / "queries.jsonl"
    with open(qfile, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            items.append({"id": d["_id"], "text": d["text"], "raw_query_text": d["text"]})
    return items


def load_qrels_test(corpus_root):
    qrels_test = set()
    qrels_path = None
    for cand in [corpus_root / "qrels" / "test.tsv", corpus_root / "qrels.test.tsv"]:
        if cand.exists():
            qrels_path = cand
            break
    if qrels_path is None:
        # fallback: collect any qrels test
        for p in (corpus_root / "qrels").glob("*test*") if (corpus_root / "qrels").exists() else []:
            qrels_path = p
            break
    if qrels_path is None:
        return set()  # all queries
    with open(qrels_path, "r", encoding="utf-8") as fh:
        next(fh, None)  # skip header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                qrels_test.add(parts[0])
    return qrels_test


def main():
    args = parse_args()
    corpus_root = Path(args.beir_root) / args.corpus
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.corpus}_top{args.top_k}.json"

    print(f"[{args.corpus}] loading BEIR data from {corpus_root}", file=sys.stderr)
    corpus_items = load_beir_corpus(corpus_root)
    query_items = load_beir_queries(corpus_root)
    qrels_test = load_qrels_test(corpus_root)
    if qrels_test:
        query_items = [q for q in query_items if q["id"] in qrels_test]
    print(f"[{args.corpus}] corpus={len(corpus_items)}, queries={len(query_items)}, qrels_test={len(qrels_test)}", file=sys.stderr)

    cache_path = args.embed_cache or Path(f"/tmp/embed_cache_{args.corpus}_bge_m3.npy")
    cache_ids_path = cache_path.with_suffix(".ids.txt")
    if cache_path.exists() and cache_ids_path.exists():
        cached_ids = cache_ids_path.read_text(encoding="utf-8").splitlines()
        if len(cached_ids) == len(corpus_items):
            print(f"[{args.corpus}] using cached doc embeddings: {cache_path}", file=sys.stderr)
            doc_embs = np.load(cache_path)
            doc_ids = cached_ids
        else:
            cache_path.unlink(missing_ok=True)

    if not cache_path.exists():
        print(f"[{args.corpus}] encoding {len(corpus_items)} docs (no cache)...", file=sys.stderr)
        doc_ids, doc_embs = encode_corpus_or_queries(
            corpus_items, args.llama_url,
            batch_size=args.batch_size, max_chars=args.max_doc_chars,
            label=f"{args.corpus}/docs",
        )
        np.save(cache_path, doc_embs)
        cache_ids_path.write_text("\n".join(doc_ids), encoding="utf-8")
        print(f"[{args.corpus}] doc cache saved: {cache_path}", file=sys.stderr)

    print(f"[{args.corpus}] encoding {len(query_items)} queries...", file=sys.stderr)
    q_ids, q_embs = encode_corpus_or_queries(
        query_items, args.llama_url,
        batch_size=args.batch_size, max_chars=args.max_doc_chars,
        label=f"{args.corpus}/queries",
    )
    print(f"[{args.corpus}] cosine ranking top-{args.top_k}...", file=sys.stderr)
    sims = q_embs @ doc_embs.T  # (n_q, n_d)
    topk_idx = np.argsort(-sims, axis=1)[:, :args.top_k]

    # build doc text lookup
    doc_text_by_id = {it["id"]: it["raw_doc_text"] for it in corpus_items}
    query_text_by_id = {it["id"]: it["raw_query_text"] for it in query_items}

    output = {}
    for qi, qid in enumerate(q_ids):
        cands = []
        for rank, dj in enumerate(topk_idx[qi]):
            did = doc_ids[dj]
            cands.append({
                "doc_id": did,
                "doc_text": doc_text_by_id.get(did, "")[:args.max_doc_chars],
                "first_stage_score": float(sims[qi, dj]),
            })
        output[qid] = {
            "qid": qid,
            "query": query_text_by_id.get(qid, ""),
            "candidates": cands,
        }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)
    print(f"[{args.corpus}] DONE: {len(output)} queries → {out_path}", file=sys.stderr)
    print(f"[{args.corpus}] file size: {out_path.stat().st_size / 1024 / 1024:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
