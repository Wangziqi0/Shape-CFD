"""
encode_beir_corpus_via_bge_m3.py
=================================
Encode BEIR corpus + queries via 9070XT BGE-M3 server (port 8080),
output bge_m3_corpus_vectors.jsonl + bge_m3_query_vectors.jsonl.

Compatible with build_first_stage_from_cache.py downstream.

Designed for new BEIR corpora not in current cache (trec-covid, webis-touche2020).

Usage:
    python benchmark/encode_beir_corpus_via_bge_m3.py --corpus trec-covid
    python benchmark/encode_beir_corpus_via_bge_m3.py --corpus webis-touche2020
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-doc-chars", type=int, default=2000)
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def embed_batch(texts, llama_url, max_chars=2000):
    truncated = [t[:max_chars] if t else "[EMPTY]" for t in texts]
    r = requests.post(
        f"{llama_url}/v1/embeddings",
        json={"input": truncated, "model": "bge-m3"},
        timeout=180,
    )
    r.raise_for_status()
    payload = r.json()
    return [d["embedding"] for d in payload["data"]]


def encode_jsonl(in_path, out_path, llama_url, batch_size, max_chars, label, id_key, text_extractor):
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[{label}] cached at {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB), skipping", file=sys.stderr)
        return

    items = []
    with open(in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            items.append({"id": str(d[id_key]), "text": text_extractor(d)})
    n = len(items)
    print(f"[{label}] encoding {n} items...", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh_out:
        for i in range(0, n, batch_size):
            batch = items[i:i + batch_size]
            texts = [b["text"] for b in batch]
            ids = [b["id"] for b in batch]
            try:
                embs = embed_batch(texts, llama_url, max_chars)
            except Exception as e:
                print(f"  WARN: batch {i} failed ({e}), skipping with zeros", file=sys.stderr)
                embs = [[0.0] * 1024 for _ in texts]

            for did, emb in zip(ids, embs):
                fh_out.write(json.dumps({"_id": did, "vector": emb, "sentences": []}, ensure_ascii=False) + "\n")
            written += len(batch)

            if (i // batch_size) % 50 == 0:
                elapsed = time.time() - t0
                rate = written / max(1.0, elapsed)
                eta = (n - written) / max(1.0, rate)
                print(f"  {label}: {written}/{n} ({rate:.1f} docs/s, ETA {eta/60:.1f} min)", file=sys.stderr)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{label}] done: {n} items in {elapsed/60:.1f} min, {size_mb:.1f} MB", file=sys.stderr)


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

    encode_jsonl(
        corpus_jsonl, out_corpus, args.llama_url,
        batch_size=args.batch_size, max_chars=args.max_doc_chars,
        label=f"{args.corpus}/corpus",
        id_key="_id",
        text_extractor=lambda d: (d.get("title", "") + " " + d.get("text", "")).strip(),
    )

    encode_jsonl(
        queries_jsonl, out_queries, args.llama_url,
        batch_size=args.batch_size, max_chars=args.max_doc_chars,
        label=f"{args.corpus}/queries",
        id_key="_id",
        text_extractor=lambda d: d["text"],
    )


if __name__ == "__main__":
    main()
