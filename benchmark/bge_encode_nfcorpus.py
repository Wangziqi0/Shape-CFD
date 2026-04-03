#!/usr/bin/env python3
"""
Encode NFCorpus corpus + queries using BGE-large (port 8082) and BGE-M3 (port 8083).
BGE queries need prefix: "Represent this sentence for searching relevant passages: "
Documents do NOT need prefix.
"""

import json
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "/home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus"
BATCH_SIZE = 32
MAX_WORKERS = 8
MAX_CHARS = 512

MODELS = {
    "bge_large": {"port": 8082, "dim": 1024},
    "bge_m3": {"port": 8083, "dim": 1024},
}

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_batch(texts, port):
    """Embed a batch of texts via llama.cpp API."""
    resp = requests.post(
        f"http://192.168.31.22:{port}/v1/embeddings",
        json={"input": texts, "encoding_format": "float"},
        timeout=120,
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def load_corpus():
    """Load corpus from corpus.jsonl"""
    docs = []
    with open(os.path.join(BASE_DIR, "corpus.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            docs.append({"_id": obj["_id"], "text": text[:MAX_CHARS]})
    return docs


def load_queries():
    """Load queries from queries.jsonl"""
    queries = []
    with open(os.path.join(BASE_DIR, "queries.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            queries.append({"_id": obj["_id"], "text": obj["text"][:MAX_CHARS]})
    return queries


def encode_items(items, port, add_prefix=False, label=""):
    """Encode a list of items in batches with concurrency."""
    # Prepare texts
    texts = []
    for item in items:
        t = item["text"]
        if add_prefix:
            t = QUERY_PREFIX + t
        texts.append(t)

    # Create batches
    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batches.append((i, texts[i : i + BATCH_SIZE]))

    results = [None] * len(texts)
    done = 0
    total = len(batches)
    t0 = time.time()

    def process_batch(batch_info):
        idx, batch_texts = batch_info
        vecs = embed_batch(batch_texts, port)
        return idx, vecs

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        for future in as_completed(futures):
            idx, vecs = future.result()
            for j, vec in enumerate(vecs):
                results[idx + j] = vec
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  {label}: {done}/{total} batches ({elapsed:.1f}s)")

    return results


def main():
    # Determine which models to encode
    models_to_run = list(MODELS.keys())
    if len(sys.argv) > 1:
        models_to_run = [m for m in sys.argv[1:] if m in MODELS]

    corpus = load_corpus()
    queries = load_queries()
    print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)} queries")

    for model_name in models_to_run:
        cfg = MODELS[model_name]
        port = cfg["port"]
        print(f"\n=== Encoding with {model_name} (port {port}) ===")

        # Check if files already exist
        corpus_file = os.path.join(BASE_DIR, f"{model_name}_corpus_vectors.jsonl")
        query_file = os.path.join(BASE_DIR, f"{model_name}_query_vectors.jsonl")

        if os.path.exists(corpus_file) and os.path.exists(query_file):
            print(f"  Files already exist, skipping. Delete to re-encode.")
            continue

        # Encode corpus (no prefix)
        print(f"  Encoding corpus ({len(corpus)} docs)...")
        corpus_vecs = encode_items(corpus, port, add_prefix=False, label="corpus")

        # Encode queries (with prefix)
        print(f"  Encoding queries ({len(queries)} queries)...")
        query_vecs = encode_items(queries, port, add_prefix=True, label="queries")

        # Write corpus vectors
        with open(corpus_file, "w") as f:
            for doc, vec in zip(corpus, corpus_vecs):
                obj = {"_id": doc["_id"], "vector": vec, "sentences": [vec]}
                f.write(json.dumps(obj) + "\n")
        print(f"  Wrote {corpus_file}")

        # Write query vectors
        with open(query_file, "w") as f:
            for q, vec in zip(queries, query_vecs):
                obj = {"_id": q["_id"], "text": q["text"], "vector": vec}
                f.write(json.dumps(obj) + "\n")
        print(f"  Wrote {query_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
