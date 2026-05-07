"""
build_first_stage_from_cache.py
================================
Build first-stage retrieval candidates from CACHED BGE-M3 v2 vectors
(bge_m3_corpus_vectors.jsonl + bge_m3_query_vectors.jsonl).

Output schema:
    {qid: {qid, query (text), candidates: [{doc_id, doc_text, first_stage_score}, ...]}}

Per-corpus runtime ~30s (cosine ranking dominated).
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beir-root", default="/home/amd/HEZIMENG/legal-assistant/beir_data")
    p.add_argument("--corpus", required=True,
                   choices=["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-doc-chars", type=int, default=2000)
    p.add_argument("--out-dir", type=Path,
                   default=Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/first_stage_bge_m3_v2"))
    return p.parse_args()


def load_text_lookup(corpus_root):
    doc_text = {}
    with open(corpus_root / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            doc_text[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    query_text = {}
    with open(corpus_root / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            query_text[d["_id"]] = d["text"]
    return doc_text, query_text


def load_test_qids(corpus_root):
    qrels_path = corpus_root / "qrels" / "test.tsv"
    if not qrels_path.exists():
        return None  # use all queries
    qids = set()
    with open(qrels_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            parts = line.strip().split("\t")
            if parts:
                qids.add(parts[0])
    return qids


def load_vectors_jsonl(path):
    """Returns (ids: list[str], embeddings: np.ndarray normalized)."""
    ids = []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            ids.append(str(d["_id"]))
            rows.append(d["vector"])
    embs = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embs / norms
    return ids, embs_norm


def main():
    args = parse_args()
    corpus_root = Path(args.beir_root) / args.corpus
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.corpus}_top{args.top_k}.json"

    t0 = time.time()
    print(f"[{args.corpus}] loading text lookup...", file=sys.stderr)
    doc_text, query_text = load_text_lookup(corpus_root)
    test_qids = load_test_qids(corpus_root)

    doc_vec_path = corpus_root / "bge_m3_corpus_vectors.jsonl"
    q_vec_path = corpus_root / "bge_m3_query_vectors.jsonl"
    if not doc_vec_path.exists() or not q_vec_path.exists():
        print(f"ERROR: missing BGE-M3 v2 vectors ({doc_vec_path}, {q_vec_path})", file=sys.stderr)
        sys.exit(2)

    print(f"[{args.corpus}] loading {doc_vec_path.name}...", file=sys.stderr)
    doc_ids, doc_embs = load_vectors_jsonl(doc_vec_path)
    print(f"  doc count: {len(doc_ids)}, embs shape: {doc_embs.shape}", file=sys.stderr)

    print(f"[{args.corpus}] loading {q_vec_path.name}...", file=sys.stderr)
    q_ids_all, q_embs_all = load_vectors_jsonl(q_vec_path)
    if test_qids:
        keep = [i for i, q in enumerate(q_ids_all) if q in test_qids]
        q_ids = [q_ids_all[i] for i in keep]
        q_embs = q_embs_all[keep]
    else:
        q_ids, q_embs = q_ids_all, q_embs_all
    print(f"  query count (test split): {len(q_ids)}, embs shape: {q_embs.shape}", file=sys.stderr)

    print(f"[{args.corpus}] cosine ranking top-{args.top_k}...", file=sys.stderr)
    sims = q_embs @ doc_embs.T  # (n_q, n_d)
    topk_idx = np.argsort(-sims, axis=1)[:, :args.top_k]

    output = {}
    for qi, qid in enumerate(q_ids):
        cands = []
        for dj in topk_idx[qi]:
            did = doc_ids[dj]
            cands.append({
                "doc_id": did,
                "doc_text": doc_text.get(did, "")[:args.max_doc_chars],
                "first_stage_score": float(sims[qi, dj]),
            })
        output[qid] = {
            "qid": qid,
            "query": query_text.get(qid, ""),
            "candidates": cands,
        }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"[{args.corpus}] DONE: {len(output)} queries -> {out_path}", file=sys.stderr)
    print(f"[{args.corpus}] file size: {out_path.stat().st_size / 1024 / 1024:.1f} MB, elapsed {elapsed:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
