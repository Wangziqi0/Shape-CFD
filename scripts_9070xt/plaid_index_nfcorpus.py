#!/usr/bin/env python3
"""
plaid_index_nfcorpus.py — Run PLAID (ColBERTv2 efficient retrieval) on NFCorpus
=================================================================================
Closes Senior Reviewer + Action Editor C-2: actually run PLAID instead of citing
Santhanam 2022 numbers. Use 9070XT ROCm 7.2 + colbert-ir 0.2.14 + faiss-cpu.

monkey-patches transformers.AdamW (deprecated) before importing colbert.
"""
import os
import sys
import time
import json
from pathlib import Path
import multiprocessing as mp

# Monkey-patch deprecated AdamW for newer transformers (4.40+ deprecated, 5.x removed)
import transformers
import torch.optim
transformers.AdamW = torch.optim.AdamW
# Also patch get_linear_schedule_with_warmup if missing
if not hasattr(transformers, "get_linear_schedule_with_warmup"):
    from transformers.optimization import get_linear_schedule_with_warmup
    transformers.get_linear_schedule_with_warmup = get_linear_schedule_with_warmup

# Patch base PreTrainedModel.all_tied_weights_keys for transformers 5.x compat
# (transformers 4.x doesn't need this; harmless on 4.x)
from transformers import PreTrainedModel
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

import numpy as np


def jsonl_iter(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


def ndcg_at_k(ranked_doc_ids, qrel, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
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
    # Force "fork" start method to avoid spawn pickling issues
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Indexer, Searcher

    # Paths
    BEIR_ROOT = Path("/home/amd/Shape-CFD-9070XT/beir_data")
    DATASET = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    DATA = BEIR_ROOT / DATASET
    WORK = Path(f"/tmp/plaid_{DATASET}")
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"[plaid] dataset = {DATASET}")
    print(f"[plaid] data    = {DATA}")
    print(f"[plaid] work    = {WORK}")

    # ---------- 1. Convert BEIR -> ColBERT TSV ----------
    print("[plaid] converting corpus.jsonl -> collection.tsv ...")
    pid_to_doc_id = {}
    with open(WORK / "collection.tsv", "w", encoding="utf-8") as f:
        for pid, rec in enumerate(jsonl_iter(DATA / "corpus.jsonl")):
            text = (rec.get("title", "") + " " + rec.get("text", "")).strip().replace("\t", " ").replace("\n", " ")
            pid_to_doc_id[pid] = rec["_id"]
            f.write(f"{pid}\t{text}\n")
    print(f"  {len(pid_to_doc_id)} docs")

    print("[plaid] converting queries.jsonl -> queries.tsv ...")
    qid_to_query_id = {}
    queries_text = {}
    with open(WORK / "queries.tsv", "w", encoding="utf-8") as f:
        for qid, rec in enumerate(jsonl_iter(DATA / "queries.jsonl")):
            text = rec["text"].strip().replace("\t", " ").replace("\n", " ")
            qid_to_query_id[qid] = rec["_id"]
            queries_text[qid] = text
            f.write(f"{qid}\t{text}\n")
    print(f"  {len(qid_to_query_id)} queries")

    qrels_path = DATA / "qrels" / "test.tsv" if (DATA / "qrels" / "test.tsv").exists() else DATA / "qrels.tsv"
    qrels = {}
    with open(qrels_path) as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 3:
                qrels.setdefault(p[0], {})[p[1]] = int(p[2])
    print(f"  {len(qrels)} test queries with qrels")

    query_id_to_qid = {v: k for k, v in qid_to_query_id.items()}

    # ---------- 2. PLAID indexing ----------
    print("\n[plaid] === PLAID indexing ===")
    EXPERIMENT = f"plaid_{DATASET}"
    INDEX_NAME = f"{DATASET}_plaid_idx"

    t_idx_start = time.time()
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT, root=str(WORK / "experiments"))):
        config = ColBERTConfig(
            nbits=2,
            doc_maxlen=180,
            query_maxlen=32,
            kmeans_niters=4,
        )
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(name=INDEX_NAME, collection=str(WORK / "collection.tsv"), overwrite=True)
    t_idx_end = time.time()
    print(f"[plaid] indexing wall: {t_idx_end - t_idx_start:.1f} s")

    # ---------- 3. PLAID retrieval ----------
    print("\n[plaid] === PLAID retrieval ===")
    t_search_start = time.time()
    rankings = {}
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT, root=str(WORK / "experiments"))):
        searcher = Searcher(index=INDEX_NAME)
        for query_id in qrels:
            if query_id not in query_id_to_qid:
                continue
            qid = query_id_to_qid[query_id]
            qtext = queries_text.get(qid, "")
            if not qtext:
                continue
            results = searcher.search(qtext, k=10)
            rankings[qid] = list(zip(results[0], results[2]))
    t_search_end = time.time()
    n_q = len(rankings)
    total_search_s = t_search_end - t_search_start
    ms_per_q = (total_search_s / n_q * 1000) if n_q else 0
    print(f"[plaid] retrieval wall: {total_search_s:.1f} s for {n_q} queries = {ms_per_q:.1f} ms/q")

    # ---------- 4. NDCG@10 ----------
    print("\n[plaid] === NDCG@10 evaluation ===")
    ndcgs = []
    for query_id, rels in qrels.items():
        if query_id not in query_id_to_qid:
            continue
        qid = query_id_to_qid[query_id]
        if qid not in rankings:
            continue
        ranked_pids = [r[0] for r in rankings[qid]]
        ranked_doc_ids = [pid_to_doc_id.get(p, "") for p in ranked_pids]
        ndcgs.append(ndcg_at_k(ranked_doc_ids, rels, k=10))
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0
    print(f"[plaid] {DATASET} PLAID NDCG@10 = {mean_ndcg:.4f} (n={len(ndcgs)})")
    print(f"  Reference: ColBERTv2 full MaxSim NDCG@10 = 0.3147 (paper Table 4)")
    print(f"  Reference: PLAID published latency on MS MARCO ~58 ms/q (Santhanam 2022 Table 6)")

    # ---------- 5. Save ----------
    out = {
        "dataset": DATASET,
        "method": "PLAID (ColBERTv2 + 2-bit centroid quantization)",
        "n_queries": len(ndcgs),
        "ndcg_at_10": mean_ndcg,
        "indexing_wall_s": t_idx_end - t_idx_start,
        "retrieval_wall_s": total_search_s,
        "latency_ms_per_query": ms_per_q,
        "config": {
            "nbits": 2,
            "doc_maxlen": 180,
            "query_maxlen": 32,
            "checkpoint": "colbert-ir/colbertv2.0",
        },
        "hardware": "AMD RX 9070 XT, ROCm 7.2, PyTorch 2.10.0+rocm7.2.0",
    }
    out_path = Path("/home/amd/Shape-CFD-9070XT/outputs") / f"plaid_{DATASET}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
