"""
colbertv2_lap_eval_7b13_cpu.py — ColBERTv2 + Graph Laplacian smoothing on 7B13 (Linux CPU only).

Paper §1 第三层防御实测: ColBERTv2 first-stage + Lap smoothing → Δ NDCG@10 vs ColBERTv2 alone.
FiQA 因 5060 32 GB RAM 装不下 53 GB doc tensor working set 触发 page-file thrash, 改 7B13 跑.
7B13: EPYC 7B13, 256 cores, 503 GB RAM, NPS4 8 NUMA. CPU only 但 RAM 装得下 FiQA.

Self-contained, depends on: torch + transformers + numpy. No colbert-ir package needed.

Usage (7B13):
    cd /home/amd/HEZIMENG/Shape-CFD/benchmark
    nohup numactl --interleave=all python3 colbertv2_lap_eval_7b13_cpu.py \
        --corpus fiqa --batch-size 16 \
        > /tmp/colbertv2_lap_fiqa_7b13.log 2>&1 &

Output:
    /home/amd/HEZIMENG/Shape-CFD/benchmark/colbertv2_lap_<corpus>_results.json

Diff vs 5060 GPU script:
    - device 强制 cpu (no cuda check)
    - torch_dtype=float32 (CPU 不支持 fp16 efficient)
    - batch_size default 16 (CPU 多核 parallel, 不受 8 GB VRAM 限制)
    - torch.set_num_threads(32) (单 NUMA node 32 cores; --interleave=all 跨 8 NUMA 总 256)
    - beir-root default 改为 7B13 path /home/amd/HEZIMENG/legal-assistant/beir_data
    - output path 改为 7B13 dir
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, choices=[
        "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
    ])
    p.add_argument("--beir-root", default="/home/amd/HEZIMENG/legal-assistant/beir_data",
                   help="BEIR data root")
    p.add_argument("--model", default="colbert-ir/colbertv2.0")
    p.add_argument("--q-max-len", type=int, default=32)
    p.add_argument("--d-max-len", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=16,
                   help="CPU 多核可加大; 16 / 32 都能试")
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--lap-alpha", type=float, default=0.15)
    p.add_argument("--lap-iter", type=int, default=5)
    p.add_argument("--max-queries", type=int, default=None, help="for sanity test")
    p.add_argument("--num-threads", type=int, default=32,
                   help="torch CPU threads; 32 = 单 NUMA node, 配 --interleave=all 时 OS 跨 NUMA 调度")
    p.add_argument("--device", default="cpu", help="cpu only on 7B13")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_beir(corpus_root):
    docs, queries, qrels = {}, {}, {}
    with open(corpus_root / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            text = (d.get("title", "") + " " + d.get("text", "")).strip()
            docs[d["_id"]] = text
    with open(corpus_root / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    with open(corpus_root / "qrels" / "test.tsv", encoding="utf-8") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
            except ValueError:
                pass
    return docs, queries, qrels


def encode_batch(texts, tokenizer, model, max_len, device, prefix=None):
    """Encode texts to (batch, n_tokens, dim) normalized embeddings (ColBERT spec)."""
    if prefix:
        texts = [f"{prefix} {t}" for t in texts]
    enc = tokenizer(texts, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (B, L, D)
    out = normalize(out, p=2, dim=-1)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    out = out * mask
    return out.cpu()  # (B, L, D)


def maxsim(query_emb, doc_emb):
    """ColBERTv2 MaxSim: sum over q_tokens of max sim with d_tokens."""
    sim = query_emb @ doc_emb.T  # (Lq, Ld)
    return sim.max(dim=1).values.sum().item()


def ndcg_at_10(ranked_doc_ids, qrels_for_query):
    rel = np.array([qrels_for_query.get(d, 0) for d in ranked_doc_ids[:10]], dtype=float)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, 12))
    dcg = float(np.sum(rel * discounts[: len(rel)]))
    ideal = np.sort(np.array(list(qrels_for_query.values()), dtype=float))[::-1]
    idcg = float(np.sum(ideal[:10] * discounts[: min(10, len(ideal))]))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    args = parse_args()
    # 强制 CPU
    torch.set_num_threads(args.num_threads)
    print(f"[setup] torch.set_num_threads({args.num_threads}); device=cpu (forced); "
          f"interop_threads={torch.get_num_interop_threads()}", file=sys.stderr, flush=True)

    corpus_root = Path(args.beir_root) / args.corpus
    if not corpus_root.exists():
        print(f"ERROR: BEIR data not found at {corpus_root}", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cpu")
    print(f"[{args.corpus}] device={device}, model={args.model}", file=sys.stderr, flush=True)

    print(f"[{args.corpus}] loading model (fp32)...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # CPU: fp32. fp16 在 CPU 上慢且部分 op 不支持
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()

    print(f"[{args.corpus}] loading BEIR data from {corpus_root}...", file=sys.stderr, flush=True)
    docs, queries, qrels = load_beir(corpus_root)
    test_qids = list(qrels.keys())
    if args.max_queries:
        test_qids = test_qids[: args.max_queries]
    queries = {q: queries[q] for q in test_qids if q in queries}
    print(f"  {len(docs)} docs, {len(queries)} test queries, "
          f"{sum(len(v) for v in qrels.values())} qrels", file=sys.stderr, flush=True)

    # ---------- Encode all docs ----------
    t0 = time.time()
    print(f"[{args.corpus}] encoding docs (batch={args.batch_size})...",
          file=sys.stderr, flush=True)
    doc_ids = list(docs.keys())
    doc_embs = []
    for i in range(0, len(doc_ids), args.batch_size):
        batch_ids = doc_ids[i:i + args.batch_size]
        batch_texts = [docs[d] for d in batch_ids]
        emb = encode_batch(batch_texts, tokenizer, model, args.d_max_len, device, prefix="[D]")
        for j, d in enumerate(batch_ids):
            doc_embs.append(emb[j])
        if (i // args.batch_size) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + args.batch_size) / max(1.0, elapsed)
            eta = (len(doc_ids) - i) / max(1.0, rate)
            print(f"  doc {i}/{len(doc_ids)} ({rate:.1f}/s, ETA {eta/60:.1f}min)",
                  file=sys.stderr, flush=True)
    print(f"[{args.corpus}] doc encoding {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    # ---------- Encode all queries + MaxSim rank ----------
    print(f"[{args.corpus}] encoding queries + MaxSim ranking...", file=sys.stderr, flush=True)
    t1 = time.time()
    rerank_results = {}
    for q_idx, qid in enumerate(test_qids):
        q_text = queries[qid]
        q_emb_batch = encode_batch([q_text], tokenizer, model, args.q_max_len, device, prefix="[Q]")
        q_emb = q_emb_batch[0]
        scores = np.zeros(len(doc_ids))
        for j, d_emb in enumerate(doc_embs):
            scores[j] = maxsim(q_emb, d_emb)
        idx = np.argsort(-scores)
        rerank_results[qid] = [(doc_ids[i], float(scores[i])) for i in idx[:200]]
        if q_idx % 20 == 0:
            elapsed = time.time() - t1
            rate = (q_idx + 1) / max(1.0, elapsed)
            eta = (len(test_qids) - q_idx) / max(1.0, rate)
            print(f"  q {q_idx}/{len(test_qids)} ({rate:.2f} q/s, ETA {eta/60:.1f}min)",
                  file=sys.stderr, flush=True)
    print(f"[{args.corpus}] MaxSim ranking {time.time()-t1:.0f}s", file=sys.stderr, flush=True)

    # ---------- Phase A: ColBERTv2-only NDCG@10 ----------
    ndcgs_only = []
    for qid in test_qids:
        ranked = [d for d, _ in rerank_results[qid]]
        ndcgs_only.append(ndcg_at_10(ranked, qrels.get(qid, {})))

    # ---------- Phase B: ColBERTv2 + Lap smoothing ----------
    print(f"[{args.corpus}] applying Lap smoothing...", file=sys.stderr, flush=True)
    t2 = time.time()
    ndcgs_lap = []
    for qi, qid in enumerate(test_qids):
        ranked = rerank_results[qid][:55]
        if len(ranked) < 5:
            ndcgs_lap.append(ndcg_at_10([d for d, _ in ranked], qrels.get(qid, {})))
            continue
        rd_ids = [d for d, _ in ranked]
        rd_embs = [doc_embs[doc_ids.index(d)] for d in rd_ids]
        n = len(rd_embs)
        sim_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                s = (maxsim(rd_embs[i], rd_embs[j]) + maxsim(rd_embs[j], rd_embs[i])) / 2.0
                sim_mat[i, j] = sim_mat[j, i] = s
        k = min(args.knn_k, n - 1)
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            top_neighbors = np.argsort(-sim_mat[i])[:k + 1]
            for j in top_neighbors:
                if j != i:
                    adj[i, j] = sim_mat[i, j]
                    adj[j, i] = sim_mat[i, j]
        scores = np.array([s for _, s in ranked], dtype=np.float32)
        deg = adj.sum(axis=1) + 1e-9
        norm_adj = adj / deg[:, None]
        for _ in range(args.lap_iter):
            scores = (1 - args.lap_alpha) * scores + args.lap_alpha * (norm_adj @ scores)
        idx = np.argsort(-scores)
        ranked_lap = [rd_ids[i] for i in idx]
        ndcgs_lap.append(ndcg_at_10(ranked_lap, qrels.get(qid, {})))
    print(f"[{args.corpus}] Lap smoothing {time.time()-t2:.0f}s", file=sys.stderr, flush=True)

    # ---------- Phase C: paired bootstrap p-value ----------
    rng = np.random.default_rng(42)
    deltas = np.array(ndcgs_lap) - np.array(ndcgs_only)
    n_iter = 10000
    means = np.zeros(n_iter)
    n_q = len(deltas)
    for i in range(n_iter):
        idx = rng.integers(0, n_q, size=n_q)
        means[i] = deltas[idx].mean()
    mean_delta = float(deltas.mean())
    p_value = float(np.mean(means <= 0)) * 2 if mean_delta > 0 else float(np.mean(means >= 0)) * 2
    p_value = max(p_value, 1.0 / n_iter)

    out = {
        "corpus": args.corpus,
        "n_queries": len(test_qids),
        "n_docs": len(doc_ids),
        "config": {
            "model": args.model,
            "q_max_len": args.q_max_len,
            "d_max_len": args.d_max_len,
            "batch_size": args.batch_size,
            "knn_k": args.knn_k,
            "lap_alpha": args.lap_alpha,
            "lap_iter": args.lap_iter,
            "device": "cpu",
            "num_threads": args.num_threads,
            "host": "7B13",
        },
        "ndcg10": {
            "colbertv2_only": float(np.mean(ndcgs_only)),
            "colbertv2_plus_lap": float(np.mean(ndcgs_lap)),
            "delta_abs": mean_delta,
            "delta_rel_pct": 100 * mean_delta / max(1e-9, np.mean(ndcgs_only)),
        },
        "p_value_paired_bootstrap": p_value,
        "verdict": (
            "POSITIVE_AND_SIGNIFICANT" if mean_delta > 0 and p_value < 0.05 else
            "POSITIVE_BUT_NOT_SIGNIFICANT" if mean_delta > 0 else
            "NEGATIVE_OR_ZERO"
        ),
    }
    out_path = args.out or Path(f"/home/amd/HEZIMENG/Shape-CFD/benchmark/colbertv2_lap_{args.corpus}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\n=== {args.corpus} verdict ===", file=sys.stderr, flush=True)
    print(f"  ColBERTv2 only:         {out['ndcg10']['colbertv2_only']:.4f}", file=sys.stderr)
    print(f"  ColBERTv2 + Lap:        {out['ndcg10']['colbertv2_plus_lap']:.4f}", file=sys.stderr)
    print(f"  Δ:                      {out['ndcg10']['delta_abs']:+.4f} "
          f"({out['ndcg10']['delta_rel_pct']:+.2f}%)", file=sys.stderr)
    print(f"  paired bootstrap p:     {out['p_value_paired_bootstrap']:.4f}", file=sys.stderr)
    print(f"  verdict:                {out['verdict']}", file=sys.stderr)
    print(f"  output:                 {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
