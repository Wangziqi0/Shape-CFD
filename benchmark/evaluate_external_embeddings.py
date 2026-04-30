#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_external_embeddings.py — Reviewer 2 cross-model 验证

读取 9070XT (192.168.31.22) 上 Agent C 落盘的 ColBERTv2 / E5-Mistral 等外部模型 embedding,
在 7b13 CPU 上跑 NDCG@10 + bootstrap 显著性, 用于 paper §RQ2 cross-model robustness.

输入:
    9070XT 上的 parquet 路径(默认通过 SSH/sshfs 访问, 也支持本地拷贝):
    /home/amd/Shape-CFD-9070XT/embeddings/<MODEL>/<DATASET>/{corpus,queries}/*.parquet

    每条 row schema:
        id          : str       — doc/query id (BEIR 原 id)
        token_emb   : bytes     — float32 token embedding, shape (n_tokens, dim)
        mean_emb    : bytes     — float32 mean-pooled embedding, shape (dim,)
        n_tokens    : int64

输出:
    /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/external_eval_<model>_<dataset>.json

支持两种 retrieval:
    1) mean_pool   : query.mean_emb · doc.mean_emb         (cosine, dense baseline)
    2) maxsim      : ColBERT-style late interaction
                       sum_q max_d (q_token · d_token)     (token-level)

NDCG@10 + paired bootstrap 95% CI (N=10000), 与现有 fusion sweep 一致 protocol.

用法:
    python evaluate_external_embeddings.py \\
        --model colbertv2 \\
        --dataset nfcorpus \\
        --remote-host 192.168.31.22 \\
        --remote-path /home/amd/Shape-CFD-9070XT/embeddings \\
        --qrels /home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data/nfcorpus/qrels.tsv \\
        --output-dir /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results

支持 --local-path 直接读本地 parquet (跳过 ssh/scp).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import io
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------- 工具函数 ----------

def log(msg: str, *args, **kw) -> None:
    print(f"[external_eval] {msg}", *args, file=sys.stderr, flush=True, **kw)


def load_qrels(qrels_tsv: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_tsv.open() as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def fetch_remote_parquet(remote_host: str, remote_path: str, local_cache_dir: Path) -> Path:
    """
    通过 scp 拉取 remote parquet 文件到本地 cache (如果尚未存在).
    返回本地路径.
    """
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    rel = Path(remote_path).name
    local_path = local_cache_dir / rel
    if local_path.exists() and local_path.stat().st_size > 0:
        log(f"cache hit: {local_path}")
        return local_path
    log(f"scp {remote_host}:{remote_path} -> {local_path}")
    cmd = ["scp", "-q", f"{remote_host}:{remote_path}", str(local_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed: {r.stderr}")
    return local_path


def list_remote_parquets(remote_host: str, remote_dir: str) -> List[str]:
    """
    通过 ssh ls 列出远程 parquet 文件路径.
    """
    cmd = ["ssh", remote_host, f"ls {remote_dir}/*.parquet 2>/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return []
    return [p for p in r.stdout.strip().split("\n") if p]


def load_parquet_rows(parquet_files: List[Path], embedding_dim: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    加载 parquet 文件, 返回 dict: id -> {'token_emb': np.ndarray (n_tokens, dim), 'mean_emb': np.ndarray (dim,), 'n_tokens': int}
    """
    import pyarrow.parquet as pq
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for pf in parquet_files:
        log(f"loading {pf} ...")
        tbl = pq.read_table(str(pf))
        ids = tbl.column("id").to_pylist()
        token_blobs = tbl.column("token_emb").to_pylist()
        mean_blobs = tbl.column("mean_emb").to_pylist()
        n_tok_list = tbl.column("n_tokens").to_pylist()
        for i, did in enumerate(ids):
            n_tok = n_tok_list[i]
            tok_arr = np.frombuffer(token_blobs[i], dtype=np.float32).reshape(n_tok, embedding_dim)
            mean_arr = np.frombuffer(mean_blobs[i], dtype=np.float32).reshape(embedding_dim)
            out[did] = {
                "token_emb": tok_arr,
                "mean_emb": mean_arr,
                "n_tokens": n_tok,
            }
    log(f"loaded {len(out)} docs")
    return out


# ---------- Retrieval 算法 ----------

def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2 normalize each row of x (in-place safe)."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def mean_pool_score(q_mean: np.ndarray, d_means: np.ndarray) -> np.ndarray:
    """
    q_mean: (dim,), d_means: (n_docs, dim) — 已 L2 normalize 后即为 cosine.
    返回 (n_docs,) score.
    """
    return d_means @ q_mean


def maxsim_score(q_tokens: np.ndarray, d_tokens: np.ndarray) -> float:
    """
    ColBERT-style MaxSim:  sum_{q in Q} max_{d in D} (q · d)
    q_tokens: (n_q, dim), d_tokens: (n_d, dim) — 都已 L2 normalize.
    """
    # (n_q, n_d) similarity matrix
    sim = q_tokens @ d_tokens.T
    # max over d for each q, then sum
    return float(sim.max(axis=1).sum())


# ---------- NDCG ----------

def ndcg_at_k(ranked_doc_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        rel = qrel.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted(qrel.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        if rel > 0:
            idcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


# ---------- Bootstrap ----------

def paired_bootstrap_ci(scores_a: List[float], scores_b: List[float], n_bootstrap: int = 10000,
                        seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """
    Paired bootstrap test on per-query score difference.
    返回: {'mean_diff': ..., 'ci_low': ..., 'ci_high': ..., 'p_value_two_sided': ...}
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b
    n = len(diff)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    mean_diff = float(diff.mean())
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # 双侧 p-value: 在 0 处的 fraction
    if mean_diff >= 0:
        p = float(2 * (boot_means <= 0).mean())
    else:
        p = float(2 * (boot_means >= 0).mean())
    p = min(p, 1.0)
    return {
        "mean_diff": mean_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p,
        "n_queries": n,
    }


# ---------- Main ----------

MODEL_DIM = {
    "colbertv2": 768,
    "e5_mistral": 4096,  # E5-Mistral-7B 输出维度, 待 Agent C 跑出后核对
}


def discover_parquet_files(remote_host: str | None, remote_path: str | None,
                            local_path: str | None, model: str, dataset: str,
                            split: str, cache_dir: Path) -> List[Path]:
    """
    返回本地 parquet 文件列表 (split = 'queries' or 'corpus')
    """
    if local_path:
        ldir = Path(local_path) / model / dataset / split
        return sorted(ldir.glob("*.parquet"))
    assert remote_host and remote_path
    rdir = f"{remote_path.rstrip('/')}/{model}/{dataset}/{split}"
    rfiles = list_remote_parquets(remote_host, rdir)
    local_files = []
    for rf in rfiles:
        lf = fetch_remote_parquet(remote_host, rf, cache_dir / model / dataset / split)
        local_files.append(lf)
    return local_files


def evaluate(args) -> Dict:
    qrels_path = Path(args.qrels)
    qrels = load_qrels(qrels_path)
    log(f"qrels: {len(qrels)} queries with relevance")

    cache_dir = Path(args.cache_dir)

    # 加载 queries / corpus
    q_files = discover_parquet_files(args.remote_host, args.remote_path, args.local_path,
                                       args.model, args.dataset, "queries", cache_dir)
    c_files = discover_parquet_files(args.remote_host, args.remote_path, args.local_path,
                                       args.model, args.dataset, "corpus", cache_dir)
    if not q_files:
        raise RuntimeError("no query parquet files found")
    if not c_files:
        raise RuntimeError("no corpus parquet files found")

    dim = MODEL_DIM.get(args.model, args.dim)
    if dim is None:
        raise ValueError(f"unknown embedding dim for model={args.model}, pass --dim")

    queries = load_parquet_rows(q_files, dim)
    corpus = load_parquet_rows(c_files, dim)

    # 预 L2 normalize 所有 mean_emb / token_emb
    log("L2-normalizing all embeddings ...")
    corpus_ids = list(corpus.keys())
    n_corpus = len(corpus_ids)
    d_means = np.stack([normalize_rows(corpus[d]["mean_emb"][None, :])[0] for d in corpus_ids])
    log(f"d_means shape: {d_means.shape}")

    for qid in queries:
        queries[qid]["mean_emb"] = normalize_rows(queries[qid]["mean_emb"][None, :])[0]
        queries[qid]["token_emb"] = normalize_rows(queries[qid]["token_emb"])
    for did in corpus:
        corpus[did]["token_emb"] = normalize_rows(corpus[did]["token_emb"])

    # 评估每个 query (限制为 qrels 里有的)
    eval_qids = [q for q in qrels if q in queries]
    log(f"evaluating {len(eval_qids)} queries (have both qrels and embedding)")

    if args.max_queries > 0:
        eval_qids = eval_qids[: args.max_queries]
        log(f"limited to {len(eval_qids)} queries")

    ndcg_mean: List[float] = []
    ndcg_maxsim: List[float] = []
    K = args.k

    t0 = time.time()
    step = max(1, len(eval_qids) // 20)

    for qi, qid in enumerate(eval_qids):
        qrel = qrels[qid]
        q_data = queries[qid]

        # mean-pool retrieval
        scores_mean = mean_pool_score(q_data["mean_emb"], d_means)
        # top-K
        topk_idx = np.argsort(-scores_mean)[:K]
        ranked_mean = [corpus_ids[i] for i in topk_idx]
        ndcg_mean.append(ndcg_at_k(ranked_mean, qrel, K))

        if not args.skip_maxsim:
            # MaxSim retrieval — 需要遍历所有 doc, O(n_q * n_d * dim) per doc
            # 优化: 先用 mean-pool top-N 重排 (默认 top-200 重排, paper 默认 top-1000)
            rerank_n = args.rerank_top_n
            cand_idx = np.argsort(-scores_mean)[:rerank_n]
            cand_ids = [corpus_ids[i] for i in cand_idx]
            maxsim_scores = []
            for did in cand_ids:
                s = maxsim_score(q_data["token_emb"], corpus[did]["token_emb"])
                maxsim_scores.append(s)
            maxsim_arr = np.asarray(maxsim_scores)
            order = np.argsort(-maxsim_arr)[:K]
            ranked_maxsim = [cand_ids[i] for i in order]
            ndcg_maxsim.append(ndcg_at_k(ranked_maxsim, qrel, K))

        if (qi + 1) % step == 0 or qi == len(eval_qids) - 1:
            dt = time.time() - t0
            eta = dt / (qi + 1) * (len(eval_qids) - qi - 1)
            print(f"  [{args.model}/{args.dataset}] {qi+1}/{len(eval_qids)} ({dt:.0f}s, ETA {eta:.0f}s)",
                  file=sys.stderr, flush=True)

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "n_queries": len(eval_qids),
        "K": K,
        "rerank_top_n": args.rerank_top_n,
        "mean_pool_ndcg10": float(np.mean(ndcg_mean)),
        "mean_pool_ndcg10_std": float(np.std(ndcg_mean)),
    }
    if not args.skip_maxsim:
        summary["maxsim_ndcg10"] = float(np.mean(ndcg_maxsim))
        summary["maxsim_ndcg10_std"] = float(np.std(ndcg_maxsim))
        # paired bootstrap maxsim vs mean_pool
        bs = paired_bootstrap_ci(ndcg_maxsim, ndcg_mean, n_bootstrap=args.n_bootstrap)
        summary["bootstrap_maxsim_vs_mean"] = bs

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["colbertv2", "e5_mistral"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--remote-host", default="192.168.31.22")
    p.add_argument("--remote-path", default="/home/amd/Shape-CFD-9070XT/embeddings")
    p.add_argument("--local-path", default=None,
                   help="若已有本地 embedding 路径, 跳过 ssh; e.g. /home/amd/Shape-CFD-9070XT/embeddings")
    p.add_argument("--qrels", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="/tmp/external_eval_cache")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rerank-top-n", type=int, default=200,
                   help="MaxSim 重排候选数; 默认 200, 论文常用 1000")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--skip-maxsim", action="store_true",
                   help="只跑 mean-pool baseline (省时, 用于 quick smoke test)")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--dim", type=int, default=None,
                   help="embedding 维度 (若 model 不在 MODEL_DIM dict 里需手填)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate(args)

    out_file = out_dir / f"external_eval_{args.model}_{args.dataset}.json"
    with out_file.open("w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_file}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
