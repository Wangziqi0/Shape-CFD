#!/usr/bin/env python3
"""
llm_rerank_bench.py — Reviewer 3 LLM-based reranker baseline
====================================================================

实现 listwise / pairwise / setwise 三种 LLM rerank（用 Qwen3-8B 通过 SiliconFlow API）。

Pipeline:
  1. 用现有 cosine top-100 拿候选（从 *_corpus_vectors.jsonl + *_query_vectors.jsonl）
  2. 用 LLM rerank 这 100 个候选
  3. 取 top-10 算 NDCG@10
  4. 与原 cosine baseline 比较

跑哪些:
  - listwise + pairwise on NFCorpus + SciFact
  - setwise (RankGPT-style sliding window) 同上
  - 其他 dataset 太慢推迟

输出:
  /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/llm_rerank_results.json
  per-query JSONL: llm_rerank_<dataset>_<method>.jsonl

用法:
  python3 llm_rerank_bench.py --datasets nfcorpus,scifact --methods listwise,pairwise \
      --max_queries 50 --candidates 100 --top_k 10
"""
import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from tqdm import tqdm

# ---------- config ----------
# 默认指向 9070XT 本地 llama-server (Qwen3-8B-Q4_K_M, 端口 8082)
API_BASE = "http://192.168.31.22:8082/v1/chat/completions"
LLM_MODEL = "qwen3-8b"
DEFAULT_BEIR_DIR = "/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data"
DEFAULT_OUT_DIR = "/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results"
# Qwen3 thinking 默认开，会消耗 max_tokens 在 reasoning 上；listwise 不需要 thinking
QWEN_NO_THINK_PREFIX = "/no_think "


def load_api_key():
    """优先读 .env，回退环境变量。本地 llama-server 不验证 key，返回 dummy 即可。"""
    for p in [Path("/home/amd/HEZIMENG/legal-assistant/.env"),
              Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/.env")]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("LOCAL_EMBED_KEY="):
                    return line.split("=", 1)[1].strip()
                if line.startswith("SILICONFLOW_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return (os.environ.get("LOCAL_EMBED_KEY") or os.environ.get("SILICONFLOW_API_KEY")
            or "no-key-needed-for-local-llama-server")


# ---------- helpers ----------
def load_jsonl(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    arr.append(json.loads(line))
                except Exception:
                    pass
    return arr


def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.startswith("query"):
                    continue
            parts = line.split("\t")
            if len(parts) >= 3:
                q, d, s = parts[0], parts[1], int(parts[2])
                qrels.setdefault(q, {})[d] = s
    return qrels


def cosine_topk(query_vec, doc_vecs, doc_ids, k=100):
    """单 query cosine top-k. doc_vecs: (N, D) ndarray; query_vec: (D,)."""
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    dn = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    sims = dn @ qn
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(doc_ids[i], float(sims[i])) for i in idx]


def compute_ndcg(ranked_ids, qrel, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k]):
        rel = qrel.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / np.log2(i + 2)
    ideal = sorted(qrel.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        if rel > 0:
            idcg += (2 ** rel - 1) / np.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------- LLM API call ----------
def call_llm(prompt, api_key, max_tokens=512, temperature=0.0, timeout=60, max_retries=3,
             api_base=None, model=None):
    """支持运行时 override api_base / model，且 Qwen3 thinking 默认禁用。"""
    base = api_base or API_BASE
    mdl = model or LLM_MODEL
    full_prompt = QWEN_NO_THINK_PREFIX + prompt if "qwen" in mdl.lower() else prompt
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "model": mdl,
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(base, json=body, headers=headers, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            if r.status_code >= 500:
                time.sleep(1 + attempt); continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"  [llm error] {e}", file=sys.stderr)
    return None


# ---------- listwise rerank ----------
LISTWISE_PROMPT = """You are a search ranker. The user has a query and {n} candidate passages numbered [1] to [{n}].
Rank the passages from most relevant to least relevant for the query.

Query: {query}

Passages:
{passages}

Output ONLY the ranking as a comma-separated list of numbers, e.g. "[3], [1], [5], [2], [4]".
No explanation. Most relevant first."""


def parse_listwise_output(text, n):
    """Parse "[3], [1], [5]..." or "3, 1, 5..." into list of ints."""
    if not text:
        return list(range(n))  # fallback: identity
    # extract all integers
    nums = [int(x) for x in re.findall(r"\d+", text)]
    # 1-indexed → 0-indexed; dedup; clip to [0, n-1]
    seen = set()
    order = []
    for x in nums:
        if 1 <= x <= n and x - 1 not in seen:
            order.append(x - 1)
            seen.add(x - 1)
    # 补上 missing
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order[:n]


def listwise_rerank(query, candidates, api_key, top_k=10, window_size=20):
    """RankGPT-style sliding window: 一次只 rank window_size 个，从最末窗口开始往前推。
    candidates: List[(doc_id, text, cos_score)]
    """
    if not candidates:
        return []
    n = len(candidates)
    # 简化版：只做一个 window 覆盖前 max(top_k, window_size) 个
    rerank_n = min(window_size, n)
    head = candidates[:rerank_n]
    tail = candidates[rerank_n:]

    passages = "\n".join([f"[{i+1}] {c[1][:300]}" for i, c in enumerate(head)])
    prompt = LISTWISE_PROMPT.format(n=rerank_n, query=query, passages=passages)
    out = call_llm(prompt, api_key, max_tokens=200)
    order = parse_listwise_output(out, rerank_n)
    reranked_head = [head[i] for i in order]
    full = reranked_head + tail
    return full[:top_k]


# ---------- pairwise rerank (一两两比较, 选 top_k via tournament) ----------
PAIRWISE_PROMPT = """Which of the two passages is more relevant to the query? Answer ONLY "A" or "B".

Query: {query}

Passage A: {a}

Passage B: {b}

More relevant:"""


def pairwise_compare(query, doc_a_text, doc_b_text, api_key):
    """返回 +1 if A wins, -1 if B wins, 0 if tie/error."""
    prompt = PAIRWISE_PROMPT.format(query=query, a=doc_a_text[:300], b=doc_b_text[:300])
    out = call_llm(prompt, api_key, max_tokens=10, temperature=0.0)
    if not out:
        return 0
    out = out.strip().upper()
    if out.startswith("A") or "ANSWER: A" in out or "A " in out[:5]:
        return 1
    if out.startswith("B") or "ANSWER: B" in out or "B " in out[:5]:
        return -1
    return 0


def pairwise_rerank(query, candidates, api_key, top_k=10, prerank_n=20):
    """简化 pairwise: 取前 prerank_n 个候选, 让 LLM bubble-sort 取 top_k.
    候选 cost = O(prerank_n * top_k) pairwise comparison.
    candidates: List[(doc_id, text, cos_score)]
    """
    if not candidates:
        return []
    pool = list(candidates[:prerank_n])
    selected = []
    while pool and len(selected) < top_k:
        # 选当前 pool 中最优 (锦标赛)
        best_idx = 0
        for i in range(1, len(pool)):
            r = pairwise_compare(query, pool[best_idx][1], pool[i][1], api_key)
            if r < 0:  # B (i) wins
                best_idx = i
        selected.append(pool.pop(best_idx))
    selected.extend(pool[:top_k - len(selected)])
    return selected[:top_k]


# ---------- setwise rerank (improved RankGPT-like with bigger window) ----------
def setwise_rerank(query, candidates, api_key, top_k=10, window_size=10, step=5):
    """Sliding window from bottom to top. RankGPT 经典做法。
    candidates 应已按 cos_score 降序，返回 reranked top_k。
    """
    n = len(candidates)
    if n == 0:
        return []
    arr = list(candidates[:max(top_k * 2, 30)])  # 取前 30 个做 sliding rerank
    n = len(arr)
    end = n
    while end > 0:
        start = max(0, end - window_size)
        window = arr[start:end]
        if len(window) < 2:
            break
        passages = "\n".join([f"[{i+1}] {c[1][:300]}" for i, c in enumerate(window)])
        prompt = LISTWISE_PROMPT.format(n=len(window), query=query, passages=passages)
        out = call_llm(prompt, api_key, max_tokens=150)
        order = parse_listwise_output(out, len(window))
        reordered = [window[i] for i in order]
        arr[start:end] = reordered
        end -= step
    return arr[:top_k]


# ---------- main per dataset ----------
def run_dataset(dataset, methods, args, api_key):
    data_dir = Path(args.beir_dir) / dataset
    print(f"\n=== {dataset.upper()} LLM rerank ({','.join(methods)}) ===")

    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels.tsv")

    corpus_text = {d["_id"]: ((d.get("title", "") + ". " + d.get("text", "")).strip(":. ")) for d in corpus}
    query_text = {q["_id"]: q["text"] for q in queries}

    # 加载向量算 cosine top-100
    cv = load_jsonl(data_dir / "corpus_vectors.jsonl")
    qv = load_jsonl(data_dir / "query_vectors.jsonl")
    if not cv or not qv:
        print(f"  [{dataset}] missing corpus/query vectors, skip")
        return None
    doc_ids = [o["_id"] for o in cv]
    doc_mat = np.array([o["vector"] for o in cv], dtype=np.float32)
    query_vec = {o["_id"]: np.array(o["vector"], dtype=np.float32) for o in qv}

    qids = [q for q in qrels if q in query_vec]
    if args.max_queries > 0:
        qids = qids[: args.max_queries]
    print(f"  {len(qids)} queries, {len(doc_ids)} docs in corpus")

    # 评测每个 method
    out = {"dataset": dataset, "n_queries": len(qids), "candidates": args.candidates,
           "top_k": args.top_k, "methods": {}}

    # cosine baseline (所有 method 共用，记一次)
    cos_ndcgs = []
    cos_topn = {}
    print(f"  Running cosine top-{args.candidates} for all queries...")
    for qid in tqdm(qids, desc=f"  cosine[{dataset}]"):
        qv_arr = query_vec[qid]
        topn = cosine_topk(qv_arr, doc_mat, doc_ids, k=args.candidates)
        cos_topn[qid] = topn
        cos_top10 = [d for d, _ in topn[: args.top_k]]
        cos_ndcgs.append(compute_ndcg(cos_top10, qrels[qid], args.top_k))
    out["cosine_ndcg10_mean"] = float(np.mean(cos_ndcgs)) if cos_ndcgs else None

    for method in methods:
        print(f"\n  ## method = {method}")
        ndcg_list = []
        per_q_path = Path(args.output_dir) / f"llm_rerank_{dataset}_{method}.jsonl"
        per_q_path.parent.mkdir(parents=True, exist_ok=True)
        per_q_stream = open(per_q_path, "w", encoding="utf-8")

        for qid in tqdm(qids, desc=f"  {method}[{dataset}]"):
            qv_arr = query_vec[qid]
            qtext = query_text.get(qid, "")
            topn = cos_topn[qid]
            cands = [(d, corpus_text.get(d, "")[:1024], s) for d, s in topn]

            try:
                if method == "listwise":
                    reranked = listwise_rerank(qtext, cands, api_key, top_k=args.top_k, window_size=20)
                elif method == "pairwise":
                    reranked = pairwise_rerank(qtext, cands, api_key, top_k=args.top_k, prerank_n=args.pairwise_prerank)
                elif method == "setwise":
                    reranked = setwise_rerank(qtext, cands, api_key, top_k=args.top_k,
                                              window_size=args.setwise_window, step=args.setwise_step)
                else:
                    raise ValueError(f"unknown method: {method}")
                top10_ids = [c[0] for c in reranked[: args.top_k]]
                ndcg = compute_ndcg(top10_ids, qrels[qid], args.top_k)
                ndcg_list.append(ndcg)
                per_q_stream.write(json.dumps({"qid": qid, "top10": top10_ids, "ndcg": ndcg}) + "\n")
                per_q_stream.flush()
            except Exception as e:
                print(f"  [{method}] qid={qid} error: {e}", file=sys.stderr)

        per_q_stream.close()
        mean_ndcg = float(np.mean(ndcg_list)) if ndcg_list else None
        out["methods"][method] = {
            "ndcg10_mean": mean_ndcg,
            "n_completed": len(ndcg_list),
            "per_query_jsonl": str(per_q_path),
        }
        print(f"  {method} NDCG@10: {mean_ndcg:.4f} (n={len(ndcg_list)})")

    return out


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="nfcorpus,scifact",
                   help="comma-separated dataset names")
    p.add_argument("--methods", default="listwise,pairwise",
                   help="comma-separated methods (listwise|pairwise|setwise)")
    p.add_argument("--max_queries", type=int, default=0,
                   help="0 = all queries; >0 = first N (for quick test)")
    p.add_argument("--candidates", type=int, default=100,
                   help="cosine top-N candidates feed to LLM")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--pairwise_prerank", type=int, default=20,
                   help="pairwise: only run on top-N cos candidates (太多太慢)")
    p.add_argument("--setwise_window", type=int, default=10)
    p.add_argument("--setwise_step", type=int, default=5)
    p.add_argument("--beir_dir", default=DEFAULT_BEIR_DIR)
    p.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: no API key (LOCAL_EMBED_KEY / SILICONFLOW_API_KEY)", file=sys.stderr)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    print(f"datasets={datasets} methods={methods} max_q={args.max_queries} candidates={args.candidates}")
    print(f"LLM model: {LLM_MODEL} via {API_BASE}")

    all_results = {"args": vars(args), "results": {}}
    final_path = Path(args.output_dir) / "llm_rerank_results.json"

    for ds in datasets:
        try:
            r = run_dataset(ds, methods, args, api_key)
            if r:
                all_results["results"][ds] = r
                with open(final_path, "w") as f:
                    json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"[{ds}] FATAL: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            all_results["results"][ds] = {"error": str(e)}
            with open(final_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\n=== DONE === final: {final_path}")


if __name__ == "__main__":
    main()
