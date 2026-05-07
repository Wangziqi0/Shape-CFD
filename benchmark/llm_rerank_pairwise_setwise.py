"""
llm_rerank_pairwise_setwise.py — Open-source LLM pairwise & setwise reranker.

Addresses Reviewer 3 weakness W1(b) of the prior IPM review: lack of
listwise/pairwise/setwise reranking baselines based on open-source LLMs.

Modes:
    --mode pairwise  : per-pair preference judgment (q, d_a, d_b) -> ∈ {a, b}.
                       Aggregates pair wins into a Bradley-Terry score per doc.
    --mode setwise   : per-set top-k judgment (q, [d1..d4]) -> top-2 indices.
                       Aggregates set memberships into a per-doc score.

Both modes use Qwen3-8B-Q4_K_M served by llama-server on 9070XT (port 8082)
with `/no_think` prompting to disable chain-of-thought; output is constrained
to a deterministic format and parsed strictly. Runs over BGE-M3 v2 (audited)
top-100 candidates per query, then reranks within that pool.

Comparison axes:
    Listwise (RankGPT-style, existing) : 1 LLM call per query, sees all 20.
    Pairwise (this script)             : C(K,2) LLM calls per query.
    Setwise (this script)              : K/4 LLM calls per query (4-doc sets).

Costs (estimated, Qwen3-8B-Q4 on 9070XT):
    Pairwise K=20, all pairs   : 190 calls × ~0.5s = ~95s/query
    Pairwise K=20, sample-200  : 200 calls × ~0.5s = ~100s/query (matched pool)
    Setwise K=20, sliding 4-grp: 5 calls × ~1.5s = ~8s/query
    Listwise (existing)        : 1 call × ~1.5s = ~1.5s/query

Usage:
    python benchmark/llm_rerank_pairwise_setwise.py \
        --mode pairwise \
        --corpus nfcorpus \
        --top-k 20 \
        --pair-budget 200 \
        --first-stage benchmark/data/results/bge_m3_v2_topk_nfcorpus.json \
        --qrels benchmark/data/qrels/nfcorpus.tsv \
        --llama-url http://192.168.31.22:8082 \
        --out benchmark/data/results/pairwise_rerank_nfcorpus_qwen3_8b.json

    python benchmark/llm_rerank_pairwise_setwise.py \
        --mode setwise \
        --corpus nfcorpus \
        --top-k 20 --set-size 4 \
        ...

Output JSON schema:
    {
      "config": {...},
      "per_query": [
          {"qid": ..., "rerank_topk": [...], "ndcg10": ...,
           "n_llm_calls": ..., "n_parse_failures": ...},
          ...
      ],
      "macro": {
          "n_queries": ..., "ndcg_mean": ..., "ndcg_std": ...,
          "n_llm_calls_total": ..., "parse_failure_rate": ...,
          "wall_time_sec": ...
      }
    }
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    import numpy as np
except ImportError:
    print("error: pip install requests numpy", file=sys.stderr)
    sys.exit(1)


# ---------- prompt templates ----------

PAIRWISE_PROMPT = """<|im_start|>user
You are a relevance judge. Given a query and two passages, decide which passage is more relevant to the query. Output exactly one character: A or B.

Query: {query}

Passage A: {doc_a}

Passage B: {doc_b}

Which is more relevant? Output only one character (A or B).<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


SETWISE_PROMPT = """<|im_start|>user
You are a relevance judge. Given a query and {n} candidate passages, output the indices of the TOP-2 most relevant passages. Output the two indices comma-separated, nothing else.

Query: {query}

{passages_block}

Top-2 indices (comma-separated, e.g. "1,3").<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


# ---------- LLM call ----------

def call_llama(prompt, llama_url, max_tokens=8, temperature=0.0):
    """Call llama-server /completion endpoint, return raw response text."""
    r = requests.post(
        f"{llama_url}/completion",
        json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "</s>"],
            "cache_prompt": True,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("content", "").strip()


def parse_pairwise(response, default="A"):
    """Strict parse: response should start with A or B; else return default."""
    response = response.strip().upper()
    if response and response[0] in {"A", "B"}:
        return response[0]
    m = re.search(r"\b([AB])\b", response)
    return m.group(1) if m else default


def parse_setwise(response, n_candidates):
    """Strict parse: extract 2 distinct integers in [1, n_candidates]."""
    nums = [int(m) for m in re.findall(r"\b(\d+)\b", response)]
    nums = [n for n in nums if 1 <= n <= n_candidates]
    seen = []
    for n in nums:
        if n not in seen:
            seen.append(n)
        if len(seen) == 2:
            break
    return seen if len(seen) == 2 else None


# ---------- pairwise rerank ----------

def rerank_pairwise(query, doc_texts, llama_url, pair_budget, rng):
    """
    Run pairwise comparisons; aggregate via Bradley-Terry-style win counts.
    Returns: (rerank_indices, n_llm_calls, n_parse_failures)
    """
    K = len(doc_texts)
    all_pairs = list(combinations(range(K), 2))
    if pair_budget is not None and pair_budget < len(all_pairs):
        chosen_idx = rng.choice(len(all_pairs), size=pair_budget, replace=False)
        pairs = [all_pairs[i] for i in chosen_idx]
    else:
        pairs = all_pairs

    wins = np.zeros(K, dtype=float)
    n_calls = 0
    n_fail = 0

    def _do_pair(ij):
        i, j = ij
        prompt = PAIRWISE_PROMPT.format(
            query=query[:512],
            doc_a=doc_texts[i][:1024],
            doc_b=doc_texts[j][:1024],
        )
        try:
            response = call_llama(prompt, llama_url)
            choice = parse_pairwise(response)
            return (i, j, choice, True)
        except Exception:
            return (i, j, None, False)

    # Workers passed via global hack (rerank_pairwise sig不变, use _PARALLEL_WORKERS)
    workers = globals().get('_PARALLEL_WORKERS', 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed(ex.submit(_do_pair, ij) for ij in pairs):
                i, j, choice, ok = fut.result()
                if ok:
                    n_calls += 1
                    if choice == "A":
                        wins[i] += 1.0
                    elif choice == "B":
                        wins[j] += 1.0
                    else:
                        n_fail += 1
                        wins[i] += 0.5
                        wins[j] += 0.5
                else:
                    n_fail += 1
                    wins[i] += 0.5
                    wins[j] += 0.5
    else:
        for ij in pairs:
            i, j, choice, ok = _do_pair(ij)
            if ok:
                n_calls += 1
                if choice == "A":
                    wins[i] += 1.0
                elif choice == "B":
                    wins[j] += 1.0
                else:
                    n_fail += 1
                    wins[i] += 0.5
                    wins[j] += 0.5
            else:
                n_fail += 1
                wins[i] += 0.5
                wins[j] += 0.5

    rerank_idx = np.argsort(-wins).tolist()  # descending
    return rerank_idx, n_calls, n_fail


# ---------- setwise rerank (sliding window of set_size=4) ----------

def rerank_setwise(query, doc_texts, llama_url, set_size=4):
    """
    Sliding-window setwise rerank: for each consecutive set of `set_size` docs,
    LLM picks top-2; promotion-aggregate.
    Returns: (rerank_indices, n_llm_calls, n_parse_failures)
    """
    K = len(doc_texts)
    promotion_count = np.zeros(K, dtype=float)
    n_calls = 0
    n_fail = 0

    # Pre-build all window prompts
    windows = []
    pos = 0
    while pos + set_size <= K:
        idxs = list(range(pos, pos + set_size))
        passages_block = "\n\n".join(
            f"Passage {k + 1}: {doc_texts[idxs[k]][:512]}" for k in range(set_size)
        )
        prompt = SETWISE_PROMPT.format(
            n=set_size, query=query[:512], passages_block=passages_block,
        )
        windows.append((idxs, prompt))
        pos += 2  # 50% overlap sliding window

    def _do_window(w):
        idxs, prompt = w
        try:
            response = call_llama(prompt, llama_url, max_tokens=12)
            top2 = parse_setwise(response, set_size)
            return (idxs, top2, True)
        except Exception:
            return (idxs, None, False)

    workers = globals().get('_PARALLEL_WORKERS', 1)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed(ex.submit(_do_window, w) for w in windows):
                idxs, top2, ok = fut.result()
                if ok:
                    n_calls += 1
                    if top2 is None:
                        n_fail += 1
                        promotion_count[idxs[0]] += 0.5
                        promotion_count[idxs[1]] += 0.5
                    else:
                        promotion_count[idxs[top2[0] - 1]] += 1.0
                        promotion_count[idxs[top2[1] - 1]] += 0.5
                else:
                    n_fail += 1
                    promotion_count[idxs[0]] += 0.5
                    promotion_count[idxs[1]] += 0.5
    else:
        for w in windows:
            idxs, top2, ok = _do_window(w)
            if ok:
                n_calls += 1
                if top2 is None:
                    n_fail += 1
                    promotion_count[idxs[0]] += 0.5
                    promotion_count[idxs[1]] += 0.5
                else:
                    promotion_count[idxs[top2[0] - 1]] += 1.0
                    promotion_count[idxs[top2[1] - 1]] += 0.5
            else:
                n_fail += 1
                promotion_count[idxs[0]] += 0.5
                promotion_count[idxs[1]] += 0.5

    # combine with first-stage rank as tie-breaker
    rerank_score = promotion_count - 0.001 * np.arange(K)
    rerank_idx = np.argsort(-rerank_score).tolist()
    return rerank_idx, n_calls, n_fail


# ---------- NDCG ----------

def compute_ndcg10(ranked_doc_ids, qrels_for_query):
    """qrels_for_query: dict {doc_id: relevance_grade}. Binary or graded."""
    rel = np.array([qrels_for_query.get(d, 0) for d in ranked_doc_ids[:10]], dtype=float)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, 12))
    dcg = float(np.sum(rel * discounts[: len(rel)]))
    ideal_rel = np.sort(np.array(list(qrels_for_query.values()), dtype=float))[::-1]
    idcg = float(np.sum(ideal_rel[:10] * discounts[: min(10, len(ideal_rel))]))
    return dcg / idcg if idcg > 0 else 0.0


# ---------- main loop ----------

def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["pairwise", "setwise"], required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--top-k", type=int, default=20, help="rerank top-K per query")
    p.add_argument("--pair-budget", type=int, default=200,
                   help="pairwise: max LLM calls per query (None = all C(K,2))")
    p.add_argument("--set-size", type=int, default=4, help="setwise: window size")
    p.add_argument("--first-stage", type=Path, required=True,
                   help="JSON: {qid: [{doc_id, doc_text}, ...]} from BGE-M3 v2")
    p.add_argument("--qrels", type=Path, required=True,
                   help="TSV qrels: qid<TAB>0<TAB>doc_id<TAB>grade")
    p.add_argument("--llama-url", default="http://192.168.31.22:8082")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="ThreadPool size for parallel client requests; pair with llama-server --parallel N to multiply throughput")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_qrels(path):
    """3-col or 4-col qrels with optional header line."""
    qrels = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("	")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            if parts[-1] in ("score", "grade", "relevance"):
                continue
            try:
                grade = int(float(parts[-1]))
            except ValueError:
                continue
            if len(parts) >= 4:
                qid, did = parts[0], parts[2]
            else:
                qid, did = parts[0], parts[1]
            qrels.setdefault(qid, {})[did] = grade
    return qrels


def main():
    args = parse_args()
    global _PARALLEL_WORKERS
    _PARALLEL_WORKERS = args.workers
    rng = np.random.default_rng(args.seed)

    qrels_all = load_qrels(args.qrels)
    with open(args.first_stage, "r", encoding="utf-8") as fh:
        first_stage = json.load(fh)
    queries_data = first_stage if isinstance(first_stage, dict) else {q["qid"]: q for q in first_stage}

    qids = list(queries_data.keys())
    if args.max_queries is not None:
        qids = qids[: args.max_queries]

    per_query_results = []
    n_calls_total = 0
    n_fail_total = 0
    t_start = time.time()

    for qi, qid in enumerate(qids):
        qd = queries_data[qid]
        if isinstance(qd, dict):
            query_text = qd.get("query") or qd.get("query_text") or ""
            candidates = qd.get("candidates") or qd.get("docs") or []
        else:
            continue
        candidates = candidates[: args.top_k]
        doc_ids = [c["doc_id"] for c in candidates]
        doc_texts = [c.get("doc_text") or c.get("text") or "" for c in candidates]

        if args.mode == "pairwise":
            rerank_idx, n_calls, n_fail = rerank_pairwise(
                query_text, doc_texts, args.llama_url, args.pair_budget, rng,
            )
        else:
            rerank_idx, n_calls, n_fail = rerank_setwise(
                query_text, doc_texts, args.llama_url, args.set_size,
            )

        rerank_doc_ids = [doc_ids[i] for i in rerank_idx]
        ndcg = compute_ndcg10(rerank_doc_ids, qrels_all.get(qid, {}))

        per_query_results.append({
            "qid": qid,
            "rerank_topk": rerank_doc_ids[:10],
            "ndcg10": round(ndcg, 4),
            "n_llm_calls": n_calls,
            "n_parse_failures": n_fail,
        })
        n_calls_total += n_calls
        n_fail_total += n_fail

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t_start
            avg_ndcg = np.mean([r["ndcg10"] for r in per_query_results])
            print(f"[{qi+1}/{len(qids)}] avg_ndcg={avg_ndcg:.4f} calls={n_calls_total} "
                  f"fails={n_fail_total} elapsed={elapsed:.0f}s", file=sys.stderr)

    wall = time.time() - t_start
    ndcg_arr = np.array([r["ndcg10"] for r in per_query_results])
    out = {
        "config": vars(args) | {"first_stage": str(args.first_stage), "qrels": str(args.qrels), "out": str(args.out)},
        "per_query": per_query_results,
        "macro": {
            "n_queries": len(per_query_results),
            "ndcg_mean": float(ndcg_arr.mean()) if len(ndcg_arr) else 0.0,
            "ndcg_std": float(ndcg_arr.std(ddof=1)) if len(ndcg_arr) > 1 else 0.0,
            "n_llm_calls_total": n_calls_total,
            "parse_failure_rate": (n_fail_total / max(1, n_calls_total)),
            "wall_time_sec": wall,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Path objects in config -> to str
    out["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in out["config"].items()}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"DONE: {args.mode} {args.corpus} n={len(per_query_results)} "
          f"ndcg={out['macro']['ndcg_mean']:.4f}±{out['macro']['ndcg_std']:.4f} "
          f"wall={wall:.0f}s out={args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
