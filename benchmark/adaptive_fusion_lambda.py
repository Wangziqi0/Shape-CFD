#!/usr/bin/env python3
"""
adaptive_fusion_lambda.py — Reviewer 2/3 真正自适应 λ 实现（不再 placeholder）

输入:
  - fusion_ablation_<dataset>_perquery.jsonl  (NFCorpus / ArguAna / SciFact)
    每行含: cosine NDCG, token_2stage NDCG, lap_T{1..20} NDCG, fusion_T5_lam{0.0..1.0} NDCG,
            kendall_tau, spearman_rho
  - queries.jsonl per dataset (查 query length)

策略对比 (test split 上汇报 NDCG@10):
  S0. Fixed λ = 0.3            (论文 v11 默认)
  S1. Per-corpus λ*            (在 train half 选每 corpus 的 best 固定 λ)
  S2. Per-query learned λ*     (linear regressor 输入 per-query 特征预测 λ)
  S3. Oracle λ*                (每 query 选最优 λ — 仅作上界 disclose)
  S4. Token only (λ=0)
  S5. Lap only   (λ=1)

per-query feature:
  - kendall_tau (between Token vs Lap T=5)
  - spearman_rho
  - cosine_ndcg                (检索难度 proxy)
  - query_length               (查询 token 数 from queries.jsonl)

train/test split: 每 corpus 内 50/50 random，seed=42，重复 5 次取均值。
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict


# ---------- IO ----------
def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def load_query_lengths(queries_path):
    """从 queries.jsonl 读 query length（按空格 split 简单 token 数）。"""
    qlen = {}
    if not os.path.exists(queries_path):
        return qlen
    for o in load_jsonl(queries_path):
        qid = o.get("_id") or o.get("id") or o.get("qid")
        text = o.get("text", "")
        if qid:
            qlen[str(qid)] = len(text.split())
    return qlen


# ---------- per-query feature 抽取 ----------
LAMBDA_GRID = [round(x * 0.1, 1) for x in range(11)]


def extract_features_and_labels(perquery_records, qlens, dataset_name):
    """
    返回:
      X: (N, F) feature matrix
      y_oracle: (N,) 每 query 的 oracle 最优 lambda
      ndcg_table: (N, 11) 每 query 在 11 档 λ 下的 NDCG（供策略 evaluate）
      ndcg_cosine: (N,)
      ndcg_token: (N,)
      ndcg_lap_T5: (N,)
      qids: list
    """
    X_list, ndcg_rows, oracle_lam, qids = [], [], [], []
    cos_list, tok_list, lap_list = [], [], []
    for rec in perquery_records:
        qid = rec.get("qid")
        n = rec.get("ndcg", {})
        # 必要字段都在
        if "token_2stage" not in n or "lap_T5" not in n:
            continue
        ndcg_per_lam = []
        for lam in LAMBDA_GRID:
            key = f"fusion_T5_lam{lam:.1f}"
            ndcg_per_lam.append(n.get(key, np.nan))
        ndcg_per_lam = np.asarray(ndcg_per_lam, dtype=np.float64)
        if np.any(np.isnan(ndcg_per_lam)):
            continue
        # oracle lam: argmax (ties → 选 0.3 最近的，但简单 argmax 就行)
        best_idx = int(np.argmax(ndcg_per_lam))
        oracle_lam.append(LAMBDA_GRID[best_idx])

        feats = [
            float(rec.get("kendall_tau", 0.0) or 0.0),
            float(rec.get("spearman_rho", 0.0) or 0.0),
            float(n.get("cosine", 0.0) or 0.0),
            float(qlens.get(str(qid), 0)),
        ]
        X_list.append(feats)
        ndcg_rows.append(ndcg_per_lam)
        qids.append(qid)
        cos_list.append(n.get("cosine", 0.0))
        tok_list.append(n.get("token_2stage", 0.0))
        lap_list.append(n.get("lap_T5", 0.0))

    return (
        np.asarray(X_list, dtype=np.float64),
        np.asarray(oracle_lam, dtype=np.float64),
        np.asarray(ndcg_rows, dtype=np.float64),
        np.asarray(cos_list, dtype=np.float64),
        np.asarray(tok_list, dtype=np.float64),
        np.asarray(lap_list, dtype=np.float64),
        qids,
    )


# ---------- 5 个策略 ----------
def eval_strategy_fixed(ndcg_table, lam_value):
    """Fixed λ。返回每 query 的 NDCG。"""
    idx = LAMBDA_GRID.index(round(lam_value, 1))
    return ndcg_table[:, idx]


def eval_strategy_per_corpus_lambda_star(ndcg_table_train, ndcg_table_test):
    """在 train 半选 best 固定 λ，再在 test 半 apply。"""
    train_means = ndcg_table_train.mean(axis=0)
    best_idx = int(np.argmax(train_means))
    best_lam = LAMBDA_GRID[best_idx]
    return ndcg_table_test[:, best_idx], best_lam


def eval_strategy_learned_lambda(X_train, ndcg_train, X_test, ndcg_test):
    """
    Linear regression: λ̂ = w · feat + b. target = oracle λ★ on train half.
    test 上预测 λ̂_q，clip 到 [0,1]，snap 到最近的 LAMBDA_GRID，取对应列 NDCG。
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    oracle_train = LAMBDA_GRID_ARR[np.argmax(ndcg_train, axis=1)]
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)
    reg = Ridge(alpha=1.0).fit(Xs_tr, oracle_train)
    pred = np.clip(reg.predict(Xs_te), 0.0, 1.0)
    snapped_idx = np.array([int(np.argmin(np.abs(LAMBDA_GRID_ARR - p))) for p in pred])
    sel_ndcg = ndcg_test[np.arange(len(snapped_idx)), snapped_idx]
    return sel_ndcg, reg.coef_, float(reg.intercept_), pred


def eval_strategy_oracle(ndcg_table):
    """每 query 选 best λ — 上界 only disclose，不当主结论。"""
    return ndcg_table.max(axis=1)


# ---------- main ----------
LAMBDA_GRID_ARR = np.asarray(LAMBDA_GRID, dtype=np.float64)


def run_corpus_5fold(perquery_path, queries_path, dataset_name, n_seeds=5, train_frac=0.5):
    perq = load_jsonl(perquery_path)
    qlens = load_query_lengths(queries_path)
    X, y_oracle, ndcg_tab, cos_arr, tok_arr, lap_arr, qids = extract_features_and_labels(perq, qlens, dataset_name)
    n = len(qids)
    if n == 0:
        return None

    seeds = list(range(n_seeds))
    results = defaultdict(list)
    chosen_per_corpus_lams = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_tr = int(n * train_frac)
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
        X_tr, X_te = X[tr_idx], X[te_idx]
        ndcg_tr, ndcg_te = ndcg_tab[tr_idx], ndcg_tab[te_idx]

        # S0. Fixed λ=0.3
        s0 = eval_strategy_fixed(ndcg_te, 0.3).mean()
        results["fixed_lam_0.3"].append(s0)

        # S1. Per-corpus λ★ (train→test)
        s1, best_lam = eval_strategy_per_corpus_lambda_star(ndcg_tr, ndcg_te)
        results["per_corpus_lam_star"].append(s1.mean())
        chosen_per_corpus_lams.append(best_lam)

        # S2. Per-query learned λ★
        s2, coefs, intercept, pred = eval_strategy_learned_lambda(X_tr, ndcg_tr, X_te, ndcg_te)
        results["per_query_learned_lam"].append(s2.mean())

        # S3. Oracle λ★ (test 半上的上界)
        s3 = eval_strategy_oracle(ndcg_te).mean()
        results["oracle_upper_bound"].append(s3)

        # S4. Token only
        results["token_only"].append(eval_strategy_fixed(ndcg_te, 0.0).mean())
        # S5. Lap only
        results["lap_only"].append(eval_strategy_fixed(ndcg_te, 1.0).mean())

    summary = {
        "dataset": dataset_name,
        "n_queries": int(n),
        "n_seeds": n_seeds,
        "train_frac": train_frac,
        "feature_names": ["kendall_tau", "spearman_rho", "cosine_ndcg", "query_length"],
        "strategies": {
            k: {
                "ndcg_mean": float(np.mean(v)),
                "ndcg_std": float(np.std(v)),
                "per_seed": [float(x) for x in v],
            }
            for k, v in results.items()
        },
        "per_corpus_lam_star_chosen_per_seed": chosen_per_corpus_lams,
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results")
    p.add_argument("--beir_dir", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data")
    p.add_argument("--datasets", default="nfcorpus,arguana,scifact,scidocs,fiqa")
    p.add_argument("--out", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/adaptive_lambda_results.json")
    p.add_argument("--n_seeds", type=int, default=5)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_res = {"datasets": {}}

    for ds in datasets:
        perq = Path(args.results_dir) / f"fusion_ablation_{ds}_perquery.jsonl"
        qpath = Path(args.beir_dir) / ds / "queries.jsonl"
        if not perq.exists():
            print(f"[{ds}] SKIP: missing {perq}")
            continue
        print(f"[{ds}] running 5-seed split eval...")
        summary = run_corpus_5fold(str(perq), str(qpath), ds, n_seeds=args.n_seeds)
        if summary:
            all_res["datasets"][ds] = summary
            print(f"  n={summary['n_queries']}")
            for sname, sres in summary["strategies"].items():
                print(f"    {sname:30s}  NDCG={sres['ndcg_mean']:.4f} ± {sres['ndcg_std']:.4f}")

    # 跨 corpus 平均
    if all_res["datasets"]:
        agg = defaultdict(list)
        for ds, summary in all_res["datasets"].items():
            for sname, sres in summary["strategies"].items():
                agg[sname].append(sres["ndcg_mean"])
        all_res["macro_avg_across_corpora"] = {
            k: {"mean": float(np.mean(v)), "datasets_used": list(all_res["datasets"].keys())}
            for k, v in agg.items()
        }
        print("\n=== Macro avg across corpora ===")
        for k, v in all_res["macro_avg_across_corpora"].items():
            print(f"  {k:30s}  {v['mean']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
