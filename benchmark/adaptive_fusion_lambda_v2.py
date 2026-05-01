#!/usr/bin/env python3
"""
adaptive_fusion_lambda_v2.py — Senior Reviewer + multi-venue agent W3 fix
========================================================================

V1 (Ridge, 4 features) only got macro +3.18% with 13.5% oracle gap.
V2 adds:
  - GradientBoostingRegressor (non-linear)
  - 11 derived features instead of 4:
    * (1) Kendall τ between Token vs Lap T=5
    * (2) Spearman ρ
    * (3) cosine NDCG@10
    * (4) query length
    * (5) token_2stage NDCG (signal quality)
    * (6) lap_T5 NDCG
    * (7) token_2stage - cosine (token improvement margin)
    * (8) lap_T5 - cosine (lap improvement margin)
    * (9) lap_T1 - lap_T20 (T-stability range; saturation signal)
    * (10) abs(token_2stage - lap_T5) (signal disagreement)
    * (11) max(lap_T1..T20) - min(lap_T1..T20) (lap T-sweep variance)

Goal: push oracle gap from 13.5% (Ridge) toward < 8%.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict


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
    qlen = {}
    if not Path(queries_path).exists():
        return qlen
    for o in load_jsonl(queries_path):
        qid = o.get("_id") or o.get("id") or o.get("qid")
        text = o.get("text", "")
        if qid:
            qlen[str(qid)] = len(text.split())
    return qlen


LAMBDA_GRID = [round(x * 0.1, 1) for x in range(11)]
LAMBDA_GRID_ARR = np.asarray(LAMBDA_GRID, dtype=np.float64)


def extract_features_v2(perquery_records, qlens):
    """11 derived features."""
    X_list, ndcg_rows, qids = [], [], []
    for rec in perquery_records:
        qid = rec.get("qid")
        n = rec.get("ndcg", {})
        if "token_2stage" not in n or "lap_T5" not in n:
            continue
        ndcg_per_lam = []
        for lam in LAMBDA_GRID:
            key = f"fusion_T5_lam{lam:.1f}"
            ndcg_per_lam.append(n.get(key, np.nan))
        ndcg_per_lam = np.asarray(ndcg_per_lam, dtype=np.float64)
        if np.any(np.isnan(ndcg_per_lam)):
            continue

        # 11 features
        cos = float(n.get("cosine", 0.0) or 0.0)
        tok = float(n.get("token_2stage", 0.0) or 0.0)
        lap5 = float(n.get("lap_T5", 0.0) or 0.0)
        lap_T_vals = []
        for T in [1, 3, 5, 7, 10, 15, 20]:
            v = n.get(f"lap_T{T}")
            if v is not None:
                lap_T_vals.append(float(v))
        if not lap_T_vals:
            lap_T_vals = [lap5]
        feats = [
            float(rec.get("kendall_tau", 0.0) or 0.0),
            float(rec.get("spearman_rho", 0.0) or 0.0),
            cos,
            float(qlens.get(str(qid), 0)),
            tok,
            lap5,
            tok - cos,
            lap5 - cos,
            lap_T_vals[0] - lap_T_vals[-1],
            abs(tok - lap5),
            max(lap_T_vals) - min(lap_T_vals),
        ]
        X_list.append(feats)
        ndcg_rows.append(ndcg_per_lam)
        qids.append(qid)

    return (
        np.asarray(X_list, dtype=np.float64),
        np.asarray(ndcg_rows, dtype=np.float64),
        qids,
    )


def eval_fixed(ndcg_table, lam_value):
    idx = LAMBDA_GRID.index(round(lam_value, 1))
    return ndcg_table[:, idx]


def eval_per_corpus(ndcg_train, ndcg_test):
    train_means = ndcg_train.mean(axis=0)
    best_idx = int(np.argmax(train_means))
    return ndcg_test[:, best_idx], LAMBDA_GRID[best_idx]


def eval_learned_ridge(X_train, ndcg_train, X_test, ndcg_test):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    oracle_train = LAMBDA_GRID_ARR[np.argmax(ndcg_train, axis=1)]
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)
    reg = Ridge(alpha=1.0).fit(Xs_tr, oracle_train)
    pred = np.clip(reg.predict(Xs_te), 0.0, 1.0)
    snapped = np.array([int(np.argmin(np.abs(LAMBDA_GRID_ARR - p))) for p in pred])
    return ndcg_test[np.arange(len(snapped)), snapped]


def eval_learned_gbm(X_train, ndcg_train, X_test, ndcg_test, n_est=200, max_depth=3):
    """GBM regression on oracle λ★."""
    from sklearn.ensemble import GradientBoostingRegressor
    oracle_train = LAMBDA_GRID_ARR[np.argmax(ndcg_train, axis=1)]
    reg = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, random_state=42,
    ).fit(X_train, oracle_train)
    pred = np.clip(reg.predict(X_test), 0.0, 1.0)
    snapped = np.array([int(np.argmin(np.abs(LAMBDA_GRID_ARR - p))) for p in pred])
    return ndcg_test[np.arange(len(snapped)), snapped]


def eval_oracle(ndcg_test):
    return ndcg_test.max(axis=1)


def run_corpus(perquery_path, queries_path, dataset, n_seeds=5, train_frac=0.5):
    perq = load_jsonl(perquery_path)
    qlens = load_query_lengths(queries_path)
    X, ndcg_tab, qids = extract_features_v2(perq, qlens)
    n = len(qids)
    if n == 0:
        return None

    results = defaultdict(list)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_tr = int(n * train_frac)
        tr, te = idx[:n_tr], idx[n_tr:]

        ndcg_tr, ndcg_te = ndcg_tab[tr], ndcg_tab[te]
        X_tr, X_te = X[tr], X[te]

        results["fixed_lam_0.3"].append(eval_fixed(ndcg_te, 0.3).mean())
        s1, _ = eval_per_corpus(ndcg_tr, ndcg_te)
        results["per_corpus_lam_star"].append(s1.mean())
        results["learned_ridge_4feat"].append(
            eval_learned_ridge(X_tr[:, :4], ndcg_tr, X_te[:, :4], ndcg_te).mean()
        )
        results["learned_ridge_11feat"].append(
            eval_learned_ridge(X_tr, ndcg_tr, X_te, ndcg_te).mean()
        )
        results["learned_gbm_11feat"].append(
            eval_learned_gbm(X_tr, ndcg_tr, X_te, ndcg_te).mean()
        )
        results["oracle_upper"].append(eval_oracle(ndcg_te).mean())

    return {
        "dataset": dataset,
        "n_queries": int(n),
        "n_seeds": n_seeds,
        "n_features": int(X.shape[1]),
        "strategies": {k: {"ndcg_mean": float(np.mean(v)),
                           "ndcg_std": float(np.std(v))}
                       for k, v in results.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results")
    p.add_argument("--beir_dir", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data")
    p.add_argument("--datasets", default="nfcorpus,arguana,scifact,scidocs,fiqa")
    p.add_argument("--out", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/adaptive_lambda_v2_gbm_results.json")
    p.add_argument("--n_seeds", type=int, default=5)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    all_res = {"datasets": {}}

    for ds in datasets:
        perq = Path(args.results_dir) / f"fusion_ablation_{ds}_perquery.jsonl"
        qpath = Path(args.beir_dir) / ds / "queries.jsonl"
        if not perq.exists():
            print(f"[{ds}] SKIP: missing {perq}")
            continue
        print(f"[{ds}] running 5-seed split eval (V2 with GBM + 11 features)...")
        s = run_corpus(str(perq), str(qpath), ds, n_seeds=args.n_seeds)
        if s:
            all_res["datasets"][ds] = s
            print(f"  n={s['n_queries']}, features={s['n_features']}")
            for sn, sr in s["strategies"].items():
                print(f"    {sn:30s}  NDCG={sr['ndcg_mean']:.4f} ± {sr['ndcg_std']:.4f}")

    if all_res["datasets"]:
        agg = defaultdict(list)
        for ds, s in all_res["datasets"].items():
            for sn, sr in s["strategies"].items():
                agg[sn].append(sr["ndcg_mean"])
        all_res["macro_avg"] = {k: float(np.mean(v)) for k, v in agg.items()}
        print("\n=== Macro avg across corpora (V2) ===")
        for k, v in all_res["macro_avg"].items():
            print(f"  {k:30s}  {v:.4f}")

        if "oracle_upper" in all_res["macro_avg"] and "learned_gbm_11feat" in all_res["macro_avg"]:
            oracle = all_res["macro_avg"]["oracle_upper"]
            gbm = all_res["macro_avg"]["learned_gbm_11feat"]
            ridge4 = all_res["macro_avg"]["learned_ridge_4feat"]
            fixed = all_res["macro_avg"]["fixed_lam_0.3"]
            gap_gbm_pct = (oracle - gbm) / fixed * 100
            gap_ridge_pct = (oracle - ridge4) / fixed * 100
            print(f"\n  oracle gap (relative-to-fixed):")
            print(f"    Ridge 4-feat (V1):       {gap_ridge_pct:.2f}%")
            print(f"    GBM 11-feat (V2):        {gap_gbm_pct:.2f}%")
            print(f"    improvement:             {gap_ridge_pct - gap_gbm_pct:+.2f} pp")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
