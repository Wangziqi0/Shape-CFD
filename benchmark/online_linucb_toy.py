#!/usr/bin/env python3
"""
online_linucb_toy.py — Online contextual bandit policy for adaptive fusion lambda
=================================================================================

Closes Reviewer 1 W1 partial gap: GBM-11 (offline supervised on oracle lambda*) is
not deployment-time online policy. This script implements a real online
contextual bandit (LinUCB) on NFCorpus per-query data, where:

  - context = 11 GBM features (kendall tau, spearman rho, cosine NDCG, query
    length, token NDCG, lap_T5 NDCG, margins, T-stability, etc.)
  - 11 arms = lambda in {0.0, 0.1, ..., 1.0}
  - reward = per-query NDCG@10 at the chosen lambda (from
    fusion_ablation_nfcorpus_perquery.jsonl, fusion_T5_lam{lambda})
  - sequential: query t observes (context_t, action_t, reward_t), updates
    LinUCB posterior, then selects action_{t+1}

Output:
  - cumulative regret curve (vs oracle per-query best lambda)
  - final macro NDCG@10 (vs GBM offline ref / fixed-lam-0.3 baselines)
  - convergence by 50/100/200-query running mean
  - lambda_mean over the run

NOTE: This is a proof-of-concept simplification of contextual bandit. Full
production policy with regret bound + theoretical analysis is deferred
to future work. Key contribution here: real online learning without oracle
label at deployment time.
"""

import json
import argparse
from pathlib import Path
import numpy as np


LAMBDA_GRID = [round(x * 0.1, 1) for x in range(11)]


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


def extract_features_and_rewards(perquery_records, qlens):
    """Return X (n x 11 features), R (n x 11 rewards / NDCG@lambda), qids."""
    X_list, R_list, qids = [], [], []
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
        R_list.append(ndcg_per_lam)
        qids.append(qid)

    return (
        np.asarray(X_list, dtype=np.float64),
        np.asarray(R_list, dtype=np.float64),
        qids,
    )


class LinUCB:
    """
    Disjoint LinUCB (Li et al. 2010, WWW). One linear model per arm.

    For arm a: A_a (d x d) = I, b_a (d) = 0
    Action selection: a_t = argmax_a [theta_a^T x_t + alpha * sqrt(x_t^T A_a^{-1} x_t)]
    Update: A_{a_t} += x_t x_t^T, b_{a_t} += r_t * x_t
    """

    def __init__(self, n_arms, d, alpha=1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A_inv = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def select(self, x, rng=None):
        ucbs = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta_a = self.A_inv[a] @ self.b[a]
            mu = float(theta_a @ x)
            sigma = float(np.sqrt(max(x @ self.A_inv[a] @ x, 0.0)))
            ucbs[a] = mu + self.alpha * sigma
        if rng is not None:
            max_val = ucbs.max()
            ties = np.where(ucbs >= max_val - 1e-12)[0]
            return int(rng.choice(ties))
        return int(np.argmax(ucbs))

    def update(self, a, x, r):
        Ax = self.A_inv[a] @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[a] = self.A_inv[a] - np.outer(Ax, Ax) / denom
        self.b[a] = self.b[a] + r * x


def run_linucb_online(X, R, alpha=1.0, seed=42, normalize_context=True):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_arms = R.shape[1]

    if normalize_context:
        mu_x = X.mean(axis=0)
        std_x = X.std(axis=0) + 1e-9
        X_norm = (X - mu_x) / std_x
    else:
        X_norm = X

    X_ext = np.concatenate([X_norm, np.ones((n, 1))], axis=1)
    d_ext = d + 1

    bandit = LinUCB(n_arms=n_arms, d=d_ext, alpha=alpha)

    chosen_arms = np.zeros(n, dtype=int)
    per_q_ndcg = np.zeros(n)
    oracle_ndcg = R.max(axis=1)
    instant_regret = np.zeros(n)

    perm = rng.permutation(n)

    for t, idx in enumerate(perm):
        x = X_ext[idx]
        a = bandit.select(x, rng=rng)
        r = R[idx, a]
        bandit.update(a, x, r)
        chosen_arms[t] = a
        per_q_ndcg[t] = r
        instant_regret[t] = oracle_ndcg[idx] - r

    cum_regret = np.cumsum(instant_regret)
    chosen_lams = np.array([LAMBDA_GRID[a] for a in chosen_arms])

    win = 50
    running_mean = np.array(
        [per_q_ndcg[max(0, t - win):t + 1].mean() for t in range(n)]
    )

    return {
        "per_q_ndcg": per_q_ndcg,
        "oracle_ndcg_perm": oracle_ndcg[perm],
        "chosen_arms": chosen_arms,
        "chosen_lams": chosen_lams,
        "cum_regret": cum_regret,
        "macro_ndcg": float(per_q_ndcg.mean()),
        "oracle_macro": float(oracle_ndcg.mean()),
        "lambda_mean": float(chosen_lams.mean()),
        "running_mean": running_mean,
        "perm": perm,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perquery", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/fusion_ablation_nfcorpus_perquery.jsonl")
    p.add_argument("--queries", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data/nfcorpus/queries.jsonl")
    p.add_argument("--out", default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/online_linucb_nfcorpus_results.json")
    p.add_argument("--seeds", type=str, default="42,7,13,21,99",
                   help="comma-separated seeds for shuffle / tie-break")
    args = p.parse_args()

    print(f"[load] {args.perquery}")
    perq = load_jsonl(args.perquery)
    print(f"  n_records = {len(perq)}")
    qlens = load_query_lengths(args.queries)
    print(f"  n_queries with length = {len(qlens)}")

    X, R, qids = extract_features_and_rewards(perq, qlens)
    n, d = X.shape
    n_arms = R.shape[1]
    print(f"  n_valid_queries = {n}, n_features = {d}, n_arms = {n_arms}")

    fixed_03_macro = float(R[:, LAMBDA_GRID.index(0.3)].mean())
    laplacian_only = float(R[:, LAMBDA_GRID.index(1.0)].mean())
    token_only = float(R[:, LAMBDA_GRID.index(0.0)].mean())
    oracle_macro = float(R.max(axis=1).mean())

    gbm_offline_ref = 0.32967905

    print(f"\n[baselines on this exact data]")
    print(f"  fixed lam=0.0 (token only) macro NDCG = {token_only:.4f}")
    print(f"  fixed lam=0.3 macro NDCG = {fixed_03_macro:.4f}")
    print(f"  fixed lam=1.0 (laplacian only) macro NDCG = {laplacian_only:.4f}")
    print(f"  oracle per-query best macro NDCG = {oracle_macro:.4f}")
    print(f"  GBM v2 offline (50/50 train/test, 5 seeds) ref = {gbm_offline_ref:.4f}")

    seeds = [int(s) for s in args.seeds.split(",")]

    runs = []
    for seed in seeds:
        for alpha in [0.5, 1.0, 2.0]:
            res = run_linucb_online(X, R, alpha=alpha, seed=seed)
            runs.append({
                "seed": seed,
                "alpha": alpha,
                "macro_ndcg": res["macro_ndcg"],
                "oracle_macro": res["oracle_macro"],
                "lambda_mean": res["lambda_mean"],
                "cum_regret_final": float(res["cum_regret"][-1]),
                "running_mean_at_50": float(res["running_mean"][min(49, n - 1)]),
                "running_mean_at_100": float(res["running_mean"][min(99, n - 1)]),
                "running_mean_at_200": float(res["running_mean"][min(199, n - 1)]),
                "running_mean_at_final": float(res["running_mean"][-1]),
            })
            print(f"  seed={seed:>3d} alpha={alpha:.1f}  macro NDCG={res['macro_ndcg']:.4f}  "
                  f"cum_regret={res['cum_regret'][-1]:.2f}  lam_mean={res['lambda_mean']:.3f}  "
                  f"win50@end={res['running_mean'][-1]:.4f}")

    alpha_1_runs = [r for r in runs if r["alpha"] == 1.0]
    macros_a1 = [r["macro_ndcg"] for r in alpha_1_runs]
    regrets_a1 = [r["cum_regret_final"] for r in alpha_1_runs]
    print(f"\n[LinUCB online alpha=1.0, mean over {len(seeds)} seeds]")
    print(f"  macro NDCG = {np.mean(macros_a1):.4f} +/- {np.std(macros_a1):.4f}")
    print(f"  cum_regret = {np.mean(regrets_a1):.2f} +/- {np.std(regrets_a1):.2f}")
    print(f"  vs GBM offline ref {gbm_offline_ref:.4f}: gap = "
          f"{(gbm_offline_ref - np.mean(macros_a1)) / gbm_offline_ref * 100:+.2f}%")
    print(f"  vs fixed lam=0.3 ({fixed_03_macro:.4f}): gain = "
          f"{(np.mean(macros_a1) - fixed_03_macro) / fixed_03_macro * 100:+.2f}%")
    print(f"  vs oracle per-query ({oracle_macro:.4f}): gap = "
          f"{(oracle_macro - np.mean(macros_a1)) / oracle_macro * 100:+.2f}%")

    summary = {
        "dataset": "nfcorpus",
        "n_queries": int(n),
        "n_features": int(d),
        "n_arms": int(n_arms),
        "lambda_grid": LAMBDA_GRID,
        "baselines": {
            "fixed_lam_0.0_token_only": token_only,
            "fixed_lam_0.3": fixed_03_macro,
            "fixed_lam_1.0_lap_only": laplacian_only,
            "oracle_per_query": oracle_macro,
            "gbm_v2_offline_supervised_ref": gbm_offline_ref,
        },
        "linucb_runs": runs,
        "linucb_alpha_1_macro_mean": float(np.mean(macros_a1)),
        "linucb_alpha_1_macro_std": float(np.std(macros_a1)),
        "linucb_alpha_1_cumregret_mean": float(np.mean(regrets_a1)),
        "n_seeds": len(seeds),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
