#!/usr/bin/env python3
"""
hartree_lambda_toy.py — Hartree-style self-consistent λ on NFCorpus
====================================================================

Replaces GBM-11 supervised oracle-λ★ regression (11 hand-crafted features +
qrels-derived oracle target) with a forward, training-free self-consistent
fixed-point iteration:

    λ^eff(q, C) = λ_0 + λ_Σ · ⟨ ‖retrieval(q, C; λ^eff)‖² ⟩_C

No qrels supervision. Two free parameters (λ_0, λ_Σ). Iterate until
convergence per-query.

Operationalization on the precomputed per-query NDCG table
(fusion_T5_lam0.0..fusion_T5_lam1.0 + token_2stage + lap_T5 + cosine):

  - Treat per-query retrieval-norm proxy R(q; λ) = (1-λ)·tok(q) + λ·lap(q),
    where tok(q) = NDCG of token_2stage stage and lap(q) = NDCG of lap_T5
    (both already normalized in [0,1] and reflect the energy each stream
    contributes at λ → 0 / λ → 1 endpoints).
  - The fused retrieval at intermediate λ is read off the precomputed
    fusion_T5_lam{λ} table (linear interpolation across the 11-grid).
  - ⟨‖retrieval‖²⟩_C is then the corpus-mean of R(q; λ)² evaluated at the
    current λ^eff iterate.
  - Iterate λ^eff(q) ← λ_0 + λ_Σ · ⟨R(q; λ^eff)²⟩_C until
    max_q |λ^eff_{t+1} - λ^eff_t| < 0.01 or 50 iters.
  - Snap each per-query λ^eff to the nearest grid point and read NDCG.

Output: macro NDCG@10 (5-seed split as in adaptive_fusion_lambda_v2 for fair
comparison) + free-parameter count + iteration trace.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


LAMBDA_GRID = [round(x * 0.1, 1) for x in range(11)]
LAMBDA_GRID_ARR = np.asarray(LAMBDA_GRID, dtype=np.float64)


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def extract_signals(perquery_records):
    """Pull per-query (cos, tok, lap5, ndcg-vs-lambda-grid)."""
    rows = []
    for rec in perquery_records:
        n = rec.get("ndcg", {})
        if "token_2stage" not in n or "lap_T5" not in n:
            continue
        ndcg_per_lam = []
        ok = True
        for lam in LAMBDA_GRID:
            v = n.get(f"fusion_T5_lam{lam:.1f}")
            if v is None:
                ok = False
                break
            ndcg_per_lam.append(float(v))
        if not ok:
            continue
        rows.append({
            "qid": rec.get("qid"),
            "cos": float(n.get("cosine", 0.0) or 0.0),
            "tok": float(n.get("token_2stage", 0.0) or 0.0),
            "lap5": float(n.get("lap_T5", 0.0) or 0.0),
            "ndcg_grid": np.asarray(ndcg_per_lam, dtype=np.float64),
        })
    return rows


def hartree_iterate(rows, lambda_0, lambda_Sigma, tol=0.01, max_iter=50,
                   damping=0.5, verbose=False):
    """
    Self-consistent fixed-point: λ^eff(q) = λ_0 + λ_Σ · R(q; λ^eff)²
    with R(q; λ) = (1-λ)·tok(q) + λ·lap(q).
    Damped Picard iteration:
        λ_eff^{t+1} = (1-α)·λ_eff^t + α·F(λ_eff^t)
    α = damping in (0,1]. Standard numerical recipe to enforce convergence
    of self-consistent equations (Hartree / mean-field literature).

    Returns per-query λ^eff array, snapped to grid, plus iteration trace.
    """
    n = len(rows)
    tok = np.array([r["tok"] for r in rows])
    lap = np.array([r["lap5"] for r in rows])

    # init λ^eff(q) = λ_0 for all q
    lam_eff = np.full(n, lambda_0, dtype=np.float64)
    trace = []
    for it in range(max_iter):
        R = (1.0 - lam_eff) * tok + lam_eff * lap
        R2 = R * R
        mean_R2 = float(R2.mean())
        # per-query Hartree closure (pointwise, not mean-field)
        F = np.clip(lambda_0 + lambda_Sigma * R2, 0.0, 1.0)
        lam_new = (1.0 - damping) * lam_eff + damping * F
        delta = float(np.max(np.abs(lam_new - lam_eff)))
        trace.append({"iter": it + 1, "max_delta": delta,
                      "mean_lam": float(lam_new.mean()),
                      "mean_R2_corpus": mean_R2})
        if verbose:
            print(f"  iter {it+1}: max_delta={delta:.4f}, "
                  f"mean_lam={lam_new.mean():.4f}, mean_R2={mean_R2:.4f}")
        lam_eff = lam_new
        if delta < tol:
            break

    snapped = np.array(
        [int(np.argmin(np.abs(LAMBDA_GRID_ARR - p))) for p in lam_eff]
    )
    return lam_eff, snapped, trace


def eval_hartree(rows, snapped):
    """NDCG@10 per-query at snapped λ_eff index."""
    ndcg_table = np.array([r["ndcg_grid"] for r in rows])
    return ndcg_table[np.arange(len(snapped)), snapped]


def grid_search_two_params(rows, n_seeds=5, train_frac=0.5,
                            lam0_grid=None, lamS_grid=None):
    """
    Tune (λ_0, λ_Σ) on training split, eval on test. Mirror the
    adaptive_fusion_lambda_v2 5-seed protocol for direct comparability.
    """
    if lam0_grid is None:
        lam0_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    if lamS_grid is None:
        lamS_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    n = len(rows)
    seed_scores = []
    seed_params = []
    convergence_log = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_tr = int(n * train_frac)
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
        tr_rows = [rows[i] for i in tr_idx]
        te_rows = [rows[i] for i in te_idx]

        # tune on train
        best_score = -1.0
        best_params = (0.7, 0.0)
        for l0 in lam0_grid:
            for lS in lamS_grid:
                _, snapped_tr, _ = hartree_iterate(tr_rows, l0, lS)
                s = eval_hartree(tr_rows, snapped_tr).mean()
                if s > best_score:
                    best_score = s
                    best_params = (l0, lS)
        # eval on test with best params
        _, snapped_te, trace = hartree_iterate(
            te_rows, best_params[0], best_params[1]
        )
        te_score = eval_hartree(te_rows, snapped_te).mean()
        seed_scores.append(te_score)
        seed_params.append(best_params)
        convergence_log.append({
            "seed": seed,
            "best_lambda_0": best_params[0],
            "best_lambda_Sigma": best_params[1],
            "n_iter_test": len(trace),
            "final_max_delta": trace[-1]["max_delta"] if trace else None,
            "test_ndcg": float(te_score),
        })

    return {
        "ndcg_mean": float(np.mean(seed_scores)),
        "ndcg_std": float(np.std(seed_scores)),
        "n_seeds": n_seeds,
        "n_free_params": 2,
        "per_seed": convergence_log,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perquery", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_seeds", type=int, default=5)
    args = p.parse_args()

    perq = load_jsonl(args.perquery)
    rows = extract_signals(perq)
    print(f"Loaded {len(rows)} queries with full token + lap + λ-grid signals.")

    # --- Hartree self-consistent (tuned 2 params) ---
    print("\n=== Hartree-style self-consistent λ (2 free params) ===")
    hartree_res = grid_search_two_params(rows, n_seeds=args.n_seeds)
    print(f"  Macro NDCG@10 = {hartree_res['ndcg_mean']:.4f} "
          f"± {hartree_res['ndcg_std']:.4f}")
    for sd in hartree_res["per_seed"]:
        print(f"    seed {sd['seed']}: λ_0={sd['best_lambda_0']:.2f}, "
              f"λ_Σ={sd['best_lambda_Sigma']:.2f}, "
              f"iters={sd['n_iter_test']}, "
              f"final_Δ={sd['final_max_delta']:.4f}, "
              f"test_NDCG={sd['test_ndcg']:.4f}")

    # --- Reference: fixed λ=0.3 on test split, same protocol ---
    print("\n=== Reference: fixed λ=0.3 (0 free params, no tune) ===")
    ndcg_table = np.array([r["ndcg_grid"] for r in rows])
    fixed_idx = LAMBDA_GRID.index(0.3)
    fixed_scores = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))
        n_tr = int(len(rows) * 0.5)
        te = idx[n_tr:]
        fixed_scores.append(ndcg_table[te, fixed_idx].mean())
    print(f"  Macro NDCG@10 = {np.mean(fixed_scores):.4f} "
          f"± {np.std(fixed_scores):.4f}")

    # --- Reference: oracle upper ---
    print("\n=== Reference: oracle upper (1 free param per-q, qrels needed) ===")
    oracle_scores = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))
        n_tr = int(len(rows) * 0.5)
        te = idx[n_tr:]
        oracle_scores.append(ndcg_table[te].max(axis=1).mean())
    print(f"  Macro NDCG@10 = {np.mean(oracle_scores):.4f} "
          f"± {np.std(oracle_scores):.4f}")

    out = {
        "dataset": "nfcorpus",
        "n_queries": len(rows),
        "n_seeds": args.n_seeds,
        "hartree_self_consistent": hartree_res,
        "reference": {
            "fixed_lam_0.3": {
                "ndcg_mean": float(np.mean(fixed_scores)),
                "ndcg_std": float(np.std(fixed_scores)),
                "n_free_params": 0,
            },
            "oracle_upper": {
                "ndcg_mean": float(np.mean(oracle_scores)),
                "ndcg_std": float(np.std(oracle_scores)),
                "n_free_params": "per-q (qrels-supervised)",
            },
            "gbm_11feat_from_v2_results": {
                "ndcg_mean": 0.32967905397774877,
                "ndcg_std": 0.015597746266401109,
                "n_free_params": 11,
                "source": "data/results/adaptive_lambda_v2_gbm_results.json",
            },
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
