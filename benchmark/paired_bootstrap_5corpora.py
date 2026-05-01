#!/usr/bin/env python3
"""
paired_bootstrap_5corpora.py — Reviewer 2 W3 / Senior Reviewer Issue 2.1 fix
====================================================================

Run paired bootstrap (10,000 iterations, two-sided) per BEIR dataset
between cosine baseline and Ours Best (per-dataset λ★ fusion or
token_2stage), reporting (effect size, p-value, 95% CI on Δ NDCG@10).

Datasets and Ours-Best definitions per paper Table 1:
  NFCorpus: fusion λ★=0.4 (vs cosine)
  ArguAna:  token_2stage (= fusion λ=0.0)
  SciFact:  fusion λ★=0.7
  SCIDOCS:  fusion λ★=0.2
  FiQA:     fusion λ★=0.2

Input: per-query JSONL produced by fusion_ablation_sweep.js
  benchmark/data/results/fusion_ablation_<dataset>_perquery.jsonl

Each line has: { qid, ndcg: { cosine, token_2stage, lap_T1..T20,
                              fusion_T5_lam0.0..1.0 } }

Output:
  benchmark/data/results/paired_bootstrap_5corpora.json
  Per-dataset: cosine mean, ours mean, abs_diff, rel_diff,
               cohen_d, p_value (two-sided), 95% CI [lo, hi].
"""

import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict


CONFIG = {
    "nfcorpus": {"ours_key": "fusion_T5_lam0.4"},
    "arguana":  {"ours_key": "token_2stage"},  # = fusion λ=0.0
    "scifact":  {"ours_key": "fusion_T5_lam0.7"},
    "scidocs":  {"ours_key": "fusion_T5_lam0.2"},
    "fiqa":     {"ours_key": "fusion_T5_lam0.2"},
}


def load_pairs(path, ours_key):
    """Returns (cosine, ours) per-query NDCG arrays."""
    cos, ours = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n = rec.get("ndcg", {})
            c = n.get("cosine")
            o = n.get(ours_key)
            if c is None or o is None:
                continue
            if not (np.isfinite(c) and np.isfinite(o)):
                continue
            cos.append(float(c))
            ours.append(float(o))
    return np.asarray(cos), np.asarray(ours)


def paired_bootstrap(cos, ours, n_iter=10000, seed=42):
    """Two-sided paired bootstrap on Δ = ours - cos.

    Returns: dict with mean_cos, mean_ours, mean_diff, cohen_d,
             p_value_two_sided, ci_low, ci_high (95%).
    """
    diffs = ours - cos
    n = len(diffs)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    obs = diffs.mean()
    # two-sided p-value: how often does bootstrapped mean cross 0?
    # Center on observed mean to test H0: μ=0 (Efron paired test
    # via shifted bootstrap distribution).
    centered = boot_means - obs
    p_two = (np.abs(centered) >= abs(obs)).mean()
    p_two = max(p_two, 1.0 / n_iter)  # floor at 1/n_iter
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    sd = diffs.std(ddof=1) if n > 1 else 0.0
    cohen_d = obs / sd if sd > 0 else 0.0
    return {
        "n_queries": int(n),
        "mean_cosine": float(cos.mean()),
        "mean_ours": float(ours.mean()),
        "abs_diff": float(obs),
        "rel_diff_pct": float(obs / cos.mean() * 100) if cos.mean() > 0 else None,
        "cohen_d": float(cohen_d),
        "p_value_two_sided": float(p_two),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",
                   default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results")
    p.add_argument("--n_iter", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",
                   default="/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/paired_bootstrap_5corpora.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    results = {"n_iter": args.n_iter, "seed": args.seed, "datasets": {}}
    print(f"# paired bootstrap (n_iter={args.n_iter}, seed={args.seed})\n")
    print(f"{'Dataset':<12} {'n':>5} {'cos':>8} {'ours':>8} {'Δ':>8} {'rel%':>8} "
          f"{'Cohen d':>8} {'p (two-side)':>14} {'95% CI':>20}")
    print("-" * 110)

    for ds, cfg in CONFIG.items():
        path = Path(args.results_dir) / f"fusion_ablation_{ds}_perquery.jsonl"
        if not path.exists():
            print(f"{ds:<12} SKIP missing {path}")
            continue
        cos, ours = load_pairs(str(path), cfg["ours_key"])
        if len(cos) == 0:
            print(f"{ds:<12} SKIP empty pairs")
            continue
        r = paired_bootstrap(cos, ours, n_iter=args.n_iter, seed=args.seed)
        r["ours_key"] = cfg["ours_key"]
        results["datasets"][ds] = r
        ci = f"[{r['ci_95_low']:+.4f},{r['ci_95_high']:+.4f}]"
        print(f"{ds:<12} {r['n_queries']:>5} {r['mean_cosine']:>8.4f} {r['mean_ours']:>8.4f} "
              f"{r['abs_diff']:>+8.4f} {r['rel_diff_pct']:>+7.2f}% {r['cohen_d']:>8.3f} "
              f"{r['p_value_two_sided']:>14.4g} {ci:>20}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
