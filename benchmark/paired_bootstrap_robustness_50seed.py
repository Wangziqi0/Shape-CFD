"""
paired_bootstrap_robustness_50seed.py
=====================================
51-seed paired bootstrap robustness check on existing 5 BEIR corpus.

paper §subsec:significance 现有数字基于 single seed=42 paired bootstrap (10000 iter).
This script reruns with 51 different seeds (42, 0, 1, ..., 49) to verify
the single-seed numbers are stable, and reports macro mean ± std across seeds.

Output JSON includes:
  - per-corpus per-seed NDCG@10 delta (Ours_Best - Cosine)
  - per-corpus 51-seed mean / std / 95% CI of delta
  - p-value 51-seed mean ± std (verifies p<0.001 stability)
  - Cohen's d 51-seed mean ± std

NUMA-aware: launch with `numactl --cpunodebind=N --membind=N python3 ...`
or run sequentially (the inner loop is single-threaded numpy).

Inputs (existing):
  benchmark/data/results/paired_bootstrap_5corpora.json (single seed=42 baseline)
  benchmark/data/results/per_query_ndcg_5corpora.json (per-query NDCG, if exists)

Or fallback: load from existing per-corpus eval JSON.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_per_query_data(corpus, results_root):
    """Load per-query NDCG@10 from fusion_ablation_<corpus>_perquery.jsonl."""
    ours_key_map = {
        'nfcorpus': 'fusion_T5_lam0.4',
        'arguana': 'token_2stage',
        'scifact': 'fusion_T5_lam0.7',
        'scidocs': 'fusion_T5_lam0.2',
        'fiqa': 'fusion_T5_lam0.2',
    }
    jsonl = results_root / f'fusion_ablation_{corpus}_perquery.jsonl'
    if not jsonl.exists():
        return None, None
    cosine_v, ours_v = [], []
    ours_key = ours_key_map.get(corpus, 'token_2stage')
    with open(jsonl, encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            ndcg = d.get('ndcg', {})
            if 'cosine' in ndcg and ours_key in ndcg:
                cosine_v.append(ndcg['cosine'])
                ours_v.append(ndcg[ours_key])
    if not cosine_v:
        return None, None
    return np.array(cosine_v, dtype=float), np.array(ours_v, dtype=float)


def paired_bootstrap_delta(cosine_v, ours_v, n_iter=10000, seed=42):
    """Returns (delta_mean, ci_low, ci_high, pvalue, cohen_d)."""
    rng = np.random.default_rng(seed)
    n = len(cosine_v)
    deltas = ours_v - cosine_v
    mean_delta = float(np.mean(deltas))

    bootstrap_means = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[i] = np.mean(deltas[idx])

    ci_low = float(np.quantile(bootstrap_means, 0.025))
    ci_high = float(np.quantile(bootstrap_means, 0.975))

    # two-sided p: fraction of bootstrap means with opposite sign
    if mean_delta > 0:
        pvalue = float(np.mean(bootstrap_means <= 0)) * 2
    else:
        pvalue = float(np.mean(bootstrap_means >= 0)) * 2
    pvalue = max(pvalue, 1.0 / n_iter)  # numerical floor

    cohen_d = mean_delta / (np.std(deltas, ddof=1) + 1e-12)
    return mean_delta, ci_low, ci_high, pvalue, float(cohen_d)


def main():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--n-seeds", type=int, default=51)
    p.add_argument("--results-root", type=Path,
                   default=Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/paired_bootstrap_robustness_50seed_20260503.json"))
    args = p.parse_args()

    corpora = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]
    seeds = [42] + list(range(args.n_seeds - 1))

    out = {
        "config": {
            "n_iter_per_seed": args.n_iter,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "corpora": corpora,
        },
        "per_corpus": {},
        "macro_summary": {},
        "verdict": {},
    }

    t0 = time.time()
    macro_deltas = []
    macro_ci_widths = []

    for c in corpora:
        cv, ov = load_per_query_data(c, args.results_root)
        if cv is None:
            print(f"[{c}] WARN: per-query data not found, skipping", file=sys.stderr)
            out["per_corpus"][c] = {"error": "per-query data not loaded"}
            continue

        per_seed = []
        for s in seeds:
            d, lo, hi, pval, cd = paired_bootstrap_delta(cv, ov, n_iter=args.n_iter, seed=s)
            per_seed.append({
                "seed": s,
                "delta_mean": round(d, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "p_value": round(pval, 6),
                "cohen_d": round(cd, 4),
            })
        deltas = np.array([r["delta_mean"] for r in per_seed])
        ci_widths = np.array([r["ci_high"] - r["ci_low"] for r in per_seed])
        pvals = np.array([r["p_value"] for r in per_seed])
        cds = np.array([r["cohen_d"] for r in per_seed])

        out["per_corpus"][c] = {
            "n_queries": len(cv),
            "per_seed": per_seed,
            "summary": {
                "delta_mean_across_seeds": float(deltas.mean()),
                "delta_std_across_seeds": float(deltas.std(ddof=1)),
                "ci_width_mean": float(ci_widths.mean()),
                "p_value_max_across_seeds": float(pvals.max()),
                "p_value_mean": float(pvals.mean()),
                "cohen_d_mean": float(cds.mean()),
                "cohen_d_std": float(cds.std(ddof=1)),
            },
        }
        macro_deltas.append(deltas.mean())
        macro_ci_widths.append(ci_widths.mean())

        print(f"[{c}] n={len(cv)} delta={deltas.mean():.4f}±{deltas.std(ddof=1):.5f} "
              f"p_max={pvals.max():.4f} cohen_d={cds.mean():.3f}±{cds.std(ddof=1):.4f}", file=sys.stderr)

    if macro_deltas:
        out["macro_summary"] = {
            "macro_delta": float(np.mean(macro_deltas)),
            "macro_ci_width": float(np.mean(macro_ci_widths)),
            "all_p_under_001": all(
                out["per_corpus"][c]["summary"]["p_value_max_across_seeds"] < 0.001
                for c in corpora if "summary" in out["per_corpus"].get(c, {})
            ),
            "all_seeds_consistent": all(
                out["per_corpus"][c]["summary"]["delta_std_across_seeds"] < 0.005
                for c in corpora if "summary" in out["per_corpus"].get(c, {})
            ),
        }
        out["verdict"]["robustness"] = "stable" if out["macro_summary"]["all_seeds_consistent"] else "unstable"

    elapsed = time.time() - t0
    out["wall_time_sec"] = elapsed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n=== robustness summary ===", file=sys.stderr)
    if out.get("macro_summary"):
        print(f"  macro delta:        {out['macro_summary']['macro_delta']:.4f}", file=sys.stderr)
        print(f"  all p < 0.001:      {out['macro_summary']['all_p_under_001']}", file=sys.stderr)
        print(f"  all seeds consist:  {out['macro_summary']['all_seeds_consistent']}", file=sys.stderr)
        print(f"  verdict:            {out['verdict'].get('robustness', 'n/a')}", file=sys.stderr)
    print(f"  wall_time: {elapsed:.0f}s", file=sys.stderr)
    print(f"  output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
