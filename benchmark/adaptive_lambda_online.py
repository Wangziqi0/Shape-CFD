"""
adaptive_lambda_online.py — Online deployment-time fusion-weight predictor.

Implements an incremental-SGD per-query lambda update with CUSUM concept-drift
detection, on top of the offline-trained GBM-11 predictor (V12 baseline).

Replay protocol:
    1. Load offline GBM-11 model (sklearn GradientBoostingRegressor, 11 features).
    2. For each query in 5-corpus replay stream:
        (a) features = build_features(query, top-K)
        (b) lambda_hat = clip( gbm.predict(features) + alpha * residual_running_mean, 0, 1 )
        (c) score = (1 - lambda_hat) * token_2stage + lambda_hat * lap_T5
        (d) ndcg_q = compute_ndcg10(score, qrels)
        (e) lambda_oracle_q = argmax_lambda(score(lambda), qrels)  # cheat for replay only
        (f) gradient = (lambda_hat - lambda_oracle_q)
        (g) residual_running_mean ← running_mean + eta * (-gradient)
        (h) cusum_stat += (gradient - cusum_baseline)
        (i) if |cusum_stat| > h_threshold: reset alpha to higher value (concept-drift)
    3. Output: per-query lambda_hat trajectory, ndcg trace, drift events.

Replay uses oracle lambda_q as supervision because real click feedback is
unavailable; in deployment, replace lambda_oracle_q by the click-derived target.

Usage:
    python benchmark/adaptive_lambda_online.py \
        --gbm-model benchmark/data/results/gbm11_5seed.pkl \
        --corpus-data benchmark/data/results/per_query_features_5corpora.json \
        --eta 0.005 \
        --cusum-threshold 2.0 \
        --window 100 \
        --out benchmark/data/results/adaptive_lambda_online_results.json

Output JSON schema:
    {
      "config": {...},
      "per_query_trajectory": [
          {"qid": ..., "corpus": ..., "lambda_hat": ..., "lambda_oracle": ...,
           "ndcg10": ..., "cusum_stat": ..., "drift_event": bool},
          ...
      ],
      "macro_ndcg10": float,
      "drift_events": int,
      "summary_per_corpus": {
          "nfcorpus": {"ndcg_mean": ..., "ndcg_std": ..., "drift_count": ...},
          ...
      }
    }
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("error: numpy required (pip install numpy)", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gbm-model", type=Path, required=True,
                   help="pickled sklearn GradientBoostingRegressor (11 features)")
    p.add_argument("--corpus-data", type=Path, required=True,
                   help="per-query features + per-lambda NDCG@10 grid (JSON)")
    p.add_argument("--eta", type=float, default=0.005,
                   help="SGD step size for residual running mean (default 0.005)")
    p.add_argument("--cusum-threshold", type=float, default=2.0,
                   help="CUSUM drift detection threshold (default 2.0)")
    p.add_argument("--cusum-baseline", type=float, default=0.0,
                   help="CUSUM baseline (default 0.0, neutral)")
    p.add_argument("--window", type=int, default=100,
                   help="exponential window for residual running mean (default 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def compute_ndcg10(ranked_relevances):
    """DCG@10 / IDCG@10 with binary relevance, log_2(rank+1) discount."""
    rel = np.asarray(ranked_relevances[:10], dtype=float)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, 12))
    dcg = float(np.sum(rel * discounts[: len(rel)]))
    ideal_rel = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal_rel * discounts[: len(ideal_rel)]))
    return dcg / idcg if idcg > 0 else 0.0


def select_lambda_oracle(per_lambda_ndcg, lambda_grid):
    """Pick the lambda in lambda_grid that maximizes NDCG@10 for this query."""
    best_idx = int(np.argmax(per_lambda_ndcg))
    return float(lambda_grid[best_idx]), float(per_lambda_ndcg[best_idx])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------- load offline GBM-11 + per-query data ----------
    with open(args.gbm_model, "rb") as fh:
        gbm = pickle.load(fh)
    with open(args.corpus_data, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    lambda_grid = np.linspace(0.0, 1.0, 11)
    queries = []
    for corpus_name, qlist in data["queries_per_corpus"].items():
        for q in qlist:
            q["corpus"] = corpus_name
            queries.append(q)
    rng.shuffle(queries)  # randomize replay order

    # ---------- replay loop with incremental SGD + CUSUM ----------
    residual_running_mean = 0.0
    cusum_stat = 0.0
    drift_events = 0
    eta_effective = args.eta
    trajectory = []

    for qi, q in enumerate(queries):
        feats = np.array(q["features"], dtype=float).reshape(1, -1)
        per_lambda_ndcg = np.array(q["per_lambda_ndcg10"], dtype=float)
        lambda_oracle, ndcg_oracle = select_lambda_oracle(per_lambda_ndcg, lambda_grid)
        lambda_gbm = float(gbm.predict(feats)[0])
        lambda_hat = float(np.clip(lambda_gbm + eta_effective * residual_running_mean, 0.0, 1.0))

        # eval ndcg at lambda_hat (interpolate to nearest lambda_grid point)
        idx_near = int(np.argmin(np.abs(lambda_grid - lambda_hat)))
        ndcg_q = float(per_lambda_ndcg[idx_near])

        # SGD update on residual running mean (target = oracle - GBM prediction)
        gradient_signal = lambda_oracle - lambda_gbm
        alpha_window = 1.0 / args.window
        residual_running_mean = (1 - alpha_window) * residual_running_mean + alpha_window * gradient_signal

        # CUSUM update for drift detection
        cusum_stat += gradient_signal - args.cusum_baseline
        drift = abs(cusum_stat) > args.cusum_threshold
        if drift:
            drift_events += 1
            cusum_stat = 0.0  # reset
            eta_effective = min(eta_effective * 1.5, 0.05)  # increase step on drift

        trajectory.append({
            "qid": q["qid"],
            "corpus": q["corpus"],
            "lambda_gbm": round(lambda_gbm, 4),
            "lambda_hat": round(lambda_hat, 4),
            "lambda_oracle": round(lambda_oracle, 4),
            "ndcg10_at_hat": round(ndcg_q, 4),
            "ndcg10_oracle": round(ndcg_oracle, 4),
            "cusum_stat": round(cusum_stat, 4),
            "drift_event": drift,
        })

    # ---------- aggregate per-corpus + macro ----------
    per_corpus_summary = {}
    for corpus_name in data["queries_per_corpus"].keys():
        rows = [r for r in trajectory if r["corpus"] == corpus_name]
        if not rows:
            continue
        ndcg_vals = np.array([r["ndcg10_at_hat"] for r in rows])
        per_corpus_summary[corpus_name] = {
            "n_queries": len(rows),
            "ndcg_mean": float(ndcg_vals.mean()),
            "ndcg_std": float(ndcg_vals.std(ddof=1)) if len(rows) > 1 else 0.0,
            "drift_count": sum(1 for r in rows if r["drift_event"]),
            "lambda_hat_mean": float(np.mean([r["lambda_hat"] for r in rows])),
        }

    macro_ndcg = float(np.mean([s["ndcg_mean"] for s in per_corpus_summary.values()]))

    out = {
        "config": {
            "gbm_model": str(args.gbm_model),
            "corpus_data": str(args.corpus_data),
            "eta_initial": args.eta,
            "cusum_threshold": args.cusum_threshold,
            "cusum_baseline": args.cusum_baseline,
            "window": args.window,
            "seed": args.seed,
            "n_queries_total": len(queries),
        },
        "macro_ndcg10": macro_ndcg,
        "drift_events": drift_events,
        "summary_per_corpus": per_corpus_summary,
        "per_query_trajectory": trajectory,  # full trajectory; reduce in production
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # ---------- summary print ----------
    print(f"adaptive_lambda_online: {len(queries)} queries replayed", file=sys.stderr)
    print(f"  macro NDCG@10:         {macro_ndcg:.4f}", file=sys.stderr)
    print(f"  drift events detected: {drift_events}", file=sys.stderr)
    for c, s in per_corpus_summary.items():
        print(f"  {c}: ndcg={s['ndcg_mean']:.4f}±{s['ndcg_std']:.4f} (n={s['n_queries']}, drift={s['drift_count']})", file=sys.stderr)
    print(f"output written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
