#!/usr/bin/env python3
"""
GBM-11 leave-one-feature-out ablation: drop each of the 11 features one at a
time and report the resulting macro NDCG@10 lift vs the full GBM-11 baseline.
Identifies which features carry signal and which are redundant.

Output: benchmark/data/results/gbm_loo_feature_ablation_results.json
"""
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, '/home/amd/HEZIMENG/Shape-CFD/benchmark')
from adaptive_fusion_lambda_v2 import (
    load_jsonl, load_query_lengths, extract_features_v2,
    LAMBDA_GRID, eval_fixed, eval_oracle,
)

DATASETS = ['nfcorpus', 'scifact', 'arguana', 'scidocs', 'fiqa']
RESULTS_DIR = Path('/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results')
BEIR_DIR = Path('/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data')

FEATURE_NAMES = [
    "kendall_tau", "spearman_rho", "cosine_ndcg", "query_length",
    "token_2stage_ndcg", "lap_T5_ndcg", "token_minus_cosine",
    "lap_minus_cosine", "lap_T1_T20_spread", "abs_token_lap_disagreement",
    "lap_Tsweep_range",
]


def load_corpus_data(ds):
    perq_path = RESULTS_DIR / f'fusion_ablation_{ds}_perquery.jsonl'
    q_path = BEIR_DIR / ds / 'queries.jsonl'
    perq = load_jsonl(str(perq_path))
    qlens = load_query_lengths(str(q_path))
    X, ndcg_tab, qids = extract_features_v2(perq, qlens)
    return X, ndcg_tab


def eval_gbm_with_features(X_train, ndcg_train, X_test, ndcg_test, n_seeds=5):
    """Train GBM on a feature subset, return mean NDCG@10."""
    scores = []
    for seed in range(n_seeds):
        # Use oracle lambda* as supervised target
        lam_idx_train = np.argmax(ndcg_train, axis=1)
        y_train = np.array(LAMBDA_GRID)[lam_idx_train]
        reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed,
        )
        reg.fit(X_train, y_train)
        lam_pred = np.clip(reg.predict(X_test), 0, 1)
        lam_grid = np.array(LAMBDA_GRID)
        lam_idx_pred = np.argmin(np.abs(lam_grid[None, :] - lam_pred[:, None]), axis=1)
        scores.append(ndcg_test[np.arange(len(ndcg_test)), lam_idx_pred].mean())
    return float(np.mean(scores)), float(np.std(scores))


def main():
    print('[gbm-loo] Loading 5 corpus per-query features...')
    data = {}
    for ds in DATASETS:
        X, ndcg = load_corpus_data(ds)
        data[ds] = {'X': X, 'ndcg': ndcg}
        print(f'  {ds:<10} n={len(X):4d}')

    # Full GBM-11 reference (in-corpus 50/50, average across 5 corpora)
    print('\n[gbm-loo] Computing in-corpus 50/50 split GBM-11 reference per corpus + leave-one-feature ablation...')

    rng = np.random.default_rng(42)
    full_per_corpus = {}
    loo_per_corpus = defaultdict(dict)

    for ds in DATASETS:
        X = data[ds]['X']
        ndcg = data[ds]['ndcg']
        n = len(X)
        # 50/50 split (one seed for time)
        idx = rng.permutation(n)
        n_tr = n // 2
        tr, te = idx[:n_tr], idx[n_tr:]

        full_score, _ = eval_gbm_with_features(X[tr], ndcg[tr], X[te], ndcg[te], n_seeds=5)
        full_per_corpus[ds] = full_score
        print(f'  {ds:<10} full GBM-11: {full_score:.4f}')

        # Leave-one-feature-out
        for f_idx, f_name in enumerate(FEATURE_NAMES):
            keep = [i for i in range(11) if i != f_idx]
            X_tr_loo = X[tr][:, keep]
            X_te_loo = X[te][:, keep]
            score_loo, _ = eval_gbm_with_features(X_tr_loo, ndcg[tr], X_te_loo, ndcg[te], n_seeds=5)
            loo_per_corpus[f_name][ds] = score_loo

    # Aggregate: macro across 5 corpora
    macro_full = np.mean(list(full_per_corpus.values()))
    print(f'\n[gbm-loo] Full GBM-11 macro: {macro_full:.4f}')
    print(f'\n{"feature":<35} {"macro w/o":<11} {"Δmacro%":<10} signal_strength')
    print('-' * 80)

    feature_impact = []
    for f_name in FEATURE_NAMES:
        macro_loo = np.mean(list(loo_per_corpus[f_name].values()))
        delta_pct = 100 * (macro_loo - macro_full) / macro_full if macro_full > 0 else 0
        importance = -delta_pct  # higher = more important
        feature_impact.append((f_name, macro_loo, delta_pct, importance))

    feature_impact.sort(key=lambda x: x[3], reverse=True)
    for f_name, macro_loo, delta_pct, imp in feature_impact:
        signal = "HIGH" if imp > 0.5 else "MED" if imp > 0.1 else "LOW" if imp > -0.1 else "NEG"
        print(f'{f_name:<35} {macro_loo:<11.4f} {delta_pct:<+10.2f} {signal}')

    out = {
        "method": "GBM-11 leave-one-feature-out ablation",
        "full_gbm11_macro": float(macro_full),
        "full_per_corpus": full_per_corpus,
        "loo_per_feature": {f_name: {
            "per_corpus_ndcg": dict(loo_per_corpus[f_name]),
            "macro_loo": float(np.mean(list(loo_per_corpus[f_name].values()))),
            "delta_macro_pct": float(100 * (np.mean(list(loo_per_corpus[f_name].values())) - macro_full) / macro_full),
        } for f_name in FEATURE_NAMES},
        "feature_importance_ranking": [{"feature": f, "importance": float(imp)} for f, _, _, imp in feature_impact],
    }

    out_path = RESULTS_DIR / 'gbm_loo_feature_ablation_results.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\n[gbm-loo] Saved: {out_path}')


if __name__ == "__main__":
    main()
