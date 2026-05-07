#!/usr/bin/env python3
"""
gbm_cross_corpus_loco.py — Leave-One-Corpus-Out cross-corpus GBM-11 transfer test.

Audits whether GBM-11's +11.5% lift is genuine cross-corpus signal or per-corpus
oracle leakage. For each held-out test corpus C_test:
  - Train GBM-11 on the union of remaining 4 corpora (all queries pooled).
  - Test on C_test, report NDCG@10 of cross-corpus GBM-11 prediction vs:
      * fixed lambda=0.3 baseline
      * in-corpus 50/50 GBM-11 (reference, from adaptive_lambda_v2_gbm_results.json)
      * per-query oracle upper bound on C_test
  - Cross-corpus NDCG close to in-corpus NDCG  =>  transferable signal (paper claim valid)
  - Cross-corpus much worse                    =>  per-corpus oracle dependency (caveat needed)
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
    LAMBDA_GRID, eval_fixed, eval_oracle, eval_learned_gbm,
)

DATASETS = ['nfcorpus', 'scifact', 'arguana', 'scidocs', 'fiqa']
RESULTS_DIR = Path('/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results')
BEIR_DIR = Path('/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data')


def load_corpus(ds):
    perq_path = RESULTS_DIR / f'fusion_ablation_{ds}_perquery.jsonl'
    q_path = BEIR_DIR / ds / 'queries.jsonl'
    perq = load_jsonl(str(perq_path))
    qlens = load_query_lengths(str(q_path))
    X, ndcg_tab, qids = extract_features_v2(perq, qlens)
    return X, ndcg_tab, qids


def main():
    print('[LOCO] Loading 5 corpus per-query features...')
    data = {}
    for ds in DATASETS:
        X, ndcg, qids = load_corpus(ds)
        data[ds] = {'X': X, 'ndcg': ndcg, 'qids': qids}
        print(f'  {ds:10s}  n={len(qids)}  features={X.shape[1]}')

    # Reference: in-corpus 50/50 GBM-11 from existing JSON
    in_corpus_ref_path = RESULTS_DIR / 'adaptive_lambda_v2_gbm_results.json'
    in_corpus_ref = json.loads(in_corpus_ref_path.read_text())
    in_corpus = {}
    for ds, sresult in in_corpus_ref['datasets'].items():
        in_corpus[ds] = sresult['strategies']['learned_gbm_11feat']['ndcg_mean']

    print('\n[LOCO] Cross-corpus GBM-11 transfer test (5 LOCO splits, 5 seeds each):')
    print(f'{"Test corpus":<12} {"in-corp GBM":>12} {"cross-corp GBM":>16} {"fixed=0.3":>11} {"oracle":>9} {"transfer Δ%":>13}')
    print('-' * 80)

    results = {'method': 'GBM-11 leave-one-corpus-out cross-corpus transfer', 'splits': {}}

    for test_ds in DATASETS:
        X_test = data[test_ds]['X']
        ndcg_test = data[test_ds]['ndcg']
        # train on union of 4 remaining
        X_train = np.vstack([data[d]['X'] for d in DATASETS if d != test_ds])
        ndcg_train = np.vstack([data[d]['ndcg'] for d in DATASETS if d != test_ds])

        # Run cross-corpus GBM-11, average across 5 seeds (only seed affects sklearn)
        cross_scores = []
        for seed in range(5):
            reg = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed,
            )
            # Target: per-query oracle lambda* selected from train set
            lam_idx_train = np.argmax(ndcg_train, axis=1)
            y_train = np.array(LAMBDA_GRID)[lam_idx_train]
            reg.fit(X_train, y_train)
            lam_pred = reg.predict(X_test)
            lam_pred = np.clip(lam_pred, 0.0, 1.0)
            # Snap to nearest grid point
            lam_grid = np.array(LAMBDA_GRID)
            lam_idx_pred = np.argmin(np.abs(lam_grid[None, :] - lam_pred[:, None]), axis=1)
            ndcg_pred = ndcg_test[np.arange(len(ndcg_test)), lam_idx_pred]
            cross_scores.append(ndcg_pred.mean())

        cross_mean = float(np.mean(cross_scores))
        cross_std = float(np.std(cross_scores))
        fixed = float(eval_fixed(ndcg_test, 0.3).mean())
        oracle = float(eval_oracle(ndcg_test).mean())
        in_corp_score = in_corpus[test_ds]
        transfer_pct = 100.0 * (cross_mean - in_corp_score) / in_corp_score if in_corp_score > 0 else 0.0

        print(f'{test_ds:<12} {in_corp_score:>12.4f} {cross_mean:>14.4f}±{cross_std:.4f}  '
              f'{fixed:>11.4f} {oracle:>9.4f} {transfer_pct:>+12.2f}%')

        results['splits'][test_ds] = {
            'in_corpus_gbm_50_50': in_corp_score,
            'cross_corpus_gbm_loco_mean': cross_mean,
            'cross_corpus_gbm_loco_std': cross_std,
            'fixed_lambda_0.3': fixed,
            'oracle_upper': oracle,
            'transfer_pct_vs_in_corpus': transfer_pct,
            'n_test': len(ndcg_test),
            'n_train_union_4corpora': len(ndcg_train),
        }

    # Summary
    macro_in = np.mean(list(in_corpus.values()))
    macro_cross = np.mean([results['splits'][d]['cross_corpus_gbm_loco_mean'] for d in DATASETS])
    macro_fixed = np.mean([results['splits'][d]['fixed_lambda_0.3'] for d in DATASETS])
    macro_oracle = np.mean([results['splits'][d]['oracle_upper'] for d in DATASETS])
    macro_transfer = 100.0 * (macro_cross - macro_in) / macro_in
    print('-' * 80)
    print(f'{"Macro (5)":<12} {macro_in:>12.4f} {macro_cross:>14.4f}         '
          f'{macro_fixed:>11.4f} {macro_oracle:>9.4f} {macro_transfer:>+12.2f}%')

    results['macro'] = {
        'in_corpus_gbm_50_50': float(macro_in),
        'cross_corpus_gbm_loco': float(macro_cross),
        'fixed_lambda_0.3': float(macro_fixed),
        'oracle_upper': float(macro_oracle),
        'transfer_pct_vs_in_corpus': float(macro_transfer),
    }

    out_path = RESULTS_DIR / 'gbm_cross_corpus_loco_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\n[LOCO] Saved: {out_path}')


if __name__ == '__main__':
    main()
