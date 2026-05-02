#!/usr/bin/env python3
"""cross_model_per_model_tuned_5corpora.py — Agent B
Run cross_model_unified.py 5 corpus × 3 model with PER-MODEL TUNED (α, T):
  Qwen3-8B    (α=0.15, T=5)   on default *_corpus_vectors.jsonl + *_query_vectors.jsonl
  BGE-M3      (α=0.10, T=10)  on bge_m3_*
  BGE-large   (α=0.02, T=20)  on bge_large_*  (only nfcorpus has vectors)

Output: benchmark/data/results/cross_model_per_model_tuned_<corpus>_<model>.json
       + cross_model_per_model_tuned_5corpora_results.json (aggregate)

This corresponds to paper Table 9 5 corpus extension with per-model tuned configs.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/amd/HEZIMENG/Shape-CFD")
DATA_BASE = ROOT / "benchmark/data/beir_data"
SCRIPT_UNIFIED = ROOT / "benchmark/cross_model_unified.py"
OUT_DIR = ROOT / "benchmark/data/results"
OUT_AGG = OUT_DIR / "cross_model_per_model_tuned_5corpora_results.json"

CORPORA = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]

# Per-model (alpha, T) per paper Table 9
MODELS = [
    # name, alpha, T, corpus_file_template, query_file_template
    ("BGE-M3",    0.10, 10, "bge_m3_corpus_vectors.jsonl", "bge_m3_query_vectors.jsonl"),
    ("BGE-large", 0.02, 20, "bge_large_corpus_vectors.jsonl", "bge_large_query_vectors.jsonl"),
    ("Qwen3-8B",  0.15,  5, "corpus_vectors.jsonl", "query_vectors.jsonl"),
]


def run_one(model_name, alpha, T, corpus_dir, c_file, q_file, out_path):
    """Use cross_model_unified.py but force a specific (alpha, T) and a single
    model by passing data through a tmp symlinked dir. Simpler: write a small
    inline driver that reuses cross_model_unified.evaluate.
    """
    c_path = corpus_dir / c_file
    q_path = corpus_dir / q_file
    if not c_path.exists() or not q_path.exists():
        return {"skipped": True, "reason": f"missing {c_file} or {q_file}"}

    # Use the cross_model_unified.evaluate function directly
    sys.path.insert(0, str(ROOT / "benchmark"))
    import importlib
    cmu = importlib.import_module("cross_model_unified")
    qrels_path = corpus_dir / "qrels.tsv"
    if not qrels_path.exists():
        qrels_path = corpus_dir / "qrels" / "test.tsv"
    res = cmu.evaluate(model_name, c_path, q_path, qrels_path,
                       K_pool=100, alpha=alpha, T=T)
    return res


def main():
    aggregate = {"per_corpus": {}}
    for corpus in CORPORA:
        print(f"\n========== {corpus.upper()} ==========")
        cdir = DATA_BASE / corpus
        per = {}
        for (mname, alpha, T, cf, qf) in MODELS:
            print(f"  -> {mname} (α={alpha}, T={T})")
            r = run_one(mname, alpha, T, cdir, cf, qf,
                        OUT_DIR / f"cross_model_per_model_tuned_{corpus}_{mname}.json")
            per[mname] = {**{"alpha": alpha, "T": T}, **r}
            if "error" in r:
                print(f"    ERROR: {r['error']}")
            elif r.get("skipped"):
                print(f"    SKIP: {r.get('reason')}")
            else:
                print(f"    cos={r.get('cosine_ndcg10',0):.4f}  lap={r.get('laplacian_unified_ndcg10',0):.4f}  gain={r.get('relative_gain_pct',0):+.2f}%")
        aggregate["per_corpus"][corpus] = per

    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AGG, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n=== aggregate -> {OUT_AGG} ===")
    print("\n=== Summary ===")
    for corpus, per in aggregate["per_corpus"].items():
        for mname, r in per.items():
            if r.get("skipped") or "error" in r:
                continue
            print(f"  {corpus:10s} {mname:10s} (α={r['alpha']}, T={r['T']}): cos={r.get('cosine_ndcg10',0):.4f} lap={r.get('laplacian_unified_ndcg10',0):.4f} gain={r.get('relative_gain_pct',0):+.2f}%")


if __name__ == "__main__":
    main()
