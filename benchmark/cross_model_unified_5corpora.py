#!/usr/bin/env python3
"""Wrapper: run cross_model_unified.py across all 5 BEIR corpora.
Aggregates per-corpus results into a single JSON for paper Section 4.7 5-corpus extension.
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path("/home/amd/HEZIMENG/Shape-CFD")
DATA_BASE = ROOT / "benchmark/data/beir_data"
SCRIPT = ROOT / "benchmark/cross_model_unified.py"
OUT_DIR = ROOT / "benchmark/data/results"
OUT = OUT_DIR / "cross_model_unified_5corpora_results.json"

CORPORA = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]

aggregate = {"alpha": 0.15, "T": 5, "K_pool": 100, "per_corpus": {}}

for corpus in CORPORA:
    data_path = DATA_BASE / corpus
    out_path = OUT_DIR / ("cross_model_unified_" + corpus + "_results.json")
    print("\n========== " + corpus.upper() + " ==========")
    cmd = [
        sys.executable, str(SCRIPT),
        "--data", str(data_path),
        "--alpha", "0.15",
        "--T", "5",
        "--K_pool", "100",
        "--out", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        with open(out_path) as f:
            aggregate["per_corpus"][corpus] = json.load(f)
    except subprocess.CalledProcessError as e:
        aggregate["per_corpus"][corpus] = {"error": str(e)}
    except FileNotFoundError as e:
        aggregate["per_corpus"][corpus] = {"error": "output JSON not produced: " + str(e)}

with open(OUT, "w") as f:
    json.dump(aggregate, f, indent=2)
print("\n=== aggregate -> " + str(OUT) + " ===")
for corpus_, res in aggregate["per_corpus"].items():
    if "error" in res:
        print("  " + corpus_ + ": ERROR " + res["error"])
        continue
    models = res.get("models", {})
    for m, v in models.items():
        line = "  " + corpus_ + " " + m + ": "
        line += "cos=" + format(v.get("cosine_ndcg10", 0), ".4f")
        line += " lap=" + format(v.get("laplacian_unified_ndcg10", 0), ".4f")
        line += " gain=" + format(v.get("relative_gain_pct", 0), "+.2f") + "%"
        print(line)
