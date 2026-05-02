#!/usr/bin/env python3
"""Aggregate all baseline NDCG@10 results into a master table for paper main table update."""
import json
from pathlib import Path

OUT_DIR = Path("/home/amd/Shape-CFD-9070XT/outputs")
DATASETS = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]
PATTERNS = {
    "BGE-M3": "bge_m3_{ds}_cosine_eval.json",
    "BGE-reranker-v2-m3": "bge_reranker_{ds}_eval.json",
    "ColBERTv2": "colbertv2_{ds}_eval.json",
    "E5-Mistral (ROCm)": "e5_mistral_{ds}_cosine_eval.json",
}

table = {}
for model, pat in PATTERNS.items():
    table[model] = {}
    for ds in DATASETS:
        f = OUT_DIR / pat.format(ds=ds)
        if f.exists():
            try:
                table[model][ds] = round(json.loads(f.read_text())["ndcg_at_10"], 4)
            except Exception as e:
                table[model][ds] = f"ERR: {e}"
        else:
            table[model][ds] = None

# Print markdown table
print("| Model | " + " | ".join([d.upper() for d in DATASETS]) + " |")
print("|" + "---|" * (len(DATASETS) + 1))
for model, scores in table.items():
    row = f"| {model} | " + " | ".join([f"{scores[d]:.4f}" if isinstance(scores[d], float) else (str(scores[d]) if scores[d] else "—") for d in DATASETS]) + " |"
    print(row)

# Plus our paper numbers (from previous fusion ablation results.json)
print("\n## Our pipeline (from fusion_ablation_results.json on 7b13)")
print("Cosine baseline: NFC 0.2300 / SciFact 0.4483 / ArguAna 0.3047")
print("Token PQ-Chamfer 2-stage: NFC 0.3220 / SciFact 0.4555 / ArguAna 0.4418")
print("Best fusion λ*: NFC 0.3270 (λ=0.4) / SciFact 0.4946 (λ=0.7) / ArguAna 0.4418 (λ=0.0)")

# Save JSON
out = {"table": table, "datasets": DATASETS, "models": list(PATTERNS.keys())}
(OUT_DIR / "_master_baseline_table.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved: {OUT_DIR}/_master_baseline_table.json")
