#!/usr/bin/env python3
"""verify_v2_zero_rate.py — Agent B
Check per corpus BGE-M3 v2 vector zero rate (vs v1 backup).
Loop 5 corpus, count zero-norm vectors in corpus + queries jsonl.
"""
import json, sys
from pathlib import Path

ROOT = Path("/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data")
CORPORA = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]


def count_zero(path):
    n = 0
    z = 0
    sum_norms = 0.0
    min_n = 10.0
    max_n = 0.0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            v = r.get("vector") or r.get("emb")
            if v is None:
                continue
            # L2 norm
            s = sum(x * x for x in v)
            norm = s ** 0.5
            n += 1
            sum_norms += norm
            if norm == 0.0:
                z += 1
            if norm < min_n:
                min_n = norm
            if norm > max_n:
                max_n = norm
    return n, z, sum_norms, min_n, max_n


def report(label, path):
    if not path.exists():
        return f"  {label}: MISSING {path}"
    n, z, sn, mn, mx = count_zero(path)
    pct = (100.0 * z / n) if n > 0 else 0.0
    avg = (sn / n) if n > 0 else 0.0
    return f"  {label}: n={n} zero={z} ({pct:.4f}%) avg_norm={avg:.4f} min={mn:.4f} max={mx:.4f}"


for c in CORPORA:
    print(f"== {c} ==")
    print(report("v2 corpus", ROOT / c / "bge_m3_corpus_vectors.jsonl"))
    print(report("v1 corpus", ROOT / c / "bge_m3_corpus_vectors_v1.jsonl"))
    print(report("v2 query ", ROOT / c / "bge_m3_query_vectors.jsonl"))
    print(report("v1 query ", ROOT / c / "bge_m3_query_vectors_v1.jsonl"))
