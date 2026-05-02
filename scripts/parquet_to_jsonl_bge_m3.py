#!/usr/bin/env python3
"""Convert 9070XT bge_m3 parquet vectors to jsonl format for 7B13 LLM-listwise pipe.
Reads bge_m3/<corpus>/queries/*.parquet + corpus/*.parquet,
writes <corpus>/bge_m3_query_vectors.jsonl + bge_m3_corpus_vectors.jsonl.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB_BASE = ROOT / "embeddings/bge_m3"
OUT_BASE = ROOT / "vectors_jsonl_for_7b13"
DIM = 1024

def parquet_to_jsonl(parquet_dir: Path, out_path: Path):
    written = 0
    with open(out_path, "w") as fout:
        for pf in sorted(parquet_dir.glob("*.parquet")):
            t = pq.read_table(pf)
            ids = t.column("id").to_pylist()
            emb_col = None
            for c in ("embedding", "emb", "vec", "vector", "mean_emb", "doc_emb", "emb_fp16"):
                if c in t.column_names:
                    emb_col = c
                    break
            if emb_col is None:
                for c in t.column_names:
                    if c != "id":
                        emb_col = c
                        break
            blobs = t.column(emb_col).to_pylist()
            for did, blob in zip(ids, blobs):
                if isinstance(blob, (bytes, bytearray)):
                    arr = np.frombuffer(blob, dtype=np.float32)
                    if arr.shape[0] != DIM:
                        arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
                    vec = arr.tolist()
                else:
                    vec = list(blob)
                fout.write(json.dumps({"_id": did, "vector": vec}) + "\n")
                written += 1
    print(f"  wrote {written} vectors -> {out_path}")

def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for corpus in ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa"]:
        cdir = EMB_BASE / corpus
        if not cdir.exists():
            print(f"skip {corpus}: dir missing")
            continue
        out_corp_dir = OUT_BASE / corpus
        out_corp_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {corpus} ==")
        parquet_to_jsonl(cdir / "queries", out_corp_dir / "bge_m3_query_vectors.jsonl")
        parquet_to_jsonl(cdir / "corpus", out_corp_dir / "bge_m3_corpus_vectors.jsonl")
    print("DONE")

if __name__ == "__main__":
    main()
