#!/bin/bash
# Master v2 — skip STAGE 1 (kill done), add HF offline env, fix HF API timeout
# Authorized by 一凡 2026-05-01 evening: "保持学术诚信 全部重做"
set -e
cd /home/amd/Shape-CFD-9070XT
mkdir -p logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[$(date)] HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[$(date)] === STAGE 1: SKIPPED (llama-server already killed at 09:06:51 by master v1) ==="

echo "[$(date)] === STAGE 2: native bf16 encode (offline, fixed) ==="
.venv/bin/python3 scripts/encode_e5_mistral_nfcorpus.py \
  --dtype bf16 --attn-impl sdpa --batch-q 1 --batch-d 2 --d-max-len 512 \
  --out-root embeddings/e5_mistral_bf16_native_control \
  > logs/e5_mistral_nfcorpus_bf16_native.log 2>&1
echo "[$(date)] bf16 encode done. meta:"
cat embeddings/e5_mistral_bf16_native_control/nfcorpus/meta.json

echo "[$(date)] === STAGE 3: native fp16 encode ==="
.venv/bin/python3 scripts/encode_e5_mistral_nfcorpus.py \
  --dtype fp16 --attn-impl sdpa --batch-q 1 --batch-d 2 --d-max-len 512 \
  --out-root embeddings/e5_mistral_fp16_native_control \
  > logs/e5_mistral_nfcorpus_fp16_native.log 2>&1
echo "[$(date)] fp16 encode done. meta:"
cat embeddings/e5_mistral_fp16_native_control/nfcorpus/meta.json

echo "[$(date)] === STAGE 4: evaluate both ==="
for variant in bf16_native_control fp16_native_control; do
  .venv/bin/python3 - <<PYEOF
import json, pyarrow.parquet as pq, numpy as np, time
from pathlib import Path
ROOT = Path("/home/amd/Shape-CFD-9070XT")
EMB = ROOT / "embeddings/e5_mistral_${variant}/nfcorpus"
DATA = ROOT / "beir_data/nfcorpus"
OUT = ROOT / "outputs/e5_mistral_nfcorpus_${variant}_eval.json"
DIM = 4096

def load_parquet_dir(d):
    out_ids, out_embs = [], []
    for f in sorted(d.glob("*.parquet")):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()
        emb_col = None
        for c in ("embedding", "emb", "vec", "vector", "mean_emb", "doc_emb"):
            if c in t.column_names:
                emb_col = c; break
        if emb_col is None:
            for c in t.column_names:
                if c != "id":
                    emb_col = c; break
        for did, blob in zip(ids, t.column(emb_col).to_pylist()):
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.shape[0] != DIM:
                arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            out_ids.append(did); out_embs.append(arr)
    return out_ids, np.array(out_embs, dtype=np.float32)

t0 = time.time()
q_ids, q_arr = load_parquet_dir(EMB / "queries")
d_ids, d_arr = load_parquet_dir(EMB / "corpus")
q_arr = q_arr / np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-12)
d_arr = d_arr / np.maximum(np.linalg.norm(d_arr, axis=1, keepdims=True), 1e-12)

qrels = {}
with open(DATA / "qrels/test.tsv") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 3:
            qrels.setdefault(p[0], {})[p[1]] = int(p[2])
q_id_to_idx = {qid: i for i, qid in enumerate(q_ids)}
test_qids = [qid for qid in qrels if qid in q_id_to_idx]

def ndcg_at_k(ranked, qrel_s, k=10):
    dcg = sum((2.0**qrel_s.get(d,0) - 1.0)/np.log2(i+2) for i, d in enumerate(ranked[:k]) if qrel_s.get(d,0)>0)
    ideal = sorted(qrel_s.values(), reverse=True)
    idcg = sum((2.0**r - 1.0)/np.log2(i+2) for i, r in enumerate(ideal[:k]) if r>0)
    return dcg / idcg if idcg > 0 else 0.0

ndcg_per_q = {}
for qid in test_qids:
    qi = q_id_to_idx[qid]
    sims = d_arr @ q_arr[qi]
    top = np.argsort(-sims)[:100]
    ndcg_per_q[qid] = ndcg_at_k([d_ids[i] for i in top], qrels[qid], 10)

avg = float(np.mean(list(ndcg_per_q.values())))
result = {
    "model": "intfloat/e5-mistral-7b-instruct",
    "dataset": "nfcorpus",
    "method": "${variant} (academic-integrity rebuild 2026-05-02 morning, HF_HUB_OFFLINE=1)",
    "n_test_queries": len(test_qids),
    "ndcg_at_10": avg,
    "ndcg_per_query": ndcg_per_q,
    "embedding_source": str(EMB),
    "embedding_meta": json.load(open(EMB / "meta.json")),
    "elapsed_sec": time.time() - t0,
    "note": "Native dtype inference rebuilt under fixed encode params (batch_q=1, batch_d=2, d_max_len=512, attn=sdpa). Encode script meta dtype hardcode bug fixed (line 170: now uses args.dtype). Replaces previous fp16_control whose dtype provenance was untraceable due to hardcoded meta string + empty shell history.",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT, "w"), indent=2)
print(f"${variant} NDCG@10 = {avg:.6f} (n={len(test_qids)})")
print(f"Saved: {OUT}")
PYEOF
done

echo "[$(date)] === STAGE 5: restart llama-server ==="
cd /home/amd
nohup llama.cpp/build/bin/llama-server --model /home/amd/models/Qwen3-8B-Q4_K_M.gguf --host 0.0.0.0 --port 8082 --ctx-size 8192 --batch-size 512 --n-gpu-layers 99 --main-gpu 0 --tensor-split 1,0 --alias qwen3-8b > /home/amd/logs/qwen3-8b-server.log 2>&1 &
disown
sleep 3
nohup llama.cpp/build/bin/llama-server --model /home/amd/models/bge-m3-f16.gguf --host 0.0.0.0 --port 8080 --embedding --pooling cls --ctx-size 8192 --batch-size 512 --n-gpu-layers 99 --main-gpu 0 --tensor-split 1,0 --alias bge-m3 > /home/amd/logs/bge-m3-server.log 2>&1 &
disown
sleep 8
echo "[$(date)] llama-server restarted; ports:"
ss -tlnp 2>/dev/null | grep -E ":(8080|8082)" | head -5
curl -s http://localhost:8080/health 2>&1 | head -1
curl -s http://localhost:8082/health 2>&1 | head -1

echo "[$(date)] === ALL DONE ==="
