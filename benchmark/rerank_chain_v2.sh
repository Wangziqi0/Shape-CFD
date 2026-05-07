#!/bin/bash
# rerank_chain_v2.sh — Day 0 today-evening detached overnight chain.
# Phase A: build first-stage from cached BGE-M3 v2 vectors (5 corpora, ~5 min)
# Phase B: setwise rerank on 5 corpora (Qwen3-8B-Q4 9070XT 8082, ~5-8 h)
#
# Total ETA: ~6-9 hours overnight.
#
# Usage: nohup bash benchmark/rerank_chain_v2.sh > /tmp/rerank_chain_v2.log 2>&1 &
#        echo "PID=$!" >> /tmp/rerank_chain_v2.log

set -e
ROOT=/home/amd/HEZIMENG/Shape-CFD
cd "$ROOT"
mkdir -p benchmark/data/results/first_stage_bge_m3_v2
mkdir -p benchmark/data/results/llm_rerank_pairwise_setwise
mkdir -p logs/rerank_chain

ts() { date '+%Y-%m-%d %H:%M:%S'; }

CORPORA=(nfcorpus scifact arguana scidocs fiqa)

LOG=logs/rerank_chain/chain_$(date +%Y%m%d_%H%M%S).log
echo "[$(ts)] === Phase A: build first-stage from BGE-M3 v2 cache ===" | tee -a "$LOG"

for c in "${CORPORA[@]}"; do
  out=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [[ -f "$out" ]] && [[ $(stat -c%s "$out") -gt 100000 ]]; then
    echo "[$(ts)] [A/$c] cached, skipping ($(stat -c%s "$out") bytes)" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [A/$c] building first-stage from cache..." | tee -a "$LOG"
  python3 benchmark/build_first_stage_from_cache.py \
    --corpus "$c" \
    --top-k 20 \
    >> "$LOG" 2>&1
  echo "[$(ts)] [A/$c] done" | tee -a "$LOG"
done

echo "[$(ts)] === Phase B: setwise rerank (5 corpora) ===" | tee -a "$LOG"

for c in "${CORPORA[@]}"; do
  out=benchmark/data/results/llm_rerank_pairwise_setwise/setwise_${c}_qwen3_8b.json
  if [[ -f "$out" ]] && [[ $(stat -c%s "$out") -gt 10000 ]]; then
    echo "[$(ts)] [B/$c] cached, skipping" | tee -a "$LOG"
    continue
  fi
  qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/${c}/qrels/test.tsv"
  if [[ ! -f "$qrels" ]]; then
    qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/${c}/qrels.test.tsv"
  fi
  first_stage=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [[ ! -f "$first_stage" ]]; then
    echo "[$(ts)] [B/$c] SKIP: no first-stage" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [B/$c] setwise rerank..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode setwise \
    --corpus "$c" \
    --top-k 20 --set-size 4 \
    --first-stage "$first_stage" \
    --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [B/$c] done" | tee -a "$LOG"
done

echo "[$(ts)] === ALL DONE ===" | tee -a "$LOG"
ls -la benchmark/data/results/llm_rerank_pairwise_setwise/ | tee -a "$LOG"

# Quick aggregate summary
python3 -c "
import json, glob
print()
print('=== setwise NDCG@10 summary ===')
for f in sorted(glob.glob('benchmark/data/results/llm_rerank_pairwise_setwise/setwise_*.json')):
    try:
        d = json.load(open(f))
        m = d['macro']
        c = f.split('setwise_')[1].split('_qwen3')[0]
        print(f'  {c}: ndcg={m[\"ndcg_mean\"]:.4f}±{m[\"ndcg_std\"]:.4f} n={m[\"n_queries\"]} fail_rate={m[\"parse_failure_rate\"]:.3f} wall={m[\"wall_time_sec\"]:.0f}s')
    except Exception as e:
        print(f'  {f}: load failed ({e})')
" | tee -a "$LOG"
