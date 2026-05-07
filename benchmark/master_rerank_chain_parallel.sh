#!/bin/bash
# master_rerank_chain_parallel.sh — like master_rerank_chain.sh but uses --workers 4
# for client-side concurrency, paired with llama-server --parallel 4 --cont-batching.

set -u
ROOT=/home/amd/HEZIMENG/Shape-CFD
cd "$ROOT"
mkdir -p logs/master_chain
LOG=logs/master_chain/master_chain_parallel_$(date +%Y%m%d_%H%M%S).log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

WORKERS=4

echo "[$(ts)] === master_rerank_chain_parallel.sh start (workers=$WORKERS) ===" | tee -a "$LOG"

CORPORA_4=(scifact arguana scidocs fiqa)

# ---------- Phase B: pairwise on 4 existing corpora ----------
echo "[$(ts)] === Phase B: pairwise on 4 existing corpora (parallel $WORKERS) ===" | tee -a "$LOG"
for c in "${CORPORA_4[@]}"; do
  out=benchmark/data/results/llm_rerank_pairwise_setwise/pairwise_${c}_qwen3_8b.json
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 10000 ]; then
    echo "[$(ts)] [B/$c] cached, skip" | tee -a "$LOG"
    continue
  fi
  qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/$c/qrels/test.tsv"
  fs="benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json"
  if [ ! -f "$fs" ]; then
    echo "[$(ts)] [B/$c] SKIP: no first-stage" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [B/$c] pairwise rerank (workers=$WORKERS, pair-budget=200)..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode pairwise --corpus "$c" \
    --top-k 20 --pair-budget 50 \
    --first-stage "$fs" --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --workers "$WORKERS" \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [B/$c] done." | tee -a "$LOG"
done

# ---------- Phase C: wait BGE-M3 robust encoding for both new corpora ----------
echo "[$(ts)] === Phase C: wait BGE-M3 encoding webis-touche2020 ===" | tee -a "$LOG"
while [ ! -f /home/amd/HEZIMENG/legal-assistant/beir_data/webis-touche2020/bge_m3_query_vectors.jsonl ]; do
  sleep 120
done
echo "[$(ts)] Phase C done" | tee -a "$LOG"

# ---------- Phase D: build first-stage for 2 new corpora ----------
echo "[$(ts)] === Phase D: build first-stage for new 2 corpora ===" | tee -a "$LOG"
for c in trec-covid webis-touche2020; do
  out=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100000 ]; then
    echo "[$(ts)] [D/$c] cached, skip" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [D/$c] building first-stage..." | tee -a "$LOG"
  python3 benchmark/build_first_stage_from_cache.py --corpus "$c" --top-k 20 >> "$LOG" 2>&1
  echo "[$(ts)] [D/$c] done." | tee -a "$LOG"
done

# ---------- Phase E: setwise on 2 new corpora ----------
echo "[$(ts)] === Phase E: setwise on 2 new corpora (parallel $WORKERS) ===" | tee -a "$LOG"
for c in trec-covid webis-touche2020; do
  out=benchmark/data/results/llm_rerank_pairwise_setwise/setwise_${c}_qwen3_8b.json
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 10000 ]; then
    echo "[$(ts)] [E/$c] cached, skip" | tee -a "$LOG"
    continue
  fi
  qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/$c/qrels/test.tsv"
  fs="benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json"
  if [ ! -f "$qrels" ] || [ ! -f "$fs" ]; then
    echo "[$(ts)] [E/$c] SKIP: missing $qrels or $fs" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [E/$c] setwise rerank (workers=$WORKERS)..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode setwise --corpus "$c" \
    --top-k 20 --set-size 4 \
    --first-stage "$fs" --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --workers "$WORKERS" \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [E/$c] done." | tee -a "$LOG"
done

# ---------- Phase F: pairwise on 2 new corpora (best effort, smaller budget for speed) ----------
echo "[$(ts)] === Phase F: pairwise on 2 new corpora (parallel $WORKERS, pair-budget=100) ===" | tee -a "$LOG"
for c in trec-covid webis-touche2020; do
  out=benchmark/data/results/llm_rerank_pairwise_setwise/pairwise_${c}_qwen3_8b.json
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 10000 ]; then
    echo "[$(ts)] [F/$c] cached, skip" | tee -a "$LOG"
    continue
  fi
  qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/$c/qrels/test.tsv"
  fs="benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json"
  if [ ! -f "$qrels" ] || [ ! -f "$fs" ]; then
    echo "[$(ts)] [F/$c] SKIP" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [F/$c] pairwise (workers=$WORKERS, budget=100)..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode pairwise --corpus "$c" \
    --top-k 20 --pair-budget 50 \
    --first-stage "$fs" --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --workers "$WORKERS" \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [F/$c] done." | tee -a "$LOG"
done

echo "[$(ts)] === ALL DONE ===" | tee -a "$LOG"

# Final summary
python3 -c "
import json, glob
print()
print('=== Final pairwise + setwise 7-corpus summary ===')
for mode in ['pairwise', 'setwise']:
    for f in sorted(glob.glob(f'benchmark/data/results/llm_rerank_pairwise_setwise/{mode}_*.json')):
        try:
            d = json.load(open(f))
            m = d['macro']
            c = f.split(f'{mode}_')[1].split('_qwen3')[0]
            print(f'  {mode}/{c}: ndcg={m[\"ndcg_mean\"]:.4f}±{m[\"ndcg_std\"]:.4f} n={m[\"n_queries\"]} fail={m[\"parse_failure_rate\"]:.3f} wall={m[\"wall_time_sec\"]:.0f}s')
        except Exception as e:
            print(f'  {f}: load fail ({e})')
" | tee -a "$LOG"
