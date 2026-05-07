#!/bin/bash
# master_rerank_chain.sh — overnight chain that fills ALL non-5060 experiments
#
# Phase A: wait current NFCorpus pairwise (PID arg) done
# Phase B: pairwise on SciFact / ArguAna / SCIDOCS / FiQA (chain after Phase A)
# Phase C: wait BGE-M3 robust encoding for trec-covid + webis-touche2020
# Phase D: build first-stage from BGE-M3 v2 cache for 2 new corpora
# Phase E: setwise on trec-covid + webis-touche2020 (5 -> 7 corpus expansion)
# Phase F: pairwise on trec-covid + webis-touche2020 (only if time permits)
#
# Total ETA: ~20-30h overnight depending on corpus sizes.
#
# Usage: nohup bash benchmark/master_rerank_chain.sh <NFCORPUS_PAIRWISE_PID> > /tmp/master_chain.log 2>&1 &

set -u
ROOT=/home/amd/HEZIMENG/Shape-CFD
cd "$ROOT"
mkdir -p logs/master_chain

NFCORPUS_PID=${1:-0}
LOG=logs/master_chain/master_chain_$(date +%Y%m%d_%H%M%S).log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === master_rerank_chain.sh starting (NFCorpus pairwise PID=$NFCORPUS_PID) ===" | tee -a "$LOG"

# ---------------------------------------------------------------
# Phase A: wait NFCorpus pairwise to complete
# ---------------------------------------------------------------
if [ "$NFCORPUS_PID" -gt 0 ]; then
  echo "[$(ts)] Phase A: waiting for NFCorpus pairwise PID $NFCORPUS_PID..." | tee -a "$LOG"
  while kill -0 "$NFCORPUS_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(ts)] Phase A: NFCorpus pairwise PID $NFCORPUS_PID done." | tee -a "$LOG"
else
  echo "[$(ts)] Phase A: skip (no PID provided)" | tee -a "$LOG"
fi

# ---------------------------------------------------------------
# Phase B: pairwise on existing 4 BEIR corpora (SciFact / ArguAna / SCIDOCS / FiQA)
# ---------------------------------------------------------------
echo "[$(ts)] === Phase B: pairwise on SciFact/ArguAna/SCIDOCS/FiQA ===" | tee -a "$LOG"
for c in scifact arguana scidocs fiqa; do
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
  echo "[$(ts)] [B/$c] pairwise rerank starting (pair-budget=200)..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode pairwise --corpus "$c" \
    --top-k 20 --pair-budget 200 \
    --first-stage "$fs" --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [B/$c] done." | tee -a "$LOG"
done

# ---------------------------------------------------------------
# Phase C: wait BGE-M3 robust encoding for both new corpora
# ---------------------------------------------------------------
echo "[$(ts)] === Phase C: wait BGE-M3 encoding for trec-covid + webis-touche2020 ===" | tee -a "$LOG"
while [ ! -f /home/amd/HEZIMENG/legal-assistant/beir_data/webis-touche2020/bge_m3_query_vectors.jsonl ]; do
  echo "[$(ts)] Phase C: still waiting webis-touche2020/bge_m3_query_vectors.jsonl..." | tee -a "$LOG"
  sleep 120
done
echo "[$(ts)] Phase C: encoding done." | tee -a "$LOG"

# ---------------------------------------------------------------
# Phase D: build first-stage for 2 new corpora
# ---------------------------------------------------------------
echo "[$(ts)] === Phase D: build first-stage for new 2 corpora ===" | tee -a "$LOG"
for c in trec-covid webis-touche2020; do
  out=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100000 ]; then
    echo "[$(ts)] [D/$c] cached, skip" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [D/$c] building first-stage from cache..." | tee -a "$LOG"
  python3 benchmark/build_first_stage_from_cache.py --corpus "$c" --top-k 20 >> "$LOG" 2>&1
  echo "[$(ts)] [D/$c] done." | tee -a "$LOG"
done

# ---------------------------------------------------------------
# Phase E: setwise on 2 new corpora
# ---------------------------------------------------------------
echo "[$(ts)] === Phase E: setwise on trec-covid + webis-touche2020 ===" | tee -a "$LOG"
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
  echo "[$(ts)] [E/$c] setwise rerank starting..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode setwise --corpus "$c" \
    --top-k 20 --set-size 4 \
    --first-stage "$fs" --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [E/$c] done." | tee -a "$LOG"
done

# ---------------------------------------------------------------
# Phase F: pairwise on 2 new corpora (best effort)
# ---------------------------------------------------------------
echo "[$(ts)] === Phase F: pairwise on trec-covid + webis-touche2020 (best effort) ===" | tee -a "$LOG"
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
  echo "[$(ts)] [F/$c] pairwise rerank (pair-budget=100, smaller for speed)..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode pairwise --corpus "$c" \
    --top-k 20 --pair-budget 100 \
    --first-stage "$fs" --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [F/$c] done." | tee -a "$LOG"
done

echo "[$(ts)] === ALL DONE ===" | tee -a "$LOG"

# Final summary
python3 -c "
import json, glob
print()
print('=== Final pairwise + setwise + 7 corpus summary ===')
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
