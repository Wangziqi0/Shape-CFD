#!/bin/bash
# rerank_chain.sh — overnight detached chain
# Phase A: build first-stage top-20 candidates for 5 BEIR corpora (BGE-M3 v2)
# Phase B: setwise rerank on 5 corpora (Qwen3-8B-Q4 9070XT 8082)
# Phase C: pairwise rerank on 2 smallest corpora (NFCorpus + SciFact)
#
# Usage: nohup bash benchmark/rerank_chain.sh > /tmp/rerank_chain.log 2>&1 &
#
# Prerequisites:
#   - 9070XT 8080 BGE-M3 server healthy
#   - 9070XT 8082 Qwen3-8B-Q4 server healthy
#   - BEIR raw data at /home/amd/HEZIMENG/legal-assistant/beir_data/<corpus>/

set -e
ROOT=/home/amd/HEZIMENG/Shape-CFD
cd "$ROOT"
mkdir -p benchmark/data/results/first_stage_bge_m3_v2
mkdir -p benchmark/data/results/llm_rerank_pairwise_setwise
mkdir -p logs/rerank_chain

ts() { date '+%Y-%m-%d %H:%M:%S'; }

CORPORA_ALL=(nfcorpus scifact arguana scidocs fiqa)
CORPORA_PAIRWISE=(nfcorpus scifact)  # smallest 2 only (pairwise expensive)

LOG=logs/rerank_chain/chain_$(date +%Y%m%d_%H%M%S).log
echo "[$(ts)] === Phase A: build first-stage candidates ===" | tee -a "$LOG"

for c in "${CORPORA_ALL[@]}"; do
  out=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [[ -f "$out" ]]; then
    echo "[$(ts)] [A/$c] cached, skipping ($(stat -c%s "$out") bytes)" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [A/$c] building first-stage..." | tee -a "$LOG"
  python3 benchmark/build_first_stage_for_rerank.py \
    --corpus "$c" \
    --top-k 20 \
    --batch-size 32 \
    --llama-url http://192.168.31.22:8080 \
    --out-dir benchmark/data/results/first_stage_bge_m3_v2 \
    >> "$LOG" 2>&1
  echo "[$(ts)] [A/$c] done" | tee -a "$LOG"
done

echo "[$(ts)] === Phase B: setwise rerank (5 corpora) ===" | tee -a "$LOG"

for c in "${CORPORA_ALL[@]}"; do
  out=benchmark/data/results/llm_rerank_pairwise_setwise/setwise_${c}_qwen3_8b.json
  if [[ -f "$out" ]]; then
    echo "[$(ts)] [B/$c] cached, skipping" | tee -a "$LOG"
    continue
  fi
  qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/${c}/qrels/test.tsv"
  if [[ ! -f "$qrels" ]]; then
    qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/${c}/qrels.test.tsv"
  fi
  first_stage=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [[ ! -f "$first_stage" ]]; then
    echo "[$(ts)] [B/$c] SKIP: no first-stage at $first_stage" | tee -a "$LOG"
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

echo "[$(ts)] === Phase C: pairwise rerank (2 corpora: nfcorpus + scifact) ===" | tee -a "$LOG"

for c in "${CORPORA_PAIRWISE[@]}"; do
  out=benchmark/data/results/llm_rerank_pairwise_setwise/pairwise_${c}_qwen3_8b.json
  if [[ -f "$out" ]]; then
    echo "[$(ts)] [C/$c] cached, skipping" | tee -a "$LOG"
    continue
  fi
  qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/${c}/qrels/test.tsv"
  if [[ ! -f "$qrels" ]]; then
    qrels="/home/amd/HEZIMENG/legal-assistant/beir_data/${c}/qrels.test.tsv"
  fi
  first_stage=benchmark/data/results/first_stage_bge_m3_v2/${c}_top20.json
  if [[ ! -f "$first_stage" ]]; then
    echo "[$(ts)] [C/$c] SKIP: no first-stage" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] [C/$c] pairwise rerank (budget=200)..." | tee -a "$LOG"
  python3 benchmark/llm_rerank_pairwise_setwise.py \
    --mode pairwise \
    --corpus "$c" \
    --top-k 20 --pair-budget 200 \
    --first-stage "$first_stage" \
    --qrels "$qrels" \
    --llama-url http://192.168.31.22:8082 \
    --out "$out" \
    >> "$LOG" 2>&1
  echo "[$(ts)] [C/$c] done" | tee -a "$LOG"
done

echo "[$(ts)] === ALL DONE ===" | tee -a "$LOG"
echo "[$(ts)] Output dir: benchmark/data/results/llm_rerank_pairwise_setwise/" | tee -a "$LOG"
ls -la benchmark/data/results/llm_rerank_pairwise_setwise/ | tee -a "$LOG"
