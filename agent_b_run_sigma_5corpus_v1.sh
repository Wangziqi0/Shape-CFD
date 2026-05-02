#!/usr/bin/env bash
# Agent B: σ measurement 5 corpus × 3 model (Qwen3 default + BGE-M3 v1 + BGE-large nfcorpus only)
# 这一步可不等 Agent A v2 BGE-M3 — 因 Qwen3 default 不受 BGE-M3 zero rate 影响
# BGE-M3 v1 跑作 baseline, v2 来后再跑覆盖文件
# 11 tasks 并行 (256 核 cap, 每 task 用 numpy 默认线程, 单 task 内不会饱核)
set -euo pipefail
ROOT=/home/amd/HEZIMENG/Shape-CFD
SCRIPT=$ROOT/benchmark/measure_sigma_subspace.py
OUT=$ROOT/benchmark/data/results
LOG=$ROOT/logs_agent_b
mkdir -p $OUT $LOG

CORPORA=(nfcorpus scifact arguana scidocs fiqa)

# 限单 task 内 numpy 线程 (BLAS 默认会吃 256 核, 让 task 间能并行)
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

run_one() {
    local model=$1
    local corpus=$2
    local file=$3
    local M=$4
    local data=$ROOT/benchmark/data/beir_data/$corpus/$file
    local out_json=$OUT/sigma_${model}_${corpus}.json
    local log=$LOG/sigma_${model}_${corpus}.log
    if [ ! -f "$data" ]; then
        echo "SKIP $model $corpus (no $file)" | tee -a $log
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start $model $corpus M=$M" | tee -a $log
    python3 $SCRIPT --corpus_vectors $data --M $M --out $out_json --limit 2000 --n_pairs 10000 >> $log 2>&1 \
        && echo "[$(date +%H:%M:%S)] done $model $corpus" | tee -a $log \
        || echo "[$(date +%H:%M:%S)] FAIL $model $corpus" | tee -a $log
}

# Qwen3-8B token centroids 4096-d → M=64 (d_s=64)
for c in ${CORPORA[@]}; do
    run_one qwen3_8b $c corpus_vectors.jsonl 64 &
done

# BGE-M3 1024-d → M=16 (d_s=64)  (v1 buggy baseline)
for c in ${CORPORA[@]}; do
    run_one bge_m3_v1 $c bge_m3_corpus_vectors.jsonl 16 &
done

# BGE-large 1024-d nfcorpus only (其他 corpus 无 vectors)
run_one bge_large nfcorpus bge_large_corpus_vectors.jsonl 16 &

wait
echo "[$(date +%H:%M:%S)] all sigma tasks done"
ls -la $OUT/sigma_*.json
