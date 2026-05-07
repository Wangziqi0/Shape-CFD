#!/bin/bash
# rerun_scidocs_fiqa.sh — retry SCIDOCS + FiQA PQ codebook with --interleave=all
# (OOM-killed previously when --membind=2 limited to single-NUMA 64 GB RAM)

set -u
ROOT=/home/amd/HEZIMENG/Shape-CFD
BIN=${ROOT}/rust-engine/standalone_tools/train_pq_codebook/target/release/train_pq_codebook
DATA=/home/amd/HEZIMENG/legal-assistant/beir_data
OUT=${ROOT}/benchmark/data/results/pq_codebook
mkdir -p "$OUT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

for c in scidocs fiqa; do
  out_codebook="${OUT}/${c}.bin"
  if [ -f "$out_codebook" ] && [ "$(stat -c%s "$out_codebook")" -eq 4194304 ]; then
    echo "[$(ts)] [$c] cached, skip"
    continue
  fi
  size=$(du -sh "$DATA/$c/token_clouds.sqlite" | cut -f1)
  echo "[$(ts)] [$c] starting (sqlite size: $size, --interleave=all to avoid OOM)"
  numactl --interleave=all "$BIN" \
    --corpus "$c" \
    --token-sqlite "$DATA/$c/token_clouds.sqlite" \
    --out-codebook "$out_codebook" \
    --numa-node 99
  if [ $? -eq 0 ]; then
    cb_size=$(du -sh "$out_codebook" | cut -f1)
    echo "[$(ts)] [$c] DONE (codebook: $cb_size)"
  else
    echo "[$(ts)] [$c] FAILED, continuing"
  fi
done

echo "[$(ts)] === SCIDOCS + FIQA chain done ==="
ls -la "$OUT/"
