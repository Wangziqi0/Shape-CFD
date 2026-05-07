#!/bin/bash
# run_pq_codebook_chain.sh — Rust standalone PQ codebook training, 5 BEIR corpora chain.
# Uses 7B13 NUMA 2 (32 cores) for one corpus at a time. Sequential.
# Total ETA: ~3-4 hours for 5 corpora (NFCorpus ~10min, SciFact ~12min, ArguAna ~20min, SCIDOCS ~65min, FiQA ~110min).
#
# Usage: nohup bash benchmark/run_pq_codebook_chain.sh > /tmp/pq_codebook_chain.log 2>&1 &

set -u
ROOT=/home/amd/HEZIMENG/Shape-CFD
BIN=${ROOT}/rust-engine/standalone_tools/train_pq_codebook/target/release/train_pq_codebook
DATA_ROOT=/home/amd/HEZIMENG/legal-assistant/beir_data
OUT_DIR=${ROOT}/benchmark/data/results/pq_codebook
mkdir -p "$OUT_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === Rust PQ codebook training chain on 7B13 NPS4 NUMA 2 ==="
echo "[$(ts)] BIN: $BIN"
echo "[$(ts)] OUT: $OUT_DIR"

# Order: smallest first to validate, largest last
CORPORA=(nfcorpus scifact arguana scidocs fiqa)

for c in "${CORPORA[@]}"; do
  out_codebook="${OUT_DIR}/${c}.bin"
  if [ -f "$out_codebook" ] && [ "$(stat -c%s "$out_codebook")" -eq 4194304 ]; then
    echo "[$(ts)] [$c] cached (4 MB), skip"
    continue
  fi

  sqlite_path="${DATA_ROOT}/${c}/token_clouds.sqlite"
  if [ ! -f "$sqlite_path" ]; then
    echo "[$(ts)] [$c] SKIP: no sqlite at $sqlite_path"
    continue
  fi

  size=$(du -sh "$sqlite_path" | cut -f1)
  echo "[$(ts)] [$c] starting training (sqlite size: $size)..."

  numactl --cpunodebind=2 --membind=2 \
    "$BIN" \
      --corpus "$c" \
      --token-sqlite "$sqlite_path" \
      --out-codebook "$out_codebook" \
      --numa-node 2

  if [ $? -eq 0 ]; then
    cb_size=$(du -sh "$out_codebook" | cut -f1)
    echo "[$(ts)] [$c] DONE (codebook: $cb_size)"
  else
    echo "[$(ts)] [$c] FAILED, continuing chain"
  fi
done

echo "[$(ts)] === ALL 5 CORPORA DONE ==="
ls -la "$OUT_DIR/"
