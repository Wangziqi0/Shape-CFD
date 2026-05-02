#!/bin/bash
# Encode 5 BEIR datasets with ColBERTv2 on 9070XT (after E5-Mistral pipeline finishes).
# ColBERTv2 BERT-110M is small (~600 MB VRAM), but we run after E5-Mistral to avoid VRAM contention.

set -e
cd /home/amd/Shape-CFD-9070XT
source .venv/bin/activate

DATASETS=(scifact arguana scidocs fiqa)
LOG_DIR=/home/amd/Shape-CFD-9070XT/logs
mkdir -p "$LOG_DIR"

# (NFCorpus already encoded earlier via encode_colbertv2_nfcorpus.py — keep it.)

for ds in "${DATASETS[@]}"; do
    log="$LOG_DIR/colbertv2_${ds}.log"
    echo "=== $(date) start ColBERTv2 $ds ===" | tee -a "$log"
    rm -rf /home/amd/Shape-CFD-9070XT/embeddings/colbertv2/$ds
    python3 scripts/encode_colbertv2_nfcorpus.py \
        --dataset "$ds" \
        --batch 64 \
        --d-max-len 180 \
        2>&1 | tee -a "$log" | tr '\r' '\n' | grep -E "queries done|corpus done|DONE|N_queries|N_corpus|VRAM|Error" | head -20
    echo "=== $(date) finished $ds ===" | tee -a "$log"
done
echo "=== ALL 4 COLBERTV2 DATASETS DONE ==="
