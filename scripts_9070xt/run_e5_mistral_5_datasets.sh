#!/bin/bash
# Encode 5 BEIR datasets with E5-Mistral on 9070XT, working ROCm config:
# bf16 + sdpa + batch_q=4 batch_d=2 + d_max_len=512 (BEIR standard)
# (Previously fp16+eager+512 hung; 256 produced poor NDCG@10=0.086.)
# NFCorpus first then SciFact / ArguAna / SCIDOCS / FiQA. Quora skipped (522K docs).

set -e
cd /home/amd/Shape-CFD-9070XT
source .venv/bin/activate

DATASETS=(nfcorpus scifact arguana scidocs fiqa)
LOG_DIR=/home/amd/Shape-CFD-9070XT/logs
mkdir -p "$LOG_DIR"

for ds in "${DATASETS[@]}"; do
    log="$LOG_DIR/e5_mistral_${ds}_512.log"
    echo "=== $(date) start E5-Mistral $ds (d_max_len=512) ===" | tee -a "$log"
    rm -rf /home/amd/Shape-CFD-9070XT/embeddings/e5_mistral/$ds
    python3 scripts/encode_e5_mistral_nfcorpus.py \
        --dataset "$ds" \
        --batch-q 4 \
        --batch-d 2 \
        --d-max-len 512 \
        --dtype bf16 \
        --attn-impl sdpa \
        2>&1 | tee -a "$log" | tr '\r' '\n' | grep -E "queries done|corpus done|DONE|N_queries|N_corpus|VRAM|Error" | head -20
    echo "=== $(date) finished $ds ===" | tee -a "$log"
done
echo "=== ALL 5 DATASETS DONE ==="
