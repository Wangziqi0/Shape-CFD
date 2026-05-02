#!/bin/bash
# Encode 6 BEIR datasets via BGE-M3 llama-server (port 8080).
# Fast: ~300 q/s via API. Sequential. Full 6 BEIR ~ 30-40 min.

set -e
cd /home/amd/Shape-CFD-9070XT
source .venv/bin/activate

DATASETS=(nfcorpus scifact arguana scidocs fiqa)
LOG_DIR=/home/amd/Shape-CFD-9070XT/logs
mkdir -p "$LOG_DIR"

for ds in "${DATASETS[@]}"; do
    log="$LOG_DIR/bge_m3_${ds}.log"
    echo "=== $(date) start BGE-M3 $ds ===" | tee -a "$log"
    rm -rf /home/amd/Shape-CFD-9070XT/embeddings/bge_m3/$ds
    python3 scripts/encode_bge_m3_via_api.py "$ds" 2>&1 | tee -a "$log" | tail -5
    echo "=== $(date) finished $ds ===" | tee -a "$log"
done
echo "=== ALL 5 BGE-M3 DATASETS DONE ==="
