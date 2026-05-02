#!/bin/bash
# Restart BGE-M3 for the 3 remaining datasets (arguana / scidocs / fiqa) using fixed script.
set +e  # don't exit on individual dataset error
cd /home/amd/Shape-CFD-9070XT
source .venv/bin/activate

DATASETS=(arguana scidocs fiqa)
LOG_DIR=/home/amd/Shape-CFD-9070XT/logs
mkdir -p "$LOG_DIR"

for ds in "${DATASETS[@]}"; do
    log="$LOG_DIR/bge_m3_${ds}_v2.log"
    echo "=== $(date) start BGE-M3 $ds ===" | tee -a "$log"
    rm -rf /home/amd/Shape-CFD-9070XT/embeddings/bge_m3/$ds
    python3 scripts/encode_bge_m3_via_api.py "$ds" 2>&1 | tee -a "$log" | tail -10
    echo "=== $(date) finished $ds ===" | tee -a "$log"
done
echo "=== ALL 3 BGE-M3 REMAINING DATASETS DONE ==="
