#!/bin/bash
# Run BGE-reranker eval on 4 remaining datasets (NFCorpus done).
set +e
cd /home/amd/Shape-CFD-9070XT
source .venv/bin/activate

DATASETS=(scifact arguana scidocs fiqa)
LOG_DIR=/home/amd/Shape-CFD-9070XT/logs

for ds in "${DATASETS[@]}"; do
    log="$LOG_DIR/bge_reranker_${ds}.log"
    echo "=== $(date) start BGE-reranker $ds ===" | tee "$log"
    python3 -u scripts/eval_bge_reranker.py "$ds" 2>&1 | tee -a "$log"
    echo "=== $(date) finished $ds ===" | tee -a "$log"
done
echo "=== ALL 4 BGE-RERANKER DATASETS DONE ==="
