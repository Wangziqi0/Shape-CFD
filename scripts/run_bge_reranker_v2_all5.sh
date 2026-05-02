#!/bin/bash
# Run BGE-reranker v2 eval on all 5 datasets serially.
set +e
cd /home/amd/Shape-CFD-9070XT
source .venv/bin/activate

DATASETS=(nfcorpus scifact arguana scidocs fiqa)
LOG_DIR=/home/amd/logs

for ds in "${DATASETS[@]}"; do
    log="$LOG_DIR/bge_rerank_v2_${ds}.log"
    echo "=== $(date) start BGE-reranker v2 $ds ===" | tee "$log"
    python3 -u scripts/eval_bge_reranker_v2.py "$ds" 2>&1 | tee -a "$log"
    echo "=== $(date) finished $ds ===" | tee -a "$log"
done
echo "=== ALL 5 BGE-RERANKER V2 DATASETS DONE ===" | tee -a "$LOG_DIR/bge_rerank_v2_ALL_DONE.flag"
