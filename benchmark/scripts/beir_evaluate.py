#!/usr/bin/env python3
"""
BEIR Benchmark 评估 — 计算 NDCG@10, MRR@10, Recall@100
用法: python beir_evaluate.py --dataset scifact --results beir_data/scifact/results.jsonl
"""

import json, sys, argparse, math
from pathlib import Path
from collections import defaultdict

def load_qrels(qrels_path):
    """加载 qrels (TSV)"""
    qrels = {}
    with open(qrels_path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(parts[2])
                if score > 0:  # 只保留正相关
                    if qid not in qrels: qrels[qid] = {}
                    qrels[qid][did] = score
    return qrels

def load_results(results_path):
    """加载 reranking 结果 (JSONL: {query_id, method, rankings: [{doc_id, score}]})"""
    results = defaultdict(dict)  # {method: {qid: [(doc_id, score)]}}
    with open(results_path) as f:
        for line in f:
            obj = json.loads(line)
            method = obj["method"]
            qid = obj["query_id"]
            rankings = [(r["doc_id"], r["score"]) for r in obj["rankings"]]
            results[method][qid] = rankings
    return dict(results)

def dcg_at_k(relevances, k):
    """DCG@k"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg

def ndcg_at_k(ranked_doc_ids, qrel, k):
    """NDCG@k for a single query"""
    relevances = [qrel.get(did, 0) for did in ranked_doc_ids[:k]]
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG
    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_rels, k)
    
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_doc_ids, qrel, k):
    """MRR@k (Mean Reciprocal Rank)"""
    for i, did in enumerate(ranked_doc_ids[:k]):
        if qrel.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked_doc_ids, qrel, k):
    """Recall@k"""
    relevant = set(qrel.keys())
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for did in ranked_doc_ids[:k] if did in relevant)
    return retrieved_relevant / len(relevant)

def evaluate_method(method_results, qrels, ks=[10]):
    """评估单个方法"""
    metrics = {}
    
    for k in ks:
        ndcgs, mrrs, recalls = [], [], []
        
        for qid, rankings in method_results.items():
            if qid not in qrels:
                continue
            
            qrel = qrels[qid]
            ranked_ids = [did for did, _ in rankings]
            
            ndcgs.append(ndcg_at_k(ranked_ids, qrel, k))
            mrrs.append(mrr_at_k(ranked_ids, qrel, k))
            recalls.append(recall_at_k(ranked_ids, qrel, k))
        
        n = len(ndcgs) if ndcgs else 1
        metrics[f"NDCG@{k}"] = sum(ndcgs) / n
        metrics[f"MRR@{k}"] = sum(mrrs) / n
        metrics[f"Recall@{k}"] = sum(recalls) / n
    
    metrics["num_queries"] = len([q for q in method_results if q in qrels])
    return metrics

def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark 评估")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", default="./beir_data")
    parser.add_argument("--results", default=None, help="结果文件路径")
    parser.add_argument("--ks", default="10", help="评估的 k 值，逗号分隔")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) / args.dataset
    qrels_path = data_dir / "qrels.tsv"
    results_path = args.results or str(data_dir / "results.jsonl")
    ks = [int(k) for k in args.ks.split(",")]
    
    print(f"📊 评估 {args.dataset}")
    print(f"  Qrels: {qrels_path}")
    print(f"  Results: {results_path}")
    print(f"  K values: {ks}")
    
    qrels = load_qrels(qrels_path)
    results = load_results(results_path)
    
    print(f"\n  Qrels: {len(qrels)} queries with relevance judgments")
    print(f"  Methods: {list(results.keys())}")
    
    # 评估每个方法
    print(f"\n{'='*70}")
    print(f"{'Method':<20}", end="")
    for k in ks:
        print(f"{'NDCG@'+str(k):<12}{'MRR@'+str(k):<12}{'Recall@'+str(k):<12}", end="")
    print(f"{'Queries':<8}")
    print(f"{'='*70}")
    
    all_metrics = {}
    for method, method_results in sorted(results.items()):
        metrics = evaluate_method(method_results, qrels, ks)
        all_metrics[method] = metrics
        
        print(f"{method:<20}", end="")
        for k in ks:
            print(f"{metrics[f'NDCG@{k}']:<12.4f}{metrics[f'MRR@{k}']:<12.4f}{metrics[f'Recall@{k}']:<12.4f}", end="")
        print(f"{metrics['num_queries']:<8}")
    
    print(f"{'='*70}")
    
    # 保存结果
    metrics_path = data_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 指标保存到 {metrics_path}")

if __name__ == "__main__":
    main()
