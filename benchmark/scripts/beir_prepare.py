#!/usr/bin/env python3
"""
BEIR Benchmark 数据准备 — 下载数据集 + Qwen3 并发编码向量
用法: python beir_prepare.py --dataset scifact --skip-sentences
"""

import os, json, sys, time, argparse, asyncio
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 配置 ──
API_BASE = "http://192.168.31.22:3000/v1/embeddings"
MODEL = "Qwen3-Embedding-8B"
DIM = 4096
BATCH_SIZE = 8
CONCURRENCY = 6  # 并发 worker 数

# API Key
def load_api_key():
    env_path = Path(__file__).parent / ".env"
    local_key = None
    sf_key = None
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("LOCAL_EMBED_KEY="):
                local_key = line.split("=", 1)[1].strip()
            elif line.startswith("SILICONFLOW_API_KEY="):
                sf_key = line.split("=", 1)[1].strip()
    return local_key or os.environ.get("LOCAL_EMBED_KEY") or sf_key or os.environ.get("SILICONFLOW_API_KEY")

# ── Embedding API 调用 ──
def embed_batch(texts, api_key, max_retries=5):
    """调用 Qwen3 Embedding API"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    body = {
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
        "dimensions": DIM,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_BASE, json=body, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if resp.status_code >= 500:
                time.sleep(1 + attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [d["embedding"] for d in sorted_data]
        except requests.exceptions.HTTPError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
    return None

# ── 下载 BEIR 数据集 ──
def download_beir_dataset(dataset_name, output_dir):
    """用 BEIR 官方 URL 下载数据集"""
    import zipfile, io
    
    data_dir = Path(output_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels.tsv"
    
    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        print(f"  📂 已有本地数据，直接加载...")
        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                obj = json.loads(line)
                corpus[obj["_id"]] = {"text": obj["text"], "title": obj.get("title", "")}
        queries = {}
        with open(queries_path) as f:
            for line in f:
                obj = json.loads(line)
                queries[obj["_id"]] = obj["text"]
        qrels = {}
        with open(qrels_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], int(parts[2])
                    if qid not in qrels: qrels[qid] = {}
                    qrels[qid][did] = score
        print(f"  Corpus: {len(corpus)}, Queries: {len(queries)}, Qrels: {sum(len(v) for v in qrels.values())}")
        return corpus, queries, qrels
    
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    print(f"📥 下载 {url} ...")
    
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    
    total_size = int(resp.headers.get('content-length', 0))
    content = io.BytesIO()
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Download") as pbar:
        for chunk in resp.iter_content(8192):
            content.write(chunk)
            pbar.update(len(chunk))
    content.seek(0)
    
    print("📦 解压中...")
    with zipfile.ZipFile(content) as zf:
        zf.extractall(data_dir.parent)
    
    beir_dir = data_dir
    
    corpus = {}
    raw_corpus = beir_dir / "corpus.jsonl"
    if raw_corpus.exists():
        with open(raw_corpus) as f:
            for line in f:
                obj = json.loads(line)
                did = str(obj["_id"])
                text = obj.get("text", "") or ""
                title = obj.get("title", "") or ""
                full_text = f"{title}. {text}" if title else text
                corpus[did] = {"text": full_text, "title": title}
    
    queries = {}
    raw_queries = beir_dir / "queries.jsonl"
    if raw_queries.exists():
        with open(raw_queries) as f:
            for line in f:
                obj = json.loads(line)
                queries[str(obj["_id"])] = obj["text"]
    
    qrels = {}
    qrels_dir = beir_dir / "qrels"
    qrels_file = qrels_dir / "test.tsv"
    if not qrels_file.exists():
        qrels_file = qrels_dir / "dev.tsv"
    
    if qrels_file.exists():
        with open(qrels_file) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], int(parts[2])
                    if qid not in qrels: qrels[qid] = {}
                    qrels[qid][did] = score
    
    with open(corpus_path, "w") as f:
        for did, doc in corpus.items():
            f.write(json.dumps({"_id": did, **doc}, ensure_ascii=False) + "\n")
    with open(queries_path, "w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
    with open(qrels_path, "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, docs in qrels.items():
            for did, score in docs.items():
                f.write(f"{qid}\t{did}\t{score}\n")
    
    print(f"  Corpus: {len(corpus)} 文档, Queries: {len(queries)} 条, Qrels: {sum(len(v) for v in qrels.values())} 条")
    return corpus, queries, qrels

# ── 并发编码 ──
def encode_corpus_concurrent(corpus, api_key, output_dir, dataset_name):
    """并发编码 corpus — 用线程池同时发 CONCURRENCY 个 API 请求"""
    data_dir = Path(output_dir) / dataset_name
    vectors_path = data_dir / "corpus_vectors.jsonl"
    
    done_ids = set()
    if vectors_path.exists():
        with open(vectors_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["_id"])
                except:
                    pass
        print(f"  已有 {len(done_ids)} 个文档向量，跳过")
    
    doc_ids = [did for did in corpus.keys() if did not in done_ids]
    if not doc_ids:
        print("  ✅ Corpus 向量已全部完成")
        return
    
    # 分 batch
    batches = []
    for i in range(0, len(doc_ids), BATCH_SIZE):
        batch_ids = doc_ids[i:i+BATCH_SIZE]
        batch_texts = [corpus[did]["text"][:2000] for did in batch_ids]
        batches.append((batch_ids, batch_texts))
    
    print(f"  编码 {len(doc_ids)} 个文档 ({len(batches)} batches × {CONCURRENCY} workers) ...")
    
    # Worker 函数
    def process_batch(batch_idx):
        batch_ids, batch_texts = batches[batch_idx]
        try:
            vectors = embed_batch(batch_texts, api_key)
            if vectors:
                results = []
                for j, did in enumerate(batch_ids):
                    if j < len(vectors) and vectors[j]:
                        results.append({
                            "_id": did,
                            "vector": vectors[j],
                            "sentences": [vectors[j]],
                        })
                return results
        except Exception as e:
            return None
        return None
    
    # 并发执行
    completed = 0
    failed = 0
    
    with open(vectors_path, "a") as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(process_batch, i): i for i in range(len(batches))}
            
            pbar = tqdm(total=len(batches), desc="Encoding corpus")
            for future in as_completed(futures):
                results = future.result()
                if results:
                    for obj in results:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    completed += len(results)
                else:
                    failed += 1
                
                f.flush()
                pbar.update(1)
                pbar.set_postfix(docs=completed, fail=failed)
            pbar.close()
    
    print(f"  ✅ 编码完成: {completed} 成功, {failed} 批次失败")

def encode_queries_concurrent(queries, api_key, output_dir, dataset_name):
    """并发编码 queries"""
    data_dir = Path(output_dir) / dataset_name
    qvec_path = data_dir / "query_vectors.jsonl"
    
    done_ids = set()
    if qvec_path.exists():
        with open(qvec_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["_id"])
                except:
                    pass
    
    remaining = {qid: text for qid, text in queries.items() if qid not in done_ids}
    if not remaining:
        print("  ✅ Query 向量已全部完成")
        return
    
    qids = list(remaining.keys())
    batches = []
    for i in range(0, len(qids), BATCH_SIZE):
        batch_ids = qids[i:i+BATCH_SIZE]
        batch_texts = [remaining[qid] for qid in batch_ids]
        batches.append((batch_ids, batch_texts))
    
    print(f"  编码 {len(remaining)} 个 queries ({len(batches)} batches) ...")
    
    def process_batch(batch_idx):
        batch_ids, batch_texts = batches[batch_idx]
        try:
            vectors = embed_batch(batch_texts, api_key)
            if vectors:
                results = []
                for j, qid in enumerate(batch_ids):
                    if j < len(vectors) and vectors[j]:
                        results.append({
                            "_id": qid,
                            "text": remaining[qid],
                            "vector": vectors[j],
                        })
                return results
        except:
            return None
        return None
    
    completed = 0
    with open(qvec_path, "a") as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(process_batch, i): i for i in range(len(batches))}
            pbar = tqdm(total=len(batches), desc="Encoding queries")
            for future in as_completed(futures):
                results = future.result()
                if results:
                    for obj in results:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    completed += len(results)
                f.flush()
                pbar.update(1)
            pbar.close()
    
    print(f"  ✅ Query 编码完成: {completed}")

# ── 主函数 ──
def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark 数据准备")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--output", default="./beir_data")
    parser.add_argument("--skip-sentences", action="store_true")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    
    
    api_key = load_api_key()
    if not api_key:
        print("❌ 未找到 API Key")
        sys.exit(1)
    
    concurrency = args.concurrency
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"⚡ 并发: {concurrency} workers, batch_size: {BATCH_SIZE}")
    
    corpus, queries, qrels = download_beir_dataset(args.dataset, args.output)
    
    print(f"\n🔄 编码 Corpus ({len(corpus)} 文档) ...")
    encode_corpus_concurrent(corpus, api_key, args.output, args.dataset)
    
    print(f"\n🔄 编码 Queries ({len(queries)} 条) ...")
    encode_queries_concurrent(queries, api_key, args.output, args.dataset)
    
    print(f"\n✅ 完成！数据保存在 {args.output}/{args.dataset}/")

if __name__ == "__main__":
    main()
