#!/usr/bin/env python3
"""
高并发内存安全编码脚本 — 10 分钟完成 57k 文档

设计:
- batch_size=16, 12 并发 workers → ~8 batch/s
- 滑动窗口 Future 管理（不一次提交所有 batch）
- 内存 < 64GB（实际 < 100MB）
- 断点续传
- 每 50 batch 打印内存+速度监控

用法: python beir_encode_safe.py --dataset fiqa --data-dir ./beir_data
"""

import os, json, sys, time, argparse, gc
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── 配置 ──
API_BASE = "http://192.168.31.22:3000/v1/embeddings"
MODEL = "Qwen3-Embedding-8B"
DIM = 4096
BATCH_SIZE = 16       # 大 batch 减少请求次数
MAX_WORKERS = 12      # 12 并发 → 吃满 API 带宽
WINDOW_SIZE = 24      # 滑动窗口：最多 24 个 Future 同时存在
TEXT_LIMIT = 2000     # 文本截断长度

# 写入锁
write_lock = threading.Lock()

def load_api_key():
    env_path = Path(__file__).parent / ".env"
    local_key = None
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("LOCAL_EMBED_KEY="):
                local_key = line.split("=", 1)[1].strip()
            elif line.startswith("SILICONFLOW_API_KEY=") and not local_key:
                local_key = line.split("=", 1)[1].strip()
    return local_key or os.environ.get("LOCAL_EMBED_KEY") or os.environ.get("SILICONFLOW_API_KEY")

def embed_batch(texts, api_key, max_retries=3):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": MODEL, "input": texts, "encoding_format": "float", "dimensions": DIM}
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_BASE, json=body, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 500 and len(texts) > 1:
                mid = len(texts) // 2
                v1 = embed_batch(texts[:mid], api_key, max_retries)
                v2 = embed_batch(texts[mid:], api_key, max_retries)
                if v1 and v2:
                    return v1 + v2
                return None
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                return None
    return None

def get_mem_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument("--data-dir", default="./beir_data")
    parser.add_argument("--mode", choices=["corpus", "query", "both"], default="both")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) / args.dataset
    api_key = load_api_key()
    workers = args.workers
    batch_size = args.batch_size
    
    print(f"🔑 API Key: {api_key[:8]}..." if api_key else "⚠️ No API Key")
    print(f"⚡ Workers: {workers}, Batch: {batch_size}, Window: {WINDOW_SIZE}")
    print(f"📊 初始内存: {get_mem_mb():.0f} MB")
    print(f"⚠️ 内存上限: 64 GB")
    
    if args.mode in ("both", "corpus"):
        encode_corpus(data_dir, api_key, workers, batch_size)
    if args.mode in ("both", "query"):
        encode_queries(data_dir, api_key, workers, batch_size)

def encode_corpus(data_dir, api_key, workers, batch_size):
    corpus_file = data_dir / "corpus.jsonl"
    vectors_file = data_dir / "corpus_vectors.jsonl"
    
    if not corpus_file.exists():
        print(f"❌ {corpus_file} 不存在"); return
    
    # 读取已完成 ID（只保存 set）
    done_ids = set()
    if vectors_file.exists():
        with open(vectors_file) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["_id"])
                except: pass
    
    # 扫描需要编码的文档：只记录 (id, byte_offset, byte_length)
    pending = []
    with open(corpus_file, 'rb') as f:
        offset = 0
        for raw_line in f:
            try:
                obj = json.loads(raw_line)
                did = str(obj["_id"])
                if did not in done_ids:
                    pending.append((did, offset, len(raw_line)))
            except: pass
            offset += len(raw_line)
    
    total = len(pending)
    if total == 0:
        print("  ✅ Corpus 向量已全部完成"); return
    
    batch_count = (total + batch_size - 1) // batch_size
    print(f"\n🔄 编码 Corpus: {total} 待编码 ({batch_count} batches × {workers} workers)")
    print(f"   已完成: {len(done_ids)}, 预计: ~{batch_count / (workers * 0.7):.0f}s")
    
    # 构建 batch 索引（不加载文本，只记录位置）
    batch_indices = []
    for i in range(0, total, batch_size):
        batch_indices.append(pending[i:i+batch_size])
    del pending  # 释放
    gc.collect()
    
    completed = 0
    failed = 0
    t0 = time.time()
    
    def process_batch(batch_items, corpus_path):
        """Worker: 从磁盘读取文本 → 编码 → 返回结果"""
        batch_ids = []
        batch_texts = []
        with open(corpus_path, 'rb') as f:
            for did, offset, length in batch_items:
                f.seek(offset)
                raw = f.read(length)
                try:
                    obj = json.loads(raw)
                    text = obj.get("text", "") or ""
                    title = obj.get("title", "") or ""
                    full = f"{title}. {text}" if title else text
                    batch_ids.append(did)
                    batch_texts.append(full[:TEXT_LIMIT])
                except: pass
        
        if not batch_texts:
            return None
        
        vectors = embed_batch(batch_texts, api_key)
        if vectors:
            results = []
            for j, did in enumerate(batch_ids):
                if j < len(vectors) and vectors[j]:
                    results.append(json.dumps({"_id": did, "vector": vectors[j], "sentences": [vectors[j]]}, ensure_ascii=False))
            return results
        return None
    
    # ★ 滑动窗口并发
    pbar = tqdm(total=batch_count, desc="Encoding corpus")
    
    with open(vectors_file, "a") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # 提交初始窗口
            active = {}
            next_idx = 0
            
            while next_idx < len(batch_indices) and len(active) < WINDOW_SIZE:
                future = pool.submit(process_batch, batch_indices[next_idx], corpus_file)
                active[future] = next_idx
                next_idx += 1
            
            while active:
                # 等任意一个完成
                done_futures = []
                for future in as_completed(active):
                    done_futures.append(future)
                    break  # 一次处理一个
                
                for future in done_futures:
                    idx = active.pop(future)
                    results = future.result()
                    
                    if results:
                        with write_lock:
                            for line in results:
                                fout.write(line + "\n")
                            fout.flush()
                            completed += len(results)
                    else:
                        failed += 1
                    
                    pbar.update(1)
                    elapsed = time.time() - t0
                    speed = (pbar.n / elapsed) if elapsed > 0 else 0
                    eta = (batch_count - pbar.n) / speed if speed > 0 else 0
                    mem = get_mem_mb()
                    pbar.set_postfix(
                        docs=completed, fail=failed,
                        speed=f"{speed:.1f}b/s",
                        eta=f"{eta:.0f}s",
                        mem=f"{mem:.0f}MB"
                    )
                    
                    # 内存安全检查
                    if mem > 60000:  # 60GB 软上限
                        print(f"\n⚠️ 内存 {mem:.0f}MB 接近上限，GC...")
                        gc.collect()
                    
                    # 补充窗口
                    if next_idx < len(batch_indices):
                        future = pool.submit(process_batch, batch_indices[next_idx], corpus_file)
                        active[future] = next_idx
                        next_idx += 1
    
    pbar.close()
    elapsed = time.time() - t0
    print(f"  ✅ 编码完成: {completed} 成功, {failed} 批次失败")
    print(f"  ⏱️ 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  📊 最终内存: {get_mem_mb():.0f} MB")

def encode_queries(data_dir, api_key, workers, batch_size):
    queries_file = data_dir / "queries.jsonl"
    vectors_file = data_dir / "query_vectors.jsonl"
    
    if not queries_file.exists():
        print(f"❌ {queries_file} 不存在"); return
    
    done_ids = set()
    if vectors_file.exists():
        with open(vectors_file) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["_id"])
                except: pass
    
    queries = []
    with open(queries_file) as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qid not in done_ids:
                queries.append((qid, obj["text"]))
    
    if not queries:
        print("  ✅ Query 向量已全部完成"); return
    
    print(f"\n🔄 编码 Queries: {len(queries)} 待编码")
    completed = 0
    
    with open(vectors_file, "a") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_count = (len(queries) + batch_size - 1) // batch_size
            pbar = tqdm(total=batch_count, desc="Encoding queries")
            
            def encode_q_batch(start):
                batch = queries[start:start+batch_size]
                texts = [q[1][:TEXT_LIMIT] for q in batch]
                vectors = embed_batch(texts, api_key)
                if vectors:
                    results = []
                    for j, (qid, text) in enumerate(batch):
                        if j < len(vectors) and vectors[j]:
                            results.append(json.dumps({"_id": qid, "vector": vectors[j], "text": text}, ensure_ascii=False))
                    return results
                return None
            
            futures = {}
            for i in range(0, len(queries), batch_size):
                future = pool.submit(encode_q_batch, i)
                futures[future] = i
            
            for future in as_completed(futures):
                results = future.result()
                if results:
                    for line in results:
                        fout.write(line + "\n")
                    fout.flush()
                    completed += len(results)
                pbar.update(1)
            pbar.close()
    
    print(f"  ✅ Query 编码完成: {completed}")

if __name__ == "__main__":
    main()
