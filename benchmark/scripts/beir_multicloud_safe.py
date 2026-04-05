#!/usr/bin/env python3
"""
FiQA 多点云编码 — 内存安全版

设计：
- 逐文档从 corpus.jsonl 读文本 → 拆句 → 编码句子向量
- 从 corpus_vectors.jsonl 按需读取已有文档向量（用 sentence_index.json 做 seek）
- 输出新的 corpus_vectors_mc.jsonl（包含多点云）
- 16 并发 × batch_size=16
- 内存 < 500MB（不加载整个 10GB 文件）

用法: python beir_multicloud_safe.py --dataset fiqa --data-dir ./beir_data --workers 16
"""

import os, json, sys, time, re, argparse, gc
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_BASE = "http://192.168.31.22:3000/v1/embeddings"
MODEL = "Qwen3-Embedding-8B"
DIM = 4096
BATCH_SIZE = 16
MAX_WORKERS = 16
TEXT_LIMIT = 2000

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
                time.sleep(0.5 * (attempt + 1)); continue
            if resp.status_code == 500 and len(texts) > 1:
                mid = len(texts) // 2
                v1 = embed_batch(texts[:mid], api_key, max_retries)
                v2 = embed_batch(texts[mid:], api_key, max_retries)
                if v1 and v2: return v1 + v2
                return None
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        except:
            if attempt < max_retries - 1: time.sleep(0.5)
            else: return None
    return None

def split_sentences(text, min_len=15):
    """拆句：按句号、问号、感叹号、分号分割"""
    parts = re.split(r'(?<=[.!?;])\s+|(?<=\n)', text)
    sents = [s.strip() for s in parts if len(s.strip()) >= min_len]
    return sents[:20] if sents else [text[:500]]

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
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) / args.dataset
    api_key = load_api_key()
    workers = args.workers
    batch_size = args.batch_size
    
    corpus_file = data_dir / "corpus.jsonl"
    old_vectors = data_dir / "corpus_vectors.jsonl"
    new_vectors = data_dir / "corpus_vectors_mc.jsonl"
    
    print(f"🔑 API Key: {api_key[:8]}..." if api_key else "⚠️ No API Key")
    print(f"⚡ Workers: {workers}, Batch: {batch_size}")
    print(f"📊 初始内存: {get_mem_mb():.0f} MB")
    
    # Step 1: 构建旧向量的 seek 索引（只读 _id 和 offset，不加载 vector 数据）
    print("\n📂 构建旧向量索引...")
    old_index = {}  # {doc_id: (byte_offset, byte_length)}
    if old_vectors.exists():
        with open(old_vectors, 'rb') as f:
            offset = 0
            for raw_line in f:
                try:
                    # 只解析 _id，不解析整个大 JSON
                    # 快速提取 _id：找到 "_id" 字段
                    line_str = raw_line.decode('utf-8', errors='replace')
                    obj = json.loads(line_str)
                    old_index[str(obj["_id"])] = (offset, len(raw_line))
                except:
                    pass
                offset += len(raw_line)
    print(f"  旧向量索引: {len(old_index)} docs")
    print(f"  内存: {get_mem_mb():.0f} MB")
    
    # Step 2: 检查已完成的多点云文档（如果 new_vectors 已存在）
    done_mc_ids = set()
    if new_vectors.exists():
        with open(new_vectors) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if len(obj.get("sentences", [])) > 1:
                        done_mc_ids.add(str(obj["_id"]))
                except:
                    pass
    print(f"  已完成多点云: {len(done_mc_ids)}")
    
    # Step 3: 收集需要多点云编码的文档信息（只记 id 和文本位置）
    print("\n📂 扫描 corpus 文档...")
    docs_to_process = []  # [(doc_id, byte_offset, byte_length)]
    with open(corpus_file, 'rb') as f:
        offset = 0
        for raw_line in f:
            try:
                obj = json.loads(raw_line)
                did = str(obj["_id"])
                if did not in done_mc_ids and did in old_index:
                    docs_to_process.append((did, offset, len(raw_line)))
            except:
                pass
            offset += len(raw_line)
    
    total = len(docs_to_process)
    print(f"  需要多点云编码: {total} 文档")
    
    if total == 0:
        print("  ✅ 所有文档已有多点云数据")
        return
    
    # Step 4: 逐文档处理 — 读文本拆句 → 编码句子 → 读旧文档向量 → 写新行
    print(f"\n🔄 开始多点云编码 ({total} docs, {workers} workers)...")
    
    t0 = time.time()
    completed = 0
    failed = 0
    total_sents = 0
    
    # 将文档分成批次（每批次处理一个文档的所有句子）
    # 为了并发效率，我们把多个文档的句子混合成大批次
    DOCS_PER_GROUP = 8  # 每组处理 8 个文档
    groups = []
    for i in range(0, total, DOCS_PER_GROUP):
        groups.append(docs_to_process[i:i + DOCS_PER_GROUP])
    
    pbar = tqdm(total=len(groups), desc="Multi-cloud encoding")
    
    def process_group(group_items):
        """处理一组文档：读文本 → 拆句 → 编码 → 读旧向量 → 返回完整记录"""
        results = []
        
        # 1. 读取文档文本并拆句
        doc_sentences = {}  # {did: [sent_text, ...]}
        with open(corpus_file, 'rb') as f:
            for did, offset, length in group_items:
                f.seek(offset)
                raw = f.read(length)
                try:
                    obj = json.loads(raw)
                    text = obj.get("text", "") or ""
                    title = obj.get("title", "") or ""
                    full = f"{title}. {text}" if title else text
                    sents = split_sentences(full)
                    doc_sentences[did] = sents
                except:
                    pass
        
        if not doc_sentences:
            return None
        
        # 2. 扁平化所有句子 → 一次性编码
        all_sents = []  # [(did, sent_idx, text)]
        for did, sents in doc_sentences.items():
            for si, s in enumerate(sents):
                all_sents.append((did, si, s[:TEXT_LIMIT]))
        
        # 分 batch 编码
        sent_vectors = {}  # {did: [(sent_idx, vector)]}
        for i in range(0, len(all_sents), batch_size):
            batch = all_sents[i:i + batch_size]
            texts = [t[2] for t in batch]
            vecs = embed_batch(texts, api_key)
            if vecs:
                for j, (did, si, _) in enumerate(batch):
                    if j < len(vecs) and vecs[j]:
                        if did not in sent_vectors:
                            sent_vectors[did] = []
                        sent_vectors[did].append((si, vecs[j]))
        
        # 3. 读取旧文档向量
        with open(old_vectors, 'rb') as f:
            for did in doc_sentences:
                if did in old_index:
                    off, ln = old_index[did]
                    f.seek(off)
                    raw = f.read(ln)
                    try:
                        old_obj = json.loads(raw)
                        doc_vec = old_obj.get("vector", [])
                        
                        # 合并句子向量
                        if did in sent_vectors:
                            sv_list = sorted(sent_vectors[did], key=lambda x: x[0])
                            sentence_vecs = [v for _, v in sv_list]
                        else:
                            sentence_vecs = [doc_vec] if doc_vec else []
                        
                        new_obj = {
                            "_id": did,
                            "vector": doc_vec,
                            "sentences": sentence_vecs
                        }
                        results.append(json.dumps(new_obj, ensure_ascii=False))
                    except:
                        pass
        
        return results
    
    with open(new_vectors, "a") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            next_idx = 0
            window = min(workers * 2, len(groups))
            
            while next_idx < len(groups) and len(active) < window:
                future = pool.submit(process_group, groups[next_idx])
                active[future] = next_idx
                next_idx += 1
            
            while active:
                for future in as_completed(active):
                    idx = active.pop(future)
                    try:
                        results = future.result()
                        if results:
                            with write_lock:
                                for line in results:
                                    fout.write(line + "\n")
                                fout.flush()
                                completed += len(results)
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                    
                    pbar.update(1)
                    elapsed = time.time() - t0
                    speed = (pbar.n / elapsed) if elapsed > 0 else 0
                    eta = (len(groups) - pbar.n) / speed if speed > 0 else 0
                    mem = get_mem_mb()
                    pbar.set_postfix(
                        docs=completed, fail=failed,
                        speed=f"{speed:.1f}g/s",
                        eta=f"{eta:.0f}s",
                        mem=f"{mem:.0f}MB"
                    )
                    
                    if mem > 60000:
                        print(f"\n⚠️ 内存 {mem:.0f}MB 接近上限，GC...")
                        gc.collect()
                    
                    if next_idx < len(groups):
                        future = pool.submit(process_group, groups[next_idx])
                        active[future] = next_idx
                        next_idx += 1
                    
                    break  # 一次循环处理一个
    
    pbar.close()
    elapsed = time.time() - t0
    print(f"\n  ✅ 多点云编码完成: {completed} docs")
    print(f"  ⏱️ 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  📊 最终内存: {get_mem_mb():.0f} MB")
    
    # Step 5: 合并 — 把没有处理到的文档（已有多点云的 + 未在 old_index 中的）也写入
    # 对于已完成的文档，直接跳过
    # 为简化，benchmark 可以直接用 new_vectors 文件

if __name__ == "__main__":
    main()
