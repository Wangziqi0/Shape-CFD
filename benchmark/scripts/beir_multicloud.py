#!/usr/bin/env python3
"""
两阶段编码：Phase1 补全文档向量 → Phase2 批量句子级多点云
比逐文档编码句子快 5-10 倍
"""

import os, json, sys, time, re, argparse
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "http://192.168.31.22:3000/v1/embeddings"
MODEL = "Qwen3-Embedding-8B"
DIM = 4096
BATCH_SIZE = 8
CONCURRENCY = 3

def load_api_key():
    env_path = Path(__file__).parent / ".env"
    local_key = sf_key = None
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("LOCAL_EMBED_KEY="): local_key = line.split("=",1)[1].strip()
            elif line.startswith("SILICONFLOW_API_KEY="): sf_key = line.split("=",1)[1].strip()
    return local_key or os.environ.get("LOCAL_EMBED_KEY") or sf_key

def embed_batch(texts, api_key, max_retries=5):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"model": MODEL, "input": texts, "encoding_format": "float", "dimensions": DIM}
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_BASE, json=body, headers=headers, timeout=60)
            if resp.status_code == 429: time.sleep(3*(attempt+1)); continue
            if resp.status_code >= 500: time.sleep(1+attempt); continue
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        except: 
            if attempt < max_retries - 1: time.sleep(1)
            else: raise
    return None

def split_sentences(text, min_len=15):
    parts = re.split(r'(?<=[.!?;])\s+|(?<=\n)', text)
    sents = [s.strip() for s in parts if len(s.strip()) >= min_len]
    return sents[:20] if sents else [text[:500]]  # 最多 20 句

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", default="./beir_data")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    
    api_key = load_api_key()
    data_dir = Path(args.data_dir) / args.dataset
    conc = args.concurrency
    
    # 加载 corpus
    corpus = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["_id"]] = obj["text"]
    
    # 加载已有向量
    vectors_path = data_dir / "corpus_vectors.jsonl"
    existing = {}  # {doc_id: {"vector": [...], "sentences": [...]}}
    if vectors_path.exists():
        with open(vectors_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing[obj["_id"]] = obj
                except: pass
    
    print(f"📊 Corpus: {len(corpus)}, 已有向量: {len(existing)}")
    
    # ═══ Phase 1: 补全文档级向量 ═══
    missing_ids = [did for did in corpus if did not in existing]
    if missing_ids:
        print(f"\n⚡ Phase 1: 补全 {len(missing_ids)} 个缺失文档向量 (并发={conc})")
        batches = []
        for i in range(0, len(missing_ids), BATCH_SIZE):
            bid = missing_ids[i:i+BATCH_SIZE]
            batches.append((bid, [corpus[d][:2000] for d in bid]))
        
        def do_batch(idx):
            ids, texts = batches[idx]
            vecs = embed_batch(texts, api_key)
            if vecs:
                return [(ids[j], vecs[j]) for j in range(len(ids)) if j < len(vecs) and vecs[j]]
            return []
        
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = {pool.submit(do_batch, i): i for i in range(len(batches))}
            pbar = tqdm(total=len(batches), desc="Phase1 doc vectors")
            for fut in as_completed(futs):
                for did, vec in fut.result():
                    existing[did] = {"_id": did, "vector": vec, "sentences": [vec]}
                pbar.update(1)
            pbar.close()
        
        print(f"  ✅ Phase 1 完成, 总向量: {len(existing)}")
    else:
        print("  ✅ 文档向量已全部完成")
    
    # ═══ Phase 2: 句子级多点云 ═══
    # 找出需要句子编码的文档（句子数 <= 1 的）
    need_sentences = []
    for did, obj in existing.items():
        sents = obj.get("sentences", [])
        if len(sents) <= 1:
            need_sentences.append(did)
    
    if not need_sentences:
        print("  ✅ 所有文档已有多点云")
    else:
        print(f"\n⚡ Phase 2: 编码 {len(need_sentences)} 个文档的句子级多点云")
        
        # 收集所有句子 → 扁平大批次
        all_tasks = []  # [(doc_id, sent_idx, sent_text)]
        for did in need_sentences:
            sents = split_sentences(corpus.get(did, ""))
            for si, s in enumerate(sents):
                all_tasks.append((did, si, s))
        
        print(f"  总句子数: {len(all_tasks)} (平均 {len(all_tasks)/max(len(need_sentences),1):.1f} 句/文档)")
        
        # 分 batch
        sent_batches = []
        for i in range(0, len(all_tasks), BATCH_SIZE):
            sent_batches.append(all_tasks[i:i+BATCH_SIZE])
        
        # 收集结果
        sent_vectors = {}  # {doc_id: [(sent_idx, vector)]}
        
        def do_sent_batch(idx):
            batch = sent_batches[idx]
            texts = [t[2] for t in batch]
            vecs = embed_batch(texts, api_key)
            if vecs:
                return [(batch[j][0], batch[j][1], vecs[j]) for j in range(len(batch)) if j < len(vecs) and vecs[j]]
            return []
        
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futs = {pool.submit(do_sent_batch, i): i for i in range(len(sent_batches))}
            pbar = tqdm(total=len(sent_batches), desc="Phase2 sentences")
            for fut in as_completed(futs):
                for did, si, vec in fut.result():
                    if did not in sent_vectors: sent_vectors[did] = []
                    sent_vectors[did].append((si, vec))
                pbar.update(1)
            pbar.close()
        
        # 合并句子向量到文档
        for did, sv_list in sent_vectors.items():
            sv_list.sort(key=lambda x: x[0])
            existing[did]["sentences"] = [v for _, v in sv_list]
        
        has_multi = sum(1 for obj in existing.values() if len(obj.get("sentences",[])) > 1)
        print(f"  ✅ Phase 2 完成, 多点云文档: {has_multi}/{len(existing)}")
    
    # ═══ 保存 ═══
    print(f"\n💾 保存 {len(existing)} 个文档向量...")
    with open(vectors_path, "w") as f:
        for did, obj in existing.items():
            if "_id" not in obj: obj["_id"] = did
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    fsize = vectors_path.stat().st_size / 1024 / 1024
    stats = {"total": len(existing), "multi_cloud": sum(1 for o in existing.values() if len(o.get("sentences",[])) > 1)}
    avg_pts = sum(len(o.get("sentences",[])) for o in existing.values()) / max(len(existing),1)
    print(f"  文件大小: {fsize:.1f} MB")
    print(f"  多点云: {stats['multi_cloud']}/{stats['total']} ({stats['multi_cloud']/max(stats['total'],1)*100:.1f}%)")
    print(f"  平均点云点数: {avg_pts:.1f}")
    print(f"\n✅ 完成！")

if __name__ == "__main__":
    main()
