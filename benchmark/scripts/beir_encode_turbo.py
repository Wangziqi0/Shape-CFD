#!/usr/bin/env python3
"""
极速编码 BEIR 数据集 — 优化吞吐到 ~250 docs/s
优化点：batch=64, concurrency=16, 缓冲写入, 最小化 IO
"""
import os, json, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_BASE = "http://192.168.31.22:3000/v1/embeddings"
MODEL = "Qwen3-Embedding-8B"
DIM = 4096
BATCH_SIZE = 64
CONCURRENCY = 16
WRITE_BUFFER = 128  # 每 128 个结果 flush 一次
OUTPUT_DIR = "./beir_data"

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

def embed_batch(texts, api_key, max_retries=5):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"model": MODEL, "input": texts, "encoding_format": "float", "dimensions": DIM}
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_BASE, json=body, headers=headers, timeout=120)
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            if resp.status_code >= 500:
                time.sleep(1 + attempt); continue
            resp.raise_for_status()
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [d["embedding"] for d in sorted_data]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
    return None

def encode_dataset(dataset_name, api_key):
    data_dir = Path(OUTPUT_DIR) / dataset_name

    # 加载 corpus
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"  ❌ {corpus_path} 不存在"); return

    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                text = obj.get("text", "") or ""
                title = obj.get("title", "") or ""
                full = f"{title}. {text}" if title else text
                corpus[obj["_id"]] = full

    # 加载 queries
    queries = {}
    queries_path = data_dir / "queries.jsonl"
    if queries_path.exists():
        with open(queries_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    queries[obj["_id"]] = obj["text"]

    print(f"  Corpus: {len(corpus):,}, Queries: {len(queries):,}")

    # === Corpus ===
    vectors_path = data_dir / "corpus_vectors.jsonl"
    done_ids = set()
    if vectors_path.exists():
        with open(vectors_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["_id"])
                except: pass
        if done_ids:
            print(f"  续传: 已有 {len(done_ids):,}")

    remaining = [did for did in corpus if did not in done_ids]
    if remaining:
        batches = []
        for i in range(0, len(remaining), BATCH_SIZE):
            batch_ids = remaining[i:i+BATCH_SIZE]
            batch_texts = [corpus[did][:512] for did in batch_ids]
            batches.append((batch_ids, batch_texts))

        print(f"  编码 {len(remaining):,} docs ({len(batches)} batches, bs={BATCH_SIZE}, workers={CONCURRENCY})")

        completed = 0
        failed = 0
        write_lock = threading.Lock()
        write_buf = []
        t0 = time.time()

        def process_batch(idx):
            ids, texts = batches[idx]
            vecs = embed_batch(texts, api_key)
            if vecs:
                return [(did, vec) for did, vec in zip(ids, vecs)]
            return None

        with open(vectors_path, "a") as f:
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = {pool.submit(process_batch, i): i for i in range(len(batches))}
                for future in as_completed(futures):
                    results = future.result()
                    if results:
                        lines = []
                        for did, vec in results:
                            lines.append(json.dumps({"_id": did, "vector": vec, "sentences": [vec]}, ensure_ascii=False))
                        with write_lock:
                            write_buf.extend(lines)
                            completed += len(results)
                            if len(write_buf) >= WRITE_BUFFER:
                                f.write("\n".join(write_buf) + "\n")
                                f.flush()
                                write_buf.clear()
                    else:
                        failed += 1

                    # 每 500 batches 报告一次
                    done_batches = completed // BATCH_SIZE + failed
                    if done_batches % 100 == 0 or done_batches == len(batches):
                        elapsed = time.time() - t0
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(remaining) - completed) / rate if rate > 0 else 0
                        print(f"\r  {completed:,}/{len(remaining):,} docs ({rate:.0f}/s, ETA {eta/60:.1f}m, fail={failed})", end="", flush=True)

            # 刷残余
            if write_buf:
                f.write("\n".join(write_buf) + "\n")
                f.flush()

        elapsed = time.time() - t0
        print(f"\n  ✅ Corpus: {completed:,} 完成, {failed} 批失败, {elapsed/60:.1f}m ({completed/elapsed:.0f}/s)")
    else:
        print(f"  ✅ Corpus 已完成")

    # === Queries ===
    qvec_path = data_dir / "query_vectors.jsonl"
    done_qids = set()
    if qvec_path.exists():
        with open(qvec_path) as f:
            for line in f:
                try: done_qids.add(json.loads(line)["_id"])
                except: pass

    remaining_q = {qid: text for qid, text in queries.items() if qid not in done_qids}
    if remaining_q:
        qids = list(remaining_q.keys())
        qbatches = []
        for i in range(0, len(qids), BATCH_SIZE):
            bids = qids[i:i+BATCH_SIZE]
            qbatches.append((bids, [remaining_q[q] for q in bids]))

        print(f"  编码 {len(remaining_q):,} queries...")
        completed_q = 0

        def process_qbatch(idx):
            ids, texts = qbatches[idx]
            vecs = embed_batch(texts, api_key)
            if vecs:
                return [(ids[j], remaining_q[ids[j]], vecs[j]) for j in range(min(len(vecs), len(ids)))]
            return None

        with open(qvec_path, "a") as f:
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = {pool.submit(process_qbatch, i): i for i in range(len(qbatches))}
                buf = []
                for future in as_completed(futures):
                    results = future.result()
                    if results:
                        for qid, text, vec in results:
                            buf.append(json.dumps({"_id": qid, "text": text, "vector": vec}, ensure_ascii=False))
                        completed_q += len(results)
                        if len(buf) >= WRITE_BUFFER:
                            f.write("\n".join(buf) + "\n")
                            f.flush()
                            buf.clear()
                if buf:
                    f.write("\n".join(buf) + "\n")
        print(f"  ✅ Queries: {completed_q:,}")
    else:
        print(f"  ✅ Queries 已完成")

    # === clouds.sqlite ===
    clouds_path = data_dir / "clouds.sqlite"
    if not clouds_path.exists() or clouds_path.stat().st_size < 1000:
        print(f"  构建 clouds.sqlite...")
        os.system(f"node build_clouds.js {data_dir}")

def main():
    api_key = load_api_key()
    if not api_key:
        print("❌ 未找到 API Key"); sys.exit(1)

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["scidocs", "trec-covid", "webis-touche2020"]
    t_total = time.time()
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"📦 {ds}")
        print(f"{'='*60}")
        t0 = time.time()
        encode_dataset(ds, api_key)
        print(f"  ⏱ {(time.time()-t0)/60:.1f} 分钟")

    print(f"\n🎉 全部完成！总耗时 {(time.time()-t_total)/60:.1f} 分钟")

if __name__ == "__main__":
    main()
