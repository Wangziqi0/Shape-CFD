#!/usr/bin/env python3
"""
批量下载+编码 7 个 BEIR 数据集
按文档数从小到大排序，逐个处理
"""
import sys, os, time, json
from pathlib import Path

# 添加当前目录到 path
sys.path.insert(0, str(Path(__file__).parent))
from beir_prepare import download_beir_dataset, encode_corpus_concurrent, encode_queries_concurrent, load_api_key

DATASETS = [
    # (name, approx_docs, beir_url_name)
    ("arguana", 8_674, "arguana"),
    ("scidocs", 25_657, "scidocs"),
    ("trec-covid", 171_332, "trec-covid"),
    ("webis-touche2020", 382_545, "webis-touche2020"),
    ("quora", 522_931, "quora"),
    ("cqadupstack", 457_199, "cqadupstack"),
    ("nq", 2_681_468, "nq"),
]

OUTPUT_DIR = "./beir_data"

def main():
    api_key = load_api_key()
    if not api_key:
        print("❌ 未找到 API Key")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    # mode: "download" = 只下载, "encode" = 只编码, "all" = 全部
    # 可选第二个参数指定从哪个数据集开始 (名字或索引)
    start_from = sys.argv[2] if len(sys.argv) > 2 else None

    started = start_from is None
    for name, approx_docs, url_name in DATASETS:
        if not started:
            if start_from == name or start_from == url_name:
                started = True
            else:
                continue

        print(f"\n{'='*60}")
        print(f"📦 {name} (~{approx_docs:,} 文档)")
        print(f"{'='*60}")

        t0 = time.time()

        # 下载
        try:
            corpus, queries, qrels = download_beir_dataset(url_name, OUTPUT_DIR)
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            continue

        actual_docs = len(corpus)
        print(f"  实际文档数: {actual_docs:,}, Queries: {len(queries):,}")

        if mode == "download":
            print(f"  ⏭  仅下载模式，跳过编码")
            continue

        # 编码
        print(f"\n  🔄 编码 corpus...")
        encode_corpus_concurrent(corpus, api_key, OUTPUT_DIR, url_name)

        print(f"\n  🔄 编码 queries...")
        encode_queries_concurrent(queries, api_key, OUTPUT_DIR, url_name)

        elapsed = time.time() - t0
        print(f"\n  ✅ {name} 完成，耗时 {elapsed/60:.1f} 分钟")

        # 构建 clouds.sqlite
        vectors_path = Path(OUTPUT_DIR) / url_name / "corpus_vectors.jsonl"
        if vectors_path.exists():
            print(f"\n  🔧 构建 clouds.sqlite...")
            os.system(f"node build_clouds.js {Path(OUTPUT_DIR) / url_name}")

    print(f"\n{'='*60}")
    print("🎉 全部完成！")

if __name__ == "__main__":
    main()
