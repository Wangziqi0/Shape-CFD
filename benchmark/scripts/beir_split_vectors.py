#!/usr/bin/env python3
"""
预处理：从 3.6GB 的 corpus_vectors.jsonl 中提取两个小文件
1. doc_vectors.jsonl — 只有文档级向量 (~56MB)
2. sentence_index.json — 句子向量的文件偏移索引

这样 Node.js benchmark 只需加载 56MB，句子按需读取
"""

import json, sys
from pathlib import Path

def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./beir_data/scifact")
    
    big_file = data_dir / "corpus_vectors.jsonl"
    doc_vec_file = data_dir / "doc_vectors.jsonl"
    sent_index_file = data_dir / "sentence_index.json"
    
    if not big_file.exists():
        print(f"❌ {big_file} 不存在"); return
    
    print(f"📂 读取 {big_file} ({big_file.stat().st_size / 1024/1024:.1f} MB)")
    
    sent_index = {}  # {doc_id: {offset, length}}
    doc_count = 0
    multi_cloud = 0
    total_sents = 0
    
    with open(big_file, 'r') as fin, open(doc_vec_file, 'w') as fout:
        offset = 0
        for line in fin:
            try:
                obj = json.loads(line)
                did = obj["_id"]
                
                # 写文档级向量（小文件）
                doc_entry = {"_id": did, "vector": obj["vector"]}
                fout.write(json.dumps(doc_entry, ensure_ascii=False) + "\n")
                
                # 记录句子偏移（用于按需读取）
                sents = obj.get("sentences", [])
                sent_index[did] = {"offset": offset, "line_len": len(line), "n_sents": len(sents)}
                
                if len(sents) > 1:
                    multi_cloud += 1
                total_sents += len(sents)
                doc_count += 1
                
            except Exception as e:
                print(f"  ⚠️ 跳过行: {e}")
            
            offset += len(line.encode('utf-8'))
    
    # 保存句子索引
    with open(sent_index_file, 'w') as f:
        json.dump(sent_index, f)
    
    dv_size = doc_vec_file.stat().st_size / 1024 / 1024
    si_size = sent_index_file.stat().st_size / 1024 / 1024
    avg_sents = total_sents / max(doc_count, 1)
    
    print(f"\n✅ 完成！")
    print(f"  doc_vectors.jsonl: {dv_size:.1f} MB ({doc_count} 文档)")
    print(f"  sentence_index.json: {si_size:.2f} MB")
    print(f"  多点云: {multi_cloud}/{doc_count} ({multi_cloud/max(doc_count,1)*100:.1f}%)")
    print(f"  平均点云: {avg_sents:.1f} 点/文档")
    print(f"  总句子: {total_sents}")

if __name__ == "__main__":
    main()
