#!/usr/bin/env python3
# beir_to_sqlite.py — 将 BEIR NFCorpus 向量数据导入 SQLite
# 输出格式兼容 law-vexus/src/cloud_store.rs 的 SQL 查询：
#   SELECT id, file_id, vector FROM chunks WHERE vector IS NOT NULL ORDER BY file_id, id
#
# 用法: python3 beir_to_sqlite.py

import json
import sqlite3
import struct
import os
import sys

# ── 路径配置 ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'beir_data', 'nfcorpus')
JSONL_PATH = os.path.join(BASE_DIR, 'corpus_vectors.jsonl')
DB_PATH    = os.path.join(BASE_DIR, 'clouds.sqlite')
MAP_PATH   = os.path.join(BASE_DIR, 'id_map.json')

def main():
    # 输入文件检查
    if not os.path.exists(JSONL_PATH):
        print(f'[错误] 找不到输入文件: {JSONL_PATH}')
        sys.exit(1)

    # 删除旧数据库（幂等重建）
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f'[info] 已删除旧数据库: {DB_PATH}')

    # 建库、建表、建索引
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            vector  BLOB
        )
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id)
    ''')
    conn.commit()

    id_map   = {}   # _id 字符串 → 整数 file_id
    file_id  = 0    # 自增文档 ID，从 1 开始
    row_count = 0   # 插入行总数

    # 关闭 WAL 日志，批量写入更快
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')

    print(f'[info] 开始读取: {JSONL_PATH}')
    with open(JSONL_PATH, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f'[警告] 第 {lineno} 行 JSON 解析失败，跳过: {e}')
                continue

            doc_id_str = obj.get('_id', '')
            if not doc_id_str:
                print(f'[警告] 第 {lineno} 行缺少 _id，跳过')
                continue

            file_id += 1
            id_map[doc_id_str] = file_id

            sentences = obj.get('sentences', [])

            if len(sentences) > 1:
                # 多句子文档：每个句子分别插入一行
                for sent in sentences:
                    # Float32 小端字节序打包（与 Rust f32::from_ne_bytes 兼容）
                    blob = struct.pack(f'<{len(sent)}f', *sent)
                    conn.execute(
                        'INSERT INTO chunks (file_id, vector) VALUES (?, ?)',
                        (file_id, blob)
                    )
                    row_count += 1
            else:
                # 单句子或无 sentences：使用文档级向量
                vec = obj.get('vector', [])
                if not vec:
                    print(f'[警告] 文档 {doc_id_str} 没有 vector，跳过')
                    continue
                blob = struct.pack(f'<{len(vec)}f', *vec)
                conn.execute(
                    'INSERT INTO chunks (file_id, vector) VALUES (?, ?)',
                    (file_id, blob)
                )
                row_count += 1

            # 每 100 个文档提交一次，避免内存积压
            if file_id % 100 == 0:
                conn.commit()
                print(f'  已处理 {file_id} 个文档，{row_count} 行...', end='\r')

    conn.commit()
    conn.close()

    # 保存 _id → file_id 映射（后续 benchmark 反查）
    with open(MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, ensure_ascii=False)

    db_size = os.path.getsize(DB_PATH)
    print(f'\n[完成] {file_id} 个文档，{row_count} 行向量')
    print(f'       数据库: {DB_PATH} ({db_size / 1024 / 1024:.1f} MB)')
    print(f'       映射表: {MAP_PATH} ({len(id_map)} 条)')

if __name__ == '__main__':
    main()
