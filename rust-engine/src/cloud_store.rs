// law-vexus/src/cloud_store.rs
// 文档点云内存存储
//
// 从 SQLite chunks 表加载所有法条分块向量，按 file_id 分组为"文档点云"，
// 并预计算 PQ 子空间范数缓存，供 PQ-Chamfer 距离计算使用。

use rusqlite::Connection;
use std::collections::HashMap;

use crate::pq_chamfer::{self, PqNormCache, FULL_DIM, NUM_SUBSPACES};

/// 单个文档的点云（同一 file_id 下所有 chunk 向量的集合）
pub struct DocumentCloud {
    pub doc_id: u32,
    /// 所有句子向量连续存储，长度 = n_sentences * FULL_DIM
    pub vectors: Vec<f32>,
    /// 句子数量
    pub n_sentences: usize,
    /// 预计算的每个句子的子空间范数，长度 = n_sentences
    pub norm_caches: Vec<PqNormCache>,
}

/// 全库文档点云存储
pub struct CloudStore {
    /// 所有文档点云
    pub documents: Vec<DocumentCloud>,
    /// doc_id -> documents 中的索引
    pub id_map: HashMap<u32, usize>,
    /// 文档总数
    pub total_docs: usize,
    /// 向量总数（所有文档的句子数之和）
    pub total_vectors: usize,
}

// ─── 内部工具 ────────────────────────────────────────────────────────────

/// 计算单个向量的 PQ 子空间范数缓存
/// 将 4096 维向量切分为 64 个 64 维子空间，分别计算 L2 范数
fn compute_pq_norms(vector: &[f32]) -> PqNormCache {
    pq_chamfer::precompute_norms(vector)
}

/// 将 SQLite BLOB 字节转为 f32 向量（native endian）
/// 与 residual.rs 中的转换逻辑一致
fn blob_to_f32_vec(bytes: &[u8], dim: usize) -> Option<Vec<f32>> {
    if bytes.len() != dim * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

// ─── DocumentCloud 实现 ──────────────────────────────────────────────────

impl DocumentCloud {
    /// 获取第 i 个句子向量的切片 (长度 = FULL_DIM)
    pub fn sentence(&self, i: usize) -> &[f32] {
        let start = i * FULL_DIM;
        &self.vectors[start..start + FULL_DIM]
    }

    /// 获取所有句子向量的切片引用列表
    pub fn as_slice_refs(&self) -> Vec<&[f32]> {
        (0..self.n_sentences)
            .map(|i| self.sentence(i))
            .collect()
    }
}

// ─── CloudStore 实现 ─────────────────────────────────────────────────────

impl CloudStore {
    /// 创建空的 CloudStore
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            id_map: HashMap::new(),
            total_docs: 0,
            total_vectors: 0,
        }
    }

    /// 从 SQLite 数据库加载全部文档点云
    ///
    /// 读取 chunks 表中所有含向量的行，按 file_id 分组，每组构成一个 DocumentCloud。
    /// 加载完成后预计算所有句子的 PQ 子空间范数缓存。
    pub fn load_from_sqlite(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("打开数据库失败: {}", e))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, file_id, vector FROM chunks \
                 WHERE vector IS NOT NULL \
                 ORDER BY file_id, id",
            )
            .map_err(|e| format!("准备查询失败: {}", e))?;

        let rows = stmt
            .query_map([], |row| -> rusqlite::Result<(i64, i64, Vec<u8>)> {
                Ok((
                    row.get::<_, i64>(0)?,   // id (未使用，但保持排序)
                    row.get::<_, i64>(1)?,   // file_id
                    row.get::<_, Vec<u8>>(2)?, // vector BLOB
                ))
            })
            .map_err(|e| format!("执行查询失败: {}", e))?;

        // 按 file_id 分组构建文档点云
        let mut store = CloudStore::new();
        let mut current_file_id: Option<i64> = None;
        let mut current_vectors: Vec<f32> = Vec::new();
        let mut current_count: usize = 0;

        // 闭包：将当前缓冲区刷新为一个 DocumentCloud
        let flush = |file_id: i64, vectors: &mut Vec<f32>, count: &mut usize, store: &mut CloudStore| {
            if *count == 0 {
                return;
            }
            // 预计算每个句子的 PQ 范数缓存
            let mut norm_caches = Vec::with_capacity(*count);
            for i in 0..*count {
                let start = i * FULL_DIM;
                let sentence = &vectors[start..start + FULL_DIM];
                norm_caches.push(compute_pq_norms(sentence));
            }

            let doc_id = file_id as u32;
            let idx = store.documents.len();
            store.documents.push(DocumentCloud {
                doc_id,
                vectors: std::mem::take(vectors),
                n_sentences: *count,
                norm_caches,
            });
            store.id_map.insert(doc_id, idx);
            store.total_vectors += *count;
            *count = 0;
        };

        for row_result in rows {
            let row: (i64, i64, Vec<u8>) = row_result
                .map_err(|e| format!("读取行失败: {}", e))?;
            let (_id, file_id, blob) = row;

            // 转换 BLOB -> f32 向量，维度不匹配则跳过
            let vec = match blob_to_f32_vec(&blob, FULL_DIM) {
                Some(v) => v,
                None => continue,
            };

            // 如果 file_id 变了，先刷新上一个文档
            if current_file_id != Some(file_id) {
                if let Some(prev_fid) = current_file_id {
                    flush(prev_fid, &mut current_vectors, &mut current_count, &mut store);
                }
                current_file_id = Some(file_id);
            }

            current_vectors.extend_from_slice(&vec);
            current_count += 1;
        }

        // 刷新最后一个文档
        if let Some(fid) = current_file_id {
            flush(fid, &mut current_vectors, &mut current_count, &mut store);
        }

        store.total_docs = store.documents.len();

        Ok(store)
    }

    /// 获取指定文档的点云引用
    pub fn get_cloud(&self, doc_id: u32) -> Option<&DocumentCloud> {
        self.id_map.get(&doc_id).map(|&idx| &self.documents[idx])
    }

    /// 文档总数
    pub fn doc_count(&self) -> usize {
        self.total_docs
    }

    /// 估算内存占用（字节）
    /// 包括：向量数据 + PQ 范数缓存
    pub fn memory_usage(&self) -> usize {
        let mut total = 0usize;
        for doc in &self.documents {
            // 向量数据: n_sentences * FULL_DIM * 4 字节
            total += doc.vectors.len() * std::mem::size_of::<f32>();
            // PQ 范数缓存: n_sentences * NUM_SUBSPACES * 4 字节
            total += doc.norm_caches.len() * NUM_SUBSPACES * std::mem::size_of::<f32>();
        }
        total
    }
}

// ─── 测试 ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建内存数据库并插入测试数据
    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();

        // 创建 chunks 表
        conn.execute(
            "CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                chunk_text TEXT,
                vector BLOB
            )",
            [],
        )
        .unwrap();

        conn
    }

    /// 生成一个 FULL_DIM 维的测试向量 BLOB
    /// 每个分量 = base + index * 0.001（方便验证范数）
    fn make_test_blob(base: f32) -> Vec<u8> {
        let mut vec = Vec::with_capacity(FULL_DIM);
        for i in 0..FULL_DIM {
            vec.push(base + i as f32 * 0.001);
        }
        vec.iter().flat_map(|f| f.to_ne_bytes()).collect()
    }

    #[test]
    fn test_compute_pq_norms() {
        // 构造全 1.0 的向量，每个子空间 L2 范数 = sqrt(64) = 8.0
        let vector = vec![1.0f32; FULL_DIM];
        let cache = compute_pq_norms(&vector);
        let expected = (crate::pq_chamfer::SUB_DIM as f32).sqrt(); // 8.0
        for s in 0..NUM_SUBSPACES {
            assert!(
                (cache.norms[s] - expected).abs() < 1e-4,
                "子空间 {} 范数错误: {} != {}",
                s, cache.norms[s], expected
            );
        }
    }

    #[test]
    fn test_blob_to_f32_vec() {
        let original: Vec<f32> = (0..FULL_DIM).map(|i| i as f32 * 0.1).collect();
        let blob: Vec<u8> = original.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let result = blob_to_f32_vec(&blob, FULL_DIM).unwrap();
        assert_eq!(result.len(), FULL_DIM);
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 0.1).abs() < 1e-6);

        // 错误长度返回 None
        assert!(blob_to_f32_vec(&blob[..100], FULL_DIM).is_none());
    }

    #[test]
    fn test_document_cloud_access() {
        let vectors: Vec<f32> = (0..FULL_DIM * 3)
            .map(|i| i as f32)
            .collect();
        let norm_caches: Vec<PqNormCache> = (0..3)
            .map(|i| compute_pq_norms(&vectors[i * FULL_DIM..(i + 1) * FULL_DIM]))
            .collect();

        let cloud = DocumentCloud {
            doc_id: 42,
            vectors,
            n_sentences: 3,
            norm_caches,
        };

        // sentence() 切片正确性
        assert_eq!(cloud.sentence(0).len(), FULL_DIM);
        assert_eq!(cloud.sentence(0)[0], 0.0);
        assert_eq!(cloud.sentence(1)[0], FULL_DIM as f32);
        assert_eq!(cloud.sentence(2)[0], (FULL_DIM * 2) as f32);

        // as_slice_refs() 数量正确
        let refs = cloud.as_slice_refs();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[1][0], FULL_DIM as f32);
    }

    #[test]
    fn test_load_from_sqlite_in_memory() {
        // rusqlite in-memory 数据库无法通过路径 load_from_sqlite 打开，
        // 所以我们手动构建并测试核心逻辑。
        // 这里测试 load 的完整路径：先写入临时文件再加载。
        let tmp_dir = std::env::temp_dir();
        let db_path = tmp_dir.join("cloud_store_test.db");
        let db_path_str = db_path.to_str().unwrap();

        // 清理可能残留的旧文件
        let _ = std::fs::remove_file(&db_path);

        // 建库并插入测试数据
        {
            let conn = Connection::open(db_path_str).unwrap();
            conn.execute(
                "CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    chunk_text TEXT,
                    vector BLOB
                )",
                [],
            )
            .unwrap();

            // 文档 1 (file_id=10): 2 个 chunk
            conn.execute(
                "INSERT INTO chunks (id, file_id, chunk_text, vector) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![1, 10, "第一条", make_test_blob(0.1)],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO chunks (id, file_id, chunk_text, vector) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![2, 10, "第二条", make_test_blob(0.2)],
            )
            .unwrap();

            // 文档 2 (file_id=20): 1 个 chunk
            conn.execute(
                "INSERT INTO chunks (id, file_id, chunk_text, vector) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![3, 20, "第三条", make_test_blob(0.5)],
            )
            .unwrap();

            // 一行 vector 为 NULL，应被跳过
            conn.execute(
                "INSERT INTO chunks (id, file_id, chunk_text, vector) VALUES (?1, ?2, ?3, NULL)",
                rusqlite::params![4, 20, "空向量"],
            )
            .unwrap();
        }

        // 加载
        let store = CloudStore::load_from_sqlite(db_path_str).unwrap();

        // 验证文档数和向量数
        assert_eq!(store.doc_count(), 2, "应有 2 个文档");
        assert_eq!(store.total_vectors, 3, "应有 3 个向量（NULL 被跳过）");

        // 验证文档 10
        let doc10 = store.get_cloud(10).expect("文档 10 应存在");
        assert_eq!(doc10.n_sentences, 2);
        assert_eq!(doc10.vectors.len(), FULL_DIM * 2);
        assert_eq!(doc10.norm_caches.len(), 2);

        // 验证文档 20
        let doc20 = store.get_cloud(20).expect("文档 20 应存在");
        assert_eq!(doc20.n_sentences, 1);

        // 验证不存在的文档
        assert!(store.get_cloud(99).is_none());

        // 验证内存占用 > 0
        let mem = store.memory_usage();
        // 3 个向量 * 4096 * 4B = 49152B 向量数据
        // 3 个缓存 * 64 * 4B = 768B 范数缓存
        let expected = 3 * FULL_DIM * 4 + 3 * NUM_SUBSPACES * 4;
        assert_eq!(mem, expected, "内存占用应为 {}", expected);

        // 清理临时文件
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_empty_database() {
        let tmp_dir = std::env::temp_dir();
        let db_path = tmp_dir.join("cloud_store_empty_test.db");
        let db_path_str = db_path.to_str().unwrap();
        let _ = std::fs::remove_file(&db_path);

        {
            let conn = Connection::open(db_path_str).unwrap();
            conn.execute(
                "CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    chunk_text TEXT,
                    vector BLOB
                )",
                [],
            )
            .unwrap();
        }

        let store = CloudStore::load_from_sqlite(db_path_str).unwrap();
        assert_eq!(store.doc_count(), 0);
        assert_eq!(store.total_vectors, 0);
        assert_eq!(store.memory_usage(), 0);

        let _ = std::fs::remove_file(&db_path);
    }
}
