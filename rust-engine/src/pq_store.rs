//! PQ 编码存储 (V14-PQ)
//!
//! 将全库 token 点云用 PQ (Product Quantization) 编码为 u8 码本 ID，
//! 每个 token 4096d -> 64 个 u8 (64 字节)，相比原始 f32 (16384 字节) 压缩 256 倍。
//! 配合 ADC (Asymmetric Distance Computation) 做近似 Chamfer 距离计算。

use crate::cloud_store::CloudStore;
use crate::inverted_index::{l2_sq_64d, NUM_CENTROIDS};
use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM};
use rayon::prelude::*;
use rusqlite::Connection;
use std::collections::HashMap;

/// PQ 码本：64 子空间 x 256 中心 x 64 维
/// flat 布局：codebook[s * 256 * 64 + c * 64 + d]
pub struct PqCodebook {
    pub data: Vec<f32>,           // [64 * 256 * 64] = 1,048,576 floats = 4 MB
    pub centroid_norms: Vec<f32>, // [64 * 256] 预计算 L2 范数
}

impl PqCodebook {
    /// 从 flat f32 数组构建（从 inverted_index 导出）
    pub fn from_flat(data: Vec<f32>) -> Self {
        assert_eq!(data.len(), NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM);
        // 预计算每个中心的 L2 范数
        let mut centroid_norms = Vec::with_capacity(NUM_SUBSPACES * NUM_CENTROIDS);
        for s in 0..NUM_SUBSPACES {
            for c in 0..NUM_CENTROIDS {
                let off = (s * NUM_CENTROIDS + c) * SUB_DIM;
                let sub = &data[off..off + SUB_DIM];
                let mut sum_sq = 0.0f32;
                for chunk in sub.chunks_exact(4) {
                    sum_sq += chunk[0] * chunk[0]
                        + chunk[1] * chunk[1]
                        + chunk[2] * chunk[2]
                        + chunk[3] * chunk[3];
                }
                centroid_norms.push(sum_sq.sqrt());
            }
        }
        Self {
            data,
            centroid_norms,
        }
    }

    /// 获取子空间 s、中心 c 的 64d 向量切片
    #[inline]
    pub fn centroid(&self, s: usize, c: usize) -> &[f32] {
        let off = (s * NUM_CENTROIDS + c) * SUB_DIM;
        &self.data[off..off + SUB_DIM]
    }

    /// 获取子空间 s、中心 c 的预计算范数
    #[inline]
    pub fn centroid_norm(&self, s: usize, c: usize) -> f32 {
        self.centroid_norms[s * NUM_CENTROIDS + c]
    }
}

/// 单个文档的 PQ 编码点云
pub struct PqDocumentCloud {
    pub doc_id: u32,
    /// PQ codes 连续存储：codes[t * NUM_SUBSPACES + s] = token t 在子空间 s 的码本 ID
    pub codes: Vec<u8>,
    pub n_tokens: usize,
}

impl PqDocumentCloud {
    /// 获取 token t 在子空间 s 的码本 ID
    #[inline]
    pub fn code(&self, t: usize, s: usize) -> usize {
        self.codes[t * NUM_SUBSPACES + s] as usize
    }
}

/// 全库 PQ 编码存储
pub struct PqStore {
    pub codebook: PqCodebook,
    pub documents: Vec<PqDocumentCloud>,
    pub id_map: HashMap<u32, usize>,
    pub total_tokens: usize,
}

impl PqStore {
    /// 从 CloudStore 编码为 PQ
    /// 对每个 token 的每个子空间，找最近码本中心，记录 ID
    pub fn encode_from_cloud_store(store: &CloudStore, codebook: &PqCodebook) -> Self {
        let total_tokens: usize = store.documents.iter().map(|d| d.n_sentences).sum();
        eprintln!(
            "[PqStore] 开始编码 {} 文档, {} tokens...",
            store.documents.len(),
            total_tokens
        );

        let documents: Vec<PqDocumentCloud> = store
            .documents
            .par_iter()
            .map(|doc| {
                let mut codes = Vec::with_capacity(doc.n_sentences * NUM_SUBSPACES);
                for t in 0..doc.n_sentences {
                    let tok = doc.sentence(t);
                    for s in 0..NUM_SUBSPACES {
                        let off = s * SUB_DIM;
                        let tok_sub = &tok[off..off + SUB_DIM];
                        // 找最近码本中心
                        let mut best_c = 0u8;
                        let mut best_d = f32::MAX;
                        for c in 0..NUM_CENTROIDS {
                            let cent = codebook.centroid(s, c);
                            let d = l2_sq_64d(tok_sub, cent);
                            if d < best_d {
                                best_d = d;
                                best_c = c as u8;
                            }
                        }
                        codes.push(best_c);
                    }
                }
                PqDocumentCloud {
                    doc_id: doc.doc_id,
                    codes,
                    n_tokens: doc.n_sentences,
                }
            })
            .collect();

        let mut id_map = HashMap::with_capacity(documents.len());
        for (idx, doc) in documents.iter().enumerate() {
            id_map.insert(doc.doc_id, idx);
        }

        eprintln!(
            "[PqStore] 编码完成: {} 文档, {} tokens, {:.1} MB",
            documents.len(),
            total_tokens,
            (total_tokens * NUM_SUBSPACES) as f64 / (1024.0 * 1024.0)
        );

        Self {
            codebook: PqCodebook::from_flat(codebook.data.clone()),
            documents,
            id_map,
            total_tokens,
        }
    }

    /// 保存到 SQLite
    pub fn save_to_sqlite(&self, db_path: &str) -> Result<(), String> {
        let conn =
            Connection::open(db_path).map_err(|e| format!("打开 SQLite 失败: {}", e))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS codebook (
                subspace INTEGER NOT NULL,
                centroid INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (subspace, centroid)
            );
            CREATE TABLE IF NOT EXISTS pq_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                codes BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pq_file ON pq_codes(file_id);
        ",
        )
        .map_err(|e| format!("建表失败: {}", e))?;

        // 保存码本
        {
            let mut stmt = conn
                .prepare(
                    "INSERT OR REPLACE INTO codebook (subspace, centroid, vector) VALUES (?1, ?2, ?3)",
                )
                .map_err(|e| format!("prepare 失败: {}", e))?;
            for s in 0..NUM_SUBSPACES {
                for c in 0..NUM_CENTROIDS {
                    let cent = self.codebook.centroid(s, c);
                    let blob: Vec<u8> = cent.iter().flat_map(|f| f.to_ne_bytes()).collect();
                    stmt.execute(rusqlite::params![s as i64, c as i64, blob])
                        .map_err(|e| format!("插入码本失败: {}", e))?;
                }
            }
        }

        // 保存 PQ codes
        {
            let mut stmt = conn
                .prepare("INSERT INTO pq_codes (file_id, codes) VALUES (?1, ?2)")
                .map_err(|e| format!("prepare 失败: {}", e))?;
            for doc in &self.documents {
                for t in 0..doc.n_tokens {
                    let code_slice =
                        &doc.codes[t * NUM_SUBSPACES..(t + 1) * NUM_SUBSPACES];
                    stmt.execute(rusqlite::params![doc.doc_id as i64, code_slice])
                        .map_err(|e| format!("插入 PQ code 失败: {}", e))?;
                }
            }
        }

        eprintln!("[PqStore] 已保存到 {}", db_path);
        Ok(())
    }

    /// 从 SQLite 加载
    pub fn load_from_sqlite(db_path: &str) -> Result<Self, String> {
        let conn =
            Connection::open(db_path).map_err(|e| format!("打开 SQLite 失败: {}", e))?;

        // 加载码本
        let mut codebook_data = vec![0.0f32; NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM];
        {
            let mut stmt = conn
                .prepare(
                    "SELECT subspace, centroid, vector FROM codebook ORDER BY subspace, centroid",
                )
                .map_err(|e| format!("查询码本失败: {}", e))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, i64>(1)? as usize,
                        row.get::<_, Vec<u8>>(2)?,
                    ))
                })
                .map_err(|e| format!("查询码本失败: {}", e))?;

            for row in rows {
                let (s, c, blob) = row.map_err(|e| format!("读取码本行失败: {}", e))?;
                let off = (s * NUM_CENTROIDS + c) * SUB_DIM;
                for (i, chunk) in blob.chunks_exact(4).enumerate() {
                    codebook_data[off + i] =
                        f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
        }
        let codebook = PqCodebook::from_flat(codebook_data);

        // 加载 PQ codes
        let mut doc_codes: HashMap<u32, Vec<u8>> = HashMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT file_id, codes FROM pq_codes ORDER BY file_id, id")
                .map_err(|e| format!("查询 PQ codes 失败: {}", e))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0)? as u32,
                        row.get::<_, Vec<u8>>(1)?,
                    ))
                })
                .map_err(|e| format!("查询失败: {}", e))?;

            for row in rows {
                let (file_id, codes) = row.map_err(|e| format!("读取行失败: {}", e))?;
                doc_codes
                    .entry(file_id)
                    .or_default()
                    .extend_from_slice(&codes);
            }
        }

        let mut documents = Vec::with_capacity(doc_codes.len());
        let mut id_map = HashMap::with_capacity(doc_codes.len());
        let mut total_tokens = 0usize;

        for (doc_id, codes) in doc_codes {
            let n_tokens = codes.len() / NUM_SUBSPACES;
            total_tokens += n_tokens;
            let idx = documents.len();
            documents.push(PqDocumentCloud {
                doc_id,
                codes,
                n_tokens,
            });
            id_map.insert(doc_id, idx);
        }

        eprintln!(
            "[PqStore] 从 {} 加载: {} 文档, {} tokens, {:.1} MB",
            db_path,
            documents.len(),
            total_tokens,
            (total_tokens * NUM_SUBSPACES) as f64 / (1024.0 * 1024.0)
        );

        Ok(Self {
            codebook,
            documents,
            id_map,
            total_tokens,
        })
    }

    /// 内存占用
    pub fn memory_usage(&self) -> usize {
        let cb = self.codebook.data.len() * 4 + self.codebook.centroid_norms.len() * 4;
        let codes: usize = self.documents.iter().map(|d| d.codes.len()).sum();
        cb + codes
    }
}
