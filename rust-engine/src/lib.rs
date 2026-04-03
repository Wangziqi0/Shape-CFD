// law-vexus/src/lib.rs
// NAPI-RS 入口 + 模块声明
// D9-B: LawVexus 主结构体导出

#![deny(clippy::all)]

use napi::Task;
use std::sync::{Arc, RwLock};

pub mod hnsw_engine;
pub mod index_manager;
pub mod residual;
pub mod search;
pub mod utils;

// V12: PQ-Chamfer 全库扫描内核
pub mod pq_chamfer;
pub mod cloud_store;
pub mod fullscan;

// V13: VT-Aligned 距离 + PDE 求解器 + Shape-CFD 完整管线
pub mod vt_distance;
pub mod pde;

// V11: Token 级点云存储（FP16 binary → f32）
pub mod token_cloud_store;
// V11: Token-level PQ-Chamfer 距离（token 点云粗筛 + 管线）
pub mod token_chamfer;

// V14: Token 倒排索引粗筛
pub mod inverted_index;

// V14-PQ: PQ 量化存储 + ADC 近似距离
pub mod pq_store;
pub mod adc_chamfer;
pub mod helmholtz_kalman;

// V15: Multi-Probe Retrieval（多路粗筛探测器）
pub mod multi_probe;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use hnsw_engine::LawHnswEngine;
use index_manager::IndexManager;
use residual::{LawResidualTask, ResidualParams};
use search::{CascadeParams, CascadeSearchResult};

// ============================================================================
// NAPI 导出返回值类型
// ============================================================================

/// 单条检索结果
#[napi(object)]
pub struct SearchResult {
    pub id: u32,
    pub score: f64,
}

/// 级联检索结果（四类分层返回）
#[napi(object)]
pub struct CascadeResult {
    pub statutes: Vec<SearchResult>,
    pub cases: Vec<SearchResult>,
    pub judgments: Vec<SearchResult>,
    pub literature: Vec<SearchResult>,
    pub elapsed_ms: f64,
}

/// 索引统计信息
#[napi(object)]
pub struct IndexStats {
    pub name: String,
    pub total_vectors: u32,
    pub dimensions: u32,
    pub capacity: u32,
    pub memory_usage_bytes: i64, // NAPI 不支持 u64，用 i64
}

/// SVD 分解结果
#[napi(object)]
pub struct SvdResult {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub k: u32,
    pub dim: u32,
}

/// 正交投影结果
#[napi(object)]
pub struct OrthogonalProjectionResult {
    pub projection: Vec<f64>,
    pub residual: Vec<f64>,
    pub basis_coefficients: Vec<f64>,
}

/// PQ-Chamfer 全库扫描结果
#[napi(object)]
pub struct FullscanHit {
    pub id: u32,
    pub score: f64, // 1.0 - chamfer_distance（越大越好）
}

/// Multi-Probe Recall 统计结果
#[napi(object)]
pub struct MultiProbeRecallResult {
    pub probe1_ids: Vec<u32>,
    pub probe2_ids: Vec<u32>,
    pub probe3_ids: Vec<u32>,
    pub merged_ids: Vec<u32>,
}

// ============================================================================
// 内部类型 → NAPI 类型转换
// ============================================================================

/// 将内部 (u64, f64) 结果列表转换为 NAPI SearchResult 列表
fn to_search_results(results: &[(u64, f64)]) -> Vec<SearchResult> {
    results
        .iter()
        .map(|&(id, score)| SearchResult {
            id: id as u32,
            score,
        })
        .collect()
}

/// 将内部 CascadeSearchResult 转换为 NAPI CascadeResult
fn to_cascade_result(result: CascadeSearchResult) -> CascadeResult {
    CascadeResult {
        statutes: to_search_results(&result.statutes),
        cases: to_search_results(&result.cases),
        judgments: to_search_results(&result.judgments),
        literature: to_search_results(&result.literature),
        elapsed_ms: result.elapsed_ms,
    }
}

// ============================================================================
// LawVexus 主入口（NAPI 导出）
// ============================================================================

/// LawVexus — 法律向量检索引擎
///
/// Node.js 侧使用:
/// ```js
/// const { LawVexus } = require('law-vexus');
/// const vexus = new LawVexus('/path/to/index/store');
/// ```
#[napi]
pub struct LawVexus {
    manager: IndexManager,
    db_path: Option<String>,
    // V12: PQ-Chamfer 全库点云存储（句子级）
    cloud_store: Option<cloud_store::CloudStore>,
    // V11: Token 级点云存储（FP16 binary 格式）
    token_store: Option<token_cloud_store::TokenCloudStore>,
    // V11: Token 级点云存储（SQLite 格式，复用 CloudStore 加载 token_clouds.sqlite）
    token_cloud_store: Option<cloud_store::CloudStore>,
    // V11: Query token 点云存储（SQLite 格式，复用 CloudStore 加载 query_token_clouds.sqlite）
    query_token_store: Option<cloud_store::CloudStore>,
    // V11: 预计算的 token 质心（每个文档的 token 均值向量，用于两阶段粗筛）
    token_centroids: Option<Vec<Vec<f32>>>,
    // V14: Token 倒排索引
    token_inverted_index: Option<inverted_index::TokenInvertedIndex>,
    // V14-PQ: PQ 量化存储
    pq_store: Option<pq_store::PqStore>,
    // V15: Multi-Probe 最强 token 代表向量
    max_token_reprs: Option<Vec<multi_probe::MaxTokenRepr>>,
}

#[napi]
impl LawVexus {
    // ===== 索引生命周期管理 =====

    /// 构造函数：初始化索引管理器
    /// @param storePath - 索引文件存储目录
    #[napi(constructor)]
    pub fn new(store_path: String) -> Self {
        Self {
            manager: IndexManager::new(&store_path),
            db_path: None,
            cloud_store: None,
            token_store: None,
            token_cloud_store: None,
            query_token_store: None,
            token_centroids: None,
            token_inverted_index: None,
            pq_store: None,
            max_token_reprs: None,
        }
    }

    /// 设置 SQLite 数据库路径（供级联检索锚点查询使用）
    #[napi]
    pub fn set_db_path(&mut self, db_path: String) {
        self.db_path = Some(db_path);
    }

    /// 创建或加载指定索引
    #[napi]
    pub fn create_index(
        &mut self,
        name: String,
        dimensions: u32,
        capacity: u32,
    ) -> Result<()> {
        self.manager
            .get_or_create(&name, dimensions, capacity)
            .map_err(|e| Error::from_reason(format!("创建索引失败: {}", e)))?;
        Ok(())
    }

    /// 从磁盘加载索引
    #[napi]
    pub fn load_index(&mut self, name: String, path: String) -> Result<()> {
        let engine = LawHnswEngine::load(&name, &path, 4096, 50_000)
            .map_err(|e| Error::from_reason(format!("加载索引失败: {}", e)))?;
        self.manager.indices.insert(name, engine);
        Ok(())
    }

    /// 保存指定索引到磁盘（原子写入）
    #[napi]
    pub fn save_index(&self, name: String, path: String) -> Result<()> {
        let engine = self
            .manager
            .indices
            .get(&name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", name)))?;
        engine
            .save(&path)
            .map_err(|e| Error::from_reason(format!("保存索引失败: {}", e)))?;
        Ok(())
    }

    /// 保存所有索引
    #[napi]
    pub fn save_all(&self) -> Result<()> {
        self.manager
            .save_all()
            .map_err(|e| Error::from_reason(format!("保存所有索引失败: {}", e)))?;
        Ok(())
    }

    // ===== 向量 CRUD =====

    /// 添加单个向量
    #[napi]
    pub fn add_vector(
        &mut self,
        index_name: String,
        id: u32,
        vector: Buffer,
    ) -> Result<()> {
        let engine = self
            .manager
            .indices
            .get(&index_name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", index_name)))?;

        let dim = engine.dimensions() as usize;
        let vec_slice = crate::utils::buffer_to_f32_slice(&vector, dim)
            .map_err(|e| Error::from_reason(format!("{}", e)))?;

        engine
            .add(id as u64, vec_slice)
            .map_err(|e| Error::from_reason(format!("添加向量失败: {}", e)))?;
        Ok(())
    }

    /// 批量添加
    #[napi]
    pub fn add_batch(
        &mut self,
        index_name: String,
        ids: Vec<u32>,
        vectors: Buffer,
    ) -> Result<u32> {
        let engine = self
            .manager
            .indices
            .get(&index_name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", index_name)))?;

        let ids64: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
        let vec_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                vectors.as_ptr() as *const f32,
                vectors.len() / std::mem::size_of::<f32>(),
            )
        };

        engine
            .add_batch(&ids64, vec_slice)
            .map_err(|e| Error::from_reason(format!("批量添加失败: {}", e)))
    }

    /// 删除向量
    #[napi]
    pub fn remove_vector(&mut self, index_name: String, id: u32) -> Result<()> {
        let engine = self
            .manager
            .indices
            .get(&index_name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", index_name)))?;
        engine
            .remove(id as u64)
            .map_err(|e| Error::from_reason(format!("删除向量失败: {}", e)))?;
        Ok(())
    }

    // ===== 检索 =====

    /// 单索引检索
    #[napi]
    pub fn search(
        &self,
        index_name: String,
        query: Buffer,
        k: u32,
    ) -> Result<Vec<SearchResult>> {
        let engine = self
            .manager
            .indices
            .get(&index_name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", index_name)))?;

        let dim = engine.dimensions() as usize;
        let query_slice = crate::utils::buffer_to_f32_slice(&query, dim)
            .map_err(|e| Error::from_reason(format!("{}", e)))?;

        let results = engine
            .search(query_slice, k)
            .map_err(|e| Error::from_reason(format!("检索失败: {}", e)))?;

        Ok(to_search_results(&results))
    }

    /// 级联检索（D9-C 核心创新）
    ///
    /// 四阶段级联管线：
    /// 1. 法条检索 → 提取命中法条 IDs
    /// 2. SQLite 锚点查询 → 案例加权检索 → 裁判书检索
    /// 3. 文献独立检索
    /// 4. RRF 融合排序
    #[napi]
    pub fn cascade_search(
        &self,
        query: Buffer,
        statute_k: u32,
        case_k: u32,
        judgment_k: u32,
        literature_k: u32,
    ) -> Result<CascadeResult> {
        // 动态获取维度（从任一已加载索引读取，默认 4096）
        let dim = self.manager.indices.values().next()
            .map(|e| e.dimensions() as usize)
            .unwrap_or(4096);

        let query_slice = crate::utils::buffer_to_f32_slice(&query, dim)
            .map_err(|e| Error::from_reason(format!("{}", e)))?;

        let db_path = self.db_path.as_deref().unwrap_or("");

        let params = CascadeParams::with_k(statute_k, case_k, judgment_k, literature_k);

        let result = search::cascade_search(&self.manager, db_path, query_slice, &params)
            .map_err(|e| Error::from_reason(format!("级联检索失败: {}", e)))?;

        Ok(to_cascade_result(result))
    }

    // ===== 异步任务 =====

    /// 从 SQLite 恢复索引（异步，不阻塞 Node.js 主线程）
    ///
    /// @param indexName - 索引名称
    /// @param dbPath - SQLite 数据库路径
    /// @param table - 表名 (chunks / cases / judgments / literature)
    /// @param filter - 可选的过滤条件
    #[napi]
    pub fn recover_from_sqlite(
        &self,
        index_name: String,
        db_path: String,
        table: String,
        filter: Option<String>,
    ) -> AsyncTask<RecoverTask> {
        let engine = self.manager.indices.get(&index_name);
        let index_arc = engine.map(|e| e.index_arc());
        let dimensions = engine.map(|e| e.dimensions()).unwrap_or(4096);

        AsyncTask::new(RecoverTask {
            index: index_arc,
            db_path,
            table,
            filter,
            dimensions,
        })
    }

    /// 内生残差预计算（异步，D9-D）
    #[napi]
    pub fn compute_intrinsic_residuals(
        &self,
        db_path: String,
        max_svd_rank: Option<u32>,
    ) -> AsyncTask<LawResidualTask> {
        AsyncTask::new(LawResidualTask {
            db_path,
            dimensions: 4096,
            params: ResidualParams {
                max_svd_rank: max_svd_rank.unwrap_or(8) as usize,
                ..Default::default()
            },
        })
    }

    // ===== 工具方法 =====

    /// 自动扩容指定索引
    #[napi]
    pub fn auto_expand(&mut self, index_name: String) -> Result<()> {
        let engine = self
            .manager
            .indices
            .get(&index_name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", index_name)))?;

        engine
            .auto_expand()
            .map_err(|e| Error::from_reason(format!("扩容失败: {}", e)))?;
        Ok(())
    }

    /// 获取指定索引的统计信息
    #[napi]
    pub fn get_stats(&self, index_name: String) -> Result<IndexStats> {
        let engine = self
            .manager
            .indices
            .get(&index_name)
            .ok_or_else(|| Error::from_reason(format!("索引不存在: {}", index_name)))?;

        let stats = engine
            .stats()
            .map_err(|e| Error::from_reason(format!("获取统计失败: {}", e)))?;

        Ok(IndexStats {
            name: stats.name,
            total_vectors: stats.total_vectors,
            dimensions: stats.dimensions,
            capacity: stats.capacity,
            memory_usage_bytes: stats.memory_usage_bytes as i64,
        })
    }

    /// 获取所有索引的统计信息
    #[napi]
    pub fn get_all_stats(&self) -> Result<Vec<IndexStats>> {
        let mut result = Vec::new();
        for (_, engine) in &self.manager.indices {
            let stats = engine
                .stats()
                .map_err(|e| Error::from_reason(format!("获取统计失败: {}", e)))?;
            result.push(IndexStats {
                name: stats.name,
                total_vectors: stats.total_vectors,
                dimensions: stats.dimensions,
                capacity: stats.capacity,
                memory_usage_bytes: stats.memory_usage_bytes as i64,
            });
        }
        Ok(result)
    }

    /// 驱逐空闲索引（D9-D TTL 驱逐）
    #[napi]
    pub fn evict_idle(&mut self) -> Result<Vec<String>> {
        let config = index_manager::EvictionConfig::default();
        self.manager
            .evict_idle(&config)
            .map_err(|e| Error::from_reason(format!("驱逐失败: {}", e)))
    }

    /// SVD 分解（供 D4 LegalDiversity 使用）
    #[napi]
    pub fn compute_svd(
        &self,
        flattened_vectors: Buffer,
        n: u32,
        dim: u32,
        max_k: u32,
    ) -> Result<SvdResult> {
        let n_usize = n as usize;
        let dim_usize = dim as usize;
        let max_k_usize = max_k as usize;

        let vec_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                flattened_vectors.as_ptr() as *const f32,
                flattened_vectors.len() / std::mem::size_of::<f32>(),
            )
        };

        if vec_slice.len() != n_usize * dim_usize {
            return Err(Error::from_reason(format!(
                "SVD 输入长度不匹配: 期望 {}, 实际 {}",
                n_usize * dim_usize,
                vec_slice.len()
            )));
        }

        use nalgebra::DMatrix;
        let matrix = DMatrix::from_row_slice(n_usize, dim_usize, vec_slice);
        let svd = matrix.svd(false, true);

        let s: Vec<f64> = svd
            .singular_values
            .as_slice()
            .iter()
            .map(|&x| x as f64)
            .collect();
        let v_t = svd.v_t.ok_or_else(|| {
            Error::from_reason("SVD 计算 V^T 失败".to_string())
        })?;

        let k = std::cmp::min(s.len(), max_k_usize);
        let mut u_flattened = Vec::with_capacity(k * dim_usize);

        for i in 0..k {
            let row = v_t.row(i);
            for &val in row.iter() {
                u_flattened.push(val as f64);
            }
        }

        Ok(SvdResult {
            u: u_flattened,
            s: s[..k].to_vec(),
            k: k as u32,
            dim,
        })
    }

    /// Gram-Schmidt 正交投影
    #[napi]
    pub fn compute_orthogonal_projection(
        &self,
        vector: Buffer,
        flattened_basis: Buffer,
        n_basis: u32,
        dim: u32,
    ) -> Result<OrthogonalProjectionResult> {
        let dim_usize = dim as usize;
        let n = n_basis as usize;

        let query: &[f32] = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const f32,
                vector.len() / 4,
            )
        };
        let basis_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                flattened_basis.as_ptr() as *const f32,
                flattened_basis.len() / 4,
            )
        };

        if query.len() != dim_usize || basis_slice.len() != n * dim_usize {
            return Err(Error::from_reason("正交投影维度不匹配".to_string()));
        }

        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut basis_coefficients = vec![0.0; n];
        let mut projection = vec![0.0; dim_usize];

        for i in 0..n {
            let start = i * dim_usize;
            let tag_vec = &basis_slice[start..start + dim_usize];
            let mut v: Vec<f64> = tag_vec.iter().map(|&x| x as f64).collect();

            // Gram-Schmidt 正交化
            for u in &basis {
                let mut dot = 0.0;
                for d in 0..dim_usize {
                    dot += v[d] * u[d];
                }
                for d in 0..dim_usize {
                    v[d] -= dot * u[d];
                }
            }

            let mut mag_sq = 0.0;
            for d in 0..dim_usize {
                mag_sq += v[d] * v[d];
            }
            let mag = mag_sq.sqrt();

            if mag > 1e-6 {
                for d in 0..dim_usize {
                    v[d] /= mag;
                }

                let mut coeff = 0.0;
                for d in 0..dim_usize {
                    coeff += (query[d] as f64) * v[d];
                }
                basis_coefficients[i] = coeff.abs();

                for d in 0..dim_usize {
                    projection[d] += coeff * v[d];
                }
                basis.push(v);
            }
        }

        let mut residual_vec = vec![0.0; dim_usize];
        for d in 0..dim_usize {
            residual_vec[d] = (query[d] as f64) - projection[d];
        }

        Ok(OrthogonalProjectionResult {
            projection,
            residual: residual_vec,
            basis_coefficients,
        })
    }

    // ===== V12: PQ-Chamfer 全库扫描 =====

    /// 从 SQLite 加载文档点云到内存
    /// 返回加载的文档数量
    #[napi]
    pub fn load_clouds(&mut self, db_path: String) -> napi::Result<u32> {
        let store = cloud_store::CloudStore::load_from_sqlite(&db_path)
            .map_err(|e| napi::Error::from_reason(format!("加载点云失败: {}", e)))?;
        let count = store.total_docs as u32;
        eprintln!(
            "[LawVexus] ☁️ 点云加载完成: {} 文档, {} 向量, {:.1} MB",
            store.total_docs,
            store.total_vectors,
            store.memory_usage() as f64 / 1_048_576.0
        );
        self.cloud_store = Some(store);
        Ok(count)
    }

    /// 从 FP16 二进制文件加载 token 级点云到内存
    /// bin_path: token_clouds_fp16.bin 路径
    /// index_path: token_index.json 路径
    /// 返回加载统计信息字符串
    #[napi]
    pub fn load_token_clouds(&mut self, bin_path: String, index_path: String) -> napi::Result<String> {
        let store = token_cloud_store::TokenCloudStore::load_from_binary(&bin_path, &index_path)
            .map_err(|e| napi::Error::from_reason(format!("加载 token 点云失败: {}", e)))?;
        let summary = format!(
            "token 点云加载完成: {} 文档, {} tokens, {:.2} GB",
            store.total_docs(),
            store.total_tokens(),
            store.memory_usage() as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        eprintln!("[LawVexus] {}", summary);
        self.token_store = Some(store);
        Ok(summary)
    }

    /// 全库 PQ-Chamfer 暴力扫描
    /// query_cloud: 每个元素是一个 4096d Float32Array 的 Buffer
    /// k: 返回 top-K
    #[napi]
    pub fn fullscan_pq_chamfer(
        &self,
        query_cloud: Vec<Buffer>,
        k: u32,
    ) -> napi::Result<Vec<FullscanHit>> {
        let store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载，请先调用 loadClouds".to_string()))?;

        // 将 Buffer 列表转为 f32 切片列表
        let query_vecs: Vec<Vec<f32>> = query_cloud
            .iter()
            .map(|buf| {
                let bytes: &[u8] = buf.as_ref();
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .collect();

        let query_refs: Vec<&[f32]> = query_vecs.iter().map(|v| v.as_slice()).collect();

        // 构造文档元组列表供 fullscan 使用
        let docs: Vec<(u32, &[f32], usize, &[pq_chamfer::PqNormCache])> = store
            .documents
            .iter()
            .map(|doc| {
                (
                    doc.doc_id,
                    doc.vectors.as_slice(),
                    doc.n_sentences,
                    doc.norm_caches.as_slice(),
                )
            })
            .collect();

        let exclude = std::collections::HashSet::new();
        let hits = fullscan::fullscan(&query_refs, &docs, k as usize, &exclude);

        Ok(hits
            .into_iter()
            .map(|h| FullscanHit {
                id: h.doc_id,
                score: 1.0 - h.distance as f64,
            })
            .collect())
    }

    // ===== V13: VT-Aligned 距离 + Shape-CFD 管线 =====

    /// 计算 N 个文档之间的 vt_aligned 距离矩阵
    /// doc_ids: 文档 ID 列表 (从 CloudStore 查找点云)
    /// 返回: N*N Float64Array (行优先)
    #[napi]
    pub fn compute_vt_distance_matrix(&self, doc_ids: Vec<u32>) -> napi::Result<Vec<f64>> {
        let store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载，请先调用 loadClouds".to_string()))?;

        // 收集每个文档的点云句子切片引用
        let clouds: Vec<Vec<&[f32]>> = doc_ids
            .iter()
            .map(|&id| {
                store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        // 过滤掉空点云的文档（保持索引对应关系）
        if clouds.iter().any(|c| c.is_empty()) {
            return Err(napi::Error::from_reason(
                "部分 doc_id 对应的点云为空或不存在".to_string(),
            ));
        }

        Ok(vt_distance::compute_vt_distance_matrix(&clouds))
    }

    /// 计算 query 到 N 个文档的 vt_aligned 距离
    /// query: 4096d Float32 Buffer
    /// doc_ids: 文档 ID 列表
    /// 返回: N 个 Float64 距离值
    #[napi]
    pub fn compute_vt_query_distances(
        &self,
        query: Buffer,
        doc_ids: Vec<u32>,
    ) -> napi::Result<Vec<f64>> {
        let store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载，请先调用 loadClouds".to_string()))?;

        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query.as_ptr() as *const f32,
                query.len() / std::mem::size_of::<f32>(),
            )
        };

        if query_slice.len() != pq_chamfer::FULL_DIM {
            return Err(napi::Error::from_reason(format!(
                "query 维度不匹配: 期望 {}，实际 {}",
                pq_chamfer::FULL_DIM,
                query_slice.len()
            )));
        }

        let clouds: Vec<Vec<&[f32]>> = doc_ids
            .iter()
            .map(|&id| {
                store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        if clouds.iter().any(|c| c.is_empty()) {
            return Err(napi::Error::from_reason(
                "部分 doc_id 对应的点云为空或不存在".to_string(),
            ));
        }

        Ok(vt_distance::compute_vt_query_distances(query_slice, &clouds))
    }

    /// 全管线一次调用：cosine 粗筛 + vt_aligned 精排 + PDE + 返回排序结果
    /// query: 4096d Float32 Buffer
    /// k: 返回 top-K
    /// top_n: 初始候选池大小 (cosine top-N)
    /// 返回: [{id, score}] 按 PDE 浓度降序
    #[napi]
    pub fn shape_cfd_pipeline(
        &self,
        query: Buffer,
        k: u32,
        top_n: u32,
    ) -> napi::Result<Vec<FullscanHit>> {
        // 检查点云是否已加载
        let _store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载，请先调用 loadClouds".to_string()))?;

        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query.as_ptr() as *const f32,
                query.len() / std::mem::size_of::<f32>(),
            )
        };

        if query_slice.len() != pq_chamfer::FULL_DIM {
            return Err(napi::Error::from_reason(format!(
                "query 维度不匹配: 期望 {}，实际 {}",
                pq_chamfer::FULL_DIM,
                query_slice.len()
            )));
        }

        let ranked = self.shape_cfd_pipeline_inner(query_slice, k as usize, top_n as usize);

        Ok(ranked
            .into_iter()
            .map(|(id, score)| FullscanHit { id, score })
            .collect())
    }

    /// Shape-CFD 完整管线内部实现
    ///
    /// 1. cosine 全库排序 top-N
    /// 2. vt_aligned 距离矩阵
    /// 3. KNN 图 (k=3)
    /// 4. 对流系数
    /// 5. 初始浓度 = exp(-2 * vt_aligned_query_doc)
    /// 6. PDE 求解
    /// 7. 按浓度排序返回 top-k
    fn shape_cfd_pipeline_inner(
        &self,
        query: &[f32],
        k: usize,
        top_n: usize,
    ) -> Vec<(u32, f64)> {
        let store = self.cloud_store.as_ref().unwrap();

        // 1. cosine 全库排序 top-N
        let candidates = vt_distance::cosine_top_n(query, store, top_n);
        let n = candidates.len();

        if n == 0 {
            return Vec::new();
        }

        // 收集候选文档的点云
        let clouds: Vec<Vec<&[f32]>> = candidates
            .iter()
            .map(|&(id, _)| {
                store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        // 过滤掉空点云（理论上不应发生，但防御性编程）
        if clouds.iter().any(|c| c.is_empty()) {
            // 退化为纯 cosine 排序
            return candidates
                .into_iter()
                .take(k)
                .map(|(id, dist)| (id, 1.0 - dist as f64))
                .collect();
        }

        // 2. vt_aligned 距离矩阵
        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);

        // 3. KNN 图 (k=3，但不超过 n-1)
        let knn_k = 3.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 4. 对流系数 (alpha=0.3)
        let u_matrix = vt_distance::compute_advection(query, &clouds, &adj, n, 0.3);

        // 5. 初始浓度 = exp(-2 * vt_aligned_query_doc)
        let c0: Vec<f64> = clouds
            .iter()
            .map(|cloud| (-2.0 * vt_distance::vt_aligned_query_doc(query, cloud) as f64).exp())
            .collect();

        // 6. PDE 求解 (D=0.15, max_iter=50, epsilon=1e-3)
        let c_final = pde::solve_pde(&c0, &adj, &u_matrix, n, 0.15, 50, 1e-3);

        // 7. 按浓度降序排序，返回 top-k
        let mut ranked: Vec<(u32, f64)> = candidates
            .iter()
            .zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }

    /// 图拉普拉斯平滑管线（PDE baseline）
    ///
    /// 与 shape_cfd_pipeline 完全相同的图构建和 C0 逻辑，
    /// 但用 laplacian_smooth 替代 solve_pde（无对流、无反应项）。
    /// 用于验证 PDE 对流项的实际增益。
    ///
    /// query: 4096d Float32 Buffer
    /// k: 返回 top-K
    /// top_n: 初始候选池大小 (cosine top-N)
    /// alpha: 平滑系数（默认 0.02）
    /// steps: 平滑步数（默认 20）
    /// 返回: [{id, score}] 按浓度降序
    #[napi]
    pub fn shape_laplacian_pipeline(
        &self,
        query: Buffer,
        k: u32,
        top_n: u32,
        alpha: Option<f64>,
        steps: Option<u32>,
    ) -> napi::Result<Vec<FullscanHit>> {
        let _store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载，请先调用 loadClouds".to_string()))?;

        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query.as_ptr() as *const f32,
                query.len() / std::mem::size_of::<f32>(),
            )
        };

        if query_slice.len() != pq_chamfer::FULL_DIM {
            return Err(napi::Error::from_reason(format!(
                "query 维度不匹配: 期望 {}，实际 {}",
                pq_chamfer::FULL_DIM,
                query_slice.len()
            )));
        }

        let alpha_val = alpha.unwrap_or(0.02);
        let steps_val = steps.unwrap_or(20) as usize;

        let ranked = self.shape_laplacian_pipeline_inner(
            query_slice,
            k as usize,
            top_n as usize,
            alpha_val,
            steps_val,
        );

        Ok(ranked
            .into_iter()
            .map(|(id, score)| FullscanHit { id, score })
            .collect())
    }

    /// Rust 原生 cosine top-K 排序（用于 benchmark baseline）
    ///
    /// query: 4096d Float32 Buffer
    /// k: 返回 top-K
    /// 返回: [{id, score}] 按 cosine similarity 降序
    #[napi]
    pub fn cosine_rank(
        &self,
        query: Buffer,
        k: u32,
    ) -> napi::Result<Vec<FullscanHit>> {
        let store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载".to_string()))?;

        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query.as_ptr() as *const f32,
                query.len() / std::mem::size_of::<f32>(),
            )
        };

        // cosine_top_n 返回 (doc_id, cosine_distance)，距离升序
        // 转换为 similarity 降序
        let candidates = vt_distance::cosine_top_n(query_slice, store, k as usize);
        Ok(candidates
            .into_iter()
            .map(|(id, dist)| FullscanHit {
                id,
                score: (1.0 - dist) as f64,
            })
            .collect())
    }

    /// 图拉普拉斯平滑管线内部实现
    ///
    /// 复用 shape_cfd_pipeline_inner 的图构建和 C0 逻辑，
    /// 仅将 PDE 求解替换为 laplacian_smooth。
    fn shape_laplacian_pipeline_inner(
        &self,
        query: &[f32],
        k: usize,
        top_n: usize,
        alpha: f64,
        steps: usize,
    ) -> Vec<(u32, f64)> {
        let store = self.cloud_store.as_ref().unwrap();

        // 1. cosine 全库排序 top-N（与 shape_cfd 完全一致）
        let candidates = vt_distance::cosine_top_n(query, store, top_n);
        let n = candidates.len();

        if n == 0 {
            return Vec::new();
        }

        // 收集候选文档的点云
        let clouds: Vec<Vec<&[f32]>> = candidates
            .iter()
            .map(|&(id, _)| {
                store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        // 过滤掉空点云
        if clouds.iter().any(|c| c.is_empty()) {
            return candidates
                .into_iter()
                .take(k)
                .map(|(id, dist)| (id, 1.0 - dist as f64))
                .collect();
        }

        // 2. vt_aligned 距离矩阵（与 shape_cfd 完全一致）
        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);

        // 3. KNN 图 (k=3，与 shape_cfd 完全一致)
        let knn_k = 3.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 4. 初始浓度 = exp(-2 * vt_aligned_query_doc)（与 shape_cfd 完全一致）
        let c0: Vec<f64> = clouds
            .iter()
            .map(|cloud| (-2.0 * vt_distance::vt_aligned_query_doc(query, cloud) as f64).exp())
            .collect();

        // 5. 图拉普拉斯平滑（替代 PDE 求解，无对流、无反应项）
        let c_final = pde::laplacian_smooth(&c0, &adj, alpha, steps);

        // 6. 按浓度降序排序，返回 top-k
        let mut ranked: Vec<(u32, f64)> = candidates
            .iter()
            .zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }

    /// 图拉普拉斯 + Allen-Cahn 管线（无对流）
    /// 扩散做平滑，Allen-Cahn 做极化，对流完全去掉
    #[napi]
    pub fn shape_laplacian_ac_pipeline(
        &self,
        query: Buffer,
        k: u32,
        top_n: u32,
        alpha: Option<f64>,
        gamma: Option<f64>,
        steps: Option<u32>,
    ) -> napi::Result<Vec<FullscanHit>> {
        let _store = self.cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载".to_string()))?;

        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query.as_ptr() as *const f32,
                query.len() / std::mem::size_of::<f32>(),
            )
        };
        if query_slice.len() != pq_chamfer::FULL_DIM {
            return Err(napi::Error::from_reason(format!(
                "query 维度不匹配: 期望 {}, 实际 {}", pq_chamfer::FULL_DIM, query_slice.len()
            )));
        }

        let store = self.cloud_store.as_ref().unwrap();
        let alpha_val = alpha.unwrap_or(0.02);
        let gamma_val = gamma.unwrap_or(0.5);
        let steps_val = steps.unwrap_or(30) as usize;

        // 复用 laplacian pipeline 的图构建逻辑
        let candidates = vt_distance::cosine_top_n(query_slice, store, top_n as usize);
        let n = candidates.len();
        if n == 0 { return Ok(Vec::new()); }

        let clouds: Vec<Vec<&[f32]>> = candidates.iter()
            .map(|&(id, _)| store.get_cloud(id).map(|d| d.as_slice_refs()).unwrap_or_default())
            .collect();

        if clouds.iter().any(|c| c.is_empty()) {
            return Ok(candidates.into_iter().take(k as usize)
                .map(|(id, dist)| FullscanHit { id, score: 1.0 - dist as f64 }).collect());
        }

        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);
        let knn_k = 3.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);
        let c0: Vec<f64> = clouds.iter()
            .map(|cloud| (-2.0 * vt_distance::vt_aligned_query_doc(query_slice, cloud) as f64).exp())
            .collect();

        // 图拉普拉斯 + Allen-Cahn（无对流）
        let c_final = pde::laplacian_allen_cahn(&c0, &adj, alpha_val, gamma_val, steps_val, 1e-3);

        let mut ranked: Vec<(u32, f64)> = candidates.iter().zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k as usize);

        Ok(ranked.into_iter().map(|(id, score)| FullscanHit { id, score }).collect())
    }

    /// V15: Helmholtz-Kalman 融合管线
    /// 句子级图平滑 + token 级 Chamfer，通过 Helmholtz 分解提取正交信号，Kalman 自适应融合
    #[napi]
    pub fn helmholtz_kalman_pipeline(
        &self,
        query_buf: Buffer,
        query_id: u32,
        k: u32,
        top_n: u32,
        beta: Option<f64>,
    ) -> napi::Result<Vec<FullscanHit>> {
        let store = self.cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("句子级点云未加载".to_string()))?;
        let token_store = self.token_cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_token_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let centroids = self.token_centroids.as_ref()
            .ok_or_else(|| napi::Error::from_reason("质心未计算".to_string()))?;

        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query_buf.as_ptr() as *const f32,
                query_buf.len() / std::mem::size_of::<f32>(),
            )
        };
        if query_slice.len() != pq_chamfer::FULL_DIM {
            return Err(napi::Error::from_reason("query 维度不匹配".to_string()));
        }

        let beta_val = beta.unwrap_or(1.0);
        let top_n = top_n as usize;
        let k = k as usize;

        // 1. Token 质心粗筛 top-100 → 精排 top_n（与 token_2stage 相同候选集）
        let query_cloud = query_token_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let token_candidates = token_chamfer::token_chamfer_two_stage(
            query_cloud, token_store, centroids, 100, top_n,
        );
        let n = token_candidates.len();
        if n == 0 { return Ok(Vec::new()); }

        let doc_ids: Vec<u32> = token_candidates.iter().map(|&(id, _)| id).collect();

        // s_token 直接来自 token_chamfer 精排分数
        let s_token: Vec<f64> = token_candidates.iter()
            .map(|&(_, dist)| (-2.0 * dist).exp())
            .collect();

        // 2. 对同一候选集做句子级图构建
        let clouds: Vec<Vec<&[f32]>> = doc_ids.iter()
            .map(|&id| store.get_cloud(id).map(|d| d.as_slice_refs()).unwrap_or_default())
            .collect();

        if clouds.iter().any(|c| c.is_empty()) {
            // 句子级点云缺失，退回 token 排序
            return Ok(token_candidates.into_iter().take(k)
                .map(|(id, dist)| FullscanHit { id, score: (-2.0 * dist).exp() }).collect());
        }

        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);
        let knn_k = 3.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 3. 句子级 C₀ → 图拉普拉斯平滑 → s_graph
        let c0: Vec<f64> = clouds.iter()
            .map(|cloud| (-2.0 * vt_distance::vt_aligned_query_doc(query_slice, cloud) as f64).exp())
            .collect();
        let s_graph = pde::laplacian_smooth(&c0, &adj, 0.02, 20);

        // 5. 计算 per-document token 方差
        let doc_clouds_for_var: Vec<&cloud_store::DocumentCloud> = doc_ids.iter()
            .filter_map(|&did| token_store.get_cloud(did))
            .collect();

        let token_variances = if doc_clouds_for_var.len() == n {
            helmholtz_kalman::compute_token_variances(query_cloud, &doc_clouds_for_var)
        } else {
            // fallback: 均匀方差
            vec![0.1; n]
        };

        // 6. 归一化分数到 [0, 1]
        let normalize = |v: &[f64]| -> Vec<f64> {
            let mn = v.iter().cloned().fold(f64::MAX, f64::min);
            let mx = v.iter().cloned().fold(f64::MIN, f64::max);
            let range = mx - mn;
            if range < 1e-12 { return vec![0.5; v.len()]; }
            v.iter().map(|&x| (x - mn) / range).collect()
        };

        let s_token_norm = normalize(&s_token);
        let s_graph_norm = normalize(&s_graph);

        // 7. Helmholtz-Kalman 融合
        let s_fused = helmholtz_kalman::helmholtz_kalman_fuse(
            &s_token_norm, &s_graph_norm, &adj, &token_variances, beta_val,
        );

        // 8. 排序返回 top-k
        let mut ranked: Vec<(u32, f64)> = doc_ids.iter().zip(s_fused.iter())
            .map(|(&id, &score)| (id, score)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);

        Ok(ranked.into_iter().map(|(id, score)| FullscanHit { id, score }).collect())
    }

    /// 探针 PQ-Chamfer 检索（含排除列表，供 Stefan 预取使用）
    #[napi]
    pub fn probe_pq_chamfer(
        &self,
        probe_cloud: Vec<Buffer>,
        k: u32,
        exclude_ids: Vec<u32>,
    ) -> napi::Result<Vec<FullscanHit>> {
        let store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("点云未加载，请先调用 loadClouds".to_string()))?;

        let probe_vecs: Vec<Vec<f32>> = probe_cloud
            .iter()
            .map(|buf| {
                let bytes: &[u8] = buf.as_ref();
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .collect();

        let probe_refs: Vec<&[f32]> = probe_vecs.iter().map(|v| v.as_slice()).collect();

        let docs: Vec<(u32, &[f32], usize, &[pq_chamfer::PqNormCache])> = store
            .documents
            .iter()
            .map(|doc| {
                (
                    doc.doc_id,
                    doc.vectors.as_slice(),
                    doc.n_sentences,
                    doc.norm_caches.as_slice(),
                )
            })
            .collect();

        let exclude: std::collections::HashSet<u32> = exclude_ids.into_iter().collect();
        let hits = fullscan::fullscan(&probe_refs, &docs, k as usize, &exclude);

        Ok(hits
            .into_iter()
            .map(|h| FullscanHit {
                id: h.doc_id,
                score: 1.0 - h.distance as f64,
            })
            .collect())
    }

    // ===== V11: Token-level PQ-Chamfer 管线 =====

    /// 从 SQLite 加载 token 级点云（语料库 + query）
    ///
    /// token_clouds.sqlite 和 query_token_clouds.sqlite 的 schema 与 clouds.sqlite 完全一致，
    /// 复用 CloudStore::load_from_sqlite 加载。
    ///
    /// @param corpusPath - token_clouds.sqlite 路径（~14GB，约 880k token 向量）
    /// @param queryPath - query_token_clouds.sqlite 路径（~331MB，约 20k token 向量）
    /// @returns 加载统计信息字符串
    #[napi]
    pub fn load_token_clouds_sqlite(
        &mut self,
        corpus_path: String,
        query_path: String,
    ) -> napi::Result<String> {
        // 加载语料库 token 点云
        eprintln!("[LawVexus] 开始加载语料库 token 点云: {}", corpus_path);
        let corpus_store = cloud_store::CloudStore::load_from_sqlite(&corpus_path)
            .map_err(|e| napi::Error::from_reason(format!("加载语料库 token 点云失败: {}", e)))?;
        let corpus_docs = corpus_store.total_docs;
        let corpus_vecs = corpus_store.total_vectors;
        let corpus_mem = corpus_store.memory_usage();
        eprintln!(
            "[LawVexus] 语料库 token 点云加载完成: {} 文档, {} tokens, {:.2} GB",
            corpus_docs,
            corpus_vecs,
            corpus_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // 加载 query token 点云
        eprintln!("[LawVexus] 开始加载 query token 点云: {}", query_path);
        let query_store = cloud_store::CloudStore::load_from_sqlite(&query_path)
            .map_err(|e| napi::Error::from_reason(format!("加载 query token 点云失败: {}", e)))?;
        let query_docs = query_store.total_docs;
        let query_vecs = query_store.total_vectors;
        let query_mem = query_store.memory_usage();
        eprintln!(
            "[LawVexus] query token 点云加载完成: {} queries, {} tokens, {:.1} MB",
            query_docs,
            query_vecs,
            query_mem as f64 / (1024.0 * 1024.0),
        );

        // 预计算 token 质心（用于两阶段粗筛）
        eprintln!("[LawVexus] 预计算 token 质心...");
        let centroids = token_chamfer::precompute_token_centroids(&corpus_store);
        eprintln!("[LawVexus] token 质心计算完成: {} 个质心", centroids.len());

        // V14: 构建 token 倒排索引（必须在 corpus_store move 之前）
        eprintln!("[LawVexus] 构建 token 倒排索引...");
        let t_idx = std::time::Instant::now();
        let inv_index = inverted_index::TokenInvertedIndex::build(&corpus_store);
        eprintln!(
            "[LawVexus] 倒排索引构建完成 ({:.1}s, {:.1} MB)",
            t_idx.elapsed().as_secs_f64(),
            inv_index.memory_usage() as f64 / (1024.0 * 1024.0),
        );

        // V14-PQ: 提取码本并编码 PQ
        eprintln!("[LawVexus] PQ 编码...");
        let t_pq = std::time::Instant::now();
        let codebook_flat = inv_index.export_codebook_flat();
        let codebook = pq_store::PqCodebook::from_flat(codebook_flat);
        let pq = pq_store::PqStore::encode_from_cloud_store(&corpus_store, &codebook);
        eprintln!(
            "[LawVexus] PQ 编码完成 ({:.1}s, {:.1} MB)",
            t_pq.elapsed().as_secs_f64(),
            pq.memory_usage() as f64 / (1024.0 * 1024.0),
        );

        self.token_cloud_store = Some(corpus_store);
        self.query_token_store = Some(query_store);
        self.token_centroids = Some(centroids);
        self.token_inverted_index = Some(inv_index);
        self.pq_store = Some(pq);

        let summary = format!(
            "token 点云加载完成: 语料库 {} 文档/{} tokens ({:.2} GB), query {} queries/{} tokens ({:.1} MB), 质心已预计算",
            corpus_docs, corpus_vecs,
            corpus_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            query_docs, query_vecs,
            query_mem as f64 / (1024.0 * 1024.0),
        );
        Ok(summary)
    }

    /// V11 完整管线: token Chamfer 粗筛 + 句子级 VT-Aligned KNN+PDE 精排
    ///
    /// 流程:
    /// 1. 从 query_token_store 获取 query 的 token 点云
    /// 2. token_chamfer_top_n 全库扫描取 top_n 候选
    /// 3. 从 cloud_store（句子级）获取 top_n 的句子点云
    /// 4. vt_aligned doc-doc 距离矩阵
    /// 5. build_knn + compute_advection + solve_pde
    /// 6. 返回 [[doc_id, score], ...]
    ///
    /// 参数沿用: k_knn=3, alpha=0.3, D=0.15, 50 步, epsilon=1e-3
    /// c0 初始浓度用 token Chamfer 距离: c0 = exp(-2 * token_chamfer_dist)
    ///
    /// @param queryId - query 文档 ID（在 query_token_clouds.sqlite 中的 file_id）
    /// @param k - 最终返回 top-K 结果数
    /// @param topN - 粗筛候选池大小
    /// @returns [[doc_id, score], ...] 按 PDE 浓度降序
    #[napi]
    pub fn token_chamfer_pipeline(
        &self,
        query_id: u32,
        k: u32,
        top_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let sentence_store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "句子级点云未加载，请先调用 loadClouds".to_string(),
            ))?;

        // 1. 获取 query 的 token 点云
        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        // 2. token Chamfer 全库扫描取 top_n
        let candidates = token_chamfer::token_chamfer_top_n(
            query_cloud,
            token_store,
            top_n as usize,
        );
        let n = candidates.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // 保存 token Chamfer 距离用于 c0 初始化
        let chamfer_dists: Vec<f64> = candidates.iter().map(|&(_, d)| d).collect();

        // 3. 从句子级 cloud_store 获取候选文档的句子点云
        let clouds: Vec<Vec<&[f32]>> = candidates
            .iter()
            .map(|&(id, _)| {
                sentence_store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        // 如果有文档在句子级点云中不存在，退化为纯 token Chamfer 排序
        if clouds.iter().any(|c| c.is_empty()) {
            let results: Vec<Vec<f64>> = candidates
                .iter()
                .take(k as usize)
                .map(|&(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
                .collect();
            return Ok(results);
        }

        // 4. vt_aligned doc-doc 距离矩阵
        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);

        // 5. KNN 图 (k=3，但不超过 n-1)
        let knn_k = 3usize.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 对流系数需要 query 向量（4096d），这里用 query token 的质心
        let query_centroid = self.compute_query_centroid(query_cloud);

        // 对流系数 (alpha=0.3)
        let u_matrix = vt_distance::compute_advection(
            &query_centroid,
            &clouds,
            &adj,
            n,
            0.3,
        );

        // 6. 初始浓度 c0 = exp(-2 * token_chamfer_dist)
        let c0: Vec<f64> = chamfer_dists
            .iter()
            .map(|&d| (-2.0 * d).exp())
            .collect();

        // 7. PDE 求解 (D=0.15, max_iter=50, epsilon=1e-3)
        let c_final = pde::solve_pde(&c0, &adj, &u_matrix, n, 0.15, 50, 1e-3);

        // 8. 按浓度降序排序，返回 top-k
        let mut ranked: Vec<(u32, f64)> = candidates
            .iter()
            .zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k as usize);

        // 返回 [[doc_id, score], ...]
        Ok(ranked
            .into_iter()
            .map(|(id, score)| vec![id as f64, score])
            .collect())
    }

    /// 纯 token Chamfer 全库扫描排序（不走 PDE，用于消融实验）
    ///
    /// @param queryId - query 文档 ID
    /// @param k - 返回 top-K
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn token_chamfer_rank(
        &self,
        query_id: u32,
        k: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;

        // 获取 query 的 token 点云
        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        // 全库扫描
        let results = token_chamfer::token_chamfer_top_n(
            query_cloud,
            token_store,
            k as usize,
        );

        // 返回 [[doc_id, score], ...]，score = exp(-2 * distance)
        Ok(results
            .into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// 对指定候选列表做 token Chamfer 重排（multigrid 用）
    #[napi]
    pub fn token_chamfer_rerank_list(
        &self,
        query_id: u32,
        doc_ids: Vec<u32>,
        top_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self.token_cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        use rayon::prelude::*;
        let mut scores: Vec<(u32, f64)> = doc_ids.par_iter().filter_map(|&did| {
            token_store.get_cloud(did).map(|doc| {
                let dist = token_chamfer::token_pq_chamfer(query_cloud, doc);
                (did, dist)
            })
        }).collect();

        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_n as usize);

        Ok(scores.into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// V11 两阶段检索：token 质心粗筛 + 全 token Chamfer 精排
    ///
    /// 第一阶段：query tokens vs 每个文档质心（快，每文档 1 个 4096d 质心）
    /// 第二阶段：对粗筛 top-M 文档做完整 token_pq_chamfer
    ///
    /// 相比 tokenChamferRank 的全库精排，这种方式可减少 80%+ 的计算量。
    ///
    /// @param queryId - query 文档 ID（在 query_token_clouds.sqlite 中的 file_id）
    /// @param k - 最终返回 top-K 结果数
    /// @param coarseTop - 粗筛候选池大小（如 200）
    /// @param topN - 精排后返回数
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn token_chamfer_two_stage(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let centroids = self
            .token_centroids
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 质心未计算，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;

        // 获取 query 的 token 点云
        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        // 两阶段检索
        let results = token_chamfer::token_chamfer_two_stage(
            query_cloud,
            token_store,
            centroids,
            coarse_top as usize,
            top_n as usize,
        );

        // 返回 [[doc_id, score], ...]，score = exp(-2 * distance)
        Ok(results
            .into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// 密度加权 Token Chamfer 两阶段检索（消融实验）
    ///
    /// 与 tokenChamferTwoStage 相同的粗筛，精排阶段使用密度加权 PQ-Chamfer。
    /// 密度 = 1 / mean_knn_distance，文档端密集区域的 token 匹配获得更高权重。
    ///
    /// @param queryId - query 文档 ID
    /// @param coarseTop - 粗筛候选池大小
    /// @param topN - 精排后返回数
    /// @param densityK - KNN K 值（如 3, 5, 7）
    /// @param weightQuery - 是否也对 query 端加权
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn token_chamfer_two_stage_density(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
        density_k: u32,
        weight_query: bool,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let centroids = self
            .token_centroids
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 质心未计算，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        let results = token_chamfer::token_chamfer_two_stage_density(
            query_cloud,
            token_store,
            centroids,
            coarse_top as usize,
            top_n as usize,
            density_k as usize,
            weight_query,
        );

        Ok(results
            .into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// PQ 重建向量消融实验：token_chamfer_two_stage 的 PQ 重建版
    ///
    /// 与 tokenChamferTwoStage 相同的粗筛，但精排阶段用 PQ 码本重建的近似向量
    /// 替代原始 f32 向量，用于验证 PQ 重建的信息损失对精排质量的影响。
    ///
    /// @param queryId - query 文档 ID
    /// @param coarseTop - 粗筛候选池大小
    /// @param topN - 精排后返回数
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn token_chamfer_two_stage_pq_recon(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let centroids = self
            .token_centroids
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 质心未计算，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let pq = self
            .pq_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "PQ store 未加载".to_string(),
            ))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        let results = token_chamfer::token_chamfer_two_stage_pq_recon(
            query_cloud,
            token_store,
            centroids,
            &pq.codebook,
            coarse_top as usize,
            top_n as usize,
        );

        Ok(results
            .into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// 方案1：Token 采样建图 + PDE 管线
    ///
    /// 与 tokenChamferPipeline 不同，这里用 token 空间建图（而非句子级 VT-Aligned），
    /// 解决 c0 语义空间与图结构不匹配的问题。
    ///
    /// 流程:
    /// 1. 从 query_token_store 获取 query 的 token 点云
    /// 2. token_chamfer_two_stage 取 top-55 候选
    /// 3. 对 55 个文档用 token_sampled_chamfer(sample_n=20) 建距离矩阵
    /// 4. build_knn(k=3) + compute_advection + solve_pde
    /// 5. c0 用 token chamfer 距离: exp(-2 * dist)
    ///
    /// @param queryId - query 文档 ID
    /// @param k - 最终返回 top-K
    /// @param topN - 粗筛后候选数
    /// @param sampleN - 每文档采样 token 数（推荐 20）
    /// @returns [[doc_id, score], ...] 按 PDE 浓度降序
    #[napi]
    pub fn token_chamfer_pipeline_sampled(
        &self,
        query_id: u32,
        k: u32,
        top_n: u32,
        sample_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let centroids = self
            .token_centroids
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 质心未计算，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;

        // 1. 获取 query 的 token 点云
        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        // 2. 两阶段检索取 top_n 候选
        let candidates = token_chamfer::token_chamfer_two_stage(
            query_cloud,
            token_store,
            centroids,
            100, // 粗筛 100
            top_n as usize,
        );
        let n = candidates.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // 保存 token Chamfer 距离用于 c0
        let chamfer_dists: Vec<f64> = candidates.iter().map(|&(_, d)| d).collect();

        // 3. 用采样 token Chamfer 建距离矩阵（token 空间建图）
        let doc_clouds: Vec<&cloud_store::DocumentCloud> = candidates
            .iter()
            .map(|&(id, _)| token_store.get_cloud(id).unwrap())
            .collect();

        let dist_matrix = token_chamfer::token_sampled_distance_matrix(
            &doc_clouds,
            sample_n as usize,
        );

        // 4. KNN 图 (k=3)
        let knn_k = 3usize.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 对流系数：用 query 质心
        let query_centroid = self.compute_query_centroid(query_cloud);
        // 需要句子级点云来计算对流系数，但这里我们用 token 质心来近似
        // 将每个文档的 token 质心作为单句子点云
        let doc_centroids_vecs: Vec<Vec<f32>> = doc_clouds
            .iter()
            .map(|doc| {
                let mut c = vec![0.0f32; pq_chamfer::FULL_DIM];
                for i in 0..doc.n_sentences {
                    let tok = doc.sentence(i);
                    for (j, v) in tok.iter().enumerate() {
                        c[j] += v;
                    }
                }
                let inv = 1.0 / doc.n_sentences as f32;
                c.iter_mut().for_each(|v| *v *= inv);
                c
            })
            .collect();

        let centroid_refs: Vec<Vec<&[f32]>> = doc_centroids_vecs
            .iter()
            .map(|c| vec![c.as_slice()])
            .collect();

        let u_matrix = vt_distance::compute_advection(
            &query_centroid,
            &centroid_refs,
            &adj,
            n,
            0.3,
        );

        // 5. 初始浓度 c0 = exp(-2 * token_chamfer_dist)
        let c0: Vec<f64> = chamfer_dists
            .iter()
            .map(|&d| (-2.0 * d).exp())
            .collect();

        // 6. PDE 求解 (D=0.15, max_iter=50, epsilon=1e-3)
        let c_final = pde::solve_pde(&c0, &adj, &u_matrix, n, 0.15, 50, 1e-3);

        // 7. 按浓度降序排序
        let mut ranked: Vec<(u32, f64)> = candidates
            .iter()
            .zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k as usize);

        Ok(ranked
            .into_iter()
            .map(|(id, score)| vec![id as f64, score])
            .collect())
    }

    /// Token 梯度对流 PDE 管线
    ///
    /// 跨粒度融合：句子级图结构 + token 级对流驱动
    ///
    /// 流程:
    /// 1. cosine 粗筛 top-N → 句子级 VT-Aligned 距离矩阵 → KNN 图
    /// 2. C0 = exp(-2 * VT句子级距离)（句子级信号）
    /// 3. 获取 query 的 token 点云，对 top-N 文档计算 token Chamfer 分数 S_i
    /// 4. 对流场 u_ij = beta * (S_j - S_i)（token 级信号）
    /// 5. 跑 token 梯度 PDE（跨粒度融合）
    /// 6. 按 PDE 浓度降序返回 top-K
    ///
    /// @param queryBuf - query 向量 (4096d f32 Buffer)
    /// @param queryId - query 在 query_token_store 中的 ID（用于获取 token 点云）
    /// @param k - 返回 top-K
    /// @param topN - 粗筛候选数
    /// @param beta - 对流强度（默认 1.0）
    /// @returns [{id, score}] 按 PDE 浓度降序
    #[napi]
    pub fn token_gradient_pde_pipeline(
        &self,
        query_buf: Buffer,
        query_id: u32,
        k: u32,
        top_n: u32,
        beta: Option<f64>,
    ) -> napi::Result<Vec<FullscanHit>> {
        // 检查句子级点云（用于图构建和 C0）
        let sentence_store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("句子级点云未加载，请先调用 loadClouds".to_string()))?;

        // 检查 token 级点云（用于对流场）
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_token_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;

        // 解析 query 向量
        let query_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                query_buf.as_ptr() as *const f32,
                query_buf.len() / std::mem::size_of::<f32>(),
            )
        };
        if query_slice.len() != pq_chamfer::FULL_DIM {
            return Err(napi::Error::from_reason(format!(
                "query 维度不匹配: 期望 {}，实际 {}",
                pq_chamfer::FULL_DIM,
                query_slice.len()
            )));
        }

        let beta_val = beta.unwrap_or(1.0);
        let k_usize = k as usize;
        let top_n_usize = top_n as usize;

        // ===== 第一部分：句子级图结构（复用 shape_cfd 前半部分） =====

        // 1. cosine 全库排序 top-N
        let candidates = vt_distance::cosine_top_n(query_slice, sentence_store, top_n_usize);
        let n = candidates.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // 收集候选文档的句子级点云
        let clouds: Vec<Vec<&[f32]>> = candidates
            .iter()
            .map(|&(id, _)| {
                sentence_store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        // 过滤掉空点云（防御性）
        if clouds.iter().any(|c| c.is_empty()) {
            return Ok(candidates
                .into_iter()
                .take(k_usize)
                .map(|(id, dist)| FullscanHit { id, score: 1.0 - dist as f64 })
                .collect());
        }

        // 2. VT-Aligned 距离矩阵 → KNN 图
        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);
        let knn_k = 3usize.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 3. C0 = exp(-2 * VT句子级距离)（句子级信号）
        let c0: Vec<f64> = clouds
            .iter()
            .map(|cloud| (-2.0 * vt_distance::vt_aligned_query_doc(query_slice, cloud) as f64).exp())
            .collect();

        // ===== 第二部分：token 级对流场 =====

        // 获取 query 的 token 点云
        let query_cloud = query_token_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        // 对每个候选文档计算 token Chamfer 分数
        let candidate_ids: Vec<u32> = candidates.iter().map(|&(id, _)| id).collect();
        let token_scores_raw: Vec<f64> = candidate_ids
            .iter()
            .map(|&doc_id| {
                if let Some(doc_cloud) = token_store.get_cloud(doc_id) {
                    let dist = token_chamfer::token_pq_chamfer(query_cloud, doc_cloud);
                    (-2.0 * dist).exp()  // 转为相似度分数 [0,1]
                } else {
                    0.0
                }
            })
            .collect();

        // 归一化到 [0, 1]（min-max）
        let s_min = token_scores_raw.iter().cloned().fold(f64::MAX, f64::min);
        let s_max = token_scores_raw.iter().cloned().fold(f64::MIN, f64::max);
        let range = s_max - s_min;
        let norm_scores: Vec<f64> = if range > 1e-8 {
            token_scores_raw.iter().map(|&s| (s - s_min) / range).collect()
        } else {
            vec![0.5; token_scores_raw.len()]
        };

        // ===== 第三部分：Token 梯度 PDE 求解 =====

        // 默认参数：D=0.15, gamma=0.2, dt=0.03, maxIter=60, epsilon=1e-3
        let c_final = pde::solve_token_gradient_pde(
            &c0,
            &adj,
            &norm_scores,
            beta_val,
            0.15,   // diff_coeff
            0.2,    // gamma (Allen-Cahn)
            0.03,   // dt
            60,     // max_iter
            1e-3,   // epsilon
        );

        // ===== 排序返回 =====
        let mut ranked: Vec<(u32, f64)> = candidates
            .iter()
            .zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k_usize);

        Ok(ranked
            .into_iter()
            .map(|(id, score)| FullscanHit { id, score })
            .collect())
    }

    /// 方案3：极弱 PDE 验证（D=0.01, alpha=0.01）
    ///
    /// 与 tokenChamferPipeline 相同流程，但 PDE 参数极弱，
    /// 用于验证"极弱 PDE 不改变排序"的数学预测。
    /// 预期 NDCG 应与纯 token Chamfer 直排相同。
    ///
    /// @param queryId - query 文档 ID
    /// @param k - 最终返回 top-K
    /// @param topN - 粗筛后候选数
    /// @returns [[doc_id, score], ...] 按 PDE 浓度降序
    #[napi]
    pub fn token_chamfer_pipeline_weak(
        &self,
        query_id: u32,
        k: u32,
        top_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "query token 点云未加载，请先调用 loadTokenCloudsSqlite".to_string(),
            ))?;
        let sentence_store = self
            .cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "句子级点云未加载，请先调用 loadClouds".to_string(),
            ))?;

        // 1. 获取 query 的 token 点云
        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 在 query token store 中不存在", query_id))
        })?;

        // 2. token Chamfer 全库扫描取 top_n
        let candidates = token_chamfer::token_chamfer_top_n(
            query_cloud,
            token_store,
            top_n as usize,
        );
        let n = candidates.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        let chamfer_dists: Vec<f64> = candidates.iter().map(|&(_, d)| d).collect();

        // 3. 从句子级 cloud_store 获取候选文档的句子点云
        let clouds: Vec<Vec<&[f32]>> = candidates
            .iter()
            .map(|&(id, _)| {
                sentence_store
                    .get_cloud(id)
                    .map(|doc| doc.as_slice_refs())
                    .unwrap_or_default()
            })
            .collect();

        // 如果有文档在句子级点云中不存在，退化为纯 token Chamfer
        if clouds.iter().any(|c| c.is_empty()) {
            let results: Vec<Vec<f64>> = candidates
                .iter()
                .take(k as usize)
                .map(|&(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
                .collect();
            return Ok(results);
        }

        // 4. vt_aligned doc-doc 距离矩阵
        let dist_matrix = vt_distance::compute_vt_distance_matrix(&clouds);

        // 5. KNN 图 (k=3)
        let knn_k = 3usize.min(n.saturating_sub(1));
        let adj = vt_distance::build_knn(n, knn_k, &dist_matrix);

        // 对流系数（用 query 质心，极弱 alpha=0.01）
        let query_centroid = self.compute_query_centroid(query_cloud);
        let u_matrix = vt_distance::compute_advection(
            &query_centroid,
            &clouds,
            &adj,
            n,
            0.01, // 极弱对流
        );

        // 6. c0 = exp(-2 * token_chamfer_dist)
        let c0: Vec<f64> = chamfer_dists
            .iter()
            .map(|&d| (-2.0 * d).exp())
            .collect();

        // 7. 极弱 PDE: D=0.01, max_iter=50, epsilon=1e-3
        let c_final = pde::solve_pde(&c0, &adj, &u_matrix, n, 0.01, 50, 1e-3);

        // 8. 按浓度降序排序
        let mut ranked: Vec<(u32, f64)> = candidates
            .iter()
            .zip(c_final.iter())
            .map(|(&(id, _), &score)| (id, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k as usize);

        Ok(ranked
            .into_iter()
            .map(|(id, score)| vec![id as f64, score])
            .collect())
    }

    /// V14: 倒排索引两阶段检索
    /// 用 token 倒排索引替代质心 cosine 粗筛
    ///
    /// @param queryId - query ID
    /// @param coarseTop - 倒排粗筛候选数（如 200）
    /// @param topN - 精排返回数
    /// @param nProbe - 每个子空间探查的码本中心数（1=快, 3=准）
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn token_inverted_two_stage(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
        n_probe: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_store = self
            .query_token_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let inv_index = self
            .token_inverted_index
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("倒排索引未构建".to_string()))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let results = inverted_index::inverted_two_stage(
            query_cloud,
            token_store,
            inv_index,
            coarse_top as usize,
            top_n as usize,
            n_probe as usize,
        );

        Ok(results
            .into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// V14c: 快速倒排索引（大候选池 + argmin/select_nth 优化）
    #[napi]
    pub fn token_inverted_fast(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
        n_probe: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self.token_cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let inv_index = self.token_inverted_index.as_ref()
            .ok_or_else(|| napi::Error::from_reason("倒排索引未构建".to_string()))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let results = inverted_index::inverted_two_stage_fast(
            query_cloud, token_store, inv_index,
            coarse_top as usize, top_n as usize, n_probe as usize,
        );

        Ok(results.into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// V14-PQ ADC 两阶段检索：质心粗筛 + ADC PQ-Chamfer 精排
    /// 用 PQ 量化后的向量做精排，无需原始 f32 向量
    ///
    /// @param queryId - query 文档 ID
    /// @param coarseTop - 粗筛候选数
    /// @param topN - 精排返回数
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn token_adc_two_stage(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let pq = self.pq_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("PQ store 未加载".to_string()))?;
        let query_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let centroids = self.token_centroids.as_ref()
            .ok_or_else(|| napi::Error::from_reason("质心未计算".to_string()))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let results = adc_chamfer::centroid_adc_two_stage(
            query_cloud, pq, centroids,
            coarse_top as usize, top_n as usize,
        );

        Ok(results.into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    // ===== V15: Multi-Probe Retrieval =====

    /// 预计算最强 token 代表向量（Probe 2 离线阶段）
    ///
    /// 对每个文档，取离质心最远的 top-K 个 token 的均值向量。
    /// 必须在 loadTokenCloudsSqlite 之后调用。
    ///
    /// @param topK - 每文档取离质心最远的前几个 token（推荐 3）
    /// @returns 预计算的文档数
    #[napi]
    pub fn precompute_max_token_repr(&mut self, top_k: u32) -> napi::Result<u32> {
        let token_store = self
            .token_cloud_store
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let centroids = self
            .token_centroids
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("质心未计算".to_string()))?;

        let t0 = std::time::Instant::now();
        let reprs = multi_probe::precompute_max_token_repr(
            token_store,
            centroids,
            top_k as usize,
        );
        let count = reprs.len() as u32;
        let mem_bytes: usize = reprs.iter().map(|r| {
            r.vector.len() * 4 + r.sub_norms.len() * 4
        }).sum();

        eprintln!(
            "[LawVexus] Max-token repr 预计算完成: {} 文档, {:.1} MB, {:.1}s",
            count,
            mem_bytes as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64(),
        );

        self.max_token_reprs = Some(reprs);
        Ok(count)
    }

    /// 多路粗筛检索（Multi-Probe Retrieve）
    ///
    /// 组合 3 个探测器的结果（质心 Chamfer + 最强 token + 倒排索引），
    /// 用指定策略合并，返回候选列表。
    ///
    /// @param queryId - query ID
    /// @param perProbeTop - 每个 probe 返回的候选数（如 200）
    /// @param mergedTop - 合并后返回的候选数（如 200）
    /// @param mergeStrategy - 合并策略: "rrf" | "max" | "hit"
    /// @param nProbeInv - 倒排索引的 n_probe 参数（如 1）
    /// @param useInverted - 是否使用倒排索引探针
    /// @returns [[doc_id, merged_score], ...] 按 merged_score 降序
    #[napi]
    pub fn multi_probe_retrieve(
        &self,
        query_id: u32,
        per_probe_top: u32,
        merged_top: u32,
        merge_strategy: String,
        n_probe_inv: u32,
        use_inverted: bool,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self.token_cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let centroids = self.token_centroids.as_ref()
            .ok_or_else(|| napi::Error::from_reason("质心未计算".to_string()))?;
        let max_token_reprs = self.max_token_reprs.as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "max_token_repr 未预计算，请先调用 precomputeMaxTokenRepr".to_string(),
            ))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let strategy = match merge_strategy.as_str() {
            "rrf" => multi_probe::MergeStrategy::Rrf,
            "max" => multi_probe::MergeStrategy::MaxScore,
            "hit" => multi_probe::MergeStrategy::HitMax,
            _ => return Err(napi::Error::from_reason(
                format!("未知的合并策略: {}，可选: rrf, max, hit", merge_strategy),
            )),
        };

        let inv_index = if use_inverted {
            self.token_inverted_index.as_ref()
        } else {
            None
        };

        let result = multi_probe::multi_probe_retrieve(
            query_cloud,
            token_store,
            centroids,
            max_token_reprs,
            inv_index,
            per_probe_top as usize,
            merged_top as usize,
            strategy,
            n_probe_inv as usize,
        );

        // 将 doc_idx 转回 doc_id
        Ok(result
            .merged
            .into_iter()
            .map(|(idx, score)| {
                let doc_id = token_store.documents[idx].doc_id;
                vec![doc_id as f64, score]
            })
            .collect())
    }

    /// 多路粗筛 + token Chamfer 精排
    ///
    /// 先用多路粗筛取 coarseTop 候选，然后对候选做全 token PQ-Chamfer 精排。
    ///
    /// @param queryId - query ID
    /// @param coarseTop - 多路粗筛候选数
    /// @param topN - 精排后返回数
    /// @param mergeStrategy - 合并策略: "rrf" | "max" | "hit"
    /// @param nProbeInv - 倒排索引的 n_probe 参数
    /// @param perProbeTop - 每个 probe 返回的候选数
    /// @param useInverted - 是否使用倒排索引探针
    /// @returns [[doc_id, score], ...] score = exp(-2 * chamfer_distance)
    #[napi]
    pub fn multi_probe_two_stage(
        &self,
        query_id: u32,
        coarse_top: u32,
        top_n: u32,
        merge_strategy: String,
        n_probe_inv: u32,
        per_probe_top: u32,
        use_inverted: bool,
    ) -> napi::Result<Vec<Vec<f64>>> {
        let token_store = self.token_cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let centroids = self.token_centroids.as_ref()
            .ok_or_else(|| napi::Error::from_reason("质心未计算".to_string()))?;
        let max_token_reprs = self.max_token_reprs.as_ref()
            .ok_or_else(|| napi::Error::from_reason(
                "max_token_repr 未预计算".to_string(),
            ))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let strategy = match merge_strategy.as_str() {
            "rrf" => multi_probe::MergeStrategy::Rrf,
            "max" => multi_probe::MergeStrategy::MaxScore,
            "hit" => multi_probe::MergeStrategy::HitMax,
            _ => return Err(napi::Error::from_reason(
                format!("未知的合并策略: {}", merge_strategy),
            )),
        };

        let inv_index = if use_inverted {
            self.token_inverted_index.as_ref()
        } else {
            None
        };

        // 1. 多路粗筛
        let coarse_result = multi_probe::multi_probe_retrieve(
            query_cloud,
            token_store,
            centroids,
            max_token_reprs,
            inv_index,
            per_probe_top as usize,
            coarse_top as usize,
            strategy,
            n_probe_inv as usize,
        );

        // 2. 精排：对候选做 token PQ-Chamfer
        use rayon::prelude::*;
        let mut fine_scores: Vec<(u32, f64)> = coarse_result
            .merged
            .par_iter()
            .filter_map(|&(idx, _)| {
                let doc = &token_store.documents[idx];
                let dist = token_chamfer::token_pq_chamfer(query_cloud, doc);
                Some((doc.doc_id, dist))
            })
            .collect();

        fine_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        fine_scores.truncate(top_n as usize);

        Ok(fine_scores
            .into_iter()
            .map(|(id, dist)| vec![id as f64, (-2.0 * dist).exp()])
            .collect())
    }

    /// 多路粗筛 recall 统计
    ///
    /// 返回各路探测器单独的候选 doc_id 列表，以及合并后的候选列表，
    /// 供 JS 端计算 recall@K。
    ///
    /// @param queryId - query ID
    /// @param perProbeTop - 每个 probe 返回的候选数
    /// @param mergedTop - 合并后返回的候选数
    /// @param mergeStrategy - 合并策略
    /// @param nProbeInv - 倒排索引 n_probe
    /// @returns { probe1Ids, probe2Ids, probe3Ids, mergedIds }
    #[napi]
    pub fn multi_probe_recall(
        &self,
        query_id: u32,
        per_probe_top: u32,
        merged_top: u32,
        merge_strategy: String,
        n_probe_inv: u32,
    ) -> napi::Result<MultiProbeRecallResult> {
        let token_store = self.token_cloud_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("token 点云未加载".to_string()))?;
        let query_store = self.query_token_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("query token 点云未加载".to_string()))?;
        let centroids = self.token_centroids.as_ref()
            .ok_or_else(|| napi::Error::from_reason("质心未计算".to_string()))?;
        let max_token_reprs = self.max_token_reprs.as_ref()
            .ok_or_else(|| napi::Error::from_reason("max_token_repr 未预计算".to_string()))?;

        let query_cloud = query_store.get_cloud(query_id).ok_or_else(|| {
            napi::Error::from_reason(format!("query_id={} 不存在", query_id))
        })?;

        let strategy = match merge_strategy.as_str() {
            "rrf" => multi_probe::MergeStrategy::Rrf,
            "max" => multi_probe::MergeStrategy::MaxScore,
            "hit" => multi_probe::MergeStrategy::HitMax,
            _ => return Err(napi::Error::from_reason(
                format!("未知的合并策略: {}", merge_strategy),
            )),
        };

        let per_top = per_probe_top as usize;

        // 各路单独跑
        let probe1 = multi_probe::centroid_probe(query_cloud, centroids, per_top);
        let probe2 = multi_probe::max_token_probe(query_cloud, max_token_reprs, per_top);
        let probe3 = if let Some(idx) = self.token_inverted_index.as_ref() {
            multi_probe::inverted_probe(query_cloud, idx, per_top, n_probe_inv as usize)
        } else {
            Vec::new()
        };

        // 合并
        let probes = if probe3.is_empty() {
            vec![probe1.clone(), probe2.clone()]
        } else {
            vec![probe1.clone(), probe2.clone(), probe3.clone()]
        };
        let merged = multi_probe::merge_probes(&probes, strategy, merged_top as usize);

        // 转 doc_idx -> doc_id
        let to_ids = |items: &[(usize, f32)]| -> Vec<u32> {
            items.iter().map(|&(idx, _)| token_store.documents[idx].doc_id).collect()
        };

        Ok(MultiProbeRecallResult {
            probe1_ids: to_ids(&probe1),
            probe2_ids: to_ids(&probe2),
            probe3_ids: to_ids(&probe3),
            merged_ids: merged.merged.iter().map(|&(idx, _)| token_store.documents[idx].doc_id).collect(),
        })
    }

    /// 保存 PQ 编码到 SQLite
    ///
    /// @param outputPath - 输出 SQLite 文件路径
    /// @returns 保存结果描述字符串
    #[napi]
    pub fn save_pq_store(&self, output_path: String) -> napi::Result<String> {
        let pq = self.pq_store.as_ref()
            .ok_or_else(|| napi::Error::from_reason("PQ store 未加载".to_string()))?;
        pq.save_to_sqlite(&output_path)
            .map_err(|e| napi::Error::from_reason(e))?;
        Ok(format!("PQ store 已保存到 {}", output_path))
    }
}

// ============================================================================
// LawVexus 内部辅助方法
// ============================================================================

impl LawVexus {
    /// 从 token 点云的 DocumentCloud 计算质心向量（用于对流系数计算）
    fn compute_query_centroid(&self, cloud: &cloud_store::DocumentCloud) -> Vec<f32> {
        let n = cloud.n_sentences;
        if n == 0 {
            return vec![0.0f32; pq_chamfer::FULL_DIM];
        }
        let mut centroid = vec![0.0f64; pq_chamfer::FULL_DIM];
        for i in 0..n {
            let sent = cloud.sentence(i);
            for d in 0..pq_chamfer::FULL_DIM {
                centroid[d] += sent[d] as f64;
            }
        }
        let inv = 1.0 / n as f64;
        centroid.iter().map(|&v| (v * inv) as f32).collect()
    }
}

// ============================================================================
// AsyncTask: SQLite 恢复任务
// ============================================================================

/// 从 SQLite 恢复索引（异步任务）
///
/// 在独立线程中从数据库加载向量数据并填充 HNSW 索引，
/// 不阻塞 Node.js 事件循环。
pub struct RecoverTask {
    index: Option<Arc<RwLock<usearch::Index>>>,
    db_path: String,
    table: String,
    filter: Option<String>,
    dimensions: u32,
}

impl Task for RecoverTask {
    type Output = u32;
    type JsValue = u32;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let index_arc = self
            .index
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("索引未初始化".to_string()))?;

        let conn = rusqlite::Connection::open(&self.db_path)
            .map_err(|e| napi::Error::from_reason(format!("打开数据库失败: {}", e)))?;

        // 根据表类型构建查询 SQL
        let sql = match self.table.as_str() {
            "chunks" => {
                if let Some(ref filter) = self.filter {
                    format!(
                        "SELECT c.id, c.vector FROM chunks c \
                         JOIN files f ON c.file_id = f.id \
                         WHERE f.category = '{}' AND c.vector IS NOT NULL",
                        filter
                    )
                } else {
                    "SELECT id, vector FROM chunks WHERE vector IS NOT NULL".to_string()
                }
            }
            "cases" => "SELECT id, vector FROM cases WHERE vector IS NOT NULL".to_string(),
            "judgments" => {
                "SELECT id, vector FROM judgments WHERE vector IS NOT NULL".to_string()
            }
            "literature" => {
                "SELECT id, vector FROM literature WHERE vector IS NOT NULL".to_string()
            }
            _ => {
                return Err(napi::Error::from_reason(format!(
                    "未知表类型: {}",
                    self.table
                )));
            }
        };

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| napi::Error::from_reason(format!("Prepare 失败: {}", e)))?;

        let expected_byte_len = self.dimensions as usize * std::mem::size_of::<f32>();

        // 获取写锁
        let index = index_arc
            .write()
            .map_err(|e| napi::Error::from_reason(format!("写锁获取失败: {}", e)))?;

        let mut count = 0u32;
        let mut skipped = 0u32;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| napi::Error::from_reason(format!("Query 失败: {}", e)))?;

        for row_result in rows {
            if let Ok((id, vector_bytes)) = row_result {
                if vector_bytes.len() == expected_byte_len {
                    let vec_slice: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            vector_bytes.as_ptr() as *const f32,
                            self.dimensions as usize,
                        )
                    };

                    // 自动扩容
                    if index.size() + 1 >= index.capacity() {
                        let new_cap = (index.capacity() as f64 * 1.5) as usize;
                        let _ = index.reserve(new_cap);
                    }

                    if index.add(id as u64, vec_slice).is_ok() {
                        count += 1;
                    }
                } else {
                    skipped += 1;
                }
            }
        }

        if skipped > 0 {
            eprintln!(
                "[LawVexus] ⚠️ 恢复跳过 {} 条记录(维度不匹配，期望 {} 字节)",
                skipped, expected_byte_len
            );
        }

        Ok(count)
    }

    fn resolve(&mut self, _env: napi::Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}
