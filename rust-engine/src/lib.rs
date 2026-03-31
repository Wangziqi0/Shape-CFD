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

        self.token_cloud_store = Some(corpus_store);
        self.query_token_store = Some(query_store);
        self.token_centroids = Some(centroids);

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
