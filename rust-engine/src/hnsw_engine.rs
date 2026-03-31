// law-vexus/src/hnsw_engine.rs
// D9-A: USearch HNSW 引擎封装（法律优化参数版）

use anyhow::{anyhow, Result};
use std::sync::{Arc, RwLock};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// 法律 HNSW 引擎参数常量（法律优化版）
pub struct HnswParams {
    pub metric: MetricKind,
    pub quantization: ScalarKind,
    pub connectivity: usize,       // M 参数
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// 法律场景优化参数
/// - cosine: 法律向量来自 SiliconFlow API，不保证归一化
/// - M=24: 法律精度优先，更高召回
/// - ef_construction=200: 更高构建质量
/// - ef_search=128: 法律场景不容许遗漏关键法条
pub const LAW_HNSW_PARAMS: HnswParams = HnswParams {
    metric: MetricKind::Cos,
    quantization: ScalarKind::F32,
    connectivity: 24,
    ef_construction: 200,
    ef_search: 128,
};

/// 索引统计信息
pub struct EngineStats {
    pub name: String,
    pub total_vectors: u32,
    pub dimensions: u32,
    pub capacity: u32,
    pub memory_usage_bytes: u64,
}

/// 法律向量索引（封装 USearch）
pub struct LawHnswEngine {
    index: Arc<RwLock<Index>>,
    dimensions: u32,
    name: String,
}

impl LawHnswEngine {
    /// 创建索引选项（复用参数）
    fn make_options(dimensions: u32) -> IndexOptions {
        IndexOptions {
            dimensions: dimensions as usize,
            metric: LAW_HNSW_PARAMS.metric,
            quantization: LAW_HNSW_PARAMS.quantization,
            connectivity: LAW_HNSW_PARAMS.connectivity,
            expansion_add: LAW_HNSW_PARAMS.ef_construction,
            expansion_search: LAW_HNSW_PARAMS.ef_search,
            multi: false,
        }
    }

    /// 创建新的空索引
    pub fn create(name: &str, dimensions: u32, capacity: u32) -> Result<Self> {
        let options = Self::make_options(dimensions);
        let index = Index::new(&options)
            .map_err(|e| anyhow!("创建索引失败: {:?}", e))?;
        index.reserve(capacity as usize)
            .map_err(|e| anyhow!("预分配容量失败: {:?}", e))?;

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            dimensions,
            name: name.to_string(),
        })
    }

    /// 从磁盘加载索引
    pub fn load(name: &str, path: &str, dimensions: u32, capacity: u32) -> Result<Self> {
        let options = Self::make_options(dimensions);
        let index = Index::new(&options)
            .map_err(|e| anyhow!("创建索引包装失败: {:?}", e))?;
        index.load(path)
            .map_err(|e| anyhow!("加载索引失败: {:?}", e))?;

        // 检查容量并扩容
        if capacity as usize > index.capacity() {
            index.reserve(capacity as usize)
                .map_err(|e| anyhow!("扩容失败: {:?}", e))?;
        }

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            dimensions,
            name: name.to_string(),
        })
    }

    /// 原子保存（.tmp + rename，借鉴 VCP）
    pub fn save(&self, path: &str) -> Result<()> {
        let index = self.index.read()
            .map_err(|e| anyhow!("读锁获取失败: {}", e))?;
        let temp_path = format!("{}.tmp", path);
        index.save(&temp_path)
            .map_err(|e| anyhow!("保存索引失败: {:?}", e))?;
        std::fs::rename(&temp_path, path)?;
        Ok(())
    }

    /// 添加单个向量（带自动扩容）
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        let index = self.index.write()
            .map_err(|e| anyhow!("写锁获取失败: {}", e))?;

        // 维度校验
        if vector.len() != self.dimensions as usize {
            return Err(anyhow!(
                "维度不匹配: 期望 {}, 实际 {}",
                self.dimensions,
                vector.len()
            ));
        }

        // 自动扩容（1.5x 几何增长，借鉴 VCP）
        if index.size() + 1 >= index.capacity() {
            let new_cap = (index.capacity() as f64 * 1.5) as usize;
            let _ = index.reserve(new_cap);
        }

        index.add(id, vector)
            .map_err(|e| anyhow!("添加向量失败: {:?}", e))?;
        Ok(())
    }

    /// 批量添加（更高效，减少锁竞争）
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<u32> {
        let index = self.index.write()
            .map_err(|e| anyhow!("写锁获取失败: {}", e))?;
        let dim = self.dimensions as usize;
        let count = ids.len();

        if vectors.len() != count * dim {
            return Err(anyhow!("批量数据长度不匹配"));
        }

        // 预扩容
        if index.size() + count >= index.capacity() {
            let new_cap = ((index.size() + count) as f64 * 1.5) as usize;
            let _ = index.reserve(new_cap);
        }

        let mut added = 0u32;
        for (i, &id) in ids.iter().enumerate() {
            let start = i * dim;
            let v = &vectors[start..start + dim];
            if index.add(id, v).is_ok() {
                added += 1;
            }
        }
        Ok(added)
    }

    /// 向量检索
    pub fn search(&self, query: &[f32], k: u32) -> Result<Vec<(u64, f64)>> {
        let index = self.index.read()
            .map_err(|e| anyhow!("读锁获取失败: {}", e))?;

        if query.len() != self.dimensions as usize {
            return Err(anyhow!(
                "查询维度不匹配: 期望 {}, 实际 {}",
                self.dimensions,
                query.len()
            ));
        }

        let matches = index.search(query, k as usize)
            .map_err(|e| anyhow!("检索失败: {:?}", e))?;

        let results: Vec<(u64, f64)> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .map(|(&key, &dist)| (key, 1.0 - dist as f64)) // 距离 → 相似度分数
            .collect();

        Ok(results)
    }

    /// 删除向量
    pub fn remove(&self, id: u64) -> Result<()> {
        let index = self.index.write()
            .map_err(|e| anyhow!("写锁获取失败: {}", e))?;
        index.remove(id)
            .map_err(|e| anyhow!("删除向量失败: {:?}", e))?;
        Ok(())
    }

    /// 自动扩容（外部触发，扩容到当前容量的 1.5 倍）
    pub fn auto_expand(&self) -> Result<()> {
        let index = self.index.write()
            .map_err(|e| anyhow!("写锁获取失败: {}", e))?;
        let new_cap = (index.capacity() as f64 * 1.5) as usize;
        index.reserve(new_cap)
            .map_err(|e| anyhow!("扩容失败: {:?}", e))?;
        Ok(())
    }

    /// 获取统计信息
    pub fn stats(&self) -> Result<EngineStats> {
        let index = self.index.read()
            .map_err(|e| anyhow!("读锁获取失败: {}", e))?;
        Ok(EngineStats {
            name: self.name.clone(),
            total_vectors: index.size() as u32,
            dimensions: self.dimensions,
            capacity: index.capacity() as u32,
            memory_usage_bytes: index.memory_usage() as u64,
        })
    }

    /// 获取索引名称
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 获取维度
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// 获取底层索引 Arc（供 RecoverTask 等异步任务使用）
    pub fn index_arc(&self) -> Arc<RwLock<Index>> {
        self.index.clone()
    }
}
