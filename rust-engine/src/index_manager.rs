// law-vexus/src/index_manager.rs
// D9-A + D9-D: 四类法律索引管理器（含懒加载 + TTL 驱逐）

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::hnsw_engine::LawHnswEngine;

/// 四类法律索引类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LawIndexType {
    Statute,     // 法条 chunks
    Literature,  // 法律文献/学术论文
    Case,        // 案例摘要
    Judgment,    // 裁判书全文分块
}

/// 索引配置条目
pub struct IndexConfig {
    pub index_type: LawIndexType,
    pub name: &'static str,
    pub dimensions: u32,
    pub capacity: u32,
    pub file_name: &'static str,
}

/// 四类索引的标准配置
pub const INDEX_CONFIGS: [IndexConfig; 4] = [
    IndexConfig {
        index_type: LawIndexType::Statute,
        name: "statute_index",
        dimensions: 4096,
        capacity: 50_000,
        file_name: "law_statutes.usearch",
    },
    IndexConfig {
        index_type: LawIndexType::Literature,
        name: "literature_index",
        dimensions: 4096,
        capacity: 100_000,
        file_name: "law_literature.usearch",
    },
    IndexConfig {
        index_type: LawIndexType::Case,
        name: "case_index",
        dimensions: 4096,
        capacity: 50_000,
        file_name: "law_cases.usearch",
    },
    IndexConfig {
        index_type: LawIndexType::Judgment,
        name: "judgment_index",
        dimensions: 4096,
        capacity: 200_000,
        file_name: "law_judgments.usearch",
    },
];

/// 索引驱逐配置（D9-D）
pub struct EvictionConfig {
    pub ttl: Duration,                      // 空闲超时（默认 2 小时）
    pub sweep_interval: Duration,           // 扫描间隔（默认 10 分钟）
    pub permanent_indices: Vec<String>,     // 永驻内存的索引名称
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(2 * 60 * 60),           // 2 小时
            sweep_interval: Duration::from_secs(10 * 60),     // 10 分钟
            permanent_indices: vec![
                "statute_index".to_string(),  // 法条索引常驻（最常用）
                "case_index".to_string(),     // 案例索引常驻
            ],
        }
    }
}

/// 索引管理器（管理四类索引的生命周期）
pub struct IndexManager {
    /// 已加载的索引
    pub indices: HashMap<String, LawHnswEngine>,
    /// 最后使用时间（TTL 驱逐用）
    pub last_used: HashMap<String, Instant>,
    /// 索引存储根目录
    pub store_path: String,
}

/// 内存使用统计
pub struct MemoryStats {
    pub total_bytes: u64,
    pub loaded_count: u32,
    pub per_index: Vec<(String, u64)>,
}

impl IndexManager {
    /// 创建新的索引管理器
    pub fn new(store_path: &str) -> Self {
        Self {
            indices: HashMap::new(),
            last_used: HashMap::new(),
            store_path: store_path.to_string(),
        }
    }

    /// 获取或创建索引（立即加载模式）
    pub fn get_or_create(
        &mut self,
        name: &str,
        dimensions: u32,
        capacity: u32,
    ) -> Result<&LawHnswEngine> {
        if !self.indices.contains_key(name) {
            let file_path = self.index_file_path(name);
            let engine = if std::path::Path::new(&file_path).exists() {
                LawHnswEngine::load(name, &file_path, dimensions, capacity)?
            } else {
                LawHnswEngine::create(name, dimensions, capacity)?
            };
            self.indices.insert(name.to_string(), engine);
        }
        self.last_used.insert(name.to_string(), Instant::now());
        Ok(self.indices.get(name).unwrap())
    }

    /// 懒加载索引（D9-D）：不存在则从磁盘加载或创建空索引
    pub fn lazy_get(
        &mut self,
        name: &str,
        dimensions: u32,
        capacity: u32,
    ) -> Result<&LawHnswEngine> {
        // 刷新访问时间
        self.last_used.insert(name.to_string(), Instant::now());

        if self.indices.contains_key(name) {
            return Ok(self.indices.get(name).unwrap());
        }

        // 懒加载流程（借鉴 VCP 研报 11 第 2.2 节）
        let file_path = self.index_file_path(name);
        let engine = if std::path::Path::new(&file_path).exists() {
            match LawHnswEngine::load(name, &file_path, dimensions, capacity) {
                Ok(e) => {
                    eprintln!(
                        "[LawVexus] ✅ 懒加载索引: {} ({} vectors)",
                        name,
                        e.stats()?.total_vectors
                    );
                    e
                }
                Err(e) => {
                    eprintln!(
                        "[LawVexus] ⚠️ 加载失败，创建空索引: {} ({})",
                        name, e
                    );
                    LawHnswEngine::create(name, dimensions, capacity)?
                }
            }
        } else {
            eprintln!("[LawVexus] 📦 创建新索引: {}", name);
            LawHnswEngine::create(name, dimensions, capacity)?
        };

        self.indices.insert(name.to_string(), engine);
        Ok(self.indices.get(name).unwrap())
    }

    /// 驱逐空闲索引（D9-D，定期调用）
    pub fn evict_idle(&mut self, config: &EvictionConfig) -> Result<Vec<String>> {
        let now = Instant::now();
        let mut evicted = Vec::new();

        let to_evict: Vec<String> = self
            .last_used
            .iter()
            .filter(|(name, last_used)| {
                // 永驻索引不驱逐
                if config.permanent_indices.contains(name) {
                    return false;
                }
                // 超过 TTL 的驱逐
                now.duration_since(**last_used) > config.ttl
            })
            .map(|(name, _)| name.clone())
            .collect();

        for name in &to_evict {
            // 1. 先保存到磁盘
            if let Some(engine) = self.indices.get(name) {
                let file_path = self.index_file_path(name);
                if let Err(e) = engine.save(&file_path) {
                    eprintln!("[LawVexus] ⚠️ 驱逐保存失败: {} ({})", name, e);
                    continue;
                }
            }
            // 2. 从内存移除
            self.indices.remove(name);
            self.last_used.remove(name);
            eprintln!("[LawVexus] 🗑️ 已驱逐空闲索引: {}", name);
            evicted.push(name.clone());
        }

        Ok(evicted)
    }

    /// 保存所有索引到磁盘
    pub fn save_all(&self) -> Result<()> {
        for (name, engine) in &self.indices {
            let file_path = self.index_file_path(name);
            engine.save(&file_path)?;
        }
        Ok(())
    }

    /// 获取内存使用统计
    pub fn memory_stats(&self) -> MemoryStats {
        let mut total = 0u64;
        let mut index_stats = Vec::new();
        for (name, engine) in &self.indices {
            if let Ok(stats) = engine.stats() {
                total += stats.memory_usage_bytes;
                index_stats.push((name.clone(), stats.memory_usage_bytes));
            }
        }
        MemoryStats {
            total_bytes: total,
            loaded_count: self.indices.len() as u32,
            per_index: index_stats,
        }
    }

    /// 根据索引名称获取文件路径
    fn index_file_path(&self, name: &str) -> String {
        let file_name = INDEX_CONFIGS
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.file_name)
            .unwrap_or("unknown.usearch");
        format!("{}/{}", self.store_path, file_name)
    }
}
