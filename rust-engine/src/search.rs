// law-vexus/src/search.rs
// D9-C: 级联检索算法设计 — 四阶段级联管线 + RRF 融合排序
//
// 级联流程:
//   Phase 1: 法条检索（最高优先级）
//   Phase 2: 案例 + 裁判书级联检索（含 SQLite 锚点加权）
//   Phase 3: 文献独立检索
//   Phase 4: RRF 融合排序（多信号融合）

use anyhow::{anyhow, Result};
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

use crate::index_manager::IndexManager;

// ============================================================================
// 数据结构
// ============================================================================

/// 级联检索参数
pub struct CascadeParams {
    /// 法条检索 Top-K
    pub statute_k: u32,
    /// 案例检索 Top-K
    pub case_k: u32,
    /// 裁判书检索 Top-K
    pub judgment_k: u32,
    /// 文献检索 Top-K
    pub literature_k: u32,
    /// 锚点法条加权倍数（引用了命中法条的案例得分提升）
    pub anchor_boost: f64,
    /// RRF 融合常数（默认 60.0，来自 VCP 报告 07/10）
    pub rrf_k: f64,
}

impl Default for CascadeParams {
    fn default() -> Self {
        Self {
            statute_k: 8,
            case_k: 5,
            judgment_k: 5,
            literature_k: 3,
            anchor_boost: 1.5,
            rrf_k: 60.0,
        }
    }
}

impl CascadeParams {
    /// 从各个 K 值快速构建参数
    pub fn with_k(statute_k: u32, case_k: u32, judgment_k: u32, literature_k: u32) -> Self {
        Self {
            statute_k,
            case_k,
            judgment_k,
            literature_k,
            ..Default::default()
        }
    }
}

/// 级联检索结果（四类分层返回）
pub struct CascadeSearchResult {
    /// 法条检索结果（最高优先级）
    pub statutes: Vec<(u64, f64)>,
    /// 案例检索结果（含锚点加权）
    pub cases: Vec<(u64, f64)>,
    /// 裁判书检索结果
    pub judgments: Vec<(u64, f64)>,
    /// 文献检索结果
    pub literature: Vec<(u64, f64)>,
    /// 总检索耗时 (ms)
    pub elapsed_ms: f64,
}

// ============================================================================
// 级联检索核心入口
// ============================================================================

/// 四阶级联检索
///
/// 管线流程:
/// 1. 法条索引 HNSW 检索 → 提取命中法条 IDs
/// 2. SQLite 锚点查询 → 案例语义检索 + 锚点加权 → 裁判书检索
/// 3. 文献独立检索
/// 4. RRF 融合排序（当有多信号时）
///
/// # Arguments
/// * `manager` - 索引管理器（包含四类索引）
/// * `db_path` - SQLite 数据库路径（含 case_statute_refs 锚点表）
/// * `query` - 查询向量 (f32 切片，维度应为 4096)
/// * `params` - 级联检索参数
pub fn cascade_search(
    manager: &IndexManager,
    db_path: &str,
    query: &[f32],
    params: &CascadeParams,
) -> Result<CascadeSearchResult> {
    let start = std::time::Instant::now();

    // ===== Phase 1: 法条检索（最先执行，最高优先级）=====
    let statute_results = if let Some(engine) = manager.indices.get("statute_index") {
        engine.search(query, params.statute_k)?
    } else {
        vec![]
    };

    // 提取命中法条 IDs（供 Phase 2 锚点查询使用）
    let statute_ids: Vec<u64> = statute_results.iter().map(|r| r.0).collect();

    // ===== Phase 2: 案例 + 裁判书级联检索 =====

    // Phase 2a: SQLite 锚点查询 — 找出引用了命中法条的案例
    let anchor_case_ids = query_anchor_cases(db_path, &statute_ids)?;

    // Phase 2b: 案例语义检索 + 锚点加权
    let case_results = if let Some(engine) = manager.indices.get("case_index") {
        // 多检索 2x 候选，以便锚点加权后仍有足够结果
        let raw = engine.search(query, params.case_k * 2)?;

        // 锚点加权：引用了命中法条的案例得分提升 anchor_boost 倍
        let boosted: Vec<(u64, f64)> = raw
            .into_iter()
            .map(|(id, score)| {
                if anchor_case_ids.contains(&id) {
                    (id, score * params.anchor_boost) // 锚点案例加权 ×1.5
                } else {
                    (id, score)
                }
            })
            .collect();

        // 按加权后分数重排，取 Top-K
        let mut sorted = boosted;
        sorted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(params.case_k as usize);
        sorted
    } else {
        vec![]
    };

    // Phase 2c: 裁判书检索（独立语义检索）
    let judgment_results = if let Some(engine) = manager.indices.get("judgment_index") {
        engine.search(query, params.judgment_k)?
    } else {
        vec![]
    };

    // ===== Phase 3: 文献独立检索 =====
    let literature_results = if let Some(engine) = manager.indices.get("literature_index") {
        engine.search(query, params.literature_k)?
    } else {
        vec![]
    };

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Ok(CascadeSearchResult {
        statutes: statute_results,
        cases: case_results,
        judgments: judgment_results,
        literature: literature_results,
        elapsed_ms: elapsed,
    })
}

// ============================================================================
// Phase 2a: SQLite 锚点查询
// ============================================================================

/// 查询引用了指定法条的案例 IDs
///
/// 从 `case_statute_refs` 表中查找哪些案例引用了命中的法条，
/// 这些案例在后续语义检索中会获得锚点加权（score × anchor_boost）。
///
/// 如果 `statute_ids` 为空或数据库不可用，返回空集合（不阻断管线）。
fn query_anchor_cases(db_path: &str, statute_ids: &[u64]) -> Result<HashSet<u64>> {
    if statute_ids.is_empty() {
        return Ok(HashSet::new());
    }

    // 尝试打开数据库，失败时返回空集合（优雅降级）
    let conn = match Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[LawVexus] ⚠️ 锚点查询数据库打开失败，跳过锚点加权: {}",
                e
            );
            return Ok(HashSet::new());
        }
    };

    // 检查表是否存在（优雅降级）
    let table_exists: bool = conn
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='case_statute_refs'")
        .and_then(|mut stmt| stmt.exists([]))
        .unwrap_or(false);

    if !table_exists {
        eprintln!(
            "[LawVexus] ⚠️ case_statute_refs 表不存在，跳过锚点加权"
        );
        return Ok(HashSet::new());
    }

    // 构建 IN 子句
    let placeholders: Vec<String> = statute_ids.iter().map(|_| "?".to_string()).collect();
    let sql = format!(
        "SELECT DISTINCT case_id FROM case_statute_refs WHERE statute_id IN ({})",
        placeholders.join(",")
    );

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| anyhow!("锚点查询准备失败: {}", e))?;

    // 构建参数
    let params: Vec<&dyn rusqlite::ToSql> = statute_ids
        .iter()
        .map(|id| id as &dyn rusqlite::ToSql)
        .collect();

    let rows = stmt
        .query_map(params.as_slice(), |row| row.get::<_, i64>(0))
        .map_err(|e| anyhow!("锚点查询执行失败: {}", e))?;

    let mut anchor_ids = HashSet::new();
    for row in rows {
        if let Ok(id) = row {
            anchor_ids.insert(id as u64);
        }
    }

    Ok(anchor_ids)
}

// ============================================================================
// Phase 4: RRF 融合排序
// ============================================================================

/// Reciprocal Rank Fusion（RRF）多信号融合排序
///
/// 当同一结果有多个排序信号时（如向量检索排位 + BM25 排位），
/// 使用 RRF 公式融合：
///
///   `score = Σ α_i / (k + rank_i)`
///
/// 其中 `k` 为常数（默认 60），`rank_i` 为第 i 个排序信号中的排位（1-based）。
///
/// # Arguments
/// * `rankings` - 多个排序列表，每个列表内按分数降序排列
/// * `weights` - 每个排序信号的权重 α
/// * `k` - RRF 常数（VCP 默认 60.0）
///
/// # Returns
/// 融合后的结果列表，按 RRF 分数降序排列
pub fn rrf_fusion(
    rankings: &[Vec<(u64, f64)>],
    weights: &[f64],
    k: f64,
) -> Vec<(u64, f64)> {
    let mut scores: HashMap<u64, f64> = HashMap::new();

    for (rank_list, &alpha) in rankings.iter().zip(weights.iter()) {
        for (rank, (id, _original_score)) in rank_list.iter().enumerate() {
            // rank 是 0-based，RRF 使用 1-based
            let rrf_score = alpha / (k + rank as f64 + 1.0);
            *scores.entry(*id).or_insert(0.0) += rrf_score;
        }
    }

    let mut results: Vec<(u64, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Softmax 温度 K 值分配（借鉴 VCP 报告 07）
///
/// 根据各索引与查询的语义相似度，使用 Softmax 分配不同的 K 值。
/// 语义相关性高的索引获得更多的检索配额。
///
/// # Arguments
/// * `similarities` - 各索引与查询的语义相似度分数
/// * `total_k` - 总检索配额
/// * `temperature` - Softmax 温度参数（默认 1.0，越小分配越集中）
///
/// # Returns
/// 各索引分配的 K 值
pub fn softmax_k_allocation(
    similarities: &[f64],
    total_k: u32,
    temperature: f64,
) -> Vec<u32> {
    if similarities.is_empty() {
        return vec![];
    }

    // Softmax 计算（含温度参数和数值稳定性处理）
    let max_sim = similarities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let exp_sims: Vec<f64> = similarities
        .iter()
        .map(|&s| ((s - max_sim) / temperature).exp())
        .collect();

    let exp_sum: f64 = exp_sims.iter().sum();

    // 分配 K 值（至少每个索引分配 1）
    let mut k_values: Vec<u32> = exp_sims
        .iter()
        .map(|&e| {
            let ratio = e / exp_sum;
            let k = (ratio * total_k as f64).round() as u32;
            std::cmp::max(k, 1) // 每个索引至少 1
        })
        .collect();

    // 调整总和确保不超过 total_k
    let allocated: u32 = k_values.iter().sum();
    if allocated > total_k && !k_values.is_empty() {
        // 从分配最多的索引中减去多余的
        let excess = allocated - total_k;
        let max_idx = k_values
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0);
        if k_values[max_idx] > excess {
            k_values[max_idx] -= excess;
        }
    }

    k_values
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion_basic() {
        // 两个排序信号
        let ranking1 = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let ranking2 = vec![(2, 0.95), (1, 0.85), (4, 0.75)];
        let weights = vec![1.0, 1.0];

        let result = rrf_fusion(&[ranking1, ranking2], &weights, 60.0);

        // ID=1 和 ID=2 都出现在两个列表中（对称排名），RRF 分数相同
        // 两者应该都排在前两名
        assert!(result.len() >= 3);
        let top2_ids: std::collections::HashSet<u64> = result[..2].iter().map(|r| r.0).collect();
        assert!(top2_ids.contains(&1));
        assert!(top2_ids.contains(&2));

        // RRF 分数验证：ID=1 和 ID=2 的 RRF 分数应相同
        let id1_score = result.iter().find(|(id, _)| *id == 1).unwrap().1;
        let id2_score = result.iter().find(|(id, _)| *id == 2).unwrap().1;
        assert!((id1_score - id2_score).abs() < 1e-10); // 对称排名 → 分数相同

        // ID=1/2 应该比只出现一次的 ID=4 得分高
        let id4_score = result.iter().find(|(id, _)| *id == 4).unwrap().1;
        assert!(id1_score > id4_score);
    }

    #[test]
    fn test_rrf_fusion_weighted() {
        let ranking1 = vec![(1, 0.9), (2, 0.8)];
        let ranking2 = vec![(2, 0.95), (3, 0.85)];
        let weights = vec![2.0, 1.0]; // ranking1 权重更高

        let result = rrf_fusion(&[ranking1, ranking2], &weights, 60.0);

        // 验证结果非空且有序
        assert!(result.len() >= 2);
        for i in 1..result.len() {
            assert!(result[i - 1].1 >= result[i].1);
        }
    }

    #[test]
    fn test_rrf_fusion_empty() {
        let result = rrf_fusion(&[], &[], 60.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_k_allocation_uniform() {
        let sims = vec![0.5, 0.5, 0.5, 0.5];
        let k_values = softmax_k_allocation(&sims, 20, 1.0);

        assert_eq!(k_values.len(), 4);
        // 均匀相似度，每个应该分到约 5
        for k in &k_values {
            assert!(*k >= 1);
        }
    }

    #[test]
    fn test_softmax_k_allocation_skewed() {
        let sims = vec![0.9, 0.1, 0.1, 0.1];
        let k_values = softmax_k_allocation(&sims, 20, 1.0);

        assert_eq!(k_values.len(), 4);
        // 第一个相似度最高，应获得最多的 K 值
        assert!(k_values[0] >= k_values[1]);
    }

    #[test]
    fn test_softmax_k_allocation_empty() {
        let result = softmax_k_allocation(&[], 10, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cascade_params_default() {
        let params = CascadeParams::default();
        assert_eq!(params.statute_k, 8);
        assert_eq!(params.case_k, 5);
        assert_eq!(params.judgment_k, 5);
        assert_eq!(params.literature_k, 3);
        assert_eq!(params.anchor_boost, 1.5);
        assert_eq!(params.rrf_k, 60.0);
    }

    #[test]
    fn test_cascade_params_with_k() {
        let params = CascadeParams::with_k(10, 8, 6, 4);
        assert_eq!(params.statute_k, 10);
        assert_eq!(params.case_k, 8);
        assert_eq!(params.judgment_k, 6);
        assert_eq!(params.literature_k, 4);
        // 默认值不变
        assert_eq!(params.anchor_boost, 1.5);
        assert_eq!(params.rrf_k, 60.0);
    }

    #[test]
    fn test_query_anchor_cases_empty_ids() {
        // 空 statute_ids 应返回空集
        let result = query_anchor_cases("/nonexistent.db", &[]).unwrap();
        assert!(result.is_empty());
    }
}
