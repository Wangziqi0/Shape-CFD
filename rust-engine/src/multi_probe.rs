//! Speculative Multi-Probe Retrieval (V15)
//!
//! 三路粗筛探测器 + RRF/max/hit 合并，验证是否比单路质心 Chamfer 粗筛
//! 捞回更多相关文档（提升 recall@200），从而提升最终 NDCG@10。
//!
//! 三个探测器：
//! - Probe 1: 质心 Chamfer（已有，复用 token_chamfer::token_centroid_chamfer）
//! - Probe 2: 最强 token 探针（每文档取 top-3 离质心最远的 token 的均值）
//! - Probe 3: PQ 签名倒排（已有，复用 inverted_index::TokenInvertedIndex）
//!
//! 合并策略：
//! - RRF: rrf_score(doc) = sum_probe 1/(60 + rank_in_probe)
//! - MaxScore: max(normalized_score_across_probes)
//! - HitMax: hit_count * max(normalized_score)

use crate::cloud_store::{CloudStore, DocumentCloud};
use crate::inverted_index::TokenInvertedIndex;
use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM, FULL_DIM};
use rayon::prelude::*;

// ============================================================================
// Probe 2: 最强 token 代表向量预计算
// ============================================================================

/// 对每个文档，计算"最强 token 代表向量" (max_token_repr)
///
/// 策略：取文档中与质心 cosine distance 最大的 top-K 个 token，
/// 求这 K 个 token 的均值向量。这个向量代表了文档中"质心无法覆盖"的独特语义。
///
/// # 参数
/// - `token_store`: 全库 token 点云
/// - `centroids`: 预计算的每文档质心
/// - `top_k`: 取离质心最远的前几个 token（默认 3）
///
/// # 返回
/// 每文档一个 4096d 向量 + 预计算的子空间范数
pub fn precompute_max_token_repr(
    token_store: &CloudStore,
    centroids: &[Vec<f32>],
    top_k: usize,
) -> Vec<MaxTokenRepr> {
    debug_assert_eq!(
        token_store.documents.len(),
        centroids.len(),
        "文档数与质心数不匹配"
    );

    token_store
        .documents
        .par_iter()
        .zip(centroids.par_iter())
        .map(|(doc, centroid)| {
            let n = doc.n_sentences;
            if n == 0 {
                return MaxTokenRepr {
                    vector: vec![0.0f32; FULL_DIM],
                    sub_norms: vec![0.0f32; NUM_SUBSPACES],
                };
            }

            // 计算每个 token 与质心的 full cosine distance
            let mut dists: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let tok = doc.sentence(i);
                    let d = full_cosine_distance(tok, centroid);
                    (i, d)
                })
                .collect();

            // 按距离降序排序（离质心最远的排前面）
            dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // 取 top-K 个 token 的均值
            let k = top_k.min(n);
            let mut repr = vec![0.0f32; FULL_DIM];
            for &(idx, _) in dists.iter().take(k) {
                let tok = doc.sentence(idx);
                for d in 0..FULL_DIM {
                    repr[d] += tok[d];
                }
            }
            let inv_k = 1.0 / k as f32;
            repr.iter_mut().for_each(|v| *v *= inv_k);

            // 预计算子空间范数
            let sub_norms = compute_sub_norms(&repr);

            MaxTokenRepr {
                vector: repr,
                sub_norms,
            }
        })
        .collect()
}

/// 最强 token 代表向量（含预计算的子空间范数）
pub struct MaxTokenRepr {
    /// 4096d 代表向量
    pub vector: Vec<f32>,
    /// 64 个子空间的 L2 范数
    pub sub_norms: Vec<f32>,
}

// ============================================================================
// Probe 2: 在线查询 — max token 探针得分
// ============================================================================

/// Probe 2 在线阶段：计算 query tokens 与每个文档 max_token_repr 的相似度
///
/// 对每个文档，score = max over query tokens: cos_sim(q_token, max_token_repr)
/// 捕捉"质心被拉偏但有一个 token 特别匹配"的文档。
///
/// # 返回
/// (doc_idx, score) 列表，按 score 降序排列，取 top_n
pub fn max_token_probe(
    query_cloud: &DocumentCloud,
    max_token_reprs: &[MaxTokenRepr],
    top_n: usize,
) -> Vec<(usize, f32)> {
    let nq = query_cloud.n_sentences;
    if nq == 0 {
        return Vec::new();
    }

    // 预计算 query token 的子空间范数
    let q_norms: Vec<Vec<f32>> = (0..nq)
        .map(|i| compute_sub_norms(query_cloud.sentence(i)))
        .collect();

    let mut scores: Vec<(usize, f32)> = max_token_reprs
        .par_iter()
        .enumerate()
        .map(|(doc_idx, repr)| {
            // max over query tokens: 对每个子空间算 cosine similarity，然后平均
            let mut best_sim = f32::MIN;
            for qi in 0..nq {
                let q_vec = query_cloud.sentence(qi);
                // 64 子空间平均 cosine similarity
                let mut sim_sum = 0.0f32;
                for s in 0..NUM_SUBSPACES {
                    let off = s * SUB_DIM;
                    let q_sub = &q_vec[off..off + SUB_DIM];
                    let d_sub = &repr.vector[off..off + SUB_DIM];
                    let q_norm = q_norms[qi][s];
                    let d_norm = repr.sub_norms[s];
                    let denom = q_norm * d_norm;
                    if denom < 1e-16 {
                        // 零向量子空间贡献 0
                    } else {
                        let dot = sub_dot_64d(q_sub, d_sub);
                        sim_sum += dot / (denom + 1e-8);
                    }
                }
                let avg_sim = sim_sum / NUM_SUBSPACES as f32;
                if avg_sim > best_sim {
                    best_sim = avg_sim;
                }
            }
            (doc_idx, best_sim)
        })
        .collect();

    // 按 score 降序排序
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_n);
    scores
}

// ============================================================================
// 多路合并策略
// ============================================================================

/// 合并策略枚举
#[derive(Clone, Copy, Debug)]
pub enum MergeStrategy {
    /// Reciprocal Rank Fusion: sum_probe 1/(60 + rank)
    Rrf,
    /// Max normalized score across probes
    MaxScore,
    /// Hit count * max normalized score
    HitMax,
}

/// 多路合并结果
pub struct MultiProbeResult {
    /// 合并后的 (doc_idx, score) 列表，按 score 降序
    pub merged: Vec<(usize, f64)>,
    /// 各路探测器分别返回的 doc_idx 集合大小（用于统计）
    pub probe_sizes: Vec<usize>,
}

/// 多路结果合并
///
/// 输入：多个 probe 的结果列表，每个是 (doc_idx, score) 按 score 降序排列。
/// 输出：合并后的 (doc_idx, merged_score) 列表。
pub fn merge_probes(
    probe_results: &[Vec<(usize, f32)>],
    strategy: MergeStrategy,
    top_n: usize,
) -> MultiProbeResult {
    use std::collections::HashMap;

    let probe_sizes: Vec<usize> = probe_results.iter().map(|p| p.len()).collect();

    match strategy {
        MergeStrategy::Rrf => {
            // RRF: score(doc) = sum_probe 1/(60 + rank_in_probe)
            let mut doc_scores: HashMap<usize, f64> = HashMap::new();
            for probe in probe_results {
                for (rank, &(doc_idx, _)) in probe.iter().enumerate() {
                    *doc_scores.entry(doc_idx).or_insert(0.0) +=
                        1.0 / (60.0 + rank as f64);
                }
            }
            let mut merged: Vec<(usize, f64)> = doc_scores.into_iter().collect();
            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            merged.truncate(top_n);
            MultiProbeResult {
                merged,
                probe_sizes,
            }
        }
        MergeStrategy::MaxScore => {
            // 每个 probe 的分数归一化到 [0,1]，然后取 max
            let normalized = normalize_probe_results(probe_results);
            let mut doc_scores: HashMap<usize, f64> = HashMap::new();
            for probe in &normalized {
                for &(doc_idx, score) in probe {
                    let entry = doc_scores.entry(doc_idx).or_insert(0.0f64);
                    if score > *entry {
                        *entry = score;
                    }
                }
            }
            let mut merged: Vec<(usize, f64)> = doc_scores.into_iter().collect();
            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            merged.truncate(top_n);
            MultiProbeResult {
                merged,
                probe_sizes,
            }
        }
        MergeStrategy::HitMax => {
            // hit_count * max(normalized_score)
            let normalized = normalize_probe_results(probe_results);
            let mut doc_max_score: HashMap<usize, f64> = HashMap::new();
            let mut doc_hit_count: HashMap<usize, f64> = HashMap::new();
            for probe in &normalized {
                for &(doc_idx, score) in probe {
                    let max_entry = doc_max_score.entry(doc_idx).or_insert(0.0f64);
                    if score > *max_entry {
                        *max_entry = score;
                    }
                    *doc_hit_count.entry(doc_idx).or_insert(0.0) += 1.0;
                }
            }
            let mut merged: Vec<(usize, f64)> = doc_max_score
                .into_iter()
                .map(|(idx, max_s)| {
                    let hits = doc_hit_count.get(&idx).copied().unwrap_or(1.0);
                    (idx, hits * max_s)
                })
                .collect();
            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            merged.truncate(top_n);
            MultiProbeResult {
                merged,
                probe_sizes,
            }
        }
    }
}

// ============================================================================
// 完整多路检索管线
// ============================================================================

/// 多路粗筛检索
///
/// 组合 3 个探测器的结果，用指定策略合并，返回候选 doc_id 列表。
///
/// # 参数
/// - `query_cloud`: query 的 token 点云
/// - `token_store`: 全库 token 点云
/// - `centroids`: 预计算的质心列表
/// - `max_token_reprs`: 预计算的最强 token 代表向量
/// - `inv_index`: 倒排索引（可选，None 则只用前两个 probe）
/// - `per_probe_top`: 每个 probe 返回的候选数
/// - `merged_top`: 合并后返回的候选数
/// - `strategy`: 合并策略
/// - `n_probe_inv`: 倒排索引的 n_probe 参数
///
/// # 返回
/// 合并后的 (doc_id, merged_score) 列表 + probe 统计
pub fn multi_probe_retrieve(
    query_cloud: &DocumentCloud,
    _token_store: &CloudStore,
    centroids: &[Vec<f32>],
    max_token_reprs: &[MaxTokenRepr],
    inv_index: Option<&TokenInvertedIndex>,
    per_probe_top: usize,
    merged_top: usize,
    strategy: MergeStrategy,
    n_probe_inv: usize,
) -> MultiProbeResult {
    // Probe 1: 质心 Chamfer 粗筛
    let probe1 = centroid_probe(query_cloud, centroids, per_probe_top);

    // Probe 2: 最强 token 探针
    let probe2 = max_token_probe(query_cloud, max_token_reprs, per_probe_top);

    // Probe 3: 倒排索引（可选）
    let probe3 = if let Some(idx) = inv_index {
        inverted_probe(query_cloud, idx, per_probe_top, n_probe_inv)
    } else {
        Vec::new()
    };

    // 合并
    let probes: Vec<Vec<(usize, f32)>> = if probe3.is_empty() {
        vec![probe1, probe2]
    } else {
        vec![probe1, probe2, probe3]
    };

    merge_probes(&probes, strategy, merged_top)
}

// ============================================================================
// Probe 1: 质心 Chamfer 粗筛（包装现有逻辑）
// ============================================================================

/// Probe 1: 用 query tokens vs 文档质心的 Chamfer 距离做粗筛
///
/// 返回 (doc_idx, similarity_score) 列表，按 score 降序
pub fn centroid_probe(
    query_cloud: &DocumentCloud,
    centroids: &[Vec<f32>],
    top_n: usize,
) -> Vec<(usize, f32)> {
    let nq = query_cloud.n_sentences;
    if nq == 0 {
        return Vec::new();
    }

    // 预计算 query token 子空间范数
    let q_norms: Vec<&[f32; NUM_SUBSPACES]> = (0..nq)
        .map(|i| &query_cloud.norm_caches[i].norms)
        .collect();

    let mut scores: Vec<(usize, f32)> = centroids
        .par_iter()
        .enumerate()
        .map(|(doc_idx, centroid)| {
            let c_norms = compute_sub_norms(centroid);
            let nq_f = nq as f32;
            let mut total = 0.0f32;

            for s in 0..NUM_SUBSPACES {
                let off = s * SUB_DIM;
                let c_sub = &centroid[off..off + SUB_DIM];
                let c_norm = c_norms[s];

                let mut sum_qd = 0.0f32;
                let mut min_dq = f32::MAX;

                for qi in 0..nq {
                    let q_sub = &query_cloud.sentence(qi)[off..off + SUB_DIM];
                    let q_norm = q_norms[qi][s];
                    let denom = q_norm * c_norm;
                    let d = if denom < 1e-16 {
                        1.0
                    } else {
                        let dot = sub_dot_64d(q_sub, c_sub);
                        1.0 - dot / (denom + 1e-8)
                    };
                    sum_qd += d;
                    if d < min_dq {
                        min_dq = d;
                    }
                }

                total += sum_qd / nq_f + min_dq;
            }

            let dist = total / NUM_SUBSPACES as f32;
            // 转为相似度分数（1 - dist），越大越好
            (doc_idx, 1.0 - dist)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_n);
    scores
}

// ============================================================================
// Probe 3: 倒排索引（包装现有逻辑）
// ============================================================================

/// Probe 3: 倒排索引粗筛
///
/// 返回 (doc_idx, hit_count as f32) 列表，按 hit_count 降序
pub fn inverted_probe(
    query_cloud: &DocumentCloud,
    index: &TokenInvertedIndex,
    top_n: usize,
    n_probe: usize,
) -> Vec<(usize, f32)> {
    let indices = index.query_top_n_fast(query_cloud, top_n, n_probe);
    // 按排名赋予递减分数（第一名最高）
    let total = indices.len() as f32;
    indices
        .into_iter()
        .enumerate()
        .map(|(rank, idx)| (idx, (total - rank as f32) / total))
        .collect()
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 全 4096 维 cosine distance
#[inline]
fn full_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (ca, cb) in a[..FULL_DIM].chunks_exact(4).zip(b[..FULL_DIM].chunks_exact(4)) {
        dot += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
        na += ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2] + ca[3] * ca[3];
        nb += cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2] + cb[3] * cb[3];
    }
    let norm_a = na.sqrt();
    let norm_b = nb.sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 1.0;
    }
    let cos_sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    (1.0 - cos_sim).max(0.0)
}

/// 64 维子空间内积
#[inline(always)]
fn sub_dot_64d(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        acc0 += ca[0] * cb[0];
        acc1 += ca[1] * cb[1];
        acc2 += ca[2] * cb[2];
        acc3 += ca[3] * cb[3];
    }
    (acc0 + acc1) + (acc2 + acc3)
}

/// 计算 4096d 向量的 64 个子空间 L2 范数
fn compute_sub_norms(vec: &[f32]) -> Vec<f32> {
    let mut norms = Vec::with_capacity(NUM_SUBSPACES);
    for s in 0..NUM_SUBSPACES {
        let off = s * SUB_DIM;
        let sub = &vec[off..off + SUB_DIM];
        let mut sum_sq = 0.0f32;
        for chunk in sub.chunks_exact(4) {
            sum_sq += chunk[0] * chunk[0]
                + chunk[1] * chunk[1]
                + chunk[2] * chunk[2]
                + chunk[3] * chunk[3];
        }
        norms.push(sum_sq.sqrt());
    }
    norms
}

/// 归一化 probe 结果到 [0,1] (min-max)
fn normalize_probe_results(
    probe_results: &[Vec<(usize, f32)>],
) -> Vec<Vec<(usize, f64)>> {
    probe_results
        .iter()
        .map(|probe| {
            if probe.is_empty() {
                return Vec::new();
            }
            let min_s = probe
                .iter()
                .map(|&(_, s)| s)
                .fold(f32::MAX, f32::min);
            let max_s = probe
                .iter()
                .map(|&(_, s)| s)
                .fold(f32::MIN, f32::max);
            let range = max_s - min_s;
            if range < 1e-8 {
                return probe.iter().map(|&(idx, _)| (idx, 0.5f64)).collect();
            }
            probe
                .iter()
                .map(|&(idx, s)| (idx, ((s - min_s) / range) as f64))
                .collect()
        })
        .collect()
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_rrf() {
        // 两个 probe，各返回 3 个结果
        let probe1 = vec![(0, 0.9f32), (1, 0.8), (2, 0.7)];
        let probe2 = vec![(1, 0.95f32), (2, 0.85), (3, 0.75)];

        let result = merge_probes(&[probe1, probe2], MergeStrategy::Rrf, 4);
        assert_eq!(result.merged.len(), 4);

        // doc 1 出现在两个 probe 中，RRF 分数应最高
        assert_eq!(result.merged[0].0, 1, "doc 1 应排第一（两路都有）");
    }

    #[test]
    fn test_merge_max_score() {
        let probe1 = vec![(0, 1.0f32), (1, 0.5)];
        let probe2 = vec![(1, 1.0f32), (2, 0.5)];

        let result = merge_probes(&[probe1, probe2], MergeStrategy::MaxScore, 3);
        assert_eq!(result.merged.len(), 3);

        // doc 0 和 doc 1 都有 max=1.0，但 doc 1 出现在两个 probe 中
        // MaxScore 只取 max，所以 doc 0 和 doc 1 分数相同
        assert!(
            result.merged[0].1 >= result.merged[1].1,
            "分数应降序排列"
        );
    }

    #[test]
    fn test_merge_hit_max() {
        let probe1 = vec![(0, 1.0f32), (1, 0.8)];
        let probe2 = vec![(1, 0.9f32), (2, 0.7)];
        let probe3 = vec![(1, 0.85f32), (3, 0.6)];

        let result = merge_probes(
            &[probe1, probe2, probe3],
            MergeStrategy::HitMax,
            4,
        );

        // doc 1 出现在 3 个 probe 中，hit=3，应排最前
        assert_eq!(result.merged[0].0, 1, "doc 1 应排第一（三路都有）");
    }

    #[test]
    fn test_full_cosine_distance_self() {
        let v: Vec<f32> = (0..FULL_DIM).map(|i| (i as f32 * 0.01).sin()).collect();
        let d = full_cosine_distance(&v, &v);
        assert!(d.abs() < 1e-5, "自身距离应接近 0，实际 = {}", d);
    }
}
