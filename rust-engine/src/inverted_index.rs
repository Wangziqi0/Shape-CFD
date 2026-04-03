//! Token 倒排索引粗筛 (V14)
//!
//! 对每个 PQ 子空间 (0..64)，用 K-means 将 880K 个 64d 子向量聚类为 256 个簇，
//! 建立倒排表。查询时对每个 query token 的每个子空间找最近的 n_probe 个簇中心，
//! 查倒排表累加文档命中次数，取 top-N 候选送精排。
//!
//! 替代原有的质心 cosine top-N 粗筛，让整条管线从粗筛到精排都是点云级别操作。

use crate::cloud_store::{CloudStore, DocumentCloud};
use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM};
use crate::token_chamfer::token_pq_chamfer;
use rayon::prelude::*;

/// 每个子空间的聚类中心数
pub const NUM_CENTROIDS: usize = 256;

// ============================================================================
// 辅助函数：64d L2 距离平方（4-way unroll）
// ============================================================================

/// 64 维 L2 距离平方，4-way unroll 帮助 LLVM 自动向量化
#[inline(always)]
pub(crate) fn l2_sq_64d(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

// ============================================================================
// K-means 训练
// ============================================================================

/// 简单 K-means (64d, K=256, max_iter=20)
///
/// 初始化：均匀采样 K 个样本作为初始中心
/// 收敛条件：分配不再变化 或 达到最大迭代次数
pub(crate) fn kmeans_64d(
    data: &[&[f32]], // N 个 64d 向量
    k: usize,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    // 1. 初始化：均匀采样
    let step = data.len() / k;
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| data[i * step].to_vec())
        .collect();

    let mut assignments = vec![0u32; data.len()];

    for _iter in 0..max_iter {
        // 2. 分配：每个点找最近中心
        let changed = assignments
            .par_iter_mut()
            .enumerate()
            .map(|(i, a)| {
                let mut best_c = 0u32;
                let mut best_d = f32::MAX;
                for (c, cent) in centroids.iter().enumerate() {
                    let d = l2_sq_64d(data[i], cent);
                    if d < best_d {
                        best_d = d;
                        best_c = c as u32;
                    }
                }
                if *a != best_c {
                    *a = best_c;
                    1u32
                } else {
                    0u32
                }
            })
            .sum::<u32>();

        // 3. 更新中心
        let mut sums = vec![vec![0.0f64; SUB_DIM]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in assignments.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1;
            for d in 0..SUB_DIM {
                sums[c][d] += data[i][d] as f64;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for d in 0..SUB_DIM {
                    centroids[c][d] = (sums[c][d] * inv) as f32;
                }
            }
        }

        if changed == 0 {
            break;
        }
    }

    centroids
}

// ============================================================================
// Per-subspace 倒排索引
// ============================================================================

/// Per-subspace 倒排索引
pub struct TokenInvertedIndex {
    /// 码本: [NUM_SUBSPACES][NUM_CENTROIDS][SUB_DIM] = 64 x 256 x 64 f32
    codebooks: Vec<Vec<Vec<f32>>>,
    /// 倒排表: [NUM_SUBSPACES][NUM_CENTROIDS] -> Vec<(doc_idx, token_idx)>
    /// doc_idx 是在 CloudStore.documents 中的索引（不是 doc_id）
    posting_lists: Vec<Vec<Vec<(u32, u16)>>>,
    /// 文档总数
    n_docs: usize,
}

impl TokenInvertedIndex {
    /// 从 CloudStore 构建倒排索引
    ///
    /// 1. 收集所有 token 的子空间向量
    /// 2. 对每个子空间做 K-means
    /// 3. 分配 token 到最近中心，建倒排表
    pub fn build(store: &CloudStore) -> Self {
        let n_docs = store.documents.len();

        // 对每个子空间并行训练 + 建倒排（64 个子空间可以并行）
        let per_sub: Vec<(Vec<Vec<f32>>, Vec<Vec<(u32, u16)>>)> = (0..NUM_SUBSPACES)
            .into_par_iter()
            .map(|s| {
                let off = s * SUB_DIM;

                // 收集该子空间所有 token 的 64d 子向量引用
                // 同时记录 (doc_idx, token_idx)
                let mut sub_vecs: Vec<&[f32]> = Vec::new();
                let mut token_ids: Vec<(u32, u16)> = Vec::new();

                for (doc_idx, doc) in store.documents.iter().enumerate() {
                    for t in 0..doc.n_sentences {
                        let tok = doc.sentence(t);
                        sub_vecs.push(&tok[off..off + SUB_DIM]);
                        token_ids.push((doc_idx as u32, t as u16));
                    }
                }

                // K-means 训练
                let centroids = kmeans_64d(&sub_vecs, NUM_CENTROIDS, 20);

                // 分配 + 建倒排表
                let mut posting: Vec<Vec<(u32, u16)>> = vec![Vec::new(); NUM_CENTROIDS];
                for (i, sv) in sub_vecs.iter().enumerate() {
                    let mut best_c = 0usize;
                    let mut best_d = f32::MAX;
                    for (c, cent) in centroids.iter().enumerate() {
                        let d = l2_sq_64d(sv, cent);
                        if d < best_d {
                            best_d = d;
                            best_c = c;
                        }
                    }
                    posting[best_c].push(token_ids[i]);
                }

                (centroids, posting)
            })
            .collect();

        let mut codebooks = Vec::with_capacity(NUM_SUBSPACES);
        let mut posting_lists = Vec::with_capacity(NUM_SUBSPACES);
        for (cb, pl) in per_sub {
            codebooks.push(cb);
            posting_lists.push(pl);
        }

        // 打印统计
        let total_entries: usize = posting_lists
            .iter()
            .flat_map(|sub| sub.iter())
            .map(|list| list.len())
            .sum();
        let avg_list_len = total_entries as f64 / (NUM_SUBSPACES * NUM_CENTROIDS) as f64;
        eprintln!(
            "[InvertedIndex] 构建完成: {} 子空间 x {} 中心, 倒排表 {:.1}M 条目, 平均 {:.0} 条/表",
            NUM_SUBSPACES,
            NUM_CENTROIDS,
            total_entries as f64 / 1e6,
            avg_list_len,
        );

        Self {
            codebooks,
            posting_lists,
            n_docs,
        }
    }

    /// 导出码本为 flat f32 数组 [64 * 256 * 64]
    pub fn export_codebook_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM);
        for s in 0..NUM_SUBSPACES {
            for c in 0..NUM_CENTROIDS {
                flat.extend_from_slice(&self.codebooks[s][c]);
            }
        }
        flat
    }

    /// 用倒排索引做粗筛，返回 top-N 候选的 doc_idx 列表
    ///
    /// 对每个 query token：
    ///   对每个子空间 s：
    ///     找到最近的 n_probe 个码本中心
    ///     查倒排表，收集 (doc_idx, token_idx) 对
    /// 按文档命中次数聚合，取 top-N
    ///
    /// n_probe=1 最快，n_probe=3 更准
    pub fn query_top_n(
        &self,
        query_cloud: &DocumentCloud,
        top_n: usize,
        n_probe: usize,
    ) -> Vec<usize> {
        let nq = query_cloud.n_sentences;

        // 文档得分计数器
        let mut doc_scores = vec![0u32; self.n_docs];

        for qi in 0..nq {
            let q_vec = query_cloud.sentence(qi);

            for s in 0..NUM_SUBSPACES {
                let off = s * SUB_DIM;
                let q_sub = &q_vec[off..off + SUB_DIM];
                let codebook = &self.codebooks[s];

                // 找最近的 n_probe 个中心
                // 用简单的 partial sort（n_probe 通常很小）
                let mut dists: Vec<(usize, f32)> = codebook
                    .iter()
                    .enumerate()
                    .map(|(c, cent)| (c, l2_sq_64d(q_sub, cent)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // 查倒排表，累加文档命中
                for &(c_idx, _) in dists.iter().take(n_probe) {
                    for &(doc_idx, _) in &self.posting_lists[s][c_idx] {
                        doc_scores[doc_idx as usize] += 1;
                    }
                }
            }
        }

        // 取 top-N by score
        let mut scored: Vec<(usize, u32)> = doc_scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(i, &s)| (i, s))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(top_n);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    /// V14c: 纯命中计数 + n_probe=1 专用快速路径
    ///
    /// n_probe=1 时用 argmin 替代排序（O(K) vs O(K log K)）
    /// n_probe>1 时用 select_nth_unstable
    pub fn query_top_n_fast(
        &self,
        query_cloud: &DocumentCloud,
        top_n: usize,
        n_probe: usize,
    ) -> Vec<usize> {
        let nq = query_cloud.n_sentences;
        let mut doc_scores = vec![0u32; self.n_docs];

        for qi in 0..nq {
            let q_vec = query_cloud.sentence(qi);

            for s in 0..NUM_SUBSPACES {
                let off = s * SUB_DIM;
                let q_sub = &q_vec[off..off + SUB_DIM];
                let codebook = &self.codebooks[s];

                if n_probe == 1 {
                    // 快速路径：直接 argmin，不排序
                    let mut best_c = 0usize;
                    let mut best_d = f32::MAX;
                    for (c, cent) in codebook.iter().enumerate() {
                        let d = l2_sq_64d(q_sub, cent);
                        if d < best_d {
                            best_d = d;
                            best_c = c;
                        }
                    }
                    for &(doc_idx, _) in &self.posting_lists[s][best_c] {
                        doc_scores[doc_idx as usize] += 1;
                    }
                } else {
                    // n_probe > 1: partial sort
                    let mut dists: Vec<(usize, f32)> = codebook
                        .iter()
                        .enumerate()
                        .map(|(c, cent)| (c, l2_sq_64d(q_sub, cent)))
                        .collect();

                    if n_probe < dists.len() {
                        dists.select_nth_unstable_by(n_probe - 1, |a, b| {
                            a.1.partial_cmp(&b.1).unwrap()
                        });
                    }

                    for &(c_idx, _) in dists.iter().take(n_probe) {
                        for &(doc_idx, _) in &self.posting_lists[s][c_idx] {
                            doc_scores[doc_idx as usize] += 1;
                        }
                    }
                }
            }
        }

        let mut scored: Vec<(usize, u32)> = doc_scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(i, &s)| (i, s))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(top_n);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    /// 内存占用估算（字节）
    pub fn memory_usage(&self) -> usize {
        // 码本: 64 子空间 x 256 中心 x 64 维 x 4 bytes
        let cb = NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM * 4;
        // 倒排表: 每条 (u32, u16) = 6 bytes
        let pl: usize = self
            .posting_lists
            .iter()
            .flat_map(|sub| sub.iter())
            .map(|list| list.len() * std::mem::size_of::<(u32, u16)>())
            .sum();
        cb + pl
    }
}

// ============================================================================
// 两阶段检索：倒排粗筛 + token PQ-Chamfer 精排
// ============================================================================

/// 倒排索引两阶段检索：倒排粗筛 + token PQ-Chamfer 精排
///
/// 1. 用倒排索引对 query token 点云做粗筛，取 coarse_top 个候选文档
/// 2. 对候选文档做 token PQ-Chamfer 精排，取 top_n 返回
pub fn inverted_two_stage(
    query_cloud: &DocumentCloud,
    token_store: &CloudStore,
    index: &TokenInvertedIndex,
    coarse_top: usize,
    top_n: usize,
    n_probe: usize,
) -> Vec<(u32, f64)> {
    // 1. 倒排粗筛
    let candidate_indices = index.query_top_n(query_cloud, coarse_top, n_probe);

    // 2. 精排
    let mut fine_scores: Vec<(u32, f64)> = candidate_indices
        .par_iter()
        .map(|&idx| {
            let doc = &token_store.documents[idx];
            let dist = token_pq_chamfer(query_cloud, doc);
            (doc.doc_id, dist)
        })
        .collect();

    fine_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    fine_scores.truncate(top_n);
    fine_scores
}

/// V14c: 快速命中计数粗筛（argmin/select_nth 优化）+ 大候选池 + 精排
pub fn inverted_two_stage_fast(
    query_cloud: &DocumentCloud,
    token_store: &CloudStore,
    index: &TokenInvertedIndex,
    coarse_top: usize,
    top_n: usize,
    n_probe: usize,
) -> Vec<(u32, f64)> {
    let candidate_indices = index.query_top_n_fast(query_cloud, coarse_top, n_probe);

    let mut fine_scores: Vec<(u32, f64)> = candidate_indices
        .par_iter()
        .map(|&idx| {
            let doc = &token_store.documents[idx];
            let dist = token_pq_chamfer(query_cloud, doc);
            (doc.doc_id, dist)
        })
        .collect();

    fine_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    fine_scores.truncate(top_n);
    fine_scores
}
