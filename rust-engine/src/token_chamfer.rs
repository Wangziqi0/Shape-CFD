//! Token-level PQ-Chamfer 距离计算 (V11) — 性能优化版
//!
//! 与 vt_distance.rs 中的句子级 VT-Aligned 不同，这里的 query 也是多点（token 点云），
//! 而非单个 4096d 向量。核心思路：
//!
//! 对每个子空间 s (0..64)：
//!   1. 一次性计算 nq x nd 距离矩阵（只做 dot product + 预计算范数查表）
//!   2. 对每行取 min → Q→D 方向
//!   3. 对每列取 min → D→Q 方向
//!   chamfer_s = mean(row_mins) + mean(col_mins)
//! 最终距离 = mean over 64 subspaces of chamfer_s
//!
//! 优化点：
//! - 距离矩阵消除 2x 重复计算（Q→D 和 D→Q 共享同一矩阵）
//! - 预计算子空间范数，距离计算只需 dot product + 除法
//! - query 子空间 buffer 重排提升缓存局部性
//! - 两阶段检索：质心粗筛 + 精排

use crate::cloud_store::{CloudStore, DocumentCloud};
use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM, FULL_DIM};
use rayon::prelude::*;

// ============================================================================
// 64 维子空间 dot product（热路径，4-way unroll）
// ============================================================================

/// 64 维子空间内积，4-way unroll 帮助 LLVM 自动向量化
#[inline(always)]
fn sub_dot_64d(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), SUB_DIM);
    debug_assert_eq!(b.len(), SUB_DIM);

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let a_chunks = a.chunks_exact(4);
    let b_chunks = b.chunks_exact(4);
    for (ca, cb) in a_chunks.zip(b_chunks) {
        acc0 += ca[0] * cb[0];
        acc1 += ca[1] * cb[1];
        acc2 += ca[2] * cb[2];
        acc3 += ca[3] * cb[3];
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// 使用预计算范数的 64d cosine distance
///
/// 返回 1 - cos(a, b)，范围 [0, 2]。
/// 当任一子空间范数接近零时，返回 1.0。
#[inline(always)]
fn cosine_distance_64d_prenorm(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
    let denom = norm_a * norm_b;
    if denom < 1e-16 {
        return 1.0;
    }
    let dot = sub_dot_64d(a, b);
    1.0 - dot / (denom + 1e-8)
}

/// 不使用预计算范数的 64d cosine distance（测试中用于验证优化版正确性）
#[cfg(test)]
#[inline(always)]
fn cosine_distance_64d(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), SUB_DIM);
    debug_assert_eq!(b.len(), SUB_DIM);

    let mut dot0 = 0.0f32;
    let mut dot1 = 0.0f32;
    let mut dot2 = 0.0f32;
    let mut dot3 = 0.0f32;

    let mut na0 = 0.0f32;
    let mut na1 = 0.0f32;
    let mut na2 = 0.0f32;
    let mut na3 = 0.0f32;

    let mut nb0 = 0.0f32;
    let mut nb1 = 0.0f32;
    let mut nb2 = 0.0f32;
    let mut nb3 = 0.0f32;

    let a_chunks = a.chunks_exact(4);
    let b_chunks = b.chunks_exact(4);
    for (ca, cb) in a_chunks.zip(b_chunks) {
        dot0 += ca[0] * cb[0];
        dot1 += ca[1] * cb[1];
        dot2 += ca[2] * cb[2];
        dot3 += ca[3] * cb[3];

        na0 += ca[0] * ca[0];
        na1 += ca[1] * ca[1];
        na2 += ca[2] * ca[2];
        na3 += ca[3] * ca[3];

        nb0 += cb[0] * cb[0];
        nb1 += cb[1] * cb[1];
        nb2 += cb[2] * cb[2];
        nb3 += cb[3] * cb[3];
    }

    let dot = (dot0 + dot1) + (dot2 + dot3);
    let norm_a = ((na0 + na1) + (na2 + na3)).sqrt();
    let norm_b = ((nb0 + nb1) + (nb2 + nb3)).sqrt();

    let denom = norm_a * norm_b;
    if denom < 1e-16 {
        return 1.0;
    }
    1.0 - dot / (denom + 1e-8)
}

// ============================================================================
// 子空间范数预计算
// ============================================================================

/// 预计算一个 token 在所有 64 个子空间上的 L2 范数
///
/// 返回 [norm_s0, norm_s1, ..., norm_s63]，长度 = NUM_SUBSPACES
#[inline]
fn precompute_sub_norms(vec: &[f32]) -> Vec<f32> {
    debug_assert!(vec.len() >= FULL_DIM);
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

// ============================================================================
// Token-level PQ-Chamfer 距离（优化版）
// ============================================================================

/// Token-level PQ-Chamfer 距离（query token 点云 vs 文档 token 点云）
///
/// 优化策略：
/// 1. 预计算所有 token 的 64 个子空间范数（消除 66% 冗余计算）
/// 2. 对每个子空间只计算一次 nq x nd 距离矩阵（消除 2x 重复）
/// 3. 对行取 min（Q→D）和对列取 min（D→Q）共享同一矩阵
///
/// # 参数
/// - `query_cloud`: query 的 token 向量列表（每个 4096d）
/// - `doc_cloud`: 文档的 token 向量列表（每个 4096d）
///
/// # 返回
/// 对称 Chamfer 距离（越小越相似）
pub fn token_pq_chamfer(query_cloud: &DocumentCloud, doc_cloud: &DocumentCloud) -> f64 {
    debug_assert!(query_cloud.n_sentences > 0, "query token 点云不能为空");
    debug_assert!(doc_cloud.n_sentences > 0, "doc token 点云不能为空");

    let nq = query_cloud.n_sentences;
    let nd = doc_cloud.n_sentences;
    let nq_f = nq as f32;
    let nd_f = nd as f32;

    // 1. 预计算所有 token 的子空间范数
    //    query: nq 个 token，每个 64 个子空间范数
    //    doc: nd 个 token，每个 64 个子空间范数
    //    （如果 DocumentCloud 已有 norm_caches，可以直接用其 norms 字段；
    //     但 norm_caches 是 PqNormCache，其 norms[s] 正好就是子空间 s 的范数）
    let q_norms: Vec<&[f32; NUM_SUBSPACES]> = (0..nq)
        .map(|i| &query_cloud.norm_caches[i].norms)
        .collect();
    let d_norms: Vec<&[f32; NUM_SUBSPACES]> = (0..nd)
        .map(|i| &doc_cloud.norm_caches[i].norms)
        .collect();

    // 2. 分配距离矩阵缓冲区（nq x nd，所有子空间共享复用）
    let mut dist_matrix = vec![0.0f32; nq * nd];
    let mut total = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let off = s * SUB_DIM;

        // 2a. 计算 nq x nd 距离矩阵
        for qi in 0..nq {
            let q_sub = &query_cloud.sentence(qi)[off..off + SUB_DIM];
            let q_norm = q_norms[qi][s];
            let row_off = qi * nd;
            for di in 0..nd {
                let d_sub = &doc_cloud.sentence(di)[off..off + SUB_DIM];
                let d_norm = d_norms[di][s];
                dist_matrix[row_off + di] =
                    cosine_distance_64d_prenorm(q_sub, d_sub, q_norm, d_norm);
            }
        }

        // 2b. 对每行取 min → Q→D 方向
        let mut sum_qd = 0.0f32;
        for qi in 0..nq {
            let row_off = qi * nd;
            let mut min_d = f32::MAX;
            for di in 0..nd {
                let d = dist_matrix[row_off + di];
                if d < min_d {
                    min_d = d;
                }
            }
            sum_qd += min_d;
        }

        // 2c. 对每列取 min → D→Q 方向
        let mut sum_dq = 0.0f32;
        for di in 0..nd {
            let mut min_d = f32::MAX;
            for qi in 0..nq {
                let d = dist_matrix[qi * nd + di];
                if d < min_d {
                    min_d = d;
                }
            }
            sum_dq += min_d;
        }

        // 2d. chamfer_s = sum_qd/nq + sum_dq/nd
        total += sum_qd / nq_f + sum_dq / nd_f;
    }

    // 3. 64 个子空间取平均
    (total / NUM_SUBSPACES as f32) as f64
}

/// 全库 Token Chamfer 扫描，返回 top_n 最近的 (doc_id, distance)
///
/// 对 token_store 中所有文档并行计算 token_pq_chamfer，
/// 排序取 top_n。使用 rayon 并行化。
///
/// # 参数
/// - `query_cloud`: query 的 token 点云（DocumentCloud 格式）
/// - `token_store`: 全库 token 点云存储
/// - `top_n`: 返回 top-N 候选数
///
/// # 返回
/// (doc_id, distance) 列表，按距离升序排列
pub fn token_chamfer_top_n(
    query_cloud: &DocumentCloud,
    token_store: &CloudStore,
    top_n: usize,
) -> Vec<(u32, f64)> {
    // 并行计算每个文档的 token Chamfer 距离
    let mut scores: Vec<(u32, f64)> = token_store
        .documents
        .par_iter()
        .map(|doc| {
            let dist = token_pq_chamfer(query_cloud, doc);
            (doc.doc_id, dist)
        })
        .collect();

    // 按距离升序排序
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_n);
    scores
}

// ============================================================================
// Token 质心粗筛 + 两阶段检索
// ============================================================================

/// 预计算每个文档的 token 质心（所有 token 的均值向量，4096d）
///
/// 返回 Vec<Vec<f32>>，索引与 token_store.documents 对应。
/// 使用 rayon 并行化。
pub fn precompute_token_centroids(token_store: &CloudStore) -> Vec<Vec<f32>> {
    token_store.documents.par_iter().map(|doc| {
        let mut centroid = vec![0.0f32; FULL_DIM];
        for i in 0..doc.n_sentences {
            let tok = doc.sentence(i);
            for (j, v) in tok.iter().enumerate() {
                centroid[j] += v;
            }
        }
        let n = doc.n_sentences as f32;
        if n > 0.0 {
            centroid.iter_mut().for_each(|v| *v /= n);
        }
        centroid
    }).collect()
}

/// query token 点云与单个质心之间的简化 Chamfer 距离
///
/// 质心视为只有 1 个点的点云，因此：
/// - Q→D 方向：每个 query token 到质心的距离取 min = 所有距离中最小值（只有 1 个 doc 点）
///   实际上 min 就是每个 query token 到质心的距离本身（因为只有 1 个 doc 点）
///   所以 Q→D = mean over query tokens of dist(qi, centroid)
/// - D→Q 方向：质心到最近 query token 的距离 = min over query tokens
/// chamfer_s = mean(dists) + min(dists) 对每个子空间
///
/// 这比全 token Chamfer 快得多（nd=1 vs nd=~数十）
#[inline]
fn token_centroid_chamfer(query_cloud: &DocumentCloud, centroid: &[f32]) -> f64 {
    let nq = query_cloud.n_sentences;
    if nq == 0 {
        return f64::MAX;
    }
    let nq_f = nq as f32;

    // 预计算质心的子空间范数
    let c_norms = precompute_sub_norms(centroid);

    // 预计算 query token 的子空间范数（直接从 norm_caches 取）
    let q_norms: Vec<&[f32; NUM_SUBSPACES]> = (0..nq)
        .map(|i| &query_cloud.norm_caches[i].norms)
        .collect();

    let mut total = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let off = s * SUB_DIM;
        let c_sub = &centroid[off..off + SUB_DIM];
        let c_norm = c_norms[s];

        let mut sum_qd = 0.0f32;  // Q→D: 每个 query token 到质心的距离之和
        let mut min_dq = f32::MAX; // D→Q: 质心到最近 query token 的距离

        for qi in 0..nq {
            let q_sub = &query_cloud.sentence(qi)[off..off + SUB_DIM];
            let q_norm = q_norms[qi][s];
            let d = cosine_distance_64d_prenorm(q_sub, c_sub, q_norm, c_norm);
            sum_qd += d;
            if d < min_dq {
                min_dq = d;
            }
        }

        // chamfer_s = mean(Q→D) + mean(D→Q)，其中 D 只有 1 个点所以 mean = 本身
        total += sum_qd / nq_f + min_dq;
    }

    (total / NUM_SUBSPACES as f32) as f64
}

/// 两阶段检索：token 质心粗筛 top-M → 全 token Chamfer 精排 top-N
///
/// 第一阶段：query tokens vs 每个文档质心的 Chamfer 距离（快，每文档只有 1 个质心点）
/// 第二阶段：对 top-M 文档做完整 token_pq_chamfer
///
/// # 参数
/// - `query_cloud`: query 的 token 点云
/// - `token_store`: 全库 token 点云存储
/// - `centroids`: 预计算的质心列表（与 token_store.documents 一一对应）
/// - `coarse_top`: 粗筛候选数（如 200）
/// - `top_n`: 最终返回数
///
/// # 返回
/// (doc_id, distance) 列表，按距离升序排列
pub fn token_chamfer_two_stage(
    query_cloud: &DocumentCloud,
    token_store: &CloudStore,
    centroids: &[Vec<f32>],
    coarse_top: usize,
    top_n: usize,
) -> Vec<(u32, f64)> {
    // 1. 粗筛：query tokens vs 每个文档质心
    let mut coarse_scores: Vec<(usize, f64)> = centroids
        .par_iter()
        .enumerate()
        .map(|(idx, centroid)| {
            let dist = token_centroid_chamfer(query_cloud, centroid);
            (idx, dist)
        })
        .collect();

    // 按距离升序排序，取 coarse_top
    coarse_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    coarse_scores.truncate(coarse_top);

    // 2. 精排：对 coarse_top 文档做完整 token_pq_chamfer
    let mut fine_scores: Vec<(u32, f64)> = coarse_scores
        .par_iter()
        .map(|&(idx, _)| {
            let doc = &token_store.documents[idx];
            let dist = token_pq_chamfer(query_cloud, doc);
            (doc.doc_id, dist)
        })
        .collect();

    // 按距离升序排序，取 top_n
    fine_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    fine_scores.truncate(top_n);
    fine_scores
}

// ============================================================================
// Token 采样 Chamfer 距离（方案1：用采样 token 建图，避免 O(n^2) 全对计算）
// ============================================================================

/// 用采样 token 计算两个文档间的 Chamfer 距离（用于建图）
///
/// 从每个文档均匀采样 sample_n 个 token，计算 PQ-Chamfer。
/// 如果文档 token 数 <= sample_n，使用全部 token。
///
/// # 参数
/// - `doc_a`: 文档 A 的 token 点云
/// - `doc_b`: 文档 B 的 token 点云
/// - `sample_n`: 每个文档的采样 token 数
///
/// # 返回
/// 采样后的对称 Chamfer 距离（越小越相似）
pub fn token_sampled_chamfer(
    doc_a: &DocumentCloud,
    doc_b: &DocumentCloud,
    sample_n: usize,
) -> f64 {
    debug_assert!(doc_a.n_sentences > 0, "doc_a token 点云不能为空");
    debug_assert!(doc_b.n_sentences > 0, "doc_b token 点云不能为空");

    let na = doc_a.n_sentences;
    let nb = doc_b.n_sentences;

    // 均匀采样索引（如果 token 数 <= sample_n，用全部）
    let indices_a = uniform_sample_indices(na, sample_n);
    let indices_b = uniform_sample_indices(nb, sample_n);

    let sa = indices_a.len();
    let sb = indices_b.len();
    let sa_f = sa as f32;
    let sb_f = sb as f32;

    // 获取采样 token 的子空间范数
    let a_norms: Vec<&[f32; NUM_SUBSPACES]> = indices_a
        .iter()
        .map(|&i| &doc_a.norm_caches[i].norms)
        .collect();
    let b_norms: Vec<&[f32; NUM_SUBSPACES]> = indices_b
        .iter()
        .map(|&i| &doc_b.norm_caches[i].norms)
        .collect();

    let mut dist_matrix = vec![0.0f32; sa * sb];
    let mut total = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let off = s * SUB_DIM;

        // 计算 sa x sb 距离矩阵
        for ai in 0..sa {
            let a_sub = &doc_a.sentence(indices_a[ai])[off..off + SUB_DIM];
            let a_norm = a_norms[ai][s];
            let row_off = ai * sb;
            for bi in 0..sb {
                let b_sub = &doc_b.sentence(indices_b[bi])[off..off + SUB_DIM];
                let b_norm = b_norms[bi][s];
                dist_matrix[row_off + bi] =
                    cosine_distance_64d_prenorm(a_sub, b_sub, a_norm, b_norm);
            }
        }

        // A→B 方向：每行取 min
        let mut sum_ab = 0.0f32;
        for ai in 0..sa {
            let row_off = ai * sb;
            let mut min_d = f32::MAX;
            for bi in 0..sb {
                let d = dist_matrix[row_off + bi];
                if d < min_d { min_d = d; }
            }
            sum_ab += min_d;
        }

        // B→A 方向：每列取 min
        let mut sum_ba = 0.0f32;
        for bi in 0..sb {
            let mut min_d = f32::MAX;
            for ai in 0..sa {
                let d = dist_matrix[ai * sb + bi];
                if d < min_d { min_d = d; }
            }
            sum_ba += min_d;
        }

        total += sum_ab / sa_f + sum_ba / sb_f;
    }

    (total / NUM_SUBSPACES as f32) as f64
}

/// 均匀采样索引：从 n 个元素中均匀采样 sample_n 个
/// 如果 n <= sample_n，返回全部索引 [0, 1, ..., n-1]
#[inline]
fn uniform_sample_indices(n: usize, sample_n: usize) -> Vec<usize> {
    if n <= sample_n {
        return (0..n).collect();
    }
    // 均匀步长采样
    let stride = n as f64 / sample_n as f64;
    (0..sample_n)
        .map(|i| ((i as f64 * stride) as usize).min(n - 1))
        .collect()
}

/// 用采样 token Chamfer 计算 N 个文档的距离矩阵（对称矩阵，行优先）
///
/// # 参数
/// - `docs`: N 个文档的 token 点云列表
/// - `sample_n`: 每个文档的采样 token 数
///
/// # 返回
/// N*N f64 矩阵（行优先），dist_matrix[i*n+j] = token_sampled_chamfer(docs[i], docs[j])
pub fn token_sampled_distance_matrix(
    docs: &[&DocumentCloud],
    sample_n: usize,
) -> Vec<f64> {
    let n = docs.len();
    let mut matrix = vec![0.0f64; n * n];

    // 只计算上三角，然后对称填充
    for i in 0..n {
        for j in (i + 1)..n {
            let d = token_sampled_chamfer(docs[i], docs[j], sample_n);
            matrix[i * n + j] = d;
            matrix[j * n + i] = d;
        }
    }
    matrix
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pq_chamfer::{precompute_norms, FULL_DIM};

    /// 构造一个包含 n_tokens 个 4096d 向量的 DocumentCloud
    /// 每个 token 向量用 seed + token_idx 生成确定性数据
    fn make_token_cloud(doc_id: u32, n_tokens: usize, seed: f32) -> DocumentCloud {
        let mut vectors = Vec::with_capacity(n_tokens * FULL_DIM);
        let mut norm_caches = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let offset = seed + t as f32 * 10.0;
            let vec: Vec<f32> = (0..FULL_DIM)
                .map(|i| ((i as f32 + offset) * 0.01).sin())
                .collect();
            norm_caches.push(precompute_norms(&vec));
            vectors.extend_from_slice(&vec);
        }
        DocumentCloud {
            doc_id,
            vectors,
            n_sentences: n_tokens,
            norm_caches,
        }
    }

    #[test]
    fn test_token_pq_chamfer_self_zero() {
        // 同一个 token 点云与自身的距离应接近 0
        let cloud = make_token_cloud(1, 3, 1.0);
        let dist = token_pq_chamfer(&cloud, &cloud);
        assert!(
            dist.abs() < 1e-4,
            "自身 token Chamfer 距离应接近 0，实际 = {}",
            dist
        );
    }

    #[test]
    fn test_token_pq_chamfer_symmetry() {
        // 对称性: d(A, B) == d(B, A)
        let cloud_a = make_token_cloud(1, 3, 1.0);
        let cloud_b = make_token_cloud(2, 2, 50.0);
        let d_ab = token_pq_chamfer(&cloud_a, &cloud_b);
        let d_ba = token_pq_chamfer(&cloud_b, &cloud_a);
        assert!(
            (d_ab - d_ba).abs() < 1e-5,
            "Token Chamfer 应对称: d(A,B)={} vs d(B,A)={}",
            d_ab, d_ba
        );
    }

    #[test]
    fn test_token_pq_chamfer_positive_for_different() {
        // 不同点云的距离应为正
        let cloud_a = make_token_cloud(1, 3, 1.0);
        let cloud_b = make_token_cloud(2, 3, 100.0);
        let dist = token_pq_chamfer(&cloud_a, &cloud_b);
        assert!(
            dist > 0.0,
            "不同 token 点云距离应 > 0，实际 = {}",
            dist
        );
    }

    #[test]
    fn test_token_pq_chamfer_single_token() {
        // 单 token vs 单 token 情况
        let cloud_a = make_token_cloud(1, 1, 1.0);
        let cloud_b = make_token_cloud(2, 1, 1.0);
        // 相同 seed 应接近 0
        let dist = token_pq_chamfer(&cloud_a, &cloud_b);
        assert!(
            dist.abs() < 1e-4,
            "相同 seed 的单 token 距离应接近 0，实际 = {}",
            dist
        );
    }

    #[test]
    fn test_token_chamfer_top_n_ordering() {
        // 构造一个小的 CloudStore，验证 top_n 排序正确
        use std::collections::HashMap;

        // 3 个文档：doc_id=1,2,3
        let query = make_token_cloud(0, 2, 1.0);
        let doc1 = make_token_cloud(1, 2, 1.0);  // 与 query seed 相同，距离最小
        let doc2 = make_token_cloud(2, 2, 50.0); // 中等距离
        let doc3 = make_token_cloud(3, 2, 200.0); // 最远

        let mut id_map = HashMap::new();
        id_map.insert(1u32, 0usize);
        id_map.insert(2u32, 1usize);
        id_map.insert(3u32, 2usize);

        let store = CloudStore {
            documents: vec![doc1, doc2, doc3],
            id_map,
            total_docs: 3,
            total_vectors: 6,
        };

        let results = token_chamfer_top_n(&query, &store, 2);
        assert_eq!(results.len(), 2, "应返回 2 个结果");
        // 第一个应该是 doc_id=1（距离最小）
        assert_eq!(results[0].0, 1, "最近文档应该是 doc_id=1");
        // 距离应按升序
        assert!(
            results[0].1 <= results[1].1,
            "距离应按升序: {} <= {}",
            results[0].1, results[1].1
        );
    }

    #[test]
    fn test_precompute_token_centroids() {
        use std::collections::HashMap;

        let doc1 = make_token_cloud(1, 3, 1.0);
        let doc2 = make_token_cloud(2, 2, 50.0);

        let mut id_map = HashMap::new();
        id_map.insert(1u32, 0usize);
        id_map.insert(2u32, 1usize);

        let store = CloudStore {
            documents: vec![doc1, doc2],
            id_map,
            total_docs: 2,
            total_vectors: 5,
        };

        let centroids = precompute_token_centroids(&store);
        assert_eq!(centroids.len(), 2, "应有 2 个质心");
        assert_eq!(centroids[0].len(), FULL_DIM, "质心维度应为 4096");
        assert_eq!(centroids[1].len(), FULL_DIM, "质心维度应为 4096");

        // 验证质心正确性：doc1 有 3 个 token，质心 = 均值
        let doc1_ref = &store.documents[0];
        let mut expected = vec![0.0f32; FULL_DIM];
        for i in 0..doc1_ref.n_sentences {
            let tok = doc1_ref.sentence(i);
            for (j, v) in tok.iter().enumerate() {
                expected[j] += v;
            }
        }
        for v in expected.iter_mut() {
            *v /= doc1_ref.n_sentences as f32;
        }
        let max_diff: f32 = centroids[0]
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "质心计算误差应 < 1e-5，实际最大误差 = {}",
            max_diff
        );
    }

    #[test]
    fn test_token_chamfer_two_stage() {
        use std::collections::HashMap;

        let query = make_token_cloud(0, 2, 1.0);
        let doc1 = make_token_cloud(1, 2, 1.0);   // 最近
        let doc2 = make_token_cloud(2, 2, 50.0);  // 中等
        let doc3 = make_token_cloud(3, 2, 200.0); // 最远

        let mut id_map = HashMap::new();
        id_map.insert(1u32, 0usize);
        id_map.insert(2u32, 1usize);
        id_map.insert(3u32, 2usize);

        let store = CloudStore {
            documents: vec![doc1, doc2, doc3],
            id_map,
            total_docs: 3,
            total_vectors: 6,
        };

        let centroids = precompute_token_centroids(&store);

        // 粗筛 top-3（全部进入精排），精排 top-2
        let results = token_chamfer_two_stage(&query, &store, &centroids, 3, 2);
        assert_eq!(results.len(), 2, "应返回 2 个结果");
        assert_eq!(results[0].0, 1, "两阶段检索最近文档应该是 doc_id=1");
        assert!(
            results[0].1 <= results[1].1,
            "距离应按升序: {} <= {}",
            results[0].1, results[1].1
        );
    }

    #[test]
    fn test_token_centroid_chamfer_consistency() {
        // 单 token 文档的质心就是该 token 本身
        // 因此 centroid chamfer 和 full chamfer 在 nd=1 时应该一致
        let query = make_token_cloud(0, 3, 1.0);
        let doc = make_token_cloud(1, 1, 50.0);

        // 全 Chamfer
        let full_dist = token_pq_chamfer(&query, &doc);

        // 质心 Chamfer（质心就是那 1 个 token）
        let centroid = doc.sentence(0).to_vec();
        let centroid_dist = token_centroid_chamfer(&query, &centroid);

        assert!(
            (full_dist - centroid_dist).abs() < 1e-4,
            "单 token 文档的 centroid chamfer 应与 full chamfer 一致: full={} centroid={}",
            full_dist, centroid_dist
        );
    }

    #[test]
    fn test_optimized_matches_original() {
        // 验证优化版与原始逻辑的数值一致性
        // 使用不同大小的点云确保矩阵方向正确
        let cloud_a = make_token_cloud(1, 4, 7.0);
        let cloud_b = make_token_cloud(2, 3, 42.0);

        // 原始逻辑：逐对计算，两个方向分别遍历
        let nq = cloud_a.n_sentences;
        let nd = cloud_b.n_sentences;
        let mut original_total = 0.0f32;

        for s in 0..NUM_SUBSPACES {
            let off = s * SUB_DIM;

            let mut sum_qd = 0.0f32;
            for qi in 0..nq {
                let q_sub = &cloud_a.sentence(qi)[off..off + SUB_DIM];
                let mut min_d = f32::MAX;
                for di in 0..nd {
                    let d_sub = &cloud_b.sentence(di)[off..off + SUB_DIM];
                    let d = cosine_distance_64d(q_sub, d_sub);
                    if d < min_d { min_d = d; }
                }
                sum_qd += min_d;
            }

            let mut sum_dq = 0.0f32;
            for di in 0..nd {
                let d_sub = &cloud_b.sentence(di)[off..off + SUB_DIM];
                let mut min_d = f32::MAX;
                for qi in 0..nq {
                    let q_sub = &cloud_a.sentence(qi)[off..off + SUB_DIM];
                    let d = cosine_distance_64d(d_sub, q_sub);
                    if d < min_d { min_d = d; }
                }
                sum_dq += min_d;
            }

            original_total += sum_qd / nq as f32 + sum_dq / nd as f32;
        }
        let original = (original_total / NUM_SUBSPACES as f32) as f64;

        // 优化版
        let optimized = token_pq_chamfer(&cloud_a, &cloud_b);

        assert!(
            (original - optimized).abs() < 1e-4,
            "优化版与原始版数值应一致: original={} optimized={}",
            original, optimized
        );
    }
}
