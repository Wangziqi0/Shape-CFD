//! VT-Aligned 距离计算 (V13)
//!
//! 对每个子空间 s (0..64)，独立做句子级 Chamfer 距离，
//! 64 个子空间取平均。与 PQ-Chamfer 的区别：
//! - PQ-Chamfer: 先对每个句子对算 64 子空间平均距离，再做 Chamfer
//! - VT-Aligned: 先对每个子空间做句子级 Chamfer，再取 64 子空间平均
//!
//! 这样做的好处是允许不同子空间对齐到不同的句子，
//! 捕捉更细粒度的语义对齐关系。

use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM, FULL_DIM};
use rayon::prelude::*;

// ============================================================================
// 64 维 cosine distance (热路径)
// ============================================================================

/// 64 维子空间 cosine distance
///
/// 返回 1 - cos(a, b)，范围 [0, 2]。
/// 当任一子空间范数接近零时，返回 1.0（最大不确定距离）。
/// 使用 4-way unroll 帮助 LLVM 自动向量化。
#[inline(always)]
fn cosine_distance_64d(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), SUB_DIM, "cosine_distance_64d: a 长度不是 {}", SUB_DIM);
    debug_assert_eq!(b.len(), SUB_DIM, "cosine_distance_64d: b 长度不是 {}", SUB_DIM);

    // 4 个独立累加器，避免依赖链阻塞流水线
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

    // 4-way unroll 主循环（SUB_DIM=64 恰好整除 4）
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

    // 对齐 JS: 1 - dot / (sqrt(na)*sqrt(nb) + 1e-8)
    let denom = norm_a * norm_b;
    if denom < 1e-16 {
        return 1.0;
    }
    1.0 - dot / (denom + 1e-8)
}

// ============================================================================
// VT-Aligned 距离
// ============================================================================

/// VT-Aligned query-doc 距离
///
/// 对每个子空间 s：
///   - 正向: query 的第 s 个子空间 vs 每个句子的第 s 个子空间，找 min
///   - 反向: 每个句子到 query 的平均距离（query 只有 1 个，所以直接求和/N）
/// 64 个子空间取平均。
///
/// # 参数
/// - `query`: 1 个 4096 维向量
/// - `doc_cloud`: N 个 4096 维句子向量的切片引用
///
/// # 返回
/// 距离值（越小越近）
pub fn vt_aligned_query_doc(query: &[f32], doc_cloud: &[&[f32]]) -> f32 {
    debug_assert_eq!(query.len(), FULL_DIM, "query 维度不是 {}", FULL_DIM);
    debug_assert!(!doc_cloud.is_empty(), "doc_cloud 不能为空");

    let n = doc_cloud.len() as f32;
    let mut total = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let off = s * SUB_DIM;
        let q_sub = &query[off..off + SUB_DIM];

        // 正向: query → doc，找最近句子
        let mut min_d = f32::MAX;
        // 反向: 每个句子到 query 的距离之和
        let mut sum_rev = 0.0f32;

        for &sent in doc_cloud {
            let s_sub = &sent[off..off + SUB_DIM];
            let d = cosine_distance_64d(q_sub, s_sub);
            if d < min_d {
                min_d = d;
            }
            // cosine_distance 是对称的，反向距离 = 正向距离
            sum_rev += d;
        }

        total += min_d + sum_rev / n;
    }

    total / NUM_SUBSPACES as f32
}

/// VT-Aligned doc-doc 距离（用于 KNN 图构建）
///
/// 对每个子空间 s，做句子级对称 Chamfer：
///   - A→B: 对 A 中每个句子，找 B 中最近句子的子空间距离
///   - B→A: 反向
/// 64 个子空间取平均。
///
/// # 参数
/// - `cloud_a`: 文档 A 的 N_a 个 4096 维句子向量
/// - `cloud_b`: 文档 B 的 N_b 个 4096 维句子向量
///
/// # 返回
/// 对称距离值（越小越近）
pub fn vt_aligned_doc_doc(cloud_a: &[&[f32]], cloud_b: &[&[f32]]) -> f32 {
    debug_assert!(!cloud_a.is_empty(), "cloud_a 不能为空");
    debug_assert!(!cloud_b.is_empty(), "cloud_b 不能为空");

    let na = cloud_a.len() as f32;
    let nb = cloud_b.len() as f32;
    let mut total = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let off = s * SUB_DIM;

        // A→B
        let mut sum_ab = 0.0f32;
        for &a in cloud_a {
            let a_sub = &a[off..off + SUB_DIM];
            let mut min_d = f32::MAX;
            for &b in cloud_b {
                let b_sub = &b[off..off + SUB_DIM];
                let d = cosine_distance_64d(a_sub, b_sub);
                if d < min_d {
                    min_d = d;
                }
            }
            sum_ab += min_d;
        }

        // B→A
        let mut sum_ba = 0.0f32;
        for &b in cloud_b {
            let b_sub = &b[off..off + SUB_DIM];
            let mut min_d = f32::MAX;
            for &a in cloud_a {
                let a_sub = &a[off..off + SUB_DIM];
                let d = cosine_distance_64d(a_sub, b_sub);
                if d < min_d {
                    min_d = d;
                }
            }
            sum_ba += min_d;
        }

        total += sum_ab / na + sum_ba / nb;
    }

    total / NUM_SUBSPACES as f32
}

// ============================================================================
// 距离矩阵批量计算
// ============================================================================

/// 计算 N 个文档之间的 VT-Aligned 距离矩阵（上三角并行）
///
/// 返回 N*N 行优先矩阵（对称），对角线为 0。
/// 使用 rayon 对 i < j 的上三角进行并行计算。
///
/// # 参数
/// - `clouds`: N 个文档点云的句子切片引用列表
///
/// # 返回
/// N*N 的 f64 距离矩阵（行优先）
pub fn compute_vt_distance_matrix(clouds: &[Vec<&[f32]>]) -> Vec<f64> {
    let n = clouds.len();
    let mut matrix = vec![0.0f64; n * n];

    if n <= 1 {
        return matrix;
    }

    // 收集上三角索引对
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    // 并行计算上三角
    let distances: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| vt_aligned_doc_doc(&clouds[i], &clouds[j]) as f64)
        .collect();

    // 填充对称矩阵
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        matrix[i * n + j] = distances[idx];
        matrix[j * n + i] = distances[idx];
    }

    matrix
}

/// 计算 query 到 N 个文档的 VT-Aligned 距离（并行）
///
/// # 参数
/// - `query`: 4096 维查询向量
/// - `clouds`: N 个文档点云
///
/// # 返回
/// N 个 f64 距离值
pub fn compute_vt_query_distances(query: &[f32], clouds: &[Vec<&[f32]>]) -> Vec<f64> {
    clouds
        .par_iter()
        .map(|cloud| vt_aligned_query_doc(query, cloud) as f64)
        .collect()
}

// ============================================================================
// cosine 全库 top-N 粗筛
// ============================================================================

/// 全 4096 维 cosine distance（用于粗筛阶段）
///
/// 4-way unroll + 独立累加器。
#[inline]
fn full_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= FULL_DIM);
    debug_assert!(b.len() >= FULL_DIM);

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

    let a_chunks = a[..FULL_DIM].chunks_exact(4);
    let b_chunks = b[..FULL_DIM].chunks_exact(4);
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

    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 1.0;
    }

    let cos_sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    (1.0 - cos_sim).max(0.0)
}

/// cosine 全库排序 top-N（质心模式）
///
/// 计算每个文档的质心（句子向量均值），然后用 query 与质心的
/// cosine distance 排序。与 JS 版 `cosSim(qVec, dv[did])` 对齐。
/// 使用 rayon 并行化。
///
/// # 参数
/// - `query`: 4096 维查询向量
/// - `store`: 全库点云存储
/// - `top_n`: 返回 top-N 候选
///
/// # 返回
/// (doc_id, cosine_distance) 列表，按距离升序
pub fn cosine_top_n(
    query: &[f32],
    store: &crate::cloud_store::CloudStore,
    top_n: usize,
) -> Vec<(u32, f32)> {
    let mut scores: Vec<(u32, f32)> = store
        .documents
        .par_iter()
        .map(|doc| {
            // 计算文档质心（句子向量均值）
            let mut centroid = vec![0.0f32; FULL_DIM];
            for i in 0..doc.n_sentences {
                let sent = doc.sentence(i);
                for d in 0..FULL_DIM {
                    centroid[d] += sent[d];
                }
            }
            let inv_n = 1.0 / doc.n_sentences as f32;
            for d in 0..FULL_DIM {
                centroid[d] *= inv_n;
            }
            let dist = full_cosine_distance(query, &centroid);
            (doc.doc_id, dist)
        })
        .collect();

    // 按距离升序排序
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_n);
    scores
}

// ============================================================================
// KNN 图构建
// ============================================================================

/// 从距离矩阵构建 KNN 图
///
/// # 参数
/// - `n`: 节点数
/// - `k`: 每个节点的邻居数
/// - `dist_matrix`: N*N 距离矩阵（行优先）
///
/// # 返回
/// 邻接表: Vec<Vec<(neighbor_idx, weight)>>，weight = 1.0 / (1.0 + distance)
pub fn build_knn(
    n: usize,
    k: usize,
    dist_matrix: &[f64],
) -> Vec<Vec<(usize, f64)>> {
    debug_assert_eq!(dist_matrix.len(), n * n, "距离矩阵大小不匹配");

    let mut adj = vec![Vec::new(); n];

    for i in 0..n {
        // 收集所有非自身节点的 (距离, 索引)
        let mut neighbors: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist_matrix[i * n + j], j))
            .collect();

        // 按距离升序排序
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 取 top-k 邻居
        let k_actual = k.min(neighbors.len());
        for idx in 0..k_actual {
            let (dist, j) = neighbors[idx];
            let weight = (-2.0 * dist).exp();
            adj[i].push((j, weight));
        }
    }

    // 图对称化: 补齐缺失的反向边
    let mut to_add: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for &(j, w) in &adj[i] {
            if !adj[j].iter().any(|&(k, _)| k == i) {
                to_add.push((j, i, w));
            }
        }
    }
    for (src, dst, w) in to_add {
        adj[src].push((dst, w));
    }

    adj
}

/// 计算对流系数矩阵
///
/// 对每条 KNN 边 (i, j)，对流系数 u_{ij} 基于 query 到 j 的亲和度：
/// u_{ij} = alpha * (c0_j - c0_i) / dist_{ij}
///
/// # 参数
/// - `query`: 查询向量
/// - `clouds`: 文档点云列表
/// - `adj`: 邻接表
/// - `n`: 节点数
/// - `alpha`: 对流强度系数
///
/// # 返回
/// N*N 对流系数矩阵（行优先，非邻居位置为 0）
pub fn compute_advection(
    query: &[f32],
    clouds: &[Vec<&[f32]>],
    adj: &[Vec<(usize, f64)>],
    n: usize,
    alpha: f64,
) -> Vec<f64> {
    // 1. query 归一化因子
    let q_norm: f64 = query.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let inv_q_norm = 1.0 / (q_norm + 1e-8);

    // 2. 每个文档的质心（句子向量均值，FULL_DIM 维）
    let centroids: Vec<Vec<f64>> = clouds.iter().map(|cloud| {
        let m = cloud.len() as f64;
        let mut centroid = vec![0.0f64; FULL_DIM];
        for &sent in cloud {
            for d in 0..FULL_DIM {
                centroid[d] += sent[d] as f64;
            }
        }
        for d in 0..FULL_DIM {
            centroid[d] /= m;
        }
        centroid
    }).collect();

    let mut u_matrix = vec![0.0f64; n * n];

    for i in 0..n {
        for &(j, _w) in &adj[i] {
            // 已计算的边跳过
            if u_matrix[i * n + j] != 0.0 || u_matrix[j * n + i] != 0.0 {
                continue;
            }

            // 3. 对流系数: u_ij = alpha * dot(centroid_j - centroid_i, q * inv_q_norm) / (|centroid_j - centroid_i| + 1e-8)
            let mut dot_val = 0.0f64;
            let mut diff_norm_sq = 0.0f64;
            for d in 0..FULL_DIM {
                let diff = centroids[j][d] - centroids[i][d];
                dot_val += diff * (query[d] as f64) * inv_q_norm;
                diff_norm_sq += diff * diff;
            }
            let diff_norm = diff_norm_sq.sqrt();
            let u_ij = alpha * dot_val / (diff_norm + 1e-8);

            u_matrix[i * n + j] = u_ij;
            // 4. 反对称
            u_matrix[j * n + i] = -u_ij;
        }
    }

    u_matrix
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成确定性测试向量
    fn make_test_vec(seed: f32) -> Vec<f32> {
        (0..FULL_DIM)
            .map(|i| ((i as f32 + seed) * 0.01).sin())
            .collect()
    }

    /// 生成伪随机向量（简易 LCG）
    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_cosine_distance_64d_self() {
        // 自身距离应接近 0
        let v: Vec<f32> = (0..SUB_DIM).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let d = cosine_distance_64d(&v, &v);
        assert!(d.abs() < 1e-5, "自身 cosine distance 应接近 0，实际 = {}", d);
    }

    #[test]
    fn test_cosine_distance_64d_symmetry() {
        let a: Vec<f32> = (0..SUB_DIM).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..SUB_DIM).map(|i| (i as f32 * 0.2 + 0.5).cos()).collect();
        let d_ab = cosine_distance_64d(&a, &b);
        let d_ba = cosine_distance_64d(&b, &a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "cosine distance 应对称: d(a,b)={} vs d(b,a)={}",
            d_ab, d_ba
        );
    }

    #[test]
    fn test_vt_aligned_query_doc_self() {
        // query 就是 doc 唯一句子时，距离应接近 0
        let v = make_test_vec(1.0);
        let cloud: Vec<&[f32]> = vec![v.as_slice()];
        let d = vt_aligned_query_doc(&v, &cloud);
        assert!(d.abs() < 1e-4, "query=doc 时距离应接近 0，实际 = {}", d);
    }

    #[test]
    fn test_vt_aligned_doc_doc_self() {
        // 同一个点云的 doc-doc 距离应接近 0
        let v1 = make_test_vec(10.0);
        let v2 = make_test_vec(20.0);
        let cloud: Vec<&[f32]> = vec![v1.as_slice(), v2.as_slice()];
        let d = vt_aligned_doc_doc(&cloud, &cloud);
        assert!(d.abs() < 1e-4, "同一点云距离应接近 0，实际 = {}", d);
    }

    #[test]
    fn test_vt_aligned_doc_doc_symmetry() {
        let a1 = make_test_vec(30.0);
        let a2 = make_test_vec(40.0);
        let b1 = make_test_vec(50.0);
        let b2 = make_test_vec(60.0);
        let b3 = make_test_vec(70.0);

        let cloud_a: Vec<&[f32]> = vec![a1.as_slice(), a2.as_slice()];
        let cloud_b: Vec<&[f32]> = vec![b1.as_slice(), b2.as_slice(), b3.as_slice()];

        let d_ab = vt_aligned_doc_doc(&cloud_a, &cloud_b);
        let d_ba = vt_aligned_doc_doc(&cloud_b, &cloud_a);
        assert!(
            (d_ab - d_ba).abs() < 1e-5,
            "VT-Aligned doc-doc 应对称: d(A,B)={} vs d(B,A)={}",
            d_ab, d_ba
        );
    }

    #[test]
    fn test_vt_aligned_positive() {
        // 不同向量之间的距离应为正
        let a = make_test_vec(100.0);
        let b = make_test_vec(200.0);
        let cloud_a: Vec<&[f32]> = vec![a.as_slice()];
        let cloud_b: Vec<&[f32]> = vec![b.as_slice()];
        let d = vt_aligned_doc_doc(&cloud_a, &cloud_b);
        assert!(d > 0.0, "不同向量的 VT-Aligned 距离应 > 0，实际 = {}", d);
    }

    #[test]
    fn test_distance_matrix() {
        let v1 = make_test_vec(1.0);
        let v2 = make_test_vec(2.0);
        let v3 = make_test_vec(3.0);

        let clouds: Vec<Vec<&[f32]>> = vec![
            vec![v1.as_slice()],
            vec![v2.as_slice()],
            vec![v3.as_slice()],
        ];

        let matrix = compute_vt_distance_matrix(&clouds);
        assert_eq!(matrix.len(), 9, "3x3 矩阵应有 9 个元素");

        // 对角线为 0
        assert!(matrix[0].abs() < 1e-10, "对角线应为 0");
        assert!(matrix[4].abs() < 1e-10, "对角线应为 0");
        assert!(matrix[8].abs() < 1e-10, "对角线应为 0");

        // 对称性
        assert!(
            (matrix[1] - matrix[3]).abs() < 1e-10,
            "矩阵应对称: M[0,1]={} vs M[1,0]={}",
            matrix[1], matrix[3]
        );
        assert!(
            (matrix[2] - matrix[6]).abs() < 1e-10,
            "矩阵应对称: M[0,2]={} vs M[2,0]={}",
            matrix[2], matrix[6]
        );
    }

    #[test]
    fn test_build_knn() {
        // 4 个节点，k=2
        #[rustfmt::skip]
        let dist = vec![
            0.0, 0.1, 0.5, 0.3,
            0.1, 0.0, 0.4, 0.2,
            0.5, 0.4, 0.0, 0.6,
            0.3, 0.2, 0.6, 0.0,
        ];

        let adj = build_knn(4, 2, &dist);
        assert_eq!(adj.len(), 4);

        // 节点 0 的 2 个最近邻应该是 1 (0.1) 和 3 (0.3)
        let n0_ids: Vec<usize> = adj[0].iter().map(|&(j, _)| j).collect();
        assert!(n0_ids.contains(&1), "节点 0 的邻居应包含 1");
        assert!(n0_ids.contains(&3), "节点 0 的邻居应包含 3");
    }

    #[test]
    fn test_full_cosine_distance_self() {
        let v = make_test_vec(42.0);
        let d = full_cosine_distance(&v, &v);
        assert!(d.abs() < 1e-5, "自身全维 cosine distance 应接近 0，实际 = {}", d);
    }

    #[test]
    fn test_query_distances_parallel() {
        let query = make_test_vec(1.0);
        let v1 = make_test_vec(10.0);
        let v2 = make_test_vec(20.0);

        let clouds: Vec<Vec<&[f32]>> = vec![
            vec![v1.as_slice()],
            vec![v2.as_slice()],
        ];

        let dists = compute_vt_query_distances(&query, &clouds);
        assert_eq!(dists.len(), 2);
        assert!(dists[0] >= 0.0, "距离应非负");
        assert!(dists[1] >= 0.0, "距离应非负");
    }
}
