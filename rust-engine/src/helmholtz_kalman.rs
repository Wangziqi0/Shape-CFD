//! Helmholtz-Kalman Fusion (V15)
//!
//! 1. Helmholtz 分解：从图信号中提取与 token 正交的结构性修正（低通滤波去噪）
//! 2. Kalman 融合：按 per-document 不确定度自适应加权
//!
//! 输入：token 分数 s_T、图平滑分数 s_G、图邻接、per-token 方差
//! 输出：融合分数 s_fused

use nalgebra::{DMatrix, DVector};

/// Helmholtz-Kalman 融合
///
/// # 参数
/// - `s_token`: token_2stage 分数 (N)
/// - `s_graph`: 图平滑分数 (N)
/// - `adj`: KNN 邻接表 [(j, w_ij), ...]
/// - `token_variances`: 每个文档的 token 匹配方差 (N)
/// - `beta`: 低通滤波器参数 (1.0 = 合理起点)
///
/// # 返回
/// 融合分数 (N)
pub fn helmholtz_kalman_fuse(
    s_token: &[f64],
    s_graph: &[f64],
    adj: &[Vec<(usize, f64)>],
    token_variances: &[f64],
    beta: f64,
) -> Vec<f64> {
    let n = s_token.len();
    debug_assert_eq!(s_graph.len(), n);
    debug_assert_eq!(adj.len(), n);
    debug_assert_eq!(token_variances.len(), n);

    if n == 0 {
        return Vec::new();
    }

    // ========================================
    // 1. Helmholtz 分解
    // ========================================

    // 1a. 正交残差: r = s_G - α*s_T
    let dot_gt: f64 = s_graph.iter().zip(s_token.iter()).map(|(g, t)| g * t).sum();
    let dot_tt: f64 = s_token.iter().map(|t| t * t).sum();
    let alpha = if dot_tt > 1e-12 { dot_gt / dot_tt } else { 0.0 };

    let r: Vec<f64> = s_graph.iter().zip(s_token.iter())
        .map(|(g, t)| g - alpha * t)
        .collect();

    // 1b. 构建归一化图拉普拉斯 L_norm = D^{-1/2}(D-W)D^{-1/2}
    //     然后解 (I + β·L_norm) · r_smooth = r

    // 构建 W 矩阵和度矩阵 D
    let mut w_mat = DMatrix::<f64>::zeros(n, n);
    let mut degrees = vec![0.0f64; n];
    for (i, neighbors) in adj.iter().enumerate() {
        for &(j, w) in neighbors {
            w_mat[(i, j)] = w;
            degrees[i] += w;
        }
    }

    // D^{-1/2}
    let d_inv_sqrt: Vec<f64> = degrees.iter()
        .map(|&d| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // A = I + β · L_norm = I + β · D^{-1/2}(D-W)D^{-1/2}
    let mut a_mat = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            let l_norm_ij = if i == j {
                if degrees[i] > 1e-12 { 1.0 } else { 0.0 }
            } else {
                -d_inv_sqrt[i] * w_mat[(i, j)] * d_inv_sqrt[j]
            };
            a_mat[(i, j)] += beta * l_norm_ij;
        }
    }

    // 解 A · r_smooth = r
    let r_vec = DVector::from_vec(r.clone());
    let r_smooth = a_mat.lu().solve(&r_vec).unwrap_or(r_vec.clone());

    // ========================================
    // 2. Kalman 融合
    // ========================================

    let mut s_fused = vec![0.0f64; n];

    for i in 0..n {
        // Token 不确定度
        let sigma2_t = token_variances[i].max(1e-10);

        // 图不确定度：邻域分数不一致性
        let mut sigma2_g = 0.0f64;
        let mut w_sum = 0.0f64;
        for &(j, w) in &adj[i] {
            sigma2_g += w * (s_graph[j] - s_graph[i]).powi(2);
            w_sum += w;
        }
        sigma2_g = if w_sum > 1e-12 { sigma2_g / w_sum } else { 1e-10 };
        sigma2_g = sigma2_g.max(1e-10);

        // Kalman 增益: K_i → 1 时几乎只用 token, K_i → 0 时采纳图修正
        let k_i = sigma2_g / (sigma2_t + sigma2_g);

        // 融合: s_fused = s_T + (1 - K_i) * r_smooth
        s_fused[i] = s_token[i] + (1.0 - k_i) * r_smooth[i];
    }

    s_fused
}

/// 计算每个文档的 token 匹配方差
///
/// 对文档 i 的每个 token t，算 max_q cos(q_token, t) 的方差
/// 这衡量文档内 token 对 query 匹配的稳定性
pub fn compute_token_variances(
    query_cloud: &crate::cloud_store::DocumentCloud,
    doc_clouds: &[&crate::cloud_store::DocumentCloud],
) -> Vec<f64> {
    use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM};

    doc_clouds.iter().map(|doc| {
        let nq = query_cloud.n_sentences;
        let nd = doc.n_sentences;
        if nq == 0 || nd == 0 {
            return 0.0;
        }

        // 对每个 doc token，算它跟最近 query token 的距离（PQ cosine）
        let mut max_sims = Vec::with_capacity(nd);
        for di in 0..nd {
            let mut best_sim = f64::MIN;
            for qi in 0..nq {
                // 快速 PQ cosine similarity（64 个子空间平均）
                let mut total_sim = 0.0f32;
                for s in 0..NUM_SUBSPACES {
                    let off = s * SUB_DIM;
                    let q_sub = &query_cloud.sentence(qi)[off..off + SUB_DIM];
                    let d_sub = &doc.sentence(di)[off..off + SUB_DIM];
                    let q_norm = query_cloud.norm_caches[qi].norms[s];
                    let d_norm = doc.norm_caches[di].norms[s];
                    let denom = q_norm * d_norm;
                    if denom < 1e-16 {
                        // total_sim += 0.0; // similarity = 0 for zero vectors
                    } else {
                        let mut dot = 0.0f32;
                        for (qa, da) in q_sub.chunks_exact(4).zip(d_sub.chunks_exact(4)) {
                            dot += qa[0]*da[0] + qa[1]*da[1] + qa[2]*da[2] + qa[3]*da[3];
                        }
                        total_sim += dot / (denom + 1e-8);
                    }
                }
                let sim = (total_sim / NUM_SUBSPACES as f32) as f64;
                if sim > best_sim { best_sim = sim; }
            }
            max_sims.push(best_sim);
        }

        // 方差
        let mean: f64 = max_sims.iter().sum::<f64>() / nd as f64;
        let var: f64 = max_sims.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / nd as f64;
        var
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helmholtz_kalman_basic() {
        // 3 个文档，token 和 graph 分数不同
        let s_t = vec![0.8, 0.5, 0.3];
        let s_g = vec![0.7, 0.6, 0.2];
        let adj = vec![
            vec![(1, 0.5), (2, 0.3)],
            vec![(0, 0.5), (2, 0.4)],
            vec![(0, 0.3), (1, 0.4)],
        ];
        let tv = vec![0.1, 0.2, 0.05];
        let fused = helmholtz_kalman_fuse(&s_t, &s_g, &adj, &tv, 1.0);
        assert_eq!(fused.len(), 3);
        // 融合结果不应该偏离 token 太远
        for i in 0..3 {
            assert!((fused[i] - s_t[i]).abs() < 0.5, "fused[{}]={} too far from s_t={}", i, fused[i], s_t[i]);
        }
    }

    #[test]
    fn test_identical_signals() {
        // token 和 graph 完全相同 → 残差为零 → 融合 = token
        let s = vec![0.9, 0.5, 0.1];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];
        let tv = vec![0.1, 0.1, 0.1];
        let fused = helmholtz_kalman_fuse(&s, &s, &adj, &tv, 1.0);
        for i in 0..3 {
            assert!((fused[i] - s[i]).abs() < 1e-6, "should equal token when signals identical");
        }
    }
}
