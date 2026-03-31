//! Upwind 对流-扩散 PDE 求解器
//!
//! 在 KNN 图上求解对流-扩散方程，用于 Shape-CFD 重排序：
//!
//!   dc/dt + div(u * c) = D * laplacian(c)
//!
//! 离散化后在图上迭代求解，浓度 c_i 表示文档 i 与 query 的相关度。
//! 使用 Upwind 格式保证数值稳定性（浓度非负 + 守恒性）。

/// Upwind 对流-扩散 PDE 求解
///
/// # 参数
/// - `c0`: 初始浓度 (长度 N)
/// - `adj`: 邻接表，adj[i] = [(neighbor_idx, weight), ...]
/// - `u`: 对流系数矩阵 (N*N, 行优先)，u[i*n+j] 是从 i 到 j 的对流系数
/// - `n`: 节点数
/// - `d_coeff`: 扩散系数 (典型值 0.15)
/// - `max_iter`: 最大迭代次数 (典型值 50)
/// - `epsilon`: 收敛阈值 (典型值 1e-3)
///
/// # 返回
/// 终态浓度向量 (长度 N)
///
/// # 算法
/// 每步迭代对每个节点 i：
///   1. 扩散项: D * sum_{j in N(i)} w_{ij} * (c_j - c_i)
///   2. 对流项: Upwind 格式，根据 u_{ij} 的正负选择上风方向
///   3. 更新: c_i += dt * (扩散项 + 对流项)
///   4. 非负截断: c_i = max(c_i, 0)
///   5. 收敛检查: max|c_new - c_old| < epsilon 时提前终止
pub fn solve_pde(
    c0: &[f64],
    adj: &[Vec<(usize, f64)>],
    u: &[f64],
    n: usize,
    d_coeff: f64,
    max_iter: usize,
    epsilon: f64,
) -> Vec<f64> {
    debug_assert_eq!(c0.len(), n, "初始浓度长度不匹配: {} vs {}", c0.len(), n);
    debug_assert_eq!(adj.len(), n, "邻接表长度不匹配: {} vs {}", adj.len(), n);
    debug_assert_eq!(u.len(), n * n, "对流矩阵大小不匹配: {} vs {}", u.len(), n * n);

    if n == 0 {
        return Vec::new();
    }

    let mut c = c0.to_vec();

    // 时间步长: 对齐 JS 实现
    let max_degree = adj.iter().map(|a| a.len()).max().unwrap_or(1) as f64;
    let dt = if max_degree > 0.0 {
        (0.8 / max_degree).min(0.1)
    } else {
        0.1
    };

    for _iter in 0..max_iter {
        let c_old = c.clone();
        let mut max_delta = 0.0f64;

        for i in 0..n {
            let mut diffusion = 0.0f64;
            let mut advection = 0.0f64;

            for &(j, w_ij) in &adj[i] {
                // 扩散项: D * w_{ij} * (c_j - c_i)
                diffusion += w_ij * (c_old[j] - c_old[i]);

                // 对流项: Upwind 格式
                let u_ij = u[i * n + j];
                if u_ij > 0.0 {
                    // 正向对流: 使用上风值 c_i
                    advection -= u_ij * c_old[i] * w_ij;
                } else {
                    // 负向对流: 使用上风值 c_j
                    advection -= u_ij * c_old[j] * w_ij;
                }
            }

            c[i] = c_old[i] + dt * (d_coeff * diffusion + advection);

            // 非负截断（物理约束：浓度不能为负）
            if c[i] < 0.0 {
                c[i] = 0.0;
            }

            let delta = (c[i] - c_old[i]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }

        // 收敛检查
        if max_delta < epsilon {
            break;
        }
    }

    c
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_concentration_stays_uniform() {
        // 均匀浓度 + 无对流 → 浓度不变
        let n = 4;
        let c0 = vec![1.0; n];
        let adj = vec![
            vec![(1, 1.0), (3, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0), (3, 1.0)],
            vec![(2, 1.0), (0, 1.0)],
        ];
        let u = vec![0.0; n * n]; // 无对流

        let c_final = solve_pde(&c0, &adj, &u, n, 0.15, 50, 1e-6);

        for i in 0..n {
            assert!(
                (c_final[i] - 1.0).abs() < 1e-6,
                "均匀浓度应保持不变，节点 {}: c = {}",
                i, c_final[i]
            );
        }
    }

    #[test]
    fn test_diffusion_smoothing() {
        // 初始浓度不均匀，纯扩散应趋向均匀
        let n = 3;
        let c0 = vec![10.0, 0.0, 0.0];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];
        let u = vec![0.0; n * n];

        let c_final = solve_pde(&c0, &adj, &u, n, 0.15, 200, 1e-6);

        // 扩散后浓度应更均匀（方差减小）
        let mean = c_final.iter().sum::<f64>() / n as f64;
        let variance: f64 = c_final.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / n as f64;

        let init_mean = c0.iter().sum::<f64>() / n as f64;
        let init_var: f64 = c0.iter().map(|&c| (c - init_mean).powi(2)).sum::<f64>() / n as f64;

        assert!(
            variance < init_var,
            "扩散后方差应减小: 初始方差={}, 终态方差={}",
            init_var, variance
        );
    }

    #[test]
    fn test_non_negative() {
        // 浓度始终非负
        let n = 3;
        let c0 = vec![0.01, 5.0, 0.01];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];
        // 强对流，可能导致负值——应被截断
        let mut u = vec![0.0; n * n];
        u[0 * n + 1] = 2.0; // 0→1 强对流
        u[2 * n + 1] = 2.0; // 2→1 强对流

        let c_final = solve_pde(&c0, &adj, &u, n, 0.15, 100, 1e-8);

        for i in 0..n {
            assert!(
                c_final[i] >= 0.0,
                "浓度不应为负，节点 {}: c = {}",
                i, c_final[i]
            );
        }
    }

    #[test]
    fn test_empty_input() {
        let c = solve_pde(&[], &[], &[], 0, 0.15, 50, 1e-3);
        assert!(c.is_empty(), "空输入应返回空结果");
    }

    #[test]
    fn test_single_node() {
        // 单节点无邻居，浓度应不变
        let c0 = vec![5.0];
        let adj = vec![vec![]];
        let u = vec![0.0];

        let c_final = solve_pde(&c0, &adj, &u, 1, 0.15, 50, 1e-3);
        assert_eq!(c_final.len(), 1);
        assert!(
            (c_final[0] - 5.0).abs() < 1e-10,
            "单节点浓度应不变: {}",
            c_final[0]
        );
    }

    #[test]
    fn test_convergence() {
        // 验证 PDE 在有限步内收敛
        let n = 4;
        let c0 = vec![1.0, 0.5, 0.3, 0.1];
        let adj = vec![
            vec![(1, 0.5), (3, 0.3)],
            vec![(0, 0.5), (2, 0.4)],
            vec![(1, 0.4), (3, 0.6)],
            vec![(2, 0.6), (0, 0.3)],
        ];
        let u = vec![0.0; n * n];

        // 用很小的 epsilon 看是否能在 max_iter 内收敛
        let c_final = solve_pde(&c0, &adj, &u, n, 0.15, 1000, 1e-8);

        // 纯扩散应趋向均匀
        let mean = c_final.iter().sum::<f64>() / n as f64;
        for &c in &c_final {
            assert!(
                (c - mean).abs() < 0.1,
                "纯扩散收敛后应接近均值 {}: 实际 {}",
                mean, c
            );
        }
    }
}
