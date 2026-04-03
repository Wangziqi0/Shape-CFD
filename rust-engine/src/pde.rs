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

/// Token 梯度对流 PDE 求解器
///
/// 与 solve_pde 的区别：对流场 u_ij = beta * (token_score_j - token_score_i)
/// 而非基于质心方向投影的对流系数矩阵。
///
/// 包含 Allen-Cahn 反应项用于双稳态聚类。
///
/// # 参数
/// - `c0`: 初始浓度（句子级 Chamfer 初始化，跨粒度）
/// - `adj`: 邻接表 [(j, w_ij), ...]
/// - `token_scores`: 每个文档的 token Chamfer 分数（归一化到 [0,1]）
/// - `beta`: 对流强度
/// - `diff_coeff`: 扩散系数 D
/// - `gamma`: Allen-Cahn 反应强度
/// - `dt`: 时间步长
/// - `max_iter`: 最大迭代次数
/// - `epsilon`: 收敛阈值
///
/// # 返回
/// 终态浓度向量（长度 N，值域 [0, 1]）
pub fn solve_token_gradient_pde(
    c0: &[f64],
    adj: &[Vec<(usize, f64)>],
    token_scores: &[f64],
    beta: f64,
    diff_coeff: f64,
    gamma: f64,
    dt: f64,
    max_iter: usize,
    epsilon: f64,
) -> Vec<f64> {
    let n = c0.len();
    debug_assert_eq!(adj.len(), n, "邻接表长度不匹配: {} vs {}", adj.len(), n);
    debug_assert_eq!(token_scores.len(), n, "token_scores 长度不匹配: {} vs {}", token_scores.len(), n);

    if n == 0 {
        return Vec::new();
    }

    let mut c = c0.to_vec();
    let mut c_new = vec![0.0; n];

    // 修正 CFL 条件：考虑对流速度和扩散系数
    let mut max_speed = 0.0f64;
    let mut max_deg = 0.0f64;
    for i in 0..n {
        let mut node_speed = 0.0;
        let mut node_deg = 0.0;
        for &(j, w) in &adj[i] {
            let u_ij = beta * (token_scores[j] - token_scores[i]);
            node_speed += u_ij.abs() * w;
            node_deg += w;
        }
        if node_speed > max_speed { max_speed = node_speed; }
        if node_deg > max_deg { max_deg = node_deg; }
    }
    let cfl_dt = if diff_coeff * max_deg + max_speed > 0.0 {
        0.8 / (diff_coeff * max_deg + max_speed)
    } else {
        dt
    };
    let effective_dt = dt.min(cfl_dt);

    for _iter in 0..max_iter {
        let mut max_change = 0.0f64;

        // Allen-Cahn 阈值：当前时间步整体均值（每步只算一次）
        let theta = c.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            let mut diffusion = 0.0;
            let mut advection = 0.0;

            for &(j, w) in &adj[i] {
                // 扩散项：标准图拉普拉斯
                diffusion += w * (c[j] - c[i]);

                // 对流项（Upwind 格式）：u_ij = beta * (S_j - S_i)
                // u_ij > 0 表示 j 的 token 分数更高，流量从 i→j（i 流失、j 获得）
                // Upwind: 流出用上风值 c_i，流入用上风值 c_j
                let u_ij = beta * (token_scores[j] - token_scores[i]);
                // net flux at node i = inflow - outflow
                //   inflow:  max(-u_ij, 0) * c_j  (当 u_ij < 0, 即 S_i > S_j, 流从 j→i)
                //   outflow: max( u_ij, 0) * c_i  (当 u_ij > 0, 即 S_j > S_i, 流从 i→j)
                advection += w * ((-u_ij).max(0.0) * c[j] - u_ij.max(0.0) * c[i]);
            }

            // Allen-Cahn 反应项：双稳态聚类
            let reaction = gamma * c[i] * (1.0 - c[i]) * (c[i] - theta);

            c_new[i] = c[i] + effective_dt * (diff_coeff * diffusion + advection + reaction);

            // 截断到 [0, 1]（物理约束：浓度为概率量）
            c_new[i] = c_new[i].clamp(0.0, 1.0);

            let change = (c_new[i] - c[i]).abs();
            if change > max_change { max_change = change; }
        }

        std::mem::swap(&mut c, &mut c_new);

        // 收敛检查
        if max_change < epsilon {
            break;
        }
    }

    c
}

/// 图拉普拉斯平滑（PDE baseline）
///
/// 归一化图拉普拉斯扩散，无对流项，无反应项：
///   C_{k+1}(i) = C_k(i) + alpha * sum_j W_ij * (C_k(j) - C_k(i)) / sum_j W_ij
///
/// 用于验证 PDE 求解器相对于纯扩散的增益。
/// 当 Pe 极小（扩散主导）时，PDE 应退化为此 baseline。
///
/// # 参数
/// - `c0`: 初始浓度（与 PDE 相同的 C0）
/// - `adj`: 邻接表：adj[i] = [(j, w_ij), ...]
/// - `alpha`: 平滑系数（对应 D * dt）
/// - `steps`: 平滑步数
///
/// # 返回
/// 终态浓度向量
pub fn laplacian_smooth(
    c0: &[f64],
    adj: &[Vec<(usize, f64)>],
    alpha: f64,
    steps: usize,
) -> Vec<f64> {
    let n = c0.len();
    if n == 0 {
        return Vec::new();
    }

    let mut c = c0.to_vec();
    let mut c_new = vec![0.0; n];

    for _ in 0..steps {
        for i in 0..n {
            let mut w_sum = 0.0;
            let mut weighted_diff = 0.0;
            for &(j, w) in &adj[i] {
                weighted_diff += w * (c[j] - c[i]);
                w_sum += w;
            }
            if w_sum > 0.0 {
                c_new[i] = c[i] + alpha * weighted_diff / w_sum;
            } else {
                c_new[i] = c[i];
            }
            // 非负截断（与 PDE 一致）
            if c_new[i] < 0.0 {
                c_new[i] = 0.0;
            }
        }
        std::mem::swap(&mut c, &mut c_new);
    }

    c
}

/// 图拉普拉斯扩散 + Allen-Cahn 反应（无对流）
///
/// dC/dt = D * L * C + γ * C(1-C)(C-θ)
/// 扩散做邻域平滑，Allen-Cahn 做双稳态极化（推向 0 或 1）
/// 对流项完全去掉——实验证明对流是负贡献
pub fn laplacian_allen_cahn(
    c0: &[f64],
    adj: &[Vec<(usize, f64)>],
    alpha: f64,    // 扩散步长 (D * dt 的效果)
    gamma: f64,    // Allen-Cahn 反应强度
    steps: usize,
    epsilon: f64,  // 收敛阈值
) -> Vec<f64> {
    let n = c0.len();
    if n == 0 {
        return Vec::new();
    }

    let mut c = c0.to_vec();
    let mut c_new = vec![0.0; n];

    for _ in 0..steps {
        // 动态阈值 θ = mean(C)
        let theta = c.iter().sum::<f64>() / n as f64;
        let mut max_change = 0.0f64;

        for i in 0..n {
            // 扩散项：归一化图拉普拉斯
            let mut w_sum = 0.0;
            let mut weighted_diff = 0.0;
            for &(j, w) in &adj[i] {
                weighted_diff += w * (c[j] - c[i]);
                w_sum += w;
            }
            let diffusion = if w_sum > 0.0 { weighted_diff / w_sum } else { 0.0 };

            // Allen-Cahn 反应项
            let reaction = gamma * c[i] * (1.0 - c[i]) * (c[i] - theta);

            c_new[i] = (c[i] + alpha * diffusion + alpha * reaction).clamp(0.0, 1.0);

            let change = (c_new[i] - c[i]).abs();
            if change > max_change { max_change = change; }
        }

        std::mem::swap(&mut c, &mut c_new);
        if max_change < epsilon { break; }
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

    // ========================================================================
    // Laplacian smoothing 测试
    // ========================================================================

    #[test]
    fn test_laplacian_uniform_stays_uniform() {
        // 均匀浓度 → 拉普拉斯平滑后不变
        let n = 4;
        let c0 = vec![1.0; n];
        let adj = vec![
            vec![(1, 1.0), (3, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0), (3, 1.0)],
            vec![(2, 1.0), (0, 1.0)],
        ];

        let c_final = laplacian_smooth(&c0, &adj, 0.02, 20);
        for i in 0..n {
            assert!(
                (c_final[i] - 1.0).abs() < 1e-6,
                "均匀浓度应不变，节点 {}: c = {}",
                i, c_final[i]
            );
        }
    }

    #[test]
    fn test_laplacian_smoothing_effect() {
        // 不均匀浓度应趋向均匀
        let n = 3;
        let c0 = vec![10.0, 0.0, 0.0];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];

        let c_final = laplacian_smooth(&c0, &adj, 0.15, 100);

        let mean = c_final.iter().sum::<f64>() / n as f64;
        let variance: f64 = c_final.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / n as f64;

        let init_mean = c0.iter().sum::<f64>() / n as f64;
        let init_var: f64 = c0.iter().map(|&c| (c - init_mean).powi(2)).sum::<f64>() / n as f64;

        assert!(
            variance < init_var,
            "拉普拉斯平滑后方差应减小: 初始方差={}, 终态方差={}",
            init_var, variance
        );
    }

    #[test]
    fn test_laplacian_empty_input() {
        let c = laplacian_smooth(&[], &[], 0.02, 20);
        assert!(c.is_empty(), "空输入应返回空结果");
    }

    #[test]
    fn test_laplacian_single_node() {
        let c0 = vec![5.0];
        let adj = vec![vec![]];
        let c_final = laplacian_smooth(&c0, &adj, 0.02, 20);
        assert_eq!(c_final.len(), 1);
        assert!(
            (c_final[0] - 5.0).abs() < 1e-10,
            "单节点浓度应不变: {}",
            c_final[0]
        );
    }

    #[test]
    fn test_laplacian_non_negative() {
        // 浓度始终非负
        let c0 = vec![0.01, 5.0, 0.01];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];
        let c_final = laplacian_smooth(&c0, &adj, 0.5, 50);
        for (i, &c) in c_final.iter().enumerate() {
            assert!(c >= 0.0, "浓度不应为负，节点 {}: c = {}", i, c);
        }
    }

    // ========================================================================
    // Token 梯度 PDE 测试
    // ========================================================================

    #[test]
    fn test_token_gradient_empty_input() {
        let c = solve_token_gradient_pde(&[], &[], &[], 1.0, 0.15, 0.2, 0.03, 60, 1e-3);
        assert!(c.is_empty(), "空输入应返回空结果");
    }

    #[test]
    fn test_token_gradient_single_node() {
        // 单节点无邻居，浓度应不变（clamp 到 [0,1]）
        let c0 = vec![0.8];
        let adj = vec![vec![]];
        let scores = vec![0.5];
        let c_final = solve_token_gradient_pde(&c0, &adj, &scores, 1.0, 0.15, 0.2, 0.03, 60, 1e-3);
        assert_eq!(c_final.len(), 1);
        // Allen-Cahn 反应项会使 0.8 向 0 或 1 偏移，但不会离开 [0,1]
        assert!(c_final[0] >= 0.0 && c_final[0] <= 1.0, "浓度应在 [0,1]，实际 = {}", c_final[0]);
    }

    #[test]
    fn test_token_gradient_bounded() {
        // 所有终态浓度应在 [0, 1] 范围内
        let _n = 4;
        let c0 = vec![0.9, 0.3, 0.6, 0.1];
        let adj = vec![
            vec![(1, 1.0), (3, 0.5)],
            vec![(0, 1.0), (2, 0.8)],
            vec![(1, 0.8), (3, 0.6)],
            vec![(2, 0.6), (0, 0.5)],
        ];
        let scores = vec![0.8, 0.2, 0.6, 0.1];

        let c_final = solve_token_gradient_pde(&c0, &adj, &scores, 1.0, 0.15, 0.2, 0.03, 60, 1e-3);

        for (i, &c) in c_final.iter().enumerate() {
            assert!(
                c >= 0.0 && c <= 1.0,
                "浓度应在 [0,1]，节点 {}: c = {}",
                i, c
            );
        }
    }

    #[test]
    fn test_token_gradient_uniform_scores_reduces_to_diffusion() {
        // 当 token 分数全相同时，对流项为 0，应退化为纯扩散 + Allen-Cahn
        let n = 3;
        let c0 = vec![0.9, 0.1, 0.5];
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];
        let scores = vec![0.5, 0.5, 0.5]; // 均匀分数 → 无对流

        let c_final = solve_token_gradient_pde(&c0, &adj, &scores, 1.0, 0.15, 0.2, 0.03, 100, 1e-6);

        // 纯扩散 + Allen-Cahn 应使浓度趋向平滑
        let init_var: f64 = {
            let mean = c0.iter().sum::<f64>() / n as f64;
            c0.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / n as f64
        };
        let final_var: f64 = {
            let mean = c_final.iter().sum::<f64>() / n as f64;
            c_final.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / n as f64
        };
        // 扩散项应降低方差（Allen-Cahn 可能增加，但扩散主导时整体方差应降低）
        assert!(
            final_var <= init_var + 0.01,
            "均匀 token 分数下，扩散应平滑浓度: 初始方差={:.4}, 终态方差={:.4}",
            init_var, final_var
        );
    }

    #[test]
    fn test_token_gradient_high_score_attracts() {
        // token 高分节点应吸引浓度
        let _n = 3;
        let c0 = vec![0.5, 0.5, 0.5]; // 均匀初始浓度
        let adj = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
        ];
        // 节点 2 的 token 分数最高
        let scores = vec![0.1, 0.5, 0.9];

        let c_final = solve_token_gradient_pde(&c0, &adj, &scores, 2.0, 0.15, 0.0, 0.03, 100, 1e-6);

        // 高 token 分数节点应获得更高浓度（对流驱动）
        assert!(
            c_final[2] >= c_final[0],
            "高 token 分数节点(2)应有更高浓度: c[0]={:.4}, c[2]={:.4}",
            c_final[0], c_final[2]
        );
    }

    #[test]
    fn test_token_gradient_convergence() {
        // 验证收敛（足够多迭代后 max_change < epsilon）
        let _n = 4;
        let c0 = vec![0.8, 0.2, 0.6, 0.4];
        let adj = vec![
            vec![(1, 0.5), (3, 0.3)],
            vec![(0, 0.5), (2, 0.4)],
            vec![(1, 0.4), (3, 0.6)],
            vec![(2, 0.6), (0, 0.3)],
        ];
        let scores = vec![0.7, 0.3, 0.5, 0.1];

        // 跑两次：一次 1000 步，一次 2000 步，结果应接近
        let c1 = solve_token_gradient_pde(&c0, &adj, &scores, 1.0, 0.15, 0.2, 0.03, 1000, 1e-8);
        let c2 = solve_token_gradient_pde(&c0, &adj, &scores, 1.0, 0.15, 0.2, 0.03, 2000, 1e-8);

        let max_diff: f64 = c1.iter().zip(c2.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(
            max_diff < 0.05,
            "1000 步和 2000 步结果应接近收敛，最大差异 = {}",
            max_diff
        );
    }
}
