//! PQ-Chamfer 距离计算核心
//!
//! PQ (Product Quantization) 将 4096 维向量切成 64 个 64 维子空间，
//! 每个子空间独立计算 cosine distance，取平均。
//! Chamfer distance 是两个点云之间的对称最近邻 PQ 距离均值。

/// 子空间数量：64 个子空间
pub const NUM_SUBSPACES: usize = 64;
/// 每个子空间的维度：64 维
pub const SUB_DIM: usize = 64;
/// 完整向量维度：4096 = 64 * 64
pub const FULL_DIM: usize = NUM_SUBSPACES * SUB_DIM;

/// 预计算的子空间范数缓存
///
/// 对一个 4096 维向量，预先计算其 64 个子空间各自的 L2 范数，
/// 避免在 PQ 距离计算中重复求范数。
pub struct PqNormCache {
    /// 每个子空间的 L2 范数
    pub norms: [f32; NUM_SUBSPACES],
}

/// 预计算一个 4096d 向量的 64 个子空间范数
///
/// 对每个子空间 s (0..64)，取 vec[s*64 .. (s+1)*64]，
/// 计算 sqrt(sum of squares) 作为该子空间的 L2 范数。
pub fn precompute_norms(vec: &[f32]) -> PqNormCache {
    debug_assert_eq!(
        vec.len(),
        FULL_DIM,
        "precompute_norms: 期望 {} 维向量，实际 {} 维",
        FULL_DIM,
        vec.len()
    );

    let mut norms = [0.0f32; NUM_SUBSPACES];
    for s in 0..NUM_SUBSPACES {
        let offset = s * SUB_DIM;
        let sub = &vec[offset..offset + SUB_DIM];
        let mut sum_sq = 0.0f32;
        // 4-way unroll 帮助 LLVM 自动向量化
        for chunk in sub.chunks_exact(4) {
            sum_sq += chunk[0] * chunk[0]
                + chunk[1] * chunk[1]
                + chunk[2] * chunk[2]
                + chunk[3] * chunk[3];
        }
        norms[s] = sum_sq.sqrt();
    }
    PqNormCache { norms }
}

/// 64 维子空间内积 (SIMD 热路径, 4-way unroll 帮助自动向量化)
///
/// 使用 chunks_exact(4) 拆分，让 LLVM 识别为可向量化的循环。
/// 调用方保证 a.len() == b.len() == SUB_DIM。
#[inline(always)]
fn sub_dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), SUB_DIM, "sub_dot_product: a 长度不是 {}", SUB_DIM);
    debug_assert_eq!(b.len(), SUB_DIM, "sub_dot_product: b 长度不是 {}", SUB_DIM);

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    // 4 个独立累加器，避免依赖链阻塞流水线
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

/// 使用预计算范数的 PQ cosine distance
///
/// 公式: d = (1/64) * sum_{s=0}^{63} (1 - dot_s / (norm_a_s * norm_b_s))
///
/// 当某个子空间的范数接近零时（零向量子空间），该子空间贡献距离 1.0。
pub fn pq_cosine_distance_cached(
    a: &[f32],
    a_norms: &PqNormCache,
    b: &[f32],
    b_norms: &PqNormCache,
) -> f32 {
    debug_assert_eq!(
        a.len(),
        FULL_DIM,
        "pq_cosine_distance_cached: a 期望 {} 维，实际 {} 维",
        FULL_DIM,
        a.len()
    );
    debug_assert_eq!(
        b.len(),
        FULL_DIM,
        "pq_cosine_distance_cached: b 期望 {} 维，实际 {} 维",
        FULL_DIM,
        b.len()
    );

    let mut total_dist = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let offset = s * SUB_DIM;
        let sub_a = &a[offset..offset + SUB_DIM];
        let sub_b = &b[offset..offset + SUB_DIM];

        let norm_a = a_norms.norms[s];
        let norm_b = b_norms.norms[s];

        // 如果任一子空间范数接近零，该子空间 cosine distance 定义为 1.0
        if norm_a < 1e-9 || norm_b < 1e-9 {
            total_dist += 1.0;
            continue;
        }

        let dot = sub_dot_product(sub_a, sub_b);
        // cosine_distance = 1 - cosine_similarity
        // clamp 避免浮点误差导致负距离
        let cos_sim = dot / (norm_a * norm_b);
        let cos_dist = (1.0 - cos_sim).max(0.0);
        total_dist += cos_dist;
    }

    total_dist / NUM_SUBSPACES as f32
}

/// 不用缓存的版本 (用于简单场景)
///
/// 内部先 precompute_norms 再调用 cached 版本。
/// 如果要对同一个向量算多次距离，建议用 cached 版本。
pub fn pq_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let a_norms = precompute_norms(a);
    let b_norms = precompute_norms(b);
    pq_cosine_distance_cached(a, &a_norms, b, &b_norms)
}

/// Chamfer distance: 两个点云之间的对称最近邻 PQ 距离均值
///
/// 公式:
///   d_chamfer = (1/|A|) * sum_{a in A} min_{b in B} d(a,b)
///             + (1/|B|) * sum_{b in B} min_{a in A} d(b,a)
///
/// 其中 d(a,b) 是 PQ cosine distance。
///
/// # Panics
/// - cloud_a / cloud_b 为空时 panic
/// - 向量维度不是 FULL_DIM 时 debug_assert 失败
pub fn chamfer_distance(
    cloud_a: &[&[f32]],
    cloud_a_norms: &[PqNormCache],
    cloud_b: &[&[f32]],
    cloud_b_norms: &[PqNormCache],
) -> f32 {
    assert!(!cloud_a.is_empty(), "chamfer_distance: cloud_a 不能为空");
    assert!(!cloud_b.is_empty(), "chamfer_distance: cloud_b 不能为空");
    debug_assert_eq!(
        cloud_a.len(),
        cloud_a_norms.len(),
        "chamfer_distance: cloud_a 和 cloud_a_norms 长度不匹配"
    );
    debug_assert_eq!(
        cloud_b.len(),
        cloud_b_norms.len(),
        "chamfer_distance: cloud_b 和 cloud_b_norms 长度不匹配"
    );

    // A -> B 方向：对 A 中每个向量找 B 中最近的
    let mut sum_a_to_b = 0.0f32;
    for (i, a_vec) in cloud_a.iter().enumerate() {
        let mut min_dist = f32::MAX;
        for (j, b_vec) in cloud_b.iter().enumerate() {
            let dist = pq_cosine_distance_cached(
                a_vec,
                &cloud_a_norms[i],
                b_vec,
                &cloud_b_norms[j],
            );
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum_a_to_b += min_dist;
    }

    // B -> A 方向：对 B 中每个向量找 A 中最近的
    let mut sum_b_to_a = 0.0f32;
    for (j, b_vec) in cloud_b.iter().enumerate() {
        let mut min_dist = f32::MAX;
        for (i, a_vec) in cloud_a.iter().enumerate() {
            let dist = pq_cosine_distance_cached(
                b_vec,
                &cloud_b_norms[j],
                a_vec,
                &cloud_a_norms[i],
            );
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum_b_to_a += min_dist;
    }

    let avg_a_to_b = sum_a_to_b / cloud_a.len() as f32;
    let avg_b_to_a = sum_b_to_a / cloud_b.len() as f32;

    avg_a_to_b + avg_b_to_a
}

// ============================================================================
// 单元测试
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// 生成一个确定性的测试向量（用索引填充，保证非零）
    fn make_test_vec(seed: f32) -> Vec<f32> {
        (0..FULL_DIM)
            .map(|i| ((i as f32 + seed) * 0.01).sin())
            .collect()
    }

    #[test]
    fn test_self_distance_is_zero() {
        // 一个向量与自身的 PQ cosine distance 应接近 0
        let v = make_test_vec(1.0);
        let dist = pq_cosine_distance(&v, &v);
        assert!(
            dist.abs() < 1e-5,
            "自身距离应接近 0，实际 = {}",
            dist
        );
    }

    #[test]
    fn test_self_distance_cached_is_zero() {
        let v = make_test_vec(2.0);
        let norms = precompute_norms(&v);
        let dist = pq_cosine_distance_cached(&v, &norms, &v, &norms);
        assert!(
            dist.abs() < 1e-5,
            "cached 版本: 自身距离应接近 0，实际 = {}",
            dist
        );
    }

    #[test]
    fn test_symmetry() {
        // PQ cosine distance 应满足对称性: d(a,b) == d(b,a)
        let a = make_test_vec(3.0);
        let b = make_test_vec(7.0);
        let d_ab = pq_cosine_distance(&a, &b);
        let d_ba = pq_cosine_distance(&b, &a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "对称性: d(a,b)={} vs d(b,a)={}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_pq_vs_naive_cosine() {
        // PQ cosine distance 是子空间 cosine distance 的均值，
        // 与朴素全空间 cosine distance 的关系：
        // 当子空间划分对齐且均匀时，两者应正相关。
        // 这里验证：朴素 cosine distance > 0 时 PQ 也 > 0，
        // 且自身距离都接近 0。
        let a = make_test_vec(10.0);
        let b = make_test_vec(20.0);

        // 朴素 cosine distance（内联计算，避免依赖 crate::utils 注册）
        let naive_dist = {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..FULL_DIM {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            1.0 - dot / (na.sqrt() * nb.sqrt())
        };

        // PQ cosine distance
        let pq_dist = pq_cosine_distance(&a, &b);

        // 两者都应为正
        assert!(naive_dist > 0.0, "朴素距离应 > 0: {}", naive_dist);
        assert!(pq_dist > 0.0, "PQ 距离应 > 0: {}", pq_dist);

        // PQ distance 应与朴素 distance 在同一数量级
        // （不要求严格相等，因为子空间独立计算 cosine 再平均不等于全空间 cosine）
        let ratio = pq_dist / naive_dist;
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "PQ 与朴素距离比值应在合理范围，ratio={}  pq={} naive={}",
            ratio,
            pq_dist,
            naive_dist
        );
    }

    #[test]
    fn test_chamfer_self_is_zero() {
        // 同一个点云与自身的 Chamfer distance 应接近 0
        let v1 = make_test_vec(100.0);
        let v2 = make_test_vec(200.0);
        let vecs: Vec<&[f32]> = vec![v1.as_slice(), v2.as_slice()];
        let norms: Vec<PqNormCache> = vecs.iter().map(|v| precompute_norms(v)).collect();

        let dist = chamfer_distance(&vecs, &norms, &vecs, &norms);
        assert!(
            dist.abs() < 1e-5,
            "点云与自身的 Chamfer distance 应接近 0，实际 = {}",
            dist
        );
    }

    #[test]
    fn test_chamfer_symmetry() {
        // Chamfer distance 天然对称: d(A,B) == d(B,A)
        let a1 = make_test_vec(30.0);
        let a2 = make_test_vec(40.0);
        let b1 = make_test_vec(50.0);
        let b2 = make_test_vec(60.0);
        let b3 = make_test_vec(70.0);

        let cloud_a: Vec<&[f32]> = vec![a1.as_slice(), a2.as_slice()];
        let cloud_b: Vec<&[f32]> = vec![b1.as_slice(), b2.as_slice(), b3.as_slice()];
        let norms_a: Vec<PqNormCache> = cloud_a.iter().map(|v| precompute_norms(v)).collect();
        let norms_b: Vec<PqNormCache> = cloud_b.iter().map(|v| precompute_norms(v)).collect();

        let d_ab = chamfer_distance(&cloud_a, &norms_a, &cloud_b, &norms_b);
        let d_ba = chamfer_distance(&cloud_b, &norms_b, &cloud_a, &norms_a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "Chamfer 对称性: d(A,B)={} vs d(B,A)={}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_precompute_norms_positive() {
        // 非零向量的每个子空间范数都应大于 0
        let v = make_test_vec(42.0);
        let cache = precompute_norms(&v);
        for (s, &norm) in cache.norms.iter().enumerate() {
            assert!(
                norm > 0.0,
                "子空间 {} 的范数应 > 0，实际 = {}",
                s,
                norm
            );
        }
    }

    #[test]
    fn test_sub_dot_product_consistency() {
        // sub_dot_product 与朴素循环结果一致
        let a: Vec<f32> = (0..SUB_DIM).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..SUB_DIM).map(|i| (i as f32) * 0.2 + 0.5).collect();

        let fast = sub_dot_product(&a, &b);
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (fast - naive).abs() < 1e-3,
            "sub_dot_product 与朴素结果应一致: fast={} naive={}",
            fast,
            naive
        );
    }
}
