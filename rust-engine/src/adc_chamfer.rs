//! ADC 近似 Chamfer 距离 (V14-PQ)
//!
//! ADC (Asymmetric Distance Computation): query 保持原始 f32，
//! doc 用 PQ code 查表，实现近似 cosine distance。
//! 查表将 O(64) 的子空间 dot product 降为 O(1) 的表查找。

use crate::cloud_store::DocumentCloud;
use crate::inverted_index::NUM_CENTROIDS;
use crate::pq_chamfer::{NUM_SUBSPACES, SUB_DIM};
use crate::pq_store::{PqCodebook, PqDocumentCloud, PqStore};
use rayon::prelude::*;

/// ADC 距离表：对一个 query token 预计算到所有码本中心的 cosine distance
/// table[s * NUM_CENTROIDS + c] = cosine_distance(query_sub_s, codebook[s][c])
struct AdcDistTable {
    table: Vec<f32>, // [NUM_SUBSPACES * NUM_CENTROIDS] = 64 * 256 = 16384 f32 = 64 KB
}

impl AdcDistTable {
    /// 为一个 query token 构建距离表
    fn build(query_vec: &[f32], codebook: &PqCodebook) -> Self {
        let mut table = Vec::with_capacity(NUM_SUBSPACES * NUM_CENTROIDS);

        for s in 0..NUM_SUBSPACES {
            let off = s * SUB_DIM;
            let q_sub = &query_vec[off..off + SUB_DIM];

            // 计算 query 子向量范数
            let mut q_norm_sq = 0.0f32;
            for chunk in q_sub.chunks_exact(4) {
                q_norm_sq += chunk[0] * chunk[0]
                    + chunk[1] * chunk[1]
                    + chunk[2] * chunk[2]
                    + chunk[3] * chunk[3];
            }
            let q_norm = q_norm_sq.sqrt();

            for c in 0..NUM_CENTROIDS {
                let c_norm = codebook.centroid_norm(s, c);
                let denom = q_norm * c_norm;
                if denom < 1e-16 {
                    table.push(1.0);
                    continue;
                }
                let cent = codebook.centroid(s, c);
                // dot product (4-way unroll)
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                let mut acc3 = 0.0f32;
                for (qa, ca) in q_sub.chunks_exact(4).zip(cent.chunks_exact(4)) {
                    acc0 += qa[0] * ca[0];
                    acc1 += qa[1] * ca[1];
                    acc2 += qa[2] * ca[2];
                    acc3 += qa[3] * ca[3];
                }
                let dot = (acc0 + acc1) + (acc2 + acc3);
                let cos_dist = (1.0 - dot / (denom + 1e-8)).max(0.0);
                table.push(cos_dist);
            }
        }

        Self { table }
    }

    /// 查表：query token 到 doc token (由 PQ code 表示) 在子空间 s 的近似距离
    #[inline(always)]
    fn lookup(&self, s: usize, code: usize) -> f32 {
        self.table[s * NUM_CENTROIDS + code]
    }
}

/// ADC 近似 PQ-Chamfer 距离
///
/// query 保持原始 f32，doc 用 PQ code 查表
/// 逻辑结构与 token_chamfer.rs 的 token_pq_chamfer 完全平行
pub fn adc_pq_chamfer(
    query_cloud: &DocumentCloud,
    doc_pq: &PqDocumentCloud,
    codebook: &PqCodebook,
) -> f64 {
    let nq = query_cloud.n_sentences;
    let nd = doc_pq.n_tokens;
    if nq == 0 || nd == 0 {
        return f64::MAX;
    }
    let nq_f = nq as f32;
    let nd_f = nd as f32;

    // 1. 为每个 query token 预计算 ADC 距离表
    let dist_tables: Vec<AdcDistTable> = (0..nq)
        .map(|qi| AdcDistTable::build(query_cloud.sentence(qi), codebook))
        .collect();

    // 2. 分配距离矩阵（所有子空间共享复用）
    let mut dist_matrix = vec![0.0f32; nq * nd];
    let mut total = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        // 2a. 用查表填充距离矩阵
        for qi in 0..nq {
            let row_off = qi * nd;
            for di in 0..nd {
                let code = doc_pq.code(di, s);
                dist_matrix[row_off + di] = dist_tables[qi].lookup(s, code);
            }
        }

        // 2b. Q->D：每行取 min
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

        // 2c. D->Q：每列取 min
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

        total += sum_qd / nq_f + sum_dq / nd_f;
    }

    (total / NUM_SUBSPACES as f32) as f64
}

/// ADC 两阶段检索：倒排粗筛 + ADC PQ-Chamfer 精排
pub fn adc_two_stage(
    query_cloud: &DocumentCloud,
    pq_store: &PqStore,
    candidate_indices: &[usize], // 粗筛结果（doc indices in pq_store）
    top_n: usize,
) -> Vec<(u32, f64)> {
    let mut fine_scores: Vec<(u32, f64)> = candidate_indices
        .par_iter()
        .map(|&idx| {
            let doc_pq = &pq_store.documents[idx];
            let dist = adc_pq_chamfer(query_cloud, doc_pq, &pq_store.codebook);
            (doc_pq.doc_id, dist)
        })
        .collect();

    fine_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    fine_scores.truncate(top_n);
    fine_scores
}

/// 质心粗筛 + ADC 精排的完整管线
/// 用预计算的质心做粗筛（与现有管线相同），但精排用 ADC 替代原始 Chamfer
pub fn centroid_adc_two_stage(
    query_cloud: &DocumentCloud,
    pq_store: &PqStore,
    centroids: &[Vec<f32>],
    coarse_top: usize,
    top_n: usize,
) -> Vec<(u32, f64)> {
    let nq = query_cloud.n_sentences;
    if nq == 0 {
        return vec![];
    }

    // 粗筛：与现有方式一致，但索引映射到 pq_store
    let mut coarse_scores: Vec<(usize, f64)> = centroids
        .par_iter()
        .enumerate()
        .map(|(idx, centroid)| {
            // 简化 Chamfer：query tokens vs 单质心
            let mut total = 0.0f32;
            for s in 0..NUM_SUBSPACES {
                let off = s * SUB_DIM;
                let c_sub = &centroid[off..off + SUB_DIM];
                let mut c_norm_sq = 0.0f32;
                for ch in c_sub.chunks_exact(4) {
                    c_norm_sq +=
                        ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3];
                }
                let c_norm = c_norm_sq.sqrt();

                let mut sum_qd = 0.0f32;
                let mut min_dq = f32::MAX;
                for qi in 0..nq {
                    let q_sub = &query_cloud.sentence(qi)[off..off + SUB_DIM];
                    let q_norm = query_cloud.norm_caches[qi].norms[s];
                    let denom = q_norm * c_norm;
                    let d = if denom < 1e-16 {
                        1.0f32
                    } else {
                        let mut acc0 = 0.0f32;
                        let mut acc1 = 0.0f32;
                        let mut acc2 = 0.0f32;
                        let mut acc3 = 0.0f32;
                        for (qa, ca) in q_sub.chunks_exact(4).zip(c_sub.chunks_exact(4)) {
                            acc0 += qa[0] * ca[0];
                            acc1 += qa[1] * ca[1];
                            acc2 += qa[2] * ca[2];
                            acc3 += qa[3] * ca[3];
                        }
                        let dot = (acc0 + acc1) + (acc2 + acc3);
                        (1.0 - dot / (denom + 1e-8)).max(0.0)
                    };
                    sum_qd += d;
                    if d < min_dq {
                        min_dq = d;
                    }
                }
                total += sum_qd / nq as f32 + min_dq;
            }
            (idx, (total / NUM_SUBSPACES as f32) as f64)
        })
        .collect();

    coarse_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    coarse_scores.truncate(coarse_top);

    // 精排：ADC
    let candidate_indices: Vec<usize> = coarse_scores.iter().map(|&(idx, _)| idx).collect();
    adc_two_stage(query_cloud, pq_store, &candidate_indices, top_n)
}
