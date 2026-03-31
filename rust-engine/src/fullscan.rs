// law-vexus/src/fullscan.rs
// 全库 PQ-Chamfer 暴力扫描模块
//
// 遍历全库每篇文档的点云，计算与 query 点云的 PQ-Chamfer 距离，返回 top-K。
// 自包含实现，不依赖其他模块。

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;

use crate::pq_chamfer::{self, PqNormCache, FULL_DIM, NUM_SUBSPACES, SUB_DIM};

// ============================================================================
// ScanHit — 扫描结果
// ============================================================================

/// 单条扫描命中结果
#[derive(Clone, Debug)]
pub struct ScanHit {
    pub doc_id: u32,
    pub distance: f32,
}

// ScanHit 的排序：按 distance 降序（大的排前面）
// 这样 BinaryHeap（默认 max-heap）的 peek() 返回距离最大的元素，
// 方便在 push 时与堆顶比较，实现 top-K 最小距离筛选。
impl Ord for ScanHit {
    fn cmp(&self, other: &Self) -> Ordering {
        // 降序：other 在前（距离大的排序值大）
        // 使用 partial_cmp + unwrap_or(Equal) 处理 NaN
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .reverse()
    }
}

impl PartialOrd for ScanHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ScanHit {}

impl PartialEq for ScanHit {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.doc_id == other.doc_id
    }
}

// ============================================================================
// TopKHeap — 维护 top-K 最小距离
// ============================================================================

/// Top-K 最小堆维护器
///
/// 内部用 max-heap（BinaryHeap），堆顶是当前 K 个中距离最大的。
/// 新元素距离 < 堆顶时才替换，保证最终留下距离最小的 K 个。
struct TopKHeap {
    heap: BinaryHeap<ScanHit>,
    k: usize,
}

impl TopKHeap {
    /// 创建容量为 k 的 TopK 维护器
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    /// 尝试将 hit 加入堆
    ///
    /// - 堆未满时直接 push
    /// - 堆满时，仅当 hit.distance < 堆顶 distance 时替换堆顶
    fn push(&mut self, hit: ScanHit) {
        if self.heap.len() < self.k {
            self.heap.push(hit);
        } else if let Some(top) = self.heap.peek() {
            if hit.distance < top.distance {
                // 弹出最大的，压入更小的
                self.heap.pop();
                self.heap.push(hit);
            }
        }
    }

    /// 消费堆，按 distance 升序返回结果
    fn into_sorted(self) -> Vec<ScanHit> {
        let mut results: Vec<ScanHit> = self.heap.into_vec();
        // 按 distance 升序排列
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        results
    }
}

// PqNormCache 和 precompute_norms 统一使用 crate::pq_chamfer 模块
fn precompute_norms(vec: &[f32]) -> PqNormCache {
    pq_chamfer::precompute_norms(vec)
}

// ============================================================================
// PQ 子空间点积 & 余弦距离
// ============================================================================

/// 子空间点积，4-way unroll
///
/// a, b: 长度为 SUB_DIM 的子空间切片
#[inline]
fn sub_dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let len = a.len();
    let main_end = len - (len % 4);

    // 4-way unroll 主循环
    let mut i = 0;
    while i < main_end {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }

    // 处理余数
    let mut tail = 0.0f32;
    while i < len {
        tail += a[i] * b[i];
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3 + tail
}

/// PQ 余弦距离（带缓存范数）
///
/// 将完整向量拆分为子空间，逐子空间计算余弦相似度后取平均，
/// 最终返回 1 - avg_cosine 作为距离。
///
/// 范数缓存避免了对同一向量的重复范数计算。
fn pq_cosine_distance_cached(
    a: &[f32],
    a_norms: &PqNormCache,
    b: &[f32],
    b_norms: &PqNormCache,
) -> f32 {
    debug_assert!(a.len() >= FULL_DIM);
    debug_assert!(b.len() >= FULL_DIM);

    let mut cosine_sum = 0.0f32;

    for s in 0..NUM_SUBSPACES {
        let offset = s * SUB_DIM;
        let sub_a = &a[offset..offset + SUB_DIM];
        let sub_b = &b[offset..offset + SUB_DIM];

        let norm_a = a_norms.norms[s];
        let norm_b = b_norms.norms[s];

        // 范数为零时跳过（零向量子空间贡献为 0）
        if norm_a < 1e-12 || norm_b < 1e-12 {
            continue;
        }

        let dot = sub_dot_product(sub_a, sub_b);
        let cosine = dot / (norm_a * norm_b);
        // clamp 到 [-1, 1] 防止浮点误差
        cosine_sum += cosine.clamp(-1.0, 1.0);
    }

    // 平均余弦相似度 → 距离
    let avg_cosine = cosine_sum / NUM_SUBSPACES as f32;
    1.0 - avg_cosine
}

// ============================================================================
// Chamfer 距离
// ============================================================================

/// 对称 Chamfer 距离
///
/// 公式: d_chamfer(A, B) = (1/|A|) * Σ_{a∈A} min_{b∈B} d(a,b)
///                        + (1/|B|) * Σ_{b∈B} min_{a∈A} d(b,a)
///
/// 其中 d(a,b) 为 PQ 余弦距离。
fn chamfer_distance_internal(
    cloud_a: &[&[f32]],
    a_norms: &[PqNormCache],
    cloud_b: &[&[f32]],
    b_norms: &[PqNormCache],
) -> f32 {
    if cloud_a.is_empty() || cloud_b.is_empty() {
        return f32::MAX;
    }

    // A→B 方向：对 A 中每个点，找 B 中最近点
    let mut sum_a_to_b = 0.0f32;
    for (i, &vec_a) in cloud_a.iter().enumerate() {
        let mut min_dist = f32::MAX;
        for (j, &vec_b) in cloud_b.iter().enumerate() {
            let dist = pq_cosine_distance_cached(vec_a, &a_norms[i], vec_b, &b_norms[j]);
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum_a_to_b += min_dist;
    }

    // B→A 方向：对 B 中每个点，找 A 中最近点
    let mut sum_b_to_a = 0.0f32;
    for (j, &vec_b) in cloud_b.iter().enumerate() {
        let mut min_dist = f32::MAX;
        for (i, &vec_a) in cloud_a.iter().enumerate() {
            let dist = pq_cosine_distance_cached(vec_b, &b_norms[j], vec_a, &a_norms[i]);
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum_b_to_a += min_dist;
    }

    // 对称 Chamfer = 两个方向的平均最近距离之和
    sum_a_to_b / cloud_a.len() as f32 + sum_b_to_a / cloud_b.len() as f32
}

// ============================================================================
// 全库暴力扫描入口
// ============================================================================

/// 全库 PQ-Chamfer 暴力扫描（单线程版本）
///
/// 遍历全库每篇文档的点云，计算与 query 点云的 PQ-Chamfer 距离，返回距离最小的 top-K。
///
/// # 参数
///
/// - `query_vecs`: query 的句子向量列表，每个是长度为 FULL_DIM 的 f32 切片
/// - `docs`: 全库文档数据元组切片
///   - `doc_id`: 文档 ID
///   - `vectors`: 该文档所有句子向量的连续存储（长度 = n_sentences * FULL_DIM）
///   - `n_sentences`: 该文档包含的句子数
///   - `norms`: 该文档每个句子的预计算范数缓存
/// - `k`: 返回 top-K 个最近文档
/// - `exclude_ids`: 需要排除的文档 ID 集合
///
/// # 返回
///
/// 按 distance 升序排列的 top-K ScanHit 列表
pub fn fullscan(
    query_vecs: &[&[f32]],
    docs: &[(u32, &[f32], usize, &[PqNormCache])],
    k: usize,
    exclude_ids: &HashSet<u32>,
) -> Vec<ScanHit> {
    if k == 0 || query_vecs.is_empty() || docs.is_empty() {
        return Vec::new();
    }

    // 预计算 query 向量的范数缓存
    let query_norms: Vec<PqNormCache> = query_vecs.iter().map(|v| precompute_norms(v)).collect();

    let mut topk = TopKHeap::new(k);

    for &(doc_id, vectors, n_sentences, ref doc_norms) in docs {
        // 跳过排除列表中的文档
        if exclude_ids.contains(&doc_id) {
            continue;
        }

        // 文档为空则跳过
        if n_sentences == 0 {
            continue;
        }

        // 将连续存储的向量拆分为句子切片
        let doc_vecs: Vec<&[f32]> = (0..n_sentences)
            .map(|i| &vectors[i * FULL_DIM..(i + 1) * FULL_DIM])
            .collect();

        // 计算 query 点云与文档点云之间的对称 Chamfer 距离
        let dist = chamfer_distance_internal(query_vecs, &query_norms, &doc_vecs, doc_norms);

        topk.push(ScanHit {
            doc_id,
            distance: dist,
        });
    }

    topk.into_sorted()
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成伪随机 f32 值（简易 LCG，不依赖外部 crate）
    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // LCG: state = (a * state + c) mod m
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // 映射到 [-1, 1]
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// 辅助：为向量集合预计算范数缓存
    fn make_norms(vecs: &[f32], n: usize) -> Vec<PqNormCache> {
        (0..n)
            .map(|i| precompute_norms(&vecs[i * FULL_DIM..(i + 1) * FULL_DIM]))
            .collect()
    }

    #[test]
    fn test_fullscan_returns_correct_count() {
        // 构造 query: 2 个句子向量
        let q0 = pseudo_random_vec(42, FULL_DIM);
        let q1 = pseudo_random_vec(43, FULL_DIM);
        let query_vecs: Vec<&[f32]> = vec![&q0, &q1];

        // 构造 10 篇文档，每篇 1~3 个句子
        let mut docs_data: Vec<(Vec<f32>, usize)> = Vec::new();
        for i in 0..10u64 {
            let n_sents = (i % 3 + 1) as usize;
            let vecs = pseudo_random_vec(100 + i, n_sents * FULL_DIM);
            docs_data.push((vecs, n_sents));
        }

        let norms_list: Vec<Vec<PqNormCache>> = docs_data
            .iter()
            .map(|(v, n)| make_norms(v, *n))
            .collect();

        let docs: Vec<(u32, &[f32], usize, &[PqNormCache])> = docs_data
            .iter()
            .zip(norms_list.iter())
            .enumerate()
            .map(|(i, ((v, n), norms))| (i as u32, v.as_slice(), *n, norms.as_slice()))
            .collect();

        let exclude = HashSet::new();

        // k=5，应该返回 5 条
        let results = fullscan(&query_vecs, &docs, 5, &exclude);
        assert_eq!(results.len(), 5, "应返回 top-5 结果");

        // k=20 > 文档数 10，应返回 10 条
        let results_all = fullscan(&query_vecs, &docs, 20, &exclude);
        assert_eq!(results_all.len(), 10, "k 超过文档总数时应返回全部");
    }

    #[test]
    fn test_fullscan_exclude_ids() {
        let q0 = pseudo_random_vec(42, FULL_DIM);
        let query_vecs: Vec<&[f32]> = vec![&q0];

        // 5 篇文档
        let mut docs_data: Vec<(Vec<f32>, usize)> = Vec::new();
        for i in 0..5u64 {
            let vecs = pseudo_random_vec(200 + i, FULL_DIM);
            docs_data.push((vecs, 1));
        }

        let norms_list: Vec<Vec<PqNormCache>> = docs_data
            .iter()
            .map(|(v, n)| make_norms(v, *n))
            .collect();

        let docs: Vec<(u32, &[f32], usize, &[PqNormCache])> = docs_data
            .iter()
            .zip(norms_list.iter())
            .enumerate()
            .map(|(i, ((v, n), norms))| (i as u32, v.as_slice(), *n, norms.as_slice()))
            .collect();

        // 排除 doc_id 1 和 3
        let mut exclude = HashSet::new();
        exclude.insert(1u32);
        exclude.insert(3u32);

        let results = fullscan(&query_vecs, &docs, 10, &exclude);
        assert_eq!(results.len(), 3, "排除 2 篇后应剩 3 篇");

        // 确认排除的 ID 不在结果中
        let result_ids: HashSet<u32> = results.iter().map(|h| h.doc_id).collect();
        assert!(!result_ids.contains(&1), "doc_id=1 应被排除");
        assert!(!result_ids.contains(&3), "doc_id=3 应被排除");
    }

    #[test]
    fn test_fullscan_sorted_ascending() {
        let q0 = pseudo_random_vec(42, FULL_DIM);
        let query_vecs: Vec<&[f32]> = vec![&q0];

        // 8 篇文档
        let mut docs_data: Vec<(Vec<f32>, usize)> = Vec::new();
        for i in 0..8u64 {
            let vecs = pseudo_random_vec(300 + i, FULL_DIM);
            docs_data.push((vecs, 1));
        }

        let norms_list: Vec<Vec<PqNormCache>> = docs_data
            .iter()
            .map(|(v, n)| make_norms(v, *n))
            .collect();

        let docs: Vec<(u32, &[f32], usize, &[PqNormCache])> = docs_data
            .iter()
            .zip(norms_list.iter())
            .enumerate()
            .map(|(i, ((v, n), norms))| (i as u32, v.as_slice(), *n, norms.as_slice()))
            .collect();

        let exclude = HashSet::new();
        let results = fullscan(&query_vecs, &docs, 5, &exclude);

        // 验证结果按 distance 升序排列
        for window in results.windows(2) {
            assert!(
                window[0].distance <= window[1].distance,
                "结果未按 distance 升序排列: {} > {}",
                window[0].distance,
                window[1].distance,
            );
        }
    }

    #[test]
    fn test_topk_heap_basic() {
        let mut heap = TopKHeap::new(3);

        // 插入 5 个元素
        heap.push(ScanHit { doc_id: 0, distance: 0.5 });
        heap.push(ScanHit { doc_id: 1, distance: 0.9 });
        heap.push(ScanHit { doc_id: 2, distance: 0.1 });
        heap.push(ScanHit { doc_id: 3, distance: 0.3 });
        heap.push(ScanHit { doc_id: 4, distance: 0.7 });

        let results = heap.into_sorted();
        assert_eq!(results.len(), 3, "top-3 应返回 3 个");
        // 最小的 3 个距离: 0.1, 0.3, 0.5
        assert_eq!(results[0].doc_id, 2);
        assert_eq!(results[1].doc_id, 3);
        assert_eq!(results[2].doc_id, 0);
    }

    #[test]
    fn test_pq_cosine_distance_identical_vectors() {
        // 相同向量的 PQ 余弦距离应接近 0
        let v = pseudo_random_vec(999, FULL_DIM);
        let norm = precompute_norms(&v);
        let dist = pq_cosine_distance_cached(&v, &norm, &v, &norm);
        assert!(
            dist.abs() < 1e-5,
            "相同向量的 PQ 余弦距离应接近 0，实际: {}",
            dist
        );
    }

    #[test]
    fn test_sub_dot_product_correctness() {
        let a: Vec<f32> = (0..SUB_DIM).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..SUB_DIM).map(|i| (SUB_DIM - i) as f32 * 0.1).collect();

        let result = sub_dot_product(&a, &b);

        // 朴素计算对照
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1e-3,
            "sub_dot_product 结果不正确: {} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_fullscan_empty_inputs() {
        let q0 = pseudo_random_vec(42, FULL_DIM);
        let query_vecs: Vec<&[f32]> = vec![&q0];
        let exclude = HashSet::new();

        // 空文档列表
        let results = fullscan(&query_vecs, &[], 5, &exclude);
        assert!(results.is_empty(), "空文档列表应返回空结果");

        // 空 query
        let empty_query: Vec<&[f32]> = vec![];
        let vecs = pseudo_random_vec(100, FULL_DIM);
        let norms = vec![precompute_norms(&vecs)];
        let docs = vec![(0u32, vecs.as_slice(), 1usize, norms.as_slice())];
        let results = fullscan(&empty_query, &docs, 5, &exclude);
        assert!(results.is_empty(), "空 query 应返回空结果");

        // k=0
        let results = fullscan(&query_vecs, &docs, 0, &exclude);
        assert!(results.is_empty(), "k=0 应返回空结果");
    }
}
