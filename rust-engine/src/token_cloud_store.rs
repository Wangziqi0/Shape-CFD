// law-vexus/src/token_cloud_store.rs
// Token 级点云内存存储
//
// 从 FP16 二进制文件 + JSON 索引加载 token-level 4096d 向量点云，
// 供 V11 Token-Chamfer 距离计算使用。
// 与 cloud_store.rs（句子级）互补，不做 PQ norm cache。

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

/// Token 级向量维度
pub const TOKEN_DIM: usize = 4096;

/// FP16 每个分量字节数
const FP16_BYTES: usize = 2;

/// 单个文档的 token 级点云
#[derive(Debug)]
pub struct TokenDocumentCloud {
    /// 所有 token 向量连续存储，长度 = n_tokens * TOKEN_DIM (f32)
    pub vectors: Vec<f32>,
    /// token 数量
    pub n_tokens: usize,
}

impl TokenDocumentCloud {
    /// 获取第 i 个 token 向量的切片 (长度 = TOKEN_DIM)
    pub fn token(&self, i: usize) -> &[f32] {
        let start = i * TOKEN_DIM;
        &self.vectors[start..start + TOKEN_DIM]
    }

    /// 计算所有 token 的均值向量（质心）
    pub fn centroid(&self) -> Vec<f32> {
        if self.n_tokens == 0 {
            return vec![0.0; TOKEN_DIM];
        }
        let mut sum = vec![0.0f64; TOKEN_DIM];
        for i in 0..self.n_tokens {
            let tok = self.token(i);
            for d in 0..TOKEN_DIM {
                sum[d] += tok[d] as f64;
            }
        }
        let inv = 1.0 / self.n_tokens as f64;
        sum.iter().map(|&s| (s * inv) as f32).collect()
    }
}

/// 全库 token 级点云存储
///
/// 使用 String key（兼容 "MED-123" 等非纯数字 doc_id 格式）
#[derive(Debug)]
pub struct TokenCloudStore {
    /// doc_id (字符串) -> token 点云
    documents: HashMap<String, TokenDocumentCloud>,
    /// 向量维度
    pub dim: usize,
}

/// token_index.json 中单个文档的条目
#[derive(serde::Deserialize)]
struct TokenIndexEntry {
    /// 在 bin 文件中的字节偏移
    offset: usize,
    /// token 数量
    n_tokens: usize,
}

impl TokenCloudStore {
    /// 从 FP16 二进制文件 + JSON 索引加载全部 token 点云
    ///
    /// - bin_path: token_clouds_fp16.bin 路径
    /// - index_path: token_index.json 路径
    ///
    /// 流程：
    /// 1. 读取 index JSON，解析每个 doc 的 offset 和 n_tokens
    /// 2. mmap 读取 bin 文件
    /// 3. 对每个 doc 区间读取 FP16 数据，转为 f32
    /// 4. 存入 HashMap
    pub fn load_from_binary(bin_path: &str, index_path: &str) -> Result<Self, String> {
        // 1. 读取并解析 index JSON
        let index_file = File::open(index_path)
            .map_err(|e| format!("打开索引文件失败: {} — {}", index_path, e))?;
        let reader = BufReader::new(index_file);
        let index_map: HashMap<String, TokenIndexEntry> = serde_json::from_reader(reader)
            .map_err(|e| format!("解析索引 JSON 失败: {}", e))?;

        // 2. mmap 打开 bin 文件
        let bin_file = File::open(bin_path)
            .map_err(|e| format!("打开二进制文件失败: {} — {}", bin_path, e))?;
        let mmap = unsafe {
            Mmap::map(&bin_file)
                .map_err(|e| format!("mmap 映射失败: {}", e))?
        };
        let bin_len = mmap.len();

        // 3. 逐文档解析 FP16 → f32
        let mut documents = HashMap::with_capacity(index_map.len());
        let mut total_tokens: usize = 0;

        for (doc_id, entry) in &index_map {
            let byte_count = entry.n_tokens * TOKEN_DIM * FP16_BYTES;
            let end = entry.offset + byte_count;

            // 边界检查
            if end > bin_len {
                return Err(format!(
                    "文档 {} 数据越界: offset={}, n_tokens={}, 需要 {} 字节, 文件仅 {} 字节",
                    doc_id, entry.offset, entry.n_tokens, end, bin_len
                ));
            }

            let region = &mmap[entry.offset..end];
            let vectors = fp16_bytes_to_f32(region, entry.n_tokens * TOKEN_DIM);

            total_tokens += entry.n_tokens;
            documents.insert(
                doc_id.clone(),
                TokenDocumentCloud {
                    vectors,
                    n_tokens: entry.n_tokens,
                },
            );
        }

        // 4. 打印加载统计
        let mem_bytes = total_tokens * TOKEN_DIM * std::mem::size_of::<f32>();
        eprintln!(
            "[TokenCloudStore] 加载完成: {} 文档, {} tokens, {:.1} GB (f32)",
            documents.len(),
            total_tokens,
            mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        Ok(Self {
            documents,
            dim: TOKEN_DIM,
        })
    }

    /// 获取指定文档的 token 点云引用
    pub fn get(&self, doc_id: &str) -> Option<&TokenDocumentCloud> {
        self.documents.get(doc_id)
    }

    /// 获取所有文档 ID 列表
    pub fn doc_ids(&self) -> Vec<String> {
        self.documents.keys().cloned().collect()
    }

    /// 总 token 数
    pub fn total_tokens(&self) -> usize {
        self.documents.values().map(|d| d.n_tokens).sum()
    }

    /// 文档总数
    pub fn total_docs(&self) -> usize {
        self.documents.len()
    }

    /// 估算内存占用（字节，仅 f32 向量数据）
    pub fn memory_usage(&self) -> usize {
        self.documents
            .values()
            .map(|d| d.vectors.len() * std::mem::size_of::<f32>())
            .sum()
    }
}

// ─── 内部工具 ────────────────────────────────────────────────────────────

/// 将 FP16 字节序列（little-endian）批量转为 f32 向量
///
/// 使用 half crate 的 f16::from_le_bytes → f32 转换
fn fp16_bytes_to_f32(bytes: &[u8], n_floats: usize) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), n_floats * FP16_BYTES);
    let mut result = Vec::with_capacity(n_floats);
    for chunk in bytes.chunks_exact(FP16_BYTES) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f = half::f16::from_bits(bits);
        result.push(f.to_f32());
    }
    result
}

// ─── 测试 ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// 将 f32 值转为 FP16 字节 (little-endian)
    fn f32_to_fp16_bytes(val: f32) -> [u8; 2] {
        half::f16::from_f32(val).to_le_bytes()
    }

    /// 创建测试用的临时 bin 文件和 index JSON
    ///
    /// 3 个文档:
    /// - "doc-0": 2 tokens, 值分别为 1.0 和 2.0（每个 token 全维同值）
    /// - "doc-1": 3 tokens, 值分别为 0.5, 1.5, 2.5
    /// - "doc-2": 1 token, 值 3.0
    fn setup_test_files(dir: &std::path::Path) -> (String, String) {
        let bin_path = dir.join("test_tokens_fp16.bin");
        let index_path = dir.join("test_token_index.json");

        // 构造 bin 数据
        let mut bin_data: Vec<u8> = Vec::new();
        let mut index_map: HashMap<String, serde_json::Value> = HashMap::new();

        // doc-0: 2 tokens
        let offset_0 = bin_data.len();
        for val in &[1.0f32, 2.0f32] {
            for _ in 0..TOKEN_DIM {
                bin_data.extend_from_slice(&f32_to_fp16_bytes(*val));
            }
        }
        index_map.insert(
            "doc-0".to_string(),
            serde_json::json!({"offset": offset_0, "n_tokens": 2}),
        );

        // doc-1: 3 tokens
        let offset_1 = bin_data.len();
        for val in &[0.5f32, 1.5f32, 2.5f32] {
            for _ in 0..TOKEN_DIM {
                bin_data.extend_from_slice(&f32_to_fp16_bytes(*val));
            }
        }
        index_map.insert(
            "doc-1".to_string(),
            serde_json::json!({"offset": offset_1, "n_tokens": 3}),
        );

        // doc-2: 1 token
        let offset_2 = bin_data.len();
        for _ in 0..TOKEN_DIM {
            bin_data.extend_from_slice(&f32_to_fp16_bytes(3.0));
        }
        index_map.insert(
            "doc-2".to_string(),
            serde_json::json!({"offset": offset_2, "n_tokens": 1}),
        );

        // 写入 bin 文件
        let mut f = File::create(&bin_path).unwrap();
        f.write_all(&bin_data).unwrap();

        // 写入 index JSON
        let mut f = File::create(&index_path).unwrap();
        serde_json::to_writer(&mut f, &index_map).unwrap();

        (
            bin_path.to_str().unwrap().to_string(),
            index_path.to_str().unwrap().to_string(),
        )
    }

    #[test]
    fn test_fp16_roundtrip() {
        // FP16 精度有限，验证常见值的往返转换
        for &val in &[0.0f32, 1.0, -1.0, 0.5, 2.5, 3.0] {
            let bytes = f32_to_fp16_bytes(val);
            let bits = u16::from_le_bytes(bytes);
            let recovered = half::f16::from_bits(bits).to_f32();
            assert!(
                (recovered - val).abs() < 0.01,
                "FP16 往返误差过大: {} -> {}",
                val,
                recovered
            );
        }
    }

    #[test]
    fn test_load_and_query() {
        let tmp = std::env::temp_dir().join("token_cloud_test_load");
        let _ = std::fs::create_dir_all(&tmp);
        let (bin_path, index_path) = setup_test_files(&tmp);

        let store = TokenCloudStore::load_from_binary(&bin_path, &index_path)
            .expect("加载失败");

        // 文档数
        assert_eq!(store.total_docs(), 3);
        // 总 token 数: 2 + 3 + 1 = 6
        assert_eq!(store.total_tokens(), 6);

        // 查询 doc-0
        let doc0 = store.get("doc-0").expect("doc-0 应存在");
        assert_eq!(doc0.n_tokens, 2);
        // 第一个 token 全 1.0 (FP16 精度)
        let tok0 = doc0.token(0);
        assert_eq!(tok0.len(), TOKEN_DIM);
        assert!((tok0[0] - 1.0).abs() < 0.01);
        assert!((tok0[TOKEN_DIM - 1] - 1.0).abs() < 0.01);
        // 第二个 token 全 2.0
        let tok1 = doc0.token(1);
        assert!((tok1[0] - 2.0).abs() < 0.01);

        // 查询 doc-1
        let doc1 = store.get("doc-1").expect("doc-1 应存在");
        assert_eq!(doc1.n_tokens, 3);

        // 查询 doc-2
        let doc2 = store.get("doc-2").expect("doc-2 应存在");
        assert_eq!(doc2.n_tokens, 1);
        assert!((doc2.token(0)[0] - 3.0).abs() < 0.01);

        // 不存在的文档
        assert!(store.get("doc-99").is_none());

        // 内存占用: 6 tokens * 4096 * 4 bytes = 98304 bytes
        let expected_mem = 6 * TOKEN_DIM * std::mem::size_of::<f32>();
        assert_eq!(store.memory_usage(), expected_mem);

        // 清理
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_centroid() {
        let tmp = std::env::temp_dir().join("token_cloud_test_centroid");
        let _ = std::fs::create_dir_all(&tmp);
        let (bin_path, index_path) = setup_test_files(&tmp);

        let store = TokenCloudStore::load_from_binary(&bin_path, &index_path)
            .expect("加载失败");

        // doc-0: 2 tokens, 值 1.0 和 2.0 → 质心 = 1.5
        let doc0 = store.get("doc-0").unwrap();
        let centroid = doc0.centroid();
        assert_eq!(centroid.len(), TOKEN_DIM);
        assert!(
            (centroid[0] - 1.5).abs() < 0.02,
            "质心[0] = {}，期望 1.5",
            centroid[0]
        );

        // doc-1: 3 tokens, 值 0.5, 1.5, 2.5 → 质心 = 1.5
        let doc1 = store.get("doc-1").unwrap();
        let centroid1 = doc1.centroid();
        assert!(
            (centroid1[0] - 1.5).abs() < 0.02,
            "质心[0] = {}，期望 1.5",
            centroid1[0]
        );

        // doc-2: 1 token, 值 3.0 → 质心 = 3.0
        let doc2 = store.get("doc-2").unwrap();
        let centroid2 = doc2.centroid();
        assert!(
            (centroid2[0] - 3.0).abs() < 0.01,
            "质心[0] = {}，期望 3.0",
            centroid2[0]
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_boundary_error() {
        // 构造一个过小的 bin 文件，触发越界检查
        let tmp = std::env::temp_dir().join("token_cloud_test_boundary");
        let _ = std::fs::create_dir_all(&tmp);

        let bin_path = tmp.join("small.bin");
        let index_path = tmp.join("bad_index.json");

        // 只写 100 字节
        let mut f = File::create(&bin_path).unwrap();
        f.write_all(&[0u8; 100]).unwrap();

        // 索引声称有 10 个 token（需要 10 * 4096 * 2 = 81920 字节）
        let mut index_map: HashMap<String, serde_json::Value> = HashMap::new();
        index_map.insert(
            "bad-doc".to_string(),
            serde_json::json!({"offset": 0, "n_tokens": 10}),
        );
        let mut f = File::create(&index_path).unwrap();
        serde_json::to_writer(&mut f, &index_map).unwrap();

        let result = TokenCloudStore::load_from_binary(
            bin_path.to_str().unwrap(),
            index_path.to_str().unwrap(),
        );
        assert!(result.is_err(), "应报越界错误");
        assert!(
            result.unwrap_err().contains("越界"),
            "错误信息应包含'越界'"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
