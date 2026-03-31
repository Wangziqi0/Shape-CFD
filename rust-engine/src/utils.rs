//! D9-E — 类型转换与工具函数
//!
//! Buffer / Float32Array 转换工具，确保 Node.js (Float32Array) 和 Rust (&[f32]) 间的类型安全。

use anyhow::{anyhow, Result};

/// 安全地将 Buffer（字节切片）转换为 f32 切片
///
/// 执行对齐检查和维度校验，防止 UB。
///
/// # Arguments
/// - `buffer`: 原始字节数据
/// - `expected_dim`: 期望的向量维度
///
/// # Safety
/// 仅在对齐检查通过后才执行 unsafe 转换
pub fn buffer_to_f32_slice(buffer: &[u8], expected_dim: usize) -> Result<&[f32]> {
    let expected_bytes = expected_dim * std::mem::size_of::<f32>();
    if buffer.len() != expected_bytes {
        return Err(anyhow!(
            "Buffer 长度不匹配: 期望 {} 字节 ({}维 × 4B), 实际 {} 字节",
            expected_bytes,
            expected_dim,
            buffer.len()
        ));
    }

    // 对齐检查
    if (buffer.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
        return Err(anyhow!("Buffer 未对齐到 4 字节边界"));
    }

    Ok(unsafe {
        std::slice::from_raw_parts(buffer.as_ptr() as *const f32, expected_dim)
    })
}

/// 将 SQLite BLOB 读取的 Vec<u8> 转换为 Vec<f32>
///
/// 用于 RecoverTask 从 SQLite 加载向量数据。
pub fn blob_to_vec_f32(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
    if blob.len() != dim * 4 {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

/// Vec<f32> → SQLite BLOB
///
/// 用于将向量写入 SQLite 的 vector BLOB 字段。
pub fn vec_f32_to_blob(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// 余弦相似度计算
///
/// 用于需要在 Rust 端计算相似度的场景（如 RRF 融合前的重排序）。
#[allow(dead_code)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "向量维度不匹配");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
