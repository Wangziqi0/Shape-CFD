// law-vexus/src/residual.rs
// D9-D: 内生残差预计算（骨架实现）
//
// 法律版内生残差：基于 VCP V7 算法，使用 nalgebra SVD 计算
// 每个法条在其共现邻居空间中的投影残差能量。
// 残差高 → 法条独特（特别法条），残差低 → 法条冗余（通则类）。

use nalgebra::DMatrix;
use napi::Task;
use napi_derive::napi;
use rusqlite::Connection;
use std::collections::HashMap;

/// 内生残差计算参数
pub struct ResidualParams {
    /// 最大保留主成分数 (默认 8)
    pub max_svd_rank: usize,
    /// 最少邻居数，少于则跳过 (默认 3)
    pub min_neighbors: usize,
    /// 最大邻居数，防 SVD 爆炸 (默认 100)
    pub max_neighbors: usize,
}

impl Default for ResidualParams {
    fn default() -> Self {
        Self {
            max_svd_rank: 8,
            min_neighbors: 3,
            max_neighbors: 100,
        }
    }
}

/// 残差计算结果（NAPI 返回值）
#[napi(object)]
pub struct ResidualTaskResult {
    pub chunk_count: u32,
    pub computed_count: u32,
    pub skipped_count: u32,
    pub elapsed_ms: f64,
}

/// 异步任务：法律内生残差预计算
/// 使用 NAPI-RS AsyncTask，在独立线程执行，不阻塞 Node.js
pub struct LawResidualTask {
    pub db_path: String,
    pub dimensions: u32,
    pub params: ResidualParams,
}

impl Task for LawResidualTask {
    type Output = ResidualTaskResult;
    type JsValue = ResidualTaskResult;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let start = std::time::Instant::now();
        let dim = self.dimensions as usize;

        let conn = Connection::open(&self.db_path)
            .map_err(|e| napi::Error::from_reason(format!("DB open failed: {}", e)))?;

        // 1. 加载所有法条向量（从 chunks 表）
        let mut chunk_vectors: HashMap<i64, Vec<f32>> = HashMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT id, vector FROM chunks WHERE vector IS NOT NULL")
                .map_err(|e| napi::Error::from_reason(format!("Prepare failed: {}", e)))?;

            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| napi::Error::from_reason(format!("Query failed: {}", e)))?;

            for row in rows {
                if let Ok((id, bytes)) = row {
                    if bytes.len() == dim * 4 {
                        let vec: Vec<f32> = bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                            .collect();
                        chunk_vectors.insert(id, vec);
                    }
                }
            }
        }

        // 2. 构建邻居关系（同一法律文件中的法条互为邻居）
        let mut adjacency: HashMap<i64, Vec<i64>> = HashMap::new();
        {
            let mut stmt = conn
                .prepare(
                    "SELECT DISTINCT c1.id, c2.id FROM chunks c1
                     JOIN chunks c2 ON c1.file_id = c2.file_id AND c1.id != c2.id
                     WHERE c1.vector IS NOT NULL AND c2.vector IS NOT NULL",
                )
                .map_err(|e| {
                    napi::Error::from_reason(format!("Adjacency query failed: {}", e))
                })?;

            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(|e| napi::Error::from_reason(format!("Query failed: {}", e)))?;

            for row in rows {
                if let Ok((src, tgt)) = row {
                    adjacency.entry(src).or_insert_with(Vec::new).push(tgt);
                }
            }
        }

        // 3. 对每个法条计算残差
        let mut results: Vec<(i64, f64, usize)> = Vec::new();
        let mut computed = 0u32;
        let mut skipped = 0u32;

        for (&chunk_id, chunk_vec) in &chunk_vectors {
            let neighbors = match adjacency.get(&chunk_id) {
                Some(n) => n,
                None => {
                    skipped += 1;
                    continue;
                }
            };

            // 收集邻居向量（限制数量）
            let mut neighbor_vecs: Vec<&Vec<f32>> = Vec::new();
            for nid in neighbors {
                if let Some(v) = chunk_vectors.get(nid) {
                    neighbor_vecs.push(v);
                    if neighbor_vecs.len() >= self.params.max_neighbors {
                        break;
                    }
                }
            }

            if neighbor_vecs.len() < self.params.min_neighbors {
                skipped += 1;
                continue;
            }

            // 构建邻居矩阵
            let n = neighbor_vecs.len();
            let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
            for v in &neighbor_vecs {
                flat.extend_from_slice(v);
            }

            // SVD 分解
            let matrix = DMatrix::from_row_slice(n, dim, &flat);
            let svd = matrix.svd(false, true);
            let v_t = match svd.v_t {
                Some(ref vt) => vt,
                None => {
                    skipped += 1;
                    continue;
                }
            };

            let k = std::cmp::min(
                self.params.max_svd_rank,
                std::cmp::min(n, dim),
            );

            // 计算投影
            let mut projection = vec![0.0f64; dim];
            for i in 0..k {
                let mut dot = 0.0f64;
                for d in 0..dim {
                    dot += (chunk_vec[d] as f64) * (v_t[(i, d)] as f64);
                }
                for d in 0..dim {
                    projection[d] += dot * (v_t[(i, d)] as f64);
                }
            }

            // 残差能量
            let mut residual_sq = 0.0f64;
            for d in 0..dim {
                let diff = (chunk_vec[d] as f64) - projection[d];
                residual_sq += diff * diff;
            }
            let residual_energy = residual_sq.sqrt();

            results.push((chunk_id, residual_energy, n));
            computed += 1;
        }

        // 4. 归一化到 [0.5, 2.0] 并写入 SQLite
        if !results.is_empty() {
            let max_r = results.iter().map(|r| r.1).fold(0.0f64, f64::max);
            let min_r = results.iter().map(|r| r.1).fold(f64::MAX, f64::min);
            let range = max_r - min_r;

            let mut conn = conn;
            let tx = conn
                .transaction()
                .map_err(|e| napi::Error::from_reason(format!("Tx failed: {}", e)))?;

            tx.execute(
                "CREATE TABLE IF NOT EXISTS law_intrinsic_residuals (
                    chunk_id INTEGER PRIMARY KEY,
                    residual_energy REAL NOT NULL,
                    neighbor_count INTEGER NOT NULL
                )",
                [],
            )
            .map_err(|e| napi::Error::from_reason(format!("Create table failed: {}", e)))?;

            tx.execute("DELETE FROM law_intrinsic_residuals", [])
                .map_err(|e| napi::Error::from_reason(format!("Clear failed: {}", e)))?;

            let mut insert = tx
                .prepare(
                    "INSERT INTO law_intrinsic_residuals (chunk_id, residual_energy, neighbor_count)
                     VALUES (?1, ?2, ?3)",
                )
                .map_err(|e| napi::Error::from_reason(format!("Prepare failed: {}", e)))?;

            for (chunk_id, raw_residual, n_count) in &results {
                let normalized = if range > 1e-9 {
                    0.5 + 1.5 * ((raw_residual - min_r) / range)
                } else {
                    1.0
                };
                insert
                    .execute(rusqlite::params![chunk_id, normalized, *n_count as i64])
                    .map_err(|e| napi::Error::from_reason(format!("Insert failed: {}", e)))?;
            }

            drop(insert);
            tx.commit()
                .map_err(|e| napi::Error::from_reason(format!("Commit failed: {}", e)))?;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(ResidualTaskResult {
            chunk_count: chunk_vectors.len() as u32,
            computed_count: computed,
            skipped_count: skipped,
            elapsed_ms: elapsed,
        })
    }

    fn resolve(&mut self, _env: napi::Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}
