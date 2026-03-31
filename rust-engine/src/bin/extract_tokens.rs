// extract_tokens.rs
// 独立二进制工具：调用 llama.cpp /embedding API (--pooling none) 提取 token-level embeddings
// 并存入 SQLite，schema 与 clouds.sqlite 完全兼容（chunks 表，vector BLOB = f32 native endian）
//
// 用法示例：
//   cargo run --release --bin extract_tokens -- \
//     --mode corpus \
//     --input /path/to/corpus.jsonl \
//     --id-map /path/to/id_map.json \
//     --output token_clouds.sqlite
//
//   cargo run --release --bin extract_tokens -- \
//     --mode query \
//     --input /path/to/query_vectors.jsonl \
//     --output query_token_clouds.sqlite

use anyhow::{Context, Result, bail};
use clap::Parser;
use rusqlite::Connection;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

// ─── 命令行参数 ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "extract_tokens", about = "从 llama.cpp 提取 token-level embeddings 存入 SQLite")]
struct Args {
    /// 模式: corpus 或 query
    #[arg(long, default_value = "corpus")]
    mode: String,

    /// llama.cpp embedding API 地址
    #[arg(long, default_value = "http://localhost:8081/embedding")]
    api_url: String,

    /// 输入 JSONL 文件路径（corpus.jsonl 或 query_vectors.jsonl）
    #[arg(long)]
    input: PathBuf,

    /// id_map.json 路径（corpus 模式必须提供）
    #[arg(long)]
    id_map: Option<PathBuf>,

    /// 输出 SQLite 数据库路径
    #[arg(long)]
    output: PathBuf,

    /// 文本最大字符数截断，防止超出 llama.cpp context 上限
    #[arg(long, default_value = "6000")]
    max_chars: usize,
}

// ─── JSONL 行结构 ────────────────────────────────────────────────────────

/// corpus.jsonl 和 query_vectors.jsonl 共用结构
#[derive(Deserialize)]
struct DocRecord {
    _id: String,
    #[serde(default)]
    title: String,
    text: String,
}

// ─── SQLite 初始化 ───────────────────────────────────────────────────────

/// 创建数据库，schema 与 clouds.sqlite 完全一致
fn create_db(path: &std::path::Path) -> Result<Connection> {
    // 如果输出文件已存在则删除，保证幂等
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=-64000;
         CREATE TABLE IF NOT EXISTS chunks (
             id INTEGER PRIMARY KEY,
             file_id INTEGER NOT NULL,
             chunk_text TEXT,
             vector BLOB
         );
         CREATE INDEX IF NOT EXISTS idx_file_id ON chunks(file_id);",
    )?;
    Ok(conn)
}

// ─── 调用 llama.cpp API ─────────────────────────────────────────────────

/// 向 llama.cpp /embedding 端点发送文本，返回 per-token 的 f32 向量列表
/// llama.cpp --pooling none 返回格式：
///   [{"embedding": [[f64; 4096]; n_tokens], "index": 0}]
/// 当只有 1 个 token 时可能返回 flat [f64; 4096]，需兼容处理
fn extract_tokens(api_url: &str, text: &str) -> Result<Vec<Vec<f32>>> {
    // 如果文本太长，截断到约 1500 token（按字符估算：英文 ~4 chars/token，中文 ~1.5 chars/token）
    let text = if text.len() > 4000 {
        &text[..4000]
    } else {
        text
    };
    let body = serde_json::json!({"content": text});

    let response = match ureq::post(api_url)
        .header("Content-Type", "application/json")
        .send_json(&body)
    {
        Ok(r) => r,
        Err(e) => bail!("HTTP 请求失败: {}", e),
    };

    let body_str = response
        .into_body()
        .with_config()
        .limit(200 * 1024 * 1024)  // 200MB limit（token级响应可达数十MB）
        .read_to_string()
        .context("读取响应 body 失败")?;

    let resp: serde_json::Value = serde_json::from_str(&body_str)
        .with_context(|| format!("JSON 解析失败, body 前200字符: {}", &body_str[..body_str.len().min(200)]))?;

    // 取出 embedding 字段
    let embedding_val = &resp[0]["embedding"];

    let token_vectors: Vec<Vec<f32>> = if let Some(outer) = embedding_val.as_array() {
        if outer.is_empty() {
            bail!("embedding 返回空数组");
        }
        // 判断第一个元素是数组（嵌套情况）还是数字（flat 情况）
        if outer[0].is_array() {
            // 嵌套 [[f64; dim]; n_tokens]
            outer
                .iter()
                .map(|tok_vec| {
                    tok_vec
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                        .collect()
                })
                .collect()
        } else {
            // flat [f64; dim] — 只有 1 个 token
            let single: Vec<f32> = outer
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            vec![single]
        }
    } else {
        bail!("embedding 字段格式异常: {:?}", embedding_val);
    };

    Ok(token_vectors)
}

/// 将 f32 向量转为 native endian 字节（与 clouds.sqlite 格式一致）
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for &val in v {
        buf.extend_from_slice(&val.to_ne_bytes());
    }
    buf
}

// ─── 主流程 ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // 验证模式参数
    if args.mode != "corpus" && args.mode != "query" {
        bail!("--mode 必须是 corpus 或 query");
    }

    // corpus 模式必须有 id_map
    if args.mode == "corpus" && args.id_map.is_none() {
        bail!("corpus 模式必须提供 --id-map 参数");
    }

    // 加载 id_map（corpus 模式）
    let id_map: HashMap<String, i64> = if let Some(ref map_path) = args.id_map {
        let f = File::open(map_path).context("打开 id_map.json 失败")?;
        serde_json::from_reader(BufReader::new(f)).context("解析 id_map.json 失败")?
    } else {
        HashMap::new()
    };

    // 读取 JSONL 文件
    let input_file = File::open(&args.input)
        .with_context(|| format!("打开输入文件失败: {:?}", args.input))?;
    let reader = BufReader::new(input_file);
    let mut records: Vec<DocRecord> = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("读取第 {} 行失败", line_num + 1))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<DocRecord>(&line) {
            Ok(rec) => records.push(rec),
            Err(e) => {
                eprintln!("[WARN] 跳过第 {} 行，JSON 解析错误: {}", line_num + 1, e);
            }
        }
    }

    let total = records.len();
    eprintln!("[INFO] 共加载 {} 条记录，模式: {}", total, args.mode);

    // 创建输出数据库
    let mut conn = create_db(&args.output)?;
    let mut global_id: i64 = 0;
    let mut success_count = 0usize;
    let mut skip_count = 0usize;

    for (idx, rec) in records.iter().enumerate() {
        // 确定 file_id
        let file_id: i64 = if args.mode == "corpus" {
            match id_map.get(&rec._id) {
                Some(&fid) => fid,
                None => {
                    eprintln!("[WARN] 文档 {} 不在 id_map 中，跳过", rec._id);
                    skip_count += 1;
                    continue;
                }
            }
        } else {
            // query 模式：直接用顺序索引作为 file_id
            idx as i64
        };

        // 拼接文本（corpus 模式包含 title）
        let text = if args.mode == "corpus" && !rec.title.is_empty() {
            format!("{} {}", rec.title, rec.text)
        } else {
            rec.text.clone()
        };

        // 截断过长文本
        let text = if text.len() > args.max_chars {
            text[..args.max_chars].to_string()
        } else {
            text
        };

        // 调用 API 提取 token embeddings
        match extract_tokens(&args.api_url, &text) {
            Ok(token_vecs) => {
                // 在一个事务内插入该文档的所有 token 向量
                let tx = conn.transaction()?;
                {
                    let mut stmt = tx.prepare(
                        "INSERT INTO chunks (id, file_id, chunk_text, vector) VALUES (?1, ?2, ?3, ?4)",
                    )?;
                    for (tok_idx, vec) in token_vecs.iter().enumerate() {
                        let blob = vec_to_blob(vec);
                        let chunk_text = format!("token_{}", tok_idx);
                        stmt.execute(rusqlite::params![
                            global_id,
                            file_id,
                            chunk_text,
                            blob,
                        ])?;
                        global_id += 1;
                    }
                }
                tx.commit()?;
                success_count += 1;
            }
            Err(e) => {
                eprintln!("[ERROR] 文档 {} (file_id={}) 提取失败: {}", rec._id, file_id, e);
                skip_count += 1;
            }
        }

        // 每 50 条打印进度
        if (idx + 1) % 50 == 0 || idx + 1 == total {
            eprintln!(
                "[PROGRESS] {}/{} (成功={}, 跳过={}), 当前总 token 行数={}",
                idx + 1,
                total,
                success_count,
                skip_count,
                global_id
            );
        }
    }

    eprintln!(
        "[DONE] 完成! 成功 {} 文档, 跳过 {} 文档, 共 {} 条 token 向量 => {:?}",
        success_count, skip_count, global_id, args.output
    );

    Ok(())
}
