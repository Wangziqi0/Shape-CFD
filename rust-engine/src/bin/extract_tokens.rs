// extract_tokens.rs — 高性能 token-level embedding 提取工具 v2
//
// 优化点：
// 1. rayon work-stealing 并发（无气泡，GPU 始终满载）
// 2. channel 异步写入（提取和写入流水线化）
// 3. 断点续传（跳过已完成的 file_id）
// 4. 批量事务写入（每 N 个文档一个事务）

use anyhow::{Context, Result, bail};
use clap::Parser;
use rusqlite::Connection;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "extract_tokens", about = "高性能 token-level embedding 提取 v2")]
struct Args {
    #[arg(long, default_value = "corpus")]
    mode: String,

    #[arg(long, default_value = "http://192.168.31.22:8081/embedding")]
    api_url: String,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    id_map: Option<PathBuf>,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value = "6000")]
    max_chars: usize,

    /// rayon 线程池大小（= 并发请求数）
    #[arg(long, default_value = "16")]
    concurrency: usize,

    /// 每多少个文档提交一次 SQLite 事务
    #[arg(long, default_value = "50")]
    tx_batch: usize,
}

#[derive(Deserialize)]
struct DocRecord {
    _id: String,
    #[serde(default)]
    title: String,
    text: String,
}

fn open_db(path: &std::path::Path, resume: bool) -> Result<Connection> {
    if !resume && path.exists() {
        std::fs::remove_file(path)?;
    }
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=-128000;
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

fn get_done_file_ids(conn: &Connection) -> Result<HashSet<i64>> {
    let mut stmt = conn.prepare("SELECT DISTINCT file_id FROM chunks")?;
    let ids: Vec<i64> = stmt.query_map([], |row| row.get(0))?.filter_map(|r| r.ok()).collect();
    Ok(ids.into_iter().collect())
}

fn get_max_chunk_id(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COALESCE(MAX(id), -1) FROM chunks", [], |row| row.get(0))?)
}

fn extract_single(api_url: &str, text: &str) -> Result<Vec<Vec<f32>>> {
    let body = serde_json::json!({"content": text});
    let response = ureq::post(api_url)
        .header("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("HTTP: {}", e))?;

    let body_str = response
        .into_body()
        .with_config()
        .limit(200 * 1024 * 1024)
        .read_to_string()
        .context("read body")?;

    let resp: serde_json::Value = serde_json::from_str(&body_str)
        .with_context(|| format!("JSON: {}", &body_str[..body_str.len().min(100)]))?;

    let emb = &resp[0]["embedding"];
    if let Some(outer) = emb.as_array() {
        if outer.is_empty() { bail!("empty"); }
        if outer[0].is_array() {
            Ok(outer.iter().map(|t| {
                t.as_array().unwrap_or(&vec![]).iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
            }).collect())
        } else {
            Ok(vec![outer.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()])
        }
    } else {
        bail!("bad format");
    }
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for &val in v { buf.extend_from_slice(&val.to_ne_bytes()); }
    buf
}

struct DocResult {
    file_id: i64,
    token_vecs: Vec<Vec<f32>>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.mode != "corpus" && args.mode != "query" {
        bail!("--mode must be corpus or query");
    }
    if args.mode == "corpus" && args.id_map.is_none() {
        bail!("corpus mode requires --id-map");
    }

    let id_map: HashMap<String, i64> = if let Some(ref p) = args.id_map {
        serde_json::from_reader(BufReader::new(File::open(p)?))?
    } else {
        HashMap::new()
    };

    let mut records: Vec<DocRecord> = Vec::new();
    for line in BufReader::new(File::open(&args.input)?).lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        if let Ok(rec) = serde_json::from_str::<DocRecord>(line.trim()) {
            records.push(rec);
        }
    }
    eprintln!("[INFO] {} records, mode={}", records.len(), args.mode);

    let resume = args.output.exists();
    let mut conn = open_db(&args.output, resume)?;
    let done_ids = if resume {
        let ids = get_done_file_ids(&conn)?;
        eprintln!("[INFO] resume: {} done", ids.len());
        ids
    } else {
        HashSet::new()
    };
    let mut global_id = if resume { get_max_chunk_id(&conn)? + 1 } else { 0 };

    // 构建任务
    let mut tasks: Vec<(i64, String)> = Vec::new();
    for (idx, rec) in records.iter().enumerate() {
        let file_id: i64 = if args.mode == "corpus" {
            match id_map.get(&rec._id) { Some(&fid) => fid, None => continue }
        } else {
            idx as i64
        };
        if done_ids.contains(&file_id) { continue; }
        let text = if args.mode == "corpus" && !rec.title.is_empty() {
            format!("{} {}", rec.title, rec.text)
        } else {
            rec.text.clone()
        };
        let text = if text.len() > args.max_chars {
            // 安全截断：找到最近的 UTF-8 字符边界
            let mut end = args.max_chars;
            while end > 0 && !text.is_char_boundary(end) { end -= 1; }
            text[..end].to_string()
        } else { text };
        tasks.push((file_id, text));
    }
    let total = tasks.len();
    eprintln!("[INFO] {} to process (skipped {})", total, done_ids.len());
    if total == 0 { eprintln!("[DONE]"); return Ok(()); }

    // 设置 rayon 线程池
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.concurrency)
        .build_global()
        .ok(); // 可能已经初始化过

    // channel: 提取线程 → 写入线程
    let (tx, rx) = mpsc::sync_channel::<DocResult>(args.concurrency * 4);

    let success = AtomicUsize::new(0);
    let fail = AtomicUsize::new(0);
    let tok_count = AtomicUsize::new(0);
    let start = Instant::now();
    let api_url = args.api_url.clone();
    let tx_batch = args.tx_batch;

    // 写入线程（单线程，顺序写 SQLite）
    let writer = std::thread::spawn(move || -> Result<()> {
        let mut buf: Vec<DocResult> = Vec::with_capacity(tx_batch);

        let flush = |conn: &mut Connection, buf: &mut Vec<DocResult>, gid: &mut i64| -> Result<()> {
            if buf.is_empty() { return Ok(()); }
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO chunks (id, file_id, chunk_text, vector) VALUES (?1, ?2, ?3, ?4)"
                )?;
                for doc in buf.iter() {
                    for (i, vec) in doc.token_vecs.iter().enumerate() {
                        stmt.execute(rusqlite::params![*gid, doc.file_id, format!("t{}", i), vec_to_blob(vec)])?;
                        *gid += 1;
                    }
                }
            }
            tx.commit()?;
            buf.clear();
            Ok(())
        };

        for doc in rx {
            buf.push(doc);
            if buf.len() >= tx_batch {
                flush(&mut conn, &mut buf, &mut global_id)?;
            }
        }
        flush(&mut conn, &mut buf, &mut global_id)?;
        Ok(())
    });

    // rayon 并行提取（work-stealing，无气泡）
    use rayon::prelude::*;
    tasks.par_iter().for_each(|(file_id, text)| {
        match extract_single(&api_url, text) {
            Ok(vecs) => {
                let n = vecs.len();
                let _ = tx.send(DocResult { file_id: *file_id, token_vecs: vecs });
                success.fetch_add(1, Ordering::Relaxed);
                tok_count.fetch_add(n, Ordering::Relaxed);
            }
            Err(e) => {
                eprintln!("[ERR] fid={}: {}", file_id, e);
                fail.fetch_add(1, Ordering::Relaxed);
            }
        }

        let s = success.load(Ordering::Relaxed) + fail.load(Ordering::Relaxed);
        if s % 100 == 0 {
            let el = start.elapsed().as_secs_f64();
            let rate = s as f64 / el;
            let eta = if rate > 0.0 { (total - s) as f64 / rate / 60.0 } else { 0.0 };
            let tc = tok_count.load(Ordering::Relaxed);
            eprintln!("[PROG] {}/{} ({:.1}/s ETA {:.0}m) ok={} fail={} toks={}",
                s, total, rate, eta,
                success.load(Ordering::Relaxed),
                fail.load(Ordering::Relaxed), tc);
        }
    });

    // 关闭 channel，等写入完成
    drop(tx);
    writer.join().unwrap()?;

    let el = start.elapsed().as_secs_f64();
    let sc = success.load(Ordering::Relaxed);
    let fc = fail.load(Ordering::Relaxed);
    let tc = tok_count.load(Ordering::Relaxed);
    eprintln!("[DONE] ok={} fail={} toks={} {:.1}s ({:.1}/s) => {:?}",
        sc, fc, tc, el, sc as f64 / el, args.output);

    Ok(())
}
