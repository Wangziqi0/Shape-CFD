#!/usr/bin/env node
'use strict';

/**
 * Shape CFD 预热脚本 (Agent 12)
 *
 * 离线遍历 23,701 个 chunk，拆句 → 批量 embed → 随机投影 4096→128 → 存 SQLite
 *
 * 运行方式：
 *   node ad_rank_shape_cache.js
 *   node ad_rank_shape_cache.js --resume   # 续跑（跳过已缓存的 chunk）
 *
 * 依赖：
 *   - 192.168.31.22:3000 Embedding API (Qwen3-Embedding-8B)
 *   - data_engine.js (SQLite)
 *   - knowledge_base/vectors/metadata.json
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const { DataEngine } = require('./data_engine');
const { loadApiKey } = require('./vectorize_engine');

// ========== 配置 ==========
const META_FILE = path.join(__dirname, 'knowledge_base', 'vectors', 'metadata.json');
const EMBED_API_URL = 'http://192.168.31.22:3000/v1/embeddings';
const EMBED_MODEL = 'Qwen3-Embedding-8B';
const FULL_DIM = 4096;
const PROJ_DIM = 128;
const BATCH_SIZE = 32;       // 每批 embed 句子数
const API_DELAY_MS = 30;     // 批次间延迟 (ms)
const SEED = 42;             // 随机投影矩阵种子

// ========== 固定随机投影矩阵 ==========

/**
 * 用 seed 生成固定的随机投影矩阵
 * Johnson-Lindenstrauss: 128 维足以保持距离结构（误差 < 10%）
 *
 * @param {number} fromDim - 源维度 (4096)
 * @param {number} toDim   - 目标维度 (128)
 * @param {number} seed
 * @returns {Float32Array} 平铺 toDim × fromDim 矩阵
 */
function generateProjectionMatrix(fromDim, toDim, seed) {
  const matrix = new Float32Array(toDim * fromDim);
  // 简单的 Mulberry32 PRNG
  let s = seed | 0;
  function rand() {
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }

  // 正态分布: Box-Muller
  const scale = 1.0 / Math.sqrt(toDim);
  for (let i = 0; i < toDim * fromDim; i += 2) {
    const u1 = rand() || 1e-10;
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    matrix[i] = r * Math.cos(theta) * scale;
    if (i + 1 < toDim * fromDim) {
      matrix[i + 1] = r * Math.sin(theta) * scale;
    }
  }
  return matrix;
}

/**
 * 将 4096 维向量投影到 128 维
 * @param {Float32Array} vec - 4096 维
 * @param {Float32Array} projMatrix - 128 × 4096 平铺矩阵
 * @returns {Float32Array} 128 维
 */
function projectVector(vec, projMatrix) {
  const result = new Float32Array(PROJ_DIM);
  for (let r = 0; r < PROJ_DIM; r++) {
    let sum = 0;
    const offset = r * FULL_DIM;
    // 8 路展开
    let d = 0;
    for (; d + 7 < FULL_DIM; d += 8) {
      sum += projMatrix[offset + d] * vec[d]
           + projMatrix[offset + d + 1] * vec[d + 1]
           + projMatrix[offset + d + 2] * vec[d + 2]
           + projMatrix[offset + d + 3] * vec[d + 3]
           + projMatrix[offset + d + 4] * vec[d + 4]
           + projMatrix[offset + d + 5] * vec[d + 5]
           + projMatrix[offset + d + 6] * vec[d + 6]
           + projMatrix[offset + d + 7] * vec[d + 7];
    }
    for (; d < FULL_DIM; d++) sum += projMatrix[offset + d] * vec[d];
    result[r] = sum;
  }
  return result;
}

// ========== Embedding API 调用 ==========

function callEmbedAPI(texts, apiKey) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: EMBED_MODEL,
      input: texts,
      encoding_format: 'float',
      dimensions: FULL_DIM,
    });

    const url = new URL(EMBED_API_URL);
    const headers = {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body),
    };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    const options = {
      hostname: url.hostname,
      port: url.port || 80,
      path: url.pathname,
      method: 'POST',
      headers,
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        if (res.statusCode !== 200) {
          reject(new Error(`API ${res.statusCode}: ${data.substring(0, 200)}`));
          return;
        }
        try {
          const json = JSON.parse(data);
          if (json.error) {
            reject(new Error(`API Error: ${json.error.message || JSON.stringify(json.error)}`));
            return;
          }
          const sorted = json.data.sort((a, b) => a.index - b.index);
          resolve(sorted.map(item => new Float32Array(item.embedding)));
        } catch (e) {
          reject(new Error(`JSON 解析失败: ${e.message}`));
        }
      });
    });

    req.on('error', e => reject(new Error(`网络错误: ${e.message}`)));
    req.setTimeout(60000, () => { req.destroy(); reject(new Error('API 超时 60s')); });
    req.write(body);
    req.end();
  });
}

// ========== 拆句 ==========

function splitSentences(text) {
  if (!text) return [''];
  const sentences = text.split(/[。；;]/).filter(s => s.trim().length > 5);
  return sentences.length > 0 ? sentences.map(s => s.trim()) : [text.trim() || '空'];
}

// ========== 主流程 ==========

(async () => {
  const resume = process.argv.includes('--resume');
  
  console.log('═'.repeat(60));
  console.log('  Shape CFD 预热脚本 (Agent 12)');
  console.log(`  投影: ${FULL_DIM} → ${PROJ_DIM} 维, seed=${SEED}`);
  console.log(`  模式: ${resume ? '续跑 (跳过已缓存)' : '全新预热'}`);
  console.log('═'.repeat(60));

  // 1. 加载元数据
  console.log('\n📦 加载元数据...');
  const metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
  console.log(`  共 ${metadata.length} 个 chunk`);

  // 2. 加载 API Key
  const apiKey = loadApiKey();
  console.log(`  API Key: ${apiKey ? apiKey.substring(0, 8) + '...' : '❌ 未找到'}`);
  const dataEngine = new DataEngine();
  dataEngine.initialize();

  // 3. 生成投影矩阵
  console.log(`\n🔢 生成随机投影矩阵 (${PROJ_DIM}×${FULL_DIM})...`);
  const projMatrix = generateProjectionMatrix(FULL_DIM, PROJ_DIM, SEED);
  console.log(`  矩阵大小: ${(projMatrix.byteLength / 1024 / 1024).toFixed(1)} MB`);

  // 4. 确定需要处理的 chunk
  let cachedChunks = new Set();
  if (resume) {
    const rows = dataEngine.db.prepare(
      'SELECT DISTINCT chunk_id FROM sentence_vectors'
    ).all();
    cachedChunks = new Set(rows.map(r => r.chunk_id));
    console.log(`\n📋 已缓存 ${cachedChunks.size} 个 chunk，跳过`);
  }

  const chunksToProcess = metadata.filter(m => m.content && !cachedChunks.has(m.id));
  console.log(`\n📋 需处理: ${chunksToProcess.length} 个 chunk`);

  // 5. 逐 chunk 拆句并收集
  let totalSentences = 0;
  let totalChunks = 0;
  let failedBatches = 0;
  const startTime = Date.now();

  // 按批次处理：每 BATCH_SIZE 个句子调一次 API
  let sentBuf = [];     // { chunkId, sentIdx, text }
  let entryBuf = [];    // 写入用的缓冲

  async function flushBatch() {
    if (sentBuf.length === 0) return;
    const texts = sentBuf.map(s => s.text);
    
    try {
      const vectors = await callEmbedAPI(texts, apiKey);
      
      for (let i = 0; i < vectors.length; i++) {
        const vec = vectors[i];
        const proj = projectVector(vec, projMatrix);
        entryBuf.push({
          chunkId: sentBuf[i].chunkId,
          sentIdx: sentBuf[i].sentIdx,
          vector: vec,
          vectorProj: proj,
        });
      }
    } catch (e) {
      console.warn(`  ⚠️ 批次失败: ${e.message}`);
      failedBatches++;
    }
    
    sentBuf = [];
  }

  function flushEntries() {
    if (entryBuf.length === 0) return;
    dataEngine.putSentenceVectors(entryBuf);
    entryBuf = [];
  }

  for (let ci = 0; ci < chunksToProcess.length; ci++) {
    const chunk = chunksToProcess[ci];
    const sentences = splitSentences(chunk.content);

    for (let si = 0; si < sentences.length; si++) {
      sentBuf.push({ chunkId: chunk.id, sentIdx: si, text: sentences[si] });

      if (sentBuf.length >= BATCH_SIZE) {
        await flushBatch();
        
        // 每 200 个句子写一次 SQLite（减少事务开销）
        if (entryBuf.length >= 200) {
          flushEntries();
        }

        // 进度报告
        totalSentences += BATCH_SIZE;
        if (totalSentences % 1000 < BATCH_SIZE) {
          const elapsed = (Date.now() - startTime) / 1000;
          const rate = totalSentences / elapsed;
          const remaining = ((chunksToProcess.length - ci) * 3 / rate).toFixed(0);
          console.log(`  [${ci + 1}/${chunksToProcess.length}] ${totalSentences} 句, ${rate.toFixed(0)} 句/s, 预计剩余 ${remaining}s`);
        }

        // 批次间延迟
        if (API_DELAY_MS > 0) {
          await new Promise(r => setTimeout(r, API_DELAY_MS));
        }
      }
    }

    totalChunks++;
  }

  // 处理剩余
  await flushBatch();
  flushEntries();

  const elapsed = (Date.now() - startTime) / 1000;

  // 6. 报告
  console.log('\n' + '═'.repeat(60));
  console.log('  预热完成');
  console.log('═'.repeat(60));
  console.log(`  处理 chunk: ${totalChunks}`);
  console.log(`  处理句子:   ~${totalSentences}`);
  console.log(`  失败批次:   ${failedBatches}`);
  console.log(`  耗时:       ${elapsed.toFixed(1)}s (${(elapsed / 60).toFixed(1)} min)`);
  
  const sentCount = dataEngine.getSentenceVectorCount();
  const chunkCount = dataEngine.getSentenceVectorChunkCount();
  console.log(`  SQLite 句子向量: ${sentCount} 条 / ${chunkCount} 个 chunk`);

  // 验证投影质量
  console.log('\n📐 投影质量验证 (随机抽样):');
  const sample = dataEngine.getSentenceVectors([metadata[0].id, metadata[100].id], true);
  if (sample.size >= 2) {
    const ids = [...sample.keys()];
    const a_full = sample.get(ids[0]).vectors;
    const b_full = sample.get(ids[1]).vectors;
    const a_proj = sample.get(ids[0]).projVectors;
    const b_proj = sample.get(ids[1]).projVectors;

    if (a_full && b_full && a_full.length > 0 && b_full.length > 0) {
      // 简单距离验证
      const cosDist = (x, y) => {
        let dot = 0, nx = 0, ny = 0;
        for (let i = 0; i < x.length; i++) {
          dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i];
        }
        return 1 - dot / (Math.sqrt(nx) * Math.sqrt(ny) + 1e-8);
      };

      const distFull = cosDist(a_full[0], b_full[0]);
      const distProj = cosDist(a_proj[0], b_proj[0]);
      console.log(`  4096 维距离: ${distFull.toFixed(4)}`);
      console.log(`  128 维距离:  ${distProj.toFixed(4)}`);
      console.log(`  相对误差:    ${(Math.abs(distFull - distProj) / (distFull + 1e-8) * 100).toFixed(1)}%`);
    }
  }

  dataEngine.close();
  console.log('\n✅ 完成');
})();
