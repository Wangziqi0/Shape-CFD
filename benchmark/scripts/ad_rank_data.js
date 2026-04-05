#!/usr/bin/env node
/**
 * ADRankData - AD-Rank 数据接口层
 * 
 * 为 AD-Rank 核心求解器提供输入数据：
 *   1. 加载 USearch HNSW 索引和元数据
 *   2. 将 query 文本转为 4096 维向量
 *   3. HNSW 粗筛候选 → 重新 embed 获取候选文档向量
 * 
 * 依赖：
 *   - usearch (已安装)
 *   - vectorize_engine.js (已存在)
 *   - knowledge_base/vectors/law.usearch + metadata.json
 */

'use strict';

const usearch = require('usearch');
const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const { VectorizeEngine, loadApiKey } = require('./vectorize_engine');
const { DataEngine } = require('./data_engine');

// ========== 路径常量 ==========
const VECTORS_DIR = path.join(__dirname, 'knowledge_base', 'vectors');
const INDEX_FILE = path.join(VECTORS_DIR, 'law.usearch');
const META_FILE = path.join(VECTORS_DIR, 'metadata.json');

// ========== Embedding API 配置 ==========
const EMBED_API_URL = 'http://192.168.31.22:3000/v1/embeddings';
const EMBED_MODEL = 'Qwen3-Embedding-8B';
const DIMENSIONS = 4096;

class ADRankData {
  constructor() {
    this.index = null;
    this.metadata = null;
    this._idMap = null;   // id → arrayIndex 映射（HNSW key = metadata.id，非数组索引）
    this.engine = null;
    this._apiKey = null;
    this._initialized = false;
    this._dataEngine = null;  // SQLite 向量缓存层
  }

  // ========== 1. 初始化 ==========

  /**
   * 加载 USearch 索引和元数据
   * 复用 search_engine.js 相同的初始化模式
   */
  async initialize() {
    if (this._initialized) return;

    // 检查文件是否存在
    if (!fs.existsSync(INDEX_FILE)) {
      throw new Error(`USearch 索引文件不存在: ${INDEX_FILE}`);
    }
    if (!fs.existsSync(META_FILE)) {
      throw new Error(`元数据文件不存在: ${META_FILE}`);
    }

    // 加载 USearch 索引
    this.index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: DIMENSIONS });
    this.index.load(INDEX_FILE);

    // 加载元数据
    this.metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));

    // 构建 id → arrayIndex 映射
    // HNSW 索引的 key 对应 metadata 的 id 字段（不连续），而非数组索引
    this._idMap = new Map();
    for (let i = 0; i < this.metadata.length; i++) {
      this._idMap.set(this.metadata[i].id, i);
    }

    // 初始化 VectorizeEngine
    this._apiKey = loadApiKey();
    if (this._apiKey) {
      this.engine = new VectorizeEngine(this._apiKey);
    }

    // 初始化 DataEngine（SQLite 向量缓存）
    try {
      this._dataEngine = new DataEngine();
      this._dataEngine.initialize();
      const cachedCount = this._dataEngine.getDocVectorCount();
      console.log(`[ADRankData] 向量缓存: ${cachedCount} 条已缓存`);
    } catch (e) {
      console.warn(`[ADRankData] DataEngine 初始化失败，将回退到全量 re-embed: ${e.message}`);
      this._dataEngine = null;
    }

    this._initialized = true;
    console.log(`[ADRankData] 加载完成: ${this.index.size()} 条向量, ${this.metadata.length} 条元数据, idMap ${this._idMap.size} 项`);
  }

  // ========== 2. Query Embedding ==========

  /**
   * 将用户查询文本转为向量
   * @param {string} queryText - 用户查询
   * @returns {Promise<Float32Array>} 4096维向量
   */
  async embedQuery(queryText) {
    if (!queryText || typeof queryText !== 'string') {
      throw new Error('queryText 必须是非空字符串');
    }

    // 优先使用 VectorizeEngine
    if (this.engine) {
      const vector = await this.engine.embed(queryText);
      // 确保返回 Float32Array
      if (vector instanceof Float32Array) return vector;
      return new Float32Array(vector);
    }

    // Fallback: 直接调用 API
    const results = await this._batchEmbed([queryText]);
    if (!results[0]) throw new Error('Query embedding 失败');
    return results[0];
  }

  // ========== 3. 获取候选 ==========

  /**
   * HNSW 粗筛 + 获取候选文档向量
   * 
   * 流程：
   *   1. HNSW 搜索 top-preFilterK 个候选
   *   2. 从 metadata 获取候选文档的文本内容
   *   3. 重新调用 embedding API 获取这些文档的 4096 维向量
   *   4. 返回向量和元数据
   * 
   * @param {Float32Array} queryVector - query 向量
   * @param {number} [preFilterK=30] - 粗筛数量
   * @returns {Promise<{vectors: Float32Array[], metadata: Object[], distances: number[], cacheStats: {hit: number, miss: number}}>}
   */
  async getCandidates(queryVector, preFilterK = 30) {
    if (!this._initialized) {
      throw new Error('请先调用 initialize()');
    }

    // 1. HNSW 粗筛
    const results = this.index.search(queryVector, preFilterK);
    const keys = Array.from(results.keys).map(Number);
    const distances = Array.from(results.distances);

    // 2. 获取元数据和文本
    const metas = [];
    const validDistances = [];
    const validKeys = [];    // HNSW key （= metadata.id）

    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const arrayIdx = this._idMap.get(k);
      const meta = arrayIdx !== undefined ? this.metadata[arrayIdx] : null;
      if (meta && meta.content) {
        metas.push(meta);
        validDistances.push(distances[i]);
        validKeys.push(k);
      }
    }

    if (validKeys.length === 0) {
      console.warn('[ADRankData] 警告: 粗筛后无有效候选文档');
      return { vectors: [], metadata: [], distances: [], cacheStats: { hit: 0, miss: 0 } };
    }

    // 3. 向量缓存层: 先查 SQLite, 未命中的才 re-embed
    let cacheHit = 0, cacheMiss = 0;
    const vectors = new Array(validKeys.length).fill(null);
    const needEmbedIndices = [];

    if (this._dataEngine) {
      const cached = this._dataEngine.getDocVectors(validKeys);
      for (let i = 0; i < validKeys.length; i++) {
        const cv = cached.get(validKeys[i]);
        if (cv) {
          vectors[i] = cv;
          cacheHit++;
        } else {
          needEmbedIndices.push(i);
          cacheMiss++;
        }
      }
    } else {
      // 无缓存层，全部需要 re-embed
      for (let i = 0; i < validKeys.length; i++) needEmbedIndices.push(i);
      cacheMiss = validKeys.length;
    }

    // 4. 批量 re-embed 缓存未命中的文档
    if (needEmbedIndices.length > 0) {
      const texts = needEmbedIndices.map(i => metas[i].content);
      let embeddedVectors;

      if (this.engine) {
        embeddedVectors = await this.engine.embedBatch(texts);
      } else {
        embeddedVectors = await this._batchEmbed(texts);
      }

      const newEntries = [];
      for (let j = 0; j < needEmbedIndices.length; j++) {
        const idx = needEmbedIndices[j];
        let v = embeddedVectors[j];
        if (!v) continue;
        if (!(v instanceof Float32Array)) v = new Float32Array(v);
        vectors[idx] = v;
        newEntries.push({ chunkId: validKeys[idx], vector: v });
      }

      // 写入缓存
      if (this._dataEngine && newEntries.length > 0) {
        try {
          this._dataEngine.putDocVectors(newEntries);
          console.log(`[ADRankData] 缓存写入 ${newEntries.length} 条新向量`);
        } catch (e) {
          console.warn(`[ADRankData] 缓存写入失败: ${e.message}`);
        }
      }
    }

    // 5. 过滤失败的向量
    const finalVectors = [];
    const finalMetas = [];
    const finalDistances = [];
    for (let i = 0; i < vectors.length; i++) {
      if (vectors[i]) {
        finalVectors.push(vectors[i]);
        finalMetas.push(metas[i]);
        finalDistances.push(validDistances[i]);
      }
    }

    return {
      vectors: finalVectors,
      metadata: finalMetas,
      distances: finalDistances,
      cacheStats: { hit: cacheHit, miss: cacheMiss },
    };
  }

  // ========== 4. 批量 Embed (Fallback) ==========

  /**
   * 直接调用 Embedding API 的批量方法
   * 当 VectorizeEngine 不可用时使用
   * 
   * @param {string[]} texts - 文本数组
   * @param {number} [batchSize=8] - 每批文本数
   * @returns {Promise<Float32Array[]>}
   */
  async _batchEmbed(texts, batchSize = 8) {
    const apiKey = this._apiKey || loadApiKey() || '';
    const results = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);

      try {
        const json = await this._callEmbedAPI(batch, apiKey);
        const sorted = json.data.sort((a, b) => a.index - b.index);
        for (const item of sorted) {
          results.push(new Float32Array(item.embedding));
        }
      } catch (err) {
        console.error(`[ADRankData] _batchEmbed 批次 ${Math.floor(i / batchSize) + 1} 失败:`, err.message);
        // 失败批次填 null
        for (let j = 0; j < batch.length; j++) {
          results.push(null);
        }
      }

      // 批次间微延迟防止过载
      if (i + batchSize < texts.length) {
        await new Promise(r => setTimeout(r, 50));
      }
    }

    return results.filter(Boolean);
  }

  /**
   * 底层 HTTP 调用 Embedding API
   * @private
   */
  _callEmbedAPI(texts, apiKey) {
    return new Promise((resolve, reject) => {
      const body = JSON.stringify({
        model: EMBED_MODEL,
        input: texts,
        encoding_format: 'float',
        dimensions: DIMENSIONS,
      });

      const url = new URL(EMBED_API_URL);
      const isHttps = url.protocol === 'https:';
      const lib = isHttps ? https : http;

      const options = {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'Content-Length': Buffer.byteLength(body),
        },
      };

      const req = lib.request(options, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          if (res.statusCode !== 200) {
            reject(new Error(`Embedding API ${res.statusCode}: ${data.substring(0, 300)}`));
            return;
          }
          try {
            const json = JSON.parse(data);
            if (json.error) {
              reject(new Error(`Embedding API Error: ${json.error.message || JSON.stringify(json.error)}`));
              return;
            }
            if (!json.data || !Array.isArray(json.data)) {
              reject(new Error(`Embedding API 返回格式异常: 缺少 data 数组`));
              return;
            }
            resolve(json);
          } catch (e) {
            reject(new Error(`JSON 解析失败: ${e.message}`));
          }
        });
      });

      req.on('error', (e) => reject(new Error(`网络错误: ${e.message}`)));
      req.setTimeout(30000, () => { req.destroy(); reject(new Error('Embedding API 超时 30s')); });
      req.write(body);
      req.end();
    });
  }

  // ========== 工具方法 ==========

  /**
   * 获取索引统计信息
   */
  getStats() {
    return {
      indexSize: this.index ? this.index.size() : 0,
      metadataCount: this.metadata ? this.metadata.length : 0,
      dimensions: DIMENSIONS,
      initialized: this._initialized,
    };
  }
}

// ========== 自测代码 ==========

if (require.main === module) {
  (async () => {
    console.log('═'.repeat(60));
    console.log('  AD-Rank 数据接口层 — 自测');
    console.log('═'.repeat(60));

    try {
      const data = new ADRankData();
      await data.initialize();

      // 1. 测试 embedQuery
      console.log('\n📌 测试 embedQuery...');
      const qVec = await data.embedQuery('劳动合同解除赔偿标准');
      console.log(`  Query 向量维度: ${qVec.length}`);
      console.log(`  前5个分量: [${Array.from(qVec.slice(0, 5)).map(v => v.toFixed(6)).join(', ')}]`);

      // 2. 测试 getCandidates
      console.log('\n📌 测试 getCandidates (top-10)...');
      const { vectors, metadata, distances } = await data.getCandidates(qVec, 10);
      console.log(`  候选: ${vectors.length} 个文档`);
      metadata.forEach((m, i) =>
        console.log(`  [${i}] ${m.law} ${m.article || ''} (dist=${distances[i].toFixed(4)})`)
      );

      // 3. 尝试调用 AD-Rank（如果已存在）
      try {
        const { adRank } = require('./ad_rank');
        console.log('\n📌 调用 AD-Rank 排序...');
        const result = adRank(qVec, vectors, 5);
        console.log('AD-Rank Top-5:');
        result.rankings.forEach((r, i) =>
          console.log(`  #${i + 1} ${metadata[r.index].law} ${metadata[r.index].article || ''} score=${r.score.toFixed(4)} ${r.topology}`)
        );
      } catch (e) {
        console.log(`\n⏭️  ad_rank.js 尚未就绪 (Agent1 产出): ${e.message}`);
      }

      // 4. 统计
      console.log('\n📊 统计:', JSON.stringify(data.getStats()));
      console.log('\n✅ 自测完成');

    } catch (err) {
      console.error('\n❌ 自测失败:', err.message);
      console.error(err.stack);
      process.exit(1);
    }
  })();
}

module.exports = { ADRankData };
