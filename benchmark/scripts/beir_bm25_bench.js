#!/usr/bin/env node
'use strict';
/**
 * beir_bm25_bench.js — BM25 粗筛 + 分级管线 Benchmark (NFCorpus)
 *
 * 使用 SQLite FTS5 实现 BM25 索引，测量：
 *   1. BM25 recall@K (K=100,200,500,1000,2000)
 *   2. BM25 单独 NDCG@10
 *   3. BM25 ∪ cosine 混合召回 recall
 *   4. 分级管线: BM25 粗筛 → cosine 精排 → NDCG@10
 *   5. 分级管线: BM25 ∪ cosine 粗筛 → cosine 精排 → NDCG@10
 *   6. 如果 LawVexus 可用: BM25 粗筛 → Shape-CFD 精排
 *
 * 用法：
 *   node beir_bm25_bench.js              # 完整 benchmark
 *   MAX_Q=10 node beir_bm25_bench.js     # 快速测试前 10 个 query
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ═══════════════════════════════════════════════════════════════
// 配置
// ═══════════════════════════════════════════════════════════════
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const BM25_DB_PATH = path.join(DATA_DIR, 'bm25.sqlite');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const K = 10;       // NDCG@K
const TOP_N = 55;   // Shape-CFD 精排候选数

// BM25 recall 测量的 K 值
const RECALL_KS = [100, 200, 500, 1000, 2000];
// 混合召回的总候选数
const HYBRID_KS = [200, 500];

// ═══════════════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════════════

/** 加载 JSONL 文件 */
function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const arr = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(fp),
      crlfDelay: Infinity
    });
    rl.on('line', line => {
      if (line.trim()) {
        try { arr.push(JSON.parse(line)); } catch (_) {}
      }
    });
    rl.on('close', () => resolve(arr));
    rl.on('error', reject);
  });
}

/** 余弦相似度 */
function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

/** NDCG@K 计算 */
function computeNDCG(ranked, qrel, k = 10) {
  let dcg = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++) {
    const rel = qrel[ranked[i]] || 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  const idealRels = Object.values(qrel).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(idealRels.length, k); i++) {
    idcg += (Math.pow(2, idealRels[i]) - 1) / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

/** recall 计算：BM25 候选集中包含了多少相关文档 */
function measureRecall(retrievedIds, qrel, topK) {
  const retrieved = new Set(retrievedIds.slice(0, topK));
  const relevant = Object.keys(qrel).filter(d => qrel[d] > 0);
  const hits = relevant.filter(d => retrieved.has(d));
  return relevant.length > 0 ? hits.length / relevant.length : 0;
}

/** 格式化时长 */
function fmtDur(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** 百分比变化字符串 */
function pctChange(val, base) {
  if (base === 0) return 'N/A';
  const pct = (val - base) / base * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
}

/** 将分数列表归一化到 [0,1] (min-max) */
function normalizeScores(scoreMap) {
  const vals = Object.values(scoreMap);
  if (vals.length === 0) return {};
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const range = mx - mn || 1e-8;
  const out = {};
  for (const [k, v] of Object.entries(scoreMap)) {
    out[k] = (v - mn) / range;
  }
  return out;
}

/**
 * 转义 FTS5 查询：按空格分词，每个词用双引号包裹，用 OR 连接
 * 去除特殊字符避免 FTS5 语法错误
 */
function escapeFts5Query(query) {
  // 移除 FTS5 特殊字符，只保留字母数字和空格
  const cleaned = query.replace(/[^\w\s]/g, ' ');
  const tokens = cleaned.split(/\s+/).filter(t => t.length > 0);
  if (tokens.length === 0) return null;
  // 用 OR 连接每个词
  return tokens.map(t => `"${t}"`).join(' OR ');
}

// ═══════════════════════════════════════════════════════════════
// BM25 索引（SQLite FTS5）
// ═══════════════════════════════════════════════════════════════

/**
 * 构建或加载 BM25 FTS5 索引
 * 返回 { db, search(query, topK) }
 */
function buildBm25Index(corpusDocs) {
  const Database = require('better-sqlite3');

  // 如果已有数据库且行数匹配，直接复用
  let needRebuild = true;
  if (fs.existsSync(BM25_DB_PATH)) {
    try {
      const db = new Database(BM25_DB_PATH);
      const row = db.prepare('SELECT COUNT(*) as cnt FROM corpus_fts').get();
      if (row && row.cnt === corpusDocs.length) {
        console.log(`  BM25 索引已存在 (${row.cnt} docs)，复用`);
        needRebuild = false;
        return wrapDb(db);
      }
      db.close();
    } catch (e) {
      // 表不存在或损坏，重建
    }
  }

  if (needRebuild) {
    // 删除旧文件
    if (fs.existsSync(BM25_DB_PATH)) fs.unlinkSync(BM25_DB_PATH);

    const db = new Database(BM25_DB_PATH);
    db.pragma('journal_mode = WAL');

    // 创建 FTS5 虚拟表
    // 注意: doc_id 不参与全文索引，用 UNINDEXED
    db.exec(`CREATE VIRTUAL TABLE corpus_fts USING fts5(
      doc_id UNINDEXED,
      title,
      content,
      tokenize='porter unicode61'
    )`);

    // 批量插入
    const insert = db.prepare(
      'INSERT INTO corpus_fts (doc_id, title, content) VALUES (?, ?, ?)'
    );
    const tx = db.transaction((docs) => {
      for (const doc of docs) {
        insert.run(doc._id, doc.title || '', doc.text || '');
      }
    });

    console.log(`  构建 BM25 FTS5 索引 (${corpusDocs.length} docs)...`);
    const t0 = Date.now();
    tx(corpusDocs);
    console.log(`  BM25 索引构建完成 (${fmtDur(Date.now() - t0)})`);

    return wrapDb(db);
  }
}

/** 包装 db 对象，提供 search 方法 */
function wrapDb(db) {
  // 预编译查询语句
  const searchStmt = db.prepare(`
    SELECT doc_id, rank as score
    FROM corpus_fts
    WHERE corpus_fts MATCH ?
    ORDER BY rank
    LIMIT ?
  `);

  return {
    db,
    /**
     * BM25 检索
     * @param {string} query - 原始查询文本
     * @param {number} topK - 返回数量
     * @returns {Array<{doc_id: string, score: number}>}
     */
    search(query, topK) {
      const ftsQuery = escapeFts5Query(query);
      if (!ftsQuery) return [];
      try {
        return searchStmt.all(ftsQuery, topK);
      } catch (e) {
        // FTS5 查询失败时返回空
        return [];
      }
    },
    close() {
      db.close();
    }
  };
}

// ═══════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════
async function main() {
  console.log('\n=== BM25 + 分级管线 Benchmark (NFCorpus) ===\n');

  // ── 1. 加载语料库 ──
  process.stdout.write('加载语料库...');
  let t0 = Date.now();
  const corpusDocs = await loadJsonl(path.join(DATA_DIR, 'corpus.jsonl'));
  console.log(` done (${corpusDocs.length} docs, ${fmtDur(Date.now() - t0)})`);

  // ── 2. 加载 query ──
  process.stdout.write('加载 queries...');
  t0 = Date.now();
  const queryDocs = await loadJsonl(path.join(DATA_DIR, 'queries.jsonl'));
  const queryMap = {};
  for (const q of queryDocs) queryMap[q._id] = q.text;
  console.log(` done (${queryDocs.length} queries, ${fmtDur(Date.now() - t0)})`);

  // ── 3. 加载 qrels ──
  const qrels = {};
  const qrelLines = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < qrelLines.length; i++) {
    const parts = qrelLines[i].split('\t');
    if (parts.length < 3) continue;
    const [qi, di, s] = parts;
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  // ── 4. 加载向量（cosine 基线）──
  process.stdout.write('加载向量...');
  t0 = Date.now();
  const corpusVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    corpusVecs[o._id] = new Float32Array(o.vector);
  }
  const allDids = Object.keys(corpusVecs);

  const queryVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl'))) {
    queryVecs[o._id] = new Float32Array(o.vector);
  }
  console.log(` done (${allDids.length} corpus, ${Object.keys(queryVecs).length} queries, ${fmtDur(Date.now() - t0)})`);

  // ── 5. 构建 BM25 索引 ──
  const bm25 = buildBm25Index(corpusDocs);

  // ── 6. 尝试加载 LawVexus ──
  let vexus = null;
  let hasTokenClouds = false;
  let hasTokenTwoStage = false;
  let idMap = null;
  let reverseMap = null;
  let queryIdMap = null;

  try {
    const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
    vexus = new LawVexus('/tmp/beir_bm25_bench');

    // 加载句子级点云
    process.stdout.write('加载句子级点云...');
    t0 = Date.now();
    const cloudInfo = vexus.loadClouds(CLOUDS_DB);
    console.log(` done (${fmtDur(Date.now() - t0)}) ${cloudInfo}`);

    // id_map
    idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
    reverseMap = {};
    for (const [strId, intId] of Object.entries(idMap)) {
      reverseMap[intId] = strId;
    }

    // query ID -> int ID 映射
    queryIdMap = {};
    const qvLines = fs.readFileSync(
      path.join(DATA_DIR, 'query_vectors.jsonl'), 'utf8'
    ).trim().split('\n');
    for (let i = 0; i < qvLines.length; i++) {
      try {
        const obj = JSON.parse(qvLines[i]);
        const qid = obj._id || obj.id;
        if (qid) queryIdMap[qid] = i;
      } catch (e) {}
    }

    // 尝试加载 token 点云
    if (typeof vexus.loadTokenCloudsSqlite === 'function') {
      process.stdout.write('加载 token 点云...');
      t0 = Date.now();
      try {
        const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
        hasTokenClouds = true;
        hasTokenTwoStage = typeof vexus.tokenChamferTwoStage === 'function';
        console.log(` done (${fmtDur(Date.now() - t0)}) ${tokenInfo}`);
      } catch (e) {
        console.log(` FAILED: ${e.message}`);
      }
    }
  } catch (e) {
    console.log(`LawVexus 不可用: ${e.message}`);
    console.log('  将只测试 BM25 recall + cosine 精排管线\n');
  }

  // ── 7. 确定测试 query ──
  let qids = Object.keys(qrels).filter(q => queryVecs[q] && queryMap[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);
  console.log(`\n测试 query 数: ${qids.length}\n`);

  // ═══════════════════════════════════════════════════════════════
  // 8. 跑 Benchmark
  // ═══════════════════════════════════════════════════════════════

  // --- BM25 Recall 累加器 ---
  const recallSums = {};
  for (const rk of RECALL_KS) recallSums[rk] = { sum: 0, count: 0 };

  // --- 混合召回 Recall 累加器 ---
  const hybridRecallSums = {};
  for (const hk of HYBRID_KS) hybridRecallSums[hk] = { sum: 0, count: 0, totalCandidates: 0 };

  // --- 方法 NDCG 累加器 ---
  const methods = {
    cosine:               { sum: 0, count: 0, totalMs: 0 },
    bm25_only:            { sum: 0, count: 0, totalMs: 0 },
    bm25_200_cosine:      { sum: 0, count: 0, totalMs: 0 },
    bm25_500_cosine:      { sum: 0, count: 0, totalMs: 0 },
    hybrid_200_cosine:    { sum: 0, count: 0, totalMs: 0 },
    hybrid_500_cosine:    { sum: 0, count: 0, totalMs: 0 },
    bm25_200_chamfer:     { sum: 0, count: 0, totalMs: 0 },
    bm25_500_chamfer:     { sum: 0, count: 0, totalMs: 0 },
    hybrid_200_chamfer:   { sum: 0, count: 0, totalMs: 0 },
    token_2stage_100:     { sum: 0, count: 0, totalMs: 0 },
    fusion_07:            { sum: 0, count: 0, totalMs: 0 },
  };

  const progressStep = Math.max(1, Math.floor(qids.length / 20));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qText = queryMap[qid];
    const qVec = queryVecs[qid];
    const qrel = qrels[qid];
    const intId = queryIdMap ? queryIdMap[qid] : undefined;

    // ── BM25 检索（取最大 K）──
    const maxK = RECALL_KS[RECALL_KS.length - 1]; // 2000
    const bm25T0 = Date.now();
    const bm25Results = bm25.search(qText, maxK);
    const bm25Ms = Date.now() - bm25T0;
    const bm25DocIds = bm25Results.map(r => r.doc_id);

    // ── BM25 Recall@K ──
    for (const rk of RECALL_KS) {
      const recall = measureRecall(bm25DocIds, qrel, rk);
      recallSums[rk].sum += recall;
      recallSums[rk].count += 1;
    }

    // ── cosine 全排序（用于基线和混合召回）──
    const cosT0 = Date.now();
    const cosScores = allDids.map(did => ({ did, s: cosSim(qVec, corpusVecs[did]) }));
    cosScores.sort((a, b) => b.s - a.s);
    const cosMs = Date.now() - cosT0;

    // cosine NDCG@10 基线
    {
      const ranked = cosScores.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.cosine.sum += ndcg;
      methods.cosine.count += 1;
      methods.cosine.totalMs += cosMs;
    }

    // ── BM25 NDCG@10（直接用 BM25 排序）──
    {
      const ranked = bm25DocIds.slice(0, K);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.bm25_only.sum += ndcg;
      methods.bm25_only.count += 1;
      methods.bm25_only.totalMs += bm25Ms;
    }

    // ── 混合召回 BM25 ∪ cosine recall@K ──
    // 每侧取 K/2
    for (const hk of HYBRID_KS) {
      const half = Math.floor(hk / 2);
      const bm25Set = new Set(bm25DocIds.slice(0, half));
      const cosSet = new Set(cosScores.slice(0, half).map(x => x.did));
      const unionSet = new Set([...bm25Set, ...cosSet]);
      const unionArr = [...unionSet];

      const recall = measureRecall(unionArr, qrel, unionArr.length);
      hybridRecallSums[hk].sum += recall;
      hybridRecallSums[hk].count += 1;
      hybridRecallSums[hk].totalCandidates += unionArr.length;
    }

    // ── 分级管线: BM25 top-K → cosine 精排 → NDCG@10 ──
    // BM25 粗筛，在候选集上用 cosine 重排
    for (const [bk, methodName] of [[200, 'bm25_200_cosine'], [500, 'bm25_500_cosine']]) {
      const t = Date.now();
      const candidateIds = bm25DocIds.slice(0, bk);
      // 在候选集上 cosine 排序
      const scored = candidateIds
        .filter(did => corpusVecs[did])
        .map(did => ({ did, s: cosSim(qVec, corpusVecs[did]) }));
      scored.sort((a, b) => b.s - a.s);
      const ranked = scored.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods[methodName].sum += ndcg;
      methods[methodName].count += 1;
      methods[methodName].totalMs += (Date.now() - t) + bm25Ms;
    }

    // ── 分级管线: (BM25 ∪ cosine) → cosine 精排 → NDCG@10 ──
    for (const [hk, methodName] of [[200, 'hybrid_200_cosine'], [500, 'hybrid_500_cosine']]) {
      const t = Date.now();
      const half = Math.floor(hk / 2);
      const bm25Set = new Set(bm25DocIds.slice(0, half));
      const cosTopIds = cosScores.slice(0, half).map(x => x.did);
      const unionSet = new Set([...bm25Set, ...cosTopIds]);

      // 在并集候选集上 cosine 精排
      const scored = [...unionSet]
        .filter(did => corpusVecs[did])
        .map(did => ({ did, s: cosSim(qVec, corpusVecs[did]) }));
      scored.sort((a, b) => b.s - a.s);
      const ranked = scored.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods[methodName].sum += ndcg;
      methods[methodName].count += 1;
      methods[methodName].totalMs += Date.now() - t;
    }

    // ── Shape-CFD 相关管线（需要 LawVexus）──
    let cfdScoreMap = null;
    let tokenScoreMap = null;

    if (vexus) {
      // shape_cfd_v10 分数（用于 fusion）
      try {
        const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
        const hits = vexus.shapeCfdPipeline(qBuf, K, TOP_N);
        cfdScoreMap = {};
        for (const h of hits) {
          const strId = reverseMap[h.id];
          if (strId) cfdScoreMap[strId] = h.score;
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN shapeCfdPipeline: ${e.message}`);
      }

      // token_2stage_100
      if (hasTokenTwoStage && intId !== undefined) {
        const t = Date.now();
        try {
          const hits = vexus.tokenChamferTwoStage(intId, 100, TOP_N);
          const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
          const ndcg = computeNDCG(ranked, qrel, K);
          methods.token_2stage_100.sum += ndcg;
          methods.token_2stage_100.count += 1;
          methods.token_2stage_100.totalMs += Date.now() - t;
          // 保存分数用于融合
          tokenScoreMap = {};
          for (const h of hits) {
            const strId = reverseMap[h[0]];
            if (strId) tokenScoreMap[strId] = h[1];
          }
        } catch (e) {
          if (qi === 0) console.error(`  WARN token_2stage: ${e.message}`);
        }
      }

      // fusion_07 (0.7 * token + 0.3 * cfd)
      if (tokenScoreMap && cfdScoreMap) {
        const t = Date.now();
        const allIds = new Set([...Object.keys(tokenScoreMap), ...Object.keys(cfdScoreMap)]);
        const normToken = normalizeScores(tokenScoreMap);
        const normCfd = normalizeScores(cfdScoreMap);
        const fusionScores = [];
        for (const did of allIds) {
          const tScore = normToken[did] || 0;
          const cScore = normCfd[did] || 0;
          fusionScores.push({ did, s: 0.7 * tScore + 0.3 * cScore });
        }
        fusionScores.sort((a, b) => b.s - a.s);
        const ranked = fusionScores.slice(0, K).map(x => x.did);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.fusion_07.sum += ndcg;
        methods.fusion_07.count += 1;
        methods.fusion_07.totalMs += Date.now() - t;
      }

      // BM25 top-200 → Shape-CFD 精排
      // 策略：BM25 候选集 ∩ tokenChamferTwoStage 全扫描结果
      if (hasTokenTwoStage && intId !== undefined && tokenScoreMap) {
        for (const [bk, methodName] of [[200, 'bm25_200_chamfer'], [500, 'bm25_500_chamfer']]) {
          const t = Date.now();
          const bm25Candidates = new Set(bm25DocIds.slice(0, bk));
          // 用 token 分数在 BM25 候选集内重排
          const scored = [];
          for (const [did, score] of Object.entries(tokenScoreMap)) {
            if (bm25Candidates.has(did)) {
              scored.push({ did, s: score });
            }
          }
          // 如果 token 分数覆盖不完全，补充 cosine 分数
          for (const did of bm25Candidates) {
            if (!tokenScoreMap[did] && corpusVecs[did]) {
              scored.push({ did, s: -999 }); // 低优先级
            }
          }
          scored.sort((a, b) => b.s - a.s);
          const ranked = scored.slice(0, K).map(x => x.did);
          const ndcg = computeNDCG(ranked, qrel, K);
          methods[methodName].sum += ndcg;
          methods[methodName].count += 1;
          methods[methodName].totalMs += (Date.now() - t) + bm25Ms;
        }
      }

      // hybrid_200 → Shape-CFD 精排
      if (hasTokenTwoStage && intId !== undefined && tokenScoreMap) {
        const t = Date.now();
        const half = 100;
        const bm25Set = new Set(bm25DocIds.slice(0, half));
        const cosTopIds = cosScores.slice(0, half).map(x => x.did);
        const unionSet = new Set([...bm25Set, ...cosTopIds]);
        // 用 token 分数在混合候选集内重排
        const scored = [];
        for (const [did, score] of Object.entries(tokenScoreMap)) {
          if (unionSet.has(did)) {
            scored.push({ did, s: score });
          }
        }
        for (const did of unionSet) {
          if (!tokenScoreMap[did]) {
            scored.push({ did, s: -999 });
          }
        }
        scored.sort((a, b) => b.s - a.s);
        const ranked = scored.slice(0, K).map(x => x.did);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.hybrid_200_chamfer.sum += ndcg;
        methods.hybrid_200_chamfer.count += 1;
        methods.hybrid_200_chamfer.totalMs += Date.now() - t;
      }
    }

    // 进度显示
    if ((qi + 1) % progressStep === 0 || qi === qids.length - 1) {
      process.stdout.write(`\r  Progress: ${qi + 1}/${qids.length}`);
    }
  }
  process.stdout.write('\n');

  // ═══════════════════════════════════════════════════════════════
  // 9. 输出结果
  // ═══════════════════════════════════════════════════════════════
  const cosAvg = methods.cosine.count > 0 ? methods.cosine.sum / methods.cosine.count : 0;

  console.log('\n' + '='.repeat(72));
  console.log('  BM25 + 分级管线 Benchmark Results (NFCorpus)');
  console.log('='.repeat(72));

  // --- BM25 Recall@K ---
  console.log('\nBM25 Recall@K:');
  for (const rk of RECALL_KS) {
    const avg = recallSums[rk].count > 0
      ? recallSums[rk].sum / recallSums[rk].count : 0;
    console.log(`  K=${String(rk).padEnd(6)} recall=${avg.toFixed(4)}`);
  }

  // --- 混合召回 Recall ---
  console.log('\nBM25 ∪ Cosine Recall@K (each side K/2):');
  for (const hk of HYBRID_KS) {
    const d = hybridRecallSums[hk];
    const avgRecall = d.count > 0 ? d.sum / d.count : 0;
    const avgCand = d.count > 0 ? (d.totalCandidates / d.count).toFixed(0) : 0;
    console.log(`  K=${String(hk).padEnd(6)} recall=${avgRecall.toFixed(4)}, avg_candidates=${avgCand}`);
  }

  // --- 方法对比 ---
  console.log('\n方法对比:');
  const header = '  ' + '方法'.padEnd(26) + 'NDCG@10'.padEnd(12)
    + 'vs cosine'.padEnd(14) + '延迟(ms/q)';
  console.log(header);
  console.log('  ' + '-'.repeat(64));

  const order = [
    'cosine',
    'bm25_only',
    'bm25_200_cosine',
    'bm25_500_cosine',
    'hybrid_200_cosine',
    'hybrid_500_cosine',
    'bm25_200_chamfer',
    'bm25_500_chamfer',
    'hybrid_200_chamfer',
    'token_2stage_100',
    'fusion_07',
  ];

  for (const name of order) {
    const m = methods[name];
    if (m.count === 0) {
      const label = name.padEnd(26);
      console.log(`  ${label}${'--'.padEnd(12)}${'(not run)'.padEnd(14)}--`);
      continue;
    }
    const avg = m.sum / m.count;
    const latency = (m.totalMs / m.count).toFixed(1);
    const label = name.padEnd(26);
    const ndcgStr = avg.toFixed(4).padEnd(12);
    const vsStr = name === 'cosine'
      ? '--'.padEnd(14)
      : pctChange(avg, cosAvg).padEnd(14);
    console.log(`  ${label}${ndcgStr}${vsStr}${latency}`);
  }

  console.log('='.repeat(72));
  console.log(`  Queries: ${qids.length}, Docs: ${allDids.length}`);
  console.log(`  LawVexus: ${vexus ? 'loaded' : 'NOT available'}`);
  console.log(`  Token clouds: ${hasTokenClouds ? 'loaded' : 'NOT loaded'}`);
  console.log();

  // ── 10. 保存结果 ──
  const resultsPath = path.join(DATA_DIR, 'bm25_bench_results.json');
  const results = {
    config: { queries: qids.length, docs: allDids.length, maxQ: MAX_Q || 'all' },
    bm25_recall: {},
    hybrid_recall: {},
    methods: {}
  };
  for (const rk of RECALL_KS) {
    results.bm25_recall[`K=${rk}`] = recallSums[rk].count > 0
      ? +(recallSums[rk].sum / recallSums[rk].count).toFixed(4) : 0;
  }
  for (const hk of HYBRID_KS) {
    const d = hybridRecallSums[hk];
    results.hybrid_recall[`K=${hk}`] = {
      recall: d.count > 0 ? +(d.sum / d.count).toFixed(4) : 0,
      avgCandidates: d.count > 0 ? +(d.totalCandidates / d.count).toFixed(0) : 0,
    };
  }
  for (const name of order) {
    const m = methods[name];
    if (m.count > 0) {
      results.methods[name] = {
        ndcg10: +(m.sum / m.count).toFixed(4),
        count: m.count,
        avgLatencyMs: +(m.totalMs / m.count).toFixed(1),
      };
    }
  }
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`  Results saved to ${resultsPath}\n`);

  // 清理
  bm25.close();
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
