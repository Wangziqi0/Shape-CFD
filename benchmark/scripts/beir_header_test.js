#!/usr/bin/env node
'use strict';
/**
 * beir_header_test.js — Header Token 粗筛实验 (NFCorpus)
 *
 * 验证：只用文档中 top-K 个 token（按不同选择策略）做粗筛，
 * 与质心 cosine 粗筛的 recall@100 对比，以及最终 NDCG@10 对比。
 *
 * 用法：
 *   MAX_Q=10 node beir_header_test.js     # 快速测试
 *   node beir_header_test.js              # 全量 323 queries
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ═══════════════════════════════════════════════════════════════
// 配置
// ═══════════════════════════════════════════════════════════════
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');

const DIM = 4096;       // 向量维度
const TOP_N = 55;       // 精排取 top-N 做 NDCG
const K_NDCG = 10;      // NDCG@K
const COARSE_TOP = 100;  // 粗筛取 top-100

const K_VALUES = [3, 5, 10, 20];
const STRATEGIES = ['max_norm', 'random', 'uniform', 'first_k'];

// ═══════════════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════════════

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

function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

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

function fmtDur(ms) {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}min`;
}

function pctChange(val, base) {
  if (base === 0) return 'N/A';
  const pct = (val - base) / base * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
}

/** 读取一个 token 向量 blob -> Float32Array */
function blobToVec(buf) {
  const vec = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) {
    vec[i] = buf.readFloatLE(i * 4);
  }
  return vec;
}

/** L2 范数 */
function l2Norm(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  return Math.sqrt(s);
}

/** PQ cosine distance: 64 subspaces x 64 dims */
function pqCosDistance(a, b) {
  const SUB = 64, SUBDIM = 64;
  let total = 0;
  for (let s = 0; s < SUB; s++) {
    const off = s * SUBDIM;
    let dot = 0, na = 0, nb = 0;
    for (let d = 0; d < SUBDIM; d++) {
      dot += a[off + d] * b[off + d];
      na += a[off + d] * a[off + d];
      nb += b[off + d] * b[off + d];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    total += denom > 1e-8 ? 1 - dot / denom : 1;
  }
  return total / SUB;
}

/**
 * 简化 Chamfer 匹配：query tokens vs document header tokens
 * 对每个 query token，找 header tokens 中最近邻（最小 PQ cosine distance）
 * 返回 mean distance（越小越好）
 */
function headerChamferScore(queryTokens, headerTokens) {
  let totalDist = 0;
  for (const qt of queryTokens) {
    let minDist = Infinity;
    for (const ht of headerTokens) {
      const d = pqCosDistance(qt, ht);
      if (d < minDist) minDist = d;
    }
    totalDist += minDist;
  }
  return totalDist / queryTokens.length;  // lower = better
}

// ═══════════════════════════════════════════════════════════════
// Token 选择策略
// ═══════════════════════════════════════════════════════════════

/** 按 L2 范数最大的 K 个 */
function selectMaxNorm(tokens, norms, k) {
  const indices = Array.from({ length: tokens.length }, (_, i) => i);
  indices.sort((a, b) => norms[b] - norms[a]);
  return indices.slice(0, k).map(i => tokens[i]);
}

/** 随机选 K 个 (seeded for reproducibility) */
function selectRandom(tokens, norms, k, seed) {
  // Simple deterministic shuffle using seed
  const indices = Array.from({ length: tokens.length }, (_, i) => i);
  let s = seed;
  for (let i = indices.length - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const j = s % (i + 1);
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, k).map(i => tokens[i]);
}

/** 均匀采样 K 个 */
function selectUniform(tokens, norms, k) {
  if (tokens.length <= k) return tokens.slice();
  const step = tokens.length / k;
  const result = [];
  for (let i = 0; i < k; i++) {
    result.push(tokens[Math.floor(i * step)]);
  }
  return result;
}

/** 取前 K 个 */
function selectFirstK(tokens, norms, k) {
  return tokens.slice(0, k);
}

const SELECTOR = {
  max_norm: selectMaxNorm,
  random: selectRandom,
  uniform: selectUniform,
  first_k: selectFirstK,
};

// ═══════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════
async function main() {
  console.log('\n=== Header Token 粗筛实验 (NFCorpus) ===\n');

  // ── 1. 加载 LawVexus (用于 token_2stage baseline) ──
  let LawVexus, vexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
    vexus = new LawVexus('/tmp/beir_header_test');
    // 加载句子级和 token 级点云
    process.stdout.write('Loading sentence clouds... ');
    let t0 = Date.now();
    vexus.loadClouds(CLOUDS_DB);
    console.log(`done (${fmtDur(Date.now() - t0)})`);

    t0 = Date.now();
    process.stdout.write('Loading token clouds (Rust)... ');
    vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
    console.log(`done (${fmtDur(Date.now() - t0)})`);
  } catch (e) {
    console.error('WARNING: LawVexus load failed:', e.message);
    console.error('Will skip token_2stage precision re-ranking.');
    vexus = null;
  }

  // ── 2. 加载评估数据 ──
  let t0 = Date.now();
  process.stdout.write('Loading evaluation data... ');

  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, intId] of Object.entries(idMap)) {
    reverseMap[intId] = strId;
  }

  const corpusVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    corpusVecs[o._id] = new Float32Array(o.vector);
  }
  const allDids = Object.keys(corpusVecs);

  const queryVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl'))) {
    queryVecs[o._id] = new Float32Array(o.vector);
  }

  // qrels
  const qrels = {};
  const qrelLines = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < qrelLines.length; i++) {
    const [qi, di, s] = qrelLines[i].split('\t');
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  let qids = Object.keys(qrels).filter(q => queryVecs[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  // query ID -> int index (for LawVexus token methods)
  const queryIdMap = {};
  const qvLines = fs.readFileSync(path.join(DATA_DIR, 'query_vectors.jsonl'), 'utf8').trim().split('\n');
  for (let i = 0; i < qvLines.length; i++) {
    try {
      const obj = JSON.parse(qvLines[i]);
      const qid = obj._id || obj.id;
      if (qid) queryIdMap[qid] = i;
    } catch (e) {}
  }

  console.log(`done (${fmtDur(Date.now() - t0)}, ${qids.length} queries, ${allDids.length} docs)`);

  // ── 3. 预加载全部文档 token 向量 + 计算范数 ──
  // 14GB 全量太大。策略：逐文档读取并只保存选中的 header tokens。
  const Database = require('better-sqlite3');

  // 3a. 首先，计算质心 cosine 粗筛 top-100 作为 baseline
  console.log('\nPhase 1: Computing centroid cosine top-100 for all queries...');
  t0 = Date.now();
  const centroidTop100 = {};  // qid -> Set of doc string ids
  const centroidRanking = {}; // qid -> [did1, did2, ...] ordered

  for (const qid of qids) {
    const qv = queryVecs[qid];
    const scores = [];
    for (const did of allDids) {
      scores.push({ did, score: cosSim(qv, corpusVecs[did]) });
    }
    scores.sort((a, b) => b.score - a.score);
    const top100 = scores.slice(0, COARSE_TOP).map(s => s.did);
    centroidTop100[qid] = new Set(top100);
    centroidRanking[qid] = top100;
  }
  console.log(`  Centroid cosine top-100 computed (${fmtDur(Date.now() - t0)})`);

  // 3b. 读取全部文档 token + 选择 header tokens
  console.log('\nPhase 2: Reading document tokens and selecting headers...');
  t0 = Date.now();

  const docDb = new Database(TOKEN_CLOUDS_DB, { readonly: true });
  const docStmt = docDb.prepare(
    'SELECT vector FROM chunks WHERE file_id = ? AND vector IS NOT NULL ORDER BY id'
  );

  // 为每个文档预计算所有策略 x K 的 header tokens
  // headerTokens[strategy][K] -> Map<fileId, Float32Array[]>
  const headerTokensMap = {};
  for (const strat of STRATEGIES) {
    headerTokensMap[strat] = {};
    for (const kv of K_VALUES) {
      headerTokensMap[strat][kv] = new Map();
    }
  }

  // 文档质心也缓存（用来验证质心 cosine 和直接向量文件计算的一致性）
  const maxK = Math.max(...K_VALUES);

  let docCount = 0;
  const totalDocs = Object.keys(idMap).length;
  const progressInterval = Math.max(1, Math.floor(totalDocs / 20));

  for (const [strId, intId] of Object.entries(idMap)) {
    const rows = docStmt.all(intId);
    if (rows.length === 0) continue;

    // 解析所有 token 向量并计算 L2 范数
    const tokens = [];
    const norms = [];
    for (const r of rows) {
      const vec = blobToVec(r.vector);
      tokens.push(vec);
      norms.push(l2Norm(vec));
    }

    // 对每个策略和 K 值，选择 header tokens
    for (const strat of STRATEGIES) {
      const selector = SELECTOR[strat];
      for (const kv of K_VALUES) {
        const k = Math.min(kv, tokens.length);
        let selected;
        if (strat === 'random') {
          selected = selector(tokens, norms, k, intId * 31 + 7);  // deterministic seed
        } else {
          selected = selector(tokens, norms, k);
        }
        headerTokensMap[strat][kv].set(intId, selected);
      }
    }

    docCount++;
    if (docCount % progressInterval === 0) {
      process.stdout.write(`  ${docCount}/${totalDocs} docs processed\r`);
    }
  }
  docDb.close();
  console.log(`  ${docCount} docs processed (${fmtDur(Date.now() - t0)})`);

  // 3c. 读取 query token 向量
  console.log('\nPhase 3: Reading query tokens...');
  t0 = Date.now();
  const queryDb = new Database(QUERY_TOKEN_CLOUDS_DB, { readonly: true });
  const queryStmt = queryDb.prepare(
    'SELECT vector FROM chunks WHERE file_id = ? AND vector IS NOT NULL ORDER BY id'
  );

  const queryTokensMap = {};  // qid -> Float32Array[]
  for (const qid of qids) {
    const qIntId = queryIdMap[qid];
    if (qIntId === undefined) continue;
    const rows = queryStmt.all(qIntId);
    queryTokensMap[qid] = rows.map(r => blobToVec(r.vector));
  }
  queryDb.close();
  console.log(`  ${Object.keys(queryTokensMap).length} queries loaded (${fmtDur(Date.now() - t0)})`);

  // ═══════════════════════════════════════════════════════════════
  // 实验 1 & 2: Header token 粗筛 recall@100
  // ═══════════════════════════════════════════════════════════════
  console.log('\nPhase 4: Computing header token coarse screening recall@100...');
  t0 = Date.now();

  // recall[strategy][K] -> average recall
  const recallResults = {};
  for (const strat of STRATEGIES) {
    recallResults[strat] = {};
    for (const kv of K_VALUES) {
      recallResults[strat][kv] = { totalRecall: 0, count: 0 };
    }
  }

  let qDone = 0;
  for (const qid of qids) {
    const qTokens = queryTokensMap[qid];
    if (!qTokens || qTokens.length === 0) continue;

    const centroidSet = centroidTop100[qid];

    for (const strat of STRATEGIES) {
      for (const kv of K_VALUES) {
        const headerMap = headerTokensMap[strat][kv];

        // 对全部文档做 header chamfer 粗筛
        const scores = [];
        for (const [strId, intId] of Object.entries(idMap)) {
          const hTokens = headerMap.get(intId);
          if (!hTokens || hTokens.length === 0) continue;
          const dist = headerChamferScore(qTokens, hTokens);
          scores.push({ did: strId, score: -dist }); // negate so higher = better
        }
        scores.sort((a, b) => b.score - a.score);
        const headerTop100 = new Set(scores.slice(0, COARSE_TOP).map(s => s.did));

        // recall: overlap with centroid top-100
        let overlap = 0;
        for (const did of headerTop100) {
          if (centroidSet.has(did)) overlap++;
        }
        recallResults[strat][kv].totalRecall += overlap / COARSE_TOP;
        recallResults[strat][kv].count++;
      }
    }

    qDone++;
    if (qDone % 5 === 0 || qDone === qids.length) {
      process.stdout.write(`  Query ${qDone}/${qids.length}\r`);
    }
  }
  console.log(`\n  Recall computation done (${fmtDur(Date.now() - t0)})`);

  // ═══════════════════════════════════════════════════════════════
  // 打印实验 1 & 2 结果
  // ═══════════════════════════════════════════════════════════════
  console.log('\n\n=== 实验 1 & 2: Token 选择策略 x K 值 recall@100 (vs 质心 cosine top-100) ===\n');

  // 表头
  const kHeader = K_VALUES.map(k => `K=${k}`.padStart(10)).join('');
  console.log(`${'策略'.padEnd(16)}${kHeader}`);
  console.log('-'.repeat(16 + K_VALUES.length * 10));

  let bestRecall = 0, bestStrat = '', bestK = 0;
  for (const strat of STRATEGIES) {
    const vals = K_VALUES.map(kv => {
      const r = recallResults[strat][kv];
      const avg = r.count > 0 ? r.totalRecall / r.count : 0;
      if (avg > bestRecall) { bestRecall = avg; bestStrat = strat; bestK = kv; }
      return avg.toFixed(4).padStart(10);
    }).join('');
    console.log(`${strat.padEnd(16)}${vals}`);
  }
  console.log(`${'centroid(基线)'.padEnd(16)}${K_VALUES.map(() => '1.0000'.padStart(10)).join('')}`);
  console.log(`\n  Best: ${bestStrat} K=${bestK}, recall=${bestRecall.toFixed(4)}`);

  // ═══════════════════════════════════════════════════════════════
  // 实验 3: Header 粗筛 -> 全量精排 -> NDCG@10
  // ═══════════════════════════════════════════════════════════════
  const SKIP_RERANK = process.env.SKIP_RERANK === '1';
  if (SKIP_RERANK) {
    console.log('\n\nPhase 5: SKIPPED (SKIP_RERANK=1)');
    console.log('\n\n=== 综合结论 ===\n');
    console.log(`1. 最佳 token 选择策略: ${bestStrat}, best recall@100 = ${bestRecall.toFixed(4)} (K=${bestK})`);
    if (bestRecall >= 0.9) {
      console.log('2. Header token 粗筛 recall@100 >= 0.90，有实用潜力');
    } else if (bestRecall >= 0.7) {
      console.log('2. Header token 粗筛 recall@100 在 0.70-0.90 之间，有一定损失但可以接受');
    } else {
      console.log('2. Header token 粗筛 recall@100 < 0.70，损失较大，质心粗筛更优');
    }
    console.log('\nDone (recall-only mode).\n');
    return;
  }

  console.log('\n\nPhase 5: Header coarse -> full token rerank -> NDCG@10...');
  console.log(`  Using best strategy: ${bestStrat}, testing all K values`);

  // 同时测质心 cosine 粗筛 -> token_2stage NDCG@10 作为 baseline
  const hasTokenTwoStage = vexus && typeof vexus.tokenChamferTwoStage === 'function';

  // NDCG results
  const ndcgResults = {};
  for (const kv of K_VALUES) {
    ndcgResults[kv] = { sum: 0, count: 0 };
  }
  const centroidNdcg = { sum: 0, count: 0 };
  // 也加一个直接 header chamfer 不精排的 NDCG
  const headerDirectNdcg = {};
  for (const kv of K_VALUES) {
    headerDirectNdcg[kv] = { sum: 0, count: 0 };
  }

  if (!hasTokenTwoStage) {
    console.log('  WARNING: tokenChamferTwoStage not available, using header-only NDCG.');
  }

  t0 = Date.now();
  qDone = 0;

  for (const qid of qids) {
    const qTokens = queryTokensMap[qid];
    if (!qTokens || qTokens.length === 0) continue;
    const qrel = qrels[qid] || {};
    const qIntId = queryIdMap[qid];

    // Centroid baseline NDCG@10 (using token_2stage if available)
    if (hasTokenTwoStage && qIntId !== undefined) {
      try {
        const hits = vexus.tokenChamferTwoStage(qIntId, COARSE_TOP, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K_NDCG);
        centroidNdcg.sum += ndcg;
        centroidNdcg.count++;
      } catch (e) {}
    }

    // Header-based coarse screening for each K
    for (const kv of K_VALUES) {
      const headerMap = headerTokensMap[bestStrat][kv];

      // Coarse: header chamfer over all docs
      const scores = [];
      for (const [strId, intId] of Object.entries(idMap)) {
        const hTokens = headerMap.get(intId);
        if (!hTokens || hTokens.length === 0) continue;
        const dist = headerChamferScore(qTokens, hTokens);
        scores.push({ did: strId, intId, score: -dist });
      }
      scores.sort((a, b) => b.score - a.score);
      const top100 = scores.slice(0, COARSE_TOP);

      // Direct header NDCG@10 (no rerank)
      const directRanked = top100.slice(0, TOP_N).map(s => s.did);
      headerDirectNdcg[kv].sum += computeNDCG(directRanked, qrel, K_NDCG);
      headerDirectNdcg[kv].count++;

      // Rerank with full token chamfer if available
      if (hasTokenTwoStage && qIntId !== undefined) {
        // We need to get the intIds of top-100 candidates
        const candidateIntIds = top100.map(s => s.intId);

        // Use tokenChamferTwoStage with prefiltered candidates
        // But tokenChamferTwoStage does its own coarse screening internally.
        // So we need to do full-token reranking manually via JS.
        // Instead, let's just read the full tokens for top-100 and do JS chamfer.

        // For efficiency, read tokens for top-100 only
        const rerankedScores = [];
        const docDb2 = new Database(TOKEN_CLOUDS_DB, { readonly: true });
        const rerankStmt = docDb2.prepare(
          'SELECT vector FROM chunks WHERE file_id = ? AND vector IS NOT NULL ORDER BY id'
        );

        for (const cand of top100) {
          const rows = rerankStmt.all(cand.intId);
          if (rows.length === 0) {
            rerankedScores.push({ did: cand.did, score: -999 });
            continue;
          }
          const docTokens = rows.map(r => blobToVec(r.vector));
          // Full Chamfer: Q->D direction (for each query token, find nearest doc token)
          let totalDist = 0;
          for (const qt of qTokens) {
            let minDist = Infinity;
            for (const dt of docTokens) {
              const d = pqCosDistance(qt, dt);
              if (d < minDist) minDist = d;
            }
            totalDist += minDist;
          }
          rerankedScores.push({ did: cand.did, score: -(totalDist / qTokens.length) });
        }
        docDb2.close();

        rerankedScores.sort((a, b) => b.score - a.score);
        const reranked = rerankedScores.slice(0, TOP_N).map(s => s.did);
        const ndcg = computeNDCG(reranked, qrel, K_NDCG);
        ndcgResults[kv].sum += ndcg;
        ndcgResults[kv].count++;
      }
    }

    qDone++;
    const elapsed = Date.now() - t0;
    const eta = qDone > 0 ? (elapsed / qDone) * (qids.length - qDone) : 0;
    process.stdout.write(`  Query ${qDone}/${qids.length} (${fmtDur(elapsed)} elapsed, ETA ${fmtDur(Math.round(eta))})\r`);
  }
  console.log(`\n  NDCG computation done (${fmtDur(Date.now() - t0)})`);

  // ═══════════════════════════════════════════════════════════════
  // 打印实验 3 结果
  // ═══════════════════════════════════════════════════════════════
  const centroidAvg = centroidNdcg.count > 0 ? centroidNdcg.sum / centroidNdcg.count : 0;

  console.log('\n\n=== 实验 3: Header 粗筛 -> NDCG@10 ===\n');
  console.log(`策略: ${bestStrat} (recall@100 最优)\n`);

  console.log(`${'方法'.padEnd(28)} ${'NDCG@10'.padStart(10)} ${'vs centroid'.padStart(12)} ${'vs cosine(0.2195)'.padStart(18)}`);
  console.log('-'.repeat(70));

  if (centroidAvg > 0) {
    console.log(`${'centroid->token_rerank'.padEnd(28)} ${centroidAvg.toFixed(4).padStart(10)} ${'--'.padStart(12)} ${pctChange(centroidAvg, 0.2195).padStart(18)}`);
  }

  for (const kv of K_VALUES) {
    // Direct (no rerank)
    const directAvg = headerDirectNdcg[kv].count > 0
      ? headerDirectNdcg[kv].sum / headerDirectNdcg[kv].count : 0;
    const directLabel = `header_K=${kv}_direct`;
    const directVsCentroid = centroidAvg > 0 ? pctChange(directAvg, centroidAvg) : 'N/A';
    console.log(`${directLabel.padEnd(28)} ${directAvg.toFixed(4).padStart(10)} ${directVsCentroid.padStart(12)} ${pctChange(directAvg, 0.2195).padStart(18)}`);

    // Reranked
    if (ndcgResults[kv].count > 0) {
      const rerankAvg = ndcgResults[kv].sum / ndcgResults[kv].count;
      const rerankLabel = `header_K=${kv}_rerank`;
      const rerankVsCentroid = centroidAvg > 0 ? pctChange(rerankAvg, centroidAvg) : 'N/A';
      console.log(`${rerankLabel.padEnd(28)} ${rerankAvg.toFixed(4).padStart(10)} ${rerankVsCentroid.padStart(12)} ${pctChange(rerankAvg, 0.2195).padStart(18)}`);
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // 内存节省分析
  // ═══════════════════════════════════════════════════════════════
  console.log('\n\n=== 内存节省分析 ===\n');
  const avgTokensPerDoc = 356;
  const bytesPerToken = 64;  // PQ code
  const numDocs = totalDocs;

  console.log(`当前: ${numDocs} docs x ${avgTokensPerDoc} tokens x ${bytesPerToken}B = ${(numDocs * avgTokensPerDoc * bytesPerToken / 1e6).toFixed(1)} MB`);
  for (const kv of K_VALUES) {
    const headerBytes = numDocs * kv * bytesPerToken;
    const savings = 1 - kv / avgTokensPerDoc;
    console.log(`K=${kv}: ${numDocs} docs x ${kv} tokens x ${bytesPerToken}B = ${(headerBytes / 1e6).toFixed(1)} MB (节省 ${(savings * 100).toFixed(1)}%)`);
  }

  // ═══════════════════════════════════════════════════════════════
  // 综合结论
  // ═══════════════════════════════════════════════════════════════
  console.log('\n\n=== 综合结论 ===\n');
  console.log(`1. 最佳 token 选择策略: ${bestStrat}, best recall@100 = ${bestRecall.toFixed(4)} (K=${bestK})`);

  if (bestRecall >= 0.9) {
    console.log('2. Header token 粗筛 recall@100 >= 0.90，有实用潜力');
  } else if (bestRecall >= 0.7) {
    console.log('2. Header token 粗筛 recall@100 在 0.70-0.90 之间，有一定损失但可以接受');
  } else {
    console.log('2. Header token 粗筛 recall@100 < 0.70，损失较大，质心粗筛更优');
  }

  const bestNdcgK = K_VALUES.reduce((best, kv) => {
    const avg = ndcgResults[kv].count > 0 ? ndcgResults[kv].sum / ndcgResults[kv].count : 0;
    return avg > best.val ? { val: avg, k: kv } : best;
  }, { val: 0, k: 0 });

  if (bestNdcgK.val > 0 && centroidAvg > 0) {
    const diff = bestNdcgK.val - centroidAvg;
    if (diff > 0) {
      console.log(`3. 最佳 NDCG@10: header_K=${bestNdcgK.k} = ${bestNdcgK.val.toFixed(4)} (超越质心 ${pctChange(bestNdcgK.val, centroidAvg)})`);
      console.log('   结论: Header token 粗筛 **超越** 质心粗筛，值得采用');
    } else {
      console.log(`3. 最佳 NDCG@10: header_K=${bestNdcgK.k} = ${bestNdcgK.val.toFixed(4)} (低于质心 ${pctChange(bestNdcgK.val, centroidAvg)})`);
      console.log('   结论: Header token 粗筛 **未超越** 质心粗筛');
    }
  }

  console.log('\nDone.\n');
}

main().catch(e => { console.error(e); process.exit(1); });
