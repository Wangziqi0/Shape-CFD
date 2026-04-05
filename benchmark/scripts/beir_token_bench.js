#!/usr/bin/env node
'use strict';
/**
 * beir_token_bench.js — V11 Token-Level Benchmark (NFCorpus)
 *
 * 单线程模式：token_clouds.sqlite 约 14GB，多 worker 内存不够，
 * 所以用单个 LawVexus 实例顺序处理全部 query。
 *
 * 对比方法：
 *   cosine          — 余弦相似度基线
 *   shape_cfd_v10   — 当前最优管线 shapeCfdPipeline (句子级)
 *   token_chamfer   — 纯 token Chamfer 直排
 *   token_v11_pipe  — V11 完整管线 (token 粗筛 + 句子 PDE)
 *   token_2stage_100 — 两阶段 (质心粗筛100 + 精排55)
 *   sampled_graph   — 方案1: token 采样建图 + PDE
 *   fusion_03/05/07 — 方案2: token_2stage + shape_cfd 分数融合
 *   weak_pde        — 方案3: 极弱 PDE (D=0.01, alpha=0.01)
 *
 * 用法：
 *   node beir_token_bench.js              # 完整 benchmark
 *   MAX_Q=10 node beir_token_bench.js     # 快速测试前 10 个 query
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ═══════════════════════════════════════════════════════════════
// 配置
// ═══════════════════════════════════════════════════════════════
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const TOP_N = 55;  // shape_cfd_v10 最优粗筛数
const K = 10;      // NDCG@K

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
  // DCG: 按 ranked 顺序累加
  let dcg = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++) {
    const rel = qrel[ranked[i]] || 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  // IDCG: 按 qrel 值降序
  const idealRels = Object.values(qrel).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(idealRels.length, k); i++) {
    idcg += (Math.pow(2, idealRels[i]) - 1) / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

/** 格式化时长 (ms -> 可读) */
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

// ═══════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════
async function main() {
  console.log('\n=== V11 Token-Level + PDE Fix Benchmark (NFCorpus) ===\n');

  // ── 1. 加载 Rust addon ──
  let LawVexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
  } catch (e) {
    console.error('ERROR: 无法加载 law-vexus addon:', e.message);
    process.exit(1);
  }
  const vexus = new LawVexus('/tmp/beir_token_bench');

  // ── 2. 加载句子级点云 ──
  let t0 = Date.now();
  process.stdout.write('Loading sentence clouds... ');
  const cloudInfo = vexus.loadClouds(CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${cloudInfo}`);

  // ── 3. 加载 token 级点云 ──
  let hasTokenClouds = false;
  if (typeof vexus.loadTokenCloudsSqlite === 'function') {
    t0 = Date.now();
    process.stdout.write('Loading token clouds... ');
    try {
      const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
      hasTokenClouds = true;
      console.log(`done (${fmtDur(Date.now() - t0)}) ${tokenInfo}`);
    } catch (e) {
      console.log(`FAILED: ${e.message}`);
    }
  } else {
    console.log('Loading token clouds... SKIPPED (loadTokenCloudsSqlite not implemented)');
  }

  // ── 4. 加载评估数据 ──
  t0 = Date.now();
  process.stdout.write('Loading evaluation data... ');

  // id_map: string_id -> internal_int_id
  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, intId] of Object.entries(idMap)) {
    reverseMap[intId] = strId;
  }

  // corpus 向量
  const corpusVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    corpusVecs[o._id] = new Float32Array(o.vector);
  }
  const allDids = Object.keys(corpusVecs);

  // query 向量
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

  // 有标注的 query
  let qids = Object.keys(qrels).filter(q => queryVecs[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  console.log(`done (${fmtDur(Date.now() - t0)}, ${qids.length} queries, ${allDids.length} docs)`);

  // ── 5. 构建 query ID -> int ID 映射 (token 方法需要) ──
  const queryIdMap = {};
  const qvLines = fs.readFileSync(path.join(DATA_DIR, 'query_vectors.jsonl'), 'utf8').trim().split('\n');
  for (let i = 0; i < qvLines.length; i++) {
    try {
      const obj = JSON.parse(qvLines[i]);
      const qid = obj._id || obj.id;
      if (qid) queryIdMap[qid] = i;
    } catch (e) {}
  }

  // ═══════════════════════════════════════════════════════════════
  // 6. 跑 Benchmark
  // ═══════════════════════════════════════════════════════════════
  console.log(`\nRunning benchmark (${qids.length} queries)...\n`);

  // 方法定义
  const methods = {
    cosine:           { sum: 0, count: 0, totalMs: 0, enabled: true },
    shape_cfd_v10:    { sum: 0, count: 0, totalMs: 0, enabled: true },
    token_chamfer:    { sum: 0, count: 0, totalMs: 0, enabled: false },
    token_v11_pipe:   { sum: 0, count: 0, totalMs: 0, enabled: false },
    token_2stage_100: { sum: 0, count: 0, totalMs: 0, enabled: false },
    sampled_graph:    { sum: 0, count: 0, totalMs: 0, enabled: false },
    fusion_03:        { sum: 0, count: 0, totalMs: 0, enabled: false },
    fusion_05:        { sum: 0, count: 0, totalMs: 0, enabled: false },
    fusion_07:        { sum: 0, count: 0, totalMs: 0, enabled: false },
    weak_pde:         { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_2stage_p1:    { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_2stage_p3:    { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_fusion_07:    { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_fast_200_p1:  { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_fast_500_p1:  { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_fast_500_p2:  { sum: 0, count: 0, totalMs: 0, enabled: false },
    inv_f_fusion07:   { sum: 0, count: 0, totalMs: 0, enabled: false },
    adc_2stage_100:   { sum: 0, count: 0, totalMs: 0, enabled: false },
    adc_2stage_200:   { sum: 0, count: 0, totalMs: 0, enabled: false },
    adc_fusion_07:    { sum: 0, count: 0, totalMs: 0, enabled: false },
  };

  // 检测方法可用性
  const hasTokenChamferRank = typeof vexus.tokenChamferRank === 'function';
  const hasTokenChamferPipeline = typeof vexus.tokenChamferPipeline === 'function';
  const hasTokenTwoStage = typeof vexus.tokenChamferTwoStage === 'function';
  const hasSampledPipeline = typeof vexus.tokenChamferPipelineSampled === 'function';
  const hasWeakPipeline = typeof vexus.tokenChamferPipelineWeak === 'function';

  // 已知分数的慢方法默认关闭，聚焦倒排索引对比
  if (hasTokenClouds && hasTokenTwoStage) {
    methods.token_2stage_100.enabled = true;
    methods.fusion_07.enabled = true;
    console.log('  [v] tokenChamferTwoStage + fusion_07 available (baseline)');
  }
  const hasInvertedTwoStage = typeof vexus.tokenInvertedTwoStage === 'function';
  const hasInvertedFast = typeof vexus.tokenInvertedFast === 'function';
  if (hasTokenClouds && hasInvertedTwoStage) {
    methods.inv_2stage_p1.enabled = true;
    methods.inv_2stage_p3.enabled = true;
    methods.inv_fusion_07.enabled = true;
    console.log('  [v] tokenInvertedTwoStage available (V14)');
  }
  if (hasTokenClouds && hasInvertedFast) {
    methods.inv_fast_200_p1.enabled = true;
    methods.inv_fast_500_p1.enabled = true;
    methods.inv_fast_500_p2.enabled = true;
    methods.inv_f_fusion07.enabled = true;
    console.log('  [v] tokenInvertedFast available (V14c optimized)');
  }
  if (!hasInvertedTwoStage && !hasInvertedFast) {
    console.log(`  [x] inverted index not available (clouds=${hasTokenClouds})`);
  }
  const hasAdcTwoStage = typeof vexus.tokenAdcTwoStage === 'function';
  if (hasTokenClouds && hasAdcTwoStage) {
    methods.adc_2stage_100.enabled = true;
    methods.adc_2stage_200.enabled = true;
    methods.adc_fusion_07.enabled = true;
    console.log('  [v] tokenAdcTwoStage available (PQ ADC)');
  }
  console.log();

  // 进度显示间隔
  const progressStep = Math.max(1, Math.floor(qids.length / 20));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = queryVecs[qid];
    const qrel = qrels[qid];
    const intId = queryIdMap[qid];

    // ── cosine baseline ──
    {
      const t = Date.now();
      const scores = allDids.map(did => ({ did, s: cosSim(qVec, corpusVecs[did]) }));
      scores.sort((a, b) => b.s - a.s);
      const ranked = scores.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.cosine.sum += ndcg;
      methods.cosine.count += 1;
      methods.cosine.totalMs += Date.now() - t;
    }

    // ── shape_cfd_v10 (现有 Rust 管线) ──
    // 收集分数用于后续融合
    let cfdScoreMap = null;
    {
      const t = Date.now();
      try {
        const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
        const hits = vexus.shapeCfdPipeline(qBuf, K, TOP_N);
        const ranked = hits.map(h => reverseMap[h.id]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.shape_cfd_v10.sum += ndcg;
        methods.shape_cfd_v10.count += 1;
        // 保存分数用于融合（id -> score）
        cfdScoreMap = {};
        for (const h of hits) {
          const strId = reverseMap[h.id];
          if (strId) cfdScoreMap[strId] = h.score;
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN shapeCfdPipeline error: ${e.message}`);
      }
      methods.shape_cfd_v10.totalMs += Date.now() - t;
    }

    // ── token_chamfer (纯 token Chamfer 直排) ──
    if (methods.token_chamfer.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferRank(intId, K);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.token_chamfer.sum += ndcg;
        methods.token_chamfer.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN tokenChamferRank error: ${e.message}`);
        methods.token_chamfer.enabled = false;
      }
      methods.token_chamfer.totalMs += Date.now() - t;
    }

    // ── token_v11_pipe (V11 完整管线) ──
    if (methods.token_v11_pipe.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferPipeline(intId, K, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.token_v11_pipe.sum += ndcg;
        methods.token_v11_pipe.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN tokenChamferPipeline error: ${e.message}`);
        methods.token_v11_pipe.enabled = false;
      }
      methods.token_v11_pipe.totalMs += Date.now() - t;
    }

    // ── token_2stage_100 + 收集分数用于融合 ──
    let tokenScoreMap = null;
    if (methods.token_2stage_100.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferTwoStage(intId, 100, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.token_2stage_100.sum += ndcg;
        methods.token_2stage_100.count += 1;
        // 保存分数用于融合
        tokenScoreMap = {};
        for (const h of hits) {
          const strId = reverseMap[h[0]];
          if (strId) tokenScoreMap[strId] = h[1]; // score = exp(-2*dist)
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN twoStage100 error: ${e.message}`);
        methods.token_2stage_100.enabled = false;
      }
      methods.token_2stage_100.totalMs += Date.now() - t;
    }

    // ── 方案2: 分数融合 (token_2stage + shape_cfd_v10) ──
    if (tokenScoreMap && cfdScoreMap) {
      // 收集所有文档 ID 的并集
      const allIds = new Set([...Object.keys(tokenScoreMap), ...Object.keys(cfdScoreMap)]);

      // 归一化到 [0,1]
      const normToken = normalizeScores(tokenScoreMap);
      const normCfd = normalizeScores(cfdScoreMap);

      for (const [lambda, methodName] of [[0.3, 'fusion_03'], [0.5, 'fusion_05'], [0.7, 'fusion_07']]) {
        if (!methods[methodName].enabled) continue;
        const t = Date.now();

        // 融合分数: lambda * token + (1-lambda) * cfd
        const fusionScores = [];
        for (const did of allIds) {
          const tScore = normToken[did] || 0;
          const cScore = normCfd[did] || 0;
          fusionScores.push({ did, s: lambda * tScore + (1 - lambda) * cScore });
        }
        fusionScores.sort((a, b) => b.s - a.s);
        const ranked = fusionScores.slice(0, K).map(x => x.did);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods[methodName].sum += ndcg;
        methods[methodName].count += 1;
        methods[methodName].totalMs += Date.now() - t;
      }
    }

    // ── 方案1: sampled_graph (token 采样建图 + PDE) ──
    if (methods.sampled_graph.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferPipelineSampled(intId, K, TOP_N, 20);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.sampled_graph.sum += ndcg;
        methods.sampled_graph.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN sampledGraph error: ${e.message}`);
        methods.sampled_graph.enabled = false;
      }
      methods.sampled_graph.totalMs += Date.now() - t;
    }

    // ── 方案3: weak_pde (极弱 PDE) ──
    if (methods.weak_pde.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferPipelineWeak(intId, K, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.weak_pde.sum += ndcg;
        methods.weak_pde.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN weakPde error: ${e.message}`);
        methods.weak_pde.enabled = false;
      }
      methods.weak_pde.totalMs += Date.now() - t;
    }

    // ── V14: 倒排索引两阶段 (n_probe=1) ──
    let invScoreMap_p1 = null;
    if (methods.inv_2stage_p1.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenInvertedTwoStage(intId, 200, TOP_N, 1);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.inv_2stage_p1.sum += ndcg;
        methods.inv_2stage_p1.count += 1;
        // 保存分数用于融合
        invScoreMap_p1 = {};
        for (const h of hits) {
          const strId = reverseMap[h[0]];
          if (strId) invScoreMap_p1[strId] = h[1];
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN inv_2stage_p1 error: ${e.message}`);
        methods.inv_2stage_p1.enabled = false;
      }
      methods.inv_2stage_p1.totalMs += Date.now() - t;
    }

    // ── V14: 倒排索引两阶段 (n_probe=3) ──
    if (methods.inv_2stage_p3.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenInvertedTwoStage(intId, 200, TOP_N, 3);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.inv_2stage_p3.sum += ndcg;
        methods.inv_2stage_p3.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN inv_2stage_p3 error: ${e.message}`);
        methods.inv_2stage_p3.enabled = false;
      }
      methods.inv_2stage_p3.totalMs += Date.now() - t;
    }

    // ── V14: 倒排 + PDE 融合 (lambda=0.7) ──
    if (methods.inv_fusion_07.enabled && invScoreMap_p1 && cfdScoreMap) {
      const t = Date.now();
      const allIds = new Set([...Object.keys(invScoreMap_p1), ...Object.keys(cfdScoreMap)]);
      const normInv = normalizeScores(invScoreMap_p1);
      const normCfd = normalizeScores(cfdScoreMap);
      const fusionScores = [];
      for (const did of allIds) {
        const tScore = normInv[did] || 0;
        const cScore = normCfd[did] || 0;
        fusionScores.push({ did, s: 0.7 * tScore + 0.3 * cScore });
      }
      fusionScores.sort((a, b) => b.s - a.s);
      const ranked = fusionScores.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.inv_fusion_07.sum += ndcg;
      methods.inv_fusion_07.count += 1;
      methods.inv_fusion_07.totalMs += Date.now() - t;
    }

    // ── V14c: 快速倒排 (coarse=200, n_probe=1, argmin优化) ──
    let invFastScoreMap = null;
    if (methods.inv_fast_200_p1.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenInvertedFast(intId, 200, TOP_N, 1);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.inv_fast_200_p1.sum += ndcg;
        methods.inv_fast_200_p1.count += 1;
        invFastScoreMap = {};
        for (const h of hits) {
          const strId = reverseMap[h[0]];
          if (strId) invFastScoreMap[strId] = h[1];
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN inv_fast_200_p1 error: ${e.message}`);
        methods.inv_fast_200_p1.enabled = false;
      }
      methods.inv_fast_200_p1.totalMs += Date.now() - t;
    }

    // ── V14c: 快速倒排 (coarse=500, n_probe=1) ──
    if (methods.inv_fast_500_p1.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenInvertedFast(intId, 500, TOP_N, 1);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.inv_fast_500_p1.sum += ndcg;
        methods.inv_fast_500_p1.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN inv_fast_500_p1 error: ${e.message}`);
        methods.inv_fast_500_p1.enabled = false;
      }
      methods.inv_fast_500_p1.totalMs += Date.now() - t;
    }

    // ── V14c: 快速倒排 (coarse=500, n_probe=2) ──
    if (methods.inv_fast_500_p2.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenInvertedFast(intId, 500, TOP_N, 2);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.inv_fast_500_p2.sum += ndcg;
        methods.inv_fast_500_p2.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN inv_fast_500_p2 error: ${e.message}`);
        methods.inv_fast_500_p2.enabled = false;
      }
      methods.inv_fast_500_p2.totalMs += Date.now() - t;
    }

    // ── V14c: 快速倒排 + PDE 融合 ──
    if (methods.inv_f_fusion07.enabled && invFastScoreMap && cfdScoreMap) {
      const t = Date.now();
      const allIds = new Set([...Object.keys(invFastScoreMap), ...Object.keys(cfdScoreMap)]);
      const normInv = normalizeScores(invFastScoreMap);
      const normCfd = normalizeScores(cfdScoreMap);
      const fusionScores = [];
      for (const did of allIds) {
        const tScore = normInv[did] || 0;
        const cScore = normCfd[did] || 0;
        fusionScores.push({ did, s: 0.7 * tScore + 0.3 * cScore });
      }
      fusionScores.sort((a, b) => b.s - a.s);
      const ranked = fusionScores.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.inv_f_fusion07.sum += ndcg;
      methods.inv_f_fusion07.count += 1;
      methods.inv_f_fusion07.totalMs += Date.now() - t;
    }

    // ── PQ ADC 两阶段 (coarse=100) ──
    let adcScoreMap = null;
    if (methods.adc_2stage_100.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenAdcTwoStage(intId, 100, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.adc_2stage_100.sum += ndcg;
        methods.adc_2stage_100.count += 1;
        adcScoreMap = {};
        for (const h of hits) {
          const strId = reverseMap[h[0]];
          if (strId) adcScoreMap[strId] = h[1];
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN adc_2stage_100 error: ${e.message}`);
        methods.adc_2stage_100.enabled = false;
      }
      methods.adc_2stage_100.totalMs += Date.now() - t;
    }

    // ── PQ ADC 两阶段 (coarse=200) ──
    if (methods.adc_2stage_200.enabled && intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenAdcTwoStage(intId, 200, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.adc_2stage_200.sum += ndcg;
        methods.adc_2stage_200.count += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN adc_2stage_200 error: ${e.message}`);
        methods.adc_2stage_200.enabled = false;
      }
      methods.adc_2stage_200.totalMs += Date.now() - t;
    }

    // ── PQ ADC + PDE 融合 ──
    if (methods.adc_fusion_07.enabled && adcScoreMap && cfdScoreMap) {
      const t = Date.now();
      const allIds = new Set([...Object.keys(adcScoreMap), ...Object.keys(cfdScoreMap)]);
      const normAdc = normalizeScores(adcScoreMap);
      const normCfd = normalizeScores(cfdScoreMap);
      const fusionScores = [];
      for (const did of allIds) {
        const tScore = normAdc[did] || 0;
        const cScore = normCfd[did] || 0;
        fusionScores.push({ did, s: 0.7 * tScore + 0.3 * cScore });
      }
      fusionScores.sort((a, b) => b.s - a.s);
      const ranked = fusionScores.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.adc_fusion_07.sum += ndcg;
      methods.adc_fusion_07.count += 1;
      methods.adc_fusion_07.totalMs += Date.now() - t;
    }

    // 进度显示
    if ((qi + 1) % progressStep === 0 || qi === qids.length - 1) {
      process.stdout.write(`\r  Progress: ${qi + 1}/${qids.length}`);
    }
  }
  process.stdout.write('\n');

  // ═══════════════════════════════════════════════════════════════
  // 7. 输出结果
  // ═══════════════════════════════════════════════════════════════
  console.log('\n' + '='.repeat(72));
  console.log('  V11 Token-Level + PDE Fix Benchmark Results (NFCorpus)');
  console.log('='.repeat(72));

  const header = '  ' + '方法'.padEnd(24) + 'NDCG@10'.padEnd(12)
    + 'vs cosine'.padEnd(14) + '延迟(ms/q)';
  console.log(header);
  console.log('  ' + '-'.repeat(62));

  const cosAvg = methods.cosine.count > 0
    ? methods.cosine.sum / methods.cosine.count : 0;

  const order = [
    'cosine', 'shape_cfd_v10',
    'token_2stage_100', 'fusion_07',
    'inv_2stage_p1', 'inv_2stage_p3', 'inv_fusion_07',
    'inv_fast_200_p1', 'inv_fast_500_p1', 'inv_fast_500_p2', 'inv_f_fusion07',
    'adc_2stage_100', 'adc_2stage_200', 'adc_fusion_07'
  ];
  for (const name of order) {
    const m = methods[name];
    if (m.count === 0) {
      const label = name.padEnd(24);
      console.log(`  ${label}${'--'.padEnd(12)}${'(not run)'.padEnd(14)}--`);
      continue;
    }
    const avg = m.sum / m.count;
    const latency = (m.totalMs / m.count).toFixed(1);
    const label = name.padEnd(24);
    const ndcgStr = avg.toFixed(4).padEnd(12);
    const vsStr = name === 'cosine'
      ? '--'.padEnd(14)
      : pctChange(avg, cosAvg).padEnd(14);
    console.log(`  ${label}${ndcgStr}${vsStr}${latency}`);
  }

  console.log('='.repeat(72));
  console.log(`  Queries: ${qids.length}, Docs: ${allDids.length}`);
  console.log(`  Token clouds: ${hasTokenClouds ? 'loaded' : 'NOT loaded'}`);
  console.log();

  // ── 8. 保存结果到 JSON ──
  const resultsPath = path.join(DATA_DIR, 'token_bench_results.json');
  const results = {};
  for (const name of order) {
    const m = methods[name];
    if (m.count > 0) {
      results[name] = {
        ndcg10: +(m.sum / m.count).toFixed(4),
        count: m.count,
        avgLatencyMs: +(m.totalMs / m.count).toFixed(1),
      };
    }
  }
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`  Results saved to ${resultsPath}\n`);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
