#!/usr/bin/env node
'use strict';
/**
 * beir_significance.js — Paired Bootstrap Significance Test
 *
 * 对比方法的 per-query NDCG@10 分数，做 paired bootstrap 显著性检验。
 * 输出：observed diff, p-value, 95% CI
 *
 * 方法：
 *   cosine          — 余弦相似度基线
 *   shape_cfd_v10   — 句子级 PDE 管线
 *   token_2stage_100 — 质心粗筛 + 精排
 *   fusion_07       — 0.7 * token_2stage + 0.3 * shape_cfd_v10
 *
 * 用法：
 *   node beir_significance.js              # 全部 323 queries
 *   MAX_Q=10 node beir_significance.js     # 快速测试前 10 个
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
const TOP_N = 55;
const K = 10;
const N_BOOT = 10000;

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

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function fmtDur(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ═══════════════════════════════════════════════════════════════
// Paired Bootstrap Significance Test
// ═══════════════════════════════════════════════════════════════

/**
 * Paired bootstrap test (two-sided).
 * H0: mean(scoresB) - mean(scoresA) <= 0
 * 返回: { observedDiff, pValue, ci95 }
 */
function pairedBootstrap(scoresA, scoresB, nBoot = N_BOOT) {
  const n = scoresA.length;
  const observedDiff = mean(scoresB) - mean(scoresA);

  // 收集 bootstrap 差异分布
  const bootDiffs = new Float64Array(nBoot);
  let countLeq0 = 0;

  for (let b = 0; b < nBoot; b++) {
    let sumDiff = 0;
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      sumDiff += scoresB[idx] - scoresA[idx];
    }
    const bootMeanDiff = sumDiff / n;
    bootDiffs[b] = bootMeanDiff;
    if (bootMeanDiff <= 0) countLeq0++;
  }

  // p-value (one-sided: P(bootstrap diff <= 0))
  const pValue = countLeq0 / nBoot;

  // 95% CI from bootstrap distribution
  const sorted = Array.from(bootDiffs).sort((a, b) => a - b);
  const lo = sorted[Math.floor(nBoot * 0.025)];
  const hi = sorted[Math.floor(nBoot * 0.975)];

  return { observedDiff, pValue, ci95: [lo, hi] };
}

// ═══════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════
async function main() {
  console.log('\n=== Paired Bootstrap Significance Test (NFCorpus NDCG@10) ===\n');
  console.log(`Bootstrap iterations: ${N_BOOT}`);

  // ── 1. 加载 Rust addon ──
  let LawVexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
  } catch (e) {
    console.error('ERROR: Cannot load law-vexus addon:', e.message);
    process.exit(1);
  }
  const vexus = new LawVexus('/tmp/beir_significance');

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
  }

  if (!hasTokenClouds) {
    console.error('ERROR: Token clouds required for this test');
    process.exit(1);
  }

  const hasTokenTwoStage = typeof vexus.tokenChamferTwoStage === 'function';
  if (!hasTokenTwoStage) {
    console.error('ERROR: tokenChamferTwoStage not available');
    process.exit(1);
  }

  // ── 4. 加载评估数据 ──
  t0 = Date.now();
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

  // query ID -> int ID 映射
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

  // ═══════════════════════════════════════════════════════════════
  // 5. 收集 per-query NDCG@10 scores
  // ═══════════════════════════════════════════════════════════════
  console.log(`\nCollecting per-query NDCG@10 scores (${qids.length} queries)...\n`);

  const perQuery = {
    cosine: [],
    shape_cfd_v10: [],
    token_2stage_100: [],
    fusion_07: [],
  };

  const progressStep = Math.max(1, Math.floor(qids.length / 20));
  let skipped = 0;

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = queryVecs[qid];
    const qrel = qrels[qid];
    const intId = queryIdMap[qid];

    if (intId === undefined) {
      skipped++;
      continue;
    }

    // ── cosine ──
    let cosNdcg;
    {
      const scores = allDids.map(did => ({ did, s: cosSim(qVec, corpusVecs[did]) }));
      scores.sort((a, b) => b.s - a.s);
      const ranked = scores.slice(0, K).map(x => x.did);
      cosNdcg = computeNDCG(ranked, qrel, K);
    }

    // ── shape_cfd_v10 ──
    let cfdNdcg = 0;
    let cfdScoreMap = null;
    try {
      const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
      const hits = vexus.shapeCfdPipeline(qBuf, K, TOP_N);
      const ranked = hits.map(h => reverseMap[h.id]).filter(Boolean);
      cfdNdcg = computeNDCG(ranked, qrel, K);
      cfdScoreMap = {};
      for (const h of hits) {
        const strId = reverseMap[h.id];
        if (strId) cfdScoreMap[strId] = h.score;
      }
    } catch (e) {
      if (qi === 0) console.error(`  WARN shapeCfdPipeline: ${e.message}`);
    }

    // ── token_2stage_100 ──
    let tokenNdcg = 0;
    let tokenScoreMap = null;
    try {
      const hits = vexus.tokenChamferTwoStage(intId, 100, TOP_N);
      const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
      tokenNdcg = computeNDCG(ranked, qrel, K);
      tokenScoreMap = {};
      for (const h of hits) {
        const strId = reverseMap[h[0]];
        if (strId) tokenScoreMap[strId] = h[1];
      }
    } catch (e) {
      if (qi === 0) console.error(`  WARN twoStage100: ${e.message}`);
    }

    // ── fusion_07 ──
    let fusionNdcg = 0;
    if (tokenScoreMap && cfdScoreMap) {
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
      fusionNdcg = computeNDCG(ranked, qrel, K);
    }

    perQuery.cosine.push(cosNdcg);
    perQuery.shape_cfd_v10.push(cfdNdcg);
    perQuery.token_2stage_100.push(tokenNdcg);
    perQuery.fusion_07.push(fusionNdcg);

    if ((qi + 1) % progressStep === 0 || qi === qids.length - 1) {
      process.stdout.write(`\r  Progress: ${qi + 1}/${qids.length}`);
    }
  }
  process.stdout.write('\n');

  if (skipped > 0) {
    console.log(`  (skipped ${skipped} queries with no int ID mapping)`);
  }

  const n = perQuery.cosine.length;
  console.log(`  Collected ${n} per-query scores for each method.\n`);

  // ═══════════════════════════════════════════════════════════════
  // 6. 描述统计
  // ═══════════════════════════════════════════════════════════════
  console.log('='.repeat(72));
  console.log('  Descriptive Statistics');
  console.log('='.repeat(72));
  console.log('  ' + 'Method'.padEnd(24) + 'Mean NDCG@10'.padEnd(16) + 'Std Dev'.padEnd(12) + 'Median');
  console.log('  ' + '-'.repeat(60));

  for (const [name, scores] of Object.entries(perQuery)) {
    const m = mean(scores);
    const sorted = [...scores].sort((a, b) => a - b);
    const median = n % 2 === 0
      ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
      : sorted[Math.floor(n / 2)];
    let variance = 0;
    for (const s of scores) variance += (s - m) * (s - m);
    const std = Math.sqrt(variance / (n - 1));
    console.log(`  ${name.padEnd(24)}${m.toFixed(4).padEnd(16)}${std.toFixed(4).padEnd(12)}${median.toFixed(4)}`);
  }

  // ═══════════════════════════════════════════════════════════════
  // 7. Paired Bootstrap Tests
  // ═══════════════════════════════════════════════════════════════
  console.log('\n' + '='.repeat(72));
  console.log('  Paired Bootstrap Significance Tests (n_boot=' + N_BOOT + ')');
  console.log('='.repeat(72));

  const comparisons = [
    { name: 'fusion_07 vs token_2stage_100', a: 'token_2stage_100', b: 'fusion_07' },
    { name: 'fusion_07 vs shape_cfd_v10',    a: 'shape_cfd_v10',    b: 'fusion_07' },
    { name: 'token_2stage_100 vs cosine',     a: 'cosine',           b: 'token_2stage_100' },
    { name: 'shape_cfd_v10 vs cosine',        a: 'cosine',           b: 'shape_cfd_v10' },
    { name: 'fusion_07 vs cosine',            a: 'cosine',           b: 'fusion_07' },
  ];

  console.log('  ' + 'Comparison'.padEnd(36) + 'Diff'.padEnd(10) + 'p-value'.padEnd(12) + '95% CI');
  console.log('  ' + '-'.repeat(72));

  const sigResults = {};

  for (const cmp of comparisons) {
    const result = pairedBootstrap(perQuery[cmp.a], perQuery[cmp.b], N_BOOT);
    const sig = result.pValue < 0.05 ? ' ***' : (result.pValue < 0.10 ? ' *' : '');
    const diffStr = (result.observedDiff >= 0 ? '+' : '') + result.observedDiff.toFixed(4);
    const pStr = result.pValue < 0.001 ? '<0.001' : result.pValue.toFixed(4);
    const ciStr = `[${result.ci95[0] >= 0 ? '+' : ''}${result.ci95[0].toFixed(4)}, ${result.ci95[1] >= 0 ? '+' : ''}${result.ci95[1].toFixed(4)}]`;

    console.log(`  ${cmp.name.padEnd(36)}${diffStr.padEnd(10)}${pStr.padEnd(12)}${ciStr}${sig}`);

    sigResults[cmp.name] = {
      observedDiff: +result.observedDiff.toFixed(6),
      pValue: +result.pValue.toFixed(6),
      ci95: [+result.ci95[0].toFixed(6), +result.ci95[1].toFixed(6)],
      significant_005: result.pValue < 0.05,
      significant_010: result.pValue < 0.10,
    };
  }

  console.log('\n  *** p < 0.05    * p < 0.10\n');

  // ═══════════════════════════════════════════════════════════════
  // 8. 结论
  // ═══════════════════════════════════════════════════════════════
  console.log('='.repeat(72));
  console.log('  Conclusions');
  console.log('='.repeat(72));

  const f07_vs_t2s = sigResults['fusion_07 vs token_2stage_100'];
  if (f07_vs_t2s.significant_005) {
    console.log('  [SIGNIFICANT] fusion_07 > token_2stage_100 (p < 0.05)');
    console.log(`    The +${(f07_vs_t2s.observedDiff).toFixed(4)} NDCG improvement is statistically significant.`);
    console.log('    PDE fusion provides a genuine orthogonal signal boost.');
  } else if (f07_vs_t2s.significant_010) {
    console.log('  [MARGINAL] fusion_07 vs token_2stage_100 (p < 0.10, not < 0.05)');
    console.log(`    The +${(f07_vs_t2s.observedDiff).toFixed(4)} NDCG improvement is marginally significant.`);
    console.log('    More queries or stronger effect needed for definitive claim.');
  } else {
    console.log('  [NOT SIGNIFICANT] fusion_07 vs token_2stage_100 (p >= 0.10)');
    console.log(`    The +${(f07_vs_t2s.observedDiff).toFixed(4)} NDCG difference is NOT statistically significant.`);
    console.log('    PDE fusion improvement may be within noise range.');
  }
  console.log();

  const t2s_vs_cos = sigResults['token_2stage_100 vs cosine'];
  if (t2s_vs_cos.significant_005) {
    console.log(`  [SIGNIFICANT] token_2stage_100 > cosine (+${t2s_vs_cos.observedDiff.toFixed(4)}, p < 0.05)`);
    console.log('    Token-level PQ-Chamfer decisively outperforms cosine baseline.');
  }

  const f07_vs_cos = sigResults['fusion_07 vs cosine'];
  if (f07_vs_cos.significant_005) {
    console.log(`  [SIGNIFICANT] fusion_07 > cosine (+${f07_vs_cos.observedDiff.toFixed(4)}, p < 0.05)`);
    console.log('    Full Shape-CFD pipeline with fusion is a definitive improvement.');
  }
  console.log();

  // ═══════════════════════════════════════════════════════════════
  // 9. 保存结果
  // ═══════════════════════════════════════════════════════════════
  const outputPath = path.join(DATA_DIR, 'significance_results.json');
  const output = {
    timestamp: new Date().toISOString(),
    nQueries: n,
    nBootstrap: N_BOOT,
    methods: {},
    comparisons: sigResults,
  };
  for (const [name, scores] of Object.entries(perQuery)) {
    output.methods[name] = {
      mean: +mean(scores).toFixed(4),
      scores: scores.map(s => +s.toFixed(6)),
    };
  }
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`  Results saved to ${outputPath}\n`);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
