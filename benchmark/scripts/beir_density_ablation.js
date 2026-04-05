#!/usr/bin/env node
'use strict';
/**
 * beir_density_ablation.js -- 密度加权 PQ-Chamfer 消融实验 (NFCorpus)
 *
 * 变体：
 *   token_2stage (baseline)      -- 标准两阶段，无密度加权
 *   density_v1_k3                -- v1: 只加权 doc->query 方向，K=3
 *   density_v1_k5                -- v1: 只加权 doc->query 方向，K=5
 *   density_v1_k7                -- v1: 只加权 doc->query 方向，K=7
 *   density_v2_k5                -- v2: 两个方向都加权，K=5
 *
 * 用法：
 *   node beir_density_ablation.js
 *   MAX_Q=10 node beir_density_ablation.js   # 快速测试
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

// ═══════════════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════════════

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

function fmtDur(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function pctChange(val, base) {
  if (base === 0) return 'N/A';
  const pct = (val - base) / base * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
}

// ═══════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════
async function main() {
  console.log('\n=== Density-Weighted PQ-Chamfer Ablation (NFCorpus) ===\n');

  // 1. 加载 Rust addon
  let LawVexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
  } catch (e) {
    console.error('ERROR: 无法加载 law-vexus addon:', e.message);
    process.exit(1);
  }
  const vexus = new LawVexus('/tmp/beir_density_ablation');

  // 2. 加载句子级点云 (shape_cfd 融合需要)
  let t0 = Date.now();
  process.stdout.write('Loading sentence clouds... ');
  const cloudInfo = vexus.loadClouds(CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${cloudInfo}`);

  // 3. 加载 token 级点云
  t0 = Date.now();
  process.stdout.write('Loading token clouds... ');
  const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${tokenInfo}`);

  // 4. 检查方法可用性
  const hasDensity = typeof vexus.tokenChamferTwoStageDensity === 'function';
  const hasTwoStage = typeof vexus.tokenChamferTwoStage === 'function';
  if (!hasDensity) {
    console.error('ERROR: tokenChamferTwoStageDensity not available');
    process.exit(1);
  }
  if (!hasTwoStage) {
    console.error('ERROR: tokenChamferTwoStage not available');
    process.exit(1);
  }
  console.log('  [v] tokenChamferTwoStage available');
  console.log('  [v] tokenChamferTwoStageDensity available');

  // 5. 加载评估数据
  t0 = Date.now();
  process.stdout.write('Loading evaluation data... ');

  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, intId] of Object.entries(idMap)) {
    reverseMap[intId] = strId;
  }

  // qrels
  const qrels = {};
  const qrelLines = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < qrelLines.length; i++) {
    const [qi, di, s] = qrelLines[i].split('\t');
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  // query ID 映射
  const queryVecLines = fs.readFileSync(path.join(DATA_DIR, 'query_vectors.jsonl'), 'utf8').trim().split('\n');
  const queryIdMap = {};
  const queryVecs = {};
  for (let i = 0; i < queryVecLines.length; i++) {
    try {
      const obj = JSON.parse(queryVecLines[i]);
      const qid = obj._id || obj.id;
      if (qid) {
        queryIdMap[qid] = i;
        queryVecs[qid] = true;
      }
    } catch (e) {}
  }

  let qids = Object.keys(qrels).filter(q => queryVecs[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  console.log(`done (${fmtDur(Date.now() - t0)}, ${qids.length} queries)`);

  // 6. 定义方法
  const methods = {
    token_2stage:   { sum: 0, count: 0, totalMs: 0, perQuery: [] },
    density_v1_k3:  { sum: 0, count: 0, totalMs: 0, perQuery: [] },
    density_v1_k5:  { sum: 0, count: 0, totalMs: 0, perQuery: [] },
    density_v1_k7:  { sum: 0, count: 0, totalMs: 0, perQuery: [] },
    density_v2_k5:  { sum: 0, count: 0, totalMs: 0, perQuery: [] },
  };

  console.log(`\nRunning benchmark (${qids.length} queries)...\n`);

  const progressStep = Math.max(1, Math.floor(qids.length / 20));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qrel = qrels[qid];
    const intId = queryIdMap[qid];
    if (intId === undefined) continue;

    // -- token_2stage baseline --
    {
      const t = Date.now();
      const hits = vexus.tokenChamferTwoStage(intId, 100, TOP_N);
      const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.token_2stage.sum += ndcg;
      methods.token_2stage.count += 1;
      methods.token_2stage.totalMs += Date.now() - t;
      methods.token_2stage.perQuery.push(ndcg);
    }

    // -- density_v1_k3: doc 加权, K=3 --
    {
      const t = Date.now();
      const hits = vexus.tokenChamferTwoStageDensity(intId, 100, TOP_N, 3, false);
      const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.density_v1_k3.sum += ndcg;
      methods.density_v1_k3.count += 1;
      methods.density_v1_k3.totalMs += Date.now() - t;
      methods.density_v1_k3.perQuery.push(ndcg);
    }

    // -- density_v1_k5: doc 加权, K=5 --
    {
      const t = Date.now();
      const hits = vexus.tokenChamferTwoStageDensity(intId, 100, TOP_N, 5, false);
      const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.density_v1_k5.sum += ndcg;
      methods.density_v1_k5.count += 1;
      methods.density_v1_k5.totalMs += Date.now() - t;
      methods.density_v1_k5.perQuery.push(ndcg);
    }

    // -- density_v1_k7: doc 加权, K=7 --
    {
      const t = Date.now();
      const hits = vexus.tokenChamferTwoStageDensity(intId, 100, TOP_N, 7, false);
      const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.density_v1_k7.sum += ndcg;
      methods.density_v1_k7.count += 1;
      methods.density_v1_k7.totalMs += Date.now() - t;
      methods.density_v1_k7.perQuery.push(ndcg);
    }

    // -- density_v2_k5: 双向加权, K=5 --
    {
      const t = Date.now();
      const hits = vexus.tokenChamferTwoStageDensity(intId, 100, TOP_N, 5, true);
      const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.density_v2_k5.sum += ndcg;
      methods.density_v2_k5.count += 1;
      methods.density_v2_k5.totalMs += Date.now() - t;
      methods.density_v2_k5.perQuery.push(ndcg);
    }

    // 进度
    if ((qi + 1) % progressStep === 0 || qi === qids.length - 1) {
      const pct = ((qi + 1) / qids.length * 100).toFixed(0);
      process.stdout.write(`\r  Progress: ${qi + 1}/${qids.length} (${pct}%)`);
    }
  }

  console.log('\n');

  // ═══════════════════════════════════════════════════════════════
  // 7. 结果输出
  // ═══════════════════════════════════════════════════════════════
  const COSINE_BASE = 0.2195;
  const TOKEN_2STAGE_BASE = 0.3220;  // 已知基线

  console.log('=== RESULTS: Density-Weighted PQ-Chamfer Ablation ===\n');
  console.log('| Method | NDCG@10 | vs cosine | vs token_2stage | Avg latency |');
  console.log('|:--|:--:|:--:|:--:|:--:|');

  const resultRows = [];
  for (const [name, m] of Object.entries(methods)) {
    if (m.count === 0) continue;
    const ndcg = m.sum / m.count;
    const avgMs = Math.round(m.totalMs / m.count);
    const vsCosine = pctChange(ndcg, COSINE_BASE);
    const vsToken = pctChange(ndcg, TOKEN_2STAGE_BASE);
    console.log(`| ${name} | ${ndcg.toFixed(4)} | ${vsCosine} | ${vsToken} | ${avgMs}ms |`);
    resultRows.push({ name, ndcg, avgMs, vsCosine, vsToken });
  }

  // ═══════════════════════════════════════════════════════════════
  // 8. Paired t-test: density_v1_k5 vs token_2stage
  // ═══════════════════════════════════════════════════════════════
  console.log('\n=== Paired t-test: density_v1_k5 vs token_2stage ===\n');
  const a = methods.token_2stage.perQuery;
  const b = methods.density_v1_k5.perQuery;
  const n = Math.min(a.length, b.length);
  if (n > 1) {
    const diffs = [];
    for (let i = 0; i < n; i++) diffs.push(b[i] - a[i]);
    const meanDiff = diffs.reduce((s, x) => s + x, 0) / n;
    const varDiff = diffs.reduce((s, x) => s + (x - meanDiff) ** 2, 0) / (n - 1);
    const seDiff = Math.sqrt(varDiff / n);
    const tStat = seDiff > 0 ? meanDiff / seDiff : 0;
    console.log(`  n = ${n}`);
    console.log(`  mean_diff = ${meanDiff.toFixed(6)}`);
    console.log(`  t-statistic = ${tStat.toFixed(4)}`);
    console.log(`  (p < 0.05 if |t| > 1.97 for df=${n-1})`);

    // Wins / ties / losses
    let wins = 0, ties = 0, losses = 0;
    for (let i = 0; i < n; i++) {
      if (diffs[i] > 1e-8) wins++;
      else if (diffs[i] < -1e-8) losses++;
      else ties++;
    }
    console.log(`  density_v1_k5 wins/ties/losses = ${wins}/${ties}/${losses}`);
  }

  // 同样测 density_v2_k5
  console.log('\n=== Paired t-test: density_v2_k5 vs token_2stage ===\n');
  const c = methods.density_v2_k5.perQuery;
  if (c.length > 1) {
    const diffs2 = [];
    for (let i = 0; i < n; i++) diffs2.push(c[i] - a[i]);
    const meanDiff2 = diffs2.reduce((s, x) => s + x, 0) / n;
    const varDiff2 = diffs2.reduce((s, x) => s + (x - meanDiff2) ** 2, 0) / (n - 1);
    const seDiff2 = Math.sqrt(varDiff2 / n);
    const tStat2 = seDiff2 > 0 ? meanDiff2 / seDiff2 : 0;
    console.log(`  n = ${n}`);
    console.log(`  mean_diff = ${meanDiff2.toFixed(6)}`);
    console.log(`  t-statistic = ${tStat2.toFixed(4)}`);

    let wins2 = 0, ties2 = 0, losses2 = 0;
    for (let i = 0; i < n; i++) {
      if (diffs2[i] > 1e-8) wins2++;
      else if (diffs2[i] < -1e-8) losses2++;
      else ties2++;
    }
    console.log(`  density_v2_k5 wins/ties/losses = ${wins2}/${ties2}/${losses2}`);
  }

  // 保存结果
  const outputPath = path.join(DATA_DIR, 'density_ablation_results.json');
  fs.writeFileSync(outputPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    n_queries: qids.length,
    baselines: { cosine: COSINE_BASE, token_2stage: TOKEN_2STAGE_BASE },
    results: resultRows,
    per_query: {
      token_2stage: methods.token_2stage.perQuery,
      density_v1_k3: methods.density_v1_k3.perQuery,
      density_v1_k5: methods.density_v1_k5.perQuery,
      density_v1_k7: methods.density_v1_k7.perQuery,
      density_v2_k5: methods.density_v2_k5.perQuery,
    },
  }, null, 2));
  console.log(`\nResults saved to ${outputPath}`);
}

main().catch(e => { console.error(e); process.exit(1); });
