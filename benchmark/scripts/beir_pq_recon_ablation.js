#!/usr/bin/env node
'use strict';
/**
 * beir_pq_recon_ablation.js -- PQ 码重建向量 vs 原始 f32 向量的精排质量消融实验
 *
 * 实验目的：
 *   验证 PQ 码本重建的近似向量（每个 64d 子空间替换为最近质心）
 *   替代原始 f32 向量后，PQ-Chamfer NDCG@10 会掉多少。
 *
 * 两个方法：
 *   token_2stage_100  -- 原始 f32 向量精排（baseline）
 *   pq_recon_2stage   -- PQ 码本重建向量精排（消融）
 *
 * 用法：
 *   node beir_pq_recon_ablation.js
 *   MAX_Q=10 node beir_pq_recon_ablation.js   # 快速测试
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
const COARSE_TOP = 100;

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
  return `${(ms / 1000).toFixed(1)}s`;
}

function pctChange(val, base) {
  if (base === 0) return 'N/A';
  const pct = (val - base) / base * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
}

// ═══════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════
async function main() {
  console.log('\n=== PQ Reconstruction Ablation Experiment (NFCorpus) ===\n');
  console.log('Purpose: Compare original f32 vectors vs PQ-reconstructed vectors');
  console.log('         in token_chamfer_two_stage fine-ranking quality.\n');

  // 1. 加载 Rust addon
  let LawVexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
  } catch (e) {
    console.error('ERROR: cannot load law-vexus addon:', e.message);
    process.exit(1);
  }
  const vexus = new LawVexus('/tmp/beir_pq_recon');

  // 2. 加载句子级点云（shape_cfd 需要）
  let t0 = Date.now();
  process.stdout.write('Loading sentence clouds... ');
  const cloudInfo = vexus.loadClouds(CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)})`);

  // 3. 加载 token 级点云
  t0 = Date.now();
  process.stdout.write('Loading token clouds (this takes ~2 min)... ');
  const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)})`);

  // 4. 检查 PQ 重建函数是否可用
  const hasPqRecon = typeof vexus.tokenChamferTwoStagePqRecon === 'function';
  if (!hasPqRecon) {
    console.error('ERROR: tokenChamferTwoStagePqRecon not available in addon.');
    console.error('Please rebuild law-vexus with the new function.');
    process.exit(1);
  }
  console.log('  [v] tokenChamferTwoStagePqRecon available\n');

  // 5. 加载评估数据
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

  console.log(`done (${qids.length} queries, ${allDids.length} docs)\n`);

  // ═══════════════════════════════════════════════════════════════
  // 6. 跑消融实验
  // ═══════════════════════════════════════════════════════════════
  console.log(`Running ablation (${qids.length} queries)...\n`);

  const methods = {
    cosine:          { sum: 0, count: 0, totalMs: 0, perQuery: [] },
    token_2stage:    { sum: 0, count: 0, totalMs: 0, perQuery: [] },
    pq_recon_2stage: { sum: 0, count: 0, totalMs: 0, perQuery: [] },
  };

  const progressStep = Math.max(1, Math.floor(qids.length / 20));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = queryVecs[qid];
    const qrel = qrels[qid];
    const intId = queryIdMap[qid];

    // cosine baseline
    {
      const t = Date.now();
      const scores = allDids.map(did => ({ did, s: cosSim(qVec, corpusVecs[did]) }));
      scores.sort((a, b) => b.s - a.s);
      const ranked = scores.slice(0, K).map(x => x.did);
      const ndcg = computeNDCG(ranked, qrel, K);
      methods.cosine.sum += ndcg;
      methods.cosine.count += 1;
      methods.cosine.totalMs += Date.now() - t;
      methods.cosine.perQuery.push(ndcg);
    }

    // token_2stage (original f32)
    if (intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferTwoStage(intId, COARSE_TOP, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.token_2stage.sum += ndcg;
        methods.token_2stage.count += 1;
        methods.token_2stage.perQuery.push(ndcg);
      } catch (e) {
        if (qi === 0) console.error(`  WARN token_2stage error: ${e.message}`);
        methods.token_2stage.perQuery.push(0);
      }
      methods.token_2stage.totalMs += Date.now() - t;
    }

    // pq_recon_2stage (PQ reconstructed vectors)
    if (intId !== undefined) {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferTwoStagePqRecon(intId, COARSE_TOP, TOP_N);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.pq_recon_2stage.sum += ndcg;
        methods.pq_recon_2stage.count += 1;
        methods.pq_recon_2stage.perQuery.push(ndcg);
      } catch (e) {
        if (qi === 0) console.error(`  WARN pq_recon error: ${e.message}`);
        methods.pq_recon_2stage.perQuery.push(0);
      }
      methods.pq_recon_2stage.totalMs += Date.now() - t;
    }

    // 进度
    if ((qi + 1) % progressStep === 0 || qi === qids.length - 1) {
      const pct = ((qi + 1) / qids.length * 100).toFixed(0);
      const cosAvg = methods.cosine.count > 0
        ? (methods.cosine.sum / methods.cosine.count).toFixed(4) : '---';
      const f32Avg = methods.token_2stage.count > 0
        ? (methods.token_2stage.sum / methods.token_2stage.count).toFixed(4) : '---';
      const pqAvg = methods.pq_recon_2stage.count > 0
        ? (methods.pq_recon_2stage.sum / methods.pq_recon_2stage.count).toFixed(4) : '---';
      process.stdout.write(
        `\r  [${pct}%] ${qi+1}/${qids.length}  cos=${cosAvg}  f32=${f32Avg}  pq_recon=${pqAvg}`
      );
    }
  }

  console.log('\n');

  // ═══════════════════════════════════════════════════════════════
  // 7. 输出结果
  // ═══════════════════════════════════════════════════════════════
  console.log('=== Results ===\n');

  const cosNdcg = methods.cosine.count > 0 ? methods.cosine.sum / methods.cosine.count : 0;
  const f32Ndcg = methods.token_2stage.count > 0 ? methods.token_2stage.sum / methods.token_2stage.count : 0;
  const pqNdcg = methods.pq_recon_2stage.count > 0 ? methods.pq_recon_2stage.sum / methods.pq_recon_2stage.count : 0;

  const f32AvgMs = methods.token_2stage.count > 0 ? methods.token_2stage.totalMs / methods.token_2stage.count : 0;
  const pqAvgMs = methods.pq_recon_2stage.count > 0 ? methods.pq_recon_2stage.totalMs / methods.pq_recon_2stage.count : 0;

  console.log('| Method             | NDCG@10 | vs cosine     | vs f32        | Avg latency |');
  console.log('|:-------------------|:-------:|:-------------:|:-------------:|:-----------:|');
  console.log(`| cosine (baseline)  | ${cosNdcg.toFixed(4)}  | --            | --            | ${f32AvgMs.toFixed(0)}ms         |`);
  console.log(`| token_2stage (f32) | ${f32Ndcg.toFixed(4)}  | ${pctChange(f32Ndcg, cosNdcg).padEnd(13)} | --            | ${f32AvgMs.toFixed(0)}ms         |`);
  console.log(`| pq_recon_2stage    | ${pqNdcg.toFixed(4)}  | ${pctChange(pqNdcg, cosNdcg).padEnd(13)} | ${pctChange(pqNdcg, f32Ndcg).padEnd(13)} | ${pqAvgMs.toFixed(0)}ms         |`);

  console.log('\n--- Per-query analysis ---\n');

  // 计算 per-query 差异统计
  const diffs = [];
  const n = Math.min(methods.token_2stage.perQuery.length, methods.pq_recon_2stage.perQuery.length);
  let wins = 0, ties = 0, losses = 0;
  for (let i = 0; i < n; i++) {
    const d = methods.pq_recon_2stage.perQuery[i] - methods.token_2stage.perQuery[i];
    diffs.push(d);
    if (Math.abs(d) < 1e-8) ties++;
    else if (d > 0) wins++;
    else losses++;
  }

  const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const stdDiff = Math.sqrt(diffs.reduce((a, b) => a + (b - meanDiff) ** 2, 0) / diffs.length);
  const maxDrop = Math.min(...diffs);
  const maxGain = Math.max(...diffs);

  console.log(`Total queries: ${n}`);
  console.log(`PQ recon wins/ties/losses: ${wins}/${ties}/${losses}`);
  console.log(`Mean per-query diff (pq - f32): ${meanDiff.toFixed(6)}`);
  console.log(`Std dev of diff: ${stdDiff.toFixed(6)}`);
  console.log(`Max drop: ${maxDrop.toFixed(6)}`);
  console.log(`Max gain: ${maxGain.toFixed(6)}`);

  // Paired t-test (approximate)
  if (n > 1 && stdDiff > 0) {
    const tStat = meanDiff / (stdDiff / Math.sqrt(n));
    console.log(`\nPaired t-statistic: ${tStat.toFixed(4)}`);
    console.log(`(|t| > 1.96 => significant at p < 0.05 for large n)`);
  }

  // Bootstrap significance test (10000 iterations)
  console.log('\n--- Bootstrap significance test (10000 iterations) ---\n');
  const B = 10000;
  let countGreater = 0;
  for (let b = 0; b < B; b++) {
    let sumDiff = 0;
    for (let i = 0; i < n; i++) {
      const j = Math.floor(Math.random() * n);
      sumDiff += diffs[j];
    }
    if (sumDiff / n >= 0) countGreater++;
  }
  const pValue = 1 - countGreater / B;
  console.log(`H0: pq_recon >= f32 (no quality loss)`);
  console.log(`p-value (one-sided, pq_recon < f32): ${pValue.toFixed(4)}`);
  console.log(`Significant quality loss? ${pValue < 0.05 ? 'YES (p < 0.05)' : 'NO (p >= 0.05)'}`);

  // Save results
  const resultsPath = path.join(DATA_DIR, 'pq_recon_ablation_results.json');
  fs.writeFileSync(resultsPath, JSON.stringify({
    experiment: 'PQ reconstruction ablation',
    date: new Date().toISOString(),
    n_queries: n,
    cosine_ndcg10: cosNdcg,
    f32_ndcg10: f32Ndcg,
    pq_recon_ndcg10: pqNdcg,
    diff_pct: ((pqNdcg - f32Ndcg) / f32Ndcg * 100),
    mean_per_query_diff: meanDiff,
    std_diff: stdDiff,
    wins, ties, losses,
    p_value: pValue,
    f32_avg_ms: f32AvgMs,
    pq_recon_avg_ms: pqAvgMs,
    per_query_f32: methods.token_2stage.perQuery,
    per_query_pq_recon: methods.pq_recon_2stage.perQuery,
  }, null, 2));
  console.log(`\nResults saved to: ${resultsPath}`);

  console.log('\n=== Experiment Complete ===\n');
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
