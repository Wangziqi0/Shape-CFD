#!/usr/bin/env node
'use strict';
/**
 * beir_multiprobe_bench.js -- V15 Speculative Multi-Probe Retrieval Benchmark (NFCorpus)
 *
 * 三路粗筛探测器消融实验：
 *   1. 单路基线（质心 only）
 *   2. 两路（质心 + max_token）
 *   3. 三路（质心 + max_token + PQ 倒排）
 *   4. 三种合并策略消融（RRF / max / hit）
 *   5. 候选集大小消融（200 / 300 / 500）
 *   6. 最优配置下跑 token_2stage 精排 + 融合
 *
 * 用法：
 *   node --max-old-space-size=16384 beir_multiprobe_bench.js
 *   MAX_Q=10 node --max-old-space-size=16384 beir_multiprobe_bench.js  # 快速测试
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ======================================================================
// 配置
// ======================================================================
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const K = 10;       // NDCG@K

// ======================================================================
// 工具函数
// ======================================================================

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

/** recall@K: 在候选列表中有多少比例的相关文档被捞到 */
function computeRecall(candidateIds, qrel) {
  const relevantDocs = Object.keys(qrel).filter(d => qrel[d] > 0);
  if (relevantDocs.length === 0) return 1.0;
  const candidateSet = new Set(candidateIds);
  let hits = 0;
  for (const d of relevantDocs) {
    if (candidateSet.has(d)) hits++;
  }
  return hits / relevantDocs.length;
}

/** 归一化分数到 [0,1] (min-max) */
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

/** 格式化时长 */
function fmtDur(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** 百分比变化 */
function pctChange(val, base) {
  if (base === 0) return 'N/A';
  const pct = (val - base) / base * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
}

// ======================================================================
// 主流程
// ======================================================================
async function main() {
  console.log('\n=== V15 Speculative Multi-Probe Retrieval Benchmark (NFCorpus) ===\n');

  // -- 1. 加载 Rust addon --
  let LawVexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
  } catch (e) {
    console.error('ERROR: 无法加载 law-vexus addon:', e.message);
    process.exit(1);
  }
  const vexus = new LawVexus('/tmp/beir_multiprobe_bench');

  // -- 2. 加载句子级点云 --
  let t0 = Date.now();
  process.stdout.write('Loading sentence clouds... ');
  const cloudCount = vexus.loadClouds(CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${cloudCount} docs`);

  // -- 3. 加载 token 级点云 --
  t0 = Date.now();
  process.stdout.write('Loading token clouds... ');
  const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${tokenInfo}`);

  // -- 4. 预计算 max_token_repr (Probe 2) --
  t0 = Date.now();
  process.stdout.write('Precomputing max_token_repr (top_k=3)... ');
  const reprCount = vexus.precomputeMaxTokenRepr(3);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${reprCount} docs`);

  // -- 5. 加载评估数据 --
  t0 = Date.now();
  process.stdout.write('Loading evaluation data... ');

  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, intId] of Object.entries(idMap)) {
    reverseMap[intId] = strId;
  }

  // corpus 向量（用于 cosine baseline）
  const corpusVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    corpusVecs[o._id] = new Float32Array(o.vector);
  }

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

  // 有标注的 query 列表
  let qids = Object.keys(qrels).filter(q => queryVecs[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  // query string ID -> internal int ID
  const queryIdMap = {};
  const qvLines = fs.readFileSync(path.join(DATA_DIR, 'query_vectors.jsonl'), 'utf8').trim().split('\n');
  for (let i = 0; i < qvLines.length; i++) {
    try {
      const obj = JSON.parse(qvLines[i]);
      const qid = obj._id || obj.id;
      if (qid) queryIdMap[qid] = i;
    } catch (e) {}
  }

  console.log(`done (${fmtDur(Date.now() - t0)}, ${qids.length} queries, ${Object.keys(corpusVecs).length} docs)`);

  // ======================================================================
  // 6. 定义实验配置
  // ======================================================================

  // 所有实验方法
  const methods = {};
  function defMethod(name) {
    methods[name] = { ndcgSum: 0, ndcgCount: 0, recallSum: 0, recallCount: 0, totalMs: 0 };
  }

  // 基线
  defMethod('centroid_only_200');    // 单路质心 recall@200 基线
  defMethod('token_2stage_100');     // 现有最优 token 管线

  // 两路消融（质心 + max_token），不同合并策略
  defMethod('2probe_rrf_200');
  defMethod('2probe_max_200');
  defMethod('2probe_hit_200');

  // 三路消融（质心 + max_token + 倒排），RRF
  defMethod('3probe_rrf_200');
  defMethod('3probe_rrf_300');
  defMethod('3probe_rrf_500');

  // 三路消融，不同合并策略
  defMethod('3probe_max_200');
  defMethod('3probe_hit_200');

  // 最优多路 + 精排 (NDCG)
  defMethod('mp_2stage_rrf_200');
  defMethod('mp_2stage_rrf_300');
  defMethod('mp_2stage_rrf_500');

  // 最优多路 + 融合
  defMethod('mp_fusion_07');

  // 各路探测器单独 recall
  defMethod('probe1_centroid_200');
  defMethod('probe2_maxtoken_200');
  defMethod('probe3_inverted_200');

  console.log(`\nRunning benchmark (${qids.length} queries)...\n`);

  const progressStep = Math.max(1, Math.floor(qids.length / 20));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = queryVecs[qid];
    const qrel = qrels[qid];
    const intId = queryIdMap[qid];

    if (intId === undefined) continue;

    // ---- 各路探测器 recall 统计 ----
    let recallResult;
    try {
      recallResult = vexus.multiProbeRecall(intId, 200, 200, 'rrf', 1);
    } catch (e) {
      if (qi === 0) console.error(`  WARN multiProbeRecall error: ${e.message}`);
      continue;
    }

    // 各路单独 recall@200
    {
      const p1Ids = recallResult.probe1Ids.map(id => reverseMap[id]).filter(Boolean);
      const p2Ids = recallResult.probe2Ids.map(id => reverseMap[id]).filter(Boolean);
      const p3Ids = recallResult.probe3Ids.map(id => reverseMap[id]).filter(Boolean);

      const r1 = computeRecall(p1Ids, qrel);
      const r2 = computeRecall(p2Ids, qrel);
      const r3 = computeRecall(p3Ids, qrel);

      methods.probe1_centroid_200.recallSum += r1;
      methods.probe1_centroid_200.recallCount += 1;
      methods.probe2_maxtoken_200.recallSum += r2;
      methods.probe2_maxtoken_200.recallCount += 1;
      methods.probe3_inverted_200.recallSum += r3;
      methods.probe3_inverted_200.recallCount += 1;
    }

    // 基线: 单路质心 recall@200
    {
      const ids = recallResult.probe1Ids.map(id => reverseMap[id]).filter(Boolean);
      const r = computeRecall(ids, qrel);
      methods.centroid_only_200.recallSum += r;
      methods.centroid_only_200.recallCount += 1;
    }

    // ---- 两路消融（质心 + max_token）----
    for (const [strat, mname] of [['rrf', '2probe_rrf_200'], ['max', '2probe_max_200'], ['hit', '2probe_hit_200']]) {
      const t = Date.now();
      try {
        const result = vexus.multiProbeRetrieve(intId, 200, 200, strat, 1, false);
        const ids = result.map(h => reverseMap[h[0]]).filter(Boolean);
        const r = computeRecall(ids, qrel);
        methods[mname].recallSum += r;
        methods[mname].recallCount += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN ${mname} error: ${e.message}`);
      }
      methods[mname].totalMs += Date.now() - t;
    }

    // ---- 三路消融（+ 倒排），不同候选集大小 ----
    for (const [mergedTop, mname] of [[200, '3probe_rrf_200'], [300, '3probe_rrf_300'], [500, '3probe_rrf_500']]) {
      const t = Date.now();
      try {
        const result = vexus.multiProbeRetrieve(intId, 200, mergedTop, 'rrf', 1, true);
        const ids = result.map(h => reverseMap[h[0]]).filter(Boolean);
        const r = computeRecall(ids, qrel);
        methods[mname].recallSum += r;
        methods[mname].recallCount += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN ${mname} error: ${e.message}`);
      }
      methods[mname].totalMs += Date.now() - t;
    }

    // 三路其他策略
    for (const [strat, mname] of [['max', '3probe_max_200'], ['hit', '3probe_hit_200']]) {
      const t = Date.now();
      try {
        const result = vexus.multiProbeRetrieve(intId, 200, 200, strat, 1, true);
        const ids = result.map(h => reverseMap[h[0]]).filter(Boolean);
        const r = computeRecall(ids, qrel);
        methods[mname].recallSum += r;
        methods[mname].recallCount += 1;
      } catch (e) {
        if (qi === 0) console.error(`  WARN ${mname} error: ${e.message}`);
      }
      methods[mname].totalMs += Date.now() - t;
    }

    // ---- 多路 + token Chamfer 精排 (NDCG@10) ----
    let mpBestScoreMap = null;
    for (const [mergedTop, perProbe, mname] of [
      [200, 200, 'mp_2stage_rrf_200'],
      [300, 200, 'mp_2stage_rrf_300'],
      [500, 200, 'mp_2stage_rrf_500'],
    ]) {
      const t = Date.now();
      try {
        const hits = vexus.multiProbeTwoStage(intId, mergedTop, 55, 'rrf', 1, perProbe, true);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods[mname].ndcgSum += ndcg;
        methods[mname].ndcgCount += 1;

        // 保存最优配置(200)的分数用于融合
        if (mname === 'mp_2stage_rrf_200') {
          mpBestScoreMap = {};
          for (const h of hits) {
            const strId = reverseMap[h[0]];
            if (strId) mpBestScoreMap[strId] = h[1];
          }
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN ${mname} error: ${e.message}`);
      }
      methods[mname].totalMs += Date.now() - t;
    }

    // ---- 基线 token_2stage_100 ----
    let tokenScoreMap = null;
    {
      const t = Date.now();
      try {
        const hits = vexus.tokenChamferTwoStage(intId, 100, 55);
        const ranked = hits.map(h => reverseMap[h[0]]).filter(Boolean);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.token_2stage_100.ndcgSum += ndcg;
        methods.token_2stage_100.ndcgCount += 1;
        tokenScoreMap = {};
        for (const h of hits) {
          const strId = reverseMap[h[0]];
          if (strId) tokenScoreMap[strId] = h[1];
        }
      } catch (e) {
        if (qi === 0) console.error(`  WARN token_2stage_100 error: ${e.message}`);
      }
      methods.token_2stage_100.totalMs += Date.now() - t;
    }

    // ---- 多路融合: 0.7 * mp_2stage + 0.3 * shape_cfd ----
    if (mpBestScoreMap) {
      const t = Date.now();
      try {
        const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
        const cfdHits = vexus.shapeLaplacianPipeline(qBuf, 55, 55, 0.02, 20);
        const cfdScoreMap = {};
        for (const h of cfdHits) {
          const strId = reverseMap[h.id];
          if (strId) cfdScoreMap[strId] = h.score;
        }

        const allIds = new Set([...Object.keys(mpBestScoreMap), ...Object.keys(cfdScoreMap)]);
        const normMp = normalizeScores(mpBestScoreMap);
        const normCfd = normalizeScores(cfdScoreMap);

        const fusionScores = [];
        for (const did of allIds) {
          const tScore = normMp[did] || 0;
          const cScore = normCfd[did] || 0;
          fusionScores.push({ did, s: 0.7 * tScore + 0.3 * cScore });
        }
        fusionScores.sort((a, b) => b.s - a.s);
        const ranked = fusionScores.slice(0, K).map(x => x.did);
        const ndcg = computeNDCG(ranked, qrel, K);
        methods.mp_fusion_07.ndcgSum += ndcg;
        methods.mp_fusion_07.ndcgCount += 1;
        methods.mp_fusion_07.totalMs += Date.now() - t;
      } catch (e) {
        if (qi === 0) console.error(`  WARN mp_fusion_07 error: ${e.message}`);
      }
    }

    // 进度显示
    if ((qi + 1) % progressStep === 0 || qi === qids.length - 1) {
      process.stdout.write(`\r  progress: ${qi + 1}/${qids.length} (${((qi + 1) / qids.length * 100).toFixed(0)}%)`);
    }
  }

  console.log('\n');

  // ======================================================================
  // 7. 报告结果
  // ======================================================================
  const baseRecall = methods.centroid_only_200.recallCount > 0
    ? methods.centroid_only_200.recallSum / methods.centroid_only_200.recallCount
    : 0;
  const baseNdcg = methods.token_2stage_100.ndcgCount > 0
    ? methods.token_2stage_100.ndcgSum / methods.token_2stage_100.ndcgCount
    : 0;

  console.log('='.repeat(80));
  console.log('  V15 Multi-Probe Retrieval Results (NFCorpus)');
  console.log('='.repeat(80));

  console.log('\n--- Recall@200 (各路探测器单独) ---');
  for (const name of ['probe1_centroid_200', 'probe2_maxtoken_200', 'probe3_inverted_200']) {
    const m = methods[name];
    if (m.recallCount > 0) {
      const r = m.recallSum / m.recallCount;
      console.log(`  ${name.padEnd(28)} recall@200 = ${r.toFixed(4)} (${pctChange(r, baseRecall)} vs centroid)`);
    }
  }

  console.log('\n--- Recall@200 (合并策略消融) ---');
  console.log(`  ${'centroid_only_200'.padEnd(28)} recall@200 = ${baseRecall.toFixed(4)} (BASELINE)`);
  for (const name of [
    '2probe_rrf_200', '2probe_max_200', '2probe_hit_200',
    '3probe_rrf_200', '3probe_max_200', '3probe_hit_200',
  ]) {
    const m = methods[name];
    if (m.recallCount > 0) {
      const r = m.recallSum / m.recallCount;
      const ms = m.totalMs / m.recallCount;
      console.log(`  ${name.padEnd(28)} recall@200 = ${r.toFixed(4)} (${pctChange(r, baseRecall)}) ${ms.toFixed(1)}ms/q`);
    }
  }

  console.log('\n--- Recall@N (候选集大小消融, 3-probe RRF) ---');
  for (const name of ['3probe_rrf_200', '3probe_rrf_300', '3probe_rrf_500']) {
    const m = methods[name];
    if (m.recallCount > 0) {
      const r = m.recallSum / m.recallCount;
      console.log(`  ${name.padEnd(28)} recall = ${r.toFixed(4)} (${pctChange(r, baseRecall)})`);
    }
  }

  console.log('\n--- NDCG@10 (精排管线) ---');
  console.log(`  ${'token_2stage_100 (baseline)'.padEnd(38)} NDCG@10 = ${baseNdcg.toFixed(4)}`);
  for (const name of [
    'mp_2stage_rrf_200', 'mp_2stage_rrf_300', 'mp_2stage_rrf_500',
    'mp_fusion_07',
  ]) {
    const m = methods[name];
    if (m.ndcgCount > 0) {
      const ndcg = m.ndcgSum / m.ndcgCount;
      const ms = m.totalMs / m.ndcgCount;
      console.log(`  ${name.padEnd(38)} NDCG@10 = ${ndcg.toFixed(4)} (${pctChange(ndcg, baseNdcg)}) ${ms.toFixed(1)}ms/q`);
    }
  }

  console.log('\n--- 延迟统计 ---');
  for (const [name, m] of Object.entries(methods)) {
    const count = m.ndcgCount || m.recallCount;
    if (count > 0 && m.totalMs > 0) {
      console.log(`  ${name.padEnd(28)} ${(m.totalMs / count).toFixed(1)} ms/query`);
    }
  }

  // 判断标准
  console.log('\n--- 关键判断 ---');
  const bestRecallName = ['3probe_rrf_200', '3probe_rrf_300', '3probe_rrf_500',
    '2probe_rrf_200', '3probe_max_200', '3probe_hit_200']
    .filter(n => methods[n].recallCount > 0)
    .sort((a, b) => {
      const ra = methods[a].recallSum / methods[a].recallCount;
      const rb = methods[b].recallSum / methods[b].recallCount;
      return rb - ra;
    })[0];

  if (bestRecallName) {
    const bestRecall = methods[bestRecallName].recallSum / methods[bestRecallName].recallCount;
    const recallDelta = (bestRecall - baseRecall) / baseRecall * 100;
    console.log(`  最佳 recall@200: ${bestRecallName} = ${bestRecall.toFixed(4)} (${pctChange(bestRecall, baseRecall)})`);
    console.log(`  recall 提升 ${recallDelta.toFixed(1)}% ${recallDelta > 5 ? '> 5% -- 有效!' : '< 5% -- 不显著'}`);
  }

  const mpFusion = methods.mp_fusion_07;
  if (mpFusion.ndcgCount > 0) {
    const ndcg = mpFusion.ndcgSum / mpFusion.ndcgCount;
    console.log(`  最优融合 NDCG@10: ${ndcg.toFixed(4)} ${ndcg > 0.3271 ? '> 0.3271 -- 超越全场最优!' : '<= 0.3271 -- 未超越'}`);
  }

  console.log('\n=== Benchmark 完成 ===\n');
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
