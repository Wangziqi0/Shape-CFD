#!/usr/bin/env node
'use strict';
/**
 * beir_pq_coarse_bench.js -- PQ 粗筛三方案对比实验 (NFCorpus)
 *
 * 对比：
 *   基线: cosine 质心粗筛 (200) -> token_pq_chamfer 精排 (55)
 *   方案1: PQ Hamming 粗筛 (200) -> token_pq_chamfer 精排 (55)
 *   方案2: 倒排索引粗筛 (200) -> token_pq_chamfer 精排 (55)
 *   方案3: ADC 全扫粗筛 (200) -> token_pq_chamfer 精排 (55)
 *
 * 指标：recall@200, NDCG@10, 粗筛延迟, 总延迟
 *
 * 用法：
 *   RAYON_NUM_THREADS=70 node --max-old-space-size=32768 beir_pq_coarse_bench.js
 *   MAX_Q=10 node --max-old-space-size=32768 beir_pq_coarse_bench.js  # 快速测试
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const COARSE_TOP = 200;
const TOP_N = 55;
const K = 10;

function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const arr = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(fp), crlfDelay: Infinity
    });
    rl.on('line', line => { if (line.trim()) try { arr.push(JSON.parse(line)); } catch (_) {} });
    rl.on('close', () => resolve(arr));
    rl.on('error', reject);
  });
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

async function main() {
  console.log('\n======================================================================');
  console.log('  PQ Coarse Filter Benchmark (NFCorpus)');
  console.log('  3 Schemes: Hamming / Inverted / ADC vs Cosine Centroid baseline');
  console.log('======================================================================\n');

  // 1. Load Rust addon
  const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
  const vexus = new LawVexus('/tmp/pq_coarse_bench');

  // 2. Load sentence clouds (for graph smoothing in fusion)
  let t0 = Date.now();
  process.stdout.write('Loading sentence clouds... ');
  const cloudInfo = vexus.loadClouds(CLOUDS_DB);
  console.log(`done (${Date.now() - t0}ms) ${cloudInfo}`);

  // 3. Load token clouds (auto builds inverted index + PQ store)
  t0 = Date.now();
  process.stdout.write('Loading token clouds + building index... ');
  const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log(`done (${Date.now() - t0}ms)`);
  console.log(`  ${tokenInfo}`);

  // 4. Load evaluation data
  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, intId] of Object.entries(idMap)) reverseMap[intId] = strId;

  const qrels = {};
  const qrelLines = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < qrelLines.length; i++) {
    const [qi, di, s] = qrelLines[i].split('\t');
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  // query vectors (for cosine baseline and lap)
  const queryVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl')))
    queryVecs[o._id] = new Float32Array(o.vector);

  // query id -> file_id mapping
  const queryIdToFileId = {};
  const queriesRaw = await loadJsonl(path.join(DATA_DIR, 'queries.jsonl'));
  for (let i = 0; i < queriesRaw.length; i++) queryIdToFileId[queriesRaw[i]._id] = i;

  let qids = Object.keys(qrels).filter(q => queryVecs[q]);
  const MQ = parseInt(process.env.MAX_Q || '0');
  if (MQ > 0) qids = qids.slice(0, MQ);
  console.log(`\n${qids.length} queries to evaluate\n`);

  // 5. Metrics accumulators
  const methods = {
    'baseline_centroid': { ndcg_sum: 0, n: 0, recall_sum: 0, coarse_ms: 0, total_ms: 0 },
    'scheme1_hamming':   { ndcg_sum: 0, n: 0, recall_sum: 0, coarse_ms: 0, total_ms: 0 },
    'scheme2_inverted':  { ndcg_sum: 0, n: 0, recall_sum: 0, coarse_ms: 0, total_ms: 0 },
    'scheme3_adc':       { ndcg_sum: 0, n: 0, recall_sum: 0, coarse_ms: 0, total_ms: 0 },
    // fusion variants
    'fus_baseline':      { ndcg_sum: 0, n: 0 },
    'fus_scheme1':       { ndcg_sum: 0, n: 0 },
    'fus_scheme2':       { ndcg_sum: 0, n: 0 },
    'fus_scheme3':       { ndcg_sum: 0, n: 0 },
    // cosine baseline
    'cosine':            { ndcg_sum: 0, n: 0 },
    'lap':               { ndcg_sum: 0, n: 0 },
  };

  const step = Math.max(1, Math.floor(qids.length / 10));
  const globalT0 = Date.now();

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qrel = qrels[qid];
    const qFileId = queryIdToFileId[qid];
    const qVec = queryVecs[qid];
    const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);

    // Ground truth relevant docs (relevance >= 1)
    const gtDocs = new Set(Object.keys(qrel).filter(d => qrel[d] >= 1));

    // --- Cosine baseline ---
    try {
      const h = vexus.cosineRank(qBuf, K);
      methods.cosine.ndcg_sum += computeNDCG(h.map(x => reverseMap[x.id]).filter(Boolean), qrel, K);
      methods.cosine.n++;
    } catch (e) { if (qi === 0) console.error('cosine:', e.message); }

    // --- Graph smoothing (lap) ---
    let lapScores = {};
    try {
      const h = vexus.shapeLaplacianPipeline(qBuf, K, TOP_N);
      methods.lap.ndcg_sum += computeNDCG(h.map(x => reverseMap[x.id]).filter(Boolean), qrel, K);
      methods.lap.n++;
      for (const x of h) lapScores[x.id] = x.score;
    } catch (e) { if (qi === 0) console.error('lap:', e.message); }

    // --- Recall measurement for all 4 coarse methods ---
    let recallResult;
    const recallT0 = Date.now();
    try {
      recallResult = vexus.coarseRecallCandidates(qFileId, COARSE_TOP, 1);
    } catch (e) {
      if (qi === 0) console.error('coarseRecall:', e.message);
      continue;
    }
    const recallMs = Date.now() - recallT0;

    // Compute recall@200 for each method
    const computeRecall = (ids) => {
      const idSet = new Set(ids.map(id => reverseMap[id]).filter(Boolean));
      let hits = 0;
      for (const d of gtDocs) {
        if (idSet.has(d)) hits++;
      }
      return gtDocs.size > 0 ? hits / gtDocs.size : 0;
    };

    methods.baseline_centroid.recall_sum += computeRecall(recallResult.centroidIds);
    methods.scheme2_inverted.recall_sum += computeRecall(recallResult.invertedIds);
    methods.scheme1_hamming.recall_sum += computeRecall(recallResult.hammingIds);
    methods.scheme3_adc.recall_sum += computeRecall(recallResult.adcIds);

    // --- Baseline: cosine centroid coarse -> f32 fine (token_2stage) ---
    {
      const t1 = Date.now();
      try {
        const h = vexus.tokenChamferTwoStage(qFileId, COARSE_TOP, TOP_N);
        const t2 = Date.now();
        const ranked = h.map(x => reverseMap[x[0]]).filter(Boolean);
        methods.baseline_centroid.ndcg_sum += computeNDCG(ranked, qrel, K);
        methods.baseline_centroid.n++;
        methods.baseline_centroid.total_ms += (t2 - t1);

        // fusion: 0.7*token + 0.3*lap (baseline version)
        const tokScores = {};
        for (const x of h) tokScores[x[0]] = x[1];
        const allIds = new Set([...Object.keys(lapScores), ...Object.keys(tokScores)]);
        const normTok = normalizeScores(tokScores);
        const normLap = normalizeScores(lapScores);
        const fused = [];
        for (const id of allIds) {
          const ts = normTok[id] || 0;
          const ls = normLap[id] || 0;
          fused.push({ id: parseInt(id), score: 0.55 * ts + 0.45 * ls });
        }
        fused.sort((a, b) => b.score - a.score);
        methods.fus_baseline.ndcg_sum += computeNDCG(fused.slice(0, K).map(x => reverseMap[x.id]).filter(Boolean), qrel, K);
        methods.fus_baseline.n++;
      } catch (e) { if (qi === 0) console.error('baseline:', e.message); }
    }

    // --- Scheme 1: PQ Hamming coarse -> f32 fine ---
    {
      const t1 = Date.now();
      try {
        const h = vexus.pqHammingTwoStage(qFileId, COARSE_TOP, TOP_N);
        const t2 = Date.now();
        const ranked = h.map(x => reverseMap[x[0]]).filter(Boolean);
        methods.scheme1_hamming.ndcg_sum += computeNDCG(ranked, qrel, K);
        methods.scheme1_hamming.n++;
        methods.scheme1_hamming.total_ms += (t2 - t1);

        // fusion
        const tokScores = {};
        for (const x of h) tokScores[x[0]] = x[1];
        const allIds = new Set([...Object.keys(lapScores), ...Object.keys(tokScores)]);
        const normTok = normalizeScores(tokScores);
        const normLap = normalizeScores(lapScores);
        const fused = [];
        for (const id of allIds) {
          fused.push({ id: parseInt(id), score: 0.55 * (normTok[id] || 0) + 0.45 * (normLap[id] || 0) });
        }
        fused.sort((a, b) => b.score - a.score);
        methods.fus_scheme1.ndcg_sum += computeNDCG(fused.slice(0, K).map(x => reverseMap[x.id]).filter(Boolean), qrel, K);
        methods.fus_scheme1.n++;
      } catch (e) { if (qi === 0) console.error('scheme1:', e.message); }
    }

    // --- Scheme 2: Inverted index coarse -> f32 fine ---
    {
      const t1 = Date.now();
      try {
        const h = vexus.tokenInvertedTwoStage(qFileId, COARSE_TOP, TOP_N, 1);
        const t2 = Date.now();
        const ranked = h.map(x => reverseMap[x[0]]).filter(Boolean);
        methods.scheme2_inverted.ndcg_sum += computeNDCG(ranked, qrel, K);
        methods.scheme2_inverted.n++;
        methods.scheme2_inverted.total_ms += (t2 - t1);

        // fusion
        const tokScores = {};
        for (const x of h) tokScores[x[0]] = x[1];
        const allIds = new Set([...Object.keys(lapScores), ...Object.keys(tokScores)]);
        const normTok = normalizeScores(tokScores);
        const normLap = normalizeScores(lapScores);
        const fused = [];
        for (const id of allIds) {
          fused.push({ id: parseInt(id), score: 0.55 * (normTok[id] || 0) + 0.45 * (normLap[id] || 0) });
        }
        fused.sort((a, b) => b.score - a.score);
        methods.fus_scheme2.ndcg_sum += computeNDCG(fused.slice(0, K).map(x => reverseMap[x.id]).filter(Boolean), qrel, K);
        methods.fus_scheme2.n++;
      } catch (e) { if (qi === 0) console.error('scheme2:', e.message); }
    }

    // --- Scheme 3: ADC coarse -> f32 fine ---
    {
      const t1 = Date.now();
      try {
        const h = vexus.adcCoarseF32Fine(qFileId, COARSE_TOP, TOP_N);
        const t2 = Date.now();
        const ranked = h.map(x => reverseMap[x[0]]).filter(Boolean);
        methods.scheme3_adc.ndcg_sum += computeNDCG(ranked, qrel, K);
        methods.scheme3_adc.n++;
        methods.scheme3_adc.total_ms += (t2 - t1);

        // fusion
        const tokScores = {};
        for (const x of h) tokScores[x[0]] = x[1];
        const allIds = new Set([...Object.keys(lapScores), ...Object.keys(tokScores)]);
        const normTok = normalizeScores(tokScores);
        const normLap = normalizeScores(lapScores);
        const fused = [];
        for (const id of allIds) {
          fused.push({ id: parseInt(id), score: 0.55 * (normTok[id] || 0) + 0.45 * (normLap[id] || 0) });
        }
        fused.sort((a, b) => b.score - a.score);
        methods.fus_scheme3.ndcg_sum += computeNDCG(fused.slice(0, K).map(x => reverseMap[x.id]).filter(Boolean), qrel, K);
        methods.fus_scheme3.n++;
      } catch (e) { if (qi === 0) console.error('scheme3:', e.message); }
    }

    // progress
    if ((qi + 1) % step === 0 || qi === qids.length - 1) {
      const el = (Date.now() - globalT0) / 1000;
      process.stdout.write(`\r  ${qi + 1}/${qids.length} (${((qi + 1) / el).toFixed(1)} q/s)    `);
    }
  }

  const totalElapsed = (Date.now() - globalT0) / 1000;
  console.log(`\n\n  Total: ${totalElapsed.toFixed(1)}s\n`);

  // 6. Print results
  const nQ = methods.baseline_centroid.n || 1;
  const cosNdcg = methods.cosine.n > 0 ? methods.cosine.ndcg_sum / methods.cosine.n : 0;

  console.log('='.repeat(90));
  console.log('  NFCorpus PQ Coarse Filter Comparison');
  console.log('='.repeat(90));
  console.log('');

  // Table 1: Coarse filter quality
  console.log('  Table 1: Coarse Filter Quality (recall@200)');
  console.log('-'.repeat(70));
  console.log('  Method'.padEnd(35) + 'recall@200'.padEnd(15) + 'vs centroid');
  console.log('-'.repeat(70));

  const centroidRecall = methods.baseline_centroid.recall_sum / nQ;
  const rows = [
    ['Cosine Centroid (baseline)', centroidRecall],
    ['Scheme 1: PQ Hamming', methods.scheme1_hamming.recall_sum / nQ],
    ['Scheme 2: Inverted Index', methods.scheme2_inverted.recall_sum / nQ],
    ['Scheme 3: ADC Full Scan', methods.scheme3_adc.recall_sum / nQ],
  ];
  for (const [name, recall] of rows) {
    const pct = name.includes('baseline') ? '--' : `${((recall - centroidRecall) / centroidRecall * 100).toFixed(1)}%`;
    console.log(`  ${name.padEnd(35)}${recall.toFixed(4).padEnd(15)}${pct}`);
  }
  console.log('');

  // Table 2: End-to-end NDCG@10
  console.log('  Table 2: End-to-End NDCG@10 (coarse@200 -> f32 fine@55)');
  console.log('-'.repeat(80));
  console.log('  Method'.padEnd(35) + 'NDCG@10'.padEnd(12) + 'vs cosine'.padEnd(14) + 'avg latency');
  console.log('-'.repeat(80));

  console.log(`  ${'Cosine baseline'.padEnd(35)}${cosNdcg.toFixed(4).padEnd(12)}${'--'.padEnd(14)}--`);
  if (methods.lap.n > 0) {
    const lapNdcg = methods.lap.ndcg_sum / methods.lap.n;
    console.log(`  ${'Graph Smoothing (lap)'.padEnd(35)}${lapNdcg.toFixed(4).padEnd(12)}${((lapNdcg - cosNdcg) / cosNdcg * 100).toFixed(1).padStart(6) + '%'.padEnd(8)}--`);
  }

  const e2eRows = [
    ['Centroid->f32 (baseline)', methods.baseline_centroid],
    ['Scheme 1: Hamming->f32', methods.scheme1_hamming],
    ['Scheme 2: Inverted->f32', methods.scheme2_inverted],
    ['Scheme 3: ADC->f32', methods.scheme3_adc],
  ];
  for (const [name, m] of e2eRows) {
    if (m.n === 0) continue;
    const ndcg = m.ndcg_sum / m.n;
    const pct = `${((ndcg - cosNdcg) / cosNdcg * 100).toFixed(1)}%`;
    const lat = `${(m.total_ms / m.n).toFixed(0)}ms`;
    console.log(`  ${name.padEnd(35)}${ndcg.toFixed(4).padEnd(12)}${pct.padStart(6).padEnd(14)}${lat}`);
  }
  console.log('');

  // Table 3: Fusion NDCG@10
  console.log('  Table 3: Fusion (0.55*token + 0.45*lap) NDCG@10');
  console.log('-'.repeat(70));
  console.log('  Method'.padEnd(35) + 'NDCG@10'.padEnd(12) + 'vs cosine');
  console.log('-'.repeat(70));

  const fusRows = [
    ['Centroid->f32 + lap (baseline)', methods.fus_baseline],
    ['Scheme 1: Hamming->f32 + lap', methods.fus_scheme1],
    ['Scheme 2: Inverted->f32 + lap', methods.fus_scheme2],
    ['Scheme 3: ADC->f32 + lap', methods.fus_scheme3],
  ];
  for (const [name, m] of fusRows) {
    if (m.n === 0) continue;
    const ndcg = m.ndcg_sum / m.n;
    const pct = `${((ndcg - cosNdcg) / cosNdcg * 100).toFixed(1)}%`;
    console.log(`  ${name.padEnd(35)}${ndcg.toFixed(4).padEnd(12)}${pct}`);
  }
  console.log('='.repeat(90));

  // 7. Save results
  const results = {
    dataset: 'nfcorpus',
    coarse_top: COARSE_TOP,
    top_n: TOP_N,
    n_queries: nQ,
    cosine_ndcg10: +cosNdcg.toFixed(4),
    methods: {}
  };
  for (const [name, m] of Object.entries(methods)) {
    if (m.n > 0) {
      results.methods[name] = {
        ndcg10: +(m.ndcg_sum / m.n).toFixed(4),
        ...(m.recall_sum !== undefined && m.n > 0 ? { recall200: +(m.recall_sum / m.n).toFixed(4) } : {}),
        ...(m.total_ms !== undefined && m.total_ms > 0 ? { avg_ms: +(m.total_ms / m.n).toFixed(1) } : {}),
      };
    }
  }
  const outPath = path.join(DATA_DIR, 'pq_coarse_bench_results.json');
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
  console.log(`\n  Results saved: ${outPath}\n`);
}

main().catch(e => { console.error(e); process.exit(1); });
