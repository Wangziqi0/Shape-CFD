#!/usr/bin/env node
'use strict';
/**
 * beir_fid_ablation.js -- Diagonal FID Coarse Screening Ablation
 *
 * Tests whether using variance (diagonal FID) in coarse screening
 * improves recall and NDCG compared to pure centroid cosine.
 *
 * Distance: d_FID(Q, D) = ||mu_Q - mu_D||^2 + alpha * ||sigma_Q - sigma_D||^2
 *
 * Pipeline: FID coarse top-200 -> Rust PQ-Chamfer fine rank -> (optional) graph smoothing fusion
 *
 * alpha sweep: 0.0 (pure centroid = baseline), 0.1, 0.5, 1.0, 2.0
 * Also tests: doc-only variance (query variance ignored), sqrt(var) variant
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ============================================================
// Config
// ============================================================
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const DOC_VAR_PATH = path.join(DATA_DIR, 'doc_variances.jsonl');
const QUERY_VAR_PATH = path.join(DATA_DIR, 'query_variances.jsonl');

const COARSE_TOP = 200;  // coarse screening pool size
const K = 10;            // NDCG@K
const TOP_N = 55;        // fine rank output size

const ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0];

// ============================================================
// Utilities
// ============================================================

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

function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

/** Squared L2 distance between two vectors */
function sqL2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

/** FID distance (normalized):
 *  d = ||mu_Q - mu_D||^2 + alpha * normFactor * ||sigma_Q - sigma_D||^2
 *  normFactor scales variance term to same magnitude as centroid term */
function fidDistance(muQ, muD, varQ, varD, alpha, useSqrt, normFactor) {
  let meanDist = sqL2(muQ, muD);
  if (alpha === 0) return meanDist;

  let varDist = 0;
  if (useSqrt) {
    // Use std dev (sqrt of variance)
    for (let i = 0; i < varQ.length; i++) {
      const d = Math.sqrt(Math.abs(varQ[i])) - Math.sqrt(Math.abs(varD[i]));
      varDist += d * d;
    }
  } else {
    // Use variance directly
    varDist = sqL2(varQ, varD);
  }

  return meanDist + alpha * normFactor * varDist;
}

/** FID distance (doc-only variance): use std(D) - std(Q) per dimension
 *  Normalized version */
function fidDistanceDocOnly(muQ, muD, varQ, varD, alpha, normFactor) {
  let meanDist = sqL2(muQ, muD);
  if (alpha === 0) return meanDist;

  // Per-dimension: (sqrt(varD_i) - sqrt(varQ_i))^2
  let varDist = 0;
  for (let i = 0; i < varD.length; i++) {
    const d = Math.sqrt(Math.abs(varD[i])) - Math.sqrt(Math.abs(varQ[i]));
    varDist += d * d;
  }
  return meanDist + alpha * normFactor * varDist;
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

function computeRecall(candidates, qrel, atN) {
  const relevant = new Set(Object.entries(qrel).filter(([_, s]) => s > 0).map(([id, _]) => id));
  if (relevant.size === 0) return 1.0;
  let found = 0;
  for (let i = 0; i < Math.min(candidates.length, atN); i++) {
    if (relevant.has(candidates[i])) found++;
  }
  return found / relevant.size;
}

/** Normalize scores to [0, 1] (min-max) */
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

function fmtDur(ms) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function pctChange(val, base) {
  if (base === 0) return 'N/A';
  const pct = (val - base) / base * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
}

// ============================================================
// Main
// ============================================================
async function main() {
  console.log('\n=== Diagonal FID Coarse Screening Ablation (NFCorpus) ===\n');

  // 1. Load Rust addon
  let LawVexus;
  try {
    ({ LawVexus } = require('/home/amd/HEZIMENG/law-vexus'));
  } catch (e) {
    console.error('ERROR: cannot load law-vexus:', e.message);
    process.exit(1);
  }
  const vexus = new LawVexus('/tmp/beir_fid_ablation');

  // 2. Load sentence clouds (needed for graph smoothing fusion)
  let t0 = Date.now();
  process.stdout.write('Loading sentence clouds... ');
  const cloudInfo = vexus.loadClouds(CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${cloudInfo}`);

  // 3. Load token clouds (needed for PQ-Chamfer rerank)
  t0 = Date.now();
  process.stdout.write('Loading token clouds... ');
  const tokenInfo = vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log(`done (${fmtDur(Date.now() - t0)}) ${tokenInfo}`);

  // 4. Load evaluation data
  t0 = Date.now();
  process.stdout.write('Loading evaluation data... ');

  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, intId] of Object.entries(idMap)) {
    reverseMap[intId] = strId;
  }

  // Corpus centroid vectors: _id -> Float32Array
  const corpusMu = {};
  const corpusIntIds = {};  // _id -> intId
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    corpusMu[o._id] = new Float32Array(o.vector);
    corpusIntIds[o._id] = idMap[o._id];
  }
  const allDids = Object.keys(corpusMu);

  // Query centroid vectors
  const queryMu = {};
  const queryLineMap = {};  // _id -> line index (= file_id in sqlite)
  const queryLines = fs.readFileSync(path.join(DATA_DIR, 'query_vectors.jsonl'), 'utf8').trim().split('\n');
  for (let i = 0; i < queryLines.length; i++) {
    const obj = JSON.parse(queryLines[i]);
    queryMu[obj._id] = new Float32Array(obj.vector);
    queryLineMap[obj._id] = i;
  }

  // Qrels
  const qrels = {};
  const qrelLines = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < qrelLines.length; i++) {
    const [qi, di, s] = qrelLines[i].split('\t');
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  let qids = Object.keys(qrels).filter(q => queryMu[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  // 5. Load variance vectors
  process.stdout.write('Loading variance vectors... ');

  // Doc variances: intId -> Float64Array
  const docVar = {};
  for (const o of await loadJsonl(DOC_VAR_PATH)) {
    docVar[o.file_id] = new Float64Array(o.variance);
  }

  // Query variances: file_id -> Float64Array
  const queryVar = {};
  for (const o of await loadJsonl(QUERY_VAR_PATH)) {
    queryVar[o.file_id] = new Float64Array(o.variance);
  }

  console.log(`done (${fmtDur(Date.now() - t0)}, ${qids.length} queries, ${allDids.length} docs, ${Object.keys(docVar).length} doc vars, ${Object.keys(queryVar).length} query vars)`);

  // 6. Map doc string IDs to arrays for fast iteration
  const docEntries = allDids.map(did => ({
    strId: did,
    intId: idMap[did],
    mu: corpusMu[did],
    variance: docVar[idMap[did]] || new Float64Array(4096),
  }));

  // ============================================================
  // 7. Compute normalization factors
  // ============================================================
  console.log('Computing normalization factors...');
  {
    // Sample 50 queries x 100 docs to estimate mean centroid dist and mean variance dist
    const sampleQ = qids.slice(0, Math.min(50, qids.length));
    const sampleD = docEntries.slice(0, Math.min(100, docEntries.length));
    let sumMuDist = 0, sumVarDist = 0, sumSqrtVarDist = 0, count = 0;
    for (const qid of sampleQ) {
      const qMu = queryMu[qid];
      const qFileId = queryLineMap[qid];
      const qVar = queryVar[qFileId] || new Float64Array(4096);
      for (const d of sampleD) {
        sumMuDist += sqL2(qMu, d.mu);
        sumVarDist += sqL2(qVar, d.variance);
        let sv = 0;
        for (let i = 0; i < 4096; i++) {
          const dd = Math.sqrt(Math.abs(qVar[i])) - Math.sqrt(Math.abs(d.variance[i]));
          sv += dd * dd;
        }
        sumSqrtVarDist += sv;
        count++;
      }
    }
    const avgMu = sumMuDist / count;
    const avgVar = sumVarDist / count;
    const avgSqrtVar = sumSqrtVarDist / count;
    // normFactor makes variance term same scale as centroid term
    // so alpha=1 means equal weight
    var normFactorVar = avgMu / (avgVar || 1);
    var normFactorSqrt = avgMu / (avgSqrtVar || 1);
    console.log(`  avg ||mu_Q - mu_D||^2 = ${avgMu.toFixed(4)}`);
    console.log(`  avg ||var_Q - var_D||^2 = ${avgVar.toFixed(2)}`);
    console.log(`  avg ||std_Q - std_D||^2 = ${avgSqrtVar.toFixed(4)}`);
    console.log(`  normFactor (var) = ${normFactorVar.toExponential(3)}`);
    console.log(`  normFactor (sqrt) = ${normFactorSqrt.toExponential(3)}`);
  }

  // ============================================================
  // 8. Run experiments
  // ============================================================
  console.log(`\nRunning FID ablation (${qids.length} queries, coarse_top=${COARSE_TOP})...\n`);

  // Method configs
  const methods = [
    { name: 'cosine_baseline', alpha: 0, useSqrt: false, docOnly: false, desc: 'pure centroid cosine (baseline)' },
    { name: 'fid_a0.0', alpha: 0.0, useSqrt: false, docOnly: false, desc: 'L2 centroid (alpha=0)' },
    { name: 'fid_a0.1', alpha: 0.1, useSqrt: false, docOnly: false, desc: 'FID alpha=0.1' },
    { name: 'fid_a0.3', alpha: 0.3, useSqrt: false, docOnly: false, desc: 'FID alpha=0.3' },
    { name: 'fid_a0.5', alpha: 0.5, useSqrt: false, docOnly: false, desc: 'FID alpha=0.5' },
    { name: 'fid_a1.0', alpha: 1.0, useSqrt: false, docOnly: false, desc: 'FID alpha=1.0' },
    { name: 'fid_a2.0', alpha: 2.0, useSqrt: false, docOnly: false, desc: 'FID alpha=2.0' },
    { name: 'fid_a5.0', alpha: 5.0, useSqrt: false, docOnly: false, desc: 'FID alpha=5.0' },
    { name: 'fid_sqrt_a0.5', alpha: 0.5, useSqrt: true, docOnly: false, desc: 'FID sqrt alpha=0.5' },
    { name: 'fid_sqrt_a1.0', alpha: 1.0, useSqrt: true, docOnly: false, desc: 'FID sqrt alpha=1.0' },
    { name: 'fid_sqrt_a2.0', alpha: 2.0, useSqrt: true, docOnly: false, desc: 'FID sqrt alpha=2.0' },
    { name: 'fid_doconly_a0.5', alpha: 0.5, useSqrt: true, docOnly: true, desc: 'FID doc-only sqrt alpha=0.5' },
    { name: 'fid_doconly_a1.0', alpha: 1.0, useSqrt: true, docOnly: true, desc: 'FID doc-only sqrt alpha=1.0' },
  ];

  // Results storage
  const results = {};
  for (const m of methods) {
    results[m.name] = {
      ndcg_scores: [],
      recall_scores: [],
      totalMs: 0,
      coarseMs: 0,
      fineMs: 0,
    };
  }

  // Also track the baseline token_2stage for comparison
  results['token_2stage_200'] = { ndcg_scores: [], recall_scores: [], totalMs: 0 };

  // Track graph smoothing scores for fusion
  const hasLaplacian = typeof vexus.shapeLaplacianPipeline === 'function';
  if (hasLaplacian) {
    results['best_fid_fusion'] = { ndcg_scores: [], recall_scores: [], totalMs: 0 };
  }

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qMu = queryMu[qid];
    const qFileId = queryLineMap[qid];
    const qVar = queryVar[qFileId] || new Float64Array(4096);
    const qrel = qrels[qid];

    if (qi % 50 === 0) {
      process.stdout.write(`  query ${qi}/${qids.length}...\r`);
    }

    // --- Cosine baseline: sort by cosine sim, take top-200 ---
    {
      const t1 = Date.now();
      const scored = docEntries.map(d => ({
        strId: d.strId,
        intId: d.intId,
        score: cosSim(qMu, d.mu),
      }));
      scored.sort((a, b) => b.score - a.score);
      const top200 = scored.slice(0, COARSE_TOP);
      const coarseMs = Date.now() - t1;

      // Recall@200 from cosine
      const recall = computeRecall(top200.map(x => x.strId), qrel, COARSE_TOP);
      results['cosine_baseline'].recall_scores.push(recall);

      // NDCG from cosine top-10 (no rerank)
      const ndcg = computeNDCG(top200.slice(0, K).map(x => x.strId), qrel, K);
      results['cosine_baseline'].ndcg_scores.push(ndcg);
      results['cosine_baseline'].coarseMs += coarseMs;
      results['cosine_baseline'].totalMs += coarseMs;
    }

    // --- Token 2-stage baseline (coarse=200) ---
    {
      const t1 = Date.now();
      const raw = vexus.tokenChamferTwoStage(qFileId, COARSE_TOP, TOP_N);
      const elapsed = Date.now() - t1;

      const ranked = raw.map(r => reverseMap[r[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      results['token_2stage_200'].ndcg_scores.push(ndcg);
      results['token_2stage_200'].totalMs += elapsed;

      // Recall: use coarse candidates from Rust (we don't have them directly)
      // We'll skip recall for this method
    }

    // --- FID methods ---
    for (const m of methods) {
      if (m.name === 'cosine_baseline') continue;  // already done

      const t1 = Date.now();

      // Compute FID distances for all docs
      const nf = m.useSqrt ? normFactorSqrt : normFactorVar;
      const scored = new Array(docEntries.length);
      for (let di = 0; di < docEntries.length; di++) {
        const d = docEntries[di];
        let dist;
        if (m.docOnly) {
          dist = fidDistanceDocOnly(qMu, d.mu, qVar, d.variance, m.alpha, nf);
        } else {
          dist = fidDistance(qMu, d.mu, qVar, d.variance, m.alpha, m.useSqrt, nf);
        }
        scored[di] = { strId: d.strId, intId: d.intId, dist };
      }

      // Sort by distance ascending
      scored.sort((a, b) => a.dist - b.dist);
      const top200 = scored.slice(0, COARSE_TOP);
      const coarseMs = Date.now() - t1;

      // Recall@200
      const recall = computeRecall(top200.map(x => x.strId), qrel, COARSE_TOP);
      results[m.name].recall_scores.push(recall);
      results[m.name].coarseMs += coarseMs;

      // Fine rank with PQ-Chamfer
      const t2 = Date.now();
      const candidateIntIds = top200.map(x => x.intId).filter(x => x !== undefined);
      const rerankRaw = vexus.tokenChamferRerankList(qFileId, candidateIntIds, TOP_N);
      const fineMs = Date.now() - t2;

      const ranked = rerankRaw.map(r => reverseMap[r[0]]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      results[m.name].ndcg_scores.push(ndcg);
      results[m.name].fineMs += fineMs;
      results[m.name].totalMs += coarseMs + fineMs;
    }

    // --- Graph smoothing fusion for best FID candidate ---
    if (hasLaplacian && qi === 0) {
      // We'll compute fusion after the main loop
    }
  }

  console.log(`\n`);

  // ============================================================
  // 8. Compute aggregates and print results
  // ============================================================
  console.log('=== Results ===\n');

  const cosineNDCG = results['cosine_baseline'].ndcg_scores.reduce((a, b) => a + b, 0) / qids.length;
  const token2stageNDCG = results['token_2stage_200'].ndcg_scores.reduce((a, b) => a + b, 0) / qids.length;

  console.log('| Method | Recall@200 | NDCG@10 | vs cosine | vs token_2stage | Coarse ms | Fine ms |');
  console.log('|:--|:--:|:--:|:--:|:--:|:--:|:--:|');

  const summaryRows = [];

  for (const m of [{ name: 'cosine_baseline' }, { name: 'token_2stage_200' }, ...methods.filter(x => x.name !== 'cosine_baseline')]) {
    const r = results[m.name];
    if (!r || r.ndcg_scores.length === 0) continue;

    const avgNDCG = r.ndcg_scores.reduce((a, b) => a + b, 0) / r.ndcg_scores.length;
    const avgRecall = r.recall_scores.length > 0
      ? r.recall_scores.reduce((a, b) => a + b, 0) / r.recall_scores.length
      : -1;
    const avgCoarse = r.coarseMs ? (r.coarseMs / qids.length).toFixed(0) : '-';
    const avgFine = r.fineMs ? (r.fineMs / qids.length).toFixed(0) : '-';

    const recallStr = avgRecall >= 0 ? avgRecall.toFixed(4) : '-';
    const vsCosine = pctChange(avgNDCG, cosineNDCG);
    const vsToken = pctChange(avgNDCG, token2stageNDCG);

    console.log(`| ${m.name} | ${recallStr} | ${avgNDCG.toFixed(4)} | ${vsCosine} | ${vsToken} | ${avgCoarse} | ${avgFine} |`);

    summaryRows.push({
      method: m.name,
      recall_200: avgRecall >= 0 ? parseFloat(avgRecall.toFixed(4)) : null,
      ndcg_10: parseFloat(avgNDCG.toFixed(4)),
      vs_cosine: vsCosine,
      vs_token_2stage: vsToken,
    });
  }

  // ============================================================
  // 9. Paired t-test: best FID vs alpha=0 (centroid L2)
  // ============================================================
  console.log('\n=== Paired t-tests ===\n');

  const baseScores = results['fid_a0.0'].ndcg_scores;
  for (const m of methods.filter(x => x.alpha > 0)) {
    const testScores = results[m.name].ndcg_scores;
    if (testScores.length !== baseScores.length) continue;

    const n = baseScores.length;
    let sumDiff = 0, sumDiffSq = 0;
    for (let i = 0; i < n; i++) {
      const d = testScores[i] - baseScores[i];
      sumDiff += d;
      sumDiffSq += d * d;
    }
    const meanDiff = sumDiff / n;
    const varDiff = (sumDiffSq - sumDiff * sumDiff / n) / (n - 1);
    const seDiff = Math.sqrt(varDiff / n);
    const t = seDiff > 0 ? meanDiff / seDiff : 0;

    const avgTest = testScores.reduce((a, b) => a + b, 0) / n;
    const avgBase = baseScores.reduce((a, b) => a + b, 0) / n;

    console.log(`${m.name} vs fid_a0.0: diff=${meanDiff.toFixed(4)}, t=${t.toFixed(2)}, base=${avgBase.toFixed(4)}, test=${avgTest.toFixed(4)}`);
  }

  // Also compare best FID recall vs cosine recall
  console.log('\n=== Recall@200 comparison ===\n');
  const cosRecall = results['cosine_baseline'].recall_scores;
  for (const m of methods.filter(x => x.name !== 'cosine_baseline')) {
    const fidRecall = results[m.name].recall_scores;
    if (fidRecall.length !== cosRecall.length) continue;

    const n = cosRecall.length;
    const avgCos = cosRecall.reduce((a, b) => a + b, 0) / n;
    const avgFid = fidRecall.reduce((a, b) => a + b, 0) / n;

    let wins = 0, ties = 0, losses = 0;
    for (let i = 0; i < n; i++) {
      if (fidRecall[i] > cosRecall[i] + 1e-6) wins++;
      else if (fidRecall[i] < cosRecall[i] - 1e-6) losses++;
      else ties++;
    }

    console.log(`${m.name}: recall=${avgFid.toFixed(4)} (cosine=${avgCos.toFixed(4)}, diff=${(avgFid - avgCos).toFixed(4)}, W/T/L=${wins}/${ties}/${losses})`);
  }

  // ============================================================
  // 10. Fusion experiment: best FID + graph smoothing
  // ============================================================
  if (hasLaplacian) {
    console.log('\n=== Fusion: best FID + graph smoothing ===\n');

    // Find the best alpha by NDCG
    let bestAlpha = 0, bestNDCG = 0, bestMethod = '', bestUseSqrt = false;
    for (const m of methods) {
      const avg = results[m.name].ndcg_scores.reduce((a, b) => a + b, 0) / qids.length;
      if (avg > bestNDCG) {
        bestNDCG = avg;
        bestAlpha = m.alpha;
        bestMethod = m.name;
        bestUseSqrt = m.useSqrt;
      }
    }
    console.log(`Best FID method: ${bestMethod} (NDCG=${bestNDCG.toFixed(4)})`);
    console.log('Running fusion with graph smoothing (lambda=0.7 token + 0.3 graph)...\n');

    let fusionNDCGs = [];

    for (let qi = 0; qi < qids.length; qi++) {
      const qid = qids[qi];
      const qMu = queryMu[qid];
      const qBuf = Buffer.from(qMu.buffer, qMu.byteOffset, qMu.byteLength);
      const qFileId = queryLineMap[qid];
      const qVar = queryVar[qFileId] || new Float64Array(4096);
      const qrel = qrels[qid];

      if (qi % 50 === 0) process.stdout.write(`  fusion query ${qi}/${qids.length}...\r`);

      // FID coarse screening with best alpha
      const scored = new Array(docEntries.length);
      for (let di = 0; di < docEntries.length; di++) {
        const d = docEntries[di];
        scored[di] = {
          strId: d.strId,
          intId: d.intId,
          dist: fidDistance(qMu, d.mu, qVar, d.variance, bestAlpha, bestUseSqrt, bestUseSqrt ? normFactorSqrt : normFactorVar),
        };
      }
      scored.sort((a, b) => a.dist - b.dist);
      const top200 = scored.slice(0, COARSE_TOP);

      // PQ-Chamfer fine rank
      const candidateIntIds = top200.map(x => x.intId).filter(x => x !== undefined);
      const tokenRaw = vexus.tokenChamferRerankList(qFileId, candidateIntIds, TOP_N);
      const tokenScores = {};
      for (const [did, score] of tokenRaw) {
        const strId = reverseMap[did];
        if (strId) tokenScores[strId] = score;
      }

      // Graph smoothing
      const graphRaw = vexus.shapeLaplacianPipeline(qBuf, TOP_N, TOP_N);
      const graphScores = {};
      for (const r of graphRaw) {
        const strId = reverseMap[r.docId];
        if (strId) graphScores[strId] = r.score;
      }

      // Fusion: 0.7 * token + 0.3 * graph (normalized)
      const normToken = normalizeScores(tokenScores);
      const normGraph = normalizeScores(graphScores);
      const allIds = new Set([...Object.keys(normToken), ...Object.keys(normGraph)]);
      const fusionScored = [];
      for (const id of allIds) {
        const ts = normToken[id] || 0;
        const gs = normGraph[id] || 0;
        fusionScored.push({ id, score: 0.7 * ts + 0.3 * gs });
      }
      fusionScored.sort((a, b) => b.score - a.score);

      const ndcg = computeNDCG(fusionScored.map(x => x.id), qrel, K);
      fusionNDCGs.push(ndcg);
    }

    const avgFusion = fusionNDCGs.reduce((a, b) => a + b, 0) / fusionNDCGs.length;
    console.log(`\nFID(${bestMethod}) + graph fusion (0.7/0.3): NDCG@10 = ${avgFusion.toFixed(4)} (${pctChange(avgFusion, cosineNDCG)} vs cosine)`);
    summaryRows.push({
      method: `fid_fusion_${bestMethod}`,
      ndcg_10: parseFloat(avgFusion.toFixed(4)),
      vs_cosine: pctChange(avgFusion, cosineNDCG),
      vs_token_2stage: pctChange(avgFusion, token2stageNDCG),
    });
  }

  // ============================================================
  // 11. Save results
  // ============================================================
  const outputPath = path.join(DATA_DIR, 'fid_ablation_results.json');
  const output = {
    timestamp: new Date().toISOString(),
    config: { coarse_top: COARSE_TOP, top_n: TOP_N, k: K, n_queries: qids.length },
    baselines: {
      cosine_ndcg10: cosineNDCG,
      token_2stage_200_ndcg10: token2stageNDCG,
      current_best_fus55: 0.3271,
    },
    methods: summaryRows,
  };
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nResults saved to ${outputPath}`);
}

main().catch(e => { console.error(e); process.exit(1); });
