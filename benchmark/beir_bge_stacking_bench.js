#!/usr/bin/env node
/**
 * BGE Stacking Benchmark: cosine baseline + graph Laplacian smoothing
 * Tests BGE-large and BGE-M3 on NFCorpus with the Shape-CFD graph smoothing pipeline.
 * Pure JS implementation (no Rust NAPI needed for 1024d).
 *
 * Matches Rust pipeline: alpha=0.02, steps=20, knn=3, C0=exp(-2*cosine_dist)
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const BASE = '/home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus';

// ─── Helpers ────────────────────────────────────────────────────────────

function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

function ndcg10(ranked, relevant) {
  let dcg = 0;
  for (let i = 0; i < Math.min(10, ranked.length); i++) {
    const rel = relevant.get(ranked[i]) || 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  const rels = [...relevant.values()].sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(10, rels.length); i++) {
    idcg += (Math.pow(2, rels[i]) - 1) / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

// ─── Laplacian Smoothing (matching Rust pde::laplacian_smooth) ──────────

function laplacianSmooth(candidates, alpha = 0.02, steps = 20, knn = 3) {
  const N = candidates.length;

  // Initial concentration: exp(-2 * cosine_distance)
  let C = candidates.map(c => Math.exp(-2 * (1 - c.score)));

  // Build similarity matrix (within candidate set)
  const simMatrix = new Array(N);
  for (let i = 0; i < N; i++) {
    simMatrix[i] = new Float32Array(N);
  }
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const s = cosSim(candidates[i].vec, candidates[j].vec);
      simMatrix[i][j] = s;
      simMatrix[j][i] = s;
    }
  }

  // KNN adjacency with edge weights = exp(-2 * cosine_distance)
  const adj = new Array(N);
  for (let i = 0; i < N; i++) {
    const dists = [];
    for (let j = 0; j < N; j++) {
      if (i !== j) dists.push({ j, d: 1 - simMatrix[i][j] });
    }
    dists.sort((a, b) => a.d - b.d);
    adj[i] = dists.slice(0, knn).map(x => ({
      j: x.j,
      w: Math.exp(-2 * x.d)
    }));
  }

  // Laplacian diffusion (matching Rust: c_new = c + alpha * sum(w*(c_j-c_i)) / sum(w))
  for (let t = 0; t < steps; t++) {
    const newC = new Array(N);
    for (let i = 0; i < N; i++) {
      let wSum = 0, wDiff = 0;
      for (const { j, w } of adj[i]) {
        wDiff += w * (C[j] - C[i]);
        wSum += w;
      }
      newC[i] = wSum > 0 ? C[i] + alpha * wDiff / wSum : C[i];
      if (newC[i] < 0) newC[i] = 0; // non-negative clamp
    }
    C = newC;
  }

  const result = candidates.map((c, i) => ({ id: c.id, score: C[i] }));
  result.sort((a, b) => b.score - a.score);
  return result;
}

// ─── Data Loading ───────────────────────────────────────────────────────

function loadVectorsStreaming(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.trim().split('\n');
  return lines.map(line => JSON.parse(line));
}

function loadQrels() {
  const lines = fs.readFileSync(path.join(BASE, 'qrels.tsv'), 'utf8').trim().split('\n');
  const qrels = new Map();
  for (let i = 1; i < lines.length; i++) {
    const [qid, docid, score] = lines[i].split('\t');
    if (!qrels.has(qid)) qrels.set(qid, new Map());
    qrels.get(qid).set(docid, parseInt(score));
  }
  return qrels;
}

// ─── Benchmark ──────────────────────────────────────────────────────────

function runBenchmark(modelName, corpusFile, queryFile, qrels) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Model: ${modelName}`);
  console.log(`${'='.repeat(60)}`);

  console.log('Loading vectors...');
  const corpusDocs = loadVectorsStreaming(corpusFile);
  const queryDocs = loadVectorsStreaming(queryFile);

  const docMap = new Map();
  for (const doc of corpusDocs) {
    docMap.set(doc._id, doc.vector);
  }
  const docIds = [...docMap.keys()];
  const docVecs = docIds.map(id => docMap.get(id));

  const validQueries = queryDocs.filter(q => qrels.has(q._id));
  console.log(`Valid queries (in qrels): ${validQueries.length}`);
  console.log(`Corpus docs: ${docIds.length}`);

  const results = {};

  // --- Cosine baseline ---
  console.log('\nRunning cosine baseline...');
  const t0 = Date.now();
  const cosineScores = [];
  const cosinePerQuery = [];

  // Precompute all cosine scores for efficiency
  const allCosineRankings = new Map();
  for (const q of validQueries) {
    const scores = docIds.map((id, idx) => ({
      id,
      score: cosSim(q.vector, docVecs[idx]),
      vec: docVecs[idx]
    }));
    scores.sort((a, b) => b.score - a.score);
    allCosineRankings.set(q._id, scores);

    const topIds = scores.slice(0, 10).map(s => s.id);
    const ndcg = ndcg10(topIds, qrels.get(q._id));
    cosineScores.push(ndcg);
    cosinePerQuery.push({ qid: q._id, ndcg });
  }

  const cosineNDCG = cosineScores.reduce((a, b) => a + b, 0) / cosineScores.length;
  const cosineTime = Date.now() - t0;
  console.log(`  cosine NDCG@10 = ${cosineNDCG.toFixed(4)} (${cosineTime}ms)`);
  results[`${modelName}_cosine`] = { ndcg10: cosineNDCG, elapsed: cosineTime, numQueries: validQueries.length, perQuery: cosinePerQuery };

  // --- Graph smoothing with different top_n and parameter configs ---
  const configs = [
    // Rust defaults: alpha=0.02, steps=20
    { alpha: 0.02, steps: 20, knn: 3, label: 'rust_default' },
    // More aggressive smoothing
    { alpha: 0.1, steps: 10, knn: 3, label: 'alpha01' },
    { alpha: 0.3, steps: 5, knn: 3, label: 'alpha03' },
    // Wider neighborhood
    { alpha: 0.02, steps: 20, knn: 5, label: 'knn5' },
  ];

  for (const topN of [55, 100, 200, 300]) {
    for (const cfg of configs) {
      const tag = `${modelName}_lap_${topN}_${cfg.label}`;
      console.log(`  Running ${tag} (alpha=${cfg.alpha}, steps=${cfg.steps}, knn=${cfg.knn})...`);
      const t1 = Date.now();
      const lapScores = [];
      const lapPerQuery = [];

      for (const q of validQueries) {
        const rankings = allCosineRankings.get(q._id);
        const candidates = rankings.slice(0, topN);
        const smoothed = laplacianSmooth(candidates, cfg.alpha, cfg.steps, cfg.knn);
        const topIds = smoothed.slice(0, 10).map(s => s.id);
        const ndcg = ndcg10(topIds, qrels.get(q._id));
        lapScores.push(ndcg);
        lapPerQuery.push({ qid: q._id, ndcg });
      }

      const lapNDCG = lapScores.reduce((a, b) => a + b, 0) / lapScores.length;
      const lapTime = Date.now() - t1;
      const gain = ((lapNDCG - cosineNDCG) / cosineNDCG * 100).toFixed(1);
      console.log(`    NDCG@10 = ${lapNDCG.toFixed(4)} (${lapTime}ms) [${gain}% vs cosine]`);
      results[tag] = { ndcg10: lapNDCG, elapsed: lapTime, numQueries: validQueries.length, gain: `${gain}%`, params: cfg, perQuery: lapPerQuery };
    }
  }

  return results;
}

// ─── Main ───────────────────────────────────────────────────────────────

function main() {
  console.log('Loading qrels...');
  const qrels = loadQrels();
  console.log(`Qrels: ${qrels.size} queries`);

  const allResults = {};

  // BGE-large
  const bgeCorpus = path.join(BASE, 'bge_large_corpus_vectors.jsonl');
  const bgeQuery = path.join(BASE, 'bge_large_query_vectors.jsonl');
  if (fs.existsSync(bgeCorpus) && fs.existsSync(bgeQuery)) {
    const r = runBenchmark('bge_large', bgeCorpus, bgeQuery, qrels);
    Object.assign(allResults, r);
  }

  // BGE-M3
  const m3Corpus = path.join(BASE, 'bge_m3_corpus_vectors.jsonl');
  const m3Query = path.join(BASE, 'bge_m3_query_vectors.jsonl');
  if (fs.existsSync(m3Corpus) && fs.existsSync(m3Query)) {
    const r = runBenchmark('bge_m3', m3Corpus, m3Query, qrels);
    Object.assign(allResults, r);
  }

  // Save results
  const outFile = path.join(BASE, 'bge_stacking_results.json');
  const summary = {};
  for (const [k, v] of Object.entries(allResults)) {
    const { perQuery, ...rest } = v;
    summary[k] = rest;
  }
  fs.writeFileSync(outFile, JSON.stringify(summary, null, 2));
  console.log(`\nResults saved to ${outFile}`);

  // Print comparison table - find best config per model per topN
  console.log('\n' + '='.repeat(90));
  console.log('BEST RESULTS PER MODEL (best alpha/steps/knn config)');
  console.log('='.repeat(90));
  console.log('| Model        | dim  | cosine | lap_55 | lap_100| lap_200| lap_300| Best Gain |');
  console.log('|:-------------|:----:|:------:|:------:|:------:|:------:|:------:|:---------:|');

  for (const model of ['bge_large', 'bge_m3']) {
    const cos = allResults[`${model}_cosine`]?.ndcg10;
    if (!cos) continue;

    const dim = 1024;
    const bestPerTopN = {};
    let overallBestGain = -Infinity;
    let overallBestConfig = '';

    for (const topN of [55, 100, 200, 300]) {
      let bestVal = -Infinity;
      let bestCfgLabel = '';
      for (const [k, v] of Object.entries(allResults)) {
        if (k.startsWith(`${model}_lap_${topN}_`) && v.ndcg10 > bestVal) {
          bestVal = v.ndcg10;
          bestCfgLabel = k;
        }
      }
      bestPerTopN[topN] = bestVal > -Infinity ? bestVal : null;
      if (bestVal > -Infinity) {
        const g = (bestVal - cos) / cos * 100;
        if (g > overallBestGain) {
          overallBestGain = g;
          overallBestConfig = bestCfgLabel;
        }
      }
    }

    const fmt = (v) => v != null ? v.toFixed(4) : '  --  ';
    console.log(`| ${model.padEnd(12)} | ${dim} | ${cos.toFixed(4)} | ${fmt(bestPerTopN[55])} | ${fmt(bestPerTopN[100])} | ${fmt(bestPerTopN[200])} | ${fmt(bestPerTopN[300])} | ${overallBestGain > 0 ? '+' : ''}${overallBestGain.toFixed(1)}% |`);
    if (overallBestConfig) {
      console.log(`  Best config: ${overallBestConfig}`);
    }
  }

  // Print detailed parameter comparison
  console.log('\n' + '='.repeat(90));
  console.log('DETAILED PARAMETER SWEEP');
  console.log('='.repeat(90));

  for (const model of ['bge_large', 'bge_m3']) {
    const cos = allResults[`${model}_cosine`]?.ndcg10;
    if (!cos) continue;
    console.log(`\n--- ${model} (cosine baseline: ${cos.toFixed(4)}) ---`);

    const entries = Object.entries(allResults)
      .filter(([k]) => k.startsWith(`${model}_lap_`))
      .sort(([,a], [,b]) => b.ndcg10 - a.ndcg10);

    for (const [k, v] of entries) {
      const gain = ((v.ndcg10 - cos) / cos * 100).toFixed(1);
      const p = v.params || {};
      console.log(`  ${k.padEnd(40)} NDCG@10=${v.ndcg10.toFixed(4)} [${gain}%] (a=${p.alpha},s=${p.steps},k=${p.knn})`);
    }
  }

  // Significance test: paired t-test for best smoothing vs cosine
  console.log('\n' + '='.repeat(90));
  console.log('SIGNIFICANCE TESTS (paired t-test, best smoothing vs cosine)');
  console.log('='.repeat(90));

  for (const model of ['bge_large', 'bge_m3']) {
    const cosKey = `${model}_cosine`;
    const cosData = allResults[cosKey];
    if (!cosData) continue;

    // Find best smoothing
    let bestKey = null, bestNDCG = -Infinity;
    for (const [k, v] of Object.entries(allResults)) {
      if (k.startsWith(`${model}_lap_`) && v.ndcg10 > bestNDCG) {
        bestNDCG = v.ndcg10;
        bestKey = k;
      }
    }
    if (!bestKey) continue;

    const bestData = allResults[bestKey];
    const n = cosData.perQuery.length;

    // Paired differences
    const diffs = [];
    for (let i = 0; i < n; i++) {
      diffs.push(bestData.perQuery[i].ndcg - cosData.perQuery[i].ndcg);
    }

    const meanDiff = diffs.reduce((a, b) => a + b, 0) / n;
    const varDiff = diffs.reduce((a, b) => a + (b - meanDiff) ** 2, 0) / (n - 1);
    const seDiff = Math.sqrt(varDiff / n);
    const tStat = meanDiff / seDiff;

    // Count wins/ties/losses
    let wins = 0, ties = 0, losses = 0;
    for (const d of diffs) {
      if (d > 0.001) wins++;
      else if (d < -0.001) losses++;
      else ties++;
    }

    console.log(`\n${model}: ${bestKey} vs ${cosKey}`);
    console.log(`  NDCG@10: ${bestNDCG.toFixed(4)} vs ${cosData.ndcg10.toFixed(4)} (diff: ${meanDiff > 0 ? '+' : ''}${meanDiff.toFixed(4)})`);
    console.log(`  Paired t = ${tStat.toFixed(2)}, SE = ${seDiff.toFixed(4)}`);
    console.log(`  Wins/Ties/Losses: ${wins}/${ties}/${losses}`);
    console.log(`  ${Math.abs(tStat) > 1.96 ? 'SIGNIFICANT (p < 0.05)' : 'NOT significant (p >= 0.05)'}`);
  }

  // Save full results
  const fullOutFile = path.join(BASE, 'bge_stacking_full_results.json');
  fs.writeFileSync(fullOutFile, JSON.stringify(allResults, null, 2));
  console.log(`\nFull results saved to ${fullOutFile}`);
}

main();
