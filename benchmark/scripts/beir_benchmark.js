#!/usr/bin/env node
'use strict';

/**
 * BEIR Benchmark — 4 种重排序方法对比
 * 
 * 方法:
 *   A. Cosine 直排 (baseline)
 *   B. AD-Rank v2 (对流-扩散)
 *   C. Shape-CFD (Chamfer + 点云 + 对流-扩散)
 *   D. BM25 (词法基线, 用简易 TF-IDF)
 * 
 * 用法: node beir_benchmark.js --dataset scifact
 */

const fs = require('fs');
const path = require('path');

// ── 命令行参数 ──
const args = process.argv.slice(2);
const datasetName = args.find((_, i) => args[i - 1] === '--dataset') || 'scifact';
const dataDir = args.find((_, i) => args[i - 1] === '--data-dir') || './beir_data';
const topN = parseInt(args.find((_, i) => args[i - 1] === '--topn') || '30');
const topK = parseInt(args.find((_, i) => args[i - 1] === '--topk') || '100');

const datasetDir = path.join(dataDir, datasetName);

// ── 加载数据 ──
const readline = require('readline');

function loadJsonlStream(filePath) {
  return new Promise((resolve, reject) => {
    const results = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(filePath, { encoding: 'utf-8' }),
      crlfDelay: Infinity,
    });
    rl.on('line', line => {
      if (line.trim()) {
        try { results.push(JSON.parse(line)); } catch(e) {}
      }
    });
    rl.on('close', () => resolve(results));
    rl.on('error', reject);
  });
}

function loadQrels(filePath) {
  const qrels = {};
  const lines = fs.readFileSync(filePath, 'utf-8').trim().split('\n');
  for (let i = 1; i < lines.length; i++) {
    const [qid, did, score] = lines[i].split('\t');
    if (!qrels[qid]) qrels[qid] = {};
    qrels[qid][did] = parseInt(score);
  }
  return qrels;
}

// ── cosine 距离工具 ──
function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

function cosineDistance(a, b) {
  return 1 - cosineSimilarity(a, b);
}

// ── 方法 A: Cosine 直排 ──
function cosineRerank(queryVec, docVecs, docIds) {
  const scored = docIds.map((id, i) => ({
    doc_id: id,
    score: cosineSimilarity(queryVec, docVecs[i]),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

// ── 方法 B: AD-Rank v2 ──
function adRankV2(queryVec, docVecs, docIds, options = {}) {
  const { D = 0.15, uStrength = 0.1, maxIter = 50, epsilon = 1e-3, knn = 3 } = options;
  const N = docVecs.length;
  if (N === 0) return [];

  // Build KNN graph (cosine similarity)
  const simMatrix = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const s = cosineSimilarity(docVecs[i], docVecs[j]);
      simMatrix[i * N + j] = s;
      simMatrix[j * N + i] = s;
    }
  }

  const beta = 2.0;
  const effectiveK = Math.min(knn, N - 1);
  const adjacency = Array.from({ length: N }, () => []);

  for (let i = 0; i < N; i++) {
    const neighbors = [];
    for (let j = 0; j < N; j++) {
      if (j !== i) neighbors.push({ j, sim: simMatrix[i * N + j] });
    }
    neighbors.sort((a, b) => b.sim - a.sim);
    for (let t = 0; t < effectiveK; t++) {
      const nb = neighbors[t];
      adjacency[i].push({ j: nb.j, w: Math.exp(-beta * (1 - nb.sim)) });
    }
  }

  // Symmetrize
  for (let i = 0; i < N; i++) {
    for (const edge of adjacency[i]) {
      if (!adjacency[edge.j].some(e => e.j === i)) {
        adjacency[edge.j].push({ j: i, w: edge.w });
      }
    }
  }

  // Initial concentration
  let C = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    C[i] = cosineSimilarity(queryVec, docVecs[i]);
  }

  // Advection coefficients
  const dim = queryVec.length;
  let qNorm = 0;
  for (let d = 0; d < dim; d++) qNorm += queryVec[d] * queryVec[d];
  const invQNorm = 1.0 / (Math.sqrt(qNorm) + 1e-8);

  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (const edge of adjacency[i]) {
      const j = edge.j;
      if (U[i * N + j] !== 0) continue;
      let edNorm = 0, dotVal = 0;
      for (let d = 0; d < dim; d++) {
        const diff = docVecs[j][d] - docVecs[i][d];
        edNorm += diff * diff;
        dotVal += diff * queryVec[d] * invQNorm;
      }
      const u_ij = (dotVal / (Math.sqrt(edNorm) + 1e-8)) * uStrength;
      U[i * N + j] = u_ij;
      U[j * N + i] = -u_ij;
    }
  }

  // CFL
  let maxDeg = 0;
  for (let i = 0; i < N; i++) if (adjacency[i].length > maxDeg) maxDeg = adjacency[i].length;
  const dt = Math.min(0.1, maxDeg > 0 ? 0.8 / maxDeg : 0.1);

  // Iterate
  let C_new = new Float64Array(N);
  for (let t = 0; t < maxIter; t++) {
    let maxDelta = 0;
    for (let i = 0; i < N; i++) {
      let diffusion = 0, advection = 0;
      for (const edge of adjacency[i]) {
        const j = edge.j, w = edge.w;
        diffusion += D * w * (C[j] - C[i]);
        const u_ij = U[i * N + j];
        const u_ji = U[j * N + i];
        advection += w * ((u_ji > 0 ? u_ji : 0) * C[j] - (u_ij > 0 ? u_ij : 0) * C[i]);
      }
      const c_new = Math.max(0, C[i] + dt * (diffusion + advection));
      C_new[i] = c_new;
      const delta = Math.abs(c_new - C[i]);
      if (delta > maxDelta) maxDelta = delta;
    }
    [C, C_new] = [C_new, C];
    if (maxDelta < epsilon) break;
  }

  return docIds.map((id, i) => ({ doc_id: id, score: C[i] }))
    .sort((a, b) => b.score - a.score);
}

// ── 方法 C: Shape-CFD ──
function shapeCFD(queryCloud, docClouds, docIds, options = {}) {
  const { D = 0.15, uStrength = 0.3, maxIter = 50, epsilon = 1e-3, knn = 3 } = options;
  const N = docClouds.length;
  if (N === 0) return [];

  // Chamfer distance
  function chamfer(cloudA, cloudB) {
    let sumAB = 0;
    for (const a of cloudA) {
      let minD = Infinity;
      for (const b of cloudB) { const d = cosineDistance(a, b); if (d < minD) minD = d; }
      sumAB += minD;
    }
    let sumBA = 0;
    for (const b of cloudB) {
      let minD = Infinity;
      for (const a of cloudA) { const d = cosineDistance(a, b); if (d < minD) minD = d; }
      sumBA += minD;
    }
    return sumAB / cloudA.length + sumBA / cloudB.length;
  }

  function centroid(cloud) {
    const dim = cloud[0].length;
    const c = new Float32Array(dim);
    for (const v of cloud) for (let d = 0; d < dim; d++) c[d] += v[d];
    const inv = 1.0 / cloud.length;
    for (let d = 0; d < dim; d++) c[d] *= inv;
    return c;
  }

  // Build Shape KNN graph
  const beta = 2.0;
  const distMatrix = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const d = chamfer(docClouds[i], docClouds[j]);
      distMatrix[i * N + j] = d;
      distMatrix[j * N + i] = d;
    }
  }

  const effectiveK = Math.min(knn, N - 1);
  const adjacency = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) {
    const neighbors = [];
    for (let j = 0; j < N; j++) {
      if (j !== i) neighbors.push({ j, dist: distMatrix[i * N + j] });
    }
    neighbors.sort((a, b) => a.dist - b.dist);
    for (let t = 0; t < effectiveK; t++) {
      adjacency[i].push({ j: neighbors[t].j, w: Math.exp(-beta * neighbors[t].dist) });
    }
  }

  // Symmetrize
  for (let i = 0; i < N; i++) {
    for (const edge of adjacency[i]) {
      if (!adjacency[edge.j].some(e => e.j === i)) {
        adjacency[edge.j].push({ j: i, w: edge.w });
      }
    }
  }

  // Initial concentration = exp(-2 * chamfer(query, doc))
  let C = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    C[i] = Math.exp(-2 * chamfer(queryCloud, docClouds[i]));
  }

  // Advection: centroid → centroid
  const queryCentroid = centroid(queryCloud);
  const docCentroids = docClouds.map(centroid);
  const dim = queryCentroid.length;
  let qNorm = 0;
  for (let d = 0; d < dim; d++) qNorm += queryCentroid[d] * queryCentroid[d];
  const invQNorm = 1.0 / (Math.sqrt(qNorm) + 1e-8);

  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (const edge of adjacency[i]) {
      const j = edge.j;
      if (U[i * N + j] !== 0) continue;
      let edNorm = 0, dotVal = 0;
      for (let d = 0; d < dim; d++) {
        const diff = docCentroids[j][d] - docCentroids[i][d];
        edNorm += diff * diff;
        dotVal += diff * queryCentroid[d] * invQNorm;
      }
      const u_ij = (dotVal / (Math.sqrt(edNorm) + 1e-8)) * uStrength;
      U[i * N + j] = u_ij;
      U[j * N + i] = -u_ij;
    }
  }

  // CFL + iterate (same as v2)
  let maxDeg = 0;
  for (let i = 0; i < N; i++) if (adjacency[i].length > maxDeg) maxDeg = adjacency[i].length;
  const dt = Math.min(0.1, maxDeg > 0 ? 0.8 / maxDeg : 0.1);

  let C_new = new Float64Array(N);
  for (let t = 0; t < maxIter; t++) {
    let maxDelta = 0;
    for (let i = 0; i < N; i++) {
      let diffusion = 0, advection = 0;
      for (const edge of adjacency[i]) {
        const j = edge.j, w = edge.w;
        diffusion += D * w * (C[j] - C[i]);
        const u_ij = U[i * N + j], u_ji = U[j * N + i];
        advection += w * ((u_ji > 0 ? u_ji : 0) * C[j] - (u_ij > 0 ? u_ij : 0) * C[i]);
      }
      const c_new = Math.max(0, C[i] + dt * (diffusion + advection));
      C_new[i] = c_new;
      const delta = Math.abs(c_new - C[i]);
      if (delta > maxDelta) maxDelta = delta;
    }
    [C, C_new] = [C_new, C];
    if (maxDelta < epsilon) break;
  }

  return docIds.map((id, i) => ({ doc_id: id, score: C[i] }))
    .sort((a, b) => b.score - a.score);
}

// ── 方法 D: 简易 BM25 ──
function bm25Rerank(queryText, docTexts, docIds, options = {}) {
  const { k1 = 1.2, b = 0.75 } = options;
  
  // Tokenize (simple whitespace + lowercase)
  const tokenize = (text) => text.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff]/g, ' ').split(/\s+/).filter(t => t.length > 1);
  
  const queryTokens = tokenize(queryText);
  const N = docTexts.length;
  
  // Document frequencies
  const docTokens = docTexts.map(tokenize);
  const avgDl = docTokens.reduce((s, d) => s + d.length, 0) / N;
  
  const df = {};
  for (const tokens of docTokens) {
    const seen = new Set(tokens);
    for (const t of seen) df[t] = (df[t] || 0) + 1;
  }
  
  // Score each document
  const scores = docIds.map((id, i) => {
    const tokens = docTokens[i];
    const dl = tokens.length;
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
    
    let score = 0;
    for (const qt of queryTokens) {
      if (!tf[qt]) continue;
      const idf = Math.log((N - (df[qt] || 0) + 0.5) / ((df[qt] || 0) + 0.5) + 1);
      const tfNorm = (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avgDl));
      score += idf * tfNorm;
    }
    
    return { doc_id: id, score };
  });
  
  scores.sort((a, b) => b.score - a.score);
  return scores;
}

// ── 主函数 ──
async function main() {
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  BEIR Benchmark — ${datasetName} (top-${topN} → rerank → top-${topK})`);
  console.log(`${'═'.repeat(60)}\n`);

  // 1. 加载数据
  console.log('📂 加载数据...');
  const corpus = {};
  for (const obj of await loadJsonlStream(path.join(datasetDir, 'corpus.jsonl'))) {
    corpus[obj._id] = obj;
  }
  
  const queryVectors = {};
  const queryTexts = {};
  for (const obj of await loadJsonlStream(path.join(datasetDir, 'query_vectors.jsonl'))) {
    queryVectors[obj._id] = new Float32Array(obj.vector);
    queryTexts[obj._id] = obj.text;
  }
  
  const corpusVectors = {};
  const corpusSentences = {};
  for (const obj of await loadJsonlStream(path.join(datasetDir, 'corpus_vectors.jsonl'))) {
    corpusVectors[obj._id] = new Float32Array(obj.vector);
    if (obj.sentences && obj.sentences.length > 0) {
      corpusSentences[obj._id] = obj.sentences.map(s => new Float32Array(s));
    }
  }
  
  const qrels = loadQrels(path.join(datasetDir, 'qrels.tsv'));
  
  console.log(`  Queries with vectors: ${Object.keys(queryVectors).length}`);
  console.log(`  Corpus with vectors: ${Object.keys(corpusVectors).length}`);
  console.log(`  Corpus with sentences: ${Object.keys(corpusSentences).length}`);
  console.log(`  Qrels: ${Object.keys(qrels).length} queries\n`);

  // 2. 对每个 query 运行 4 种 reranking
  const results = [];
  const queryIds = Object.keys(qrels).filter(qid => queryVectors[qid]);
  
  console.log(`🔄 Running ${queryIds.length} queries × 4 methods...\n`);

  for (let qi = 0; qi < queryIds.length; qi++) {
    const qid = queryIds[qi];
    const queryVec = queryVectors[qid];
    const queryText = queryTexts[qid] || '';

    // First-stage retrieval: top-N by cosine similarity
    const allDocIds = Object.keys(corpusVectors);
    const cosineScores = allDocIds.map(did => ({
      did,
      score: cosineSimilarity(queryVec, corpusVectors[did]),
    }));
    cosineScores.sort((a, b) => b.score - a.score);
    const topNDocs = cosineScores.slice(0, topN);
    const candidateIds = topNDocs.map(d => d.did);
    const candidateVecs = candidateIds.map(did => corpusVectors[did]);

    // A. Cosine 直排 (just top-N cosine scores)
    const cosineResult = topNDocs.map(d => ({ doc_id: d.did, score: d.score }));
    results.push({ query_id: qid, method: 'cosine', rankings: cosineResult.slice(0, topK) });

    // B. AD-Rank v2
    const v2Result = adRankV2(queryVec, candidateVecs, candidateIds);
    results.push({ query_id: qid, method: 'ad_rank_v2', rankings: v2Result.slice(0, topK) });

    // C. Shape-CFD (only if sentence vectors available)
    const candidateClouds = candidateIds.map(did => 
      corpusSentences[did] || [corpusVectors[did]]
    );
    const queryCloud = [queryVec]; // query as single-point cloud
    const shapeResult = shapeCFD(queryCloud, candidateClouds, candidateIds);
    results.push({ query_id: qid, method: 'shape_cfd', rankings: shapeResult.slice(0, topK) });

    // D. BM25
    const candidateTexts = candidateIds.map(did => corpus[did]?.text || '');
    const bm25Result = bm25Rerank(queryText, candidateTexts, candidateIds);
    results.push({ query_id: qid, method: 'bm25', rankings: bm25Result.slice(0, topK) });

    if ((qi + 1) % 10 === 0 || qi === queryIds.length - 1) {
      process.stdout.write(`\r  Progress: ${qi + 1}/${queryIds.length} queries`);
    }
  }

  console.log('\n');

  // 3. 保存结果
  const outputPath = path.join(datasetDir, 'results.jsonl');
  const fd = fs.openSync(outputPath, 'w');
  for (const r of results) {
    fs.writeSync(fd, JSON.stringify(r) + '\n');
  }
  fs.closeSync(fd);

  console.log(`💾 结果保存到 ${outputPath}`);
  console.log(`   共 ${results.length} 条 (${queryIds.length} queries × 4 methods)`);
  console.log(`\n✅ 完成！运行 beir_evaluate.py 计算指标。`);
}

main().catch(err => {
  console.error('❌', err);
  process.exit(1);
});
