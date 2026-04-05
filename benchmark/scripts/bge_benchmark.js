/**
 * BGE-large-en-v1.5 Benchmark on NFCorpus
 * - Cosine baseline
 * - Laplacian smoothing (图平滑) with various top_n
 *
 * 纯 JS 实现，不依赖 Rust NAPI
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = '/home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus';

// ===== 向量操作 =====
function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(v) {
  return Math.sqrt(dotProduct(v, v));
}

function cosSim(a, b) {
  const d = norm(a) * norm(b);
  return d > 0 ? dotProduct(a, b) / d : 0;
}

// ===== 数据加载 =====
function loadJsonl(filePath) {
  console.log(`Loading ${filePath}...`);
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  return lines.map(l => JSON.parse(l));
}

function loadQrels(filePath) {
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  const qrels = {};
  for (let i = 1; i < lines.length; i++) {  // skip header
    const parts = lines[i].split('\t');
    if (parts.length < 3) continue;
    const [qid, did, score] = parts;
    if (!qrels[qid]) qrels[qid] = {};
    qrels[qid][did] = parseInt(score);
  }
  return qrels;
}

// ===== NDCG@K 计算 =====
function dcg(scores, k) {
  let sum = 0;
  for (let i = 0; i < Math.min(scores.length, k); i++) {
    sum += (Math.pow(2, scores[i]) - 1) / Math.log2(i + 2);
  }
  return sum;
}

function ndcgAtK(ranked, qrel, k) {
  // ranked = [{id, score}, ...] 已排序
  const relevances = ranked.slice(0, k).map(r => qrel[r.id] || 0);
  const idealRelevances = Object.values(qrel).sort((a, b) => b - a).slice(0, k);

  const d = dcg(relevances, k);
  const ideal = dcg(idealRelevances, k);
  return ideal > 0 ? d / ideal : 0;
}

// ===== 方法实现 =====

// 1. Cosine baseline: query vs all docs
function cosineRank(queryVec, docVecs, docIds) {
  const scored = docIds.map((id, i) => ({
    id,
    score: cosSim(queryVec, docVecs[i])
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

// 2. Laplacian smoothing
function laplacianSmooth(queryVec, docVecs, docIds, topN, alpha = 0.15, steps = 5, knn = 3) {
  // 1. cosine 粗筛 top-N
  const scored = docIds.map((id, i) => ({
    id, i,
    score: cosSim(queryVec, docVecs[i])
  }));
  scored.sort((a, b) => b.score - a.score);
  const candidates = scored.slice(0, topN);
  const n = candidates.length;

  // 2. 初始浓度 C0 = exp(-2 * cosine_distance)
  let C = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    C[i] = Math.exp(-2 * (1 - candidates[i].score));
  }

  // 3. 建 KNN 图（候选集内部），预计算距离矩阵
  const distMatrix = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const sim = cosSim(docVecs[candidates[i].i], docVecs[candidates[j].i]);
      const dist = 1 - sim;
      distMatrix[i * n + j] = dist;
      distMatrix[j * n + i] = dist;
    }
  }

  // 每个节点取最近 knn 个邻居，边权 = exp(-2 * distance)
  const neighbors = [];  // neighbors[i] = [{j, w}, ...]
  for (let i = 0; i < n; i++) {
    const dists = [];
    for (let j = 0; j < n; j++) {
      if (j !== i) dists.push({ j, d: distMatrix[i * n + j] });
    }
    dists.sort((a, b) => a.d - b.d);
    neighbors.push(dists.slice(0, knn).map(d => ({
      j: d.j,
      w: Math.exp(-2 * d.d)
    })));
  }

  // 4. Laplacian 扩散 steps 步
  for (let step = 0; step < steps; step++) {
    const Cnew = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let wSum = 0;
      let wCSum = 0;
      for (const { j, w } of neighbors[i]) {
        wSum += w;
        wCSum += w * C[j];
      }
      Cnew[i] = (1 - alpha) * C[i] + (wSum > 0 ? alpha * wCSum / wSum : 0);
    }
    C = Cnew;
  }

  // 5. 按 C 降序排序
  const result = candidates.map((c, i) => ({ id: c.id, score: C[i] }));
  result.sort((a, b) => b.score - a.score);
  return result;
}

// ===== 主函数 =====
async function main() {
  // 加载数据
  const corpusDocs = loadJsonl(path.join(DATA_DIR, 'bge_corpus_vectors.jsonl'));
  const queryDocs = loadJsonl(path.join(DATA_DIR, 'bge_query_vectors.jsonl'));
  const qrels = loadQrels(path.join(DATA_DIR, 'qrels.tsv'));

  console.log(`Corpus: ${corpusDocs.length} docs, Query: ${queryDocs.length} queries`);
  console.log(`Qrels: ${Object.keys(qrels).length} queries with relevance judgments`);
  console.log(`Vector dim: ${corpusDocs[0].vector.length}`);

  // 构建索引
  const docIds = corpusDocs.map(d => d._id);
  const docVecs = corpusDocs.map(d => d.vector);

  // 找有 qrels 且有 embedding 的 query
  const queryMap = {};
  for (const q of queryDocs) queryMap[q._id] = q;

  const testQueries = Object.keys(qrels)
    .filter(qid => queryMap[qid])
    .map(qid => ({ _id: qid, vector: queryMap[qid].vector }));

  console.log(`Test queries (have both qrels and embeddings): ${testQueries.length}`);
  console.log();

  // 配置
  const configs = [
    { name: 'bge_cosine', method: 'cosine' },
    { name: 'bge_lap_30', method: 'laplacian', topN: 30 },
    { name: 'bge_lap_55', method: 'laplacian', topN: 55 },
    { name: 'bge_lap_100', method: 'laplacian', topN: 100 },
    { name: 'bge_lap_200', method: 'laplacian', topN: 200 },
    { name: 'bge_lap_300', method: 'laplacian', topN: 300 },
  ];

  const K = 10;
  const allResults = {};

  for (const config of configs) {
    console.log(`\n=== ${config.name} ===`);
    const startTime = Date.now();

    let ndcgSum = 0;
    const perQuery = [];

    for (let qi = 0; qi < testQueries.length; qi++) {
      const q = testQueries[qi];
      const qrel = qrels[q._id];

      let ranked;
      if (config.method === 'cosine') {
        ranked = cosineRank(q.vector, docVecs, docIds);
      } else {
        ranked = laplacianSmooth(q.vector, docVecs, docIds, config.topN);
      }

      const ndcg = ndcgAtK(ranked, qrel, K);
      ndcgSum += ndcg;
      perQuery.push({ qid: q._id, ndcg });

      if ((qi + 1) % 100 === 0 || qi === testQueries.length - 1) {
        process.stdout.write(`\r  Progress: ${qi + 1}/${testQueries.length}, running NDCG@${K}: ${(ndcgSum / (qi + 1)).toFixed(4)}`);
      }
    }

    const avgNdcg = ndcgSum / testQueries.length;
    const elapsed = Date.now() - startTime;
    console.log(`\n  NDCG@${K} = ${avgNdcg.toFixed(4)} (${elapsed}ms, ${testQueries.length} queries)`);

    allResults[config.name] = {
      ndcg10: avgNdcg,
      elapsed,
      numQueries: testQueries.length,
      perQuery
    };
  }

  // ===== 输出结果报告 =====
  console.log('\n\n========================================');
  console.log('BGE-large-en-v1.5 NFCorpus Benchmark Results');
  console.log('========================================\n');

  const cosineBase = allResults['bge_cosine'].ndcg10;

  console.log('| Method | NDCG@10 | vs Cosine | Elapsed |');
  console.log('|:--|:--:|:--:|:--:|');

  for (const config of configs) {
    const r = allResults[config.name];
    const delta = r.ndcg10 - cosineBase;
    const pct = cosineBase > 0 ? (delta / cosineBase * 100).toFixed(1) : 'N/A';
    const deltaStr = config.name === 'bge_cosine' ? '--' : `${delta >= 0 ? '+' : ''}${delta.toFixed(4)} (${delta >= 0 ? '+' : ''}${pct}%)`;
    console.log(`| ${config.name} | ${r.ndcg10.toFixed(4)} | ${deltaStr} | ${r.elapsed}ms |`);
  }

  // 对比 Qwen3 结果
  console.log('\n\n========================================');
  console.log('Cross-Embedding Comparison (NFCorpus)');
  console.log('========================================\n');

  const qwenResults = {
    'qwen3_cosine': 0.2195,
    'qwen3_lap_55': 0.2900,
    'qwen3_token_2stage': 0.3220,
    'qwen3_fusion_07': 0.3232,
  };

  console.log('| Embedding | Method | NDCG@10 | vs own cosine |');
  console.log('|:--|:--|:--:|:--:|');

  // Qwen3 rows
  for (const [name, val] of Object.entries(qwenResults)) {
    const delta = val - qwenResults.qwen3_cosine;
    const pct = (delta / qwenResults.qwen3_cosine * 100).toFixed(1);
    const deltaStr = name === 'qwen3_cosine' ? '--' : `+${pct}%`;
    console.log(`| Qwen3-8B (4096d) | ${name.replace('qwen3_', '')} | ${val.toFixed(4)} | ${deltaStr} |`);
  }

  // BGE rows
  for (const config of configs) {
    const r = allResults[config.name];
    const delta = r.ndcg10 - cosineBase;
    const pct = cosineBase > 0 ? (delta / cosineBase * 100).toFixed(1) : 'N/A';
    const deltaStr = config.name === 'bge_cosine' ? '--' : `+${pct}%`;
    console.log(`| BGE-large (1024d) | ${config.name.replace('bge_', '')} | ${r.ndcg10.toFixed(4)} | ${deltaStr} |`);
  }

  // 保存结果
  const resultsPath = path.join(DATA_DIR, 'bge_benchmark_results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(allResults, null, 2));
  console.log(`\nResults saved to: ${resultsPath}`);

  // Stacking 结论
  console.log('\n========================================');
  console.log('STACKING CONCLUSION');
  console.log('========================================\n');

  const bestLap = Math.max(
    allResults['bge_lap_55']?.ndcg10 || 0,
    allResults['bge_lap_100']?.ndcg10 || 0,
    allResults['bge_lap_200']?.ndcg10 || 0,
    allResults['bge_lap_300']?.ndcg10 || 0
  );
  const lapImprovement = (bestLap - cosineBase) / cosineBase * 100;

  console.log(`BGE cosine baseline: ${cosineBase.toFixed(4)}`);
  console.log(`BGE best laplacian:  ${bestLap.toFixed(4)} (+${lapImprovement.toFixed(1)}%)`);
  console.log(`Qwen3 cosine:        0.2195`);
  console.log(`Qwen3 lap_55:        0.2900 (+32.1%)`);
  console.log();

  if (bestLap > cosineBase * 1.01) {
    console.log('VERDICT: Stacking EFFECTIVE -- Laplacian smoothing improves BGE-large embeddings.');
    console.log('The post-processing method is embedding-agnostic and generalizes beyond Qwen3.');
  } else if (bestLap > cosineBase) {
    console.log('VERDICT: Stacking MARGINAL -- Small improvement, may not be significant.');
  } else {
    console.log('VERDICT: Stacking NOT EFFECTIVE on BGE-large -- The method may be embedding-specific.');
  }
}

main().catch(e => { console.error(e); process.exit(1); });
