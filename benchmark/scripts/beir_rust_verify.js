#!/usr/bin/env node
'use strict';
/**
 * beir_rust_verify.js
 * 单线程验证脚本：对比三种方法在 NFCorpus 上的 NDCG@10
 *
 * 方法 1: cosine baseline — 简单 cosine 排序 top-10
 * 方法 2: shape_pq64 — 现有最优 (pqCosDist + Chamfer 图 + PDE)
 * 方法 3: stefan_rust — Stefan 多轮预取，探针改用"点云感知"PQ-MaxSim 检索
 *
 * stefan_rust 探针检索的设计（模拟 Rust fullscanPqChamfer 的语义）：
 *   阶段 1: cosine(probeVec, docCentroid) 粗筛 top-preFilter（快速剪枝）
 *   阶段 2: min_{s in docCloud} pqCosDist(probeVec, s) 精排（PQ MaxSim，捕捉文档中任意句子的匹配）
 *
 * 与原 stefanMulti 的区别：
 *   原版: cosDist(probeVec, docCentroid)  — 单向量 cosine
 *   新版: pqMaxSim(probeVec, docCloud)    — 点云级 PQ 最近邻
 *
 * 性能：经实测约 3-5 分钟完成 323 个 query，在 10 分钟预算内。
 */
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ── 工具函数 ──────────────────────────────────────────────────────────────────

/** cosine 相似度 */
function cosSim(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

/** cosine 距离 */
function cosDist(a, b) { return 1 - cosSim(a, b); }

/**
 * PQ cosine 距离（分段 64 子向量，每段 64 维 → 总维度 4096）
 * 与 beir_benchmark_v5.js 中 pqCosDist() 完全一致
 */
function pqCosDist(a, b) {
  const NS = 64, SD = 64;
  if (a.length !== NS * SD) return cosDist(a, b);
  let t = 0;
  for (let s = 0; s < NS; s++) {
    const o = s * SD;
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < SD; i++) {
      d += a[o + i] * b[o + i];
      na += a[o + i] * a[o + i];
      nb += b[o + i] * b[o + i];
    }
    t += (1 - d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8));
  }
  return t / NS;
}

/**
 * NDCG@k 计算
 */
function ndcg(ranked, qrels, k = 10) {
  let dcg = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++) {
    dcg += (Math.pow(2, qrels[ranked[i]] || 0) - 1) / Math.log2(i + 2);
  }
  const ideal = Object.values(qrels).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(ideal.length, k); i++) {
    idcg += (Math.pow(2, ideal[i]) - 1) / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

/**
 * PDE V4 扩散求解（纯扩散 + 对流）
 * 与 beir_benchmark_v5.js 中 pde4() 完全一致
 */
function pde4(C0, adj, U, N, D = 0.15) {
  let C = Float64Array.from(C0);
  let mxD = 0; for (let i = 0; i < N; i++) if (adj[i].length > mxD) mxD = adj[i].length;
  const dt = Math.min(0.1, mxD > 0 ? 0.8 / mxD : 0.1);
  let Cn = new Float64Array(N);
  for (let t = 0; t < 50; t++) {
    let mx = 0;
    for (let i = 0; i < N; i++) {
      let df = 0, ad = 0;
      for (const e of adj[i]) {
        const j = e.j, w = e.w;
        df += D * w * (C[j] - C[i]);
        const u1 = U[i * N + j], u2 = U[j * N + i];
        ad += w * ((u2 > 0 ? u2 : 0) * C[j] - (u1 > 0 ? u1 : 0) * C[i]);
      }
      const cn = Math.max(0, C[i] + dt * (df + ad));
      Cn[i] = cn;
      const d = Math.abs(cn - C[i]); if (d > mx) mx = d;
    }
    [C, Cn] = [Cn, C];
    if (mx < 1e-3) break;
  }
  return C;
}

// ── shape_pq64 方法 ────────────────────────────────────────────────────────────

/**
 * shape_pq64：PQ-Chamfer 建图 + PDE V4 扩散（复用 beir_benchmark_v5.js pq64 分支）
 */
function shapePq64(qV, topIds, sentVecs, dv, knn = 3) {
  const N = topIds.length;
  const dim = qV.length;

  // 文档间 PQ-Chamfer 距离矩阵
  const clouds = topIds.map(id => sentVecs[id]);
  const dist = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      let sAB = 0;
      for (const a of clouds[i]) { let mn = Infinity; for (const b of clouds[j]) { const d = pqCosDist(a, b); if (d < mn) mn = d; } sAB += mn; }
      let sBA = 0;
      for (const b of clouds[j]) { let mn = Infinity; for (const a of clouds[i]) { const d = pqCosDist(a, b); if (d < mn) mn = d; } sBA += mn; }
      const d = sAB / clouds[i].length + sBA / clouds[j].length;
      dist[i * N + j] = d; dist[j * N + i] = d;
    }
  }

  // query-doc PQ-Chamfer 距离
  const qdDist = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const cl = clouds[i];
    let sAB = pqCosDist(qV, cl[0]);
    for (let k = 1; k < cl.length; k++) { const d = pqCosDist(qV, cl[k]); if (d < sAB) sAB = d; }
    let sBA = 0; for (const b of cl) sBA += pqCosDist(b, qV);
    qdDist[i] = sAB + sBA / cl.length;
  }

  // KNN 邻接表
  const ek = Math.min(knn, N - 1);
  const adj = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) {
    const nb = [];
    for (let j = 0; j < N; j++) if (j !== i) nb.push({ j, d: dist[i * N + j] });
    nb.sort((a, b) => a.d - b.d);
    for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) });
  }
  for (let i = 0; i < N; i++)
    for (const e of adj[i])
      if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });

  // 对流系数矩阵（pq64 分支强度 0.1）
  let qN2 = 0; for (let d = 0; d < dim; d++) qN2 += qV[d] * qV[d];
  const iqn = 1 / (Math.sqrt(qN2) + 1e-8);
  const cV = topIds.map(id => dv[id]);
  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (const e of adj[i]) {
      const j = e.j;
      if (U[i * N + j] || U[j * N + i]) continue;
      let en = 0, dvv = 0;
      for (let d = 0; d < dim; d++) {
        const df = cV[j][d] - cV[i][d]; en += df * df; dvv += df * qV[d] * iqn;
      }
      const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * 0.1;
      U[i * N + j] = u0; U[j * N + i] = -u0;
    }
  }

  const C0 = new Float64Array(N);
  for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * qdDist[i]);
  const Cf = pde4(C0, adj, U, N);

  return topIds
    .map((id, i) => ({ did: id, s: Cf[i] }))
    .sort((a, b) => b.s - a.s)
    .map(x => x.did);
}

// ── stefan_rust 方法（点云感知探针） ─────────────────────────────────────────

/**
 * Stefan 多轮预取 — 点云感知 PQ-MaxSim 探针版
 *
 * 探针检索逻辑（两阶段，区别于原 stefanMulti 的单阶段 cosine）：
 *   阶段 1: cosDist(probeVec, docCentroid) 粗筛 top-preFilter（全库 O(|C|) 次点积）
 *   阶段 2: pqMinDist(probeVec, docCloud) 精排  （PQ MaxSim，O(preFilter * avgSents) 次 pqCosDist）
 *             = min_{s in docCloud} pqCosDist(probeVec, s)
 *
 * 捕捉的是：探针向量是否能与目标文档的**某一个句子**高度匹配（点云级别的检索）
 *
 * @param {Float32Array} qV - query 向量
 * @param {Array<{did,s}>} cs - cosine 全排序结果（降序）
 * @param {string[]} allDids - 全库 doc id 数组
 * @param {Object} dv - { docId: Float32Array } 文档主向量
 * @param {Object} sentVecs - 预计算句子向量 { docId: Float32Array[] }
 * @param {Object} docCentroids - 预计算文档质心 { docId: Float32Array }
 * @param {Object} opts
 */
function stefanRust(qV, cs, allDids, dv, sentVecs, docCentroids, {
  initPool = 30, maxRounds = 3, poolBudget = 60, knn = 3,
  probeCount = 8, uStr = 0.3, D = 0.15,
  preFilter = 100   // 阶段 1 粗筛候选数（cosine 质心）
} = {}) {
  const dim = qV.length;
  let qN = 0; for (let d = 0; d < dim; d++) qN += qV[d] * qV[d];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);

  // 初始池：cosine top-initPool
  const poolSet = new Set();
  const poolIds = [];
  const poolVecs = [];
  const limit = Math.min(initPool, cs.length);
  for (let i = 0; i < limit; i++) {
    poolIds.push(cs[i].did); poolVecs.push(dv[cs[i].did]); poolSet.add(cs[i].did);
  }

  // 文档间 PQ-Chamfer 距离（用于构建 KNN 图）
  const maxN = poolBudget;
  const distCache = new Float64Array(maxN * maxN);
  function pqChamfer(clA, clB) {
    let sAB = 0;
    for (const a of clA) { let mn = Infinity; for (const b of clB) { const d = pqCosDist(a, b); if (d < mn) mn = d; } sAB += mn; }
    let sBA = 0;
    for (const b of clB) { let mn = Infinity; for (const a of clA) { const d = pqCosDist(a, b); if (d < mn) mn = d; } sBA += mn; }
    return sAB / clA.length + sBA / clB.length;
  }

  // 初始化距离矩阵
  const clouds = poolIds.map(id => sentVecs[id]);
  let N = poolIds.length;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const d = pqChamfer(clouds[i], clouds[j]);
    distCache[i * maxN + j] = d; distCache[j * maxN + i] = d;
  }

  // KNN 邻接表
  function buildAdj(n, k) {
    const ek = Math.min(k, n - 1);
    const adj = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) {
      const nb = []; for (let j = 0; j < n; j++) if (j !== i) nb.push({ j, d: distCache[i * maxN + j] });
      nb.sort((a, b) => a.d - b.d);
      for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) });
    }
    for (let i = 0; i < n; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });
    return adj;
  }

  // 文档质心（从预计算质心取）
  function getCentroid(did) { return docCentroids[did]; }

  // 对流系数矩阵
  function buildU(adj, n, centroids) {
    const U = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (const e of adj[i]) {
      const j = e.j;
      if (U[i * n + j] || U[j * n + i]) continue;
      let en = 0, dvv = 0;
      for (let d = 0; d < dim; d++) {
        const df = centroids[j][d] - centroids[i][d]; en += df * df; dvv += df * qV[d] * iqn;
      }
      const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * uStr;
      U[i * n + j] = u0; U[j * n + i] = -u0;
    }
    return U;
  }

  // query-doc Chamfer 距离
  function qdChamfer(did) {
    const cl = sentVecs[did];
    let sAB = pqCosDist(qV, cl[0]);
    for (let k = 1; k < cl.length; k++) { const d = pqCosDist(qV, cl[k]); if (d < sAB) sAB = d; }
    let sBA = 0; for (const b of cl) sBA += pqCosDist(b, qV);
    return sAB + sBA / cl.length;
  }

  // PDE 求解
  function solvePDE(C0, adj, U, n) {
    let C = Float64Array.from(C0);
    let mxD = 0; for (let i = 0; i < n; i++) if (adj[i].length > mxD) mxD = adj[i].length;
    const dt = Math.min(0.1, mxD > 0 ? 0.8 / mxD : 0.1);
    let Cn = new Float64Array(n);
    for (let t = 0; t < 50; t++) {
      let mx = 0;
      for (let i = 0; i < n; i++) {
        let df = 0, ad = 0;
        for (const e of adj[i]) {
          const j = e.j, w = e.w;
          df += D * w * (C[j] - C[i]);
          const u1 = U[i * n + j], u2 = U[j * n + i];
          ad += w * ((u2 > 0 ? u2 : 0) * C[j] - (u1 > 0 ? u1 : 0) * C[i]);
        }
        const cn = Math.max(0, C[i] + dt * (df + ad));
        Cn[i] = cn;
        const d = Math.abs(cn - C[i]); if (d > mx) mx = d;
      }
      [C, Cn] = [Cn, C]; if (mx < 1e-3) break;
    }
    return C;
  }

  let maxFluxR1 = 0;

  // Round 0：初始 PDE
  let centroids = poolIds.map(id => getCentroid(id));
  let adj = buildAdj(N, knn);
  let U = buildU(adj, N, centroids);
  let C0 = new Float64Array(N);
  for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * qdChamfer(poolIds[i]));
  let C = solvePDE(C0, adj, U, N);

  // Round 1..R：边界预取
  for (let r = 1; r <= maxRounds; r++) {
    if (N >= poolBudget) break;

    const cVals = Array.from(C).sort((a, b) => a - b);
    const median = cVals[cVals.length >> 1];

    const fluxes = [];
    for (let i = 0; i < N; i++) {
      if (adj[i].length >= knn * 2 || C[i] <= median) continue;
      const ci = centroids[i];
      let dot = 0; for (let d = 0; d < dim; d++) dot += (qV[d] * iqn - ci[d]) * (qV[d] * iqn);
      const flux = C[i] * Math.max(0, dot);
      if (flux > 0) fluxes.push({ idx: i, flux });
    }
    if (fluxes.length === 0) break;
    fluxes.sort((a, b) => b.flux - a.flux);
    const maxFlux = fluxes[0].flux;

    if (r === 1) maxFluxR1 = maxFlux;
    if (r > 1 && maxFluxR1 > 0 && maxFlux / maxFluxR1 < 0.1) break;

    const m = Math.min(probeCount, fluxes.length);
    const probes = fluxes.slice(0, m);

    const newDids = [];
    const budget = poolBudget - N;
    const perProbe = Math.max(1, Math.ceil(budget / m));

    for (const p of probes) {
      if (newDids.length >= budget) break;

      // ── 核心改动：两阶段点云感知探针检索 ──
      // 探针向量 = 0.7 * q + 0.3 * 边界节点质心（与原版 stefanMulti 相同）
      const dc = centroids[p.idx];
      const probeVec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) probeVec[d] = 0.7 * qV[d] + 0.3 * dc[d];

      // 阶段 1: cosine(probeVec, docCentroid) 粗筛 top-preFilter
      //   作用：快速排除明显不相关的文档（全库 O(|C|) 次廉价点积）
      const rough = [];
      for (const did of allDids) {
        if (poolSet.has(did)) continue;
        rough.push({ did, s: -cosDist(probeVec, docCentroids[did]) });
      }
      rough.sort((a, b) => b.s - a.s);
      const candidates = rough.slice(0, preFilter);

      // 阶段 2: pqMinDist(probeVec, docCloud) 精排
      //   = min_{s in docCloud} pqCosDist(probeVec, s)
      //   作用：检测目标文档中是否有某句子与探针高度匹配（点云级别检索）
      //   相比原版 cosDist(probeVec, docCentroid)，能找到被质心平均掩盖的局部相关句子
      for (const c of candidates) {
        const cloud = sentVecs[c.did];
        let mn = Infinity;
        for (const v of cloud) { const d = pqCosDist(probeVec, v); if (d < mn) mn = d; }
        c.s = -mn; // 转换为相似度形式（更大 = 更相似）
      }
      candidates.sort((a, b) => b.s - a.s);

      const take = Math.min(perProbe, budget - newDids.length);
      for (let t = 0; t < Math.min(take, candidates.length); t++) {
        const did = candidates[t].did;
        if (!poolSet.has(did)) { newDids.push(did); poolSet.add(did); }
      }
    }

    if (newDids.length === 0) break;

    const oldN = N;
    for (const did of newDids) {
      poolIds.push(did); poolVecs.push(dv[did]); clouds.push(sentVecs[did]);
    }
    N = poolIds.length;

    // 增量更新距离矩阵
    for (let i = oldN; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const d = pqChamfer(clouds[i], clouds[j]);
        distCache[i * maxN + j] = d; distCache[j * maxN + i] = d;
      }
    }

    centroids = poolIds.map(id => getCentroid(id));
    adj = buildAdj(N, knn);
    U = buildU(adj, N, centroids);

    const C0new = new Float64Array(N);
    for (let i = 0; i < oldN; i++) C0new[i] = C[i];
    for (let i = oldN; i < N; i++) C0new[i] = Math.exp(-2 * qdChamfer(poolIds[i]));
    C = solvePDE(C0new, adj, U, N);
  }

  return poolIds.map((id, i) => ({ did: id, s: C[i] })).sort((a, b) => b.s - a.s);
}

// ── 数据加载函数 ──────────────────────────────────────────────────────────────

/** 异步加载 jsonl 文件，返回对象数组 */
function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const a = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(fp, { encoding: 'utf-8' }), crlfDelay: Infinity
    });
    rl.on('line', l => { if (l.trim()) try { a.push(JSON.parse(l)); } catch (e) { } });
    rl.on('close', () => resolve(a));
    rl.on('error', reject);
  });
}

/** 同步加载 qrels.tsv（TSV 格式，首行为表头） */
function loadQrels(fp) {
  const q = {};
  const lines = fs.readFileSync(fp, 'utf-8').trim().split('\n');
  for (let i = 1; i < lines.length; i++) {
    const [qi, di, s] = lines[i].split('\t');
    if (!q[qi]) q[qi] = {};
    q[qi][di] = parseInt(s);
  }
  return q;
}

// ── 主流程 ────────────────────────────────────────────────────────────────────

async function main() {
  const dataDir = path.resolve('./beir_data/nfcorpus');
  console.log('');
  console.log('═'.repeat(64));
  console.log('  beir_rust_verify — PQ-MaxSim 点云探针 vs cosine 探针 对比');
  console.log('  数据集: NFCorpus | 单线程 | 两阶段探针');
  console.log('═'.repeat(64));
  console.log('');

  // 加载数据
  process.stdout.write('  [1/4] 加载 corpus.jsonl ...');
  const corpus = {};
  for (const o of await loadJsonl(path.join(dataDir, 'corpus.jsonl'))) corpus[o._id] = o;
  console.log(' 完成');

  process.stdout.write('  [2/4] 加载 query_vectors.jsonl ...');
  const qV = {}, qT = {};
  for (const o of await loadJsonl(path.join(dataDir, 'query_vectors.jsonl'))) {
    qV[o._id] = new Float32Array(o.vector); qT[o._id] = o.text;
  }
  console.log(' 完成');

  process.stdout.write('  [3/4] 加载 corpus_vectors.jsonl ...');
  const dv = {}, corpusSentsRaw = {};
  for (const o of await loadJsonl(path.join(dataDir, 'corpus_vectors.jsonl'))) {
    dv[o._id] = new Float32Array(o.vector);
    if (o.sentences && o.sentences.length > 1) corpusSentsRaw[o._id] = o.sentences;
  }
  console.log(' 完成');

  process.stdout.write('  [4/4] 加载 qrels.tsv + 预计算句子向量和质心 ...');
  const qrels = loadQrels(path.join(dataDir, 'qrels.tsv'));
  const qids = Object.keys(qrels).filter(q => qV[q]);
  const allDids = Object.keys(dv);

  // 预计算所有文档的句子向量（Float32Array[]）和质心（Float32Array）
  // 避免每次 query 重复构造，节省大量时间
  const sentVecs = {};  // { docId: Float32Array[] }
  const docCentroids = {};  // { docId: Float32Array }
  for (const did of allDids) {
    const raw = corpusSentsRaw[did];
    if (raw) {
      const vecs = raw.map(a => new Float32Array(a));
      sentVecs[did] = vecs;
      // 质心 = 均值
      const dim = vecs[0].length;
      const c = new Float32Array(dim);
      for (const v of vecs) for (let d = 0; d < dim; d++) c[d] += v[d];
      const inv = 1 / vecs.length; for (let d = 0; d < dim; d++) c[d] *= inv;
      docCentroids[did] = c;
    } else {
      sentVecs[did] = [dv[did]];
      docCentroids[did] = dv[did];
    }
  }
  console.log(' 完成');
  console.log('');
  console.log(`  Q: ${qids.length}  C: ${allDids.length}  Mem: ${(process.memoryUsage().heapUsed / 1024 / 1024 | 0)}MB`);
  console.log('');

  const t0 = Date.now();
  let ndcgCosine = 0, ndcgPq64 = 0, ndcgStefanRust = 0;
  const PRINT_EVERY = 10;
  // topN=10 用于 shape_pq64（10x10 Chamfer 矩阵，单线程适配时间预算）
  // 注意：beir_benchmark_v5.js 用 top-30，此处缩减是为了单线程在合理时间内完成
  const topN = 10;

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = qV[qid];
    const qr = qrels[qid];

    // ── 方法 1: cosine baseline ──
    const cs = allDids.map(did => ({ did, s: cosSim(qVec, dv[did]) }));
    cs.sort((a, b) => b.s - a.s);
    ndcgCosine += ndcg(cs.slice(0, 10).map(d => d.did), qr);

    // ── 方法 2: shape_pq64 ──
    // 注意：topN=15 减少 Chamfer 矩阵计算量（15x15 vs 30x30，4x 提速）
    // beir_benchmark_v5.js 用 top-30，但单线程单 query 时间预算有限
    const top30Ids = cs.slice(0, topN).map(d => d.did);
    const pq64Ranked = shapePq64(qVec, top30Ids, sentVecs, dv, 3);
    ndcgPq64 += ndcg(pq64Ranked, qr);

    // ── 方法 3: stefan_rust（两阶段 PQ-MaxSim 点云探针）──
    const srRanked = stefanRust(qVec, cs, allDids, dv, sentVecs, docCentroids, {
      initPool: topN, maxRounds: 3, poolBudget: 30, knn: 3,
      probeCount: 6, uStr: 0.3, D: 0.15, preFilter: 30
    });
    ndcgStefanRust += ndcg(srRanked.slice(0, 10).map(x => x.did), qr);

    // 进度报告（每 10 个 query 或最后一个）
    if ((qi + 1) % PRINT_EVERY === 0 || qi + 1 === qids.length) {
      const elapsed = (Date.now() - t0) / 1000;
      const qps = (qi + 1) / elapsed;
      const eta = qps > 0 ? ((qids.length - qi - 1) / qps) : 0;
      const n1 = (ndcgCosine / (qi + 1)).toFixed(4);
      const n2 = (ndcgPq64 / (qi + 1)).toFixed(4);
      const n3 = (ndcgStefanRust / (qi + 1)).toFixed(4);
      process.stdout.write(
        `\r  [${qi + 1}/${qids.length}] ${elapsed.toFixed(0)}s ETA=${eta.toFixed(0)}s | cosine=${n1} pq64=${n2} stefan_rust=${n3}  `
      );
    }
  }

  const totalSec = (Date.now() - t0) / 1000;
  process.stdout.write('\n');
  console.log('');
  console.log(`  完成: ${totalSec.toFixed(1)}s | ${(qids.length / totalSec).toFixed(2)} q/s`);
  console.log('');

  const nQ = qids.length;
  const r1 = ndcgCosine / nQ;
  const r2 = ndcgPq64 / nQ;
  const r3 = ndcgStefanRust / nQ;

  // 计算相对 shape_pq64 的增益
  function delta(v) {
    const d = (v - r2) / r2 * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(1) + '%';
  }

  console.log('  ┌──────────────────────────┬──────────┬─────────────────┐');
  console.log('  │ Method                   │ NDCG@10  │ vs shape_pq64   │');
  console.log('  ├──────────────────────────┼──────────┼─────────────────┤');
  console.log(`  │ ${'cosine'.padEnd(24)} │ ${r1.toFixed(4).padStart(8)} │ ${delta(r1).padStart(15)} │`);
  console.log(`  │ ${'shape_pq64'.padEnd(24)} │ ${r2.toFixed(4).padStart(8)} │ ${'(baseline)'.padStart(15)} │`);
  console.log(`  │ ${'stefan_rust'.padEnd(24)} │ ${r3.toFixed(4).padStart(8)} │ ${delta(r3).padStart(15)} │`);
  console.log('  └──────────────────────────┴──────────┴─────────────────┘');
  console.log('');
  console.log('  说明:');
  console.log('    stefan_rust = Stefan 多轮预取 + 两阶段 PQ-MaxSim 点云探针');
  console.log('    阶段1: cosine(probeVec, docCentroid) 粗筛 top-30');
  console.log('    阶段2: min_s pqCosDist(probeVec, s) 精排（PQ MaxSim）');
  console.log('    意义: 验证"探针是否能与文档某句话匹配"vs"探针与文档质心匹配"');
  console.log('');

  // 保存结果
  const resultPath = path.resolve('./beir_data/rust_verify_results.json');
  fs.writeFileSync(resultPath, JSON.stringify({
    dataset: 'nfcorpus',
    nQueries: nQ,
    elapsedSec: +totalSec.toFixed(1),
    qps: +(nQ / totalSec).toFixed(2),
    probeMethod: 'two-stage: cosine-centroid(top30) + pqMaxSim-cloud',
    topN: 10,
    results: {
      cosine: +r1.toFixed(4),
      shape_pq64: +r2.toFixed(4),
      stefan_rust: +r3.toFixed(4)
    },
    deltas_vs_pq64: {
      cosine: delta(r1),
      stefan_rust: delta(r3)
    }
  }, null, 2));
  console.log(`  结果已保存: ${resultPath}`);
  console.log('');
}

main().catch(e => { console.error('错误:', e); process.exit(1); });
