#!/usr/bin/env node
'use strict';
/**
 * beir_rust_bench.js — BEIR NFCorpus Benchmark：Rust PQ-Chamfer 内核 + Stefan 预取
 *
 * 对比三种方法：
 *   1. cosine       — 标准余弦相似度基线
 *   2. shape_pq64   — JS 侧 PQ-Chamfer + PDE（复制 V5 逻辑）
 *   3. stefan_rust  — Stefan 多轮预取，探针改用 Rust fullscanPqChamfer
 *
 * 用法: node beir_rust_bench.js
 */

const fs      = require('fs');
const path    = require('path');
const readline = require('readline');

// ── Rust Addon ──────────────────────────────────────────────────────────────
// 优先使用有 loadClouds / probePqChamfer 的最新构建（law-vexus 主目录）
// legal-assistant/law-vexus 是旧版本，不含 CloudStore 功能
let LawVexus;
const LAWVEXUS_PATHS = [
  '/home/amd/HEZIMENG/law-vexus',          // 主目录（最新构建，含 CloudStore）
  path.join(__dirname, 'law-vexus'),        // 本地子目录（备用）
];
let loadErr = null;
for (const p of LAWVEXUS_PATHS) {
  try {
    ({ LawVexus } = require(p));
    // 验证 loadClouds 存在
    const tmp = new LawVexus('/tmp/_probe_check');
    if (typeof tmp.loadClouds === 'function') {
      break; // 找到有效版本
    }
    LawVexus = null; // 该版本不支持 loadClouds，继续找
  } catch (e) {
    loadErr = e;
    LawVexus = null;
  }
}
if (!LawVexus) {
  console.error('[致命] 无法加载含 loadClouds 功能的 LawVexus 模块');
  if (loadErr) console.error('  最后错误:', loadErr.message);
  process.exit(1);
}

// ── 路径配置 ─────────────────────────────────────────────────────────────────
const DATA_DIR      = path.join(__dirname, 'beir_data', 'nfcorpus');
const SQLITE_PATH   = path.join(DATA_DIR, 'clouds.sqlite');
const ID_MAP_PATH   = path.join(DATA_DIR, 'id_map.json');
const CORPUS_VECS   = path.join(DATA_DIR, 'corpus_vectors.jsonl');
const QUERY_VECS    = path.join(DATA_DIR, 'query_vectors.jsonl');
const QRELS_PATH    = path.join(DATA_DIR, 'qrels.tsv');
const TOP_N         = 30;     // 初始候选池大小
const DIM           = 4096;
const NS            = 64;     // PQ 子空间数
const SD            = 64;     // 每子空间维度

// ── 工具函数 ─────────────────────────────────────────────────────────────────

/** 余弦相似度 */
function cosSim(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

/** 余弦距离 */
function cosDist(a, b) { return 1 - cosSim(a, b); }

/** PQ 余弦距离：64 个子空间的平均余弦距离 */
function pqCosDist(a, b) {
  if (a.length !== NS * SD) return cosDist(a, b);
  let t = 0;
  for (let s = 0; s < NS; s++) {
    const o = s * SD;
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < SD; i++) { d += a[o + i] * b[o + i]; na += a[o + i] * a[o + i]; nb += b[o + i] * b[o + i]; }
    t += (1 - d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8));
  }
  return t / NS;
}

/** NDCG@k 计算 */
function ndcg(ranked, qrels, k = 10) {
  let d = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++) {
    d += (Math.pow(2, qrels[ranked[i]] || 0) - 1) / Math.log2(i + 2);
  }
  const ideal = Object.values(qrels).sort((a, b) => b - a);
  let id = 0;
  for (let i = 0; i < Math.min(ideal.length, k); i++) {
    id += (Math.pow(2, ideal[i]) - 1) / Math.log2(i + 2);
  }
  return id > 0 ? d / id : 0;
}

/** PDE 对流-扩散求解器（V4 版本，50 步迭代） */
function pde4(C0, adj, U, N) {
  let C = Float64Array.from(C0);
  let mxD = 0;
  for (let i = 0; i < N; i++) if (adj[i].length > mxD) mxD = adj[i].length;
  const dt = Math.min(0.1, mxD > 0 ? 0.8 / mxD : 0.1);
  let Cn = new Float64Array(N);
  for (let t = 0; t < 50; t++) {
    let mx = 0;
    for (let i = 0; i < N; i++) {
      let df = 0, ad = 0;
      for (const e of adj[i]) {
        const j = e.j, w = e.w;
        df += 0.15 * w * (C[j] - C[i]);
        const u1 = U[i * N + j], u2 = U[j * N + i];
        ad += w * ((u2 > 0 ? u2 : 0) * C[j] - (u1 > 0 ? u1 : 0) * C[i]);
      }
      const cn = Math.max(0, C[i] + dt * (df + ad));
      Cn[i] = cn;
      const d = Math.abs(cn - C[i]);
      if (d > mx) mx = d;
    }
    [C, Cn] = [Cn, C];
    if (mx < 1e-3) break;
  }
  return C;
}

// ── 异步加载 JSONL ────────────────────────────────────────────────────────────
function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const arr = [];
    const rl = readline.createInterface({ input: fs.createReadStream(fp, { encoding: 'utf-8' }), crlfDelay: Infinity });
    rl.on('line', l => { if (l.trim()) { try { arr.push(JSON.parse(l)); } catch (e) {} } });
    rl.on('close', () => resolve(arr));
    rl.on('error', reject);
  });
}

/** 加载 qrels TSV（首行为标题行，跳过） */
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

// ── shape_pq64 方法：JS 侧 PQ-Chamfer + PDE ──────────────────────────────────
/**
 * 对给定候选池用 PQ-Chamfer 建图，运行 PDE，按浓度排序
 * @param {Float32Array} qV - query 向量
 * @param {string[]} poolIds - 候选文档 ID 列表
 * @param {Object} dv - {_id: Float32Array} 文档向量
 * @param {Object} corpusSents - {_id: number[][]} 文档句子向量（原始数组）
 * @returns {string[]} - 按 PDE 浓度降序排列的文档 ID
 */
function shapePq64(qV, poolIds, dv, corpusSents) {
  const N = poolIds.length;
  if (N === 0) return [];

  // 获取文档点云（句子列表）
  function getCloud(did) {
    const s = corpusSents[did];
    return s ? s.map(a => new Float32Array(a)) : [dv[did]];
  }

  // PQ-Chamfer 距离：两文档点云间的对称 Chamfer
  function pqChamfer(clA, clB) {
    let sAB = 0;
    for (const a of clA) {
      let mn = Infinity;
      for (const b of clB) { const d = pqCosDist(a, b); if (d < mn) mn = d; }
      sAB += mn;
    }
    let sBA = 0;
    for (const b of clB) {
      let mn = Infinity;
      for (const a of clA) { const d = pqCosDist(a, b); if (d < mn) mn = d; }
      sBA += mn;
    }
    return sAB / clA.length + sBA / clB.length;
  }

  // query-doc 单向 Chamfer（query 为单点）
  function qdChamfer(did) {
    const cl = getCloud(did);
    let sAB = pqCosDist(qV, cl[0]);
    for (let k = 1; k < cl.length; k++) { const d = pqCosDist(qV, cl[k]); if (d < sAB) sAB = d; }
    let sBA = 0;
    for (const b of cl) sBA += pqCosDist(b, qV);
    return sAB + sBA / cl.length;
  }

  const clouds = poolIds.map(id => getCloud(id));
  const knn = 3;

  // 文档-文档距离矩阵
  const dist = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const d = pqChamfer(clouds[i], clouds[j]);
    dist[i * N + j] = d; dist[j * N + i] = d;
  }

  // KNN 邻接表（对称化）
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

  // 对流系数矩阵 U（query 方向场）
  let qN = 0;
  for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);
  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (const e of adj[i]) {
    const j = e.j;
    if (U[i * N + j] || U[j * N + i]) continue;
    let en = 0, dvv = 0;
    for (let d = 0; d < DIM; d++) {
      const df = dv[poolIds[j]][d] - dv[poolIds[i]][d];
      en += df * df; dvv += df * qV[d] * iqn;
    }
    const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * 0.1;
    U[i * N + j] = u0; U[j * N + i] = -u0;
  }

  // 初始浓度场：exp(-2 * chamfer)
  const C0 = new Float64Array(N);
  for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * qdChamfer(poolIds[i]));

  // 求解 PDE
  const Cf = pde4(C0, adj, U, N);
  return poolIds.map((id, i) => ({ id, s: Cf[i] })).sort((a, b) => b.s - a.s).map(x => x.id);
}

// ── stefan_rust 方法：Stefan 多轮预取 + Rust 探针 ─────────────────────────────
/**
 * Stefan 预取：用 Rust fullscanPqChamfer 做全库探针检索
 *
 * @param {Float32Array} qV - query 向量
 * @param {Object[]} initCandidates - [{did, s}] 初始候选（cosine 排序）
 * @param {Object} dv - {_id: Float32Array} 文档向量
 * @param {Object} corpusSents - {_id: number[][]} 文档句子向量
 * @param {LawVexus} vexus - Rust addon 实例（已 loadClouds）
 * @param {Object} idMap - {_id: file_id} 字符串 ID → 整数 file_id
 * @param {Object} reverseMap - {file_id: _id} 整数 file_id → 字符串 ID
 * @param {Object} opts - 超参数
 * @returns {{did: string, s: number}[]} - 按 PDE 浓度降序排列
 */
function stefanRust(qV, initCandidates, dv, corpusSents, vexus, idMap, reverseMap, opts = {}) {
  const {
    initPool   = 30,
    maxRounds  = 3,
    poolBudget = 60,
    knn        = 3,
    probeCount = 8,
    uStr       = 0.3,
    D          = 0.15,
  } = opts;

  // query 单位向量（用于对流方向）
  let qN = 0;
  for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);

  // 获取文档点云（句子向量列表）
  function getCloud(did) {
    const s = corpusSents[did];
    return s ? s.map(a => new Float32Array(a)) : [dv[did]];
  }

  // 获取文档质心
  function getCentroid(did) {
    const cl = getCloud(did);
    const c = new Float32Array(DIM);
    for (const v of cl) for (let d = 0; d < DIM; d++) c[d] += v[d];
    const inv = 1 / cl.length;
    for (let d = 0; d < DIM; d++) c[d] *= inv;
    return c;
  }

  // PQ-Chamfer 距离（两文档点云间）
  function pqChamfer(clA, clB) {
    let sAB = 0;
    for (const a of clA) {
      let mn = Infinity;
      for (const b of clB) { const d = pqCosDist(a, b); if (d < mn) mn = d; }
      sAB += mn;
    }
    let sBA = 0;
    for (const b of clB) {
      let mn = Infinity;
      for (const a of clA) { const d = pqCosDist(a, b); if (d < mn) mn = d; }
      sBA += mn;
    }
    return sAB / clA.length + sBA / clB.length;
  }

  // query-doc Chamfer（query 为单点）
  function qdChamfer(did) {
    const cl = getCloud(did);
    let sAB = pqCosDist(qV, cl[0]);
    for (let k = 1; k < cl.length; k++) { const d = pqCosDist(qV, cl[k]); if (d < sAB) sAB = d; }
    let sBA = 0;
    for (const b of cl) sBA += pqCosDist(b, qV);
    return sAB + sBA / cl.length;
  }

  // 初始池（cosine top-initPool）
  const poolSet = new Set();
  const poolIds = [];
  const limit = Math.min(initPool, initCandidates.length);
  for (let i = 0; i < limit; i++) {
    poolIds.push(initCandidates[i].did);
    poolSet.add(initCandidates[i].did);
  }

  // 距离缓存
  const maxN = poolBudget;
  const distCache = new Float64Array(maxN * maxN);
  const clouds = poolIds.map(id => getCloud(id));
  let N = poolIds.length;

  // 初始距离矩阵
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const d = pqChamfer(clouds[i], clouds[j]);
    distCache[i * maxN + j] = d;
    distCache[j * maxN + i] = d;
  }

  // 从距离缓存构建 KNN 邻接表
  function buildAdj(n, k) {
    const ek = Math.min(k, n - 1);
    const adj = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) {
      const nb = [];
      for (let j = 0; j < n; j++) if (j !== i) nb.push({ j, d: distCache[i * maxN + j] });
      nb.sort((a, b) => a.d - b.d);
      for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) });
    }
    // 对称化
    for (let i = 0; i < n; i++)
      for (const e of adj[i])
        if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });
    return adj;
  }

  // 构建对流系数矩阵 U
  function buildU(adj, n, centroids) {
    const U = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (const e of adj[i]) {
      const j = e.j;
      if (U[i * n + j] || U[j * n + i]) continue;
      let en = 0, dvv = 0;
      for (let d = 0; d < DIM; d++) {
        const df = centroids[j][d] - centroids[i][d];
        en += df * df; dvv += df * qV[d] * iqn;
      }
      const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * uStr;
      U[i * n + j] = u0; U[j * n + i] = -u0;
    }
    return U;
  }

  // PDE 求解（与 V5 逻辑一致，扩散系数 D）
  function solvePDE(C0, adj, U, n) {
    let C = Float64Array.from(C0);
    let mxD = 0;
    for (let i = 0; i < n; i++) if (adj[i].length > mxD) mxD = adj[i].length;
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
        const d = Math.abs(cn - C[i]);
        if (d > mx) mx = d;
      }
      [C, Cn] = [Cn, C];
      if (mx < 1e-3) break;
    }
    return C;
  }

  // Round 0：初始 PDE
  let centroids = poolIds.map(id => getCentroid(id));
  let adj = buildAdj(N, knn);
  let U = buildU(adj, N, centroids);
  const C0 = new Float64Array(N);
  for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * qdChamfer(poolIds[i]));
  let C = solvePDE(C0, adj, U, N);

  let maxFluxR1 = 0;

  // Round 1..R：边界预取，探针改用 Rust fullscanPqChamfer
  for (let r = 1; r <= maxRounds; r++) {
    if (N >= poolBudget) break;

    // 找边界节点：degree < knn*2 且 C_i > median
    const cVals = Array.from(C).sort((a, b) => a - b);
    const median = cVals[cVals.length >> 1];

    const fluxes = [];
    for (let i = 0; i < N; i++) {
      if (adj[i].length >= knn * 2 || C[i] <= median) continue;
      const ci = centroids[i];
      let dot = 0;
      for (let d = 0; d < DIM; d++) dot += (qV[d] * iqn - ci[d]) * (qV[d] * iqn);
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
    const budget = poolBudget - N;
    const perProbe = Math.max(1, Math.ceil(budget / m));

    // 收集已在池中的 file_id（Rust 侧排除列表）
    const excludeFileIds = [];
    for (const did of poolSet) {
      const fid = idMap[did];
      if (fid !== undefined) excludeFileIds.push(fid);
    }

    const newDids = [];
    for (const p of probes) {
      if (newDids.length >= budget) break;

      // 构造探针点云：边界节点的句子向量，转为 Buffer[]
      // probe_cloud = 边界文档的所有句子向量
      const probeDocId = poolIds[p.idx];
      const probeCloud = getCloud(probeDocId).map(v => Buffer.from(v.buffer, v.byteOffset, v.byteLength));

      // 调用 Rust 探针检索：全库 PQ-Chamfer，返回 [{id: file_id, score}]
      let rustHits;
      try {
        rustHits = vexus.probePqChamfer(probeCloud, perProbe + excludeFileIds.length, excludeFileIds);
      } catch (e) {
        // Rust 调用失败时回退到 JS 侧 cosine 检索
        console.warn(`[警告] probePqChamfer 失败，回退 cosine: ${e.message}`);
        const probeVec = new Float32Array(DIM);
        const dc = centroids[p.idx];
        for (let d = 0; d < DIM; d++) probeVec[d] = 0.7 * qV[d] + 0.3 * dc[d];
        rustHits = [];
        // 此处不做全库扫描回退（避免超时），直接跳过
        break;
      }

      // 将 file_id 转回字符串 _id，过滤已在池中的
      let taken = 0;
      for (const hit of rustHits) {
        if (taken >= perProbe || newDids.length >= budget) break;
        const did = reverseMap[hit.id];
        if (!did || poolSet.has(did)) continue;
        newDids.push(did);
        poolSet.add(did);
        taken++;
      }
    }

    if (newDids.length === 0) break;

    // 将新文档加入池
    const oldN = N;
    for (const did of newDids) {
      poolIds.push(did);
      clouds.push(getCloud(did));
    }
    N = poolIds.length;

    // 增量更新距离矩阵（只计算新文档与已有文档的距离）
    for (let i = oldN; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const d = pqChamfer(clouds[i], clouds[j]);
        distCache[i * maxN + j] = d;
        distCache[j * maxN + i] = d;
      }
    }

    // 重建图、对流系数、初始浓度
    centroids = poolIds.map(id => getCentroid(id));
    adj = buildAdj(N, knn);
    U = buildU(adj, N, centroids);

    // 新初始浓度：旧节点保留上轮 PDE 结果，新节点用 exp(-2*chamfer)
    const C0new = new Float64Array(N);
    for (let i = 0; i < oldN; i++) C0new[i] = C[i];
    for (let i = oldN; i < N; i++) C0new[i] = Math.exp(-2 * qdChamfer(poolIds[i]));
    C = solvePDE(C0new, adj, U, N);
  }

  return poolIds.map((id, i) => ({ did: id, s: C[i] })).sort((a, b) => b.s - a.s);
}

// ── 主函数 ───────────────────────────────────────────────────────────────────
async function main() {
  console.log('');
  console.log('═'.repeat(60));
  console.log('  BEIR NFCorpus — Rust PQ-Chamfer + Stefan 预取 Benchmark');
  console.log('═'.repeat(60));

  // 1. 检查 SQLite 数据库
  if (!fs.existsSync(SQLITE_PATH)) {
    console.error(`[错误] 未找到 SQLite 数据库: ${SQLITE_PATH}`);
    console.error('       请先运行: python3 beir_to_sqlite.py');
    process.exit(1);
  }
  if (!fs.existsSync(ID_MAP_PATH)) {
    console.error(`[错误] 未找到 ID 映射表: ${ID_MAP_PATH}`);
    process.exit(1);
  }

  // 2. 加载 ID 映射表，并构建反向映射
  console.log('[1/5] 加载 ID 映射表...');
  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [strId, fileId] of Object.entries(idMap)) {
    reverseMap[fileId] = strId;
  }
  console.log(`      ${Object.keys(idMap).length} 个文档 ID 映射`);

  // 3. 初始化 Rust addon 并加载点云
  console.log('[2/5] 初始化 LawVexus 并加载点云数据库...');
  const vexus = new LawVexus('/tmp/beir_rust_bench_store');
  const t0Load = Date.now();
  const nDocs = vexus.loadClouds(SQLITE_PATH);
  const tLoad = (Date.now() - t0Load) / 1000;
  console.log(`      加载完成：${nDocs} 个文档点云，耗时 ${tLoad.toFixed(1)}s`);

  // 4. 加载语料库向量和查询数据
  console.log('[3/5] 加载 BEIR 数据...');
  const corpusObjs = await loadJsonl(CORPUS_VECS);
  const queryObjs  = await loadJsonl(QUERY_VECS);
  const qrels      = loadQrels(QRELS_PATH);

  // 构建语料库向量和句子映射
  const dv = {};          // {_id: Float32Array}
  const corpusSents = {}; // {_id: number[][]}（原始 JSON 数组，按需转 Float32Array）
  for (const obj of corpusObjs) {
    dv[obj._id] = new Float32Array(obj.vector);
    if (obj.sentences && obj.sentences.length > 1) {
      corpusSents[obj._id] = obj.sentences;
    }
  }

  // 过滤有 qrels 和 query 向量的查询
  const qV = {};
  for (const obj of queryObjs) {
    if (obj.vector) qV[obj._id] = new Float32Array(obj.vector);
  }
  const allDids = Object.keys(dv);
  const qids    = Object.keys(qrels).filter(q => qV[q]);

  console.log(`      语料库: ${allDids.length} 个文档，查询: ${qids.length} 个`);

  // 5. 运行 Benchmark
  console.log('[4/5] 运行 Benchmark...');
  console.log(`      方法: cosine | shape_pq64 | stefan_rust`);
  console.log(`      TOP_N=${TOP_N}, poolBudget=60, maxRounds=3\n`);

  const scores = { cosine: 0, shape_pq64: 0, stefan_rust: 0 };
  const t0Bench = Date.now();

  const MAX_Q = parseInt(process.env.MAX_Q || '0') || qids.length;
  const runQ = Math.min(MAX_Q, qids.length);
  console.log(`      实际运行: ${runQ} queries\n`);
  for (let qi = 0; qi < runQ; qi++) {
    const qid = qids[qi];
    const qVec = qV[qid];
    const qr   = qrels[qid];

    // cosine baseline：全库排序，取 top-10
    const cosHits = allDids
      .map(did => ({ did, s: cosSim(qVec, dv[did]) }))
      .sort((a, b) => b.s - a.s);
    scores.cosine += ndcg(cosHits.slice(0, 10).map(x => x.did), qr);

    // 候选池（cosine top-N）
    const candidatePool = cosHits.slice(0, TOP_N);
    const poolIds = candidatePool.map(x => x.did);

    // shape_pq64
    const pq64Ranked = shapePq64(qVec, poolIds, dv, corpusSents);
    scores.shape_pq64 += ndcg(pq64Ranked.slice(0, 10), qr);

    // stefan_rust：Stefan 多轮预取，Rust 探针
    const stefanRanked = stefanRust(
      qVec, candidatePool, dv, corpusSents, vexus, idMap, reverseMap,
      { initPool: TOP_N, maxRounds: 1, poolBudget: 45, knn: 3, probeCount: 8, uStr: 0.3, D: 0.15 }
    );
    scores.stefan_rust += ndcg(stefanRanked.slice(0, 10).map(x => x.did), qr);

    // 每 50 个查询打印进度
    if ((qi + 1) % 50 === 0 || qi === qids.length - 1) {
      const elapsed = (Date.now() - t0Bench) / 1000;
      const qps = (qi + 1) / elapsed;
      const eta = qps > 0 ? (qids.length - qi - 1) / qps : 0;
      const n = qi + 1;
      process.stdout.write(
        `\r  进度: ${n}/${qids.length} (${(n / qids.length * 100).toFixed(1)}%) ` +
        `| ${qps.toFixed(1)} q/s | ETA ${eta.toFixed(0)}s` +
        ' '.repeat(10)
      );
    }
  }

  const totalSec = (Date.now() - t0Bench) / 1000;
  process.stdout.write('\n');
  console.log(`\n[5/5] Benchmark 完成，总耗时 ${totalSec.toFixed(1)}s\n`);

  // 6. 输出结果表格
  const nQ = runQ;
  const cosNdcg = scores.cosine / nQ;
  const pq64Ndcg = scores.shape_pq64 / nQ;
  const stefanNdcg = scores.stefan_rust / nQ;

  function fmtDiff(val, base) {
    const pct = ((val - base) / base * 100);
    return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
  }

  console.log('  ┌──────────────────┬──────────┬─────────────┬─────────────┐');
  console.log('  │ 方法             │ NDCG@10  │ vs cosine   │ vs pq64     │');
  console.log('  ├──────────────────┼──────────┼─────────────┼─────────────┤');
  console.log(`  │ cosine           │ ${cosNdcg.toFixed(4).padStart(8)} │ ${'baseline'.padStart(11)} │ ${fmtDiff(cosNdcg, pq64Ndcg).padStart(11)} │`);
  console.log(`  │ shape_pq64       │ ${pq64Ndcg.toFixed(4).padStart(8)} │ ${fmtDiff(pq64Ndcg, cosNdcg).padStart(11)} │ ${'baseline'.padStart(11)} │`);
  console.log(`  │ stefan_rust      │ ${stefanNdcg.toFixed(4).padStart(8)} │ ${fmtDiff(stefanNdcg, cosNdcg).padStart(11)} │ ${fmtDiff(stefanNdcg, pq64Ndcg).padStart(11)} │`);
  console.log('  └──────────────────┴──────────┴─────────────┴─────────────┘');
  console.log('');
  console.log(`  查询总数: ${nQ} | 语料库: ${allDids.length} | 每查询平均: ${(totalSec / nQ * 1000).toFixed(1)}ms`);
  console.log('');
}

main().catch(err => {
  console.error('[致命错误]', err);
  process.exit(1);
});
