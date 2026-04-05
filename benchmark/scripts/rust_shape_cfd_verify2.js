#!/usr/bin/env node
'use strict';
/**
 * rust_shape_cfd_verify2.js
 * 精细对比验证：4 种方法
 *   1. cosine_centroid    — JS: cosSim(q, docCentroid)，top-30
 *   2. js_vt_aligned_r0  — JS: vtAligned PDE，初始池 cosine-centroid top-30，无预取（等效 Rust 结构）
 *   3. rust_shape_cfd    — Rust: shapeCfdPipeline(q, k=10, top_n=30)
 *   4. js_vt_v7          — JS: vtV7Combined (vtAligned + v7 cosine 预取)  [当前 SOTA = 0.2844]
 *
 * 目标：确认 Rust 与方法2（同结构 JS）是否对齐，并找出与 vt_v7 的差距来源
 *
 * 运行: MAX_Q=10 node rust_shape_cfd_verify2.js  (烟雾)
 *       MAX_Q=50 node rust_shape_cfd_verify2.js  (快速验证)
 *       node rust_shape_cfd_verify2.js            (完整 323 query)
 */
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const SQLITE_PATH = path.join(DATA_DIR, 'clouds.sqlite');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');
const MAX_Q = parseInt(process.env.MAX_Q || '0');

// ── 工具函数 ──────────────────────────────────────────────────────────────────
function cosSim(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

const DIM = 4096, NS = 64, SD = 64;

function ndcg(ranked, qrels, k = 10) {
  let dcg = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++)
    dcg += (Math.pow(2, qrels[ranked[i]] || 0) - 1) / Math.log2(i + 2);
  const ideal = Object.values(qrels).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(ideal.length, k); i++)
    idcg += (Math.pow(2, ideal[i]) - 1) / Math.log2(i + 2);
  return idcg > 0 ? dcg / idcg : 0;
}

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
        const u1 = U[i*N+j], u2 = U[j*N+i];
        ad += w * ((u2 > 0 ? u2 : 0) * C[j] - (u1 > 0 ? u1 : 0) * C[i]);
      }
      const cn = Math.max(0, C[i] + dt * (df + ad));
      Cn[i] = cn; const d2 = Math.abs(cn - C[i]); if (d2 > mx) mx = d2;
    }
    [C, Cn] = [Cn, C]; if (mx < 1e-3) break;
  }
  return C;
}

function vtAlignedDist(qV, docCloud) {
  let totalDist = 0;
  for (let s = 0; s < NS; s++) {
    const off = s * SD;
    let minDist = Infinity;
    for (const sent of docCloud) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += qV[off+i]*sent[off+i]; na += qV[off+i]*qV[off+i]; nb += sent[off+i]*sent[off+i]; }
      const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
      if (d < minDist) minDist = d;
    }
    totalDist += minDist;
    let sumReverse = 0;
    for (const sent of docCloud) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += sent[off+i]*qV[off+i]; na += sent[off+i]*sent[off+i]; nb += qV[off+i]*qV[off+i]; }
      sumReverse += 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
    }
    totalDist += sumReverse / docCloud.length;
  }
  return totalDist / NS;
}

function vtAlignedChamfer(clA, clB) {
  let totalDist = 0;
  for (let s = 0; s < NS; s++) {
    const off = s * SD;
    let sAB = 0;
    for (const a of clA) {
      let mn = Infinity;
      for (const b of clB) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < SD; i++) { dot += a[off+i]*b[off+i]; na += a[off+i]*a[off+i]; nb += b[off+i]*b[off+i]; }
        const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        if (d < mn) mn = d;
      }
      sAB += mn;
    }
    let sBA = 0;
    for (const b of clB) {
      let mn = Infinity;
      for (const a of clA) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < SD; i++) { dot += a[off+i]*b[off+i]; na += a[off+i]*a[off+i]; nb += b[off+i]*b[off+i]; }
        const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        if (d < mn) mn = d;
      }
      sBA += mn;
    }
    totalDist += sAB / clA.length + sBA / clB.length;
  }
  return totalDist / NS;
}

// JS vtAligned Round0（与 Rust shapeCfdPipeline 同结构：cosine-centroid top-30，无预取）
function jsVtAlignedR0(qV, poolIds, dv, getSentVecs, knn = 3, uStr = 0.3) {
  const N = poolIds.length; if (N === 0) return [];
  const clouds = poolIds.map(id => getSentVecs(id));
  let qN = 0; for (let d2 = 0; d2 < DIM; d2++) qN += qV[d2] * qV[d2];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);

  // doc-doc vtAlignedChamfer 距离矩阵
  const dist = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i+1; j < N; j++) {
    const d2 = vtAlignedChamfer(clouds[i], clouds[j]); dist[i*N+j] = d2; dist[j*N+i] = d2;
  }

  // KNN 图
  const ek = Math.min(knn, N-1); const adj = Array.from({length: N}, () => []);
  for (let i = 0; i < N; i++) {
    const nb = []; for (let j = 0; j < N; j++) if (j !== i) nb.push({j, d: dist[i*N+j]});
    nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek; t++) adj[i].push({j: nb[t].j, w: Math.exp(-2*nb[t].d)});
  }
  for (let i = 0; i < N; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({j: i, w: e.w});

  // 对流系数（alpha=0.3，对齐 Rust）
  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (const e of adj[i]) {
    const j = e.j; if (U[i*N+j] || U[j*N+i]) continue;
    let en = 0, dvv = 0;
    for (let d2 = 0; d2 < DIM; d2++) {
      const df = dv[poolIds[j]][d2] - dv[poolIds[i]][d2]; en += df*df; dvv += df*qV[d2]*iqn;
    }
    const u0 = (dvv / (Math.sqrt(en)+1e-8)) * uStr; U[i*N+j] = u0; U[j*N+i] = -u0;
  }

  // 初始浓度
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
  const Cf = pde4(C0, adj, U, N);

  return poolIds.map((id, i) => ({did: id, s: Cf[i]})).sort((a, b) => b.s - a.s);
}

// JS vt_v7 = vtV7Combined（initPool=30, maxRounds=1, poolBudget=45, cosine 探针预取）
function vtV7Combined(qV, initCandidates, allDids, dv, getSentVecs, getCentroid) {
  const initPool = 30, maxRounds = 1, poolBudget = 45, knn = 3, probeCount = 8, uStr = 0.3;
  let qN = 0; for (let d2 = 0; d2 < DIM; d2++) qN += qV[d2] * qV[d2];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);
  const qHat = new Float32Array(DIM);
  for (let d2 = 0; d2 < DIM; d2++) qHat[d2] = qV[d2] * iqn;

  const poolSet = new Set(); const poolIds = [];
  const limit = Math.min(initPool, initCandidates.length);
  for (let i = 0; i < limit; i++) { poolIds.push(initCandidates[i].did); poolSet.add(initCandidates[i].did); }
  const maxN = poolBudget; const distCache = new Float64Array(maxN * maxN);
  const clouds = poolIds.map(id => getSentVecs(id)); let N = poolIds.length;
  for (let i = 0; i < N; i++) for (let j = i+1; j < N; j++) {
    const d2 = vtAlignedChamfer(clouds[i], clouds[j]); distCache[i*maxN+j] = d2; distCache[j*maxN+i] = d2;
  }
  function buildAdj(n, k) {
    const ek2 = Math.min(k, n-1); const adj2 = Array.from({length: n}, () => []);
    for (let i = 0; i < n; i++) {
      const nb = []; for (let j = 0; j < n; j++) if (j !== i) nb.push({j, d: distCache[i*maxN+j]});
      nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek2; t++) adj2[i].push({j: nb[t].j, w: Math.exp(-2*nb[t].d)});
    }
    for (let i = 0; i < n; i++) for (const e of adj2[i]) if (!adj2[e.j].some(x => x.j === i)) adj2[e.j].push({j: i, w: e.w});
    return adj2;
  }
  function buildU(adj2, n, centroids) {
    const U2 = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (const e of adj2[i]) {
      const j = e.j; if (U2[i*n+j] || U2[j*n+i]) continue;
      let en = 0, dvv = 0;
      for (let d2 = 0; d2 < DIM; d2++) { const df = centroids[j][d2]-centroids[i][d2]; en += df*df; dvv += df*qV[d2]*iqn; }
      const u0 = (dvv / (Math.sqrt(en)+1e-8)) * uStr; U2[i*n+j] = u0; U2[j*n+i] = -u0;
    }
    return U2;
  }
  let centroids = poolIds.map(id => getCentroid(id));
  let adj = buildAdj(N, knn); let U = buildU(adj, N, centroids);
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
  let C = pde4(C0, adj, U, N);
  for (let r = 1; r <= maxRounds; r++) {
    if (N >= poolBudget) break;
    const cVals = Array.from(C).sort((a, b) => a - b); const median = cVals[cVals.length >> 1];
    function outwardFlux(i) {
      let flux = 0;
      for (const e of adj[i]) { const u_ij = U[i*N+e.j]; if (u_ij > 0) flux += u_ij * e.w * C[e.j]; }
      return flux;
    }
    const fluxes = [];
    for (let i = 0; i < N; i++) {
      if (adj[i].length >= knn * 2 || C[i] <= median) continue;
      const f = C[i] * Math.max(0, outwardFlux(i));
      if (f > 0) fluxes.push({idx: i, flux: f});
    }
    if (fluxes.length === 0) break;
    fluxes.sort((a, b) => b.flux - a.flux);
    const m = Math.min(probeCount, fluxes.length);
    const probes = fluxes.slice(0, m);
    const budget = poolBudget - N;
    const newDids = [];
    const perProbe = Math.max(1, Math.ceil(budget / m));
    for (const p of probes) {
      if (newDids.length >= budget) break;
      const docCentroid = centroids[p.idx];
      const probe = new Float32Array(DIM);
      for (let d2 = 0; d2 < DIM; d2++) probe[d2] = 0.7 * qHat[d2] + 0.3 * docCentroid[d2];
      const hits = [];
      for (const did of allDids) {
        if (poolSet.has(did)) continue;
        hits.push({did, s: cosSim(probe, dv[did])});
      }
      hits.sort((a, b) => b.s - a.s);
      let taken = 0;
      for (const h of hits) {
        if (taken >= perProbe || newDids.length >= budget) break;
        newDids.push(h.did); poolSet.add(h.did); taken++;
      }
    }
    if (newDids.length === 0) break;
    const oldN = N;
    for (const did of newDids) { poolIds.push(did); clouds.push(getSentVecs(did)); }
    N = poolIds.length;
    for (let i = oldN; i < N; i++) for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const d2 = vtAlignedChamfer(clouds[i], clouds[j]); distCache[i*maxN+j] = d2; distCache[j*maxN+i] = d2;
    }
    centroids = poolIds.map(id => getCentroid(id));
    adj = buildAdj(N, knn); U = buildU(adj, N, centroids);
    const C0n = new Float64Array(N);
    for (let i = 0; i < oldN; i++) C0n[i] = C[i];
    for (let i = oldN; i < N; i++) C0n[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
    C = pde4(C0n, adj, U, N);
  }
  return poolIds.map((id, i) => ({did: id, s: C[i]})).sort((a, b) => b.s - a.s);
}

// ── 数据加载 ──────────────────────────────────────────────────────────────────
function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const a = [];
    const rl = readline.createInterface({input: fs.createReadStream(fp, {encoding: 'utf-8'}), crlfDelay: Infinity});
    rl.on('line', l => { if (l.trim()) try { a.push(JSON.parse(l)); } catch(e) {} });
    rl.on('close', () => resolve(a));
    rl.on('error', reject);
  });
}
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
  console.log('\n' + '═'.repeat(70));
  console.log('  Rust shapeCfdPipeline 精细对比验证 (4 方法)');
  console.log('  cosine_centroid | js_vt_aligned_r0 | rust_shape_cfd | js_vt_v7');
  console.log('═'.repeat(70) + '\n');

  // 加载 Rust addon
  let vexus = null;
  let reverseMap = {};
  try {
    const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
    vexus = new LawVexus('/tmp/cfd_verify2_' + process.pid);
    const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
    for (const [k, v] of Object.entries(idMap)) reverseMap[v] = k;
    const n = vexus.loadClouds(SQLITE_PATH);
    console.log(`  [Rust] loadClouds: ${n} 文档`);
  } catch(e) {
    console.error('  [Rust] 加载失败:', e.message);
    process.exit(1);
  }

  process.stdout.write('  [1/3] 加载 corpus_vectors.jsonl ...');
  const dv = {}, corpusSentsRaw = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    dv[o._id] = new Float32Array(o.vector);
    if (o.sentences && o.sentences.length > 1) corpusSentsRaw[o._id] = o.sentences;
  }
  console.log(' 完成');

  process.stdout.write('  [2/3] 加载 query_vectors.jsonl + qrels ...');
  const qV = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl'))) qV[o._id] = new Float32Array(o.vector);
  const qrels = loadQrels(path.join(DATA_DIR, 'qrels.tsv'));
  let qids = Object.keys(qrels).filter(q => qV[q]);
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);
  console.log(' 完成');

  process.stdout.write('  [3/3] 预计算句子向量和质心 ...');
  const sentVecs = {}, docCentroids = {};
  const allDids = Object.keys(dv);
  for (const did of allDids) {
    const raw = corpusSentsRaw[did];
    if (raw) {
      const vecs = raw.map(a => new Float32Array(a));
      sentVecs[did] = vecs;
      const c = new Float32Array(DIM);
      for (const v of vecs) for (let d2 = 0; d2 < DIM; d2++) c[d2] += v[d2];
      const inv = 1 / vecs.length; for (let d2 = 0; d2 < DIM; d2++) c[d2] *= inv;
      docCentroids[did] = c;
    } else {
      sentVecs[did] = [dv[did]]; docCentroids[did] = dv[did];
    }
  }
  console.log(' 完成\n');
  console.log(`  Q: ${qids.length}  C: ${allDids.length}  topN: 30  Mem: ${(process.memoryUsage().heapUsed/1024/1024)|0}MB\n`);

  function getSentVecs(did) { return sentVecs[did] || [dv[did]]; }
  function getCentroid(did) { return docCentroids[did]; }

  const t0 = Date.now();
  let acc = {cosine: 0, jsR0: 0, rustCfd: 0, jsVtV7: 0};
  const PRINT_EVERY = Math.max(1, Math.floor(qids.length / 10));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = qV[qid];
    const qr = qrels[qid];

    // cosine centroid baseline（与 JS vt_v7 的初始池筛选方式相同）
    const cs = allDids.map(did => ({did, s: cosSim(qVec, dv[did])}));
    cs.sort((a, b) => b.s - a.s);
    acc.cosine += ndcg(cs.slice(0, 10).map(d => d.did), qr);

    // JS vtAligned Round0（无预取，对齐 Rust 结构，使用 cosine-centroid top-30 初始池）
    const r0Ranked = jsVtAlignedR0(qVec, cs.slice(0, 30).map(x => x.did), dv, getSentVecs);
    acc.jsR0 += ndcg(r0Ranked.slice(0, 10).map(x => x.did), qr);

    // Rust shapeCfdPipeline（内部用 MaxSim cosine 粗筛 top-30）
    const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
    const rustHits = vexus.shapeCfdPipeline(qBuf, 10, 30);
    const rustRanked = rustHits.map(h => reverseMap[h.id]).filter(Boolean);
    acc.rustCfd += ndcg(rustRanked, qr);

    // JS vt_v7（vtAligned + v7 cosine 预取，当前 SOTA）
    const jsVtV7Ranked = vtV7Combined(qVec, cs.slice(0, 30), allDids, dv, getSentVecs, getCentroid);
    acc.jsVtV7 += ndcg(jsVtV7Ranked.slice(0, 10).map(x => x.did), qr);

    if ((qi + 1) % PRINT_EVERY === 0 || qi + 1 === qids.length) {
      const elapsed = (Date.now() - t0) / 1000;
      const n1 = (acc.cosine / (qi+1)).toFixed(4);
      const n2 = (acc.jsR0 / (qi+1)).toFixed(4);
      const n3 = (acc.rustCfd / (qi+1)).toFixed(4);
      const n4 = (acc.jsVtV7 / (qi+1)).toFixed(4);
      process.stdout.write(`\r  [${qi+1}/${qids.length}] ${elapsed.toFixed(0)}s | cos=${n1} jsR0=${n2} rust=${n3} vtv7=${n4}  `);
    }
  }
  process.stdout.write('\n\n');

  const nQ = qids.length;
  const totalSec = (Date.now() - t0) / 1000;
  const r1 = acc.cosine / nQ;
  const r2 = acc.jsR0 / nQ;
  const r3 = acc.rustCfd / nQ;
  const r4 = acc.jsVtV7 / nQ;

  function delta(v, base) {
    const d = (v - base) / base * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(2) + '%';
  }

  console.log(`  完成: ${totalSec.toFixed(1)}s\n`);
  console.log('  ┌──────────────────────────────┬──────────┬──────────────┬──────────────┬──────────────┐');
  console.log('  │ Method                       │ NDCG@10  │ vs cosine    │ vs js_r0     │ vs js_vt_v7  │');
  console.log('  ├──────────────────────────────┼──────────┼──────────────┼──────────────┼──────────────┤');
  console.log(`  │ cosine_centroid (baseline)   │ ${r1.toFixed(4).padStart(8)} │  (baseline)  │ ${delta(r1,r2).padStart(12)} │ ${delta(r1,r4).padStart(12)} │`);
  console.log(`  │ JS vtAligned R0 (no prefetch)│ ${r2.toFixed(4).padStart(8)} │ ${delta(r2,r1).padStart(12)} │  (baseline)  │ ${delta(r2,r4).padStart(12)} │`);
  console.log(`  │ Rust shapeCfdPipeline        │ ${r3.toFixed(4).padStart(8)} │ ${delta(r3,r1).padStart(12)} │ ${delta(r3,r2).padStart(12)} │ ${delta(r3,r4).padStart(12)} │`);
  console.log(`  │ JS vt_v7 (SOTA target)       │ ${r4.toFixed(4).padStart(8)} │ ${delta(r4,r1).padStart(12)} │ ${delta(r4,r2).padStart(12)} │  (baseline)  │`);
  console.log('  └──────────────────────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n');

  const gapRustVsJsR0 = (r3 - r2) / r2 * 100;
  const gapRustVsVtV7 = (r3 - r4) / r4 * 100;
  const rustJsAligned = Math.abs(gapRustVsJsR0) < 3.0;

  console.log('  分析:');
  console.log(`  • Rust vs JS-R0 (同结构): ${gapRustVsJsR0 >= 0 ? '+' : ''}${gapRustVsJsR0.toFixed(2)}% — ${rustJsAligned ? '对齐 (< 3%)' : '存在偏差 (>= 3%)'}`);
  console.log(`  • Rust vs SOTA vt_v7:     ${gapRustVsVtV7 >= 0 ? '+' : ''}${gapRustVsVtV7.toFixed(2)}%`);
  console.log(`  • Rust 超越 cosine:       ${delta(r3, r1)}`);

  if (!rustJsAligned) {
    console.log('\n  可能的偏差来源:');
    console.log('  1. Rust cosine_top_n 使用 MaxSim (min sentence cosine distance)');
    console.log('     JS cosine_centroid 使用文档主向量 cosine');
    console.log('  2. Rust 对流系数使用文档点云的第 0 句向量 vs JS 使用 dv[docId] 主向量');
    console.log('  这些不是 bug，而是 Rust 与 JS 的设计差异。');
  }

  // 保存结果
  const resultPath = path.join(DATA_DIR, 'rust_shape_cfd_verify2.json');
  fs.writeFileSync(resultPath, JSON.stringify({
    dataset: 'nfcorpus', nQueries: nQ, elapsedSec: +totalSec.toFixed(1),
    results: {
      cosine_centroid: +r1.toFixed(4),
      js_vt_aligned_r0: +r2.toFixed(4),
      rust_shape_cfd: +r3.toFixed(4),
      js_vt_v7: +r4.toFixed(4)
    },
    gaps: {
      rust_vs_js_r0_pct: +gapRustVsJsR0.toFixed(2),
      rust_vs_vt_v7_pct: +gapRustVsVtV7.toFixed(2),
      rust_vs_cosine_pct: +((r3-r1)/r1*100).toFixed(2),
    },
    rust_js_aligned_within_3pct: rustJsAligned
  }, null, 2));
  console.log(`\n  结果保存: ${resultPath}\n`);
}

main().catch(e => { console.error('错误:', e); process.exit(1); });
