#!/usr/bin/env node
'use strict';
/**
 * rust_shape_cfd_verify.js
 * 验证修复后 Rust shapeCfdPipeline 与 JS vt_v7 (vtV7Combined) 的 NDCG 对齐
 *
 * 烟雾模式: MAX_Q=10 node rust_shape_cfd_verify.js
 * 完整模式: node rust_shape_cfd_verify.js
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
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

const DIM = 4096, NS = 64, SD = 64;
function pqCosDist(a, b) {
  if (a.length !== NS * SD) return 1 - cosSim(a, b);
  let t = 0;
  for (let s = 0; s < NS; s++) {
    const o = s * SD; let d = 0, na = 0, nb = 0;
    for (let i = 0; i < SD; i++) { d += a[o+i]*b[o+i]; na += a[o+i]*a[o+i]; nb += b[o+i]*b[o+i]; }
    t += (1 - d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8));
  }
  return t / NS;
}

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

// VT-Aligned query-doc 距离（对齐 beir_rust_parallel.js 中 vtAlignedDist）
function vtAlignedDist(qV, docCloud) {
  let totalDist = 0;
  for (let s = 0; s < NS; s++) {
    const off = s * SD;
    // 正向：query 第 s 子空间在 docCloud 中找最近句子
    let minDist = Infinity;
    for (const sent of docCloud) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += qV[off+i]*sent[off+i]; na += qV[off+i]*qV[off+i]; nb += sent[off+i]*sent[off+i]; }
      const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
      if (d < minDist) minDist = d;
    }
    totalDist += minDist;
    // 反向：每句子第 s 子空间到 query，取平均
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

// VT-Aligned doc-doc Chamfer（对齐 beir_rust_parallel.js vtAlignedChamfer）
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

// JS vt_v7 = vtV7Combined（对齐 beir_rust_parallel.js）
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
    const ek = Math.min(k, n-1); const adj = Array.from({length: n}, () => []);
    for (let i = 0; i < n; i++) {
      const nb = []; for (let j = 0; j < n; j++) if (j !== i) nb.push({j, d: distCache[i*maxN+j]});
      nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek; t++) adj[i].push({j: nb[t].j, w: Math.exp(-2*nb[t].d)});
    }
    for (let i = 0; i < n; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({j: i, w: e.w});
    return adj;
  }
  function buildU(adj, n, centroids) {
    const U = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (const e of adj[i]) {
      const j = e.j; if (U[i*n+j] || U[j*n+i]) continue;
      let en = 0, dvv = 0;
      for (let d2 = 0; d2 < DIM; d2++) { const df = centroids[j][d2]-centroids[i][d2]; en += df*df; dvv += df*qV[d2]*iqn; }
      const u0 = (dvv / (Math.sqrt(en)+1e-8)) * uStr; U[i*n+j] = u0; U[j*n+i] = -u0;
    }
    return U;
  }

  let centroids = poolIds.map(id => getCentroid(id));
  let adj = buildAdj(N, knn); let U = buildU(adj, N, centroids);
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
  let C = pde4(C0, adj, U, N);

  // Round 1: v7 伴随通量预取（cosine 探针，不用 Rust）
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
      const lambda = 0.7;
      const docCentroid = centroids[p.idx];
      const probe = new Float32Array(DIM);
      for (let d2 = 0; d2 < DIM; d2++) probe[d2] = lambda * qHat[d2] + (1 - lambda) * docCentroid[d2];
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
  console.log('\n' + '═'.repeat(64));
  console.log('  Rust shapeCfdPipeline vs JS vt_v7 对齐验证');
  console.log('  数据集: NFCorpus | 修复后 liblaw_vexus.so');
  console.log('═'.repeat(64) + '\n');

  // 加载 Rust addon
  let vexus = null;
  let reverseMap = {};
  try {
    const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
    vexus = new LawVexus('/tmp/cfd_verify_' + process.pid);
    const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
    for (const [k, v] of Object.entries(idMap)) reverseMap[v] = k;
    const n = vexus.loadClouds(SQLITE_PATH);
    console.log(`  [Rust] loadClouds 成功: ${n} 文档`);
  } catch(e) {
    console.error('  [Rust] 加载失败:', e.message);
    process.exit(1);
  }

  process.stdout.write('  [1/4] 加载 corpus_vectors.jsonl ...');
  const dv = {}, corpusSentsRaw = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    dv[o._id] = new Float32Array(o.vector);
    if (o.sentences && o.sentences.length > 1) corpusSentsRaw[o._id] = o.sentences;
  }
  console.log(' 完成');

  process.stdout.write('  [2/4] 加载 query_vectors.jsonl ...');
  const qV = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl'))) qV[o._id] = new Float32Array(o.vector);
  console.log(' 完成');

  process.stdout.write('  [3/4] 加载 qrels.tsv ...');
  const qrels = loadQrels(path.join(DATA_DIR, 'qrels.tsv'));
  let qids = Object.keys(qrels).filter(q => qV[q]);
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);
  console.log(' 完成');

  process.stdout.write('  [4/4] 预计算句子向量和质心 ...');
  const sentVecs = {};
  const docCentroids = {};
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
      sentVecs[did] = [dv[did]];
      docCentroids[did] = dv[did];
    }
  }
  console.log(' 完成\n');
  console.log(`  Q: ${qids.length}  C: ${allDids.length}  Mem: ${(process.memoryUsage().heapUsed/1024/1024)|0}MB\n`);

  function getSentVecs(did) { return sentVecs[did] || [dv[did]]; }
  function getCentroid(did) { return docCentroids[did]; }

  const t0 = Date.now();
  let ndcgCosine = 0, ndcgJsVtV7 = 0, ndcgRustCfd = 0;
  const PRINT_EVERY = Math.max(1, Math.floor(qids.length / 10));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = qV[qid];
    const qr = qrels[qid];

    // cosine baseline
    const cs = allDids.map(did => ({did, s: cosSim(qVec, dv[did])}));
    cs.sort((a, b) => b.s - a.s);
    ndcgCosine += ndcg(cs.slice(0, 10).map(d => d.did), qr);

    // JS vt_v7
    const jsRanked = vtV7Combined(qVec, cs.slice(0, 30), allDids, dv, getSentVecs, getCentroid);
    ndcgJsVtV7 += ndcg(jsRanked.slice(0, 10).map(x => x.did), qr);

    // Rust shapeCfdPipeline
    const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
    const rustHits = vexus.shapeCfdPipeline(qBuf, 10, 30);
    const rustRanked = rustHits.map(h => reverseMap[h.id]).filter(Boolean);
    ndcgRustCfd += ndcg(rustRanked, qr);

    if ((qi + 1) % PRINT_EVERY === 0 || qi + 1 === qids.length) {
      const elapsed = (Date.now() - t0) / 1000;
      const n1 = (ndcgCosine / (qi+1)).toFixed(4);
      const n2 = (ndcgJsVtV7 / (qi+1)).toFixed(4);
      const n3 = (ndcgRustCfd / (qi+1)).toFixed(4);
      process.stdout.write(`\r  [${qi+1}/${qids.length}] ${elapsed.toFixed(0)}s | cosine=${n1} js_vt_v7=${n2} rust_cfd=${n3}  `);
    }
  }
  process.stdout.write('\n\n');

  const nQ = qids.length;
  const totalSec = (Date.now() - t0) / 1000;
  const r1 = ndcgCosine / nQ;
  const r2 = ndcgJsVtV7 / nQ;
  const r3 = ndcgRustCfd / nQ;

  function delta(v, base) {
    const d = (v - base) / base * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(2) + '%';
  }

  console.log(`  完成: ${totalSec.toFixed(1)}s\n`);
  console.log('  ┌──────────────────────────┬──────────┬─────────────────┬─────────────────┐');
  console.log('  │ Method                   │ NDCG@10  │ vs cosine       │ vs js_vt_v7     │');
  console.log('  ├──────────────────────────┼──────────┼─────────────────┼─────────────────┤');
  console.log(`  │ ${'cosine (baseline)'.padEnd(24)} │ ${r1.toFixed(4).padStart(8)} │    (baseline)   │ ${delta(r1,r2).padStart(15)} │`);
  console.log(`  │ ${'JS vt_v7'.padEnd(24)} │ ${r2.toFixed(4).padStart(8)} │ ${delta(r2,r1).padStart(15)} │    (baseline)   │`);
  console.log(`  │ ${'Rust shapeCfdPipeline'.padEnd(24)} │ ${r3.toFixed(4).padStart(8)} │ ${delta(r3,r1).padStart(15)} │ ${delta(r3,r2).padStart(15)} │`);
  console.log('  └──────────────────────────┴──────────┴─────────────────┴─────────────────┘\n');

  const gapAbs = Math.abs(r3 - r2);
  const gapPct = Math.abs((r3 - r2) / r2 * 100);
  const aligned = gapPct < 5.0;

  console.log(`  对齐判断: Rust vs JS 绝对差 = ${gapAbs.toFixed(4)}，相对差 = ${gapPct.toFixed(2)}%`);
  if (aligned) {
    console.log('  结论: Rust shapeCfdPipeline 与 JS vt_v7 对齐 (误差 < 5%)');
  } else {
    console.log('  结论: 存在对齐偏差 (>= 5%)，需进一步排查');
  }

  // 保存结果
  const resultPath = path.join(DATA_DIR, 'rust_shape_cfd_verify.json');
  fs.writeFileSync(resultPath, JSON.stringify({
    dataset: 'nfcorpus', nQueries: nQ, elapsedSec: +totalSec.toFixed(1),
    results: {cosine: +r1.toFixed(4), js_vt_v7: +r2.toFixed(4), rust_shape_cfd: +r3.toFixed(4)},
    gap_abs: +gapAbs.toFixed(4), gap_pct: +gapPct.toFixed(2), aligned
  }, null, 2));
  console.log(`\n  结果保存: ${resultPath}\n`);
}

main().catch(e => { console.error('错误:', e); process.exit(1); });
