#!/usr/bin/env node
'use strict';
/**
 * beir_dual_vt_bench.js — 双边PQ展开实验
 *
 * 实验目的：探索虚拟 token (VT) 子空间的不同聚合策略对 NDCG@10 的影响。
 * 基于 beir_rust_parallel.js 的 64-worker 并行框架，测试以下方法：
 *
 *   1. dual_vt_aligned     — 基线确认，与 vt_aligned 完全一致
 *   2. dual_vt_hierarchical — 分层 Chamfer：子空间内距离 + 子空间间二阶聚合
 *   3. dual_vt_weighted_T01 — 加权虚拟 token（temperature=0.1）
 *   4. dual_vt_weighted_T05 — 加权虚拟 token（temperature=0.5）
 *   5. dual_vt_weighted_T10 — 加权虚拟 token（temperature=1.0）
 *   6. dual_vt_chamfer_full — 全展开 Chamfer（跨子空间匹配，对照组）
 *
 * 数据：BEIR NFCorpus
 * 用法：node beir_dual_vt_bench.js [MAX_Q=0 表示全量]
 */
const {Worker, isMainThread, parentPort, workerData} = require('worker_threads');
const fs = require('fs'), path = require('path'), readline = require('readline');
const NW = 64; // worker 数量

// ════════════════════════════════════════════════════════════════
// WORKER
// ════════════════════════════════════════════════════════════════
if (!isMainThread) {
  const {queries, corpusVecs, corpusSents, allDids, topN} = workerData;
  const DIM = 4096, NS = 64, SD = 64;

  const dv = corpusVecs;

  // ── 基础工具函数 ──────────────────────────────────────────────

  /** 余弦相似度 */
  function cosSim(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
  }

  /** 子空间级 cosine 距离（在指定偏移 off 处取 SD 维） */
  function subCosDist(a, b, off) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < SD; i++) {
      dot += a[off + i] * b[off + i];
      na += a[off + i] * a[off + i];
      nb += b[off + i] * b[off + i];
    }
    return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
  }

  /** 子空间级 cosine 相似度（在指定偏移 off 处取 SD 维） */
  function subCosSim(a, b, off) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < SD; i++) {
      dot += a[off + i] * b[off + i];
      na += a[off + i] * a[off + i];
      nb += b[off + i] * b[off + i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
  }

  /** NDCG@k 计算 */
  function ndcg(r, q, k = 10) {
    let d = 0;
    for (let i = 0; i < Math.min(r.length, k); i++)
      d += (Math.pow(2, q[r[i]] || 0) - 1) / Math.log2(i + 2);
    const ir = Object.values(q).sort((a, b) => b - a);
    let id = 0;
    for (let i = 0; i < Math.min(ir.length, k); i++)
      id += (Math.pow(2, ir[i]) - 1) / Math.log2(i + 2);
    return id > 0 ? d / id : 0;
  }

  /** 获取文档点云（多句向量或单个文档向量） */
  function getCloud(did) {
    const s = corpusSents[did];
    return s ? s.map(a => new Float32Array(a)) : [dv[did]];
  }

  // ── PDE 扩散求解器（与原版一致） ─────────────────────────────

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

  // ── 方法 1: dual_vt_aligned（基线确认，与原版 vtAlignedDist 完全一致） ──

  function dualVtAlignedDist(qV, docCloud) {
    let totalDist = 0;
    for (let s = 0; s < NS; s++) {
      const off = s * SD;
      // 正向：query 的第 s 个子空间在 docCloud 中找最近句子
      let minDist = Infinity;
      for (const sent of docCloud) {
        const d = subCosDist(qV, sent, off);
        if (d < minDist) minDist = d;
      }
      totalDist += minDist;
      // 反向：每个句子的第 s 个子空间到 query（query 只有 1 点），取平均
      let sumReverse = 0;
      for (const sent of docCloud) {
        sumReverse += subCosDist(sent, qV, off);
      }
      totalDist += sumReverse / docCloud.length;
    }
    return totalDist / NS;
  }

  // doc-doc 对齐虚拟 token Chamfer（用于建 KNN 图）
  function vtAlignedChamfer(clA, clB) {
    let totalDist = 0;
    for (let s = 0; s < NS; s++) {
      const off = s * SD;
      // 正向
      let sAB = 0;
      for (const a of clA) {
        let mn = Infinity;
        for (const b of clB) {
          const d = subCosDist(a, b, off);
          if (d < mn) mn = d;
        }
        sAB += mn;
      }
      // 反向
      let sBA = 0;
      for (const b of clB) {
        let mn = Infinity;
        for (const a of clA) {
          const d = subCosDist(a, b, off);
          if (d < mn) mn = d;
        }
        sBA += mn;
      }
      totalDist += sAB / clA.length + sBA / clB.length;
    }
    return totalDist / NS;
  }

  // ── 方法 2: dual_vt_hierarchical（分层 Chamfer） ──
  //
  // 第一层（子空间内）：对每个子空间 s，计算 query_sub[s] 与每个 doc 句子
  //   的 sub[s] 之间的 cosine 距离 → 得到 64×N 的距离矩阵
  // 第二层（子空间间）：将 64 个子空间视为 64 个"超级点"，
  //   每个超级点的"坐标"是该子空间对 N 个句子的距离向量（N 维）。
  //   然后在这 64 个超级点之间做 top-k 最小值聚合。
  //
  // 核心思路：既保持子空间对齐，又引入子空间间的交互。

  function dualVtHierarchicalDist(qV, docCloud) {
    const N = docCloud.length;

    // 第一层：构建 64×N 距离矩阵
    // distMatrix[s][j] = cosine_dist(query_sub[s], sent_j_sub[s])
    const distMatrix = new Float64Array(NS * N);
    for (let s = 0; s < NS; s++) {
      const off = s * SD;
      for (let j = 0; j < N; j++) {
        distMatrix[s * N + j] = subCosDist(qV, docCloud[j], off);
      }
    }

    // 第二层：64 个超级点（子空间），每个超级点的坐标是 N 维距离向量
    // 对这些超级点做 Chamfer-like 聚合：
    //   - 对每个超级点 s，计算它与所有其他超级点 t 的 L2 距离（N 维空间中）
    //   - 找最近的 top-k 个邻居，取平均距离作为该超级点的"交互得分"
    // 最终得分 = 基础 aligned 得分 + 交互调制项

    // 先计算基础 aligned 得分（每个子空间的最小距离 + 反向平均）
    let baseDist = 0;
    const subMinDist = new Float64Array(NS); // 每个子空间的最小正向距离
    for (let s = 0; s < NS; s++) {
      let minD = Infinity;
      for (let j = 0; j < N; j++) {
        const d = distMatrix[s * N + j];
        if (d < minD) minD = d;
      }
      subMinDist[s] = minD;
      baseDist += minD;
      // 反向
      let sumR = 0;
      for (let j = 0; j < N; j++) sumR += distMatrix[s * N + j];
      baseDist += sumR / N;
    }
    baseDist /= NS;

    // 子空间间 L2 距离矩阵（64×64），在 N 维距离向量空间中
    // 选 top-K 最近邻（排除自身），用它们的"距离一致性"来调制
    const K_SUPER = 8; // 超级点 top-k 邻居数
    let interactionScore = 0;

    for (let s = 0; s < NS; s++) {
      // 计算子空间 s 与其他所有子空间的 L2 距离（在 N 维距离向量空间中）
      const neighbors = [];
      for (let t = 0; t < NS; t++) {
        if (t === s) continue;
        let l2sq = 0;
        for (let j = 0; j < N; j++) {
          const diff = distMatrix[s * N + j] - distMatrix[t * N + j];
          l2sq += diff * diff;
        }
        neighbors.push({t, l2: Math.sqrt(l2sq)});
      }
      neighbors.sort((a, b) => a.l2 - b.l2);

      // top-K 最近邻的子空间，取它们的 minDist 平均值
      // 直觉：如果子空间 s 的近邻们也有低距离 → 信号更可靠 → 贡献更大
      let neighborMinSum = 0;
      const kActual = Math.min(K_SUPER, neighbors.length);
      for (let k = 0; k < kActual; k++) {
        neighborMinSum += subMinDist[neighbors[k].t];
      }
      // 该子空间的调制后得分 = 自身最小距离 × (1 + 近邻一致性因子)
      // 近邻一致性：近邻平均距离 / 自身距离 → 接近1表示一致
      const neighborAvg = neighborMinSum / kActual;
      // 使用几何平均来融合自身和近邻信号
      interactionScore += Math.sqrt(subMinDist[s] * neighborAvg);
    }
    interactionScore /= NS;

    // 最终距离 = 0.7 * 基础aligned + 0.3 * 交互调制
    return 0.7 * baseDist + 0.3 * interactionScore;
  }

  // ── 方法 3: dual_vt_weighted（加权虚拟 token） ──
  //
  // 计算每个子空间 s 对 query 的"响应强度"：
  //   response_s = max over doc_sentences of cos_sim(query_sub[s], sent_sub[s])
  // 用 softmax(response / temperature) 作为权重，加权平均替代简单平均

  function dualVtWeightedDist(qV, docCloud, temperature) {
    // 计算每个子空间的响应强度和距离
    const responses = new Float64Array(NS);
    const subDists = new Float64Array(NS); // 正向最小距离
    const subReverse = new Float64Array(NS); // 反向平均距离

    for (let s = 0; s < NS; s++) {
      const off = s * SD;
      let maxSim = -Infinity;
      let minDist = Infinity;
      for (const sent of docCloud) {
        const sim = subCosSim(qV, sent, off);
        if (sim > maxSim) maxSim = sim;
        const d = 1 - sim;
        if (d < minDist) minDist = d;
      }
      responses[s] = maxSim;
      subDists[s] = minDist;

      // 反向距离
      let sumR = 0;
      for (const sent of docCloud) {
        sumR += subCosDist(sent, qV, off);
      }
      subReverse[s] = sumR / docCloud.length;
    }

    // softmax 加权
    // 先找最大值防溢出
    let maxR = -Infinity;
    for (let s = 0; s < NS; s++) if (responses[s] > maxR) maxR = responses[s];

    let sumExp = 0;
    const weights = new Float64Array(NS);
    for (let s = 0; s < NS; s++) {
      weights[s] = Math.exp((responses[s] - maxR) / temperature);
      sumExp += weights[s];
    }
    // 归一化
    for (let s = 0; s < NS; s++) weights[s] /= sumExp;

    // 加权聚合
    let totalDist = 0;
    for (let s = 0; s < NS; s++) {
      totalDist += weights[s] * (subDists[s] + subReverse[s]);
    }
    return totalDist;
  }

  // ── 方法 4: dual_vt_chamfer_full（全展开 Chamfer，跨子空间匹配） ──
  //
  // query 展开为 64 个 64d 点；doc 每个句子展开为 64 个 64d 点 → N×64 个点
  // 计算 64 vs N×64 的全 Chamfer（允许跨子空间匹配）
  // 预期比 aligned 差（vt_unaligned 的教训），用作对照组

  function dualVtChamferFullDist(qV, docCloud) {
    const maxSent = 5; // 限制句子数防止过慢
    const useSents = docCloud.length <= maxSent ? docCloud : docCloud.slice(0, maxSent);

    // query 展开为 64 个 64d token
    const qTokens = [];
    for (let s = 0; s < NS; s++) {
      const tok = new Float32Array(SD);
      for (let i = 0; i < SD; i++) tok[i] = qV[s * SD + i];
      qTokens.push(tok);
    }

    // doc 展开：每个句子 × 64 个子空间 → N*64 个 64d token
    const dTokens = [];
    for (const sent of useSents) {
      for (let s = 0; s < NS; s++) {
        const tok = new Float32Array(SD);
        for (let i = 0; i < SD; i++) tok[i] = sent[s * SD + i];
        dTokens.push(tok);
      }
    }

    // Chamfer: query→doc（64 个 query token 各找最近的 doc token）
    let sAB = 0;
    for (const qt of qTokens) {
      let mn = Infinity;
      for (const dt of dTokens) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < SD; i++) {
          dot += qt[i] * dt[i]; na += qt[i] * qt[i]; nb += dt[i] * dt[i];
        }
        const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        if (d < mn) mn = d;
      }
      sAB += mn;
    }

    // Chamfer: doc→query（N*64 个 doc token 各找最近的 query token）
    let sBA = 0;
    for (const dt of dTokens) {
      let mn = Infinity;
      for (const qt of qTokens) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < SD; i++) {
          dot += qt[i] * dt[i]; na += qt[i] * qt[i]; nb += dt[i] * dt[i];
        }
        const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        if (d < mn) mn = d;
      }
      sBA += mn;
    }

    return sAB / qTokens.length + sBA / dTokens.length;
  }

  // ── Shape 重排序框架（通用）──────────────────────────────────
  // 接受一个 query-doc 距离函数和一个 doc-doc 距离函数，
  // 构建 KNN 图 + PDE 扩散，输出重排序结果

  function shapeRerank(qV, poolIds, qdDistFn, ddDistFn) {
    const N = poolIds.length;
    if (N === 0) return [];
    const clouds = poolIds.map(id => getCloud(id));
    const knn = 3;

    // doc-doc 距离矩阵
    const dist = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const d = ddDistFn(clouds[i], clouds[j]);
        dist[i * N + j] = d;
        dist[j * N + i] = d;
      }
    }

    // KNN 图
    const ek = Math.min(knn, N - 1);
    const adj = Array.from({length: N}, () => []);
    for (let i = 0; i < N; i++) {
      const nb = [];
      for (let j = 0; j < N; j++) if (j !== i) nb.push({j, d: dist[i * N + j]});
      nb.sort((a, b) => a.d - b.d);
      for (let t = 0; t < ek; t++) adj[i].push({j: nb[t].j, w: Math.exp(-2 * nb[t].d)});
    }
    for (let i = 0; i < N; i++) {
      for (const e of adj[i]) {
        if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({j: i, w: e.w});
      }
    }

    // 对流系数
    let qN = 0;
    for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d];
    const iqn = 1 / (Math.sqrt(qN) + 1e-8);
    const U = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (const e of adj[i]) {
        const j = e.j;
        if (U[i * N + j] || U[j * N + i]) continue;
        let en = 0, dvv = 0;
        for (let d = 0; d < DIM; d++) {
          const df = dv[poolIds[j]][d] - dv[poolIds[i]][d];
          en += df * df;
          dvv += df * qV[d] * iqn;
        }
        const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * 0.3;
        U[i * N + j] = u0;
        U[j * N + i] = -u0;
      }
    }

    // 初始浓度
    const C0 = new Float64Array(N);
    for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * qdDistFn(qV, clouds[i]));
    const Cf = pde4(C0, adj, U, N);
    return poolIds.map((id, i) => ({id, s: Cf[i]})).sort((a, b) => b.s - a.s).map(x => x.id);
  }

  // ── Worker 主循环 ─────────────────────────────────────────────

  // 方法名列表
  const methods = [
    'cosine',
    'dual_vt_aligned',
    'dual_vt_hierarchical',
    'dual_vt_weighted_T01',
    'dual_vt_weighted_T05',
    'dual_vt_weighted_T10',
    'dual_vt_chamfer_full',
  ];
  const R = {};
  for (const m of methods) R[m] = 0;

  for (const q of queries) {
    const qV = new Float32Array(q.vec), qr = q.qrels;

    // cosine 初始排序
    const cs = allDids.map(did => ({did, s: cosSim(qV, dv[did])}));
    cs.sort((a, b) => b.s - a.s);
    R.cosine += ndcg(cs.slice(0, 10).map(d => d.did), qr);

    // top-N 候选池
    const pool = cs.slice(0, topN).map(x => x.did);

    // 方法1: dual_vt_aligned（基线）— 使用 vtAlignedChamfer 做 doc-doc
    R.dual_vt_aligned += ndcg(
      shapeRerank(qV, pool, dualVtAlignedDist, vtAlignedChamfer), qr
    );

    // 方法2: dual_vt_hierarchical — 用分层距离做 query-doc，doc-doc 仍用 aligned
    R.dual_vt_hierarchical += ndcg(
      shapeRerank(qV, pool, dualVtHierarchicalDist, vtAlignedChamfer), qr
    );

    // 方法3: dual_vt_weighted（三个温度）
    R.dual_vt_weighted_T01 += ndcg(
      shapeRerank(qV, pool,
        (q, cl) => dualVtWeightedDist(q, cl, 0.1),
        vtAlignedChamfer), qr
    );
    R.dual_vt_weighted_T05 += ndcg(
      shapeRerank(qV, pool,
        (q, cl) => dualVtWeightedDist(q, cl, 0.5),
        vtAlignedChamfer), qr
    );
    R.dual_vt_weighted_T10 += ndcg(
      shapeRerank(qV, pool,
        (q, cl) => dualVtWeightedDist(q, cl, 1.0),
        vtAlignedChamfer), qr
    );

    // 方法4: dual_vt_chamfer_full（全展开 Chamfer，对照组）
    R.dual_vt_chamfer_full += ndcg(
      shapeRerank(qV, pool, dualVtChamferFullDist, vtAlignedChamfer), qr
    );

    parentPort.postMessage({type: 'progress'});
  }

  parentPort.postMessage({type: 'result', data: R});
  process.exit(0);
}

// ════════════════════════════════════════════════════════════════
// MAIN THREAD
// ════════════════════════════════════════════════════════════════
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');

/** 逐行加载 JSONL 文件 */
function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const arr = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(fp, {encoding: 'utf-8'}),
      crlfDelay: Infinity,
    });
    rl.on('line', l => { if (l.trim()) try { arr.push(JSON.parse(l)); } catch(e) {} });
    rl.on('close', () => resolve(arr));
    rl.on('error', reject);
  });
}

/** 加载 qrels（TSV 格式） */
function loadQrels(fp) {
  const q = {};
  const ls = fs.readFileSync(fp, 'utf-8').trim().split('\n');
  for (let i = 1; i < ls.length; i++) {
    const [qi, di, s] = ls[i].split('\t');
    if (!q[qi]) q[qi] = {};
    q[qi][di] = parseInt(s);
  }
  return q;
}

(async () => {
  const methods = [
    'cosine',
    'dual_vt_aligned',
    'dual_vt_hierarchical',
    'dual_vt_weighted_T01',
    'dual_vt_weighted_T05',
    'dual_vt_weighted_T10',
    'dual_vt_chamfer_full',
  ];

  console.log(`\n${'='.repeat(70)}`);
  console.log(`  BEIR NFCorpus -- 双边PQ展开实验 (${NW} workers)`);
  console.log(`  ${methods.join(' | ')}`);
  console.log(`${'='.repeat(70)}\n`);

  const MAX_Q = parseInt(process.env.MAX_Q || '0');

  // 加载数据
  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));
  const reverseMap = {};
  for (const [k, v] of Object.entries(idMap)) reverseMap[v] = k;

  const cV = {}, cS = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl'))) {
    cV[o._id] = Array.from(new Float32Array(o.vector));
    if (o.sentences?.length > 1) cS[o._id] = o.sentences;
  }

  const qV = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl'))) {
    qV[o._id] = Array.from(new Float32Array(o.vector));
  }

  const qrels = loadQrels(path.join(DATA_DIR, 'qrels.tsv'));
  let qids = Object.keys(qrels).filter(q => qV[q]);
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  const allDids = Object.keys(cV);
  const topN = 30;

  console.log(`  Q:${qids.length} C:${allDids.length} W:${NW}\n`);

  // 分配 query 到 worker
  const batches = Array.from({length: NW}, () => []);
  for (let i = 0; i < qids.length; i++) {
    const qi = qids[i];
    batches[i % NW].push({vec: qV[qi], qrels: qrels[qi]});
  }

  let completed = 0;
  const t0 = performance.now();
  const progressTimer = setInterval(() => {
    const sec = (performance.now() - t0) / 1000;
    const qps = completed / sec || 0;
    const eta = qps > 0 ? ((qids.length - completed) / qps) : 0;
    process.stdout.write(`\r  ${completed}/${qids.length} | ${qps.toFixed(1)} q/s | ${sec.toFixed(0)}s | ETA ${eta.toFixed(0)}s  `);
  }, 3000);

  // 启动 worker
  const emptyResult = {};
  for (const m of methods) emptyResult[m] = 0;

  const wr = await Promise.all(batches.map((b, i) => {
    if (!b.length) return Promise.resolve({...emptyResult});
    return new Promise((resolve, reject) => {
      const w = new Worker(__filename, {
        workerData: {
          queries: b,
          corpusVecs: cV,
          corpusSents: cS,
          allDids,
          topN,
          idMap,
          reverseMap,
        },
      });
      w.on('message', msg => {
        if (msg.type === 'progress') completed++;
        else if (msg.type === 'result') resolve(msg.data);
      });
      w.on('error', reject);
      w.on('exit', c => { if (c !== 0) reject(new Error(`Worker ${i} exit: ${c}`)); });
    });
  }));

  clearInterval(progressTimer);
  const sec = (performance.now() - t0) / 1000;
  process.stdout.write('\r' + ' '.repeat(80) + '\r');

  // 汇总结果
  const nd = {};
  for (const m of methods) nd[m] = 0;
  for (const w of wr) for (const m of methods) nd[m] += (w[m] || 0);
  const nQ = qids.length;

  // 计算各方法 NDCG@10
  const scores = {};
  for (const m of methods) scores[m] = nd[m] / nQ;

  const cosVal = scores.cosine;
  const baseVal = scores.dual_vt_aligned;
  const fmt = (v, b) => {
    const p = ((v - b) / b * 100);
    return `${p >= 0 ? '+' : ''}${p.toFixed(1)}%`;
  };

  console.log(`\n  Done in ${sec.toFixed(1)}s (${(nQ / sec).toFixed(1)} q/s)\n`);
  console.log('  +----------------------+----------+-------------+------------------+');
  console.log('  | Method               | NDCG@10  | vs cosine   | vs vt_aligned    |');
  console.log('  +----------------------+----------+-------------+------------------+');
  for (const m of methods) {
    const v = scores[m];
    const label = m.padEnd(20);
    const ndcgStr = v.toFixed(4).padStart(8);
    const vsCos = m === 'cosine' ? '   baseline ' : fmt(v, cosVal).padStart(11) + ' ';
    const vsBase = m === 'dual_vt_aligned' ? '     baseline    ' : fmt(v, baseVal).padStart(16) + ' ';
    console.log(`  | ${label} | ${ndcgStr} | ${vsCos}| ${vsBase}|`);
  }
  console.log('  +----------------------+----------+-------------+------------------+');

  // 保存结果到 JSON
  const resultPath = path.join(DATA_DIR, 'dual_vt_bench_results.json');
  fs.writeFileSync(resultPath, JSON.stringify({
    ...scores,
    queries: nQ,
    workers: NW,
    seconds: +sec.toFixed(1),
  }, null, 2));
  console.log(`\n  Results saved to ${resultPath}`);
})().catch(e => { console.error('ERROR:', e); process.exit(1); });
