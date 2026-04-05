'use strict';

/**
 * AD-Rank V3 — 多方向对流 (MDA) + 分块对流 (BAA)
 *
 * 解决 v2 对流信号弱的问题（Pe=0.065, σ=1/√4096≈0.0156）。
 * 通过降维/分块，在低维子空间中放大对流信号：
 *   - MDA: 随机投影到 M 个 k 维子空间, σ=1/√k (k=32 → 11x 放大)
 *   - BAA: 按连续块分割维度, 每块独立对流
 *
 * 约束：
 *   - 不修改 ad_rank.js (v2 保留为基线)
 *   - 随机投影用固定种子 seed=42
 *   - 投影矩阵在模块加载时一次性生成
 */

// ══════════════════════════════════════════════════
// 1. adRankCore — 纯函数核心求解器
// ══════════════════════════════════════════════════

/**
 * 核心求解器（不含排序后处理，接受任意维度向量）
 *
 * @param {Float32Array} qVec - query 向量 (任意维度)
 * @param {Float32Array[]} docVecs - 文档向量
 * @param {number} N - 文档数
 * @param {object} opts - 超参数
 * @returns {{ C: Float64Array, reynolds: number, peclet: number, iterations: number, convergence: boolean }}
 */
function adRankCore(qVec, docVecs, N, opts = {}) {
  const {
    D = 0.15,
    uStrength = 0.3,
    knn = 3,
    maxIter = 50,
    epsilon = 1e-3,
    dt = 0.1,
  } = opts;

  if (N === 0) {
    return { C: new Float64Array(0), reynolds: 0, peclet: 0, iterations: 0, convergence: true };
  }

  const dim = qVec.length;

  // ── Step 1: 建 KNN 稀疏图 ──
  const beta = 2.0;

  // 预计算 L2 范数
  const norms = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    const v = docVecs[i];
    for (let d = 0; d < dim; d++) s += v[d] * v[d];
    norms[i] = Math.sqrt(s) + 1e-8;
  }

  // 相似度矩阵 (上三角 → 对称)
  const simMatrix = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    const vi = docVecs[i];
    const invNi = 1.0 / norms[i];
    for (let j = i + 1; j < N; j++) {
      const vj = docVecs[j];
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += vi[d] * vj[d];
      const s = dot * invNi / norms[j];
      simMatrix[i * N + j] = s;
      simMatrix[j * N + i] = s;
    }
  }

  // KNN 选择
  const adjacency = Array.from({ length: N }, () => []);
  const effectiveK = Math.min(knn, N - 1);
  const sortIdx = new Int32Array(N);
  const sortVal = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    let count = 0;
    for (let j = 0; j < N; j++) {
      if (j !== i) {
        sortIdx[count] = j;
        sortVal[count] = simMatrix[i * N + j];
        count++;
      }
    }
    for (let t = 0; t < effectiveK; t++) {
      let maxPos = t, maxVal = sortVal[t];
      for (let p = t + 1; p < count; p++) {
        if (sortVal[p] > maxVal) { maxVal = sortVal[p]; maxPos = p; }
      }
      if (maxPos !== t) {
        sortVal[maxPos] = sortVal[t]; sortVal[t] = maxVal;
        const tmpIdx = sortIdx[maxPos]; sortIdx[maxPos] = sortIdx[t]; sortIdx[t] = tmpIdx;
      }
      adjacency[i].push({ j: sortIdx[t], w: Math.exp(-beta * (1 - maxVal)) });
    }
  }

  // 对称化
  for (let i = 0; i < N; i++) {
    for (const edge of adjacency[i]) {
      if (!adjacency[edge.j].some(e => e.j === i)) {
        adjacency[edge.j].push({ j: i, w: edge.w });
      }
    }
  }

  // ── Step 2: 初始化浓度场 C[i] = cosineSim(query, doc[i]) ──
  let C = new Float64Array(N);
  let qNorm = 0;
  for (let d = 0; d < dim; d++) qNorm += qVec[d] * qVec[d];
  const invQNorm = 1.0 / (Math.sqrt(qNorm) + 1e-8);

  for (let i = 0; i < N; i++) {
    let dot = 0;
    const v = docVecs[i];
    for (let d = 0; d < dim; d++) dot += qVec[d] * v[d];
    C[i] = dot * invQNorm / norms[i];
  }

  // ── Step 3: 预计算对流方向 ──
  const U = new Float64Array(N * N);
  const queryDir = new Float32Array(dim);
  for (let d = 0; d < dim; d++) queryDir[d] = qVec[d] * invQNorm;

  for (let i = 0; i < N; i++) {
    const vi = docVecs[i];
    for (const edge of adjacency[i]) {
      const j = edge.j;
      if (U[i * N + j] !== 0 || U[j * N + i] !== 0) continue;

      const vj = docVecs[j];
      let edNorm = 0, dotVal = 0;
      for (let d = 0; d < dim; d++) {
        const diff = vj[d] - vi[d];
        edNorm += diff * diff;
        dotVal += diff * queryDir[d];
      }
      const u_ij = (dotVal / (Math.sqrt(edNorm) + 1e-8)) * uStrength;
      U[i * N + j] = u_ij;
      U[j * N + i] = -u_ij;
    }
  }

  // ── Step 4: 迭代求解 ──
  let maxDeg = 0;
  for (let i = 0; i < N; i++) {
    if (adjacency[i].length > maxDeg) maxDeg = adjacency[i].length;
  }
  const cfl_dt = maxDeg > 0 ? 0.8 / maxDeg : dt;
  const effective_dt = Math.min(dt, cfl_dt);

  let C_new = new Float64Array(N);
  let iterations = 0;
  let converged = false;
  let totalAdvectionSum = 0;
  let totalDiffusionSum = 0;

  for (let t = 0; t < maxIter; t++) {
    let maxDelta = 0;
    let iterAdvection = 0, iterDiffusion = 0;

    for (let i = 0; i < N; i++) {
      let diffusion = 0, advection = 0;
      const adj_i = adjacency[i];
      const C_i = C[i];
      const offset = i * N;

      for (let e = 0; e < adj_i.length; e++) {
        const edge = adj_i[e];
        const j = edge.j;
        const w = edge.w;
        const dC = C[j] - C_i;

        diffusion += D * w * dC;

        const u_ij = U[offset + j];
        const u_ji = U[j * N + i];
        advection += w * ((u_ji > 0 ? u_ji : 0) * C[j] - (u_ij > 0 ? u_ij : 0) * C_i);
      }

      iterDiffusion += (diffusion > 0 ? diffusion : -diffusion);
      iterAdvection += (advection > 0 ? advection : -advection);

      let c_new = C_i + effective_dt * (diffusion + advection);
      if (c_new < 0) c_new = 0;
      if (c_new > 1) c_new = 1;
      C_new[i] = c_new;

      const delta = c_new > C_i ? c_new - C_i : C_i - c_new;
      if (delta > maxDelta) maxDelta = delta;
    }

    totalAdvectionSum = iterAdvection;
    totalDiffusionSum = iterDiffusion;
    iterations = t + 1;

    // 交替缓冲区
    const tmp = C; C = C_new; C_new = tmp;

    if (maxDelta < epsilon) { converged = true; break; }
  }

  // Reynolds & Péclet
  const reynolds = totalDiffusionSum > 1e-12 ? totalAdvectionSum / totalDiffusionSum : 0;
  let uAbsSum = 0, edgeCount = 0;
  for (let i = 0; i < N; i++) {
    const offset = i * N;
    for (const edge of adjacency[i]) {
      if (edge.j > i) {
        const u = U[offset + edge.j];
        uAbsSum += (u > 0 ? u : -u);
        edgeCount++;
      }
    }
  }
  const peclet = edgeCount > 0 && D > 1e-12 ? (uAbsSum / edgeCount) / D : 0;

  return { C, reynolds, peclet, iterations, convergence: converged };
}

// ══════════════════════════════════════════════════
// 2. 随机投影工具
// ══════════════════════════════════════════════════

/**
 * 生成 M 个随机投影矩阵，每个 k × dim，行为归一化随机向量
 * 固定种子，确保可复现
 */
function generateRandomProjections(M, k, dim, seed) {
  let rng = seed;
  const nextRand = () => {
    rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (rng >>> 0) / 0xFFFFFFFF - 0.5;
  };

  const projections = [];
  for (let m = 0; m < M; m++) {
    const P = [];
    for (let r = 0; r < k; r++) {
      const row = new Float32Array(dim);
      let norm = 0;
      for (let d = 0; d < dim; d++) {
        row[d] = nextRand();
        norm += row[d] * row[d];
      }
      norm = Math.sqrt(norm) + 1e-8;
      for (let d = 0; d < dim; d++) row[d] /= norm;
      P.push(row);
    }
    projections.push(P);
  }
  return projections;
}

/**
 * 将向量投影到子空间
 * @param {Float32Array} vec - 原始向量
 * @param {Float32Array[]} P - 投影矩阵 (k 行, 每行 dim 维)
 * @returns {Float32Array} k 维投影向量
 */
function projectVector(vec, P) {
  const k = P.length;
  const result = new Float32Array(k);
  for (let r = 0; r < k; r++) {
    let dot = 0;
    const row = P[r];
    for (let d = 0; d < row.length; d++) dot += row[d] * vec[d];
    result[r] = dot;
  }
  return result;
}

// ══════════════════════════════════════════════════
// 3. MDA — 多方向对流
// ══════════════════════════════════════════════════

// 预生成投影矩阵缓存（模块加载时按需生成）
const _projectionCache = new Map();

function _getProjections(M, k, dim, seed) {
  const key = `${M}_${k}_${dim}_${seed}`;
  if (!_projectionCache.has(key)) {
    _projectionCache.set(key, generateRandomProjections(M, k, dim, seed));
  }
  return _projectionCache.get(key);
}

/**
 * 多方向对流 (MDA, Multi-Directional Advection)
 *
 * 生成 M 个随机投影矩阵，将高维投影到 k 维子空间。
 * 每个子空间独立运行 adRankCore，按 query 在该方向的能量加权融合。
 *
 * @param {Float32Array} queryVector - query 向量
 * @param {Float32Array[]} docVectors - 文档向量
 * @param {number} topK - 返回 top-k
 * @param {object} [options]
 * @returns {ADRankResult}
 */
function adRankMDA(queryVector, docVectors, topK, options = {}) {
  const {
    M = 8,
    k = 32,
    D = 0.15,
    uStrength = 0.3,
    knn = 3,
    maxIter = 50,
    epsilon = 1e-3,
    dt = 0.1,
    seed = 42,
  } = options;

  const N = docVectors.length;
  if (N === 0) {
    return {
      rankings: [], reynolds: 0, peclet: 0, iterations: 0,
      convergence: true, advectionContribution: 0, method: 'MDA',
    };
  }

  const dim = docVectors[0].length;
  const projections = _getProjections(M, k, dim, seed);

  const concentrations = [];
  const weights = [];
  let totalReynolds = 0, totalPeclet = 0, totalIter = 0;
  let allConverged = true;

  for (let m = 0; m < M; m++) {
    const P = projections[m];

    // 投影 query 和所有 doc
    const qProj = projectVector(queryVector, P);
    const docsProj = docVectors.map(v => projectVector(v, P));

    // 在投影空间中跑 adRankCore
    const result = adRankCore(qProj, docsProj, N, { D, uStrength, knn, maxIter, epsilon, dt });
    concentrations.push(result.C);
    totalReynolds += result.reynolds;
    totalPeclet += result.peclet;
    totalIter += result.iterations;
    if (!result.convergence) allConverged = false;

    // 权重 = query 投影后的 L2 能量
    let qEnergy = 0;
    for (let d = 0; d < k; d++) qEnergy += qProj[d] * qProj[d];
    weights.push(Math.sqrt(qEnergy));
  }

  // 归一化权重
  const wSum = weights.reduce((a, b) => a + b, 0) || 1;
  for (let m = 0; m < M; m++) weights[m] /= wSum;

  // 加权融合浓度场
  const C_final = new Float64Array(N);
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < N; i++) {
      C_final[i] += weights[m] * concentrations[m][i];
    }
  }

  // 计算对流贡献率: 对比纯扩散（u=0）的结果
  // 纯扩散: 每个方向 u=0 跑一遍, 加权融合
  const C_diffOnly = new Float64Array(N);
  for (let m = 0; m < M; m++) {
    const P = projections[m];
    const qProj = projectVector(queryVector, P);
    const docsProj = docVectors.map(v => projectVector(v, P));
    const result = adRankCore(qProj, docsProj, N, { D, uStrength: 0, knn, maxIter, epsilon, dt });
    for (let i = 0; i < N; i++) {
      C_diffOnly[i] += weights[m] * result.C[i];
    }
  }

  // 对流贡献率 = 1 - correlation(C_final, C_diffOnly)
  // 用排序差异更合理: 比较排序顺序的差异比例
  const rankFull = argsort(C_final);
  const rankDiff = argsort(C_diffOnly);
  let orderDiffs = 0;
  for (let i = 0; i < N; i++) {
    if (rankFull[i] !== rankDiff[i]) orderDiffs++;
  }
  const advectionContribution = orderDiffs / N;

  // 排序返回 top-k
  const ranked = new Array(N);
  for (let i = 0; i < N; i++) {
    ranked[i] = { index: i, score: C_final[i], topology: 'flow' };
  }
  ranked.sort((a, b) => b.score - a.score);
  const topKResults = ranked.slice(0, Math.min(topK, N));

  return {
    rankings: topKResults,
    reynolds: totalReynolds / M,
    peclet: totalPeclet / M,
    iterations: Math.round(totalIter / M),
    convergence: allConverged,
    advectionContribution,
    method: 'MDA',
    weights: weights.map(w => +w.toFixed(4)),
  };
}

// ══════════════════════════════════════════════════
// 4. BAA — 分块对流
// ══════════════════════════════════════════════════

/**
 * 分块对流 (BAA, Block-Adaptive Advection)
 *
 * 将高维向量按连续块分割（如 8×512），每块独立建 KNN、算对流。
 * 按 query 在每块的 L2 能量加权融合。
 *
 * @param {Float32Array} queryVector - query 向量
 * @param {Float32Array[]} docVectors - 文档向量
 * @param {number} topK - 返回 top-k
 * @param {object} [options]
 * @returns {ADRankResult}
 */
function adRankBAA(queryVector, docVectors, topK, options = {}) {
  const {
    B = 8,
    D = 0.15,
    uStrength = 0.3,
    knn = 3,
    maxIter = 50,
    epsilon = 1e-3,
    dt = 0.1,
  } = options;

  const N = docVectors.length;
  if (N === 0) {
    return {
      rankings: [], reynolds: 0, peclet: 0, iterations: 0,
      convergence: true, advectionContribution: 0, method: 'BAA',
    };
  }

  const dim = docVectors[0].length;
  const blockSize = Math.floor(dim / B);

  const concentrations = [];
  const weights = [];
  let totalReynolds = 0, totalPeclet = 0, totalIter = 0;
  let allConverged = true;

  for (let b = 0; b < B; b++) {
    const start = b * blockSize;
    const end = (b === B - 1) ? dim : start + blockSize;

    // 提取块向量
    const qBlock = queryVector.slice(start, end);
    const docsBlock = docVectors.map(v => v.slice(start, end));

    // 在块空间中跑 adRankCore
    const result = adRankCore(qBlock, docsBlock, N, { D, uStrength, knn, maxIter, epsilon, dt });
    concentrations.push(result.C);
    totalReynolds += result.reynolds;
    totalPeclet += result.peclet;
    totalIter += result.iterations;
    if (!result.convergence) allConverged = false;

    // 权重 = query 在此块的 L2 能量
    let qEnergy = 0;
    const bDim = end - start;
    for (let d = 0; d < bDim; d++) qEnergy += qBlock[d] * qBlock[d];
    weights.push(Math.sqrt(qEnergy));
  }

  // 归一化权重
  const wSum = weights.reduce((a, b) => a + b, 0) || 1;
  for (let b = 0; b < B; b++) weights[b] /= wSum;

  // 加权融合浓度场
  const C_final = new Float64Array(N);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < N; i++) {
      C_final[i] += weights[b] * concentrations[b][i];
    }
  }

  // 对流贡献率
  const C_diffOnly = new Float64Array(N);
  for (let b = 0; b < B; b++) {
    const start = b * blockSize;
    const end = (b === B - 1) ? dim : start + blockSize;
    const qBlock = queryVector.slice(start, end);
    const docsBlock = docVectors.map(v => v.slice(start, end));
    const result = adRankCore(qBlock, docsBlock, N, { D, uStrength: 0, knn, maxIter, epsilon, dt });
    for (let i = 0; i < N; i++) {
      C_diffOnly[i] += weights[b] * result.C[i];
    }
  }

  const rankFull = argsort(C_final);
  const rankDiff = argsort(C_diffOnly);
  let orderDiffs = 0;
  for (let i = 0; i < N; i++) {
    if (rankFull[i] !== rankDiff[i]) orderDiffs++;
  }
  const advectionContribution = orderDiffs / N;

  // 排序
  const ranked = new Array(N);
  for (let i = 0; i < N; i++) {
    ranked[i] = { index: i, score: C_final[i], topology: 'flow' };
  }
  ranked.sort((a, b) => b.score - a.score);
  const topKResults = ranked.slice(0, Math.min(topK, N));

  return {
    rankings: topKResults,
    reynolds: totalReynolds / B,
    peclet: totalPeclet / B,
    iterations: Math.round(totalIter / B),
    convergence: allConverged,
    advectionContribution,
    method: 'BAA',
    weights: weights.map(w => +w.toFixed(4)),
  };
}

// ══════════════════════════════════════════════════
// 工具函数
// ══════════════════════════════════════════════════

/** 返回排序后的索引数组 (降序) */
function argsort(arr) {
  const indices = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) indices[i] = i;
  indices.sort((a, b) => arr[b] - arr[a]);
  // 转为 rank 数组
  const rank = new Array(arr.length);
  for (let i = 0; i < indices.length; i++) rank[indices[i]] = i;
  return rank;
}

// ══════════════════════════════════════════════════
// 导出
// ══════════════════════════════════════════════════

module.exports = { adRankMDA, adRankBAA, adRankCore };

// ══════════════════════════════════════════════════
// 自测
// ══════════════════════════════════════════════════

if (require.main === module) {
  const N = 30, D = 128;
  const seed = 42;

  // 固定种子随机数
  let rng = seed;
  const rand = () => { rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF; return (rng >>> 0) / 0xFFFFFFFF - 0.5; };

  const query = new Float32Array(D).map(() => rand());
  const docs = Array.from({ length: N }, () => new Float32Array(D).map(() => rand()));

  console.log('═'.repeat(60));
  console.log('  AD-Rank V3 自测 — adRankCore + MDA + BAA');
  console.log('═'.repeat(60));

  // 1. adRankCore 测试
  console.log('\n── adRankCore ──');
  const t0 = performance.now();
  const coreResult = adRankCore(query, docs, N, { D: 0.15, uStrength: 0.3, knn: 3 });
  const t0e = performance.now() - t0;
  console.log(`  收敛: ${coreResult.convergence}, 迭代: ${coreResult.iterations}`);
  console.log(`  Re: ${coreResult.reynolds.toFixed(3)}, Pe: ${coreResult.peclet.toFixed(3)}`);
  console.log(`  耗时: ${t0e.toFixed(2)}ms`);
  console.log(`  Top-5 scores: [${Array.from(coreResult.C).sort((a,b) => b-a).slice(0,5).map(v => v.toFixed(4)).join(', ')}]`);

  // 2. MDA 测试
  console.log('\n── MDA (M=8, k=32) ──');
  const t1 = performance.now();
  const mdaResult = adRankMDA(query, docs, 5, { M: 8, k: 32, uStrength: 0.3 });
  const t1e = performance.now() - t1;
  console.log(`  收敛: ${mdaResult.convergence}, 迭代: ${mdaResult.iterations}`);
  console.log(`  Re: ${mdaResult.reynolds.toFixed(3)}, Pe: ${mdaResult.peclet.toFixed(3)}`);
  console.log(`  对流贡献率: ${(mdaResult.advectionContribution * 100).toFixed(1)}%`);
  console.log(`  权重分布: [${mdaResult.weights.join(', ')}]`);
  console.log(`  耗时: ${t1e.toFixed(2)}ms`);
  console.log(`  Top-5:`);
  mdaResult.rankings.forEach((r, i) =>
    console.log(`    #${i + 1} doc[${r.index}] score=${r.score.toFixed(4)}`)
  );

  // 3. BAA 测试
  console.log('\n── BAA (B=8) ──');
  const t2 = performance.now();
  const baaResult = adRankBAA(query, docs, 5, { B: 8, uStrength: 0.3 });
  const t2e = performance.now() - t2;
  console.log(`  收敛: ${baaResult.convergence}, 迭代: ${baaResult.iterations}`);
  console.log(`  Re: ${baaResult.reynolds.toFixed(3)}, Pe: ${baaResult.peclet.toFixed(3)}`);
  console.log(`  对流贡献率: ${(baaResult.advectionContribution * 100).toFixed(1)}%`);
  console.log(`  权重分布: [${baaResult.weights.join(', ')}]`);
  console.log(`  耗时: ${t2e.toFixed(2)}ms`);
  console.log(`  Top-5:`);
  baaResult.rankings.forEach((r, i) =>
    console.log(`    #${i + 1} doc[${r.index}] score=${r.score.toFixed(4)}`)
  );

  // 4. v2 基线对照
  console.log('\n── v2 基线（全局 adRankCore, dim=128）──');
  const t3 = performance.now();
  const v2Result = adRankCore(query, docs, N, { D: 0.15, uStrength: 0.1, knn: 3 });
  const t3e = performance.now() - t3;
  console.log(`  Re: ${v2Result.reynolds.toFixed(3)}, Pe: ${v2Result.peclet.toFixed(3)}`);
  console.log(`  耗时: ${t3e.toFixed(2)}ms`);

  // 5. Pe 对比
  console.log('\n── 信号放大验证 ──');
  console.log(`  v2 Pe (dim=${D}):   ${v2Result.peclet.toFixed(4)}`);
  console.log(`  MDA Pe (k=32):  ${mdaResult.peclet.toFixed(4)} (${(mdaResult.peclet / (v2Result.peclet || 0.001)).toFixed(1)}x)`);
  console.log(`  BAA Pe (B=8):   ${baaResult.peclet.toFixed(4)} (${(baaResult.peclet / (v2Result.peclet || 0.001)).toFixed(1)}x)`);
  console.log(`  对流贡献率: MDA ${(mdaResult.advectionContribution*100).toFixed(1)}% / BAA ${(baaResult.advectionContribution*100).toFixed(1)}%`);

  console.log('\n✅ 自测完成');
}
