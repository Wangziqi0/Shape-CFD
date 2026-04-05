#!/usr/bin/env node
'use strict';

/**
 * AD-Rank Shape CFD V5 — 多灵感融合版
 *
 * 基于 ad_rank_shape.js (V4) 的增强版本，包含以下可开关的灵感模块：
 *   A: MaxSim 初始场解耦 (useMaxSimInit)
 *   B: Allen-Cahn 反应项 (useAllenCahn)
 *   D: 对流场硬截断 (useCutoff)
 *   Adaptive: 自适应 PDE 参数 (useAdaptive)
 *   Ensemble: 多参数投票 (useEnsemble)
 *   E: C 型分数分布诊断 (自动输出)
 *
 * 原始 V4 代码完全不修改，本文件独立运行。
 *
 * @author Chen, Yifan
 * @date 2026-03-27
 */

// 复用 V4 的基础设施
const {
  chamferDistance, centroid, cosineDistance: _cosDistV4,
  buildPointCloudsCached, buildQueryCloudCached,
  projectVector, PROJ_DIM, FULL_DIM,
} = require('./ad_rank_shape');

// ──────────────────────────────────────────────
// 工具函数
// ──────────────────────────────────────────────

/** 余弦距离 (8路展开，复制自 V4 以避免依赖问题) */
function cosineDistance(a, b) {
  const dim = a.length;
  let dot = 0, na = 0, nb = 0, d = 0;
  for (; d + 7 < dim; d += 8) {
    dot += a[d]*b[d] + a[d+1]*b[d+1] + a[d+2]*b[d+2] + a[d+3]*b[d+3]
         + a[d+4]*b[d+4] + a[d+5]*b[d+5] + a[d+6]*b[d+6] + a[d+7]*b[d+7];
    na  += a[d]*a[d] + a[d+1]*a[d+1] + a[d+2]*a[d+2] + a[d+3]*a[d+3]
         + a[d+4]*a[d+4] + a[d+5]*a[d+5] + a[d+6]*a[d+6] + a[d+7]*a[d+7];
    nb  += b[d]*b[d] + b[d+1]*b[d+1] + b[d+2]*b[d+2] + b[d+3]*b[d+3]
         + b[d+4]*b[d+4] + b[d+5]*b[d+5] + b[d+6]*b[d+6] + b[d+7]*b[d+7];
  }
  for (; d < dim; d++) { dot += a[d]*b[d]; na += a[d]*a[d]; nb += b[d]*b[d]; }
  return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

// ──────────────────────────────────────────────
// 灵感 A: MaxSim 距离（非对称）
// ──────────────────────────────────────────────

/**
 * 非对称 MaxSim 距离: 对 query 的每个句子向量，找 doc 中余弦最近的
 * MaxSim(q, d) = 1 - (1/|q|) Σ_{v∈q} max_{u∈d} cos(v, u)
 *
 * 解决对称 Chamfer 的"噪声稀释"问题：长文档只需一句命中即可得高分
 *
 * @param {Float32Array[]} queryCloud
 * @param {Float32Array[]} docCloud
 * @returns {number} 距离值 [0, 2]，越小越相关
 */
function maxSimDistance(queryCloud, docCloud) {
  let sumMaxSim = 0;
  for (const q of queryCloud) {
    let maxSim = -Infinity;
    for (const d of docCloud) {
      const sim = 1 - cosineDistance(q, d);
      if (sim > maxSim) maxSim = sim;
    }
    sumMaxSim += maxSim;
  }
  return 1 - (sumMaxSim / queryCloud.length);
}

/**
 * 对称 MaxSim (用于图边权)
 * SymMaxSim(A, B) = (MaxSim(A,B) + MaxSim(B,A)) / 2
 */
function symmetricMaxSimDistance(cloudA, cloudB) {
  return (maxSimDistance(cloudA, cloudB) + maxSimDistance(cloudB, cloudA)) / 2;
}

// ──────────────────────────────────────────────
// 灵感 D: 硬截断辅助
// ──────────────────────────────────────────────

/**
 * 计算边距离的第 p 百分位，用于自适应 ε 截断
 * @param {Float64Array} edgeNorms - 所有边的 ||ci - cj||² 值
 * @param {number} percentile - 百分位 (0-100)
 * @returns {number}
 */
function computePercentile(values, percentile) {
  const sorted = Array.from(values).filter(v => v > 0).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const idx = Math.floor(sorted.length * percentile / 100);
  return sorted[Math.min(idx, sorted.length - 1)];
}

// ──────────────────────────────────────────────
// 灵感 E: 分数分布诊断
// ──────────────────────────────────────────────

/**
 * 分析最终分数分布，检测是否形成 C 型间隔
 * @param {Float64Array} C - 最终浓度场
 * @param {number} theta - 阈值
 * @returns {Object} 分布诊断结果
 */
function diagnoseDistribution(C, theta) {
  const N = C.length;
  const above = [];
  const below = [];

  for (let i = 0; i < N; i++) {
    if (C[i] >= theta) above.push(C[i]);
    else below.push(C[i]);
  }

  above.sort((a, b) => b - a);
  below.sort((a, b) => b - a);

  // 计算簇间间隔
  const minAbove = above.length > 0 ? above[above.length - 1] : theta;
  const maxBelow = below.length > 0 ? below[0] : theta;
  const gap = minAbove - maxBelow;

  // 簇内离散度
  const aboveSpread = above.length > 1 ? above[0] - above[above.length - 1] : 0;
  const belowSpread = below.length > 1 ? below[0] - below[below.length - 1] : 0;

  // C 型判定：间隔 > 簇内离散度 * 0.3
  const isCType = gap > 0.02 && gap > Math.max(aboveSpread, belowSpread) * 0.3;

  return {
    type: isCType ? 'C-type' : (gap > 0.1 ? 'A-type' : 'smooth'),
    theta,
    aboveCount: above.length,
    belowCount: below.length,
    gap: gap,
    aboveSpread,
    belowSpread,
    aboveScores: above.slice(0, 5),  // 前5个高分
    belowScores: below.slice(0, 5),  // 前5个低分
  };
}

// ──────────────────────────────────────────────
// KNN 图构建（支持 MaxSim 和 Chamfer 两种度量）
// ──────────────────────────────────────────────

/**
 * 构建 KNN 图
 * @param {Float32Array[][]} clouds
 * @param {number} knn
 * @param {string} metric - 'chamfer' | 'maxsim'
 * @returns {{adjacency: Array, distMatrix: Float32Array}}
 */
function buildKNNGraph(clouds, knn, metric = 'chamfer') {
  const N = clouds.length;
  const beta = 2.0;
  const distFn = metric === 'maxsim' ? symmetricMaxSimDistance : chamferDistance;

  const distMatrix = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const d = distFn(clouds[i], clouds[j]);
      distMatrix[i * N + j] = d;
      distMatrix[j * N + i] = d;
    }
  }

  const adjacency = Array.from({ length: N }, () => []);
  const effectiveK = Math.min(knn, N - 1);
  const sortIdx = new Int32Array(N);
  const sortVal = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    let count = 0;
    for (let j = 0; j < N; j++) {
      if (j !== i) { sortIdx[count] = j; sortVal[count] = distMatrix[i * N + j]; count++; }
    }
    for (let t = 0; t < effectiveK; t++) {
      let minPos = t, minVal = sortVal[t];
      for (let p = t + 1; p < count; p++) {
        if (sortVal[p] < minVal) { minVal = sortVal[p]; minPos = p; }
      }
      if (minPos !== t) {
        sortVal[minPos] = sortVal[t]; sortVal[t] = minVal;
        const tmpIdx = sortIdx[minPos]; sortIdx[minPos] = sortIdx[t]; sortIdx[t] = tmpIdx;
      }
      adjacency[i].push({ j: sortIdx[t], w: Math.exp(-beta * minVal) });
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

  return { adjacency, distMatrix };
}

// ──────────────────────────────────────────────
// 主函数: adRankShapeV5
// ──────────────────────────────────────────────

/**
 * Shape-CFD V5 核心函数
 *
 * @param {Float32Array[]} queryProjCloud - query 128 维投影点云
 * @param {Float32Array[][]} docProjClouds - N 个文档 128 维投影点云
 * @param {Float32Array} queryCentroid_ - query 质心 (4096 维)
 * @param {Float32Array[]} docCentroids - N 个文档质心 (4096 维)
 * @param {number} k - 返回 top-k
 * @param {Object} [options]
 * @returns {Object} 排序结果
 */
function adRankShapeV5(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, options = {}) {
  let {
    // V4 基础参数
    D: diffCoeff  = 0.15,
    uStrength     = 0.1,
    maxIter       = 80,
    epsilon       = 5e-3,
    knn           = 3,
    dt            = 0.05,
    // ─── V5 灵感开关 ───
    useMaxSimInit = false,     // 灵感 A: MaxSim 初始场
    useCutoff     = false,     // 灵感 D: 硬截断
    useAllenCahn  = false,     // 灵感 B: Allen-Cahn 反应项
    allenCahnGamma = 0.5,      // γ 反应强度
    thetaMode     = 'fixed',   // θ 模式: 'fixed'|'dynamic'|'ema'
    thetaEmaAlpha = 0.1,       // EMA 平滑系数
    useAdaptive   = false,     // 处理器灵感: 自适应参数
    cutoffEpsilon = 'auto',    // ε: 'auto' 或具体数值
    graphMetric   = 'chamfer', // 图边权度量: 'chamfer'|'maxsim'
  } = options;

  const N = docProjClouds.length;
  if (N === 0) {
    return {
      rankings: [], reynolds: 0, peclet: 0, damkohler: 0,
      iterations: 0, convergence: true, distribution: null,
      convergencePoints: [], divergencePoints: [], stagnationPoints: [],
    };
  }

  // ─── 处理器灵感 Adaptive: 自适应参数 ───
  if (useAdaptive) {
    // 检测 query 复杂度
    const qSize = queryProjCloud.length;
    let querySpread = 0;
    if (qSize > 1) {
      for (let i = 0; i < qSize; i++) {
        for (let j = i + 1; j < qSize; j++) {
          const d = cosineDistance(queryProjCloud[i], queryProjCloud[j]);
          if (d > querySpread) querySpread = d;
        }
      }
    }

    if (qSize <= 1 && querySpread < 0.15) {
      // 浅层 query → 掩码 = 扩散主导
      diffCoeff = 0.2;
      uStrength = 0.05;
      maxIter = 40;
      if (useAllenCahn) allenCahnGamma = 0.2;
    } else if (qSize >= 3 || querySpread > 0.4) {
      // 深层 query → 掩码 = 对流+反应主导
      diffCoeff = 0.1;
      uStrength = 0.15;
      maxIter = 100;
      if (useAllenCahn) allenCahnGamma = 0.8;
    }
    // else: 中间层，保持默认参数
  }

  // ─── Step 1: 构建 KNN 图 ───
  const { adjacency } = buildKNNGraph(docProjClouds, knn, graphMetric);

  // ─── Step 2: 初始浓度 C₀ ───
  let C = new Float64Array(N);

  if (useMaxSimInit) {
    // 灵感 A: 非对称 MaxSim 初始场
    for (let i = 0; i < N; i++) {
      const d = maxSimDistance(queryProjCloud, docProjClouds[i]);
      C[i] = Math.exp(-2 * d);
    }
  } else {
    // V4 默认: 对称 Chamfer
    for (let i = 0; i < N; i++) {
      const d = chamferDistance(queryProjCloud, docProjClouds[i]);
      C[i] = Math.exp(-2 * d);
    }
  }

  // 记录初始 θ（用于 fixed 模式）
  let theta0 = 0;
  for (let i = 0; i < N; i++) theta0 += C[i];
  theta0 /= N;
  let thetaCurrent = theta0;

  // ─── Step 3: 对流方向 — 质心→质心 ───
  const dim = queryCentroid_.length;
  let qNorm = 0;
  for (let d = 0; d < dim; d++) qNorm += queryCentroid_[d] * queryCentroid_[d];
  const invQNorm = 1.0 / (Math.sqrt(qNorm) + 1e-8);
  const queryDir = new Float32Array(dim);
  for (let d = 0; d < dim; d++) queryDir[d] = queryCentroid_[d] * invQNorm;

  // 预计算对流系数 + 灵感 D 硬截断
  const U = new Float64Array(N * N);
  const edgeNormsSq = []; // 收集所有边的 ||ci-cj||²

  // 第一遍：收集 edgeNorms（用于 auto epsilon）
  for (let i = 0; i < N; i++) {
    const ci = docCentroids[i];
    for (const edge of adjacency[i]) {
      const j = edge.j;
      if (j <= i) continue; // 只算上三角
      const cj = docCentroids[j];
      let edNormSq = 0;
      for (let d = 0; d < dim; d++) {
        const diff = cj[d] - ci[d];
        edNormSq += diff * diff;
      }
      edgeNormsSq.push(edNormSq);
    }
  }

  // 灵感 D: 计算截断阈值
  let epsilonSq = 0;
  if (useCutoff) {
    if (cutoffEpsilon === 'auto') {
      epsilonSq = computePercentile(new Float64Array(edgeNormsSq), 5);
    } else {
      epsilonSq = cutoffEpsilon * cutoffEpsilon;
    }
  }

  let cutoffCount = 0;

  // 第二遍：计算对流系数
  for (let i = 0; i < N; i++) {
    const ci = docCentroids[i];
    for (const edge of adjacency[i]) {
      const j = edge.j;
      if (U[i * N + j] !== 0 || U[j * N + i] !== 0) continue;

      const cj = docCentroids[j];
      let edNormSq = 0, dotVal = 0;
      for (let d = 0; d < dim; d++) {
        const diff = cj[d] - ci[d];
        edNormSq += diff * diff;
        dotVal += diff * queryDir[d];
      }

      // 灵感 D: 方向不可靠时关闭对流
      if (useCutoff && edNormSq < epsilonSq) {
        U[i * N + j] = 0;
        U[j * N + i] = 0;
        cutoffCount++;
        continue;
      }

      const u_ij = (dotVal / (Math.sqrt(edNormSq) + 1e-8)) * uStrength;
      U[i * N + j] = u_ij;
      U[j * N + i] = -u_ij;
    }
  }

  // ─── Step 4: PDE 迭代求解 ───
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
  let totalReactionSum = 0;

  const threshold = 0.01;
  const dCdt_threshold = epsilon * 10;
  const topologies = new Array(N);
  const convergencePoints = [];
  const divergencePoints = [];
  const stagnationPoints = [];

  // θ 追踪历史（用于诊断震荡）
  const thetaHistory = [theta0];

  for (let t = 0; t < maxIter; t++) {
    let maxDelta = 0;
    let iterAdvection = 0;
    let iterDiffusion = 0;
    let iterReaction = 0;

    // 灵感 B: 计算当前 θ
    if (useAllenCahn && t > 0) {
      if (thetaMode === 'dynamic') {
        let sum = 0;
        for (let i = 0; i < N; i++) sum += C[i];
        thetaCurrent = sum / N;
      } else if (thetaMode === 'ema') {
        let sum = 0;
        for (let i = 0; i < N; i++) sum += C[i];
        const meanC = sum / N;
        thetaCurrent = thetaEmaAlpha * meanC + (1 - thetaEmaAlpha) * thetaCurrent;
      }
      // 'fixed' 模式保持 thetaCurrent = theta0
      thetaHistory.push(thetaCurrent);
    }

    for (let i = 0; i < N; i++) {
      let diffusion = 0;
      let advection = 0;
      const adj_i = adjacency[i];
      const C_i = C[i];
      const offset = i * N;

      for (let e = 0; e < adj_i.length; e++) {
        const edge = adj_i[e];
        const j = edge.j;
        const w = edge.w;
        const dC = C[j] - C_i;

        // 扩散项
        diffusion += diffCoeff * w * dC;

        // 守恒型 upwind 对流
        const u_ij = U[offset + j];
        const u_ji = U[j * N + i];
        advection += w * ((u_ji > 0 ? u_ji : 0) * C[j] - (u_ij > 0 ? u_ij : 0) * C_i);
      }

      // 灵感 B: Allen-Cahn 反应项
      let reaction = 0;
      if (useAllenCahn) {
        // γ × C × (1-C) × (C - θ)
        // C > θ → 推向 1; C < θ → 推向 0; 在 θ 附近制造鸿沟
        reaction = allenCahnGamma * C_i * (1 - C_i) * (C_i - thetaCurrent);
      }

      iterDiffusion += Math.abs(diffusion);
      iterAdvection += Math.abs(advection);
      iterReaction += Math.abs(reaction);

      let c_new = C_i + effective_dt * (diffusion + advection + reaction);
      if (c_new < 0) c_new = 0;
      if (c_new > 1) c_new = 1;
      C_new[i] = c_new;

      const delta = Math.abs(c_new - C_i);
      if (delta > maxDelta) maxDelta = delta;
    }

    totalAdvectionSum = iterAdvection;
    totalDiffusionSum = iterDiffusion;
    totalReactionSum = iterReaction;
    iterations = t + 1;

    // 交替缓冲区
    const tmp = C; C = C_new; C_new = tmp;

    // 收敛判断 + 拓扑分析
    if (maxDelta < epsilon || t === maxIter - 1) {
      for (let i = 0; i < N; i++) {
        let div_i = 0;
        const adj_i = adjacency[i];
        const offset = i * N;
        for (let e = 0; e < adj_i.length; e++) {
          const edge = adj_i[e];
          div_i += U[offset + edge.j] * edge.w * C[edge.j];
        }
        if (div_i < -threshold) { topologies[i] = 'convergence'; convergencePoints.push(i); }
        else if (div_i > threshold) { topologies[i] = 'divergence'; divergencePoints.push(i); }
        else {
          let diff = 0, adv = 0;
          const C_i = C[i];
          for (let e = 0; e < adj_i.length; e++) {
            const edge = adj_i[e]; const j = edge.j; const w = edge.w;
            diff += diffCoeff * w * (C[j] - C_i);
            const u_ij = U[offset + j]; const u_ji = U[j * N + i];
            adv += w * ((u_ji > 0 ? u_ji : 0) * C[j] - (u_ij > 0 ? u_ij : 0) * C_i);
          }
          const rate = diff + adv;
          if (Math.abs(rate) < dCdt_threshold) { topologies[i] = 'stagnation'; stagnationPoints.push(i); }
          else { topologies[i] = 'flow'; }
        }
      }
      if (maxDelta < epsilon) converged = true;
      break;
    }
  }

  // ─── Step 5: 物理量计算 ───
  const reynolds = totalDiffusionSum > 1e-12 ? totalAdvectionSum / totalDiffusionSum : 0;

  let uAbsSum = 0, edgeCount = 0;
  for (let i = 0; i < N; i++) {
    const adj_i = adjacency[i]; const offset = i * N;
    for (let e = 0; e < adj_i.length; e++) {
      const j = adj_i[e].j;
      if (j > i) { uAbsSum += Math.abs(U[offset + j]); edgeCount++; }
    }
  }
  const peclet = edgeCount > 0 && diffCoeff > 1e-12 ? (uAbsSum / edgeCount) / diffCoeff : 0;
  const damkohler = diffCoeff > 1e-12 ? allenCahnGamma / diffCoeff : 0;

  // ─── 灵感 E: 分数分布诊断 ───
  const distribution = diagnoseDistribution(C, thetaCurrent);

  // ─── Step 6: 排序 ───
  const ranked = new Array(N);
  for (let i = 0; i < N; i++) ranked[i] = { index: i, score: C[i], topology: topologies[i] };
  ranked.sort((a, b) => b.score - a.score);
  const topK = ranked.slice(0, Math.min(k, N));

  return {
    rankings: topK,
    reynolds,
    peclet,
    damkohler,
    iterations,
    convergence: converged,
    distribution,
    thetaHistory,
    cutoffCount,
    convergencePoints,
    divergencePoints,
    stagnationPoints,
    // V5 额外诊断
    v5Config: {
      useMaxSimInit, useCutoff, useAllenCahn, useAdaptive,
      allenCahnGamma, thetaMode, diffCoeff, uStrength,
      cutoffEpsilon: useCutoff ? Math.sqrt(epsilonSq) : null,
    },
  };
}

// ──────────────────────────────────────────────
// 处理器灵感 Ensemble: 多参数投票
// ──────────────────────────────────────────────

/**
 * 多参数 ensemble: 用 3 组不同 (Pe, Da) 参数跑 PDE，选最优
 * 类比 Zen 5 乱序执行：多个配置并行跑，退役队列选最好的
 */
function adRankShapeV5Ensemble(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, options = {}) {
  const configs = [
    { D: 0.2,  uStrength: 0.05, allenCahnGamma: 0.2, label: 'diffusion-dominant' },
    { D: 0.15, uStrength: 0.1,  allenCahnGamma: 0.5, label: 'balanced' },
    { D: 0.1,  uStrength: 0.15, allenCahnGamma: 0.8, label: 'reaction-dominant' },
  ];

  let bestResult = null;
  let bestSpread = -1;
  let bestLabel = '';

  for (const cfg of configs) {
    const mergedOpts = {
      ...options,
      D: cfg.D,
      uStrength: cfg.uStrength,
      allenCahnGamma: cfg.allenCahnGamma,
      useEnsemble: false, // 防止递归
    };

    const result = adRankShapeV5(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, mergedOpts);

    if (!result.convergence) continue; // 跳过未收敛的

    // 选择标准：最大分数展开率（top-1 与 top-last 的差距）
    const scores = result.rankings.map(r => r.score);
    const spread = scores.length > 1 ? scores[0] - scores[scores.length - 1] : 0;

    if (spread > bestSpread) {
      bestSpread = spread;
      bestResult = result;
      bestLabel = cfg.label;
    }
  }

  if (!bestResult) {
    // 全部未收敛，用默认参数
    bestResult = adRankShapeV5(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, {
      ...options, useEnsemble: false,
    });
    bestLabel = 'fallback';
  }

  bestResult.ensembleWinner = bestLabel;
  return bestResult;
}

// ──────────────────────────────────────────────
// 统一入口
// ──────────────────────────────────────────────

/**
 * V5 统一入口 — 根据 useEnsemble 决定走单次还是多参数投票
 */
function adRankShapeV5Entry(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, options = {}) {
  if (options.useEnsemble) {
    return adRankShapeV5Ensemble(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, options);
  }
  return adRankShapeV5(queryProjCloud, docProjClouds, queryCentroid_, docCentroids, k, options);
}

// ──────────────────────────────────────────────
// 预设配置
// ──────────────────────────────────────────────

const V5_PRESETS = {
  // 单灵感测试
  'v5-maxsim':     { useMaxSimInit: true },
  'v5-cutoff':     { useMaxSimInit: true, useCutoff: true },
  'v5-allen-cahn': { useMaxSimInit: true, useCutoff: true, useAllenCahn: true, thetaMode: 'fixed' },
  'v5-adaptive':   { useMaxSimInit: true, useCutoff: true, useAllenCahn: true, thetaMode: 'fixed', useAdaptive: true },
  'v5-ensemble':   { useMaxSimInit: true, useCutoff: true, useAllenCahn: true, thetaMode: 'fixed', useEnsemble: true },
  // 完整 V5
  'v5-full':       { useMaxSimInit: true, useCutoff: true, useAllenCahn: true, thetaMode: 'ema', useAdaptive: true },
  // 消融对照
  'v5-B-only':     { useAllenCahn: true, thetaMode: 'fixed' },
  'v5-D-only':     { useCutoff: true },
  'v5-AB':         { useMaxSimInit: true, useAllenCahn: true, thetaMode: 'fixed' },
  'v5-AD':         { useMaxSimInit: true, useCutoff: true },
};

// ──────────────────────────────────────────────
// 导出
// ──────────────────────────────────────────────

module.exports = {
  adRankShapeV5: adRankShapeV5Entry,
  adRankShapeV5Core: adRankShapeV5,
  adRankShapeV5Ensemble,
  maxSimDistance,
  symmetricMaxSimDistance,
  diagnoseDistribution,
  V5_PRESETS,
  // 复用 V4 基础设施
  buildPointCloudsCached,
  buildQueryCloudCached,
  projectVector,
  PROJ_DIM,
  FULL_DIM,
};

// ──────────────────────────────────────────────
// 自测
// ──────────────────────────────────────────────

if (require.main === module) {
  console.log('═'.repeat(60));
  console.log('  Shape-CFD V5 自测 (合成数据)');
  console.log('═'.repeat(60));

  const D = 128;
  const makeVec = () => new Float32Array(D).map(() => Math.random() - 0.5);

  // 模拟 8 个文档点云
  const docClouds = Array.from({ length: 8 }, () => {
    const n = 2 + Math.floor(Math.random() * 4); // 2-5 句
    return Array.from({ length: n }, makeVec);
  });
  const queryCloud = [makeVec(), makeVec()];
  const docCentroids = docClouds.map(c => centroid(c));
  const queryCentroid_ = centroid(queryCloud);

  // 测试 MaxSim 距离
  console.log('\n📐 MaxSim 距离测试:');
  for (let i = 0; i < 3; i++) {
    const msd = maxSimDistance(queryCloud, docClouds[i]);
    const cd = chamferDistance(queryCloud, docClouds[i]);
    console.log(`  query ↔ doc[${i}]: MaxSim=${msd.toFixed(4)} Chamfer=${cd.toFixed(4)}`);
  }

  // 测试各配置
  const configs = ['v5-maxsim', 'v5-cutoff', 'v5-allen-cahn', 'v5-adaptive', 'v5-ensemble', 'v5-full'];

  for (const preset of configs) {
    console.log(`\n🏄 ${preset}:`);
    const t0 = performance.now();
    const result = adRankShapeV5Entry(queryCloud, docClouds, queryCentroid_, docCentroids, 8, {
      ...V5_PRESETS[preset],
    });
    const elapsed = performance.now() - t0;

    console.log(`  迭代: ${result.iterations} 收敛: ${result.convergence} Pe: ${result.peclet.toFixed(3)} Da: ${result.damkohler.toFixed(3)}`);
    console.log(`  分布: ${result.distribution.type} (gap=${result.distribution.gap.toFixed(4)}, above=${result.distribution.aboveCount}, below=${result.distribution.belowCount})`);
    if (result.cutoffCount) console.log(`  截断边: ${result.cutoffCount}`);
    if (result.ensembleWinner) console.log(`  Ensemble 胜出: ${result.ensembleWinner}`);
    console.log(`  Top-3: ${result.rankings.slice(0, 3).map(r => `doc${r.index}=${r.score.toFixed(3)}`).join(' | ')}`);
    console.log(`  耗时: ${elapsed.toFixed(1)}ms`);
  }

  // 验证点
  console.log('\n' + '─'.repeat(60));
  console.log('✅ V5 自测完成');
  console.log('─'.repeat(60));
}
