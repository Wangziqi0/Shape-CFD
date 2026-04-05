# Agent 9 指令：自适应参数（Fiedler Value + 跨域检测）

> 工作量：中（~150 行新代码）
> 依赖：无，可与 Agent 7/8 并行

## 目标

让 AD-Rank 的超参数根据候选集的图结构自动调整，消除手动调参需求。

## 修改文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank.js` — 新增自适应逻辑

## 方案 A：Fiedler Value 自动确定 D

**原理**：拉普拉斯矩阵的第二小特征值 λ₂（Fiedler value）衡量图的代数连通度。
- λ₂ 大 → 图连通性强（候选紧密聚集）→ D 应小（防止过度平滑）
- λ₂ 小 → 图连通性弱（候选分散）→ D 应大（信号需要更强扩散传播）

```js
/**
 * 计算 Fiedler Value (图拉普拉斯第二小特征值)
 * 对 N=30 的小图，用幂迭代即可
 */
function computeFiedlerValue(adjacency) {
  const N = adjacency.length;
  
  // 构建拉普拉斯矩阵 L = D - W
  const L = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    let degree = 0;
    for (const edge of adjacency[i]) {
      L[i * N + edge.j] = -edge.w;
      degree += edge.w;
    }
    L[i * N + i] = degree;
  }
  
  // 用移位逆幂迭代求第二小特征值
  // (最小特征值恒为 0，对应常数特征向量)
  // ... 具体实现用 Rayleigh quotient 迭代
  
  return lambda2;
}

/**
 * 根据 Fiedler value 自动确定 D
 */
function autoD(lambda2, baseD = 0.15) {
  // λ₂ 越大 → D 越小；λ₂ 越小 → D 越大
  // D = baseD / (1 + λ₂)
  return baseD / (1 + lambda2);
}
```

## 方案 B：跨域检测自动调 uStrength

**原理**：分析 KNN 子图的结构，判断候选是否来自多个不同聚类（跨域）。

```js
/**
 * 检测候选集是否跨域
 * 用图密度和特征值分布判断
 */
function detectCrossDomain(adjacency, simMatrix, N) {
  // 方法1: 图密度（边数 / 最大可能边数）
  let totalEdges = 0;
  for (let i = 0; i < N; i++) totalEdges += adjacency[i].length;
  const density = totalEdges / (N * (N - 1));
  
  // 方法2: 平均相似度的方差
  let simValues = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      simValues.push(simMatrix[i * N + j]);
    }
  }
  const simMean = simValues.reduce((a, b) => a + b, 0) / simValues.length;
  const simVar = simValues.reduce((a, b) => a + (b - simMean) ** 2, 0) / simValues.length;
  
  // 高方差 = 候选分散 = 可能跨域 → 增大 uStrength
  // 低方差 = 候选聚集 = 单一领域 → 减小 uStrength
  const isCrossDomain = simVar > 0.02; // 阈值从实验中定
  
  return {
    isCrossDomain,
    density,
    simVar,
    suggestedU: isCrossDomain ? 0.3 : 0.1,
  };
}
```

## 实现要求

1. 在 `adRank()` 的 options 中加入：
   - `autoD: false` — 是否自动确定 D
   - `autoU: false` — 是否自动调整 uStrength
   - 两者当 `true` 时覆盖手动值

2. 在返回值中加入：
   - `fiedlerValue` — λ₂ 值
   - `crossDomain` — 是否跨域
   - `autoParams` — 自动确定的 D 和 uStrength

## 验证

对 10 个标准 query 跑，对比：
- 手动最优参数 (D=0.15, u=0.1)
- 自适应参数 (autoD=true, autoU=true)

期望：
1. 跨域 query（离婚财产分割）自动增大 u
2. 单域 query（醉驾量刑）自动减小 u
3. 整体盲评分数不低于手动调参
