# Agent 7 指令：多方向对流 (MDA) + 分块对流 (BAA) 实现与验证

> 实现两种对流增强策略，与 v2 基线对比，找出最优方案。

## 背景

AD-Rank v2 的对流信号极弱（Pe=0.065），根因是维度诅咒：
d=4096 时 dot product 标准差 σ = 1/√4096 ≈ 0.0156。
实验中对流仅贡献 16% 排序差异。

**核心思路**：通过降维或分块放大对流信号。

## 要创建的文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank_v3.js`

## 方案 A：多方向对流 (MDA, Multi-Directional Advection)

### 原理

生成 M 个随机投影矩阵，将 4096 维投影到 k 维子空间。
每个子空间独立计算对流方向和浓度场，最后加权融合。

```
信号增强：σ = 1/√k
  k=32:  σ ≈ 0.177 → 比全局强 11x
  k=64:  σ ≈ 0.125 → 比全局强 8x
  k=128: σ ≈ 0.088 → 比全局强 5.6x
```

### 实现

```js
function adRankMDA(queryVector, docVectors, topK, options = {}) {
  const {
    M = 8,           // 投影方向数
    k = 32,          // 每个方向的投影维度
    D = 0.15,
    uStrength = 0.3, // 注意：MDA 中可以用更大的 uStrength
    knn = 3,
    maxIter = 50,
    epsilon = 1e-3,
    dt = 0.1,
    seed = 42,       // 随机种子（确保可复现）
  } = options;

  const N = docVectors.length;
  const dim = docVectors[0].length;

  // 1. 生成 M 个随机投影矩阵 (正交随机投影)
  // 用固定种子，每次查询用相同的投影
  const projections = generateRandomProjections(M, k, dim, seed);

  // 2. 对每个方向独立求解
  const concentrations = []; // M 个浓度场
  const weights = [];        // M 个权重

  for (let m = 0; m < M; m++) {
    const P = projections[m]; // k × dim 投影矩阵

    // 投影 query 和所有 doc
    const qProj = projectVector(queryVector, P);
    const docsProj = docVectors.map(v => projectVector(v, P));

    // 在投影空间中跑 adRank
    // 注意：这里直接复用现有的 adRank 核心逻辑
    const result = adRankCore(qProj, docsProj, N, { D, uStrength, knn, maxIter, epsilon, dt });
    concentrations.push(result.C);

    // 权重 = query 投影后的能量 (L2 范数)
    // 如果 query 在某个方向投影很弱，说明这个方向对当前 query 没意义
    let qEnergy = 0;
    for (let d = 0; d < k; d++) qEnergy += qProj[d] * qProj[d];
    weights.push(Math.sqrt(qEnergy));
  }

  // 3. 归一化权重
  const wSum = weights.reduce((a, b) => a + b, 0) || 1;
  for (let m = 0; m < M; m++) weights[m] /= wSum;

  // 4. 加权融合浓度场
  const C_final = new Float64Array(N);
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < N; i++) {
      C_final[i] += weights[m] * concentrations[m][i];
    }
  }

  // 5. 排序和拓扑分析（同 v2）
  // ...
}

// 随机投影矩阵：每行是 dim 维的随机单位向量
function generateRandomProjections(M, k, dim, seed) {
  // 用简单的线性同余随机数（固定种子）
  let rng = seed;
  const nextRand = () => { rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF; return (rng >>> 0) / 0xFFFFFFFF - 0.5; };

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
```

## 方案 B：分块对流 (BAA, Block-Adaptive Advection)

### 原理

将 4096 维按连续块分割（如 8×512），每块独立建 KNN 图、算对流。

```
Block 0: dim[0:511]   → 独立 KNN 图 → 独立 C_0
Block 1: dim[512:1023] → 独立 KNN 图 → 独立 C_1
...
Block 7: dim[3584:4095] → 独立 KNN 图 → 独立 C_7

C_final = Σ w_b × C_b
```

### 实现

```js
function adRankBAA(queryVector, docVectors, topK, options = {}) {
  const {
    B = 8,            // 分块数
    D = 0.15,
    uStrength = 0.3,
    knn = 3,
    maxIter = 50,
    epsilon = 1e-3,
    dt = 0.1,
  } = options;

  const N = docVectors.length;
  const dim = docVectors[0].length;
  const blockSize = Math.floor(dim / B);

  const concentrations = [];
  const weights = [];

  for (let b = 0; b < B; b++) {
    const start = b * blockSize;
    const end = (b === B - 1) ? dim : start + blockSize;
    const bDim = end - start;

    // 提取块向量
    const qBlock = queryVector.slice(start, end);
    const docsBlock = docVectors.map(v => v.slice(start, end));

    // 在块空间中跑 adRank
    const result = adRankCore(qBlock, docsBlock, N, { D, uStrength, knn, maxIter, epsilon, dt });
    concentrations.push(result.C);

    // 权重 = query 在此块的 L2 能量
    let qEnergy = 0;
    for (let d = 0; d < bDim; d++) qEnergy += qBlock[d] * qBlock[d];
    weights.push(Math.sqrt(qEnergy));
  }

  // 归一化权重 + 融合（同 MDA）
  // ...
}
```

## 共用的 adRankCore 函数

从现有 `ad_rank.js` 提取核心求解逻辑为一个纯函数：

```js
/**
 * 核心求解器（不含初始化、不含拓扑分析后处理）
 * @param {Float32Array} qVec - query 向量 (任意维度)
 * @param {Float32Array[]} docVecs - 文档向量
 * @param {number} N - 文档数
 * @param {object} opts - 超参数
 * @returns {{ C: Float64Array, reynolds: number, iterations: number, convergence: boolean }}
 */
function adRankCore(qVec, docVecs, N, opts) {
  // ... 复用 ad_rank.js 的核心逻辑
  // 建 KNN 图 → 初始化 C → 迭代对流-扩散 → 返回 C
}
```

## 验证实验

创建 `ad_rank_v3_benchmark.js`，四组对比：

```
组 A: cosine (基线)
组 B: AD-Rank v2 (全局对流, D=0.15 u=0.1 knn=3)
组 C: MDA (M=8, k=32, u=0.3)
组 D: BAA (B=8, u=0.3)
```

对 10 个标准 query 跑：
1. 速度
2. Top-10 重叠率（相对组 A）
3. 对流贡献率（与纯扩散对照的 B vs C/D 差异）
4. 调 DeepSeek 做 A vs C 和 A vs D 的盲评（沿用 `ad_rank_blind_eval.js` 的模式）

### 参数搜索

MDA 关键参数：
- M: [4, 8, 16]
- k: [16, 32, 64]
- uStrength: [0.1, 0.3, 0.5]  ← 注意 MDA 中可以用更大的 u

BAA 关键参数：
- B: [4, 8, 16]
- uStrength: [0.1, 0.3, 0.5]

### 验证标准

1. MDA 或 BAA 的对流贡献率 > 16%（超过 v2 的 16%）
2. DeepSeek 盲评分数 > v2 的 3.3/5
3. 速度 < 30ms

## 依赖

- `ad_rank.js` (现有 v2，提取 core 逻辑)
- `ad_rank_data.js` (数据接口，含缓存)
- `ad_rank_blind_eval.js` (盲评模式参考)
- DeepSeek API (`.env` 中的 DEEPSEEK_API_KEY)

## 文件清单

| 文件 | 操作 |
|:---|:---|
| `ad_rank_v3.js` | 新建：MDA + BAA + adRankCore |
| `ad_rank_v3_benchmark.js` | 新建：四组对比 + DeepSeek 盲评 |
| `word/cfd创新/experiment_v3.json` | 输出：实验结果 |

## ⚠️ 重要约束

- `ad_rank.js` 不要改（v2 保留为基线）
- 新文件 `ad_rank_v3.js` 独立实现
- 随机投影矩阵用固定种子（seed=42），确保**可复现**
- 投影矩阵在模块加载时一次性生成，不要每次查询都重新生成
