# Agent 5 指令：AD-Rank 求解器极致性能优化

> 目标：将 `ad_rank.js` 的求解时间从 ~26ms 优化到 <5ms (30 候选, 4096 维)
> 优化后必须跑 `node ad_rank_benchmark.js` 验证速度和排序质量不退化

## 当前性能基线

```
30 候选, 4096 维, 20 轮迭代:
  总求解时间: ~26ms
  瓶颈分布 (估算):
    N² cosine sim:          ~8ms (900 对 × 4096 维)
    Map<string> u_ij 查找:  ~6ms (字符串拼接 + hash)
    迭代循环:               ~5ms
    normalize + 临时分配:   ~4ms
    拓扑分析:               ~3ms
```

## 要修改的文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank.js`

## 优化清单（按优先级排序）

### 优化 1：Map<string> → 平铺 Float64Array（必做，省 ~6ms）

**当前**：`uMap = new Map(); uMap.set(\`${i}-${j}\`, value);` — 每次查找都做字符串拼接 + hash
**替换**：`const U = new Float64Array(N * N); U[i * N + j] = value;` — O(1) 数组下标访问

```js
// 替换 computeAdvection 函数
function computeAdvectionFlat(docVectors, queryVector, adjacency, uStrength) {
  const N = adjacency.length;
  const dim = queryVector.length;
  const U = new Float64Array(N * N); // 平铺二维数组
  
  // 预计算 normalize(queryVector) 一次
  const queryDir = normalize(queryVector);
  
  for (let i = 0; i < N; i++) {
    for (const edge of adjacency[i]) {
      const j = edge.j;
      if (U[i * N + j] !== 0 || U[j * N + i] !== 0) continue; // 已算过
      
      // edgeDir = normalize(doc[j] - doc[i]) → 直接内联，避免临时数组
      let edNorm = 0;
      let dotVal = 0;
      for (let d = 0; d < dim; d++) {
        const diff = docVectors[j][d] - docVectors[i][d];
        edNorm += diff * diff;
        dotVal += diff * queryDir[d]; // 合并 normalize 和 dot 为一步
      }
      edNorm = Math.sqrt(edNorm) + 1e-8;
      const u_ij = (dotVal / edNorm) * uStrength;
      
      U[i * N + j] = u_ij;
      U[j * N + i] = -u_ij; // 反对称：u_ji = -u_ij（边方向反转）
    }
  }
  
  return U;
}
```

迭代循环中的查找也要改：
```js
// 替换: const u_ij = uMap.get(`${i}-${j}`) || 0;
// 为:    const u_ij = U[i * N + j];
```

### 优化 2：消除临时数组分配（必做，省 ~4ms）

**当前**：`normalize()` 和边方向计算都创建 `new Float32Array(dim)`
**替换**：预分配缓冲区，重复使用

```js
// 在 adRank 函数顶部预分配
const _edgeBuf = new Float32Array(dim);  // 复用缓冲区

// normalize 也内联到计算中（见优化 1 的合并写法）
```

### 优化 3：相似度矩阵用 Float32 + 4 路展开（必做，省 ~3ms）

```js
function buildKNNGraphFast(docVectors, knn) {
  const N = docVectors.length;
  const dim = docVectors[0].length;
  const beta = 2.0;

  // 预计算每个向量的 L2 范数
  const norms = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    const v = docVectors[i];
    for (let d = 0; d < dim; d++) s += v[d] * v[d];
    norms[i] = Math.sqrt(s) + 1e-8;
  }

  // N×N 相似度（只算上三角）
  const simMatrix = new Float32Array(N * N);  // Float32 够用（省内存+快）
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const vi = docVectors[i], vj = docVectors[j];
      let dot = 0;
      // 4 路展开
      let d = 0;
      for (; d + 3 < dim; d += 4) {
        dot += vi[d]*vj[d] + vi[d+1]*vj[d+1] + vi[d+2]*vj[d+2] + vi[d+3]*vj[d+3];
      }
      for (; d < dim; d++) dot += vi[d] * vj[d];
      
      const s = dot / (norms[i] * norms[j]);
      simMatrix[i * N + j] = s;
      simMatrix[j * N + i] = s;
    }
  }

  // KNN 选择（用 partial sort 代替全排序）
  const adjacency = Array.from({ length: N }, () => []);
  const effectiveK = Math.min(knn, N - 1);
  const candidates = new Float32Array(N); // 复用

  for (let i = 0; i < N; i++) {
    // 复制一行相似度
    for (let j = 0; j < N; j++) candidates[j] = simMatrix[i * N + j];
    candidates[i] = -Infinity; // 排除自身

    // 选 top-k（partial quickselect 代替全排序）
    // 简化版：N=30 时全排序也够快
    const indexed = [];
    for (let j = 0; j < N; j++) {
      if (j !== i) indexed.push({ j, sim: candidates[j] });
    }
    indexed.sort((a, b) => b.sim - a.sim);

    for (let t = 0; t < effectiveK; t++) {
      const { j, sim } = indexed[t];
      adjacency[i].push({ j, w: Math.exp(-beta * (1 - sim)) });
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

  return adjacency;
}
```

### 优化 4：合并拓扑分析到最后一轮迭代（必做，省 ~3ms）

**当前**：迭代结束后单独遍历一轮做拓扑分析
**替换**：在最后一轮迭代中同时计算拓扑

```js
for (let t = 0; t < maxIter; t++) {
  const isLastIter = (t === maxIter - 1); // 或已收敛的当前轮
  // ... 正常迭代 ...
  
  if (isLastIter || maxDelta < epsilon) {
    // 在同一轮中计算拓扑
    for (let i = 0; i < N; i++) {
      let div_i = 0;
      for (const edge of adjacency[i]) {
        div_i += U[i * N + edge.j] * edge.w * C[edge.j];
      }
      // 设置 topology...
    }
    if (maxDelta < epsilon) converged = true;
    iterations = t + 1;
    break;
  }
}
```

### 优化 5：提前收敛（建议做，省 ~2ms）

```js
const defaults = {
  maxIter: 50,      // 20→50 但通常会提前收敛
  epsilon: 1e-3,    // 1e-6→1e-3 排序不需要高精度
  dt: 0.1,          // 0.05→0.1 加速收敛
};

// 自适应 dt (CFL 条件)
const maxDegree = Math.max(...adjacency.map(a => a.length));
const cfl_dt = maxDegree > 0 ? 0.8 / maxDegree : dt;
const effective_dt = Math.min(dt, cfl_dt);
```

### 优化 6：减少候选数（可选，按需调整）

在 benchmark 配置中把 `preFilterK` 从 30 减到 20：
- 30 候选: ~5ms (优化后)
- 20 候选: ~2ms (N² 效应)

## 修改后的完整 adRank 函数签名（不变）

```js
function adRank(queryVector, docVectors, k, options = {}) { ... }
module.exports = { adRank };
```

接口和返回值格式**完全不变**，只有内部实现变快。

## 验证步骤

1. 自测：
```bash
node ad_rank.js
# 期望: 20 docs 128 维 < 2ms, 100 docs 128 维 < 5ms
```

2. 基线实验：
```bash
node ad_rank_benchmark.js
# 期望: 
#   AD-Rank 求解 < 5ms (之前 ~26ms)
#   Top-10 重叠率 ≈ 50% (不退化)
#   收敛率 > 50% (之前 0/10)
```

3. 排序质量对比：
   - 优化前后对同样的 10 个 query 跑，对比 Top-10 排序是否一致
   - 允许因 epsilon 放宽导致的微小差异，但 Top-3 不应改变

## ⚠️ 重要约束

- **不能改接口签名和返回值结构**
- **不能改物理方程**（守恒型 upwind 格式不能动）
- **不能引入外部依赖**（纯 JS 数学计算）
- **优化只限于实现层面**（数据结构、内存分配、循环优化）
