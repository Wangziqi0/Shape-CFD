# Agent 1 指令：AD-Rank 核心求解器

> 你是负责实现 AD-Rank 核心对流-扩散求解器的 agent。
> 请严格按照以下规格实现。

## 背景

AD-Rank 是一个通用的向量检索增强排序算法，核心思想是在文档构成的图上运行对流-扩散方程，利用 query 的方向性（对流项）和文档间的语义传播（扩散项）来重新排序检索结果。

## 要创建的文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank.js`

## 接口定义

```js
/**
 * AD-Rank 对流-扩散图排序
 * 
 * @param {Float32Array} queryVector   - query 的 embedding (4096维)
 * @param {Float32Array[]} docVectors  - N 个候选文档的 embedding 数组
 * @param {number} k                  - 返回 top-k
 * @param {Object} [options]          - 可选参数
 * @param {number} [options.D=0.15]        - 扩散系数
 * @param {number} [options.uStrength=0.3] - 对流强度
 * @param {number} [options.maxIter=20]    - 最大迭代次数
 * @param {number} [options.epsilon=1e-6]  - 收敛阈值
 * @param {number} [options.knn=5]         - KNN 图的 k 值
 * @param {number} [options.dt=0.05]       - 时间步长
 * @returns {ADRankResult}
 */
function adRank(queryVector, docVectors, k, options = {}) { ... }

module.exports = { adRank };
```

## 返回值类型

```js
/**
 * @typedef {Object} ADRankResult
 * @property {Array<{index: number, score: number, topology: string}>} rankings
 * @property {number} reynolds       - 雷诺数
 * @property {number} iterations     - 实际迭代步数
 * @property {boolean} convergence   - 是否收敛
 * @property {number[]} convergencePoints - 汇聚点索引
 * @property {number[]} divergencePoints  - 发散点索引
 * @property {number[]} stagnationPoints  - 停滞点索引
 */
```

## 算法步骤（伪代码）

### Step 1: 计算余弦相似度

```js
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
```

### Step 2: 建 KNN 稀疏图

```
对每个文档 i:
  计算 i 与所有其他文档 j 的 cosine similarity
  取 top-k_nn 个最近邻
  边权重 W_ij = exp(-β × (1 - sim(i, j)))，β = 2.0

结果：adjacency[i] = [{ j, w }, ...]  (对称图)
```

### Step 3: 初始化浓度场

```js
C[i] = cosineSim(queryVector, docVectors[i])  // 每个文档的初始相关度
```

### Step 4: 计算对流方向（upwind 格式）

```
对每条边 (i, j):
  edgeDir = normalize(docVectors[j] - docVectors[i])  // 边所在的方向
  queryDir = normalize(queryVector)                      // query 方向
  u_ij = dot(edgeDir, queryDir) * uStrength              // 投影

注意：这里在 4096 维做 dot，不降维
```

### Step 5: 迭代求解

```
for t = 0 to maxIter:
  for each node i:
    // 扩散项: D × Σ W_ij × (C[j] - C[i])
    diffusion = D * Σ_j W_ij * (C[j] - C[i])
    
    // 对流项 (upwind): -Σ max(u_ij, 0) × W_ij × (C[j] - C[i])
    advection = 0
    for each neighbor j:
      if u_ij > 0:
        advection -= u_ij * W_ij * (C[j] - C[i])
    
    C_new[i] = C[i] + dt * (diffusion + advection)
    C_new[i] = clamp(C_new[i], 0, 1)
  
  if ||C_new - C|| < epsilon: break  // 收敛
  C = C_new
```

### Step 6: 流场拓扑分析

```
for each node i:
  div_i = Σ_j u_ij × W_ij × C[j]
  
  if div_i < -threshold:    topology = 'convergence'  (汇聚点)
  else if div_i > threshold: topology = 'divergence'  (发散点)
  else if |dC/dt| ≈ 0:      topology = 'stagnation'  (停滞点)
  else:                      topology = 'flow'        (流动)
```

### Step 7: 计算雷诺数

```
Re = |Σ 对流项| / |Σ 扩散项|
```

### Step 8: 排序返回

```
按 C_final 降序排列，返回 top-k
每个结果附带 { index, score: C_final[i], topology }
```

## 性能要求

- 30 个候选文档：< 5ms
- 100 个候选文档：< 20ms
- 不使用任何外部 npm 包，纯 JS 数学计算

## 测试

在文件末尾添加自测代码：

```js
if (require.main === module) {
  // 生成 20 个随机 128 维向量作为测试数据（小维度快速验证）
  const N = 20, D = 128;
  const query = new Float32Array(D).map(() => Math.random() - 0.5);
  const docs = Array.from({ length: N }, () =>
    new Float32Array(D).map(() => Math.random() - 0.5)
  );
  
  const result = adRank(query, docs, 5);
  console.log('=== AD-Rank 自测 ===');
  console.log('迭代:', result.iterations, '收敛:', result.convergence);
  console.log('Re:', result.reynolds.toFixed(3));
  console.log('Top-5:');
  result.rankings.forEach((r, i) =>
    console.log(`  #${i + 1} doc[${r.index}] score=${r.score.toFixed(4)} ${r.topology}`)
  );
  console.log('汇聚点:', result.convergencePoints.length);
  console.log('发散点:', result.divergencePoints.length);
}
```

## 参考

- 研究文档：`word/cfd创新/CFD-RAG研究完整总结.md` §3-4
- 全景图：`word/cfd创新/AD-Rank技术全景图.md`
