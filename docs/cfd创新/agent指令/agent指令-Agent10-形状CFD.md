# Agent 10 指令：形状 CFD (Shape CFD MVP)

> 工作量：大（~250 行新代码 + 实验）
> 依赖：需 embedding API 在线
> 这是最具创新价值的实验

## 目标

验证"向量不是点，是形状"的核心假设：将每个文档从单个向量扩展为点云（按句子拆分），用 Chamfer 距离替代 cosine 构建 KNN 图。

## 核心思想

```
当前 (点 CFD):
  doc_i = 1 个 4096 维向量
  sim(i,j) = cosine(vec_i, vec_j)

形状 CFD:
  doc_i = {sent_i_1, sent_i_2, ..., sent_i_K} (K 个句子的向量)
  sim(i,j) = 1 / chamfer_distance(cloud_i, cloud_j)
  
  对流方向 = 文档点云质心 → query 点云质心
```

## 要创建的文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank_shape.js`

## 实现步骤

### Step 1：文本拆句 + 批量 Embedding

```js
const { ADRankData } = require('./ad_rank_data');

/**
 * 将候选法条拆成句子并 embed
 * @param {Object[]} metas - 候选元数据 (含 content 字段)
 * @returns {Float32Array[][]} - 每个文档的句子向量数组（点云）
 */
async function buildPointClouds(metas, engine) {
  const clouds = [];
  
  for (const meta of metas) {
    const text = meta.content || '';
    // 按句号/分号拆句
    const sentences = text.split(/[。；;]/).filter(s => s.trim().length > 5);
    
    if (sentences.length <= 1) {
      // 只有一句，用原始整句向量，退化为点
      const vec = await engine.embed(text);
      clouds.push([vec instanceof Float32Array ? vec : new Float32Array(vec)]);
    } else {
      // 多句，批量 embed
      const vecs = await engine.embedBatch(sentences);
      const validVecs = vecs
        .map(v => v ? (v instanceof Float32Array ? v : new Float32Array(v)) : null)
        .filter(Boolean);
      clouds.push(validVecs.length > 0 ? validVecs : [await engine.embed(text)]);
    }
  }
  
  return clouds;
}
```

### Step 2：Chamfer 距离

```js
/**
 * 两个点云之间的 Chamfer 距离
 * Chamfer(A, B) = (1/|A|) Σ min_b d(a,b) + (1/|B|) Σ min_a d(a,b)
 * 
 * @param {Float32Array[]} cloudA
 * @param {Float32Array[]} cloudB
 * @returns {number}
 */
function chamferDistance(cloudA, cloudB) {
  let sumAB = 0;
  for (const a of cloudA) {
    let minDist = Infinity;
    for (const b of cloudB) {
      const d = cosineDistance(a, b);  // 用 1 - cosine 作为距离
      if (d < minDist) minDist = d;
    }
    sumAB += minDist;
  }
  
  let sumBA = 0;
  for (const b of cloudB) {
    let minDist = Infinity;
    for (const a of cloudA) {
      const d = cosineDistance(a, b);
      if (d < minDist) minDist = d;
    }
    sumBA += minDist;
  }
  
  return sumAB / cloudA.length + sumBA / cloudB.length;
}

function cosineDistance(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < a.length; d++) {
    dot += a[d] * b[d];
    na += a[d] * a[d];
    nb += b[d] * b[d];
  }
  return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
```

### Step 3：点云质心对流

```js
/**
 * 计算点云质心
 */
function centroid(cloud) {
  const dim = cloud[0].length;
  const c = new Float32Array(dim);
  for (const v of cloud) {
    for (let d = 0; d < dim; d++) c[d] += v[d];
  }
  for (let d = 0; d < dim; d++) c[d] /= cloud.length;
  return c;
}

/**
 * 形状 CFD 对流方向：质心→质心
 * u_ij = dot(normalize(centroid_j - centroid_i), normalize(centroid_query))
 */
```

### Step 4：用 Chamfer 距离替代 cosine 构建 KNN 图

```js
function buildShapeKNNGraph(clouds, knn) {
  const N = clouds.length;
  
  // N×N Chamfer 距离矩阵
  const distMatrix = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const d = chamferDistance(clouds[i], clouds[j]);
      distMatrix[i * N + j] = d;
      distMatrix[j * N + i] = d;
    }
  }
  
  // KNN 选择（距离最小的 k 个）
  // 边权重 W_ij = exp(-beta * d_ij)
  // ...
}
```

### Step 5：在新图上跑现有 adRank 核心

```js
function adRankShape(queryCloud, docClouds, topK, options = {}) {
  // 1. 构建 Chamfer KNN 图
  const adjacency = buildShapeKNNGraph(docClouds, options.knn || 3);
  
  // 2. 初始浓度 = query 点云与各文档点云的 Chamfer 相似度
  const queryCentroid = centroid(queryCloud);
  const C0 = docClouds.map(cloud => {
    const d = chamferDistance(queryCloud, cloud);
    return Math.exp(-2 * d);  // 转为相似度
  });
  
  // 3. 对流方向用质心
  const docCentroids = docClouds.map(centroid);
  // ... 计算 u_ij 用质心方向
  
  // 4. 迭代求解（复用现有核心逻辑）
  // ...
}
```

## 验证实验

对 5 个 query 做 A/B/C 对比：
```
组 A: cosine (基线)
组 B: AD-Rank v2 (点 CFD)
组 C: Shape CFD (Chamfer 距离)
```

记录：
1. Chamfer 距离 vs cosine 的排序差异
2. 形状 CFD 的 Pe 是否比点 CFD 更高
3. DeepSeek 盲评

**注意**：形状 CFD 需要额外的 embedding API 调用（拆句 embed），可能较慢。记录耗时。

## ⚠️ 重要约束

- **缓存**：句子级 embedding 结果也要缓存到 SQLite（避免重复 API 调用）
- **降级**：如果法条只有一句话，退化为点 → 与 v2 一致
- **Query 也要拆**：query 文本也按逗号分割成点云
- 不要修改 `ad_rank.js`
