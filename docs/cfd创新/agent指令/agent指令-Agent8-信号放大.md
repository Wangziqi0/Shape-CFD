# Agent 8 指令：对流信号放大（局部 PCA + 温度缩放）

> 工作量：小（~100 行新代码）
> 依赖：无，可与 Agent 7/9 并行

## 目标

在现有 `ad_rank.js` 的对流计算中应用两种信号放大技术，提升对流贡献率。

## 修改文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank.js` — 新增两个可选模式

## 方案 A：局部 PCA 降维

在计算对流方向前，对 30 个候选 + query 做局部 SVD，投影到前 k 个主成分。

```js
/**
 * 局部 PCA：将对流方向计算限制在候选集的主子空间中
 * @param {Float32Array} queryVec - 4096 维
 * @param {Float32Array[]} docVecs - 30 × 4096
 * @param {number} k - 主成分数 (默认 32)
 * @returns {{ qProj, docsProj }} - k 维投影
 */
function localPCA(queryVec, docVecs, k = 32) {
  const N = docVecs.length;
  const dim = queryVec.length;
  
  // 1. 计算均值
  const mean = new Float64Array(dim);
  for (const v of docVecs) for (let d = 0; d < dim; d++) mean[d] += v[d];
  for (let d = 0; d < dim; d++) mean[d] /= N;
  
  // 2. 中心化
  const centered = docVecs.map(v => {
    const c = new Float32Array(dim);
    for (let d = 0; d < dim; d++) c[d] = v[d] - mean[d];
    return c;
  });
  
  // 3. 幂迭代法求前 k 个主成分 (N=30, 幂迭代比全 SVD 快)
  // 或：直接用 N×N 协方差矩阵 (30×30) 的特征分解，然后回投
  // (因为 N << d, 用 AA^T 比 A^T A 快得多)
  const G = new Float64Array(N * N); // Gram 矩阵 30×30
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += centered[i][d] * centered[j][d];
      G[i * N + j] = dot;
      G[j * N + i] = dot;
    }
  }
  
  // 4. 对 30×30 Gram 矩阵做特征分解 (幂迭代求前 k 个)
  // ... 得到主方向矩阵 P (k × dim)
  
  // 5. 投影
  // qProj = P × (query - mean), docsProj = P × centered
}
```

**注意**：只在计算对流方向 `u_ij` 时使用 PCA 投影。KNN 图构建和扩散仍在原始空间进行。

## 方案 B：温度缩放

对流系数的非线性放大：

```js
// 原始: u_ij = dot(edgeDir, queryDir) × uStrength
// 温度缩放: u_ij = sgn(x) × |x|^alpha × uStrength (alpha < 1 放大微弱信号)

function temperatureScale(u, alpha = 0.5) {
  return Math.sign(u) * Math.pow(Math.abs(u), alpha);
}
```

alpha=0.5 时：
- |u|=0.01 → 0.1 (放大 10 倍)
- |u|=0.1 → 0.316 (放大 3 倍)
- |u|=0.5 → 0.707 (放大 1.4 倍)

## 实现要求

1. 在 `adRank()` 的 options 中加入：
   - `pcaDim: 0` — 0=不用 PCA, >0 = PCA 降维到 pcaDim 维
   - `tempAlpha: 1.0` — 1.0=不做温度缩放, <1.0=放大弱信号

2. 两个技术可以叠加：先 PCA 再温度缩放

## 验证

```bash
# 对比 4 种配置
node -e "
const {adRank} = require('./ad_rank');
// ... 用同样的 30 个候选测试
// 配置1: 原始 (pcaDim=0, tempAlpha=1.0)
// 配置2: 仅 PCA (pcaDim=32)
// 配置3: 仅温度 (tempAlpha=0.5)
// 配置4: PCA+温度 (pcaDim=32, tempAlpha=0.5)
// 对比 Pe 值变化
"
```

期望：Pe 从 0.065 提升到 > 0.3
