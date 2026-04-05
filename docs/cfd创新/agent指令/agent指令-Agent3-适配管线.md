# Agent 3 指令：AD-Rank 适配到 search_engine.js

> 将 AD-Rank v1 集成到现有 RAG 搜索管线中，作为可选的精排层。

## 背景

AD-Rank 核心求解器（`ad_rank.js`）和数据接口层（`ad_rank_data.js`）已完成。
基线实验结果：
- AD-Rank 求解器本身仅 18.8ms，但 re-embed 候选文档需要 ~1000ms
- Top-10 排序重叠率 51%（对流项确实在改变排序）
- 0/10 收敛（20 轮迭代不够）

## 要修改的文件

`/home/amd/HEZIMENG/legal-assistant/search_engine.js` (2256行)

## 修改方案

### 1. 在 constructor 中添加 AD-Rank 配置

在 `this.config` 中添加（约 L62-69 区域）：

```js
// AD-Rank 对流-扩散精排
adrank_enabled: false,           // 默认关闭，通过 rag_config.json 开启
adrank_preFilterK: 30,           // HNSW 粗筛候选数
adrank_D: 0.15,                  // 扩散系数
adrank_uStrength: 0.3,           // 对流强度
adrank_maxIter: 50,              // 最大迭代数 (基线实验 20 不够收敛)
adrank_dt: 0.05,                 // 时间步长
adrank_knn: 5,                   // KNN 图的 k 值
```

### 2. 在 `_searchInner()` 方法中插入 AD-Rank 精排

位置：在 Step 5 (HNSW 向量搜索) 之后、Step 6 (LAP 法条关联金字塔) 之前。

```js
// Step 5.5 AD-Rank: 对流-扩散精排 (可选)
if (this.config.adrank_enabled) {
  enrichedResults = await this._adRankRerank(queryVector, rawResults);
}
```

### 3. 实现 `_adRankRerank()` 方法

添加新方法（大约 50 行）：

```js
/**
 * AD-Rank 对流-扩散精排
 * 
 * 在 HNSW 粗筛结果上运行对流-扩散方程，利用 query 方向性重新排序。
 * 注意：需要重新 embed 候选文本以获取原始向量（因为 USearch 不支持 get()）。
 * 
 * @param {Float32Array} queryVector - query 向量
 * @param {Object} rawResults - HNSW 搜索结果 {keys, distances}
 * @returns {Object} 重新排序后的结果（与 rawResults 格式兼容）
 */
async _adRankRerank(queryVector, rawResults) {
  const { adRank } = require('./ad_rank');
  const keys = Array.from(rawResults.keys).map(Number);
  const distances = Array.from(rawResults.distances);

  // 1. 取 top preFilterK 候选的文本
  const candidates = [];
  for (let i = 0; i < Math.min(keys.length, this.config.adrank_preFilterK); i++) {
    const meta = this.metadata[keys[i]];
    if (meta && meta.content) {
      candidates.push({ key: keys[i], meta, dist: distances[i] });
    }
  }

  if (candidates.length < 3) return rawResults; // 候选太少，跳过

  // 2. 重新 embed 获取原始向量 (瓶颈! ~1s)
  const texts = candidates.map(c => c.meta.content);
  let vectors;
  try {
    vectors = await this.engine.embedBatch(texts);
    vectors = vectors
      .map(v => v ? (v instanceof Float32Array ? v : new Float32Array(v)) : null)
      .filter(Boolean);
  } catch (e) {
    console.warn('[AD-Rank] re-embed 失败，跳过精排:', e.message);
    return rawResults;
  }

  if (vectors.length < 3) return rawResults;

  // 3. 运行 AD-Rank
  const adResult = adRank(queryVector, vectors, keys.length, {
    D: this.config.adrank_D,
    uStrength: this.config.adrank_uStrength,
    maxIter: this.config.adrank_maxIter,
    dt: this.config.adrank_dt,
    knn: this.config.adrank_knn,
  });

  console.log(`  🌊 AD-Rank: Re=${adResult.reynolds.toFixed(3)} Pe=${adResult.peclet.toFixed(3)} iter=${adResult.iterations} conv=${adResult.convergence}`);
  console.log(`     汇聚${adResult.convergencePoints.length} 发散${adResult.divergencePoints.length} 停滞${adResult.stagnationPoints.length}`);

  // 4. 用 AD-Rank 排序替代原始排序
  // 构造与 rawResults 格式兼容的输出
  const newKeys = new BigUint64Array(adResult.rankings.length);
  const newDists = new Float32Array(adResult.rankings.length);
  for (let i = 0; i < adResult.rankings.length; i++) {
    const r = adResult.rankings[i];
    newKeys[i] = BigInt(candidates[r.index].key);
    newDists[i] = 1 - r.score; // score → distance
  }

  return {
    keys: newKeys,
    distances: newDists,
    count: () => newKeys.length,
    size: () => newKeys.length,
    _adrank: adResult, // 附带 AD-Rank 元信息
  };
}
```

### 4. 在 rag_config.json 中添加开关

```json
{
  "adrank_enabled": false,
  "adrank_preFilterK": 30,
  "adrank_D": 0.15,
  "adrank_uStrength": 0.3,
  "adrank_maxIter": 50,
  "adrank_dt": 0.05,
  "adrank_knn": 5
}
```

### 5. 同样适配 `searchMultiVector()` 方法

在 `searchMultiVector()` 中（约 L284 区域），在合并结果后，同样加入 AD-Rank 精排逻辑。

## 注意事项

1. **不要破坏现有功能** — AD-Rank 默认关闭 (`adrank_enabled: false`)
2. **兼容 rawResults 格式** — 返回值必须有 `keys`, `distances`，与 USearch 格式兼容
3. **错误容忍** — re-embed 或 AD-Rank 失败时 fallback 到原始结果
4. **日志** — 用 `🌊 AD-Rank:` 前缀打印物理指标
5. **性能** — 瓶颈是 re-embed (~1s)，AD-Rank 本身只要 ~19ms

## 验证

适配完成后运行现有测试确保不 break：

```bash
cd /home/amd/HEZIMENG/legal-assistant
# 启动后端测试基本搜索（adrank_enabled = false）
node -e "
const {SearchEngine} = require('./search_engine');
(async () => {
  const se = new SearchEngine();
  const r = await se.search('劳动合同解除赔偿');
  console.log('成功:', r.success, '结果数:', r.articles?.length);
})();
"
```

然后在 rag_config.json 中设 `"adrank_enabled": true`，重新搜索对比结果。

## 依赖

- `ad_rank.js` (已存在，Agent 1 产出)
- `search_engine.js` (要修改)
- `rag_config.json` (要添加配置项)
