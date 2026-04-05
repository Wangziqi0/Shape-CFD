# Agent 2 指令：AD-Rank 数据接口层

> 你是负责实现 AD-Rank 数据接口层的 agent。
> 为 AD-Rank 核心求解器提供输入数据（query 向量 + 候选文档向量）。

## 要创建的文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank_data.js`

## 项目现状

```
向量索引: knowledge_base/vectors/law.usearch (23,701 条, USearch HNSW, cos, 4096维)
元数据:   knowledge_base/vectors/metadata.json (JSON 数组, 每项含 law/path/article/content)
Embedding API: http://192.168.31.22:3000/v1/embeddings
模型: Qwen3-Embedding-8B (4096维)
```

## 接口定义

```js
class ADRankData {
  /**
   * 初始化：加载 USearch 索引和元数据
   */
  async initialize() { ... }

  /**
   * 将用户查询文本转为向量
   * @param {string} queryText - 用户查询
   * @returns {Promise<Float32Array>} 4096维向量
   */
  async embedQuery(queryText) { ... }

  /**
   * HNSW 粗筛 + 获取候选文档向量
   * 
   * 流程：
   *   1. HNSW 搜索 top-preFilterK 个候选
   *   2. 从 metadata 获取候选文档的文本内容
   *   3. 重新调用 embedding API 获取这些文档的 4096 维向量
   *   4. 返回向量和元数据
   * 
   * @param {Float32Array} queryVector - query 向量
   * @param {number} [preFilterK=30] - 粗筛数量
   * @returns {Promise<{vectors: Float32Array[], metadata: Object[], distances: number[]}>}
   */
  async getCandidates(queryVector, preFilterK = 30) { ... }
}

module.exports = { ADRankData };
```

## 实现细节

### 1. 初始化

复用现有的 usearch 和 metadata 加载逻辑：

```js
const usearch = require('usearch');
const fs = require('fs');
const path = require('path');

const VECTORS_DIR = path.join(__dirname, 'knowledge_base', 'vectors');
const INDEX_FILE = path.join(VECTORS_DIR, 'law.usearch');
const META_FILE = path.join(VECTORS_DIR, 'metadata.json');

async initialize() {
  this.index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 4096 });
  this.index.load(INDEX_FILE);
  this.metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
  console.log(`[ADRankData] 加载完成: ${this.index.size()} 条向量`);
}
```

### 2. Query Embedding

复用 vectorize_engine.js 的逻辑：

```js
const { VectorizeEngine, loadApiKey } = require('./vectorize_engine');

async embedQuery(queryText) {
  if (!this.engine) {
    this.engine = new VectorizeEngine(loadApiKey());
  }
  const vectors = await this.engine.embed(queryText);
  return vectors;  // Float32Array(4096)
}
```

### 3. 获取候选

关键：USearch 不支持 get() 取回原始向量，所以需要重新 embed：

```js
async getCandidates(queryVector, preFilterK = 30) {
  // 1. HNSW 粗筛
  const results = this.index.search(queryVector, preFilterK);
  const keys = Array.from(results.keys).map(Number);
  const distances = Array.from(results.distances);

  // 2. 获取元数据和文本
  const metas = keys.map(k => this.metadata[k]).filter(Boolean);

  // 3. 批量重新 embed 候选文本
  const texts = metas.map(m => m.content);
  const vectors = await this.engine.embedAll(texts);

  return {
    vectors: vectors,          // Float32Array[](preFilterK)
    metadata: metas,           // [{law, path, article, content}]
    distances: distances,      // number[]
  };
}
```

### 4. embedAll 批量调用

如果 vectorize_engine.js 的 embedAll 不适合，自己实现批量：

```js
async _batchEmbed(texts, batchSize = 8) {
  const results = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const response = await fetch('http://192.168.31.22:3000/v1/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'Qwen3-Embedding-8B',
        input: batch,
      }),
    });
    const json = await response.json();
    for (const item of json.data) {
      results.push(new Float32Array(item.embedding));
    }
  }
  return results;
}
```

## 测试

```js
if (require.main === module) {
  (async () => {
    const data = new ADRankData();
    await data.initialize();
    
    const qVec = await data.embedQuery("劳动合同解除赔偿标准");
    console.log('Query 向量维度:', qVec.length);
    
    const { vectors, metadata, distances } = await data.getCandidates(qVec, 10);
    console.log(`候选: ${vectors.length} 个文档`);
    metadata.forEach((m, i) =>
      console.log(`  [${i}] ${m.law} ${m.article} (dist=${distances[i].toFixed(4)})`)
    );
    
    // 调用 AD-Rank
    const { adRank } = require('./ad_rank');
    const result = adRank(qVec, vectors, 5);
    console.log('\nAD-Rank Top-5:');
    result.rankings.forEach((r, i) =>
      console.log(`  #${i+1} ${metadata[r.index].law} ${metadata[r.index].article} score=${r.score.toFixed(4)} ${r.topology}`)
    );
  })();
}
```

## 依赖

- `usearch` (已安装)
- `vectorize_engine.js` (已存在)
- `ad_rank.js` (Agent 1 的产出)
