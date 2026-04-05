# Agent 4 指令：AD-Rank 性能优化

> 基于基线实验数据进行优化，解决收敛问题和 re-embed 瓶颈。

## 基线实验数据（10 个 query 平均值）

```
AD-Rank 总延迟: 1031ms
  ├ HNSW 粗筛:   1.2ms
  ├ re-embed:    1011.2ms  ← 98% 瓶颈！
  └ AD-Rank 求解: 18.8ms
收敛率: 0/10（20 轮全部未收敛）
Re: 1.093  Pe: 0.202
```

## 优化任务

### 任务 1：解决收敛问题（必做）

**问题**：20 轮迭代全部未收敛，maxDelta > epsilon。

**修改文件**：`/home/amd/HEZIMENG/legal-assistant/ad_rank.js`

**方案**：
1. 增大 maxIter（20→50）
2. 增大 epsilon（1e-6→1e-4，检索排序不需要高精度）
3. 增大 dt（0.05→0.1，加速收敛但小心数值不稳定）
4. 加入**自适应时间步长**：`dt = min(0.2, 1 / max_degree)` 防止震荡

在 `adRank()` 函数中修改默认参数并添加自适应 dt：

```js
// 自适应 dt: CFL 条件
const maxDegree = Math.max(...adjacency.map(adj => adj.length));
const cfl_dt = maxDegree > 0 ? 0.8 / maxDegree : dt;
const effective_dt = Math.min(dt, cfl_dt);
```

### 任务 2：消除 re-embed 瓶颈（必做）

**问题**：每次搜索需调 embedding API 重新 embed 30 个候选文本，耗时 ~1s。

**方案 A：向量缓存层**

修改 `ad_rank_data.js`，增加 **SQLite 向量持久化**：

```js
// 在 data_engine.js 的 schema 中新增表
CREATE TABLE IF NOT EXISTS doc_vectors (
  chunk_id INTEGER PRIMARY KEY,
  vector BLOB NOT NULL,
  model TEXT DEFAULT 'Qwen3-Embedding-8B',
  created_at TEXT DEFAULT (datetime('now'))
);
```

流程：
1. 首次查询时 re-embed + 存入 SQLite
2. 后续查询时直接从 SQLite 读取（~1ms）
3. 逐步填充，最终覆盖全部 23,701 个 chunk

```js
async getCandidateVectors(keys) {
  const cached = [];
  const needEmbed = [];
  
  for (const k of keys) {
    const row = this.db.prepare('SELECT vector FROM doc_vectors WHERE chunk_id = ?').get(k);
    if (row) {
      cached.push({ key: k, vector: new Float32Array(row.vector.buffer) });
    } else {
      needEmbed.push(k);
    }
  }
  
  if (needEmbed.length > 0) {
    // 只 re-embed 缓存未命中的
    const texts = needEmbed.map(k => this.metadata[k].content);
    const vectors = await this.engine.embedBatch(texts);
    // 写入缓存
    for (let i = 0; i < needEmbed.length; i++) {
      if (vectors[i]) {
        const buf = Buffer.from(vectors[i].buffer);
        this.db.prepare('INSERT OR REPLACE INTO doc_vectors (chunk_id, vector) VALUES (?, ?)').run(needEmbed[i], buf);
        cached.push({ key: needEmbed[i], vector: vectors[i] });
      }
    }
  }
  
  return cached;
}
```

**效果预估**：首次查询 ~1s（冷缓存），后续 <5ms（热缓存）。

**方案 B：批量预导出**（可选，一次性操作）

编写脚本一次性 embed 全部 23,701 个 chunk 到 SQLite：

```bash
node scripts/export_doc_vectors.js  # 预计 30 分钟
```

### 任务 3：参数调优（建议做）

基于实验数据调优超参数，修改 `ad_rank.js` 的默认值：

| 参数 | 当前值 | 建议值 | 原因 |
|:---|:---:|:---:|:---|
| maxIter | 20 | 50 | 收敛率 0/10 |
| epsilon | 1e-6 | 1e-4 | 排序不需要高精度 |
| dt | 0.05 | 自适应 | CFL 稳定性 |
| D | 0.15 | 0.15 | Re≈1，平衡不错 |
| uStrength | 0.3 | 0.3 | 对流有效但不过强 |
| knn | 5 | 5 | 30 个候选 k=5 合理 |

### 任务 4：SSE 流式拓扑推送（可选）

在 `server.js` 的 SSE 管线中添加 AD-Rank 拓扑数据推送：

```js
emit('algorithm_step', {
  step: 'adrank_topology',
  status: 'done',
  detail: `Re=${re.toFixed(2)} 汇聚${conv}点 发散${div}点`,
  topology: { reynolds, peclet, convergencePoints, divergencePoints },
});
```

## 验证

优化完成后重新跑基线实验：

```bash
node ad_rank_benchmark.js 2>&1
```

对比：
1. 收敛率应提升到 >80%
2. 首次查询延迟 ≈ 1s（冷缓存），后续 <50ms（热缓存）
3. AD-Rank 求解 <20ms（不变）
4. 排序质量无明显退化

## 文件清单

| 文件 | 操作 |
|:---|:---|
| `ad_rank.js` | 修改默认参数、加自适应 dt |
| `ad_rank_data.js` | 添加 SQLite 向量缓存 |
| `data_engine.js` | 新增 `doc_vectors` 表 |
| `ad_rank_benchmark.js` | 重新跑对比实验 |
