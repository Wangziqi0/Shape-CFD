# Agent 11 指令：全策略横评 + 集成到管线

> 工作量：中（~200 行实验代码 + 修改 search_engine.js）
> 依赖：Agent 7/8/9/10 全部完成后执行

## 目标

1. 对所有实现的策略做统一横评，找出全局最优方案
2. 将最优方案集成到 `search_engine.js` 的实际管线中

## Phase 1：全策略横评

### 创建文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank_final_eval.js`

### 参赛选手

```
组 A:  cosine (基线)
组 B:  AD-Rank v2 (全局对流, D=0.15 u=0.1 knn=3)
组 C:  MDA (多方向对流, Agent 7 最优参数)
组 D:  BAA (分块对流, Agent 7 最优参数)
组 E:  v2 + PCA (局部 PCA 放大, Agent 8)
组 F:  v2 + 温度缩放 (Agent 8)
组 G:  v2 + 自适应参数 (Agent 9)
组 H:  Shape CFD (形状 CFD, Agent 10)
```

### 评测维度

对 10 个标准 query：

| 维度 | 权重 | 说明 |
|:---|:---:|:---|
| DeepSeek 盲评分 | 40% | 5 分制，最终排名质量指标 |
| 对流贡献率 | 20% | B vs C/D 差异的百分比 |
| 速度 (ms) | 20% | < 20ms 为合格 |
| 收敛率 | 10% | 需 > 80% |
| Pe 值 | 10% | 对流信号强度 |

### 盲评设计

每个 query 做 **A vs X** 的成对盲评（X 遍历 B-H），不做所有组之间的两两对比（否则太多组合）。

### 输出

```json
{
  "rankings": [
    { "rank": 1, "method": "MDA", "score": 4.2, "pe": 0.3, ... },
    { "rank": 2, "method": "Shape", "score": 4.0, ... },
    ...
  ],
  "winner": "MDA",
  "winnerParams": { ... }
}
```

保存到 `word/cfd创新/experiment_final_eval.json`

## Phase 2：集成到 search_engine.js

### 修改文件

`/home/amd/HEZIMENG/legal-assistant/search_engine.js`

### 方案

参照 Agent 3 的指令（`agent指令-Agent3-适配管线.md`），但使用 Phase 1 的获胜方案：

```js
// 在 _searchInner() 中，Step 5 (HNSW) 之后插入：

if (this.config.adrank_enabled) {
  // 根据获胜方案选择对应的实现
  const method = this.config.adrank_method || 'v2'; // 'v2' | 'mda' | 'baa' | 'shape'
  
  switch (method) {
    case 'mda':
      enrichedResults = await this._adRankMDA(queryVector, rawResults);
      break;
    case 'baa':
      enrichedResults = await this._adRankBAA(queryVector, rawResults);
      break;
    case 'shape':
      enrichedResults = await this._adRankShape(queryVector, rawResults);
      break;
    default:
      enrichedResults = await this._adRankV2(queryVector, rawResults);
  }
}
```

### rag_config.json 添加

```json
{
  "adrank_enabled": false,
  "adrank_method": "v2",
  "adrank_preFilterK": 30,
  "adrank_D": 0.15,
  "adrank_uStrength": 0.1,
  "adrank_knn": 3,
  "adrank_autoD": false,
  "adrank_autoU": false
}
```

### 验证

1. `adrank_enabled: false` → 现有功能不受影响
2. `adrank_enabled: true, adrank_method: "v2"` → v2 基线
3. `adrank_enabled: true, adrank_method: "<winner>"` → 最优方案

```bash
# 基本功能测试
node -e "
const {SearchEngine} = require('./search_engine');
(async () => {
  const se = new SearchEngine();
  const r = await se.search('劳动合同解除赔偿');
  console.log('成功:', r.success, '结果数:', r.articles?.length);
})();
"
```

## 最终产出

1. `experiment_final_eval.json` — 全策略横评数据
2. `search_engine.js` — 集成最优方案
3. 更新 `word/cfd创新/CFD-RAG研究完整总结.md` — 加入 v3 实验结果

## ⚠️ 约束

- 只有在 Phase 1 找到**比 v2 更好的方案**时才做 Phase 2 集成
- 如果所有新方案都不如 v2，就只集成 v2（参照 Agent 3 指令）
- 必须保证 `adrank_enabled: false` 时零影响
