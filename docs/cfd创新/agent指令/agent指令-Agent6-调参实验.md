# Agent 6 指令：AD-Rank 调参实验与三组对比

> 对 AD-Rank 进行参数网格搜索，找到最优超参数组合，
> 然后与原方案做三组对比实验 (A/B/C)，生成实验报告。

## 背景

当前 AD-Rank v2 (极致优化版) 基线：
- 求解 ~10ms, 缓存命中后总延迟 ~13.6ms
- 默认参数: D=0.15, uStrength=0.3, dt=0.1, knn=5, preFilterK=30, maxIter=50, epsilon=1e-3
- 收敛率 10/10, Re≈1.1, Pe≈0.2, Top-10 重叠率 53%

## 要创建的文件

`/home/amd/HEZIMENG/legal-assistant/ad_rank_param_sweep.js`

## 实验设计

### Phase 1：参数网格搜索

测试以下参数组合（4×3×3×2 = 72 组，每组跑 5 个 query）：

```js
const PARAM_GRID = {
  D:         [0.05, 0.10, 0.15, 0.25],    // 扩散系数
  uStrength: [0.1, 0.3, 0.5],              // 对流强度
  knn:       [3, 5, 7],                     // KNN 图 k 值
  preFilterK:[20, 30],                      // HNSW 候选数
};

const SWEEP_QUERIES = [
  '劳动合同解除赔偿标准',
  '醉驾量刑标准',
  '房屋租赁合同违约金',
  '工伤认定标准和赔偿',
  '离婚财产分割',
];
```

对每组参数记录：
- 求解时间 (ms)
- 收敛率 (0-5)
- 迭代数 (avg)
- Re, Pe
- 与基线 cosine 的 Top-10 重叠率
- 排序变化量（Kendall tau 距离）

### Phase 2：最优参数三组对比

用网格搜索找到的最优参数，跑完整 10 个 query 的三组实验：

```
组 A (基线):     纯 cosine HNSW 排序
组 B (纯扩散):   AD-Rank uStrength=0（只有扩散项，无对流）
组 C (AD-Rank):  AD-Rank 最优参数（对流+扩散）
```

**组 B 的目的**：证明对流项的价值。如果 B 和 C 差不多，说明对流没用；如果 C 显著优于 B，证明对流是关键创新。

## 代码框架

```js
#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const usearch = require('usearch');
const { adRank } = require('./ad_rank');
const { ADRankData } = require('./ad_rank_data');
const { VectorizeEngine, loadApiKey } = require('./vectorize_engine');

const VECTORS_DIR = path.join(__dirname, 'knowledge_base', 'vectors');
const INDEX_FILE = path.join(VECTORS_DIR, 'law.usearch');
const META_FILE = path.join(VECTORS_DIR, 'metadata.json');

// ========== 参数网格 ==========
const PARAM_GRID = {
  D:          [0.05, 0.10, 0.15, 0.25],
  uStrength:  [0.1,  0.3,  0.5],
  knn:        [3, 5, 7],
  preFilterK: [20, 30],
};

const SWEEP_QUERIES = [
  '劳动合同解除赔偿标准',
  '醉驾量刑标准',
  '房屋租赁合同违约金',
  '工伤认定标准和赔偿',
  '离婚财产分割',
];

const FULL_QUERIES = [
  ...SWEEP_QUERIES,
  '借款合同利息上限',
  '交通事故责任划分',
  '商标侵权赔偿',
  '行政处罚听证程序',
  '未成年人犯罪处罚',
];

(async () => {
  // 初始化
  const index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 4096 });
  index.load(INDEX_FILE);
  const metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
  const idMap = new Map();
  for (let i = 0; i < metadata.length; i++) {
    if (metadata[i].id !== undefined) idMap.set(metadata[i].id, i);
  }
  const engine = new VectorizeEngine(loadApiKey());
  const adData = new ADRankData();
  await adData.initialize();

  // ========== Phase 1: 参数网格搜索 ==========
  console.log('═'.repeat(70));
  console.log('  Phase 1: 参数网格搜索');
  console.log('═'.repeat(70));

  const sweepResults = [];

  // 预先 embed 所有 query
  const queryVectors = {};
  for (const q of SWEEP_QUERIES) {
    queryVectors[q] = await engine.embed(q);
    if (!(queryVectors[q] instanceof Float32Array))
      queryVectors[q] = new Float32Array(queryVectors[q]);
  }

  // 遍历参数组合
  const Ds = PARAM_GRID.D;
  const Us = PARAM_GRID.uStrength;
  const Ks = PARAM_GRID.knn;
  const Ps = PARAM_GRID.preFilterK;
  let total = Ds.length * Us.length * Ks.length * Ps.length;
  let idx = 0;

  for (const D of Ds) {
    for (const uStr of Us) {
      for (const knn of Ks) {
        for (const pfK of Ps) {
          idx++;
          process.stdout.write(`\r  [${idx}/${total}] D=${D} u=${uStr} knn=${knn} pf=${pfK}    `);

          let totalTime = 0, converged = 0, totalIter = 0;
          let totalRe = 0, totalPe = 0, totalOverlap = 0;

          for (const q of SWEEP_QUERIES) {
            const qVec = queryVectors[q];

            // 基线 cosine
            const baseResults = index.search(qVec, 10);
            const baseKeys = Array.from(baseResults.keys).map(Number);
            const baseLaws = baseKeys.map(k => {
              const ai = idMap.get(k);
              const m = ai !== undefined ? metadata[ai] : null;
              return m ? `${m.law}|${m.article}` : '';
            });

            // AD-Rank
            const { vectors, metadata: metas } = await adData.getCandidates(qVec, pfK);
            if (vectors.length < 3) continue;

            const t0 = performance.now();
            const result = adRank(qVec, vectors, 10, {
              D, uStrength: uStr, knn, maxIter: 50, epsilon: 1e-3, dt: 0.1,
            });
            totalTime += performance.now() - t0;

            if (result.convergence) converged++;
            totalIter += result.iterations;
            totalRe += result.reynolds;
            totalPe += result.peclet;

            // Top-10 重叠率
            const adLaws = result.rankings.map(r => {
              const m = metas[r.index];
              return m ? `${m.law}|${m.article}` : '';
            });
            const overlap = baseLaws.filter(x => adLaws.includes(x)).length;
            totalOverlap += overlap / 10 * 100;
          }

          const n = SWEEP_QUERIES.length;
          sweepResults.push({
            D, uStrength: uStr, knn, preFilterK: pfK,
            avgTimeMs: +(totalTime / n).toFixed(1),
            convergenceRate: converged + '/' + n,
            avgIter: +(totalIter / n).toFixed(1),
            avgRe: +(totalRe / n).toFixed(3),
            avgPe: +(totalPe / n).toFixed(3),
            avgOverlap: +(totalOverlap / n).toFixed(0),
          });

          await new Promise(r => setTimeout(r, 50));
        }
      }
    }
  }

  // 排序：按收敛率 desc → 重叠率变化大（排序改变多=对流价值高）→ 速度 asc
  sweepResults.sort((a, b) => {
    const convA = parseInt(a.convergenceRate);
    const convB = parseInt(b.convergenceRate);
    if (convA !== convB) return convB - convA;
    // 寻找重叠率在 40-60% 的（排序有变化但不过激）
    const midA = Math.abs(a.avgOverlap - 50);
    const midB = Math.abs(b.avgOverlap - 50);
    if (midA !== midB) return midA - midB;
    return a.avgTimeMs - b.avgTimeMs;
  });

  console.log('\n\n  Top 10 参数组合:');
  console.log('  ' + '─'.repeat(90));
  console.log('  D    | u    | knn | pf | time  | conv | iter | Re    | Pe    | overlap%');
  console.log('  ' + '─'.repeat(90));
  for (const r of sweepResults.slice(0, 10)) {
    console.log(`  ${r.D.toFixed(2)} | ${r.uStrength.toFixed(1)} | ${r.knn}   | ${r.preFilterK} | ${r.avgTimeMs.toString().padStart(5)}ms | ${r.convergenceRate}  | ${r.avgIter.toString().padStart(4)} | ${r.avgRe} | ${r.avgPe} | ${r.avgOverlap}%`);
  }

  // 最优参数
  const best = sweepResults[0];
  console.log(`\n  🏆 最优参数: D=${best.D} u=${best.uStrength} knn=${best.knn} pf=${best.preFilterK}`);

  // ========== Phase 2: 三组对比 (A/B/C) ==========
  console.log('\n\n' + '═'.repeat(70));
  console.log('  Phase 2: 三组对比 A(cosine) / B(纯扩散) / C(AD-Rank最优)');
  console.log('═'.repeat(70));

  const comparison = [];

  for (const q of FULL_QUERIES) {
    let qVec = await engine.embed(q);
    if (!(qVec instanceof Float32Array)) qVec = new Float32Array(qVec);

    // 组 A: cosine
    const aResults = index.search(qVec, 10);
    const aKeys = Array.from(aResults.keys).map(Number);
    const aLaws = aKeys.map(k => {
      const ai = idMap.get(k);
      const m = ai !== undefined ? metadata[ai] : null;
      return m ? `${m.law}|${m.article}` : '';
    });

    // 获取候选
    const { vectors, metadata: metas } = await adData.getCandidates(qVec, best.preFilterK);
    if (vectors.length < 3) continue;

    // 组 B: 纯扩散 (uStrength=0)
    const bResult = adRank(qVec, vectors, 10, {
      D: best.D, uStrength: 0, knn: best.knn, maxIter: 50, epsilon: 1e-3,
    });
    const bLaws = bResult.rankings.map(r => metas[r.index] ? `${metas[r.index].law}|${metas[r.index].article}` : '');

    // 组 C: AD-Rank 最优
    const cResult = adRank(qVec, vectors, 10, {
      D: best.D, uStrength: best.uStrength, knn: best.knn, maxIter: 50, epsilon: 1e-3,
    });
    const cLaws = cResult.rankings.map(r => metas[r.index] ? `${metas[r.index].law}|${metas[r.index].article}` : '');

    // 重叠率
    const abOverlap = aLaws.filter(x => bLaws.includes(x)).length / 10 * 100;
    const acOverlap = aLaws.filter(x => cLaws.includes(x)).length / 10 * 100;
    const bcOverlap = bLaws.filter(x => cLaws.includes(x)).length / 10 * 100;

    comparison.push({
      query: q,
      a_vs_b: abOverlap,  // cosine vs 纯扩散
      a_vs_c: acOverlap,  // cosine vs AD-Rank
      b_vs_c: bcOverlap,  // 纯扩散 vs AD-Rank (差值=对流的贡献)
      c_re: cResult.reynolds,
      c_conv: cResult.convergence,
    });

    console.log(`\n  "${q}"`);
    console.log(`    A vs B(纯扩散): ${abOverlap}%  |  A vs C(AD-Rank): ${acOverlap}%  |  B vs C: ${bcOverlap}%`);
    console.log(`    → 对流贡献: ${(100 - bcOverlap).toFixed(0)}% 排序差异来自对流项`);

    await new Promise(r => setTimeout(r, 200));
  }

  // ========== 汇总 ==========
  console.log('\n\n' + '═'.repeat(70));
  console.log('  实验汇总');
  console.log('═'.repeat(70));

  const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

  console.log(`\n  最优参数: D=${best.D} u=${best.uStrength} knn=${best.knn} pf=${best.preFilterK}`);
  console.log(`\n  三组对比 (${comparison.length} 个 query 平均):`);
  console.log(`    A vs B(纯扩散) 重叠率: ${avg(comparison.map(c => c.a_vs_b)).toFixed(0)}%`);
  console.log(`    A vs C(AD-Rank) 重叠率: ${avg(comparison.map(c => c.a_vs_c)).toFixed(0)}%`);
  console.log(`    B vs C 重叠率: ${avg(comparison.map(c => c.b_vs_c)).toFixed(0)}%`);
  console.log(`    → 对流贡献: ${(100 - avg(comparison.map(c => c.b_vs_c))).toFixed(0)}% 排序差异归因于对流项`);

  // 保存完整结果
  const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_param_sweep.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    bestParams: best,
    sweepTop10: sweepResults.slice(0, 10),
    comparison,
    summary: {
      bestD: best.D,
      bestU: best.uStrength,
      bestKnn: best.knn,
      bestPf: best.preFilterK,
      avgAvsB: +avg(comparison.map(c => c.a_vs_b)).toFixed(0),
      avgAvsC: +avg(comparison.map(c => c.a_vs_c)).toFixed(0),
      avgBvsC: +avg(comparison.map(c => c.b_vs_c)).toFixed(0),
      advectionContribution: +(100 - avg(comparison.map(c => c.b_vs_c))).toFixed(0),
    },
  }, null, 2));
  console.log(`\n  📁 结果已保存: ${reportPath}`);
  console.log('\n✅ 调参实验完成');
})();
```

## 验证标准

1. 网格搜索 72 组全部跑完
2. 找到收敛率 5/5 且排序有意义变化的最优参数
3. 三组对比证明 **B vs C 重叠率 < 90%**（即对流项至少改变了 10% 的排序）
4. 实验结果保存到 `experiment_param_sweep.json`

## 依赖

- `ad_rank.js` (已优化)
- `ad_rank_data.js` (已有缓存层)
- `vectorize_engine.js`
- usearch
