#!/usr/bin/env node
'use strict';

/**
 * AD-Rank 调参实验 — Agent 6
 *
 * Phase 1: 参数网格搜索 (4×3×3×2 = 72 组, 每组 5 query)
 * Phase 2: 三组对比 A(cosine) / B(纯扩散) / C(AD-Rank最优), 10 query
 *
 * 输出: word/cfd创新/experiment_param_sweep.json
 */

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
  try {
    // ========== 初始化 ==========
    console.log('═'.repeat(70));
    console.log('  AD-Rank 调参实验 — Agent 6');
    console.log('═'.repeat(70));

    // USearch 索引
    const index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 4096 });
    index.load(INDEX_FILE);
    const metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
    const idMap = new Map();
    for (let i = 0; i < metadata.length; i++) {
      if (metadata[i].id !== undefined) idMap.set(metadata[i].id, i);
    }

    // Embedding 引擎
    const apiKey = loadApiKey();
    if (!apiKey) {
      console.error('❌ 未找到 API Key');
      process.exit(1);
    }
    const engine = new VectorizeEngine(apiKey);

    // ADRankData (含 SQLite 缓存层)
    const adData = new ADRankData();
    await adData.initialize();

    console.log(`  索引: ${index.size()} 条向量, 元数据: ${metadata.length} 条`);

    // ========== Phase 1: 参数网格搜索 ==========
    console.log('\n' + '═'.repeat(70));
    console.log('  Phase 1: 参数网格搜索 (72 组 × 5 query)');
    console.log('═'.repeat(70));

    const sweepResults = [];

    // 预先 embed 所有 sweep query
    console.log('\n  📡 Pre-embedding sweep queries...');
    const queryVectors = {};
    for (const q of SWEEP_QUERIES) {
      queryVectors[q] = await engine.embed(q);
      if (!(queryVectors[q] instanceof Float32Array))
        queryVectors[q] = new Float32Array(queryVectors[q]);
      process.stdout.write('.');
    }
    console.log(' done\n');

    // 预先获取每个 query 的基线 cosine 排序和候选集
    // 用最大 preFilterK=30 获取候选，较小的 preFilterK=20 取子集即可
    const maxPfK = Math.max(...PARAM_GRID.preFilterK);
    const sweepCandidates = {};
    const sweepBaselines = {};

    for (const q of SWEEP_QUERIES) {
      const qVec = queryVectors[q];

      // 基线 cosine 排序
      const baseResults = index.search(qVec, 10);
      const baseKeys = Array.from(baseResults.keys).map(Number);
      sweepBaselines[q] = baseKeys.map(k => {
        const ai = idMap.get(k);
        const m = ai !== undefined ? metadata[ai] : null;
        return m ? `${m.law}|${m.article}` : '';
      });

      // 候选集（使用最大 preFilterK）
      const { vectors, metadata: metas, cacheStats } = await adData.getCandidates(qVec, maxPfK);
      sweepCandidates[q] = { vectors, metas, cacheStats };
      console.log(`  \"${q}\" → ${vectors.length} 候选 (缓存 hit=${cacheStats.hit} miss=${cacheStats.miss})`);
    }

    // 遍历参数组合
    const Ds = PARAM_GRID.D;
    const Us = PARAM_GRID.uStrength;
    const Ks = PARAM_GRID.knn;
    const Ps = PARAM_GRID.preFilterK;
    const total = Ds.length * Us.length * Ks.length * Ps.length;
    let idx = 0;

    console.log(`\n  开始搜索 ${total} 组参数...\n`);

    for (const D of Ds) {
      for (const uStr of Us) {
        for (const knn of Ks) {
          for (const pfK of Ps) {
            idx++;
            process.stdout.write(`\r  [${idx}/${total}] D=${D} u=${uStr} knn=${knn} pf=${pfK}    `);

            let totalTime = 0, converged = 0, totalIter = 0;
            let totalRe = 0, totalPe = 0, totalOverlap = 0;
            let validCount = 0;

            for (const q of SWEEP_QUERIES) {
              const qVec = queryVectors[q];
              const baseLaws = sweepBaselines[q];
              const { vectors, metas } = sweepCandidates[q];

              // 根据 preFilterK 截取候选子集
              const useVectors = pfK < vectors.length ? vectors.slice(0, pfK) : vectors;
              const useMetas = pfK < metas.length ? metas.slice(0, pfK) : metas;

              if (useVectors.length < 3) continue;

              const t0 = performance.now();
              const result = adRank(qVec, useVectors, 10, {
                D, uStrength: uStr, knn, maxIter: 50, epsilon: 1e-3, dt: 0.1,
              });
              totalTime += performance.now() - t0;

              if (result.convergence) converged++;
              totalIter += result.iterations;
              totalRe += result.reynolds;
              totalPe += result.peclet;

              // Top-10 重叠率
              const adLaws = result.rankings.map(r => {
                const m = useMetas[r.index];
                return m ? `${m.law}|${m.article}` : '';
              });
              const overlap = baseLaws.filter(x => adLaws.includes(x)).length;
              totalOverlap += overlap / 10 * 100;
              validCount++;
            }

            if (validCount === 0) continue;

            sweepResults.push({
              D, uStrength: uStr, knn, preFilterK: pfK,
              avgTimeMs: +(totalTime / validCount).toFixed(1),
              convergenceRate: converged + '/' + validCount,
              avgIter: +(totalIter / validCount).toFixed(1),
              avgRe: +(totalRe / validCount).toFixed(3),
              avgPe: +(totalPe / validCount).toFixed(3),
              avgOverlap: +(totalOverlap / validCount).toFixed(0),
            });
          }
        }
      }
    }

    // 排序：按收敛率 desc → 重叠率在 40-60% 的（排序有变化但不过激）→ 速度 asc
    sweepResults.sort((a, b) => {
      const convA = parseInt(a.convergenceRate);
      const convB = parseInt(b.convergenceRate);
      if (convA !== convB) return convB - convA;
      // 寻找重叠率接近 50% 的（排序有变化但不过激）
      const midA = Math.abs(a.avgOverlap - 50);
      const midB = Math.abs(b.avgOverlap - 50);
      if (midA !== midB) return midA - midB;
      return a.avgTimeMs - b.avgTimeMs;
    });

    console.log('\n\n  Top 10 参数组合:');
    console.log('  ' + '─'.repeat(95));
    console.log('  D    | u    | knn | pf | time    | conv | iter | Re      | Pe      | overlap%');
    console.log('  ' + '─'.repeat(95));
    for (const r of sweepResults.slice(0, 10)) {
      console.log(
        `  ${r.D.toFixed(2)} | ${r.uStrength.toFixed(1)} | ${String(r.knn).padStart(3)} | ${String(r.preFilterK).padStart(2)} | ` +
        `${String(r.avgTimeMs).padStart(6)}ms | ${r.convergenceRate.padEnd(4)} | ${String(r.avgIter).padStart(4)} | ` +
        `${r.avgRe.toFixed(3).padStart(7)} | ${r.avgPe.toFixed(3).padStart(7)} | ${r.avgOverlap}%`
      );
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
      let qVec;
      // 复用已有的 embedding
      if (queryVectors[q]) {
        qVec = queryVectors[q];
      } else {
        qVec = await engine.embed(q);
        if (!(qVec instanceof Float32Array)) qVec = new Float32Array(qVec);
      }

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
      if (vectors.length < 3) {
        console.log(`\n  ⚠️  \"${q}\" 候选不足 (${vectors.length}), 跳过`);
        continue;
      }

      // 组 B: 纯扩散 (uStrength=0)
      const t_b = performance.now();
      const bResult = adRank(qVec, vectors, 10, {
        D: best.D, uStrength: 0, knn: best.knn, maxIter: 50, epsilon: 1e-3, dt: 0.1,
      });
      const t_b_ms = performance.now() - t_b;
      const bLaws = bResult.rankings.map(r => metas[r.index] ? `${metas[r.index].law}|${metas[r.index].article}` : '');

      // 组 C: AD-Rank 最优
      const t_c = performance.now();
      const cResult = adRank(qVec, vectors, 10, {
        D: best.D, uStrength: best.uStrength, knn: best.knn, maxIter: 50, epsilon: 1e-3, dt: 0.1,
      });
      const t_c_ms = performance.now() - t_c;
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
        c_pe: cResult.peclet,
        c_conv: cResult.convergence,
        c_iter: cResult.iterations,
        b_time_ms: +t_b_ms.toFixed(1),
        c_time_ms: +t_c_ms.toFixed(1),
      });

      console.log(`\n  "${q}"`);
      console.log(`    A vs B(纯扩散): ${abOverlap}%  |  A vs C(AD-Rank): ${acOverlap}%  |  B vs C: ${bcOverlap}%`);
      console.log(`    → 对流贡献: ${(100 - bcOverlap).toFixed(0)}% 排序差异来自对流项`);
      console.log(`    Re=${cResult.reynolds.toFixed(3)} Pe=${cResult.peclet.toFixed(3)} conv=${cResult.convergence} iter=${cResult.iterations}`);
      console.log(`    solve: B=${t_b_ms.toFixed(1)}ms C=${t_c_ms.toFixed(1)}ms`);

      await new Promise(r => setTimeout(r, 200));
    }

    // ========== 汇总 ==========
    console.log('\n\n' + '═'.repeat(70));
    console.log('  实验汇总');
    console.log('═'.repeat(70));

    const avg = arr => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    const avgAvsB = avg(comparison.map(c => c.a_vs_b));
    const avgAvsC = avg(comparison.map(c => c.a_vs_c));
    const avgBvsC = avg(comparison.map(c => c.b_vs_c));
    const advectionContrib = 100 - avgBvsC;

    console.log(`\n  最优参数: D=${best.D} u=${best.uStrength} knn=${best.knn} pf=${best.preFilterK}`);
    console.log(`\n  三组对比 (${comparison.length} 个 query 平均):`);
    console.log(`    A vs B(纯扩散) 重叠率: ${avgAvsB.toFixed(0)}%`);
    console.log(`    A vs C(AD-Rank) 重叠率: ${avgAvsC.toFixed(0)}%`);
    console.log(`    B vs C 重叠率: ${avgBvsC.toFixed(0)}%`);
    console.log(`    → 对流贡献: ${advectionContrib.toFixed(0)}% 排序差异归因于对流项`);

    console.log(`\n  物理指标:`);
    console.log(`    平均 Re: ${avg(comparison.map(c => c.c_re)).toFixed(3)}`);
    console.log(`    平均 Pe: ${avg(comparison.map(c => c.c_pe)).toFixed(3)}`);
    console.log(`    收敛率: ${comparison.filter(c => c.c_conv).length}/${comparison.length}`);
    console.log(`    平均迭代: ${avg(comparison.map(c => c.c_iter)).toFixed(1)}`);

    console.log(`\n  求解速度:`);
    console.log(`    组 B(纯扩散) 平均: ${avg(comparison.map(c => c.b_time_ms)).toFixed(1)}ms`);
    console.log(`    组 C(AD-Rank) 平均: ${avg(comparison.map(c => c.c_time_ms)).toFixed(1)}ms`);

    // ========== 验证标准检查 ==========
    console.log('\n  ✅ 验证标准:');
    console.log(`    [${sweepResults.length >= 72 ? '✓' : '✗'}] 网格搜索 ${sweepResults.length}/72 组全部跑完`);
    const bestConv = parseInt(best.convergenceRate);
    console.log(`    [${bestConv === SWEEP_QUERIES.length ? '✓' : '✗'}] 最优参数收敛率: ${best.convergenceRate}`);
    console.log(`    [${avgBvsC < 90 ? '✓' : '✗'}] B vs C 重叠率 ${avgBvsC.toFixed(0)}% < 90% (对流项改变了 ${advectionContrib.toFixed(0)}% 排序)`);

    // 保存完整结果
    const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_param_sweep.json');
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      bestParams: best,
      sweepTop10: sweepResults.slice(0, 10),
      sweepAll: sweepResults,
      comparison,
      summary: {
        bestD: best.D,
        bestU: best.uStrength,
        bestKnn: best.knn,
        bestPf: best.preFilterK,
        avgAvsB: +avgAvsB.toFixed(0),
        avgAvsC: +avgAvsC.toFixed(0),
        avgBvsC: +avgBvsC.toFixed(0),
        advectionContribution: +advectionContrib.toFixed(0),
        avgReynolds: +avg(comparison.map(c => c.c_re)).toFixed(3),
        avgPeclet: +avg(comparison.map(c => c.c_pe)).toFixed(3),
        convergenceRate: `${comparison.filter(c => c.c_conv).length}/${comparison.length}`,
      },
    }, null, 2));
    console.log(`\n  📁 结果已保存: ${reportPath}`);
    console.log('\n✅ 调参实验完成');

  } catch (err) {
    console.error('\n❌ 实验失败:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
})();
