#!/usr/bin/env node
'use strict';

/**
 * AD-Rank Agent8 信号放大验证 — 4 组配置对比
 *
 * 配置1: 原始 (pcaDim=0, tempAlpha=1.0) — 基线
 * 配置2: 仅 PCA (pcaDim=32)
 * 配置3: 仅温度 (tempAlpha=0.5)
 * 配置4: PCA+温度 (pcaDim=32, tempAlpha=0.5)
 *
 * 对每组测试 5 个真实 query，记录 Pe、Re、排序变化和耗时
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

const QUERIES = [
  '劳动合同解除赔偿标准',
  '醉驾量刑标准',
  '房屋租赁合同违约金',
  '工伤认定标准和赔偿',
  '离婚财产分割',
];

const CONFIGS = [
  { name: '基线 (原始)',       pcaDim: 0,  tempAlpha: 1.0 },
  { name: '仅 PCA (dim=32)',  pcaDim: 32, tempAlpha: 1.0 },
  { name: '仅温度 (α=0.5)',   pcaDim: 0,  tempAlpha: 0.5 },
  { name: 'PCA+温度',         pcaDim: 32, tempAlpha: 0.5 },
];

// 使用调参实验确定的最优基线参数
const BASE_OPTS = { D: 0.15, uStrength: 0.1, knn: 3, maxIter: 50, epsilon: 1e-3, dt: 0.1 };
const PRE_FILTER_K = 30;

(async () => {
  try {
    console.log('═'.repeat(70));
    console.log('  Agent8 信号放大验证 — 4 组配置对比');
    console.log('═'.repeat(70));

    // 初始化
    const index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 4096 });
    index.load(INDEX_FILE);
    const metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));
    const idMap = new Map();
    for (let i = 0; i < metadata.length; i++) {
      if (metadata[i].id !== undefined) idMap.set(metadata[i].id, i);
    }

    const apiKey = loadApiKey();
    if (!apiKey) { console.error('❌ 未找到 API Key'); process.exit(1); }
    const engine = new VectorizeEngine(apiKey);

    const adData = new ADRankData();
    await adData.initialize();
    console.log(`  索引: ${index.size()} 条向量\n`);

    // Pre-embed queries
    console.log('📡 Embedding queries...');
    const queryVectors = {};
    for (const q of QUERIES) {
      queryVectors[q] = await engine.embed(q);
      if (!(queryVectors[q] instanceof Float32Array))
        queryVectors[q] = new Float32Array(queryVectors[q]);
      process.stdout.write('.');
    }
    console.log(' done\n');

    // Pre-fetch candidates
    const candidates = {};
    for (const q of QUERIES) {
      const { vectors, metadata: metas } = await adData.getCandidates(queryVectors[q], PRE_FILTER_K);
      candidates[q] = { vectors, metas };
    }

    // === 实验 ===
    const allResults = {};
    for (const cfg of CONFIGS) {
      allResults[cfg.name] = { peList: [], reList: [], timeList: [], iterList: [], convList: [], rankings: {} };
    }

    for (const q of QUERIES) {
      console.log(`\n${'─'.repeat(70)}`);
      console.log(`  Query: "${q}"`);
      console.log('─'.repeat(70));

      const qVec = queryVectors[q];
      const { vectors, metas } = candidates[q];

      // 基线 cosine 排序
      const baseResults = index.search(qVec, 10);
      const baseKeys = Array.from(baseResults.keys).map(Number);
      const baseLaws = baseKeys.map(k => {
        const ai = idMap.get(k);
        const m = ai !== undefined ? metadata[ai] : null;
        return m ? `${m.law}|${m.article}` : '';
      });

      for (const cfg of CONFIGS) {
        const opts = { ...BASE_OPTS, pcaDim: cfg.pcaDim, tempAlpha: cfg.tempAlpha };
        const t0 = performance.now();
        const result = adRank(qVec, vectors, 10, opts);
        const elapsed = performance.now() - t0;

        const adLaws = result.rankings.map(r => {
          const m = metas[r.index];
          return m ? `${m.law}|${m.article}` : '';
        });
        const overlap = baseLaws.filter(x => adLaws.includes(x)).length;

        allResults[cfg.name].peList.push(result.peclet);
        allResults[cfg.name].reList.push(result.reynolds);
        allResults[cfg.name].timeList.push(elapsed);
        allResults[cfg.name].iterList.push(result.iterations);
        allResults[cfg.name].convList.push(result.convergence);
        allResults[cfg.name].rankings[q] = adLaws;

        console.log(`  [${cfg.name}]  Pe=${result.peclet.toFixed(3)}  Re=${result.reynolds.toFixed(3)}  iter=${result.iterations}  conv=${result.convergence}  overlap=${overlap}/10  ${elapsed.toFixed(1)}ms`);
      }

      await new Promise(r => setTimeout(r, 200));
    }

    // === 汇总 ===
    console.log('\n\n' + '═'.repeat(70));
    console.log('  实验汇总');
    console.log('═'.repeat(70));

    const avg = arr => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    console.log('\n  配置              | 平均 Pe  | 平均 Re  | 平均耗时   | 收敛率    | 平均迭代');
    console.log('  ' + '─'.repeat(80));
    const summaryRows = [];
    for (const cfg of CONFIGS) {
      const r = allResults[cfg.name];
      const avgPe = avg(r.peList);
      const avgRe = avg(r.reList);
      const avgTime = avg(r.timeList);
      const convRate = r.convList.filter(Boolean).length + '/' + r.convList.length;
      const avgIter = avg(r.iterList);
      console.log(`  ${cfg.name.padEnd(18)}| ${avgPe.toFixed(3).padStart(8)} | ${avgRe.toFixed(3).padStart(8)} | ${avgTime.toFixed(1).padStart(8)}ms | ${convRate.padEnd(9)} | ${avgIter.toFixed(1)}`);
      summaryRows.push({ config: cfg.name, pcaDim: cfg.pcaDim, tempAlpha: cfg.tempAlpha, avgPe, avgRe, avgTimeMs: +avgTime.toFixed(1), convergenceRate: convRate, avgIter: +avgIter.toFixed(1) });
    }

    // Pe 提升评估
    const basePe = avg(allResults[CONFIGS[0].name].peList);
    const bestPe = Math.max(...CONFIGS.map(c => avg(allResults[c.name].peList)));
    const bestConfig = CONFIGS[CONFIGS.map(c => avg(allResults[c.name].peList)).indexOf(bestPe)];

    console.log(`\n  📊 Baseline Pe: ${basePe.toFixed(3)}`);
    console.log(`  📊 Best Pe:     ${bestPe.toFixed(3)} (${bestConfig.name})`);
    console.log(`  📊 提升倍数:    ${(bestPe / Math.max(basePe, 1e-6)).toFixed(1)}x`);

    const peTarget = 0.3;
    console.log(`\n  ✅ 验证标准:`);
    console.log(`    [${bestPe >= peTarget ? '✓' : '✗'}] Pe >= ${peTarget} (actual: ${bestPe.toFixed(3)})`);

    // 保存结果
    const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_signal_amplify.json');
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      baseParams: BASE_OPTS,
      configs: CONFIGS,
      summary: summaryRows,
      baseline_pe: basePe,
      best_pe: bestPe,
      best_config: bestConfig.name,
      pe_boost: +(bestPe / Math.max(basePe, 1e-6)).toFixed(1),
    }, null, 2));
    console.log(`\n  📁 结果已保存: ${reportPath}`);
    console.log('\n✅ Agent8 信号放大验证完成');

  } catch (err) {
    console.error('\n❌ 实验失败:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
})();
