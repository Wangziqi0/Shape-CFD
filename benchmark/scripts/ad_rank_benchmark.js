#!/usr/bin/env node
'use strict';

/**
 * AD-Rank 基线实验 — A/B 对比
 * 
 * 组 A (基线): 纯 cosine similarity 排序 (当前管线)
 * 组 B (AD-Rank v1): HNSW 粗筛 → 对流-扩散重排序
 * 
 * 实验目标：
 * 1. 记录组 A 的结果和速度
 * 2. 记录组 B 的结果和速度
 * 3. 对比排序差异 (Kendall tau, Top-K 命中变化)
 */

const fs = require('fs');
const path = require('path');
const usearch = require('usearch');

// ========== 实验配置 ==========
const EXPERIMENT = {
  queries: [
    '劳动合同解除赔偿标准',
    '醉驾量刑标准',
    '房屋租赁合同违约金',
    '工伤认定标准和赔偿',
    '离婚财产分割',
    '借款合同利息上限',
    '交通事故责任划分',
    '商标侵权赔偿',
    '行政处罚听证程序',
    '未成年人犯罪处罚',
  ],
  topK: 10,
  preFilterK: 30,  // HNSW 粗筛数
};

// ========== 加载模块 ==========
const VECTORS_DIR = path.join(__dirname, 'knowledge_base', 'vectors');
const INDEX_FILE = path.join(VECTORS_DIR, 'law.usearch');
const META_FILE = path.join(VECTORS_DIR, 'metadata.json');

(async () => {
  console.log('═'.repeat(70));
  console.log('  AD-Rank 基线实验 — 组 A (cosine) vs 组 B (AD-Rank v1)');
  console.log('═'.repeat(70));

  // 1. 加载索引
  console.log('\n📦 加载索引...');
  const index = new usearch.Index({ metric: 'cos', connectivity: 16, dimensions: 4096 });
  index.load(INDEX_FILE);
  const metadata = JSON.parse(fs.readFileSync(META_FILE, 'utf-8'));

  // 构建 id → arrayIndex 映射
  const idMap = new Map();
  for (let i = 0; i < metadata.length; i++) {
    if (metadata[i].id !== undefined) idMap.set(metadata[i].id, i);
  }

  console.log(`  索引: ${index.size()} 条向量, 元数据: ${metadata.length} 条`);

  // 2. 加载 Embedding 引擎
  const { VectorizeEngine, loadApiKey } = require('./vectorize_engine');
  const apiKey = loadApiKey();
  if (!apiKey) {
    console.error('❌ 未找到 API Key');
    process.exit(1);
  }
  const engine = new VectorizeEngine(apiKey);

  // 3. 加载 AD-Rank
  const { adRank } = require('./ad_rank');

  // 4. 初始化 ADRankData（含向量缓存层）
  const { ADRankData } = require('./ad_rank_data');
  const adData = new ADRankData();
  await adData.initialize();

  // ========== 实验结果存储 ==========
  const results = [];

  for (let qi = 0; qi < EXPERIMENT.queries.length; qi++) {
    const query = EXPERIMENT.queries[qi];
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`  Query ${qi + 1}/${EXPERIMENT.queries.length}: "${query}"`);
    console.log('─'.repeat(70));

    // ── embed query ──
    const t_embed_start = performance.now();
    let queryVector;
    try {
      queryVector = await engine.embed(query);
      if (!(queryVector instanceof Float32Array)) {
        queryVector = new Float32Array(queryVector);
      }
    } catch (e) {
      console.error(`  ❌ Query embedding 失败: ${e.message}`);
      continue;
    }
    const t_embed = performance.now() - t_embed_start;

    // ══════════════════════════════════════
    // 组 A: 纯 cosine (HNSW 直接排序)
    // ══════════════════════════════════════
    const t_a_start = performance.now();
    const rawResults = index.search(queryVector, EXPERIMENT.topK);
    const t_a = performance.now() - t_a_start;

    const groupA = [];
    const aKeys = Array.from(rawResults.keys).map(Number);
    const aDists = Array.from(rawResults.distances);
    for (let i = 0; i < aKeys.length; i++) {
      const arrayIdx = idMap.get(aKeys[i]);
      const meta = arrayIdx !== undefined ? metadata[arrayIdx] : metadata[aKeys[i]];
      if (meta) {
        groupA.push({
          rank: i + 1,
          law: meta.law || '?',
          article: meta.article || '',
          score: (1 - aDists[i]).toFixed(4),
          content: (meta.content || '').substring(0, 50),
        });
      }
    }

    // ══════════════════════════════════════
    // 组 B: AD-Rank v1（通过 ADRankData 缓存层）
    // ══════════════════════════════════════
    const t_b_hnsw_start = performance.now();

    // 使用 ADRankData 的缓存路径：SQLite 命中 → ~1ms, 未命中 → re-embed + 写入缓存
    const { vectors: candidateVectors, metadata: candidateMetas, distances: coarseDists, cacheStats } =
      await adData.getCandidates(queryVector, EXPERIMENT.preFilterK);

    const t_b_hnsw_and_embed = performance.now() - t_b_hnsw_start;

    if (candidateVectors.length === 0) {
      console.error(`  ❌ 无有效候选向量`);
      continue;
    }

    // Step 4: 跑 AD-Rank
    const t_b_adrank_start = performance.now();
    const adResult = adRank(queryVector, candidateVectors, EXPERIMENT.topK);
    const t_b_adrank = performance.now() - t_b_adrank_start;

    const t_b_total = t_b_hnsw_and_embed + t_b_adrank;

    const groupB = adResult.rankings.map((r, i) => ({
      rank: i + 1,
      law: candidateMetas[r.index]?.law || '?',
      article: candidateMetas[r.index]?.article || '',
      score: r.score.toFixed(4),
      topology: r.topology,
      content: (candidateMetas[r.index]?.content || '').substring(0, 50),
    }));

    // ══════════════════════════════════════
    // 输出对比
    // ══════════════════════════════════════
    console.log(`\n  ⏱  速度对比:`);
    console.log(`     Query embedding:  ${t_embed.toFixed(1)}ms`);
    console.log(`     组 A (cosine):    ${t_a.toFixed(1)}ms`);
    console.log(`     组 B (AD-Rank):   ${t_b_total.toFixed(1)}ms (HNSW+embed ${t_b_hnsw_and_embed.toFixed(1)} + solve ${t_b_adrank.toFixed(1)}ms)`);
    console.log(`     缓存: hit=${cacheStats.hit} miss=${cacheStats.miss}`);
    console.log(`     Re: ${adResult.reynolds.toFixed(3)}, Pe: ${adResult.peclet.toFixed(3)}, iter: ${adResult.iterations}, conv: ${adResult.convergence}`);

    console.log(`\n  📊 排序对比 (Top-${EXPERIMENT.topK}):`);
    console.log('     组A (cosine)                          | 组B (AD-Rank)');
    console.log('     ' + '─'.repeat(42) + '|' + '─'.repeat(42));
    const maxLen = Math.max(groupA.length, groupB.length);
    for (let i = 0; i < maxLen; i++) {
      const a = groupA[i];
      const b = groupB[i];
      const aStr = a ? `#${a.rank} ${a.law} ${a.article} (${a.score})`.padEnd(42) : ' '.repeat(42);
      const bStr = b ? `#${b.rank} ${b.law} ${b.article} (${b.score}) [${b.topology}]` : '';
      console.log(`     ${aStr}| ${bStr}`);
    }

    // 计算排序变化
    const aLawArticles = groupA.map(a => `${a.law}|${a.article}`);
    const bLawArticles = groupB.map(b => `${b.law}|${b.article}`);

    // Top-K 重叠率
    const overlap = aLawArticles.filter(x => bLawArticles.includes(x)).length;
    const overlapRate = (overlap / Math.max(aLawArticles.length, 1) * 100).toFixed(0);

    // 新引入的文档（AD-Rank 引入但 cosine 没有的）
    const newInB = bLawArticles.filter(x => !aLawArticles.includes(x));

    console.log(`\n  📐 排序变化: Top-${EXPERIMENT.topK} 重叠率 ${overlapRate}%, 新引入 ${newInB.length} 个`);
    if (newInB.length > 0) {
      console.log(`     新引入: ${newInB.slice(0, 3).join(', ')}${newInB.length > 3 ? '...' : ''}`);
    }

    // 汇聚/发散/停滞点
    console.log(`  🌊 拓扑: 汇聚${adResult.convergencePoints.length} 发散${adResult.divergencePoints.length} 停滞${adResult.stagnationPoints.length}`);

    results.push({
      query,
      t_embed: +t_embed.toFixed(1),
      t_a: +t_a.toFixed(1),
      t_b_total: +t_b_total.toFixed(1),
      t_b_hnsw_embed: +t_b_hnsw_and_embed.toFixed(1),
      t_b_adrank: +t_b_adrank.toFixed(1),
      cache_hit: cacheStats.hit,
      cache_miss: cacheStats.miss,
      reynolds: +adResult.reynolds.toFixed(3),
      peclet: +adResult.peclet.toFixed(3),
      overlapRate: +overlapRate,
      newInB: newInB.length,
      convergence: adResult.convergence,
      iterations: adResult.iterations,
    });

    // 每个 query 之间稍停一下，避免 API 过载
    await new Promise(r => setTimeout(r, 200));
  }

  // ══════════════════════════════════════
  // 汇总报告
  // ══════════════════════════════════════
  console.log('\n\n' + '═'.repeat(70));
  console.log('  实验汇总');
  console.log('═'.repeat(70));

  if (results.length === 0) {
    console.log('  ⚠️  没有成功的实验结果');
    return;
  }

  const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

  console.log(`\n  查询数: ${results.length}`);
  console.log(`\n  ⏱  平均速度:`);
  console.log(`     Query embed:   ${avg(results.map(r => r.t_embed)).toFixed(1)}ms`);
  console.log(`     组 A (cosine): ${avg(results.map(r => r.t_a)).toFixed(1)}ms`);
  console.log(`     组 B (total):  ${avg(results.map(r => r.t_b_total)).toFixed(1)}ms`);
  console.log(`       ├ HNSW+embed: ${avg(results.map(r => r.t_b_hnsw_embed)).toFixed(1)}ms`);
  console.log(`       └ AD-Rank:   ${avg(results.map(r => r.t_b_adrank)).toFixed(1)}ms`);
  console.log(`     缓存命中率:  ${(results.reduce((s,r) => s + r.cache_hit, 0) / Math.max(results.reduce((s,r) => s + r.cache_hit + r.cache_miss, 0), 1) * 100).toFixed(0)}%`);

  console.log(`\n  📐 排序变化:`);
  console.log(`     平均 Top-${EXPERIMENT.topK} 重叠率: ${avg(results.map(r => r.overlapRate)).toFixed(0)}%`);
  console.log(`     平均新引入文档:   ${avg(results.map(r => r.newInB)).toFixed(1)} 个`);

  console.log(`\n  🌊 物理指标:`);
  console.log(`     平均 Re:  ${avg(results.map(r => r.reynolds)).toFixed(3)}`);
  console.log(`     平均 Pe:  ${avg(results.map(r => r.peclet)).toFixed(3)}`);
  console.log(`     收敛率:   ${results.filter(r => r.convergence).length}/${results.length}`);
  console.log(`     平均迭代: ${avg(results.map(r => r.iterations)).toFixed(1)}`);

  // 保存结果到 JSON
  const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_baseline.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    config: EXPERIMENT,
    results,
    summary: {
      avgEmbedMs: +avg(results.map(r => r.t_embed)).toFixed(1),
      avgCosineMs: +avg(results.map(r => r.t_a)).toFixed(1),
      avgADRankTotalMs: +avg(results.map(r => r.t_b_total)).toFixed(1),
      avgADRankSolveMs: +avg(results.map(r => r.t_b_adrank)).toFixed(1),
      avgOverlapRate: +avg(results.map(r => r.overlapRate)).toFixed(0),
      avgNewDocs: +avg(results.map(r => r.newInB)).toFixed(1),
      avgReynolds: +avg(results.map(r => r.reynolds)).toFixed(3),
      avgPeclet: +avg(results.map(r => r.peclet)).toFixed(3),
    },
  }, null, 2));
  console.log(`\n  📁 结果已保存: ${reportPath}`);
  console.log('\n✅ 实验完成');
})();
