#!/usr/bin/env node
'use strict';

/**
 * AD-Rank V3 四组对比实验 + DeepSeek 盲评
 *
 * 组 A: cosine (基线)
 * 组 B: AD-Rank v2 (全局对流, D=0.15 u=0.1 knn=3)
 * 组 C: MDA (M=8, k=32, u=0.3)
 * 组 D: BAA (B=8, u=0.3)
 *
 * 对 10 个标准 query 跑：速度、Top-10 重叠率、对流贡献率、DeepSeek 盲评
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const usearch = require('usearch');
const { adRank } = require('./ad_rank');
const { adRankMDA, adRankBAA } = require('./ad_rank_v3');
const { ADRankData } = require('./ad_rank_data');
const { VectorizeEngine, loadApiKey } = require('./vectorize_engine');

// ========== 配置 ==========
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || (() => {
  try {
    const envContent = fs.readFileSync(path.join(__dirname, '.env'), 'utf-8');
    const match = envContent.match(/DEEPSEEK_API_KEY=(.+)/);
    return match ? match[1].trim() : '';
  } catch { return ''; }
})();

const QUERIES = [
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
];

const PARAMS = {
  v2: { D: 0.15, uStrength: 0.1, knn: 3, maxIter: 50, epsilon: 1e-3 },
  mda: { M: 8, k: 32, D: 0.15, uStrength: 0.3, knn: 3, maxIter: 50, epsilon: 1e-3 },
  baa: { B: 8, D: 0.15, uStrength: 0.3, knn: 3, maxIter: 50, epsilon: 1e-3 },
  preFilterK: 30,
};

// ========== DeepSeek API ==========
function callDeepSeek(prompt) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: 'deepseek-chat',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      max_tokens: 800,
      stream: false,
    });

    const req = https.request({
      hostname: 'api.deepseek.com',
      port: 443,
      path: '/v1/chat/completions',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          if (json.error) { reject(new Error(json.error.message || JSON.stringify(json.error))); return; }
          resolve(json.choices?.[0]?.message?.content || '');
        } catch (e) { reject(e); }
      });
    });

    req.on('error', reject);
    req.setTimeout(60000, () => { req.destroy(); reject(new Error('超时 60s')); });
    req.write(body);
    req.end();
  });
}

// ========== 盲评函数 ==========
async function blindEval(query, groupA, groupTarget, targetLabel) {
  if (!DEEPSEEK_API_KEY) return null;

  const aIsFirst = Math.random() > 0.5;
  const first = aIsFirst ? groupA : groupTarget;
  const second = aIsFirst ? groupTarget : groupA;

  const formatGroup = (group) => group.map(r =>
    `  ${r.rank}. 【${r.law}】${r.article}\n     ${r.content}`
  ).join('\n');

  const prompt = `你是一位中国法律专家。以下是一个法律问题，以及两种不同检索算法返回的 Top-5 法条结果。
请你判断哪个方案检索的法条更精准、更切题、更有实际参考价值。

## 法律问题
"${query}"

## 方案甲的检索结果
${formatGroup(first)}

## 方案乙的检索结果
${formatGroup(second)}

## 评估要求
1. 分别给两个方案打分（1-5分，5分最好），评估标准：
   - 法条与问题的相关性（最重要）
   - 法条的专业性和精确程度
   - 覆盖面（是否涵盖了问题的关键方面）
2. 明确选择哪个方案更好，或判定平局
3. 简要说明理由（2-3句话）

请按以下格式回答：
方案甲评分: X/5
方案乙评分: X/5
胜出方: 方案甲/方案乙/平局
理由: ...`;

  try {
    const response = await callDeepSeek(prompt);
    const firstScoreMatch = response.match(/方案甲评分[：:]*\s*(\d)[/／](\d)/);
    const secondScoreMatch = response.match(/方案乙评分[：:]*\s*(\d)[/／](\d)/);
    const winnerMatch = response.match(/胜出方[：:]*\s*(方案[甲乙]|平局)/);

    const firstScore = firstScoreMatch ? parseInt(firstScoreMatch[1]) : 0;
    const secondScore = secondScoreMatch ? parseInt(secondScoreMatch[1]) : 0;
    const winner = winnerMatch ? winnerMatch[1] : '未知';

    const aScore = aIsFirst ? firstScore : secondScore;
    const targetScore = aIsFirst ? secondScore : firstScore;

    let realWinner;
    if (winner === '平局') realWinner = '平局';
    else if ((winner === '方案甲' && aIsFirst) || (winner === '方案乙' && !aIsFirst)) realWinner = 'cosine';
    else realWinner = targetLabel;

    return { aScore, targetScore, winner: realWinner, response: response.substring(0, 300) };
  } catch (e) {
    console.error(`    ❌ DeepSeek 盲评失败 (${targetLabel}): ${e.message}`);
    return null;
  }
}

// ========== 主流程 ==========
(async () => {
  console.log('═'.repeat(70));
  console.log('  AD-Rank V3 四组对比实验');
  console.log('═'.repeat(70));

  // 初始化
  const VECTORS_DIR = path.join(__dirname, 'knowledge_base', 'vectors');
  const INDEX_FILE = path.join(VECTORS_DIR, 'law.usearch');
  const META_FILE = path.join(VECTORS_DIR, 'metadata.json');

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

  console.log(`  索引: ${index.size()} 条, 元数据: ${metadata.length} 条`);
  console.log(`  DeepSeek API: ${DEEPSEEK_API_KEY ? '✅' : '❌ (跳过盲评)'}`);

  const results = [];

  for (let qi = 0; qi < QUERIES.length; qi++) {
    const query = QUERIES[qi];
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`  [${qi + 1}/${QUERIES.length}] "${query}"`);
    console.log('─'.repeat(70));

    // embed query
    let qVec;
    try {
      qVec = await engine.embed(query);
      if (!(qVec instanceof Float32Array)) qVec = new Float32Array(qVec);
    } catch (e) {
      console.error(`  ❌ Query embedding 失败: ${e.message}`);
      continue;
    }

    // ══ 组 A: cosine ══
    const tA = performance.now();
    const aResults = index.search(qVec, 10);
    const tA_ms = performance.now() - tA;

    const aKeys = Array.from(aResults.keys).map(Number);
    const aDists = Array.from(aResults.distances);
    const groupA = aKeys.map((k, i) => {
      const ai = idMap.get(k);
      const m = ai !== undefined ? metadata[ai] : null;
      return m ? { rank: i + 1, law: m.law || '?', article: m.article || '', content: (m.content || '').substring(0, 200), score: +(1 - aDists[i]).toFixed(4) } : null;
    }).filter(Boolean);

    // ══ 获取候选向量（共用） ══
    const { vectors: candVecs, metadata: candMetas, cacheStats } =
      await adData.getCandidates(qVec, PARAMS.preFilterK);

    if (candVecs.length === 0) { console.error('  ❌ 无候选'); continue; }

    // ══ 组 B: AD-Rank v2 ══
    const tB = performance.now();
    const v2Result = adRank(qVec, candVecs, 10, PARAMS.v2);
    const tB_ms = performance.now() - tB;

    const groupB = v2Result.rankings.map((r, i) => ({
      rank: i + 1, law: candMetas[r.index]?.law || '?', article: candMetas[r.index]?.article || '',
      content: (candMetas[r.index]?.content || '').substring(0, 200), score: +r.score.toFixed(4),
    }));

    // ══ 组 C: MDA ══
    const tC = performance.now();
    const mdaResult = adRankMDA(qVec, candVecs, 10, PARAMS.mda);
    const tC_ms = performance.now() - tC;

    const groupC = mdaResult.rankings.map((r, i) => ({
      rank: i + 1, law: candMetas[r.index]?.law || '?', article: candMetas[r.index]?.article || '',
      content: (candMetas[r.index]?.content || '').substring(0, 200), score: +r.score.toFixed(4),
    }));

    // ══ 组 D: BAA ══
    const tD = performance.now();
    const baaResult = adRankBAA(qVec, candVecs, 10, PARAMS.baa);
    const tD_ms = performance.now() - tD;

    const groupD = baaResult.rankings.map((r, i) => ({
      rank: i + 1, law: candMetas[r.index]?.law || '?', article: candMetas[r.index]?.article || '',
      content: (candMetas[r.index]?.content || '').substring(0, 200), score: +r.score.toFixed(4),
    }));

    // ══ 计算指标 ══
    const aLaws = groupA.map(a => `${a.law}|${a.article}`);
    const bLaws = groupB.map(b => `${b.law}|${b.article}`);
    const cLaws = groupC.map(c => `${c.law}|${c.article}`);
    const dLaws = groupD.map(d => `${d.law}|${d.article}`);

    const overlapBC = bLaws.filter(x => aLaws.includes(x)).length / Math.max(aLaws.length, 1);
    const overlapCA = cLaws.filter(x => aLaws.includes(x)).length / Math.max(aLaws.length, 1);
    const overlapDA = dLaws.filter(x => aLaws.includes(x)).length / Math.max(aLaws.length, 1);

    console.log(`\n  ⏱  速度: A=${tA_ms.toFixed(1)}ms  B(v2)=${tB_ms.toFixed(1)}ms  C(MDA)=${tC_ms.toFixed(1)}ms  D(BAA)=${tD_ms.toFixed(1)}ms`);
    console.log(`  📊 重叠率(vs A): B=${(overlapBC*100).toFixed(0)}%  C=${(overlapCA*100).toFixed(0)}%  D=${(overlapDA*100).toFixed(0)}%`);
    console.log(`  🌊 Pe: B=${v2Result.peclet.toFixed(3)}  C=${mdaResult.peclet.toFixed(3)}  D=${baaResult.peclet.toFixed(3)}`);
    console.log(`  🔥 对流贡献率: C=${(mdaResult.advectionContribution*100).toFixed(1)}%  D=${(baaResult.advectionContribution*100).toFixed(1)}%`);
    console.log(`  缓存: hit=${cacheStats.hit} miss=${cacheStats.miss}`);

    // ══ DeepSeek 盲评: A vs C 和 A vs D ══
    const top5A = groupA.slice(0, 5);
    const top5C = groupC.slice(0, 5);
    const top5D = groupD.slice(0, 5);

    let evalAC = null, evalAD = null;
    if (DEEPSEEK_API_KEY) {
      console.log(`  🤖 DeepSeek 盲评...`);
      evalAC = await blindEval(query, top5A, top5C, 'MDA');
      if (evalAC) {
        console.log(`    A vs C(MDA): cosine=${evalAC.aScore}/5 MDA=${evalAC.targetScore}/5 胜出=${evalAC.winner}`);
      }
      await new Promise(r => setTimeout(r, 1500));

      evalAD = await blindEval(query, top5A, top5D, 'BAA');
      if (evalAD) {
        console.log(`    A vs D(BAA): cosine=${evalAD.aScore}/5 BAA=${evalAD.targetScore}/5 胜出=${evalAD.winner}`);
      }
      await new Promise(r => setTimeout(r, 1500));
    }

    results.push({
      query,
      speed: { A: +tA_ms.toFixed(1), B: +tB_ms.toFixed(1), C: +tC_ms.toFixed(1), D: +tD_ms.toFixed(1) },
      overlap: { B: +(overlapBC*100).toFixed(0), C: +(overlapCA*100).toFixed(0), D: +(overlapDA*100).toFixed(0) },
      peclet: { B: +v2Result.peclet.toFixed(3), C: +mdaResult.peclet.toFixed(3), D: +baaResult.peclet.toFixed(3) },
      advContrib: { C: +(mdaResult.advectionContribution*100).toFixed(1), D: +(baaResult.advectionContribution*100).toFixed(1) },
      convergence: { B: v2Result.convergence, C: mdaResult.convergence, D: baaResult.convergence },
      blindEval: {
        AC: evalAC ? { aScore: evalAC.aScore, cScore: evalAC.targetScore, winner: evalAC.winner } : null,
        AD: evalAD ? { aScore: evalAD.aScore, dScore: evalAD.targetScore, winner: evalAD.winner } : null,
      },
    });
  }

  // ══════════════════════════════════════
  // 汇总
  // ══════════════════════════════════════
  console.log('\n\n' + '═'.repeat(70));
  console.log('  V3 四组对比汇总');
  console.log('═'.repeat(70));

  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  console.log(`\n  查询数: ${results.length}`);

  console.log(`\n  ⏱  平均速度:`);
  console.log(`     A (cosine): ${avg(results.map(r => r.speed.A)).toFixed(1)}ms`);
  console.log(`     B (v2):     ${avg(results.map(r => r.speed.B)).toFixed(1)}ms`);
  console.log(`     C (MDA):    ${avg(results.map(r => r.speed.C)).toFixed(1)}ms`);
  console.log(`     D (BAA):    ${avg(results.map(r => r.speed.D)).toFixed(1)}ms`);

  console.log(`\n  📊 平均重叠率 (vs cosine):`);
  console.log(`     B (v2):  ${avg(results.map(r => r.overlap.B)).toFixed(0)}%`);
  console.log(`     C (MDA): ${avg(results.map(r => r.overlap.C)).toFixed(0)}%`);
  console.log(`     D (BAA): ${avg(results.map(r => r.overlap.D)).toFixed(0)}%`);

  console.log(`\n  🌊 平均 Péclet:`);
  console.log(`     B (v2):  ${avg(results.map(r => r.peclet.B)).toFixed(3)}`);
  console.log(`     C (MDA): ${avg(results.map(r => r.peclet.C)).toFixed(3)}`);
  console.log(`     D (BAA): ${avg(results.map(r => r.peclet.D)).toFixed(3)}`);

  console.log(`\n  🔥 平均对流贡献率:`);
  console.log(`     C (MDA): ${avg(results.map(r => r.advContrib.C)).toFixed(1)}%`);
  console.log(`     D (BAA): ${avg(results.map(r => r.advContrib.D)).toFixed(1)}%`);

  // 盲评汇总
  const acEvals = results.filter(r => r.blindEval.AC);
  const adEvals = results.filter(r => r.blindEval.AD);

  if (acEvals.length > 0) {
    const acAvgA = avg(acEvals.map(r => r.blindEval.AC.aScore));
    const acAvgC = avg(acEvals.map(r => r.blindEval.AC.cScore));
    const acWins = { cosine: 0, MDA: 0, tie: 0 };
    acEvals.forEach(r => {
      if (r.blindEval.AC.winner === 'cosine') acWins.cosine++;
      else if (r.blindEval.AC.winner === 'MDA') acWins.MDA++;
      else acWins.tie++;
    });

    console.log(`\n  🤖 DeepSeek 盲评 A vs C(MDA):`);
    console.log(`     平均分: cosine ${acAvgA.toFixed(1)}/5  MDA ${acAvgC.toFixed(1)}/5`);
    console.log(`     胜负: cosine ${acWins.cosine} : ${acWins.MDA} MDA (平局 ${acWins.tie})`);
  }

  if (adEvals.length > 0) {
    const adAvgA = avg(adEvals.map(r => r.blindEval.AD.aScore));
    const adAvgD = avg(adEvals.map(r => r.blindEval.AD.dScore));
    const adWins = { cosine: 0, BAA: 0, tie: 0 };
    adEvals.forEach(r => {
      if (r.blindEval.AD.winner === 'cosine') adWins.cosine++;
      else if (r.blindEval.AD.winner === 'BAA') adWins.BAA++;
      else adWins.tie++;
    });

    console.log(`\n  🤖 DeepSeek 盲评 A vs D(BAA):`);
    console.log(`     平均分: cosine ${adAvgA.toFixed(1)}/5  BAA ${adAvgD.toFixed(1)}/5`);
    console.log(`     胜负: cosine ${adWins.cosine} : ${adWins.BAA} BAA (平局 ${adWins.tie})`);
  }

  // 保存
  const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_v3.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    params: PARAMS,
    results,
    summary: {
      avgSpeed: {
        A: +avg(results.map(r => r.speed.A)).toFixed(1),
        B: +avg(results.map(r => r.speed.B)).toFixed(1),
        C: +avg(results.map(r => r.speed.C)).toFixed(1),
        D: +avg(results.map(r => r.speed.D)).toFixed(1),
      },
      avgOverlap: {
        B: +avg(results.map(r => r.overlap.B)).toFixed(0),
        C: +avg(results.map(r => r.overlap.C)).toFixed(0),
        D: +avg(results.map(r => r.overlap.D)).toFixed(0),
      },
      avgPeclet: {
        B: +avg(results.map(r => r.peclet.B)).toFixed(3),
        C: +avg(results.map(r => r.peclet.C)).toFixed(3),
        D: +avg(results.map(r => r.peclet.D)).toFixed(3),
      },
      avgAdvContrib: {
        C: +avg(results.map(r => r.advContrib.C)).toFixed(1),
        D: +avg(results.map(r => r.advContrib.D)).toFixed(1),
      },
      blindEval: {
        AC: acEvals.length > 0 ? {
          avgCosine: +avg(acEvals.map(r => r.blindEval.AC.aScore)).toFixed(1),
          avgMDA: +avg(acEvals.map(r => r.blindEval.AC.cScore)).toFixed(1),
        } : null,
        AD: adEvals.length > 0 ? {
          avgCosine: +avg(adEvals.map(r => r.blindEval.AD.aScore)).toFixed(1),
          avgBAA: +avg(adEvals.map(r => r.blindEval.AD.dScore)).toFixed(1),
        } : null,
      },
    },
  }, null, 2));

  console.log(`\n  📁 结果已保存: ${reportPath}`);
  console.log('\n✅ V3 实验完成');
})();
