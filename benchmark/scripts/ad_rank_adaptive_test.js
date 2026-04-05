#!/usr/bin/env node
'use strict';

/**
 * AD-Rank Agent9 自适应参数验证实验
 * 
 * 对 10 个标准 query 跑两组对比：
 *   组 A (手动参数): D=0.15, uStrength=0.1
 *   组 B (自适应参数): autoD=true, autoU=true
 * 
 * 验证：
 * 1. 跨域 query（离婚财产分割）自动增大 u
 * 2. 单域 query（醉驾量刑）自动减小 u
 * 3. 整体盲评分数不低于手动调参
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { adRank } = require('./ad_rank');
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

const MANUAL_PARAMS = { D: 0.15, uStrength: 0.1, knn: 3, preFilterK: 30 };

// ========== DeepSeek 盲评 ==========
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
      hostname: 'api.deepseek.com', port: 443,
      path: '/v1/chat/completions', method: 'POST',
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
          if (json.error) { reject(new Error(json.error.message)); return; }
          resolve(json.choices?.[0]?.message?.content || '');
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('超时 30s')); });
    req.write(body);
    req.end();
  });
}

// ========== 主流程 ==========
(async () => {
  console.log('═'.repeat(70));
  console.log('  Agent9 自适应参数验证实验');
  console.log('  组 A: 手动参数 (D=0.15, u=0.1)');
  console.log('  组 B: 自适应参数 (autoD=true, autoU=true)');
  console.log('═'.repeat(70));

  const engine = new VectorizeEngine(loadApiKey());
  const adData = new ADRankData();
  await adData.initialize();

  const hasDeepSeek = !!DEEPSEEK_API_KEY;
  if (!hasDeepSeek) {
    console.log('⚠️  未找到 DEEPSEEK_API_KEY，跳过盲评环节\n');
  }

  const results = [];
  let aWins = 0, bWins = 0, ties = 0;

  for (let qi = 0; qi < QUERIES.length; qi++) {
    const query = QUERIES[qi];
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`  [${qi + 1}/${QUERIES.length}] "${query}"`);
    console.log('─'.repeat(70));

    // embed query
    let qVec = await engine.embed(query);
    if (!(qVec instanceof Float32Array)) qVec = new Float32Array(qVec);

    // 获取候选
    const { vectors, metadata: metas } = await adData.getCandidates(qVec, MANUAL_PARAMS.preFilterK);
    if (vectors.length === 0) { console.error('  ❌ 无候选'); continue; }

    // === 组 A: 手动参数 ===
    const tA = performance.now();
    const resultA = adRank(qVec, vectors, 5, {
      D: MANUAL_PARAMS.D, uStrength: MANUAL_PARAMS.uStrength,
      knn: MANUAL_PARAMS.knn, maxIter: 50, epsilon: 1e-3,
    });
    const tA_elapsed = performance.now() - tA;

    // === 组 B: 自适应参数 ===
    const tB = performance.now();
    const resultB = adRank(qVec, vectors, 5, {
      D: MANUAL_PARAMS.D, uStrength: MANUAL_PARAMS.uStrength,
      knn: MANUAL_PARAMS.knn, maxIter: 50, epsilon: 1e-3,
      autoD: true, autoU: true,
    });
    const tB_elapsed = performance.now() - tB;

    const groupA = resultA.rankings.map((r, i) => ({
      rank: i + 1,
      law: metas[r.index]?.law || '?',
      article: metas[r.index]?.article || '',
      content: (metas[r.index]?.content || '').substring(0, 200),
    }));
    const groupB = resultB.rankings.map((r, i) => ({
      rank: i + 1,
      law: metas[r.index]?.law || '?',
      article: metas[r.index]?.article || '',
      content: (metas[r.index]?.content || '').substring(0, 200),
    }));

    // 输出自适应参数信息
    console.log(`\n  📊 自适应参数:`);
    console.log(`     Fiedler λ₂ = ${resultB.fiedlerValue.toFixed(4)}`);
    if (resultB.autoParams) {
      console.log(`     D: ${resultB.autoParams.originalD} → ${resultB.autoParams.adaptiveD?.toFixed(4)}`);
      console.log(`     U: ${resultB.autoParams.originalU} → ${resultB.autoParams.adaptiveU?.toFixed(4)}`);
      console.log(`     跨域: ${resultB.autoParams.crossDomain ? '是 ✅' : '否'} (密度=${resultB.autoParams.density?.toFixed(4)}, 方差=${resultB.autoParams.simVar?.toFixed(6)})`);
    }

    console.log(`  ⏱  速度: A=${tA_elapsed.toFixed(1)}ms, B=${tB_elapsed.toFixed(1)}ms`);

    // 排序对比
    console.log(`\n  组A (手动)                              | 组B (自适应)`);
    console.log('  ' + '─'.repeat(42) + '|' + '─'.repeat(42));
    for (let i = 0; i < 5; i++) {
      const a = groupA[i];
      const b = groupB[i];
      const aStr = a ? `#${a.rank} ${a.law} ${a.article}`.padEnd(42) : ' '.repeat(42);
      const bStr = b ? `#${b.rank} ${b.law} ${b.article}` : '';
      console.log(`  ${aStr}| ${bStr}`);
    }

    // 盲评
    let aScore = 0, bScore = 0, winner = '跳过';
    if (hasDeepSeek) {
      const aIsFirst = Math.random() > 0.5;
      const first = aIsFirst ? groupA : groupB;
      const second = aIsFirst ? groupB : groupA;

      const formatGroup = (group) => group.map(r =>
        `  ${r.rank}. 【${r.law}】${r.article}\n     ${r.content}`
      ).join('\n');

      const prompt = `你是一位中国法律专家。以下是一个法律问题，以及两种不同检索算法返回的 Top-5 法条结果。
请你判断哪个方案检索的法条更精准、更切题。

## 法律问题
"${query}"

## 方案甲的检索结果
${formatGroup(first)}

## 方案乙的检索结果
${formatGroup(second)}

## 评估要求
1. 分别给两个方案打分（1-5分）
2. 明确选择哪个方案更好，或判定平局
3. 简要说明理由

请按以下格式回答：
方案甲评分: X/5
方案乙评分: X/5
胜出方: 方案甲/方案乙/平局
理由: ...`;

      try {
        const response = await callDeepSeek(prompt);
        const firstMatch = response.match(/方案甲评分[：:]*\s*(\d)[/／](\d)/);
        const secondMatch = response.match(/方案乙评分[：:]*\s*(\d)[/／](\d)/);
        const winnerMatch = response.match(/胜出方[：:]*\s*(方案[甲乙]|平局)/);

        const firstScore = firstMatch ? parseInt(firstMatch[1]) : 0;
        const secondScore = secondMatch ? parseInt(secondMatch[1]) : 0;
        aScore = aIsFirst ? firstScore : secondScore;
        bScore = aIsFirst ? secondScore : firstScore;

        const rawWinner = winnerMatch ? winnerMatch[1] : '未知';
        if (rawWinner === '平局') {
          winner = '平局'; ties++;
        } else if (rawWinner === '方案甲') {
          winner = aIsFirst ? '手动' : '自适应';
        } else if (rawWinner === '方案乙') {
          winner = aIsFirst ? '自适应' : '手动';
        }
        if (winner === '手动') aWins++;
        if (winner === '自适应') bWins++;

        console.log(`\n  🏆 盲评: 手动=${aScore}/5  自适应=${bScore}/5  胜出=${winner}`);
      } catch (e) {
        console.error(`  ❌ DeepSeek 调用失败: ${e.message}`);
      }
    }

    results.push({
      query,
      fiedlerValue: resultB.fiedlerValue,
      adaptiveD: resultB.autoParams?.adaptiveD,
      adaptiveU: resultB.autoParams?.adaptiveU,
      isCrossDomain: resultB.autoParams?.crossDomain,
      simVar: resultB.autoParams?.simVar,
      density: resultB.autoParams?.density,
      aScore, bScore, winner,
      tA: +tA_elapsed.toFixed(1),
      tB: +tB_elapsed.toFixed(1),
    });

    // 防止 API 限速
    if (hasDeepSeek) await new Promise(r => setTimeout(r, 1500));
  }

  // ========== 汇总 ==========
  console.log('\n\n' + '═'.repeat(70));
  console.log('  Agent9 自适应参数实验汇总');
  console.log('═'.repeat(70));

  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  console.log(`\n  📊 自适应参数分析:`);
  console.log(`     平均 Fiedler λ₂:  ${avg(results.map(r => r.fiedlerValue)).toFixed(4)}`);
  console.log(`     平均自适应 D:      ${avg(results.map(r => r.adaptiveD || 0)).toFixed(4)} (原始: ${MANUAL_PARAMS.D})`);
  console.log(`     平均自适应 U:      ${avg(results.map(r => r.adaptiveU || 0)).toFixed(4)} (原始: ${MANUAL_PARAMS.uStrength})`);
  console.log(`     跨域 query 数:    ${results.filter(r => r.isCrossDomain).length}/${results.length}`);

  // 分域分析
  console.log(`\n  🔍 分域分析:`);
  for (const r of results) {
    const marker = r.isCrossDomain ? '🌐' : '📍';
    console.log(`     ${marker} "${r.query}" → D=${r.adaptiveD?.toFixed(4)} U=${r.adaptiveU?.toFixed(4)} var=${r.simVar?.toFixed(6)}`);
  }

  if (hasDeepSeek && results.some(r => r.aScore > 0)) {
    console.log(`\n  🏆 盲评汇总:`);
    console.log(`     手动平均分:    ${avg(results.filter(r => r.aScore > 0).map(r => r.aScore)).toFixed(1)}/5`);
    console.log(`     自适应平均分:  ${avg(results.filter(r => r.bScore > 0).map(r => r.bScore)).toFixed(1)}/5`);
    console.log(`     胜负: 手动 ${aWins} : ${bWins} 自适应 (平局 ${ties})`);
  }

  // 保存
  const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_adaptive.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    manualParams: MANUAL_PARAMS,
    results,
    summary: {
      avgFiedler: +avg(results.map(r => r.fiedlerValue)).toFixed(4),
      avgAdaptiveD: +avg(results.map(r => r.adaptiveD || 0)).toFixed(4),
      avgAdaptiveU: +avg(results.map(r => r.adaptiveU || 0)).toFixed(4),
      crossDomainCount: results.filter(r => r.isCrossDomain).length,
      manualWins: aWins,
      adaptiveWins: bWins,
      ties,
    },
  }, null, 2));
  console.log(`\n  📁 结果已保存: ${reportPath}`);
  console.log('\n✅ Agent9 实验完成');
})();
