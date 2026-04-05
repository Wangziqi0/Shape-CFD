#!/usr/bin/env node
'use strict';

/**
 * AD-Rank 盲评实验 — DeepSeek 作为裁判
 * 
 * 对每个 query，将组 A (cosine) 和组 C (AD-Rank) 的 Top-5 法条
 * 匿名呈现给 DeepSeek，让它判断哪组更佳。
 * 
 * 关键设计：
 * - A/B 随机分配为"方案甲/方案乙"，消除位置偏差
 * - DeepSeek 看不到算法名称，只看法条内容
 * - 要求给出 1-5 分打分 + 理由
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const usearch = require('usearch');
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

const BEST_PARAMS = { D: 0.15, uStrength: 0.1, knn: 3, preFilterK: 30 };

// ========== DeepSeek API 调用 ==========
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
          if (json.error) {
            reject(new Error(json.error.message || JSON.stringify(json.error)));
            return;
          }
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
  console.log('  AD-Rank 盲评实验 — DeepSeek 裁判');
  console.log('═'.repeat(70));

  if (!DEEPSEEK_API_KEY) {
    console.error('❌ 未找到 DEEPSEEK_API_KEY');
    process.exit(1);
  }

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

  const results = [];
  let aWins = 0, cWins = 0, ties = 0;

  for (let qi = 0; qi < QUERIES.length; qi++) {
    const query = QUERIES[qi];
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`  [${qi + 1}/${QUERIES.length}] "${query}"`);
    console.log('─'.repeat(70));

    // embed query
    let qVec = await engine.embed(query);
    if (!(qVec instanceof Float32Array)) qVec = new Float32Array(qVec);

    // 组 A: cosine Top-5
    const aResults = index.search(qVec, 5);
    const aKeys = Array.from(aResults.keys).map(Number);
    const groupA = aKeys.map((k, i) => {
      const ai = idMap.get(k);
      const m = ai !== undefined ? metadata[ai] : null;
      return m ? {
        rank: i + 1,
        law: m.law || '?',
        article: m.article || '',
        content: (m.content || '').substring(0, 200),
      } : null;
    }).filter(Boolean);

    // 组 C: AD-Rank Top-5
    const { vectors, metadata: metas } = await adData.getCandidates(qVec, BEST_PARAMS.preFilterK);
    const adResult = adRank(qVec, vectors, 5, {
      D: BEST_PARAMS.D, uStrength: BEST_PARAMS.uStrength,
      knn: BEST_PARAMS.knn, maxIter: 50, epsilon: 1e-3,
    });
    const groupC = adResult.rankings.map((r, i) => ({
      rank: i + 1,
      law: metas[r.index]?.law || '?',
      article: metas[r.index]?.article || '',
      content: (metas[r.index]?.content || '').substring(0, 200),
    }));

    // 随机分配甲乙（消除位置偏差）
    const aIsFirst = Math.random() > 0.5;
    const first = aIsFirst ? groupA : groupC;
    const second = aIsFirst ? groupC : groupA;
    const firstLabel = '方案甲';
    const secondLabel = '方案乙';

    // 构造 prompt
    const formatGroup = (group) => group.map(r =>
      `  ${r.rank}. 【${r.law}】${r.article}\n     ${r.content}`
    ).join('\n');

    const prompt = `你是一位中国法律专家。以下是一个法律问题，以及两种不同检索算法返回的 Top-5 法条结果。
请你判断哪个方案检索的法条更精准、更切题、更有实际参考价值。

## 法律问题
"${query}"

## ${firstLabel}的检索结果
${formatGroup(first)}

## ${secondLabel}的检索结果
${formatGroup(second)}

## 评估要求
1. 分别给两个方案打分（1-5分，5分最好），评估标准：
   - 法条与问题的相关性（最重要）
   - 法条的专业性和精确程度
   - 覆盖面（是否涵盖了问题的关键方面）
2. 明确选择哪个方案更好，或判定平局
3. 简要说明理由（2-3句话）

请按以下格式回答：
${firstLabel}评分: X/5
${secondLabel}评分: X/5
胜出方: 方案甲/方案乙/平局
理由: ...`;

    // 调用 DeepSeek
    let response;
    try {
      response = await callDeepSeek(prompt);
    } catch (e) {
      console.error(`  ❌ DeepSeek 调用失败: ${e.message}`);
      continue;
    }

    console.log(`\n  DeepSeek 裁决:\n${response.split('\n').map(l => '    ' + l).join('\n')}`);

    // 解析结果
    const firstScoreMatch = response.match(new RegExp(`${firstLabel}评分[：:]*\\s*(\\d)[/／](\\d)`));
    const secondScoreMatch = response.match(new RegExp(`${secondLabel}评分[：:]*\\s*(\\d)[/／](\\d)`));
    const winnerMatch = response.match(/胜出方[：:]*\s*(方案[甲乙]|平局)/);

    const firstScore = firstScoreMatch ? parseInt(firstScoreMatch[1]) : 0;
    const secondScore = secondScoreMatch ? parseInt(secondScoreMatch[1]) : 0;
    const winner = winnerMatch ? winnerMatch[1] : '未知';

    // 还原真实身份
    let realWinner;
    if (winner === '平局') {
      realWinner = '平局';
      ties++;
    } else if (winner === firstLabel) {
      realWinner = aIsFirst ? 'cosine' : 'AD-Rank';
    } else if (winner === secondLabel) {
      realWinner = aIsFirst ? 'AD-Rank' : 'cosine';
    } else {
      realWinner = '未知';
    }

    if (realWinner === 'cosine') aWins++;
    if (realWinner === 'AD-Rank') cWins++;

    const aScore = aIsFirst ? firstScore : secondScore;
    const cScore = aIsFirst ? secondScore : firstScore;

    console.log(`\n  📊 还原: cosine=${aScore}/5  AD-Rank=${cScore}/5  胜出=${realWinner}`);
    console.log(`     (${aIsFirst ? '甲=cosine 乙=AD-Rank' : '甲=AD-Rank 乙=cosine'})`);

    results.push({
      query,
      aIsFirst,
      aScore,
      cScore,
      winner: realWinner,
      reason: response.match(/理由[：:]*\s*(.+)/s)?.[1]?.trim()?.substring(0, 200) || '',
    });

    // 防止 API 限速
    await new Promise(r => setTimeout(r, 1500));
  }

  // ========== 汇总 ==========
  console.log('\n\n' + '═'.repeat(70));
  console.log('  盲评汇总');
  console.log('═'.repeat(70));

  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  console.log(`\n  评测 query 数: ${results.length}`);
  console.log(`  cosine 平均分:  ${avg(results.map(r => r.aScore)).toFixed(1)}/5`);
  console.log(`  AD-Rank 平均分: ${avg(results.map(r => r.cScore)).toFixed(1)}/5`);
  console.log(`\n  胜负: cosine ${aWins} : ${cWins} AD-Rank (平局 ${ties})`);

  if (cWins > aWins) {
    console.log(`\n  ✅ AD-Rank 在 DeepSeek 盲评中胜出`);
  } else if (aWins > cWins) {
    console.log(`\n  ❌ cosine 在 DeepSeek 盲评中胜出（AD-Rank 排序质量不如基线）`);
  } else {
    console.log(`\n  🟰 两者持平`);
  }

  console.log('\n  逐 query 结果:');
  for (const r of results) {
    const marker = r.winner === 'AD-Rank' ? '✅' : r.winner === 'cosine' ? '❌' : '🟰';
    console.log(`    ${marker} "${r.query}" → cosine ${r.aScore}/5 vs AD-Rank ${r.cScore}/5 → ${r.winner}`);
  }

  // 保存
  const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_blind_eval.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    judge: 'DeepSeek V3.2 (deepseek-chat)',
    params: BEST_PARAMS,
    results,
    summary: {
      avgCosineScore: +avg(results.map(r => r.aScore)).toFixed(1),
      avgADRankScore: +avg(results.map(r => r.cScore)).toFixed(1),
      cosineWins: aWins,
      adrankWins: cWins,
      ties,
    },
  }, null, 2));
  console.log(`\n  📁 结果已保存: ${reportPath}`);
  console.log('\n✅ 盲评实验完成');
})();
