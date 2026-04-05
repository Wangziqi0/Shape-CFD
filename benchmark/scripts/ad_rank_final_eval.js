#!/usr/bin/env node
'use strict';

/**
 * AD-Rank 全策略横评 — Agent 11
 *
 * 8 组对比 + DeepSeek 盲评：
 *   组 A: cosine (基线)
 *   组 B: AD-Rank v2 (D=0.15, u=0.1, knn=3)
 *   组 C: MDA (M=8, k=32, u=0.3)
 *   组 D: BAA (B=8, u=0.3)
 *   组 E: v2 + PCA (pcaDim=32)
 *   组 F: v2 + 温度缩放 (tempAlpha=0.5)
 *   组 G: v2 + 自适应参数 (autoD + autoU)
 *   组 H: Shape CFD (Chamfer 距离 + 质心对流) [可选, --skip-shape 跳过]
 *
 * 评测维度：DeepSeek 盲评分(40%)、对流贡献率(20%)、速度(20%)、收敛率(10%)、Pe(10%)
 * 盲评设计：每个 query 做 A vs X (X=B-H) 的成对盲评
 *
 * Usage:
 *   node ad_rank_final_eval.js               # 完整 8 组
 *   node ad_rank_final_eval.js --skip-shape  # 跳过 Shape CFD (节省时间)
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
const SKIP_SHAPE = process.argv.includes('--skip-shape');

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

// 各组参数配置
const METHOD_CONFIGS = {
  B: { name: 'AD-Rank v2',  params: { D: 0.15, uStrength: 0.1, knn: 3, maxIter: 50, epsilon: 1e-3 } },
  C: { name: 'MDA',         type: 'mda', params: { M: 8, k: 32, D: 0.15, uStrength: 0.3, knn: 3, maxIter: 50, epsilon: 1e-3 } },
  D: { name: 'BAA',         type: 'baa', params: { B: 8, D: 0.15, uStrength: 0.3, knn: 3, maxIter: 50, epsilon: 1e-3 } },
  E: { name: 'v2+PCA',      params: { D: 0.15, uStrength: 0.1, knn: 3, maxIter: 50, epsilon: 1e-3, pcaDim: 32 } },
  F: { name: 'v2+温度',     params: { D: 0.15, uStrength: 0.1, knn: 3, maxIter: 50, epsilon: 1e-3, tempAlpha: 0.5 } },
  G: { name: 'v2+自适应',   params: { D: 0.15, uStrength: 0.1, knn: 3, maxIter: 50, epsilon: 1e-3, autoD: true, autoU: true } },
};

const PRE_FILTER_K = 30;

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
          if (json.error) {
            reject(new Error(json.error.message || JSON.stringify(json.error)));
            return;
          }
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

/**
 * 成对盲评: A(cosine) vs X(target)
 */
async function blindEval(query, groupA, groupX, labelX) {
  if (!DEEPSEEK_API_KEY) return null;

  const aIsFirst = Math.random() > 0.5;
  const first = aIsFirst ? groupA : groupX;
  const second = aIsFirst ? groupX : groupA;

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
    const xScore = aIsFirst ? secondScore : firstScore;

    let realWinner;
    if (winner === '平局') realWinner = '平局';
    else if ((winner === '方案甲' && aIsFirst) || (winner === '方案乙' && !aIsFirst)) realWinner = 'cosine';
    else realWinner = labelX;

    const reason = response.match(/理由[：:]*\s*(.+)/s)?.[1]?.trim()?.substring(0, 300) || '';

    return { aScore, xScore, winner: realWinner, reason };
  } catch (e) {
    console.error(`    ❌ DeepSeek 盲评失败 (${labelX}): ${e.message}`);
    return null;
  }
}

/**
 * 格式化排序结果供盲评使用
 */
function formatRanking(result, metas, maxK = 5) {
  return result.rankings.slice(0, maxK).map((r, i) => ({
    rank: i + 1,
    law: metas[r.index]?.law || '?',
    article: metas[r.index]?.article || '',
    content: (metas[r.index]?.content || '').substring(0, 200),
  }));
}

// ========== 主流程 ==========
(async () => {
  try {
    console.log('═'.repeat(70));
    console.log('  AD-Rank 全策略横评 — Agent 11 (完整 8 组)');
    const activeGroups = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
    if (!SKIP_SHAPE) activeGroups.push('H');
    console.log(`  参赛组: ${activeGroups.join(', ')}${SKIP_SHAPE ? ' (已跳过 Shape)' : ''}`);
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

    const apiKey = loadApiKey();
    const engine = new VectorizeEngine(apiKey);
    const adData = new ADRankData();
    await adData.initialize();

    // Shape CFD 模块 (可选)
    let adRankShape, buildPointClouds, buildQueryCloud;
    if (!SKIP_SHAPE) {
      ({ adRankShape, buildPointClouds, buildQueryCloud } = require('./ad_rank_shape'));
    }

    console.log(`  索引: ${index.size()} 条向量`);
    console.log(`  DeepSeek API: ✅`);

    const results = [];
    // 各组累计指标
    const methodStats = {};
    for (const key of activeGroups) {
      if (key !== 'A') methodStats[key] = { scores: [], peList: [], speedList: [], convList: [], wins: 0, losses: 0, ties: 0 };
    }

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

      // ══ 组 A: cosine Top-5 ══
      const tA = performance.now();
      const aResults = index.search(qVec, 5);
      const tA_ms = performance.now() - tA;
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

      // ══ 获取候选向量 (B-G 共用) ══
      const { vectors: candVecs, metadata: candMetas } = await adData.getCandidates(qVec, PRE_FILTER_K);
      if (candVecs.length === 0) { console.error('  ❌ 无候选'); continue; }

      const queryResult = {
        query,
        timing: { A: +tA_ms.toFixed(1) },
        physics: {},
        blindEval: {},
      };

      // ══ 组 B: AD-Rank v2 ══
      const tB = performance.now();
      const bResult = adRank(qVec, candVecs, 5, METHOD_CONFIGS.B.params);
      const tB_ms = performance.now() - tB;
      const groupB = formatRanking(bResult, candMetas);
      queryResult.timing.B = +tB_ms.toFixed(1);
      queryResult.physics.B = { re: +bResult.reynolds.toFixed(3), pe: +bResult.peclet.toFixed(3), conv: bResult.convergence, iter: bResult.iterations };
      methodStats.B.peList.push(bResult.peclet);
      methodStats.B.speedList.push(tB_ms);
      methodStats.B.convList.push(bResult.convergence);

      // ══ 组 C: MDA ══
      const tC = performance.now();
      const cResult = adRankMDA(qVec, candVecs, 5, METHOD_CONFIGS.C.params);
      const tC_ms = performance.now() - tC;
      const groupC = formatRanking(cResult, candMetas);
      queryResult.timing.C = +tC_ms.toFixed(1);
      queryResult.physics.C = { re: +cResult.reynolds.toFixed(3), pe: +cResult.peclet.toFixed(3), conv: cResult.convergence, iter: cResult.iterations, advContrib: +(cResult.advectionContribution * 100).toFixed(1) };
      methodStats.C.peList.push(cResult.peclet);
      methodStats.C.speedList.push(tC_ms);
      methodStats.C.convList.push(cResult.convergence);

      // ══ 组 D: BAA ══
      const tD = performance.now();
      const dResult = adRankBAA(qVec, candVecs, 5, METHOD_CONFIGS.D.params);
      const tD_ms = performance.now() - tD;
      const groupD = formatRanking(dResult, candMetas);
      queryResult.timing.D = +tD_ms.toFixed(1);
      queryResult.physics.D = { re: +dResult.reynolds.toFixed(3), pe: +dResult.peclet.toFixed(3), conv: dResult.convergence, iter: dResult.iterations, advContrib: +(dResult.advectionContribution * 100).toFixed(1) };
      methodStats.D.peList.push(dResult.peclet);
      methodStats.D.speedList.push(tD_ms);
      methodStats.D.convList.push(dResult.convergence);

      // ══ 组 E: v2 + PCA ══
      const tE = performance.now();
      const eResult = adRank(qVec, candVecs, 5, METHOD_CONFIGS.E.params);
      const tE_ms = performance.now() - tE;
      const groupE = formatRanking(eResult, candMetas);
      queryResult.timing.E = +tE_ms.toFixed(1);
      queryResult.physics.E = { re: +eResult.reynolds.toFixed(3), pe: +eResult.peclet.toFixed(3), conv: eResult.convergence, iter: eResult.iterations };
      methodStats.E.peList.push(eResult.peclet);
      methodStats.E.speedList.push(tE_ms);
      methodStats.E.convList.push(eResult.convergence);

      // ══ 组 F: v2 + 温度缩放 ══
      const tF = performance.now();
      const fResult = adRank(qVec, candVecs, 5, METHOD_CONFIGS.F.params);
      const tF_ms = performance.now() - tF;
      const groupF = formatRanking(fResult, candMetas);
      queryResult.timing.F = +tF_ms.toFixed(1);
      queryResult.physics.F = { re: +fResult.reynolds.toFixed(3), pe: +fResult.peclet.toFixed(3), conv: fResult.convergence, iter: fResult.iterations };
      methodStats.F.peList.push(fResult.peclet);
      methodStats.F.speedList.push(tF_ms);
      methodStats.F.convList.push(fResult.convergence);

      // ══ 组 G: v2 + 自适应参数 ══
      const tG = performance.now();
      const gResult = adRank(qVec, candVecs, 5, METHOD_CONFIGS.G.params);
      const tG_ms = performance.now() - tG;
      const groupG = formatRanking(gResult, candMetas);
      queryResult.timing.G = +tG_ms.toFixed(1);
      queryResult.physics.G = { re: +gResult.reynolds.toFixed(3), pe: +gResult.peclet.toFixed(3), conv: gResult.convergence, iter: gResult.iterations, autoParams: gResult.autoParams };
      methodStats.G.peList.push(gResult.peclet);
      methodStats.G.speedList.push(tG_ms);
      methodStats.G.convList.push(gResult.convergence);

      // ══ 组 H: Shape CFD (可选) ══
      let groupH = null;
      if (!SKIP_SHAPE) {
        try {
          const tH = performance.now();
          const queryCloud = await buildQueryCloud(query, engine);
          const docClouds = await buildPointClouds(candMetas, engine);
          const tH_embed = performance.now() - tH;
          const tH_solve = performance.now();
          const hResult = adRankShape(queryCloud, docClouds, 5, {
            D: 0.15, uStrength: 0.3, knn: 3, maxIter: 50, epsilon: 1e-3,
          });
          const tH_solve_ms = performance.now() - tH_solve;
          const tH_total = performance.now() - tH;
          groupH = formatRanking(hResult, candMetas);
          queryResult.timing.H = +tH_total.toFixed(1);
          queryResult.timing.H_embed = +tH_embed.toFixed(0);
          queryResult.timing.H_solve = +tH_solve_ms.toFixed(1);
          queryResult.physics.H = { re: +hResult.reynolds.toFixed(3), pe: +hResult.peclet.toFixed(3), conv: hResult.convergence, iter: hResult.iterations };
          methodStats.H.peList.push(hResult.peclet);
          methodStats.H.speedList.push(tH_total);
          methodStats.H.convList.push(hResult.convergence);
        } catch (e) {
          console.error(`  ❌ Shape CFD 失败: ${e.message}`);
        }
      }

      // 速度概要
      const speedLine = Object.entries(queryResult.timing)
        .filter(([k]) => !k.includes('_'))
        .map(([k, v]) => `${k}=${v}ms`)
        .join('  ');
      console.log(`\n  ⏱  速度: ${speedLine}`);

      // 物理指标概要
      for (const [key, phys] of Object.entries(queryResult.physics)) {
        console.log(`  🌊 ${key}: Re=${phys.re} Pe=${phys.pe} iter=${phys.iter} conv=${phys.conv}${phys.advContrib ? ' adv=' + phys.advContrib + '%' : ''}`);
      }

      // ══ DeepSeek 盲评: A vs B/C/D/E/F/G/H ══
      const groups = { B: groupB, C: groupC, D: groupD, E: groupE, F: groupF, G: groupG };
      if (groupH) groups.H = groupH;

      const groupLabels = { B: 'AD-Rank v2', C: 'MDA', D: 'BAA', E: 'v2+PCA', F: 'v2+温度', G: 'v2+自适应', H: 'Shape' };

      for (const [key, group] of Object.entries(groups)) {
        const label = groupLabels[key];
        console.log(`  🤖 盲评 A vs ${key}(${label})...`);
        const evalResult = await blindEval(query, groupA, group, label);

        if (evalResult) {
          queryResult.blindEval[key] = {
            cosine: evalResult.aScore,
            target: evalResult.xScore,
            winner: evalResult.winner,
            reason: evalResult.reason,
          };
          methodStats[key].scores.push(evalResult.xScore);
          if (evalResult.winner === label) methodStats[key].wins++;
          else if (evalResult.winner === 'cosine') methodStats[key].losses++;
          else methodStats[key].ties++;

          console.log(`     cosine=${evalResult.aScore}/5 ${label}=${evalResult.xScore}/5 胜出=${evalResult.winner}`);
        }

        // API 限速保护
        await new Promise(r => setTimeout(r, 1200));
      }

      results.push(queryResult);
      await new Promise(r => setTimeout(r, 300));
    }

    // ══════════════════════════════════════
    // 汇总
    // ══════════════════════════════════════
    console.log('\n\n' + '═'.repeat(70));
    console.log('  全策略横评汇总');
    console.log('═'.repeat(70));

    const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    // 综合排名表
    const rankings = [];
    for (const [key, stats] of Object.entries(methodStats)) {
      const avgScore = avg(stats.scores);
      const avgPe = avg(stats.peList);
      const avgSpeed = avg(stats.speedList);
      const convRate = stats.convList.filter(Boolean).length / stats.convList.length * 100;

      // 综合加权分
      // 盲评分(40%) + 对流贡献率(20%,用Pe近似) + 速度(20%) + 收敛率(10%) + Pe(10%)
      const speedScore = avgSpeed <= 20 ? 1.0 : avgSpeed <= 50 ? 0.8 : avgSpeed <= 200 ? 0.4 : 0.1;
      const convScore = convRate / 100;
      const peScore = Math.min(avgPe / 0.5, 1.0);
      const composite = avgScore / 5 * 0.4 + peScore * 0.2 + speedScore * 0.2 + convScore * 0.1 + peScore * 0.1;

      rankings.push({
        key,
        name: key === 'H' ? 'Shape CFD' : (METHOD_CONFIGS[key]?.name || key),
        avgScore: +avgScore.toFixed(2),
        avgPe: +avgPe.toFixed(3),
        avgSpeed: +avgSpeed.toFixed(1),
        convRate: +convRate.toFixed(0),
        composite: +composite.toFixed(3),
        wins: stats.wins,
        losses: stats.losses,
        ties: stats.ties,
      });
    }

    rankings.sort((a, b) => b.composite - a.composite);

    console.log('\n  📊 综合排名:');
    console.log('  ┌─────┬────────────────┬──────────┬────────┬──────────┬──────────┬───────────┬─────────────────┐');
    console.log('  │Rank │ 方案           │ 盲评分   │ Pe     │ 速度(ms) │ 收敛率   │ 综合分   │ 胜/负/平         │');
    console.log('  ├─────┼────────────────┼──────────┼────────┼──────────┼──────────┼───────────┼─────────────────┤');
    for (let i = 0; i < rankings.length; i++) {
      const r = rankings[i];
      const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '  ';
      console.log(`  │${medal}${String(i+1).padStart(2)} │ ${r.name.padEnd(14)} │ ${String(r.avgScore + '/5').padStart(8)} │ ${String(r.avgPe).padStart(6)} │ ${String(r.avgSpeed).padStart(8)} │ ${String(r.convRate + '%').padStart(8)} │ ${String(r.composite).padStart(9)} │ ${r.wins}/${r.losses}/${r.ties}`.padEnd(17) + '│');
    }
    console.log('  └─────┴────────────────┴──────────┴────────┴──────────┴──────────┴───────────┴─────────────────┘');

    const winner = rankings[0];
    console.log(`\n  🏆 获胜方案: ${winner.name} (综合分=${winner.composite}, 盲评=${winner.avgScore}/5, Pe=${winner.avgPe}, 速度=${winner.avgSpeed}ms)`);

    // 保存
    const reportPath = path.join(__dirname, 'word', 'cfd创新', 'experiment_final_eval.json');
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      judge: 'DeepSeek V3.2 (deepseek-chat)',
      methodConfigs: METHOD_CONFIGS,
      skipShape: SKIP_SHAPE,
      results,
      rankings,
      winner: winner.name,
      winnerKey: winner.key,
      winnerComposite: winner.composite,
    }, null, 2));
    console.log(`\n  📁 结果已保存: ${reportPath}`);
    console.log('\n✅ 全策略横评完成');

  } catch (err) {
    console.error('\n❌ 实验失败:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
})();
