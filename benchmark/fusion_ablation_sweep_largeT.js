#!/usr/bin/env node
'use strict';
/**
 * fusion_ablation_sweep.js — Reviewer 2 三问 ablation sweep
 *
 * 实现：
 *   (i)   Graph Laplacian 迭代次数 sweep T ∈ {1, 3, 5, 7, 10, 15, 20}  (alpha=0.15 固定)
 *   (ii)  互补性度量: Kendall τ, Spearman ρ, Jaccard@{5,10,20}, 失败案例 complementarity
 *   (iii) λ sweep ∈ {0.0, 0.1, ..., 1.0} on NFCorpus + ArguAna
 *
 * 用法:
 *   node --max-old-space-size=32768 fusion_ablation_sweep.js [--datasets nfcorpus,arguana] [--max_queries 0]
 *
 * 输出:
 *   /home/amd/HEZIMENG/Shape-CFD/benchmark/data/results/fusion_ablation_results.json
 *   每个 dataset 落盘 JSON Lines: fusion_ablation_<dataset>.jsonl  (中途 crash 不全丢)
 *
 * 依赖: law-vexus napi 模块 (复用现有 shapeLaplacianPipeline + tokenChamferTwoStage + cosineRank)
 */
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// ---------- argparse ----------
function parseArgs() {
  const args = { datasets: ['nfcorpus', 'arguana'], max_queries: 0,
                 output_dir: '/home/amd/HEZIMENG/Shape-CFD/benchmark/data/results',
                 top_n: 55, K: 10 };
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a === '--datasets') args.datasets = process.argv[++i].split(',').map(s => s.trim());
    else if (a === '--max_queries') args.max_queries = parseInt(process.argv[++i]);
    else if (a === '--output_dir') args.output_dir = process.argv[++i];
    else if (a === '--top_n') args.top_n = parseInt(process.argv[++i]);
    else if (a === '--K') args.K = parseInt(process.argv[++i]);
    else if (a === '--help' || a === '-h') {
      console.log('Usage: node fusion_ablation_sweep.js [--datasets nfcorpus,arguana] [--max_queries 0] [--output_dir ...] [--top_n 55] [--K 10]');
      process.exit(0);
    }
  }
  return args;
}

// ---------- helpers ----------
function loadJsonl(fp) {
  return new Promise((res, rej) => {
    const arr = [];
    const r = readline.createInterface({ input: fs.createReadStream(fp), crlfDelay: Infinity });
    r.on('line', l => { if (l.trim()) try { arr.push(JSON.parse(l)); } catch (_) {} });
    r.on('close', () => res(arr));
    r.on('error', rej);
  });
}

function computeNDCG(ranked, qrel, k = 10) {
  let dcg = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++) {
    const r = qrel[ranked[i]] || 0;
    dcg += (Math.pow(2, r) - 1) / Math.log2(i + 2);
  }
  const ideal = Object.values(qrel).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(ideal.length, k); i++) {
    idcg += (Math.pow(2, ideal[i]) - 1) / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

function normalizeScores(scoreMap) {
  const vs = Object.values(scoreMap);
  if (!vs.length) return {};
  const mn = Math.min(...vs), mx = Math.max(...vs), r = mx - mn || 1e-8;
  const out = {};
  for (const [k, v] of Object.entries(scoreMap)) out[k] = (v - mn) / r;
  return out;
}

// Kendall τ-b (handles ties), inputs: ranking arrays (lists of doc ids in order)
// We compute τ between two score-vectors over common docs.
function kendallTau(scoresA, scoresB) {
  const keys = Object.keys(scoresA).filter(k => k in scoresB);
  const n = keys.length;
  if (n < 2) return NaN;
  let concordant = 0, discordant = 0, tieA = 0, tieB = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dA = scoresA[keys[i]] - scoresA[keys[j]];
      const dB = scoresB[keys[i]] - scoresB[keys[j]];
      if (dA === 0 && dB === 0) continue;
      if (dA === 0) { tieA++; continue; }
      if (dB === 0) { tieB++; continue; }
      if (Math.sign(dA) === Math.sign(dB)) concordant++; else discordant++;
    }
  }
  const denom = Math.sqrt((concordant + discordant + tieA) * (concordant + discordant + tieB));
  return denom === 0 ? NaN : (concordant - discordant) / denom;
}

// Spearman ρ over common docs
function spearmanRho(scoresA, scoresB) {
  const keys = Object.keys(scoresA).filter(k => k in scoresB);
  const n = keys.length;
  if (n < 2) return NaN;
  function rank(arr) {
    const indexed = arr.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    const r = new Array(arr.length);
    for (let i = 0; i < indexed.length; i++) r[indexed[i].i] = i + 1;
    return r;
  }
  const aArr = keys.map(k => scoresA[k]);
  const bArr = keys.map(k => scoresB[k]);
  const ra = rank(aArr), rb = rank(bArr);
  const meanA = ra.reduce((a, b) => a + b, 0) / n;
  const meanB = rb.reduce((a, b) => a + b, 0) / n;
  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < n; i++) {
    const da = ra[i] - meanA, db = rb[i] - meanB;
    num += da * db; denA += da * da; denB += db * db;
  }
  return denA === 0 || denB === 0 ? NaN : num / Math.sqrt(denA * denB);
}

// Jaccard@k between two ranked lists (top-k sets)
function jaccardAtK(rankedA, rankedB, k) {
  const sA = new Set(rankedA.slice(0, k));
  const sB = new Set(rankedB.slice(0, k));
  const inter = [...sA].filter(x => sB.has(x)).length;
  const uni = new Set([...sA, ...sB]).size;
  return uni === 0 ? 0 : inter / uni;
}

// 失败案例互补性: 当 A 完全失败 (NDCG=0) 时，B 是否成功 (NDCG>0) ？
function rescuePair(ndcgA, ndcgB) {
  return { aFails: ndcgA === 0, bRescues: ndcgA === 0 && ndcgB > 0 };
}

function ndcgFromScoreMap(scoreMap, qrel, k) {
  const ranked = Object.entries(scoreMap)
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(x => x[0]);
  return computeNDCG(ranked, qrel, k);
}

function topKFromMap(scoreMap, k) {
  return Object.entries(scoreMap)
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(x => x[0]);
}

// 加权融合: λ * laplacian + (1-λ) * token_2stage  (注意：manuscript 是 0.7*token + 0.3*lap，
// 所以 λ=0.3 对应 manuscript 设置；λ=0 即纯 token；λ=1 即纯 lap)
function fuseScores(lapMap, tokMap, lambda) {
  const all = new Set([...Object.keys(lapMap), ...Object.keys(tokMap)]);
  const nl = normalizeScores(lapMap), nt = normalizeScores(tokMap);
  const out = {};
  for (const d of all) out[d] = lambda * (nl[d] || 0) + (1 - lambda) * (nt[d] || 0);
  return out;
}

// ---------- main per dataset ----------
async function runDataset(dataset, args) {
  const DATA_DIR = path.join('/home/amd/HEZIMENG/Shape-CFD/benchmark/data/beir_data', dataset);
  const TOP_N = args.top_n;
  const K = args.K;

  const tokenCloudsPath = path.join(DATA_DIR, 'token_clouds.sqlite');
  const queryTokenCloudsPath = path.join(DATA_DIR, 'query_token_clouds.sqlite');
  const cloudsPath = path.join(DATA_DIR, 'clouds.sqlite');
  const idMapPath = path.join(DATA_DIR, 'id_map.json');
  const queryVecPath = path.join(DATA_DIR, 'query_vectors.jsonl');
  const queriesPath = path.join(DATA_DIR, 'queries.jsonl');
  const qrelsPath = path.join(DATA_DIR, 'qrels.tsv');

  for (const p of [tokenCloudsPath, queryTokenCloudsPath, cloudsPath, idMapPath, queryVecPath, queriesPath, qrelsPath]) {
    if (!fs.existsSync(p)) {
      console.error(`[${dataset}] MISSING: ${p}, skip dataset`);
      return null;
    }
  }

  console.log(`\n=== ${dataset.toUpperCase()} fusion ablation sweep ===`);

  const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
  const v = new LawVexus(`/tmp/${dataset}_fusion_sweep`);

  process.stdout.write(`[${dataset}] Loading sentence clouds... `);
  v.loadClouds(cloudsPath);
  console.log('done');

  process.stdout.write(`[${dataset}] Loading token clouds... `);
  v.loadTokenCloudsSqlite(tokenCloudsPath, queryTokenCloudsPath);
  console.log('done');

  // qrels
  const qrels = {};
  const ql = fs.readFileSync(qrelsPath, 'utf-8').trim().split('\n');
  for (let i = 1; i < ql.length; i++) {
    const p = ql[i].split('\t');
    if (!qrels[p[0]]) qrels[p[0]] = {};
    qrels[p[0]][p[1]] = parseInt(p[2]);
  }

  // id map (string id → int id)
  const idMap = JSON.parse(fs.readFileSync(idMapPath, 'utf-8'));
  const rev = {};
  for (const [s, i] of Object.entries(idMap)) rev[i] = s;

  // query vectors
  const queryVecs = {};
  for (const o of await loadJsonl(queryVecPath)) {
    queryVecs[o._id] = new Float32Array(o.vector);
  }

  // query id → file_id mapping (for token-level)
  const queryIdToFileId = {};
  const queriesRaw = await loadJsonl(queriesPath);
  for (let i = 0; i < queriesRaw.length; i++) {
    queryIdToFileId[queriesRaw[i]._id] = i;
  }

  let qids = Object.keys(qrels).filter(q => queryVecs[q] && queryIdToFileId[q] !== undefined);
  if (args.max_queries > 0) qids = qids.slice(0, args.max_queries);
  console.log(`[${dataset}] ${qids.length} queries\n`);

  // ===== Sweep 配置 =====
  const T_VALUES = [25, 50, 100, 200];                  // (i) Laplacian iterations
  const LAMBDA_VALUES = [];                                    // (iii) λ ∈ {0.0, 0.1, ..., 1.0}
  for (let l = 0; l <= 10; l++) LAMBDA_VALUES.push(l / 10);
  const ALPHA_FIXED = 0.15;                                    // 论文设置

  // 累加器
  const metrics = {
    cosine: { sum: 0, n: 0 },
    token_2stage: { sum: 0, n: 0 },
  };
  // 每个 T: laplacian NDCG 的累加
  for (const T of T_VALUES) metrics[`laplacian_T${T}`] = { sum: 0, n: 0 };
  // 每个 (T=5, λ): fusion NDCG 累加 (主 sweep)
  for (const lam of LAMBDA_VALUES) {
    metrics[`fusion_T5_lam${lam.toFixed(1)}`] = { sum: 0, n: 0 };
  }

  // 互补性度量累加 (Lap T=5 vs Token)
  const compl = {
    kendall_tau: [],   // per-query τ
    spearman_rho: [],
    jaccard_5: [], jaccard_10: [], jaccard_20: [],
    lap_only_success: 0,    // lap 成功 (>0) 但 token 失败 (=0)
    token_only_success: 0,  // token 成功但 lap 失败
    both_success: 0,
    both_fail: 0,
  };

  // per-query 输出 (供深度分析用，落盘到 jsonl)
  const perQOut = path.join(args.output_dir, `fusion_ablation_${dataset}_perquery.jsonl`);
  fs.mkdirSync(args.output_dir, { recursive: true });
  const perQStream = fs.createWriteStream(perQOut, { flags: 'w' });

  const t0 = Date.now();
  const step = Math.max(1, Math.floor(qids.length / 20));

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qVec = queryVecs[qid];
    const qrel = qrels[qid];
    const qBuf = Buffer.from(qVec.buffer, qVec.byteOffset, qVec.byteLength);
    const qFileId = queryIdToFileId[qid];

    const perQ = { qid, ndcg: {} };

    // cosine baseline
    try {
      const h = v.cosineRank(qBuf, K);
      const ranked = h.map(x => rev[x.id]).filter(Boolean);
      const ndcg = computeNDCG(ranked, qrel, K);
      metrics.cosine.sum += ndcg; metrics.cosine.n++;
      perQ.ndcg.cosine = ndcg;
    } catch (e) { if (qi === 0) console.error('cosine err:', e.message); }

    // token_2stage
    let tokMap = null;
    try {
      const h = v.tokenChamferTwoStage(qFileId, 100, TOP_N);
      tokMap = {};
      for (const [intId, sc] of h) {
        const s = rev[intId]; if (s) tokMap[s] = sc;
      }
      const tokNdcg = ndcgFromScoreMap(tokMap, qrel, K);
      metrics.token_2stage.sum += tokNdcg; metrics.token_2stage.n++;
      perQ.ndcg.token_2stage = tokNdcg;
    } catch (e) { if (qi === 0) console.error('token err:', e.message); }

    // laplacian sweep over T
    const lapMaps = {};   // T → score map
    for (const T of T_VALUES) {
      try {
        const h = v.shapeLaplacianPipeline(qBuf, K, TOP_N, ALPHA_FIXED, T);
        const m = {};
        for (const x of h) { const s = rev[x.id]; if (s) m[s] = x.score; }
        lapMaps[T] = m;
        const ndcg = ndcgFromScoreMap(m, qrel, K);
        metrics[`laplacian_T${T}`].sum += ndcg;
        metrics[`laplacian_T${T}`].n++;
        perQ.ndcg[`lap_T${T}`] = ndcg;
      } catch (e) { if (qi === 0) console.error(`lap T=${T} err:`, e.message); }
    }

    // λ sweep with T=5 (论文设置)
    if (tokMap && lapMaps[5]) {
      for (const lam of LAMBDA_VALUES) {
        const fused = fuseScores(lapMaps[5], tokMap, lam);
        const ndcg = ndcgFromScoreMap(fused, qrel, K);
        const key = `fusion_T5_lam${lam.toFixed(1)}`;
        metrics[key].sum += ndcg;
        metrics[key].n++;
        perQ.ndcg[key] = ndcg;
      }
    }

    // 互补性 (Lap T=5 vs Token)
    if (tokMap && lapMaps[5]) {
      const tau = kendallTau(lapMaps[5], tokMap);
      const rho = spearmanRho(lapMaps[5], tokMap);
      if (!isNaN(tau)) compl.kendall_tau.push(tau);
      if (!isNaN(rho)) compl.spearman_rho.push(rho);
      const lapRanked = topKFromMap(lapMaps[5], 20);
      const tokRanked = topKFromMap(tokMap, 20);
      compl.jaccard_5.push(jaccardAtK(lapRanked, tokRanked, 5));
      compl.jaccard_10.push(jaccardAtK(lapRanked, tokRanked, 10));
      compl.jaccard_20.push(jaccardAtK(lapRanked, tokRanked, 20));

      const lapNdcg = perQ.ndcg.lap_T5 || 0;
      const tokNdcg = perQ.ndcg.token_2stage || 0;
      const lapOK = lapNdcg > 0, tokOK = tokNdcg > 0;
      if (lapOK && !tokOK) compl.lap_only_success++;
      else if (!lapOK && tokOK) compl.token_only_success++;
      else if (lapOK && tokOK) compl.both_success++;
      else compl.both_fail++;

      perQ.kendall_tau = tau;
      perQ.spearman_rho = rho;
    }

    perQStream.write(JSON.stringify(perQ) + '\n');

    if ((qi + 1) % step === 0 || qi === qids.length - 1) {
      const dt = (Date.now() - t0) / 1000;
      const eta = dt / (qi + 1) * (qids.length - qi - 1);
      process.stdout.write(`\r  [${dataset}] ${qi + 1}/${qids.length} (${dt.toFixed(0)}s, ETA ${eta.toFixed(0)}s)`);
    }
  }
  console.log('');
  perQStream.end();

  // ===== 汇总 =====
  const summary = { dataset, n_queries: qids.length, alpha: ALPHA_FIXED, top_n: TOP_N, K };

  function avg(name) {
    const m = metrics[name];
    return m.n > 0 ? m.sum / m.n : null;
  }
  function mean(arr) { return arr.length === 0 ? null : arr.reduce((a, b) => a + b, 0) / arr.length; }

  summary.cosine = avg('cosine');
  summary.token_2stage = avg('token_2stage');
  summary.laplacian_T_sweep = {};
  for (const T of T_VALUES) summary.laplacian_T_sweep[T] = avg(`laplacian_T${T}`);
  summary.fusion_T5_lambda_sweep = {};
  for (const lam of LAMBDA_VALUES) {
    summary.fusion_T5_lambda_sweep[lam.toFixed(1)] = avg(`fusion_T5_lam${lam.toFixed(1)}`);
  }
  summary.complementarity = {
    n_pairs: compl.kendall_tau.length,
    kendall_tau_mean: mean(compl.kendall_tau),
    spearman_rho_mean: mean(compl.spearman_rho),
    jaccard5_mean: mean(compl.jaccard_5),
    jaccard10_mean: mean(compl.jaccard_10),
    jaccard20_mean: mean(compl.jaccard_20),
    lap_only_success: compl.lap_only_success,
    token_only_success: compl.token_only_success,
    both_success: compl.both_success,
    both_fail: compl.both_fail,
    rescue_rate_lap_to_token: compl.both_success + compl.token_only_success > 0
      ? compl.lap_only_success / (compl.lap_only_success + compl.both_fail + 1e-9) : null,
  };

  console.log(`\n[${dataset}] === Summary ===`);
  console.log(`  cosine:        ${summary.cosine?.toFixed(4)}`);
  console.log(`  token_2stage:  ${summary.token_2stage?.toFixed(4)}`);
  console.log(`  Laplacian T sweep:`);
  for (const T of T_VALUES) console.log(`    T=${T}:   ${summary.laplacian_T_sweep[T]?.toFixed(4)}`);
  console.log(`  Fusion λ sweep (T=5):`);
  for (const lam of LAMBDA_VALUES) console.log(`    λ=${lam.toFixed(1)}: ${summary.fusion_T5_lambda_sweep[lam.toFixed(1)]?.toFixed(4)}`);
  console.log(`  Complementarity (Lap T=5 vs Token):`);
  console.log(`    Kendall τ mean:  ${summary.complementarity.kendall_tau_mean?.toFixed(4)}`);
  console.log(`    Spearman ρ mean: ${summary.complementarity.spearman_rho_mean?.toFixed(4)}`);
  console.log(`    Jaccard@5/10/20: ${summary.complementarity.jaccard5_mean?.toFixed(4)}/${summary.complementarity.jaccard10_mean?.toFixed(4)}/${summary.complementarity.jaccard20_mean?.toFixed(4)}`);
  console.log(`    failure modes: lap_only=${compl.lap_only_success}, tok_only=${compl.token_only_success}, both_succ=${compl.both_success}, both_fail=${compl.both_fail}`);

  return summary;
}

// ---------- main ----------
(async () => {
  const args = parseArgs();
  fs.mkdirSync(args.output_dir, { recursive: true });

  const allResults = { generated_at: new Date().toISOString(), args, datasets: {} };
  const outFinal = path.join(args.output_dir, 'fusion_ablation_results.json');
  // 中途落盘文件（每完成一个 dataset 即写）
  const outStream = path.join(args.output_dir, 'fusion_ablation_results.jsonl');
  const stream = fs.createWriteStream(outStream, { flags: 'w' });

  for (const ds of args.datasets) {
    try {
      const summary = await runDataset(ds, args);
      if (summary) {
        allResults.datasets[ds] = summary;
        stream.write(JSON.stringify(summary) + '\n');
        // 每个 dataset 完成立刻 dump 完整 JSON
        fs.writeFileSync(outFinal, JSON.stringify(allResults, null, 2));
      }
    } catch (e) {
      console.error(`[${ds}] FATAL:`, e.message, e.stack);
      allResults.datasets[ds] = { error: e.message };
      fs.writeFileSync(outFinal, JSON.stringify(allResults, null, 2));
    }
  }
  stream.end();

  console.log('\n=== ALL DONE ===');
  console.log(`  Final result: ${outFinal}`);
})();
