#!/usr/bin/env node
'use strict';
/**
 * beir_pq_coarse_latency.js -- Coarse filter latency micro-benchmark
 *
 * Measures ONLY the coarse filter stage latency (no fine ranking).
 * Also tests inverted index with n_probe=1 and n_probe=3.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const COARSE_TOP = 200;

function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const arr = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(fp), crlfDelay: Infinity
    });
    rl.on('line', line => { if (line.trim()) try { arr.push(JSON.parse(line)); } catch (_) {} });
    rl.on('close', () => resolve(arr));
    rl.on('error', reject);
  });
}

function computeNDCG(ranked, qrel, k = 10) {
  let dcg = 0;
  for (let i = 0; i < Math.min(ranked.length, k); i++) {
    const rel = qrel[ranked[i]] || 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  const idealRels = Object.values(qrel).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(idealRels.length, k); i++) {
    idcg += (Math.pow(2, idealRels[i]) - 1) / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

async function main() {
  console.log('\n=== Coarse Filter Latency & Recall Micro-Benchmark ===\n');

  const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
  const vexus = new LawVexus('/tmp/pq_lat_bench');

  process.stdout.write('Loading sentence clouds... ');
  vexus.loadClouds(CLOUDS_DB);
  console.log('done');

  process.stdout.write('Loading token clouds... ');
  vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log('done');

  // Load eval data
  const idMap = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'id_map.json'), 'utf-8'));
  const reverseMap = {};
  for (const [s, i] of Object.entries(idMap)) reverseMap[i] = s;

  const qrels = {};
  const ql = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < ql.length; i++) {
    const [qi, di, s] = ql[i].split('\t');
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  const queryVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl')))
    queryVecs[o._id] = new Float32Array(o.vector);

  const queryIdToFileId = {};
  const queriesRaw = await loadJsonl(path.join(DATA_DIR, 'queries.jsonl'));
  for (let i = 0; i < queriesRaw.length; i++) queryIdToFileId[queriesRaw[i]._id] = i;

  let qids = Object.keys(qrels).filter(q => queryVecs[q]);
  const MQ = parseInt(process.env.MAX_Q || '0');
  if (MQ > 0) qids = qids.slice(0, MQ);
  console.log(`${qids.length} queries\n`);

  // Test inverted with n_probe=1 and n_probe=3 for recall comparison
  // Also measure per-method coarse latency
  const stats = {
    centroid: { recall_sum: 0, n: 0, ms: 0 },
    inverted_p1: { recall_sum: 0, n: 0, ms: 0, ndcg_sum: 0 },
    inverted_p3: { recall_sum: 0, n: 0, ms: 0, ndcg_sum: 0 },
    hamming: { recall_sum: 0, n: 0, ms: 0 },
    adc: { recall_sum: 0, n: 0, ms: 0 },
  };

  const TOP_N = 55;

  for (let qi = 0; qi < qids.length; qi++) {
    const qid = qids[qi];
    const qrel = qrels[qid];
    const qFileId = queryIdToFileId[qid];
    const gtDocs = new Set(Object.keys(qrel).filter(d => qrel[d] >= 1));

    const computeRecall = (ids) => {
      const idSet = new Set(ids.map(id => reverseMap[id]).filter(Boolean));
      let hits = 0;
      for (const d of gtDocs) if (idSet.has(d)) hits++;
      return gtDocs.size > 0 ? hits / gtDocs.size : 0;
    };

    // Measure each coarse method individually with timing
    // Centroid
    {
      const t0 = performance.now();
      const r = vexus.coarseRecallCandidates(qFileId, COARSE_TOP, 1);
      const ms = performance.now() - t0;
      stats.centroid.recall_sum += computeRecall(r.centroidIds);
      stats.centroid.ms += ms;
      stats.centroid.n++;
    }

    // Inverted n_probe=1 (full pipeline)
    {
      const t0 = performance.now();
      const h = vexus.tokenInvertedTwoStage(qFileId, COARSE_TOP, TOP_N, 1);
      const ms = performance.now() - t0;
      stats.inverted_p1.ms += ms;
      stats.inverted_p1.n++;
      const ranked = h.map(x => reverseMap[x[0]]).filter(Boolean);
      stats.inverted_p1.ndcg_sum += computeNDCG(ranked, qrel, 10);
    }

    // Inverted n_probe=3
    {
      const t0 = performance.now();
      const h = vexus.tokenInvertedTwoStage(qFileId, COARSE_TOP, TOP_N, 3);
      const ms = performance.now() - t0;
      stats.inverted_p3.ms += ms;
      stats.inverted_p3.n++;
      const ranked = h.map(x => reverseMap[x[0]]).filter(Boolean);
      stats.inverted_p3.ndcg_sum += computeNDCG(ranked, qrel, 10);
    }

    if ((qi + 1) % 32 === 0 || qi === qids.length - 1)
      process.stdout.write(`\r  ${qi + 1}/${qids.length}`);
  }

  console.log('\n');

  // Print
  console.log('='.repeat(70));
  console.log('  Inverted Index n_probe Sensitivity');
  console.log('-'.repeat(70));
  console.log('  Method'.padEnd(30) + 'NDCG@10'.padEnd(12) + 'avg latency');
  console.log('-'.repeat(70));
  const n1 = stats.inverted_p1;
  const n3 = stats.inverted_p3;
  console.log(`  ${'Inverted n_probe=1'.padEnd(30)}${(n1.ndcg_sum / n1.n).toFixed(4).padEnd(12)}${(n1.ms / n1.n).toFixed(1)}ms`);
  console.log(`  ${'Inverted n_probe=3'.padEnd(30)}${(n3.ndcg_sum / n3.n).toFixed(4).padEnd(12)}${(n3.ms / n3.n).toFixed(1)}ms`);
  console.log('='.repeat(70));
}

main().catch(e => { console.error(e); process.exit(1); });
