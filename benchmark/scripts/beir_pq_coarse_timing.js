#!/usr/bin/env node
'use strict';
/**
 * beir_pq_coarse_timing.js -- Individual coarse filter timing
 *
 * Measures total pipeline latency for each scheme separately.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'token_clouds.sqlite');
const QUERY_TOKEN_CLOUDS_DB = path.join(DATA_DIR, 'query_token_clouds.sqlite');
const CLOUDS_DB = path.join(DATA_DIR, 'clouds.sqlite');
const COARSE_TOP = 200;
const TOP_N = 55;

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

async function main() {
  console.log('\n=== Per-Scheme Timing Benchmark ===\n');

  const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
  const vexus = new LawVexus('/tmp/pq_timing_bench');

  process.stdout.write('Loading... ');
  vexus.loadClouds(CLOUDS_DB);
  vexus.loadTokenCloudsSqlite(TOKEN_CLOUDS_DB, QUERY_TOKEN_CLOUDS_DB);
  console.log('done');

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

  // Warm up (first call is slow)
  const warmId = queryIdToFileId[qids[0]];
  vexus.tokenChamferTwoStage(warmId, COARSE_TOP, TOP_N);
  vexus.pqHammingTwoStage(warmId, COARSE_TOP, TOP_N);
  vexus.tokenInvertedTwoStage(warmId, COARSE_TOP, TOP_N, 1);
  vexus.adcCoarseF32Fine(warmId, COARSE_TOP, TOP_N);

  // Measure each method's pipeline latency separately
  const methods = [
    { name: 'Centroid->f32 (baseline)', fn: (qfid) => vexus.tokenChamferTwoStage(qfid, COARSE_TOP, TOP_N) },
    { name: 'Hamming->f32 (scheme1)', fn: (qfid) => vexus.pqHammingTwoStage(qfid, COARSE_TOP, TOP_N) },
    { name: 'Inverted->f32 (scheme2)', fn: (qfid) => vexus.tokenInvertedTwoStage(qfid, COARSE_TOP, TOP_N, 1) },
    { name: 'ADC->f32 (scheme3)', fn: (qfid) => vexus.adcCoarseF32Fine(qfid, COARSE_TOP, TOP_N) },
  ];

  for (const m of methods) {
    let totalMs = 0;
    for (let qi = 0; qi < qids.length; qi++) {
      const qfid = queryIdToFileId[qids[qi]];
      const t0 = performance.now();
      m.fn(qfid);
      totalMs += performance.now() - t0;
    }
    const avg = totalMs / qids.length;
    console.log(`  ${m.name.padEnd(35)} avg=${avg.toFixed(1)}ms  total=${(totalMs/1000).toFixed(1)}s`);
  }

  console.log('\nDone.\n');
}

main().catch(e => { console.error(e); process.exit(1); });
