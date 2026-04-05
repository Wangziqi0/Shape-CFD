#!/usr/bin/env node
'use strict';
/**
 * rust_cfd_bench_parallel.js — 128-worker 并行跑 Rust shapeCfdPipeline
 * 对比 cosine baseline 和 Rust 管线的 NDCG@10
 */
const {Worker, isMainThread, parentPort, workerData} = require('worker_threads');
const fs = require('fs'), path = require('path'), readline = require('readline');
const NW = parseInt(process.env.NW || '128');

// ═══════════════════════════════════════════════════════════════
// WORKER
// ═══════════════════════════════════════════════════════════════
if (!isMainThread) {
  const {queries, corpusVecs, allDids, sqlitePath, idMap} = workerData;
  const reverseMap = {};
  for (const [k, v] of Object.entries(idMap)) reverseMap[v] = k;

  // 加载 Rust
  const {LawVexus} = require('/home/amd/HEZIMENG/law-vexus');
  const vexus = new LawVexus('/tmp/bench_w' + process.pid);
  vexus.loadClouds(sqlitePath);

  function cosSim(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
  }
  function ndcg(ranked, qr, k = 10) {
    let d = 0;
    for (let i = 0; i < Math.min(ranked.length, k); i++)
      d += (Math.pow(2, qr[ranked[i]] || 0) - 1) / Math.log2(i + 2);
    const ir = Object.values(qr).sort((a, b) => b - a);
    let id = 0;
    for (let i = 0; i < Math.min(ir.length, k); i++)
      id += (Math.pow(2, ir[i]) - 1) / Math.log2(i + 2);
    return id > 0 ? d / id : 0;
  }

  const results = [];
  for (const {qid, qVec, qr} of queries) {
    const q = new Float32Array(qVec);

    // cosine baseline
    const cs = allDids.map(did => ({did, s: cosSim(q, new Float32Array(corpusVecs[did]))}));
    cs.sort((a, b) => b.s - a.s);
    const ndcgCos = ndcg(cs.slice(0, 10).map(d => d.did), qr);

    // Rust shapeCfdPipeline
    const qBuf = Buffer.from(q.buffer, q.byteOffset, q.byteLength);
    const rustHits = vexus.shapeCfdPipeline(qBuf, 10, 55);
    const rustRanked = rustHits.map(h => reverseMap[h.id]).filter(Boolean);
    const ndcgRust = ndcg(rustRanked, qr);

    results.push({qid, ndcgCos, ndcgRust});
  }
  parentPort.postMessage(results);
  process.exit(0);
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════
async function main() {
  const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
  const SQLITE_PATH = path.join(DATA_DIR, 'clouds.sqlite');
  const ID_MAP_PATH = path.join(DATA_DIR, 'id_map.json');

  console.log(`\n  Rust shapeCfdPipeline — ${NW}-worker 并行 Benchmark\n`);

  // 加载数据
  process.stdout.write('  加载数据...');
  const idMap = JSON.parse(fs.readFileSync(ID_MAP_PATH, 'utf-8'));

  function loadJsonl(fp) {
    return new Promise((resolve, reject) => {
      const a = [];
      const rl = readline.createInterface({input: fs.createReadStream(fp), crlfDelay: Infinity});
      rl.on('line', l => { if (l.trim()) try { a.push(JSON.parse(l)); } catch(e) {} });
      rl.on('close', () => resolve(a));
      rl.on('error', reject);
    });
  }

  const corpusVecs = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'corpus_vectors.jsonl')))
    corpusVecs[o._id] = Array.from(new Float32Array(o.vector));
  const allDids = Object.keys(corpusVecs);

  const qV = {};
  for (const o of await loadJsonl(path.join(DATA_DIR, 'query_vectors.jsonl')))
    qV[o._id] = Array.from(new Float32Array(o.vector));

  const qrels = {};
  const lines = fs.readFileSync(path.join(DATA_DIR, 'qrels.tsv'), 'utf-8').trim().split('\n');
  for (let i = 1; i < lines.length; i++) {
    const [qi, di, s] = lines[i].split('\t');
    if (!qrels[qi]) qrels[qi] = {};
    qrels[qi][di] = parseInt(s);
  }

  let qids = Object.keys(qrels).filter(q => qV[q]);
  const MAX_Q = parseInt(process.env.MAX_Q || '0');
  if (MAX_Q > 0) qids = qids.slice(0, MAX_Q);

  console.log(` ${qids.length} queries, ${allDids.length} docs, ${NW} workers`);

  // 分配 queries 到 workers
  const chunks = Array.from({length: NW}, () => []);
  qids.forEach((qid, i) => {
    chunks[i % NW].push({qid, qVec: qV[qid], qr: qrels[qid]});
  });

  const t0 = Date.now();
  const promises = chunks.filter(c => c.length > 0).map(queries => {
    return new Promise((resolve, reject) => {
      const w = new Worker(__filename, {
        workerData: {queries, corpusVecs, allDids, sqlitePath: SQLITE_PATH, idMap}
      });
      w.on('message', resolve);
      w.on('error', reject);
    });
  });

  const allResults = (await Promise.all(promises)).flat();
  const elapsed = (Date.now() - t0) / 1000;

  let sumCos = 0, sumRust = 0;
  for (const r of allResults) { sumCos += r.ndcgCos; sumRust += r.ndcgRust; }
  const nQ = allResults.length;
  const avgCos = sumCos / nQ;
  const avgRust = sumRust / nQ;
  const delta = ((avgRust - avgCos) / avgCos * 100);

  console.log(`\n  完成: ${elapsed.toFixed(1)}s (${(nQ/elapsed).toFixed(1)} q/s)\n`);
  console.log(`  cosine baseline:        ${avgCos.toFixed(4)}`);
  console.log(`  Rust shapeCfdPipeline:  ${avgRust.toFixed(4)} (${delta >= 0 ? '+' : ''}${delta.toFixed(2)}% vs cosine)`);
  console.log();
}

main().catch(e => { console.error(e); process.exit(1); });
