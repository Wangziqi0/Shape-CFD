#!/usr/bin/env node
'use strict';
/** 测量 token Chamfer vs sentence Chamfer 排名的 Kendall tau */
const fs = require('fs'), path = require('path'), readline = require('readline');
const DATA_DIR = path.join(__dirname, 'beir_data', 'nfcorpus');
const TOP_N = 55;

function loadJsonl(fp) {
  return new Promise((resolve, reject) => {
    const arr = [];
    const rl = readline.createInterface({ input: fs.createReadStream(fp), crlfDelay: Infinity });
    rl.on('line', l => { if(l.trim()) try{arr.push(JSON.parse(l))}catch(_){} });
    rl.on('close', () => resolve(arr)); rl.on('error', reject);
  });
}

function kendallTau(a, b) {
  const n = a.length;
  let concordant = 0, discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const ai = a[i] - a[j], bi = b[i] - b[j];
      if (ai * bi > 0) concordant++;
      else if (ai * bi < 0) discordant++;
    }
  }
  const total = concordant + discordant;
  return total > 0 ? (concordant - discordant) / total : 0;
}

async function main() {
  console.log('\n=== Kendall Tau: Token Chamfer vs Sentence Chamfer ===\n');
  const { LawVexus } = require('/home/amd/HEZIMENG/law-vexus');
  const vexus = new LawVexus('/tmp/tau_test');
  
  process.stdout.write('Loading clouds... ');
  vexus.loadClouds(path.join(DATA_DIR, 'clouds.sqlite'));
  vexus.loadTokenCloudsSqlite(path.join(DATA_DIR,'token_clouds.sqlite'), path.join(DATA_DIR,'query_token_clouds.sqlite'));
  console.log('done');

  const idMap = JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev = {}; for(const[s,i]of Object.entries(idMap)) rev[i]=s;
  const corpusVecs = {}; for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl'))) corpusVecs[o._id]=new Float32Array(o.vector);
  const queryVecs = {}; for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl'))) queryVecs[o._id]=new Float32Array(o.vector);
  const qrels = {};
  const ql = fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const[q,d,s]=ql[i].split('\t');if(!qrels[q])qrels[q]={};qrels[q][d]=parseInt(s)}
  const qidMap = {};
  const qvl = fs.readFileSync(path.join(DATA_DIR,'query_vectors.jsonl'),'utf8').trim().split('\n');
  for(let i=0;i<qvl.length;i++){try{const o=JSON.parse(qvl[i]);if(o._id)qidMap[o._id]=i}catch(_){}}

  let qids = Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0'); if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries\n`);

  const taus = [];
  const step = Math.max(1,Math.floor(qids.length/10));

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi], qVec=queryVecs[qid], intId=qidMap[qid];
    if(intId===undefined) continue;
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // 句子级 Shape-CFD scores (top-55 文档)
    let sentScores;
    try {
      const hits = vexus.shapeCfdPipeline(qBuf, TOP_N, TOP_N);
      sentScores = {}; for(const h of hits) sentScores[h.id] = h.score;
    } catch(e){ continue; }

    // Token 级 2-stage scores (top-55 文档)
    let tokScores;
    try {
      const hits = vexus.tokenChamferTwoStage(intId, 100, TOP_N);
      tokScores = {}; for(const h of hits) tokScores[h[0]] = h[1];
    } catch(e){ continue; }

    // 取共同文档集
    const common = Object.keys(sentScores).filter(id => tokScores[id] !== undefined);
    if(common.length < 10) continue;

    const sentArr = common.map(id => sentScores[id]);
    const tokArr = common.map(id => tokScores[id]);
    const tau = kendallTau(sentArr, tokArr);
    taus.push(tau);

    if((qi+1)%step===0||qi===qids.length-1) process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  const mean = taus.reduce((a,b)=>a+b,0)/taus.length;
  const sorted = [...taus].sort((a,b)=>a-b);
  const median = sorted[Math.floor(sorted.length/2)];
  const p25 = sorted[Math.floor(sorted.length*0.25)];
  const p75 = sorted[Math.floor(sorted.length*0.75)];
  const std = Math.sqrt(taus.reduce((s,t)=>s+(t-mean)**2,0)/(taus.length-1));

  console.log('='.repeat(50));
  console.log(`  Kendall Tau (token vs sentence Chamfer)`);
  console.log('-'.repeat(50));
  console.log(`  N queries:  ${taus.length}`);
  console.log(`  Mean:       ${mean.toFixed(4)}`);
  console.log(`  Median:     ${median.toFixed(4)}`);
  console.log(`  Std:        ${std.toFixed(4)}`);
  console.log(`  P25:        ${p25.toFixed(4)}`);
  console.log(`  P75:        ${p75.toFixed(4)}`);
  console.log(`  Min:        ${sorted[0].toFixed(4)}`);
  console.log(`  Max:        ${sorted[sorted.length-1].toFixed(4)}`);
  console.log('='.repeat(50));

  if(mean < 0.7) console.log('\n  ✓ tau < 0.7: 跨粒度融合有充足信息增量空间');
  else if(mean < 0.9) console.log('\n  ~ tau 0.7-0.9: 有一定信息增量但不大');
  else console.log('\n  ✗ tau > 0.9: 两个排名高度一致，跨粒度融合意义不大');
}
main().catch(e=>{console.error(e);process.exit(1)});
