#!/usr/bin/env node
'use strict';
/**
 * 通用 token_2stage benchmark
 * 用法: node beir_token2stage_bench.js <dataset> [top_n]
 * 例: RAYON_NUM_THREADS=70 node --max-old-space-size=32768 beir_token2stage_bench.js fiqa 55
 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const dataset=process.argv[2]||'arguana';
const TOP_N=parseInt(process.argv[3]||'55');
const K=10;
const DATA_DIR=path.join(__dirname,'beir_data',dataset);

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}

async function main(){
  console.log(`\n=== ${dataset.toUpperCase()} Token-2Stage Benchmark ===`);
  console.log(`top_n=${TOP_N}\n`);

  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus(`/tmp/${dataset}_t2s_bench`);

  // 加载句子级点云
  process.stdout.write('Loading sentence clouds... ');
  const ci=v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  console.log(`done (${ci})`);

  // 加载 token 点云
  const tcPath=path.join(DATA_DIR,'token_clouds.sqlite');
  const qtcPath=path.join(DATA_DIR,'query_token_clouds.sqlite');
  let hasToken=false;
  if(typeof v.loadTokenCloudsSqlite==='function'){
    process.stdout.write('Loading token clouds... ');
    try{
      const ti=v.loadTokenCloudsSqlite(tcPath,qtcPath);
      hasToken=true;
      console.log(`done (${ti})`);
    }catch(e){console.log(`FAILED: ${e.message}`)}
  }

  if(!hasToken){console.error('Token clouds not available!');process.exit(1)}

  // 加载评估数据
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const p=ql[i].split('\t');if(!qrels[p[0]])qrels[p[0]]={};qrels[p[0]][p[1]]=parseInt(p[2])}

  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;

  // query 向量（用于 cosine baseline）
  const queryVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))
    queryVecs[o._id]=new Float32Array(o.vector);

  // query id → token file_id 映射（token_2stage 需要 query_id 不是 buffer）
  // query_token_clouds.sqlite 中 file_id = 按 queries.jsonl 顺序的索引
  const queryIdToFileId={};
  const queriesRaw=await loadJsonl(path.join(DATA_DIR,'queries.jsonl'));
  for(let i=0;i<queriesRaw.length;i++) queryIdToFileId[queriesRaw[i]._id]=i;

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries\n`);

  // 指标
  const metrics={
    cosine:{s:0,n:0},
    lap:{s:0,n:0},
    token_2stage:{s:0,n:0},
    fusion_07:{s:0,n:0},
  };

  const step=Math.max(1,Math.floor(qids.length/20));
  const t0=Date.now();

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid];
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine baseline (Rust)
    try{
      const h=v.cosineRank(qBuf,K);
      metrics.cosine.s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);
      metrics.cosine.n++;
    }catch(e){if(qi===0)console.error('cosine:',e.message)}

    // 图平滑 (句子级)
    try{
      const h=v.shapeLaplacianPipeline(qBuf,K,TOP_N);
      metrics.lap.s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);
      metrics.lap.n++;
    }catch(e){if(qi===0)console.error('lap:',e.message)}

    // token_2stage（需要 query file_id，不是 buffer）
    // 返回 [[doc_id, score], ...]
    const qFileId=queryIdToFileId[qid];
    try{
      const h=v.tokenChamferTwoStage(qFileId,100,TOP_N);
      const ranked=h.map(x=>rev[x[0]]).filter(Boolean);
      metrics.token_2stage.s+=computeNDCG(ranked,qrel,K);
      metrics.token_2stage.n++;

      // fusion: 0.7*token + 0.3*lap
      const lapH=v.shapeLaplacianPipeline(qBuf,K,TOP_N);
      const lapScores={};for(const x of lapH)lapScores[x.id]=x.score;
      const tokScores={};for(const x of h)tokScores[x[0]]=x[1];

      // 归一化
      const allIds=new Set([...Object.keys(lapScores),...Object.keys(tokScores)]);
      const tVals=Object.values(tokScores),lVals=Object.values(lapScores);
      const tMin=Math.min(...tVals),tMax=Math.max(...tVals),tR=tMax-tMin||1e-8;
      const lMin=Math.min(...lVals),lMax=Math.max(...lVals),lR=lMax-lMin||1e-8;

      const fused=[];
      for(const id of allIds){
        const ts=tokScores[id]!==undefined?(tokScores[id]-tMin)/tR:0;
        const ls=lapScores[id]!==undefined?(lapScores[id]-lMin)/lR:0;
        fused.push({id:parseInt(id),score:0.7*ts+0.3*ls});
      }
      fused.sort((a,b)=>b.score-a.score);
      metrics.fusion_07.s+=computeNDCG(fused.slice(0,K).map(x=>rev[x.id]).filter(Boolean),qrel,K);
      metrics.fusion_07.n++;
    }catch(e){if(qi===0)console.error('token:',e.message)}

    if((qi+1)%step===0||qi===qids.length-1){
      const el=(Date.now()-t0)/1000;
      process.stdout.write(`\r  ${qi+1}/${qids.length} (${((qi+1)/el).toFixed(1)} q/s)`);
    }
  }
  console.log(`\n  Done: ${((Date.now()-t0)/1000).toFixed(1)}s\n`);

  // 输出
  const cosAvg=metrics.cosine.n>0?metrics.cosine.s/metrics.cosine.n:0;
  console.log('='.repeat(60));
  console.log(`  ${dataset.toUpperCase()} Token-2Stage NDCG@10`);
  console.log('-'.repeat(60));
  console.log('  方法'.padEnd(28)+'NDCG@10'.padEnd(12)+'vs cosine');
  console.log('-'.repeat(60));
  for(const[name,v2]of Object.entries(metrics)){
    if(v2.n===0)continue;
    const avg=v2.s/v2.n;
    const pct=name==='cosine'?'—':`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
    console.log(`  ${name.padEnd(28)}${avg.toFixed(4).padEnd(12)}${pct}`);
  }
  console.log('='.repeat(60));

  // 保存
  const results={dataset,top_n:TOP_N,queries:qids.length,metrics:{}};
  for(const[name,v2]of Object.entries(metrics)){
    if(v2.n>0)results.metrics[name]=+(v2.s/v2.n).toFixed(4);
  }
  const outPath=path.join(DATA_DIR,'token2stage_results.json');
  fs.writeFileSync(outPath,JSON.stringify(results,null,2));
  console.log(`  Saved: ${outPath}`);
}
main().catch(e=>{console.error(e);process.exit(1)});
