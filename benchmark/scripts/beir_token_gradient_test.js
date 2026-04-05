#!/usr/bin/env node
'use strict';
/** Token 梯度对流 PDE vs 旧 PDE vs 图平滑 vs 无处理 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','nfcorpus');
const K=10,TOP_N=55;

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}
function normalizeScores(m){const v=Object.values(m);if(!v.length)return{};const mn=Math.min(...v),mx=Math.max(...v),r=mx-mn||1e-8;const o={};for(const[k,val]of Object.entries(m))o[k]=(val-mn)/r;return o}

async function main(){
  console.log('\n=== Token 梯度对流 PDE Benchmark ===\n');
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/tg_test');

  process.stdout.write('Loading... ');
  v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  v.loadTokenCloudsSqlite(path.join(DATA_DIR,'token_clouds.sqlite'),path.join(DATA_DIR,'query_token_clouds.sqlite'));
  console.log('done');

  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;
  const corpusVecs={};for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl')))corpusVecs[o._id]=new Float32Array(o.vector);
  const allDids=Object.keys(corpusVecs);
  const queryVecs={};for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))queryVecs[o._id]=new Float32Array(o.vector);
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const[q,d,s]=ql[i].split('\t');if(!qrels[q])qrels[q]={};qrels[q][d]=parseInt(s)}
  const qidMap={};
  const qvl=fs.readFileSync(path.join(DATA_DIR,'query_vectors.jsonl'),'utf8').trim().split('\n');
  for(let i=0;i<qvl.length;i++){try{const o=JSON.parse(qvl[i]);if(o._id)qidMap[o._id]=i}catch(_){}}

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);

  const hasTG=typeof v.tokenGradientPdePipeline==='function';
  const hasLap=typeof v.shapeLaplacianPipeline==='function';
  const hasT2S=typeof v.tokenChamferTwoStage==='function';
  console.log(`tokenGradientPdePipeline: ${hasTG}`);
  console.log(`${qids.length} queries\n`);

  if(!hasTG){console.error('tokenGradientPdePipeline not available!');process.exit(1)}

  const m={cosine:{s:0,n:0,ms:0},shape_cfd:{s:0,n:0,ms:0},laplacian:{s:0,n:0,ms:0},
           token_2stage:{s:0,n:0,ms:0},fusion_pde:{s:0,n:0,ms:0},fusion_lap:{s:0,n:0,ms:0},
           tg_pde_b1:{s:0,n:0,ms:0},tg_pde_b3:{s:0,n:0,ms:0},tg_pde_b5:{s:0,n:0,ms:0},
           tg_fusion_b1:{s:0,n:0,ms:0}};
  const step=Math.max(1,Math.floor(qids.length/10));

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid],intId=qidMap[qid];
    if(intId===undefined)continue;
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine
    {const cs=allDids.map(d=>({d,s:cosSim(qVec,corpusVecs[d])})).sort((a,b)=>b.s-a.s);
    m.cosine.s+=computeNDCG(cs.slice(0,K).map(x=>x.d),qrel,K);m.cosine.n++}

    // shape_cfd (旧 PDE)
    let cfdMap=null;
    try{const h=v.shapeCfdPipeline(qBuf,K,TOP_N);
    m.shape_cfd.s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);m.shape_cfd.n++;
    cfdMap={};for(const x of h){const s=rev[x.id];if(s)cfdMap[s]=x.score}}catch(e){}

    // laplacian
    if(hasLap){try{const h=v.shapeLaplacianPipeline(qBuf,K,TOP_N);
    m.laplacian.s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);m.laplacian.n++}catch(e){}}

    // token_2stage
    let tokMap=null;
    if(hasT2S){try{const h=v.tokenChamferTwoStage(intId,100,TOP_N);
    m.token_2stage.s+=computeNDCG(h.map(x=>rev[x[0]]).filter(Boolean),qrel,K);m.token_2stage.n++;
    tokMap={};for(const x of h){const s=rev[x[0]];if(s)tokMap[s]=x[1]}}catch(e){}}

    // 旧融合 (token + PDE)
    if(tokMap&&cfdMap){
      const all=new Set([...Object.keys(tokMap),...Object.keys(cfdMap)]);
      const nt=normalizeScores(tokMap),nc=normalizeScores(cfdMap);
      const fs2=[];for(const d of all)fs2.push({d,s:0.7*(nt[d]||0)+0.3*(nc[d]||0)});
      fs2.sort((a,b)=>b.s-a.s);
      m.fusion_pde.s+=computeNDCG(fs2.slice(0,K).map(x=>x.d),qrel,K);m.fusion_pde.n++}

    // Token 梯度 PDE (beta=1.0, 3.0, 5.0)
    for(const[beta,name]of[[1.0,'tg_pde_b1'],[3.0,'tg_pde_b3'],[5.0,'tg_pde_b5']]){
      const t=Date.now();
      try{const h=v.tokenGradientPdePipeline(qBuf,intId,K,TOP_N,beta);
      const ranked=h.map(x=>rev[x.id]).filter(Boolean);
      m[name].s+=computeNDCG(ranked,qrel,K);m[name].n++;
      m[name].ms+=Date.now()-t;

      // 融合 (token + tg_pde) 只对 beta=1
      if(name==='tg_pde_b1'&&tokMap){
        const tgMap={};for(const x of h){const s=rev[x.id];if(s)tgMap[s]=x.score}
        const all=new Set([...Object.keys(tokMap),...Object.keys(tgMap)]);
        const nt=normalizeScores(tokMap),ng=normalizeScores(tgMap);
        const fs2=[];for(const d of all)fs2.push({d,s:0.7*(nt[d]||0)+0.3*(ng[d]||0)});
        fs2.sort((a,b)=>b.s-a.s);
        m.tg_fusion_b1.s+=computeNDCG(fs2.slice(0,K).map(x=>x.d),qrel,K);m.tg_fusion_b1.n++}
      }catch(e){if(qi===0)console.error(`${name}:`,e.message)}}

    if((qi+1)%step===0||qi===qids.length-1)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  console.log('='.repeat(64));
  console.log('  方法'.padEnd(24)+'NDCG@10'.padEnd(12)+'vs cosine'.padEnd(14)+'延迟');
  console.log('-'.repeat(64));
  const cosAvg=m.cosine.n>0?m.cosine.s/m.cosine.n:0;
  for(const[name,v2]of Object.entries(m)){
    if(v2.n===0){console.log(`  ${name.padEnd(24)}-- (not run)`);continue}
    const avg=v2.s/v2.n;const pct=name==='cosine'?'--':`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
    const lat=v2.ms>0?`${(v2.ms/v2.n).toFixed(0)}ms`:'';
    console.log(`  ${name.padEnd(24)}${avg.toFixed(4).padEnd(12)}${pct.padEnd(14)}${lat}`)}
  console.log('='.repeat(64));
}
main().catch(e=>{console.error(e);process.exit(1)});
