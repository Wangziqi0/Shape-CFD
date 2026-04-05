#!/usr/bin/env node
'use strict';
/** top_n 扫描：token_2stage + 图平滑融合，不同候选集大小 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','nfcorpus');
const K=10;

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}
function normalizeScores(m){const v=Object.values(m);if(!v.length)return{};const mn=Math.min(...v),mx=Math.max(...v),r=mx-mn||1e-8;const o={};for(const[k,val]of Object.entries(m))o[k]=(val-mn)/r;return o}

async function main(){
  console.log('\n=== top_n 扫描: token_2stage + 图平滑融合 ===\n');
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/topn_test');

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
  console.log(`${qids.length} queries\n`);

  const topNs=[55,100,200];
  const methods={cosine:{s:0,n:0}};
  for(const tn of topNs){
    methods[`t2s_${tn}`]={s:0,n:0,ms:0};
    methods[`lap_${tn}`]={s:0,n:0,ms:0};
    methods[`fus_${tn}`]={s:0,n:0};
  }

  const step=Math.max(1,Math.floor(qids.length/10));

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid],intId=qidMap[qid];
    if(intId===undefined)continue;
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine
    const cs=allDids.map(d=>({d,s:cosSim(qVec,corpusVecs[d])})).sort((a,b)=>b.s-a.s);
    methods.cosine.s+=computeNDCG(cs.slice(0,K).map(x=>x.d),qrel,K);methods.cosine.n++;

    for(const tn of topNs){
      // token_2stage (粗筛100 → 精排 tn)
      let tokMap=null;
      {const t=Date.now();
      try{
        const coarse=Math.max(100,tn);
        const h=v.tokenChamferTwoStage(intId,coarse,tn);
        const ranked=h.map(x=>rev[x[0]]).filter(Boolean);
        methods[`t2s_${tn}`].s+=computeNDCG(ranked,qrel,K);methods[`t2s_${tn}`].n++;
        methods[`t2s_${tn}`].ms+=Date.now()-t;
        tokMap={};for(const x of h){const s=rev[x[0]];if(s)tokMap[s]=x[1]}
      }catch(e){if(qi===0)console.error(`t2s_${tn}:`,e.message)}}

      // 图平滑 (top_n=tn)
      let lapMap=null;
      {const t=Date.now();
      try{
        const h=v.shapeLaplacianPipeline(qBuf,tn,tn);
        methods[`lap_${tn}`].s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);
        methods[`lap_${tn}`].n++;methods[`lap_${tn}`].ms+=Date.now()-t;
        lapMap={};for(const x of h){const s=rev[x.id];if(s)lapMap[s]=x.score}
      }catch(e){if(qi===0)console.error(`lap_${tn}:`,e.message)}}

      // 融合 0.7*token + 0.3*lap
      if(tokMap&&lapMap){
        const all=new Set([...Object.keys(tokMap),...Object.keys(lapMap)]);
        const nt=normalizeScores(tokMap),nl=normalizeScores(lapMap);
        const fs2=[];for(const d of all)fs2.push({d,s:0.7*(nt[d]||0)+0.3*(nl[d]||0)});
        fs2.sort((a,b)=>b.s-a.s);
        methods[`fus_${tn}`].s+=computeNDCG(fs2.slice(0,K).map(x=>x.d),qrel,K);
        methods[`fus_${tn}`].n++;
      }
    }
    if((qi+1)%step===0||qi===qids.length-1)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  console.log('='.repeat(64));
  console.log('  方法'.padEnd(20)+'NDCG@10'.padEnd(12)+'vs cosine'.padEnd(12)+'延迟');
  console.log('-'.repeat(64));
  const cosAvg=methods.cosine.n>0?methods.cosine.s/methods.cosine.n:0;
  const order=['cosine',...topNs.flatMap(tn=>[`t2s_${tn}`,`lap_${tn}`,`fus_${tn}`])];
  for(const name of order){
    const m=methods[name];if(!m||m.n===0)continue;
    const avg=m.s/m.n;
    const pct=name==='cosine'?'--':`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
    const lat=m.ms?`${(m.ms/m.n).toFixed(0)}ms`:'';
    console.log(`  ${name.padEnd(20)}${avg.toFixed(4).padEnd(12)}${pct.padEnd(12)}${lat}`);
    if(name.startsWith('fus_'))console.log('');
  }
  console.log('='.repeat(64));
}
main().catch(e=>{console.error(e);process.exit(1)});
