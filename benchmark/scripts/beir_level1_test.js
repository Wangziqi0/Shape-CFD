#!/usr/bin/env node
'use strict';
/** Level 1 验证：对流在大候选集上是否复活 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','nfcorpus');
const K=10;

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}

async function main(){
  console.log('\n=== Level 1 验证：对流在大候选集上复活？ ===\n');
  console.log('核心假设：top_n=55 时 Pe≈0.064（对流死），top_n=500 时 Pe≈1（对流活）\n');

  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/l1_test');

  process.stdout.write('Loading... ');
  v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  console.log('done');

  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;
  const corpusVecs={};for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl')))corpusVecs[o._id]=new Float32Array(o.vector);
  const allDids=Object.keys(corpusVecs);
  const queryVecs={};for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))queryVecs[o._id]=new Float32Array(o.vector);
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const[q,d,s]=ql[i].split('\t');if(!qrels[q])qrels[q]={};qrels[q][d]=parseInt(s)}

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries\n`);

  // 测试不同 top_n 下的 PDE vs 图平滑
  const topNs = [55, 100, 200, 500];
  const methods = {};
  for(const tn of topNs){
    methods[`pde_${tn}`] = {s:0,n:0,ms:0};
    methods[`lap_${tn}`] = {s:0,n:0,ms:0};
  }
  methods['cosine'] = {s:0,n:0,ms:0};

  const step=Math.max(1,Math.floor(qids.length/10));

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid];
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine
    const cs=allDids.map(d=>({d,s:cosSim(qVec,corpusVecs[d])})).sort((a,b)=>b.s-a.s);
    methods.cosine.s+=computeNDCG(cs.slice(0,K).map(x=>x.d),qrel,K);methods.cosine.n++;

    for(const tn of topNs){
      // PDE (对流+扩散)
      {const t=Date.now();
      try{const h=v.shapeCfdPipeline(qBuf,K,tn);
      methods[`pde_${tn}`].s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);
      methods[`pde_${tn}`].n++;methods[`pde_${tn}`].ms+=Date.now()-t}catch(e){}}

      // 图平滑 (纯扩散)
      {const t=Date.now();
      try{const h=v.shapeLaplacianPipeline(qBuf,K,tn);
      methods[`lap_${tn}`].s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);
      methods[`lap_${tn}`].n++;methods[`lap_${tn}`].ms+=Date.now()-t}catch(e){}}
    }

    if((qi+1)%step===0||qi===qids.length-1)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  // 输出
  console.log('='.repeat(70));
  console.log('  方法'.padEnd(20)+'NDCG@10'.padEnd(10)+'vs cos'.padEnd(10)+'PDE-LAP'.padEnd(10)+'延迟');
  console.log('-'.repeat(70));
  const cosAvg=methods.cosine.n>0?methods.cosine.s/methods.cosine.n:0;

  console.log(`  ${'cosine'.padEnd(20)}${cosAvg.toFixed(4).padEnd(10)}${'--'.padEnd(10)}${''.padEnd(10)}`);

  for(const tn of topNs){
    const pde=methods[`pde_${tn}`], lap=methods[`lap_${tn}`];
    const pAvg=pde.n>0?pde.s/pde.n:0, lAvg=lap.n>0?lap.s/lap.n:0;
    const diff=(pAvg-lAvg).toFixed(4);
    const winner=pAvg>lAvg?'PDE胜':'LAP胜';
    const pLat=pde.n>0?`${(pde.ms/pde.n).toFixed(0)}ms`:'';
    const lLat=lap.n>0?`${(lap.ms/lap.n).toFixed(0)}ms`:'';

    console.log(`  ${'pde_'+tn}`.padEnd(20)+`${pAvg.toFixed(4)}`.padEnd(10)+`${((pAvg-cosAvg)/cosAvg*100).toFixed(1)}%`.padEnd(10)+`${diff}`.padEnd(10)+pLat);
    console.log(`  ${'lap_'+tn}`.padEnd(20)+`${lAvg.toFixed(4)}`.padEnd(10)+`${((lAvg-cosAvg)/cosAvg*100).toFixed(1)}%`.padEnd(10)+`${winner}`.padEnd(10)+lLat);
    console.log('');
  }
  console.log('='.repeat(70));
  console.log('\n  关键看：top_n 增大时 PDE-LAP 差值是否从负变正');
  console.log('  如果 top_n=500 时 PDE > LAP，对流在大候选集复活\n');
}
main().catch(e=>{console.error(e);process.exit(1)});
