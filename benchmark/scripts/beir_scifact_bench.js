#!/usr/bin/env node
'use strict';
/** SciFact 句子级 Shape-CFD Benchmark */
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','scifact');
const K=10, TOP_N=55;

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}

async function main(){
  console.log('\n=== SciFact 句子级 Benchmark ===\n');

  // 加载 Rust addon
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/scifact_bench');

  // 加载句子级点云
  process.stdout.write('Loading sentence clouds... ');
  const ci=v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  console.log(`done (${ci})`);

  const hasLap=typeof v.shapeLaplacianPipeline==='function';
  const hasCfd=typeof v.shapeCfdPipeline==='function';
  console.log(`shapeCfdPipeline: ${hasCfd}, shapeLaplacianPipeline: ${hasLap}`);

  // 加载向量
  process.stdout.write('Loading vectors... ');
  const corpusVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl')))
    corpusVecs[o._id]=new Float32Array(o.vector);
  const allDids=Object.keys(corpusVecs);

  const queryVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))
    queryVecs[o._id]=new Float32Array(o.vector);
  console.log(`done (${allDids.length} docs, ${Object.keys(queryVecs).length} queries)`);

  // 加载 qrels
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const p=ql[i].split('\t');if(!qrels[p[0]])qrels[p[0]]={};qrels[p[0]][p[1]]=parseInt(p[2])}

  // id_map
  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries with relevance labels\n`);

  const m={cosine:{s:0,n:0},shape_cfd:{s:0,n:0},laplacian:{s:0,n:0}};
  const step=Math.max(1,Math.floor(qids.length/10));

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid];
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine
    const cs=allDids.map(d=>({d,s:cosSim(qVec,corpusVecs[d])})).sort((a,b)=>b.s-a.s);
    m.cosine.s+=computeNDCG(cs.slice(0,K).map(x=>x.d),qrel,K);m.cosine.n++;

    // shape_cfd (PDE)
    if(hasCfd){try{
      const h=v.shapeCfdPipeline(qBuf,K,TOP_N);
      m.shape_cfd.s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);m.shape_cfd.n++;
    }catch(e){if(qi===0)console.error('cfd:',e.message)}}

    // laplacian
    if(hasLap){try{
      const h=v.shapeLaplacianPipeline(qBuf,K,TOP_N);
      m.laplacian.s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);m.laplacian.n++;
    }catch(e){if(qi===0)console.error('lap:',e.message)}}

    if((qi+1)%step===0||qi===qids.length-1)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  console.log('='.repeat(56));
  console.log('  SciFact NDCG@10 Results');
  console.log('-'.repeat(56));
  console.log('  方法'.padEnd(24)+'NDCG@10'.padEnd(12)+'vs cosine');
  console.log('-'.repeat(56));
  const cosAvg=m.cosine.n>0?m.cosine.s/m.cosine.n:0;
  for(const[name,v2]of Object.entries(m)){
    if(v2.n===0){console.log(`  ${name.padEnd(24)}--`);continue}
    const avg=v2.s/v2.n;
    const pct=name==='cosine'?'--':`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
    console.log(`  ${name.padEnd(24)}${avg.toFixed(4).padEnd(12)}${pct}`)}
  console.log('='.repeat(56));
  console.log(`  ${qids.length} queries, ${allDids.length} corpus docs`);
}
main().catch(e=>{console.error(e);process.exit(1)});
