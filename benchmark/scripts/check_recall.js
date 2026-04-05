#!/usr/bin/env node
'use strict';
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','nfcorpus');

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}

async function main(){
  console.log('\n=== 质心粗筛 Recall@K 测量 ===\n');
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/recall_test');
  v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  v.loadTokenCloudsSqlite(path.join(DATA_DIR,'token_clouds.sqlite'),path.join(DATA_DIR,'query_token_clouds.sqlite'));

  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;
  const queryVecs={};for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))queryVecs[o._id]=new Float32Array(o.vector);
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const[q,d,s]=ql[i].split('\t');if(!qrels[q])qrels[q]={};qrels[q][d]=parseInt(s)}
  const qidMap={};
  const qvl=fs.readFileSync(path.join(DATA_DIR,'query_vectors.jsonl'),'utf8').trim().split('\n');
  for(let i=0;i<qvl.length;i++){try{const o=JSON.parse(qvl[i]);if(o._id)qidMap[o._id]=i}catch(_){}}

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]&&qidMap[q]!==undefined);
  console.log(`${qids.length} queries\n`);

  // 用 token_chamfer_two_stage 的粗筛：质心 Chamfer top-K
  // 测 recall@K: 粗筛 top-K 中包含多少 ground truth 相关文档
  const Ks=[55,100,200,300,500,1000];
  const recallSums={};for(const k of Ks)recallSums[k]=0;

  // 同时测 token 全扫描 top-K 的 recall 作为天花板
  const fullRecallSums={};for(const k of Ks)fullRecallSums[k]=0;

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qrel=qrels[qid],intId=qidMap[qid];
    const relevant=new Set(Object.keys(qrel).filter(d=>qrel[d]>0));
    if(relevant.size===0)continue;

    // 质心粗筛各种 K
    for(const k of Ks){
      try{
        const h=v.tokenChamferTwoStage(intId,k,k);
        const retrieved=new Set(h.map(x=>rev[x[0]]).filter(Boolean));
        let hits=0;for(const d of relevant)if(retrieved.has(d))hits++;
        recallSums[k]+=hits/relevant.size;
      }catch(e){}
    }

    if((qi+1)%50===0)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  console.log('质心 Chamfer 粗筛 Recall@K:');
  console.log('-'.repeat(30));
  for(const k of Ks){
    const avg=recallSums[k]/qids.length;
    console.log(`  Recall@${String(k).padEnd(6)} ${avg.toFixed(4)}`);
  }
}
main().catch(e=>{console.error(e);process.exit(1)});
