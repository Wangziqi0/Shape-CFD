#!/usr/bin/env node
'use strict';
/** FiQA 句子级融合：cosine + 图平滑 各种权重 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','fiqa');
const K=10;

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}
function normalizeScores(m){const v=Object.values(m);if(!v.length)return{};const mn=Math.min(...v),mx=Math.max(...v),r=mx-mn||1e-8;const o={};for(const[k,val]of Object.entries(m))o[k]=(val-mn)/r;return o}

async function main(){
  console.log('\n=== FiQA 句子级融合 Benchmark ===\n');
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/fiqa_fus');

  process.stdout.write('Loading clouds... ');
  v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  console.log('done');

  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;

  process.stdout.write('Loading vectors... ');
  const corpusVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl')))
    corpusVecs[o._id]=new Float32Array(o.vector);
  const allDids=Object.keys(corpusVecs);
  const queryVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))
    queryVecs[o._id]=new Float32Array(o.vector);
  console.log(`done (${allDids.length} docs)`);

  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const[q,d,s]=ql[i].split('\t');if(!qrels[q])qrels[q]={};qrels[q][d]=parseInt(s)}

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries\n`);

  const topNs=[55,100,200];
  const methods={cosine:{s:0,n:0}};
  for(const tn of topNs){
    methods[`lap_${tn}`]={s:0,n:0};
    methods[`pde_${tn}`]={s:0,n:0};
    for(const lam of [0.3,0.5,0.7]){
      methods[`fus_${tn}_${lam}`]={s:0,n:0};
    }
  }

  const step=Math.max(1,Math.floor(qids.length/10));
  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid];
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine
    const cs=allDids.map(d=>({d,s:cosSim(qVec,corpusVecs[d])})).sort((a,b)=>b.s-a.s);
    const cosMap={};for(const x of cs.slice(0,500))cosMap[x.d]=x.s;
    methods.cosine.s+=computeNDCG(cs.slice(0,K).map(x=>x.d),qrel,K);methods.cosine.n++;

    for(const tn of topNs){
      // 图平滑
      let lapMap=null;
      try{
        const h=v.shapeLaplacianPipeline(qBuf,tn,tn);
        const ranked=h.map(x=>rev[x.id]).filter(Boolean);
        methods[`lap_${tn}`].s+=computeNDCG(ranked,qrel,K);methods[`lap_${tn}`].n++;
        lapMap={};for(const x of h){const s=rev[x.id];if(s)lapMap[s]=x.score}
      }catch(e){}

      // PDE
      try{
        const h=v.shapeCfdPipeline(qBuf,tn,tn);
        methods[`pde_${tn}`].s+=computeNDCG(h.map(x=>rev[x.id]).filter(Boolean),qrel,K);
        methods[`pde_${tn}`].n++;
      }catch(e){}

      // 融合 cosine + 图平滑
      if(lapMap){
        for(const lam of [0.3,0.5,0.7]){
          const all=new Set([...Object.keys(cosMap),...Object.keys(lapMap)]);
          const nc=normalizeScores(cosMap),nl=normalizeScores(lapMap);
          const fs2=[];for(const d of all)fs2.push({d,s:lam*(nc[d]||0)+(1-lam)*(nl[d]||0)});
          fs2.sort((a,b)=>b.s-a.s);
          methods[`fus_${tn}_${lam}`].s+=computeNDCG(fs2.slice(0,K).map(x=>x.d),qrel,K);
          methods[`fus_${tn}_${lam}`].n++;
        }
      }
    }
    if((qi+1)%step===0||qi===qids.length-1)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  console.log('='.repeat(60));
  console.log('  FiQA NDCG@10');
  console.log('-'.repeat(60));
  const cosAvg=methods.cosine.n>0?methods.cosine.s/methods.cosine.n:0;
  const order=['cosine',...topNs.flatMap(tn=>[`pde_${tn}`,`lap_${tn}`,...[0.3,0.5,0.7].map(l=>`fus_${tn}_${l}`)])];
  for(const name of order){
    const m=methods[name];if(!m||m.n===0)continue;
    const avg=m.s/m.n;
    const pct=name==='cosine'?'--':`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
    console.log(`  ${name.padEnd(20)}${avg.toFixed(4).padEnd(12)}${pct}`);
    if(name.startsWith('fus_')&&name.endsWith('0.7'))console.log('');
  }
  console.log('='.repeat(60));
}
main().catch(e=>{console.error(e);process.exit(1)});
