#!/usr/bin/env node
'use strict';
/**
 * 通用多 top_n 句子级 Benchmark（全 Rust 计算）
 * 用法: node beir_multi_bench.js <dataset_name> [top_n1,top_n2,...]
 * 例: node beir_multi_bench.js arguana 55,100,200
 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const dataset=process.argv[2]||'arguana';
const topNs=(process.argv[3]||'55,100,200').split(',').map(Number);
const K=10;
const DATA_DIR=path.join(__dirname,'beir_data',dataset);

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}

async function main(){
  console.log(`\n=== ${dataset.toUpperCase()} 句子级 Benchmark (Rust) ===`);
  console.log(`top_n: [${topNs.join(', ')}]\n`);

  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus(`/tmp/${dataset}_bench`);

  // 加载句子级点云到 Rust
  process.stdout.write('Loading clouds... ');
  const ci=v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  console.log(`done (${ci})`);

  const hasLap=typeof v.shapeLaplacianPipeline==='function';
  const hasCfd=typeof v.shapeCfdPipeline==='function';
  console.log(`Laplacian: ${hasLap}, PDE: ${hasCfd}`);

  // 只加载 query 向量（cosine 粗筛在 Rust 内部做）
  process.stdout.write('Loading query vectors... ');
  const queryVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))
    queryVecs[o._id]=new Float32Array(o.vector);
  console.log(`${Object.keys(queryVecs).length} queries`);

  // qrels
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const p=ql[i].split('\t');if(!qrels[p[0]])qrels[p[0]]={};qrels[p[0]][p[1]]=parseInt(p[2])}

  // id_map
  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries with relevance labels\n`);

  // 初始化指标：cosine 从 Rust lap pipeline 的粗筛阶段拿
  const metrics={};
  for(const tn of topNs){
    if(hasLap) metrics[`lap_${tn}`]={s:0,n:0};
    if(hasCfd) metrics[`pde_${tn}`]={s:0,n:0};
  }
  // 用最大 top_n 的 Rust cosine 粗筛结果算 cosine baseline
  metrics.cosine={s:0,n:0};

  const step=Math.max(1,Math.floor(qids.length/20));
  const t0=Date.now();

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid];
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // 各 top_n: Laplacian
    for(const tn of topNs){
      if(hasLap){try{
        const h=v.shapeLaplacianPipeline(qBuf,K,tn);
        const ranked=h.map(x=>rev[x.id]).filter(Boolean);
        metrics[`lap_${tn}`].s+=computeNDCG(ranked,qrel,K);metrics[`lap_${tn}`].n++;
      }catch(e){if(qi===0)console.error(`lap_${tn}:`,e.message)}}

      if(hasCfd){try{
        const h=v.shapeCfdPipeline(qBuf,K,tn);
        const ranked=h.map(x=>rev[x.id]).filter(Boolean);
        metrics[`pde_${tn}`].s+=computeNDCG(ranked,qrel,K);metrics[`pde_${tn}`].n++;
      }catch(e){if(qi===0)console.error(`pde_${tn}:`,e.message)}}
    }

    if((qi+1)%step===0||qi===qids.length-1){
      const elapsed=(Date.now()-t0)/1000;
      const rate=((qi+1)/elapsed).toFixed(1);
      process.stdout.write(`\r  ${qi+1}/${qids.length} (${rate} q/s)`);
    }
  }
  const totalTime=(Date.now()-t0)/1000;
  console.log(`\n  Rust pipelines done: ${totalTime.toFixed(1)}s\n`);

  // cosine baseline: Rust cosineRank
  const hasCosRank=typeof v.cosineRank==='function';
  if(hasCosRank){
    console.log('  Computing cosine baseline (Rust)...');
    const t1=Date.now();
    for(let qi=0;qi<qids.length;qi++){
      const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid];
      const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);
      const h=v.cosineRank(qBuf,K);
      const ranked=h.map(x=>rev[x.id]).filter(Boolean);
      metrics.cosine.s+=computeNDCG(ranked,qrel,K);metrics.cosine.n++;
      if((qi+1)%step===0)process.stdout.write(`\r  cosine: ${qi+1}/${qids.length}`);
    }
    console.log(`\r  cosine done: ${((Date.now()-t1)/1000).toFixed(1)}s`);
  } else {
    console.log('  ⚠ cosineRank not available, skipping cosine baseline');
  }

  // 输出结果
  const cosAvg=metrics.cosine.n>0?metrics.cosine.s/metrics.cosine.n:0;
  console.log('\n'+'='.repeat(60));
  console.log(`  ${dataset.toUpperCase()} NDCG@10 Results`);
  console.log('-'.repeat(60));
  console.log('  方法'.padEnd(28)+'NDCG@10'.padEnd(12)+'vs cosine');
  console.log('-'.repeat(60));
  // cosine first
  console.log(`  ${'cosine'.padEnd(28)}${cosAvg.toFixed(4).padEnd(12)}—`);
  // 按 top_n 排序输出
  for(const tn of topNs){
    for(const prefix of ['lap','pde']){
      const key=`${prefix}_${tn}`;
      const v2=metrics[key];
      if(!v2||v2.n===0)continue;
      const avg=v2.s/v2.n;
      const pct=`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
      console.log(`  ${key.padEnd(28)}${avg.toFixed(4).padEnd(12)}${pct}`);
    }
  }
  console.log('='.repeat(60));
  console.log(`  ${qids.length} queries`);

  // 保存 JSON
  const results={dataset,queries:qids.length,metrics:{}};
  for(const[name,v2]of Object.entries(metrics)){
    if(v2.n>0)results.metrics[name]=+(v2.s/v2.n).toFixed(4);
  }
  const outPath=path.join(DATA_DIR,'bench_results.json');
  fs.writeFileSync(outPath,JSON.stringify(results,null,2));
  console.log(`  Saved: ${outPath}`);
}
main().catch(e=>{console.error(e);process.exit(1)});
