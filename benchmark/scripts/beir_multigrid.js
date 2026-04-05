#!/usr/bin/env node
'use strict';
/** Multigrid V-Cycle: 残差驱动的候选扩展 */
const fs=require('fs'),path=require('path'),readline=require('readline');
const DATA_DIR=path.join(__dirname,'beir_data','nfcorpus');
const K=10,TOP_N=55;

function loadJsonl(fp){return new Promise((res,rej)=>{const a=[];const r=readline.createInterface({input:fs.createReadStream(fp),crlfDelay:Infinity});r.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l))}catch(_){}});r.on('close',()=>res(a));r.on('error',rej)})}
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)}
function computeNDCG(ranked,qrel,k=10){let dcg=0;for(let i=0;i<Math.min(ranked.length,k);i++){const r=qrel[ranked[i]]||0;dcg+=(Math.pow(2,r)-1)/Math.log2(i+2)}const ir=Object.values(qrel).sort((a,b)=>b-a);let idcg=0;for(let i=0;i<Math.min(ir.length,k);i++){idcg+=(Math.pow(2,ir[i])-1)/Math.log2(i+2)}return idcg>0?dcg/idcg:0}
function normalizeScores(m){const v=Object.values(m);if(!v.length)return{};const mn=Math.min(...v),mx=Math.max(...v),r=mx-mn||1e-8;const o={};for(const[k,val]of Object.entries(m))o[k]=(val-mn)/r;return o}
function rankNormalize(arr){const sorted=[...arr].map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v);const out=new Array(arr.length);sorted.forEach((x,rank)=>{out[x.i]=1-rank/(arr.length-1||1)});return out}

async function main(){
  console.log('\n=== Multigrid V-Cycle Benchmark ===\n');
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  const v=new LawVexus('/tmp/mg_test');

  process.stdout.write('Loading... ');
  v.loadClouds(path.join(DATA_DIR,'clouds.sqlite'));
  v.loadTokenCloudsSqlite(path.join(DATA_DIR,'token_clouds.sqlite'),path.join(DATA_DIR,'query_token_clouds.sqlite'));
  console.log('done');

  const hasRerank=typeof v.tokenChamferRerankList==='function';
  console.log(`tokenChamferRerankList: ${hasRerank}`);
  if(!hasRerank){console.error('NOT AVAILABLE');process.exit(1)}

  const idMap=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'id_map.json'),'utf-8'));
  const rev={};for(const[s,i]of Object.entries(idMap))rev[i]=s;
  const intToStr={};for(const[s,i]of Object.entries(idMap))intToStr[i]=s;

  // 加载全库质心向量（用于邻域搜索）
  const corpusVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl')))
    corpusVecs[o._id]=new Float32Array(o.vector);
  const allDids=Object.keys(corpusVecs);
  const queryVecs={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))
    queryVecs[o._id]=new Float32Array(o.vector);
  const qrels={};
  const ql=fs.readFileSync(path.join(DATA_DIR,'qrels.tsv'),'utf-8').trim().split('\n');
  for(let i=1;i<ql.length;i++){const[q,d,s]=ql[i].split('\t');if(!qrels[q])qrels[q]={};qrels[q][d]=parseInt(s)}
  const qidMap={};
  const qvl=fs.readFileSync(path.join(DATA_DIR,'query_vectors.jsonl'),'utf8').trim().split('\n');
  for(let i=0;i<qvl.length;i++){try{const o=JSON.parse(qvl[i]);if(o._id)qidMap[o._id]=i}catch(_){}}

  let qids=Object.keys(qrels).filter(q=>queryVecs[q]&&qidMap[q]!==undefined);
  const MQ=parseInt(process.env.MAX_Q||'0');if(MQ>0)qids=qids.slice(0,MQ);
  console.log(`${qids.length} queries\n`);

  // strId -> intId 映射
  const strToInt={};for(const[s,i]of Object.entries(idMap))strToInt[s]=i;

  const methods={
    cosine:{s:0,n:0},
    t2s_200:{s:0,n:0},          // 基线：质心粗筛200+精排55
    fus_55:{s:0,n:0},           // 当前最优融合
    multigrid_m10:{s:0,n:0,ms:0}, // V-cycle m=10
    multigrid_m20:{s:0,n:0,ms:0}, // V-cycle m=20
    multigrid_m30:{s:0,n:0,ms:0}, // V-cycle m=30
    blind_300:{s:0,n:0},        // 对照：直接 top-300 无 multigrid
  };

  const step=Math.max(1,Math.floor(qids.length/10));

  for(let qi=0;qi<qids.length;qi++){
    const qid=qids[qi],qVec=queryVecs[qid],qrel=qrels[qid],intId=qidMap[qid];
    const qBuf=Buffer.from(qVec.buffer,qVec.byteOffset,qVec.byteLength);

    // cosine
    const cs=allDids.map(d=>({d,s:cosSim(qVec,corpusVecs[d])})).sort((a,b)=>b.s-a.s);
    methods.cosine.s+=computeNDCG(cs.slice(0,K).map(x=>x.d),qrel,K);methods.cosine.n++;

    // ── Round 1: 质心粗筛 top-200 + token 精排 top-55 ──
    let r1Hits;
    try{ r1Hits=v.tokenChamferTwoStage(intId,200,200); }catch(e){continue}

    // t2s_200 基线
    {const ranked=r1Hits.slice(0,TOP_N).map(x=>rev[x[0]]).filter(Boolean);
    methods.t2s_200.s+=computeNDCG(ranked,qrel,K);methods.t2s_200.n++}

    // 图平滑分数（句子级 top-200）
    let lapHits;
    try{ lapHits=v.shapeLaplacianPipeline(qBuf,200,200); }catch(e){continue}

    // fus_55 基线（token top-55 + 图平滑 top-55 融合）
    {const tokMap={},lapMap={};
    for(const h of r1Hits.slice(0,TOP_N)){const s=rev[h[0]];if(s)tokMap[s]=h[1]}
    for(const h of lapHits.slice(0,TOP_N)){const s=rev[h.id];if(s)lapMap[s]=h.score}
    const all=new Set([...Object.keys(tokMap),...Object.keys(lapMap)]);
    const nt=normalizeScores(tokMap),nl=normalizeScores(lapMap);
    const fs2=[];for(const d of all)fs2.push({d,s:0.7*(nt[d]||0)+0.3*(nl[d]||0)});
    fs2.sort((a,b)=>b.s-a.s);
    methods.fus_55.s+=computeNDCG(fs2.slice(0,K).map(x=>x.d),qrel,K);methods.fus_55.n++}

    // ── 残差计算 ──
    // 取 token top-200 和 lap top-200 的交集文档
    const tokScores={},lapScores={};
    for(const h of r1Hits){const s=rev[h[0]];if(s)tokScores[s]=h[1]}
    for(const h of lapHits){const s=rev[h.id];if(s)lapScores[s]=h.score}

    const commonDocs=Object.keys(tokScores).filter(d=>lapScores[d]!==undefined);
    if(commonDocs.length<20)continue;

    // 排名归一化
    const tokVals=commonDocs.map(d=>tokScores[d]);
    const lapVals=commonDocs.map(d=>lapScores[d]);
    const tokRank=rankNormalize(tokVals);
    const lapRank=rankNormalize(lapVals);

    // 残差 = 图平滑排名 - token排名（图觉得该高但token给低的）
    const residuals=commonDocs.map((d,i)=>({d,r:lapRank[i]-tokRank[i]}));
    residuals.sort((a,b)=>b.r-a.r);

    // ── V-cycle 各种 m 值 ──
    for(const[m,name]of[[10,'multigrid_m10'],[20,'multigrid_m20'],[30,'multigrid_m30']]){
      const t=Date.now();

      // 取残差最大的 m 个文档
      const anomalies=residuals.slice(0,m).map(x=>x.d);

      // 对每个异常文档，在全库中找邻居
      const existingSet=new Set(Object.keys(tokScores));
      const expanded=new Set(existingSet);

      for(const anom of anomalies){
        const aVec=corpusVecs[anom];
        if(!aVec)continue;

        // 算跟所有文档的 cosine 距离，找最近的不在候选集中的
        const neighbors=[];
        for(const did of allDids){
          if(existingSet.has(did))continue;
          neighbors.push({d:did, s:cosSim(aVec,corpusVecs[did])});
        }
        neighbors.sort((a,b)=>b.s-a.s);

        // 取 top-5 邻居加入扩展集
        for(let i=0;i<Math.min(5,neighbors.length);i++){
          expanded.add(neighbors[i].d);
        }
      }

      // 扩展后的候选列表 → token Chamfer 重排
      const expandedIds=[...expanded].map(d=>strToInt[d]).filter(x=>x!==undefined);

      try{
        const reranked=v.tokenChamferRerankList(intId,expandedIds,TOP_N);
        const ranked=reranked.map(x=>rev[x[0]]).filter(Boolean);
        methods[name].s+=computeNDCG(ranked,qrel,K);methods[name].n++;
        methods[name].ms+=Date.now()-t;
      }catch(e){if(qi===0)console.error(`${name}:`,e.message)}
    }

    // 对照：直接 top-300（盲扩）
    {try{const h=v.tokenChamferTwoStage(intId,300,TOP_N);
    methods.blind_300.s+=computeNDCG(h.map(x=>rev[x[0]]).filter(Boolean),qrel,K);
    methods.blind_300.n++}catch(e){}}

    if((qi+1)%step===0||qi===qids.length-1)process.stdout.write(`\r  ${qi+1}/${qids.length}`);
  }
  console.log('\n');

  console.log('='.repeat(64));
  console.log('  方法'.padEnd(24)+'NDCG@10'.padEnd(12)+'vs cosine'.padEnd(12)+'延迟');
  console.log('-'.repeat(64));
  const cosAvg=methods.cosine.n>0?methods.cosine.s/methods.cosine.n:0;
  for(const[name,m]of Object.entries(methods)){
    if(m.n===0)continue;
    const avg=m.s/m.n;
    const pct=name==='cosine'?'--':`${((avg-cosAvg)/cosAvg*100).toFixed(1)}%`;
    const lat=m.ms>0?`${(m.ms/m.n).toFixed(0)}ms`:'';
    console.log(`  ${name.padEnd(24)}${avg.toFixed(4).padEnd(12)}${pct.padEnd(12)}${lat}`);
  }
  console.log('='.repeat(64));
}
main().catch(e=>{console.error(e);process.exit(1)});
