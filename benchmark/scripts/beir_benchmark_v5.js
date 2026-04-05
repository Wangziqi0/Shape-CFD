#!/usr/bin/env node
'use strict';
/**
 * BEIR V5.2 — 域适应 Allen-Cahn + Zelnik-Manor + 拓扑扩容
 * 
 * 修复: 大候选池(100/200)用 cosine 距离建图(避免 Chamfer N²爆炸)
 *       小候选池(30)用 Chamfer 距离(V4 兼容)
 */
const{Worker,isMainThread,parentPort,workerData}=require('worker_threads');
const fs=require('fs'),path=require('path'),readline=require('readline'),os=require('os');
// 32 worker — workerData 结构化克隆 32 份 ~6GB, 避免 128/240 克隆导致的 OOM
const NW=32;
// NUMA: 8 nodes, 32 threads each
const NUMA_NODES=8, THREADS_PER_NUMA=32;

if(!isMainThread){
const{queries,corpusVecs,corpusSents,corpusTexts,topN}=workerData;
function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8);}
function cosDist(a,b){return 1-cosSim(a,b);}
function pqCosDist(a,b){const NS=64,SD=64;if(a.length!==NS*SD)return cosDist(a,b);
  let t=0;for(let s=0;s<NS;s++){const o=s*SD;let d=0,na=0,nb=0;
    for(let i=0;i<SD;i++){d+=a[o+i]*b[o+i];na+=a[o+i]*a[o+i];nb+=b[o+i]*b[o+i];}
    t+=(1-d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8));}return t/NS;}
function ndcg(r,q,k=10){let d=0;for(let i=0;i<Math.min(r.length,k);i++)d+=(Math.pow(2,q[r[i]]||0)-1)/Math.log2(i+2);
  const ir=Object.values(q).sort((a,b)=>b-a);let id=0;for(let i=0;i<Math.min(ir.length,k);i++)id+=(Math.pow(2,ir[i])-1)/Math.log2(i+2);return id>0?d/id:0;}

function otsu(vals){const s=Array.from(vals).sort((a,b)=>a-b);const n=s.length;if(n<2)return s[0]||0.5;
  let best=s[n>>1],bv=-1;const step=Math.max(1,n/30|0);
  for(let t=1;t<n;t+=step){const w0=t/n,w1=1-w0;let m0=0;for(let i=0;i<t;i++)m0+=s[i];m0/=t;
    let m1=0;for(let i=t;i<n;i++)m1+=s[i];m1/=(n-t);const v=w0*w1*(m0-m1)*(m0-m1);if(v>bv){bv=v;best=(s[t-1]+s[t])/2;}}return best;}

function pde52(C0,adj,U,N,gb=1.0){
  let C=Float64Array.from(C0);
  let mn=Infinity,mx=-Infinity;for(let i=0;i<N;i++){if(C[i]<mn)mn=C[i];if(C[i]>mx)mx=C[i];}
  const delta=mx-mn+1e-6, gEff=gb/(delta*delta), theta=otsu(C);
  let mxD=0;for(let i=0;i<N;i++)if(adj[i].length>mxD)mxD=adj[i].length;
  const dt=Math.min(0.05,mxD>0?0.8/mxD:0.05,2/(gb*0.25+1e-8)*0.5);
  let Cn=new Float64Array(N);
  for(let t=0;t<80;t++){let mx2=0;
    for(let i=0;i<N;i++){let df=0,ad=0;
      for(const e of adj[i]){const j=e.j,w=e.w;df+=0.15*w*(C[j]-C[i]);
        const u1=U[i*N+j],u2=U[j*N+i];ad+=w*((u2>0?u2:0)*C[j]-(u1>0?u1:0)*C[i]);}
      const rx=gEff*(C[i]-mn)*(mx-C[i])*(C[i]-theta);
      let cn=C[i]+dt*(df+ad+rx);if(cn<mn)cn=mn;if(cn>mx)cn=mx;Cn[i]=cn;
      const d=Math.abs(cn-C[i]);if(d>mx2)mx2=d;}[C,Cn]=[Cn,C];if(mx2<5e-3)break;}return C;}

function pde4(C0,adj,U,N){
  let C=Float64Array.from(C0);let mxD=0;for(let i=0;i<N;i++)if(adj[i].length>mxD)mxD=adj[i].length;
  const dt=Math.min(0.1,mxD>0?0.8/mxD:0.1);let Cn=new Float64Array(N);
  for(let t=0;t<50;t++){let mx=0;for(let i=0;i<N;i++){let df=0,ad=0;
    for(const e of adj[i]){const j=e.j,w=e.w;df+=0.15*w*(C[j]-C[i]);
      const u1=U[i*N+j],u2=U[j*N+i];ad+=w*((u2>0?u2:0)*C[j]-(u1>0?u1:0)*C[i]);}
    const cn=Math.max(0,C[i]+dt*(df+ad));Cn[i]=cn;const d=Math.abs(cn-C[i]);if(d>mx)mx=d;}
    [C,Cn]=[Cn,C];if(mx<1e-3)break;}return C;}

const allDids=Object.keys(corpusVecs);
const dv={};for(const[id,a]of Object.entries(corpusVecs))dv[id]=new Float32Array(a);

// ── JL Projection: 4096 → 128 ──
const JL_DIM=128, ORIG_DIM=4096;
// Seeded PRNG (mulberry32) for reproducible JL matrix
function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const rng=mulberry32(42);
function boxMuller(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(u1+1e-10))*Math.cos(2*Math.PI*u2);}
const jlMatrix=new Float32Array(JL_DIM*ORIG_DIM);
for(let i=0;i<JL_DIM*ORIG_DIM;i++)jlMatrix[i]=boxMuller(rng)/Math.sqrt(JL_DIM);

function jlProject(vec){
  const out=new Float32Array(JL_DIM);
  for(let i=0;i<JL_DIM;i++){let s=0;for(let j=0;j<ORIG_DIM;j++)s+=jlMatrix[i*ORIG_DIM+j]*vec[j];out[i]=s;}
  // L2 normalize
  let norm=0;for(let i=0;i<JL_DIM;i++)norm+=out[i]*out[i];norm=Math.sqrt(norm)+1e-8;
  for(let i=0;i<JL_DIM;i++)out[i]/=norm;
  return out;
}

// Pre-project all doc vectors and sentence clouds
const dvJL={};for(const[id,v]of Object.entries(dv))dvJL[id]=jlProject(v);
const sentJL={};for(const[id,sents]of Object.entries(corpusSents)){
  if(sents)sentJL[id]=sents.map(s=>jlProject(new Float32Array(s)));
}

// ── Stefan-CFD 多轮预取核心函数 ──
// 在 PDE 收敛后，找边界高压节点，沿扩散梯度方向从全库预取新文档，扩展候选池后重跑 PDE
function stefanMulti(qV, cs, allDids, dv, corpusSents, {
  initPool=30, maxRounds=3, poolBudget=60, knn=3, probeCount=8, uStr=0.3, D=0.15
}={}){
  const dim=qV.length;
  // query 单位向量
  let qN=0;for(let d=0;d<dim;d++)qN+=qV[d]*qV[d];const iqn=1/(Math.sqrt(qN)+1e-8);

  // 初始池: HNSW top-initPool (用 cosine 排序模拟)
  const poolSet=new Set();
  const poolIds=[];
  const poolVecs=[];
  const limit=Math.min(initPool,cs.length);
  for(let i=0;i<limit;i++){
    poolIds.push(cs[i].did);poolVecs.push(dv[cs[i].did]);poolSet.add(cs[i].did);
  }

  // 获取文档点云 (句子级向量)
  function getCloud(did){
    const s=corpusSents[did];
    return s?s.map(a=>new Float32Array(a)):[dv[did]];
  }
  // 获取文档质心
  function getCentroid(did){
    const cl=getCloud(did);
    const c=new Float32Array(dim);
    for(const v of cl)for(let d=0;d<dim;d++)c[d]+=v[d];
    const inv=1/cl.length;for(let d=0;d<dim;d++)c[d]*=inv;
    return c;
  }

  // 构建 KNN 图 + PQ-Chamfer 距离矩阵 (只对池内文档)
  // distCache[i*maxN+j] 缓存已计算的距离，增量更新时复用
  const maxN=poolBudget;
  const distCache=new Float64Array(maxN*maxN);
  // 用 PQ-Chamfer 计算两个文档点云之间的距离
  function pqChamfer(clA,clB){
    let sAB=0;for(const a of clA){let mn=Infinity;for(const b of clB){const d=pqCosDist(a,b);if(d<mn)mn=d;}sAB+=mn;}
    let sBA=0;for(const b of clB){let mn=Infinity;for(const a of clA){const d=pqCosDist(a,b);if(d<mn)mn=d;}sBA+=mn;}
    return sAB/clA.length+sBA/clB.length;
  }

  // 初始距离矩阵
  const clouds=poolIds.map(id=>getCloud(id));
  let N=poolIds.length;
  for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){
    const d=pqChamfer(clouds[i],clouds[j]);
    distCache[i*maxN+j]=d;distCache[j*maxN+i]=d;
  }

  // 从距离矩阵构建 KNN 邻接表
  function buildAdj(n,k){
    const ek=Math.min(k,n-1);
    const adj=Array.from({length:n},()=>[]);
    for(let i=0;i<n;i++){
      const nb=[];for(let j=0;j<n;j++)if(j!==i)nb.push({j,d:distCache[i*maxN+j]});
      nb.sort((a,b)=>a.d-b.d);
      for(let t=0;t<ek;t++)adj[i].push({j:nb[t].j,w:Math.exp(-2*nb[t].d)});
    }
    // 对称化
    for(let i=0;i<n;i++)for(const e of adj[i])if(!adj[e.j].some(x=>x.j===i))adj[e.j].push({j:i,w:e.w});
    return adj;
  }

  // 计算对流系数矩阵 U
  function buildU(adj,n,centroids){
    const U=new Float64Array(n*n);
    for(let i=0;i<n;i++)for(const e of adj[i]){const j=e.j;
      if(U[i*n+j]||U[j*n+i])continue;
      let en=0,dvv=0;
      for(let d=0;d<dim;d++){const df=centroids[j][d]-centroids[i][d];en+=df*df;dvv+=df*qV[d]*iqn;}
      const u0=(dvv/(Math.sqrt(en)+1e-8))*uStr;
      U[i*n+j]=u0;U[j*n+i]=-u0;
    }
    return U;
  }

  // 用 pde4 求解浓度场
  function solvePDE(C0,adj,U,n){
    let C=Float64Array.from(C0);let mxD=0;
    for(let i=0;i<n;i++)if(adj[i].length>mxD)mxD=adj[i].length;
    const dt=Math.min(0.1,mxD>0?0.8/mxD:0.1);let Cn=new Float64Array(n);
    for(let t=0;t<50;t++){let mx=0;for(let i=0;i<n;i++){let df=0,ad=0;
      for(const e of adj[i]){const j=e.j,w=e.w;df+=D*w*(C[j]-C[i]);
        const u1=U[i*n+j],u2=U[j*n+i];ad+=w*((u2>0?u2:0)*C[j]-(u1>0?u1:0)*C[i]);}
      const cn=Math.max(0,C[i]+dt*(df+ad));Cn[i]=cn;const d=Math.abs(cn-C[i]);if(d>mx)mx=d;}
      [C,Cn]=[Cn,C];if(mx<1e-3)break;}
    return C;
  }

  // query-doc Chamfer 距离 (用 PQ cosine)
  function qdChamfer(did){
    const cl=getCloud(did);
    let sAB=pqCosDist(qV,cl[0]);for(let k=1;k<cl.length;k++){const d=pqCosDist(qV,cl[k]);if(d<sAB)sAB=d;}
    let sBA=0;for(const b of cl)sBA+=pqCosDist(b,qV);
    return sAB+sBA/cl.length;
  }

  let maxFluxR1=0; // 第 1 轮最大通量，用于停止条件

  // ── Round 0: 初始 PDE ──
  let centroids=poolIds.map(id=>getCentroid(id));
  let adj=buildAdj(N,knn);
  let U=buildU(adj,N,centroids);
  let C0=new Float64Array(N);
  for(let i=0;i<N;i++)C0[i]=Math.exp(-2*qdChamfer(poolIds[i]));
  let C=solvePDE(C0,adj,U,N);

  // ── Round 1..R: 边界预取 ──
  for(let r=1;r<=maxRounds;r++){
    if(N>=poolBudget)break; // 池已满

    // 找边界节点: degree < knn 且 C_i > median(C)
    const cVals=Array.from(C).sort((a,b)=>a-b);
    const median=cVals[cVals.length>>1];

    // 计算通量
    const fluxes=[];
    for(let i=0;i<N;i++){
      if(adj[i].length>=knn*2||C[i]<=median)continue; // 非边界或浓度太低
      // flux_i = C_i * max(0, dot(n_i, q_hat))
      // n_i = 从节点 i 质心指向 query 方向的单位向量
      const ci=centroids[i];
      let dot=0;for(let d=0;d<dim;d++)dot+=(qV[d]*iqn-ci[d])*(qV[d]*iqn);
      const flux=C[i]*Math.max(0,dot);
      if(flux>0)fluxes.push({idx:i,flux});
    }
    if(fluxes.length===0)break;
    fluxes.sort((a,b)=>b.flux-a.flux);
    const maxFlux=fluxes[0].flux;

    // 第 1 轮记录基准通量
    if(r===1)maxFluxR1=maxFlux;
    // 停止条件: 相对阈值
    if(r>1&&maxFluxR1>0&&maxFlux/maxFluxR1<0.1)break;

    // 取 top-m 探针
    const m=Math.min(probeCount,fluxes.length);
    const probes=fluxes.slice(0,m);

    // 构造探针向量并从全库检索
    const newDids=[];
    const budget=poolBudget-N; // 剩余配额
    const perProbe=Math.max(1,Math.ceil(budget/m));

    for(const p of probes){
      if(newDids.length>=budget)break;
      // probe_i = 0.7 * q_centroid + 0.3 * doc_centroid_i
      const probeVec=new Float32Array(dim);
      const dc=centroids[p.idx];
      for(let d=0;d<dim;d++)probeVec[d]=0.7*qV[d]+0.3*dc[d];

      // 用探针向量对全库做 cosine 检索 (模拟 HNSW)
      const hits=[];
      for(const did of allDids){
        if(poolSet.has(did))continue;
        let dot=0,na=0,nb=0;
        const dVec=dv[did];
        for(let d=0;d<dim;d++){dot+=probeVec[d]*dVec[d];na+=probeVec[d]*probeVec[d];nb+=dVec[d]*dVec[d];}
        hits.push({did,s:dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-8)});
      }
      hits.sort((a,b)=>b.s-a.s);

      const take=Math.min(perProbe,budget-newDids.length);
      for(let t=0;t<Math.min(take,hits.length);t++){
        const did=hits[t].did;
        if(!poolSet.has(did)){newDids.push(did);poolSet.add(did);}
      }
    }

    if(newDids.length===0)break;

    // 将新文档加入池
    const oldN=N;
    for(const did of newDids){
      poolIds.push(did);poolVecs.push(dv[did]);
      clouds.push(getCloud(did));
    }
    N=poolIds.length;

    // 增量更新距离矩阵: 只计算新文档与旧池 + 新文档之间的距离
    for(let i=oldN;i<N;i++){
      for(let j=0;j<N;j++){
        if(i===j)continue;
        const d=pqChamfer(clouds[i],clouds[j]);
        distCache[i*maxN+j]=d;distCache[j*maxN+i]=d;
      }
    }

    // 重建图、对流系数、初始浓度
    centroids=poolIds.map(id=>getCentroid(id));
    adj=buildAdj(N,knn);
    U=buildU(adj,N,centroids);

    // 新初始浓度: 旧节点保留上轮 PDE 浓度，新节点用 exp(-2*chamfer)
    const C0new=new Float64Array(N);
    for(let i=0;i<oldN;i++)C0new[i]=C[i];
    for(let i=oldN;i<N;i++)C0new[i]=Math.exp(-2*qdChamfer(poolIds[i]));
    C=solvePDE(C0new,adj,U,N);
  }

  // 最终按浓度降序排列
  const ranked=poolIds.map((id,i)=>({did:id,s:C[i]})).sort((a,b)=>b.s-a.s);
  return ranked;
}

const methods=['cosine','ad_rank_v2','shape_v4','shape_pq64','bm25',
  'shape_jl128','stefan_multi',
  'mix_a0.5','mix_a0.6','mix_a0.7','mix_a0.8','mix_a0.9','mix_a1.0'];
const R={};for(const m of methods)R[m]=0;

for(const q of queries){
  const qV=new Float32Array(q.vec),qr=q.qrels;
  const cs=allDids.map(did=>({did,s:cosSim(qV,dv[did])}));cs.sort((a,b)=>b.s-a.s);
  R.cosine+=ndcg(cs.slice(0,10).map(d=>d.did),qr);

  // ── Stefan-CFD 多轮预取 ──
  {
    const ranked=stefanMulti(qV,cs,allDids,dv,corpusSents,{
      initPool:topN, maxRounds:3, poolBudget:60, knn:3, probeCount:8, uStr:0.3, D:0.15
    });
    R.stefan_multi+=ndcg(ranked.slice(0,10).map(x=>x.did),qr);
  }

  const cfgs=[
    {n:'ad_rank_v2',p:topN,k:3,m:'v2'},
    {n:'shape_v4',p:topN,k:3,m:'v4'},
    {n:'shape_pq64',p:topN,k:3,m:'pq64'},
    {n:'bm25',p:topN,k:3,m:'bm25'},
    {n:'shape_jl128',p:topN,k:3,m:'jl128'},
    // 混合初始场 α 精细网格: Chamfer图 + V4纯扩散
    {n:'mix_a0.5',p:topN,k:3,m:'mix',alpha:0.5},
    {n:'mix_a0.6',p:topN,k:3,m:'mix',alpha:0.6},
    {n:'mix_a0.7',p:topN,k:3,m:'mix',alpha:0.7},
    {n:'mix_a0.8',p:topN,k:3,m:'mix',alpha:0.8},
    {n:'mix_a0.9',p:topN,k:3,m:'mix',alpha:0.9},
    {n:'mix_a1.0',p:topN,k:3,m:'mix',alpha:1.0},
  ];

  for(const c of cfgs){
    const pool=Math.min(c.p,cs.length);
    const cIds=cs.slice(0,pool).map(x=>x.did);
    const cV=cIds.map(id=>dv[id]);
    const N=cIds.length,dim=qV.length;

    if(c.m==='bm25'){
      const tok=t=>t.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff]/g,' ').split(/\s+/).filter(t=>t.length>1);
      const qt=tok(q.text||''),dt=cIds.map(id=>tok(corpusTexts[id]||''));
      const al=dt.reduce((s,d)=>s+d.length,0)/N;
      const df={};for(const ts of dt){const sn=new Set(ts);for(const t of sn)df[t]=(df[t]||0)+1;}
      const r=cIds.map((id,i)=>{const ts=dt[i],dl=ts.length,tf={};for(const t of ts)tf[t]=(tf[t]||0)+1;
        let sc=0;for(const ql of qt){if(!tf[ql])continue;const idf=Math.log((N-(df[ql]||0)+0.5)/((df[ql]||0)+0.5)+1);
          sc+=idf*(tf[ql]*2.2)/(tf[ql]+1.2*(0.25+0.75*dl/al));}return{did:id,s:sc};}).sort((a,b)=>b.s-a.s);
      R[c.n]+=ndcg(r.map(x=>x.did),qr);continue;
    }

    // 图距离: top-30用Chamfer(V4兼容), >30用cosine(快)
    // jl128模式: 用JL投影后的128维向量计算Chamfer
    const useJL=c.m==='jl128';
    const usePQ=c.m==='pq64';
    const useCham=pool<=30;
    const dist=new Float64Array(N*N);
    if(useCham){
      const cCl=useJL
        ? cIds.map(id=>sentJL[id]||[dvJL[id]])
        : cIds.map(id=>{const s=corpusSents[id];return s?s.map(a=>new Float32Array(a)):[dv[id]];});
      const dFn=usePQ?pqCosDist:cosDist;
      for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){
        let sAB=0;for(const a of cCl[i]){let mn=Infinity;for(const b of cCl[j]){const d=dFn(a,b);if(d<mn)mn=d;}sAB+=mn;}
        let sBA=0;for(const b of cCl[j]){let mn=Infinity;for(const a of cCl[i]){const d=dFn(a,b);if(d<mn)mn=d;}sBA+=mn;}
        const d=sAB/cCl[i].length+sBA/cCl[j].length;dist[i*N+j]=d;dist[j*N+i]=d;}
    } else {
      for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){const d=cosDist(cV[i],cV[j]);dist[i*N+j]=d;dist[j*N+i]=d;}
    }

    // query-doc
    const qdDist=new Float64Array(N);
    if(useCham){
      const qVp=useJL?jlProject(qV):qV;
      const cCl=useJL
        ? cIds.map(id=>sentJL[id]||[dvJL[id]])
        : cIds.map(id=>{const s=corpusSents[id];return s?s.map(a=>new Float32Array(a)):[dv[id]];});
      const dFn2=usePQ?pqCosDist:cosDist;
      for(let i=0;i<N;i++){const cl=cCl[i];
        let sAB=dFn2(qVp,cl[0]);for(let k=1;k<cl.length;k++){const d=dFn2(qVp,cl[k]);if(d<sAB)sAB=d;}
        let sBA=0;for(const b of cl){sBA+=dFn2(b,qVp);}
        qdDist[i]=sAB+sBA/cl.length;}
    } else {
      for(let i=0;i<N;i++)qdDist[i]=cosDist(qV,cV[i]);
    }

    // KNN
    const ek=Math.min(c.k,N-1);
    const adj=Array.from({length:N},()=>[]);
    for(let i=0;i<N;i++){const nb=[];for(let j=0;j<N;j++)if(j!==i)nb.push({j,d:dist[i*N+j]});
      nb.sort((a,b)=>a.d-b.d);for(let t=0;t<ek;t++)adj[i].push({j:nb[t].j,w:Math.exp(-2*nb[t].d)});}
    for(let i=0;i<N;i++)for(const e of adj[i])if(!adj[e.j].some(x=>x.j===i))adj[e.j].push({j:i,w:e.w});

    // 质心 + advection (jl128模式用投影+L2归一化后的质心)
    const advVecs=useJL?cIds.map(id=>dvJL[id]):cV;
    const advQ=useJL?jlProject(qV):qV;
    const advDim=useJL?JL_DIM:dim;
    let qN2=0;for(let d=0;d<advDim;d++)qN2+=advQ[d]*advQ[d];const iqn=1/(Math.sqrt(qN2)+1e-8);
    const sigmas=new Float64Array(N);
    for(let i=0;i<N;i++){
      const norms=[];for(let j=0;j<N;j++){if(j===i)continue;
        let en=0;for(let d=0;d<advDim;d++){const df=advVecs[j][d]-advVecs[i][d];en+=df*df;}norms.push(en);}
      norms.sort((a,b)=>a-b);sigmas[i]=norms[Math.min(c.k-1,norms.length-1)]+1e-10;
    }

    const U=new Float64Array(N*N);
    for(let i=0;i<N;i++)for(const e of adj[i]){const j=e.j;if(U[i*N+j]||U[j*N+i])continue;
      let en=0,dvv=0;for(let d=0;d<advDim;d++){const df=advVecs[j][d]-advVecs[i][d];en+=df*df;dvv+=df*advQ[d]*iqn;}
      const u0=(dvv/(Math.sqrt(en)+1e-8))*((c.m==='v4'||c.m==='jl128')?0.3:0.1);
      if(c.m==='v52'){
        const soft=Math.exp(-en/(sigmas[i]*sigmas[j]));
        U[i*N+j]=u0*soft;U[j*N+i]=-u0*soft;
      }else{U[i*N+j]=u0;U[j*N+i]=-u0;}
    }

    const C0=new Float64Array(N);
    if(c.m==='v2'){for(let i=0;i<N;i++)C0[i]=cosSim(qV,cV[i]);}
    else if(c.m==='mix'){
      // 混合初始场: α·MaxSim + (1-α)·MeanSim
      // 必须用 Chamfer 图（pool<=30 已保证）
      const cCl=cIds.map(id=>{const s=corpusSents[id];return s?s.map(a=>new Float32Array(a)):[dv[id]];});
      for(let i=0;i<N;i++){const cl=cCl[i];
        // MaxSim: query 到 doc 最近句子的余弦相似度
        let maxS=-Infinity;for(const v of cl){const s=cosSim(qV,v);if(s>maxS)maxS=s;}
        // MeanSim: query 到 doc 所有句子的平均余弦相似度
        let sumS=0;for(const v of cl)sumS+=cosSim(qV,v);const meanS=sumS/cl.length;
        C0[i]=c.alpha*maxS+(1-c.alpha)*meanS;
      }
    }
    else{for(let i=0;i<N;i++)C0[i]=Math.exp(-2*qdDist[i]);}

    let Cf;
    if(c.m==='v52') Cf=pde52(C0,adj,U,N,c.g||1.0);
    else Cf=pde4(C0,adj,U,N); // mix 用 V4 纯扩散

    const ranked=cIds.map((id,i)=>({did:id,s:Cf[i]})).sort((a,b)=>b.s-a.s).map(x=>x.did);
    R[c.n]+=ndcg(ranked,qr);
  }
  // 进度探针: 每完成一个 query 上报
  parentPort.postMessage({type:'progress',wid:workerData.wid});
}
parentPort.postMessage({type:'result',data:R});process.exit(0);
}

// ── MAIN ──
const args=process.argv.slice(2);
const runAll=args.includes('--all');
const datasets=runAll?['nfcorpus','scifact']:[args.find((_,i)=>args[i-1]==='--dataset')||'nfcorpus'];
const dataDir=args.find((_,i)=>args[i-1]==='--data-dir')||'./beir_data';
const topN=parseInt(args.find((_,i)=>args[i-1]==='--topn')||'30');

function loadJsonl(fp){return new Promise((r,j)=>{const a=[];const rl=readline.createInterface({input:fs.createReadStream(fp,{encoding:'utf-8'}),crlfDelay:Infinity});
  rl.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l));}catch(e){}});rl.on('close',()=>r(a));rl.on('error',j);});}
function loadQrels(fp){const q={};const ls=fs.readFileSync(fp,'utf-8').trim().split('\n');
  for(let i=1;i<ls.length;i++){const[qi,di,s]=ls[i].split('\t');if(!q[qi])q[qi]={};q[qi][di]=parseInt(s);}return q;}

// NUMA node map for CPU affinity hints
const NUMA_MAP=[
  {node:0,cpus:'0-15,128-143'},{node:1,cpus:'16-31,144-159'},
  {node:2,cpus:'32-47,160-175'},{node:3,cpus:'48-63,176-191'},
  {node:4,cpus:'64-79,192-207'},{node:5,cpus:'80-95,208-223'},
  {node:6,cpus:'96-111,224-239'},{node:7,cpus:'112-127,240-255'},
];

async function run(ds){
  const dd=path.join(dataDir,ds);
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  V5.2 — ${ds} (域适应PDE + Zelnik-Manor + cosine图扩容)`);
  console.log(`  ${NW} worker | 进度探针`);
  console.log(`${'═'.repeat(60)}\n`);
  const corpus={};for(const o of await loadJsonl(path.join(dd,'corpus.jsonl')))corpus[o._id]=o;
  const qV={},qT={};for(const o of await loadJsonl(path.join(dd,'query_vectors.jsonl'))){qV[o._id]=Array.from(new Float32Array(o.vector));qT[o._id]=o.text;}
  const cV={},cS={},cT={};
  for(const o of await loadJsonl(path.join(dd,'corpus_vectors.jsonl'))){
    cV[o._id]=Array.from(new Float32Array(o.vector));if(o.sentences?.length>1)cS[o._id]=o.sentences;cT[o._id]=corpus[o._id]?.text||'';}
  const qrels=loadQrels(path.join(dd,'qrels.tsv'));
  const qids=Object.keys(qrels).filter(q=>qV[q]);
  const effectiveNW=Math.min(NW,qids.length); // 不创建空 worker
  console.log(`  Q:${qids.length} C:${Object.keys(cV).length} W:${effectiveNW} Mem:${(process.memoryUsage().heapUsed/1024/1024|0)}MB`);
  console.log(`  NUMA:${NUMA_NODES} nodes × ${THREADS_PER_NUMA} threads\n`);

  // 均匀分配 query 到 worker（round-robin NUMA 感知）
  const batches=Array.from({length:effectiveNW},()=>[]);
  for(let i=0;i<qids.length;i++){const qi=qids[i];batches[i%effectiveNW].push({vec:qV[qi],text:qT[qi],qrels:qrels[qi]});}

  // 进度探针
  let completed=0;
  const totalQ=qids.length;
  const t0=performance.now();
  const progressTimer=setInterval(()=>{
    const elapsed=(performance.now()-t0)/1000;
    const qps=completed/elapsed||0;
    const eta=qps>0?((totalQ-completed)/qps):0;
    const pct=(completed/totalQ*100).toFixed(1);
    const bar='█'.repeat(Math.floor(completed/totalQ*30))+'░'.repeat(30-Math.floor(completed/totalQ*30));
    const mem=(process.memoryUsage().heapUsed/1024/1024|0);
    process.stdout.write(`\r  ${bar} ${pct}% | ${completed}/${totalQ} | ${qps.toFixed(1)} q/s | ETA ${eta.toFixed(0)}s | ${mem}MB  `);
  },2000);

  console.log(`  ⏳ ${effectiveNW} workers launching...`);

  const wr=await Promise.all(batches.map((b,i)=>{
    if(!b.length)return Promise.resolve({});
    return new Promise((r,j)=>{
      const w=new Worker(__filename,{
        workerData:{queries:b,corpusVecs:cV,corpusSents:cS,corpusTexts:cT,topN,wid:i},
      });
      w.on('message',msg=>{
        if(msg.type==='progress')completed++;
        else if(msg.type==='result')r(msg.data);
      });
      w.on('error',j);
      w.on('exit',c=>{if(c!==0)j(new Error(`W${i}:${c}`));});
    });
  }));

  clearInterval(progressTimer);
  const sec=(performance.now()-t0)/1000;
  process.stdout.write('\r' + ' '.repeat(100) + '\r'); // clear progress line

  const methods=['cosine','ad_rank_v2','shape_v4','shape_pq64','bm25','shape_jl128','stefan_multi','mix_a0.5','mix_a0.6','mix_a0.7','mix_a0.8','mix_a0.9','mix_a1.0'];
  const nd={};for(const m of methods)nd[m]=0;for(const w of wr)for(const m of methods)nd[m]+=(w[m]||0);
  const nQ=qids.length,v4=nd.shape_v4/nQ;
  console.log(`\n  ✅ ${sec.toFixed(1)}s (${(nQ/sec).toFixed(1)} q/s) | Peak Mem: ${(process.memoryUsage().heapUsed/1024/1024|0)}MB\n`);
  console.log('  ┌─────────────────────┬──────────┬────────────┐');
  console.log('  │ Method              │ NDCG@10  │ vs V4      │');
  console.log('  ├─────────────────────┼──────────┼────────────┤');
  for(const m of methods){const n=nd[m]/nQ;
    const d=(m.startsWith('v5')||m.startsWith('mix')||m.startsWith('stefan'))?`${n>v4?'+':''}${((n-v4)/v4*100).toFixed(1)}%`:'';
    console.log(`  │ ${m.padEnd(19)} │ ${n.toFixed(4).padStart(8)} │ ${d.padStart(10)} │`);}
  console.log('  └─────────────────────┴──────────┴────────────┘');
  return Object.fromEntries(methods.map(m=>[m,+(nd[m]/nQ).toFixed(4)]));
}
(async()=>{const a={};for(const d of datasets)a[d]=await run(d);
  fs.writeFileSync(path.join(dataDir,'v5.2_results.json'),JSON.stringify(a,null,2));
  console.log(`\n💾 v5.2_results.json\n✅ 全部完成`);
})().catch(e=>{console.error('❌',e);process.exit(1);});
