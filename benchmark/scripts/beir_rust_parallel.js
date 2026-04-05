#!/usr/bin/env node
'use strict';
/**
 * beir_rust_parallel.js — 64-worker 并行 Benchmark
 * cosine | shape_pq64 | vt_aligned | vt_unaligned | stefan_rust | stefan_full | v7_adjoint | vt_v7
 */
const {Worker,isMainThread,parentPort,workerData}=require('worker_threads');
const fs=require('fs'),path=require('path'),readline=require('readline');
const NW=64;

// ════════════════════════════════════════════════════════════════
// WORKER
// ════════════════════════════════════════════════════════════════
if(!isMainThread){
const{queries,corpusVecs,corpusSents,allDids,topN,idMap,reverseMap,sqlitePath}=workerData;
const DIM=4096,NS=64,SD=64;

// Rust addon
let vexus=null;
try{
  const{LawVexus}=require('/home/amd/HEZIMENG/law-vexus');
  vexus=new LawVexus('/tmp/beir_w'+process.pid);
  vexus.loadClouds(sqlitePath);
}catch(e){console.error('Worker Rust加载失败:',e.message);}

const dv=corpusVecs;

function cosSim(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8);}
function pqCosDist(a,b){if(a.length!==NS*SD)return 1-cosSim(a,b);let t=0;for(let s=0;s<NS;s++){const o=s*SD;let d=0,na=0,nb=0;for(let i=0;i<SD;i++){d+=a[o+i]*b[o+i];na+=a[o+i]*a[o+i];nb+=b[o+i]*b[o+i];}t+=(1-d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8));}return t/NS;}
function ndcg(r,q,k=10){let d=0;for(let i=0;i<Math.min(r.length,k);i++)d+=(Math.pow(2,q[r[i]]||0)-1)/Math.log2(i+2);const ir=Object.values(q).sort((a,b)=>b-a);let id=0;for(let i=0;i<Math.min(ir.length,k);i++)id+=(Math.pow(2,ir[i])-1)/Math.log2(i+2);return id>0?d/id:0;}

function pde4(C0,adj,U,N){let C=Float64Array.from(C0);let mxD=0;for(let i=0;i<N;i++)if(adj[i].length>mxD)mxD=adj[i].length;
  const dt=Math.min(0.1,mxD>0?0.8/mxD:0.1);let Cn=new Float64Array(N);
  for(let t=0;t<50;t++){let mx=0;for(let i=0;i<N;i++){let df=0,ad=0;
    for(const e of adj[i]){const j=e.j,w=e.w;df+=0.15*w*(C[j]-C[i]);const u1=U[i*N+j],u2=U[j*N+i];ad+=w*((u2>0?u2:0)*C[j]-(u1>0?u1:0)*C[i]);}
    const cn=Math.max(0,C[i]+dt*(df+ad));Cn[i]=cn;const d=Math.abs(cn-C[i]);if(d>mx)mx=d;}
    [C,Cn]=[Cn,C];if(mx<1e-3)break;}return C;}

function getCloud(did){const s=corpusSents[did];return s?s.map(a=>new Float32Array(a)):[dv[did]];}
function getCentroid(did){const cl=getCloud(did);const c=new Float32Array(DIM);for(const v of cl)for(let d=0;d<DIM;d++)c[d]+=v[d];const inv=1/cl.length;for(let d=0;d<DIM;d++)c[d]*=inv;return c;}

function pqChamfer(clA,clB){
  let sAB=0;for(const a of clA){let mn=1e9;for(const b of clB){const d=pqCosDist(a,b);if(d<mn)mn=d;}sAB+=mn;}
  let sBA=0;for(const b of clB){let mn=1e9;for(const a of clA){const d=pqCosDist(a,b);if(d<mn)mn=d;}sBA+=mn;}
  return sAB/clA.length+sBA/clB.length;
}

function qdChamfer(qV,did){const cl=getCloud(did);let sAB=pqCosDist(qV,cl[0]);for(let k=1;k<cl.length;k++){const d=pqCosDist(qV,cl[k]);if(d<sAB)sAB=d;}let sBA=0;for(const b of cl)sBA+=pqCosDist(b,qV);return sAB+sBA/cl.length;}

// ── 虚拟 token 距离函数 ──────────────────────────────────────
// vt_aligned: 对齐虚拟 token — 每个子空间 s 独立做句子级 Chamfer，64 个子空间取平均
function vtAlignedDist(qV, docCloud) {
  const NS = 64, SD = 64;
  let totalDist = 0;
  for (let s = 0; s < NS; s++) {
    const off = s * SD;
    // 正向：query 的第 s 个子空间在 docCloud 中找最近句子
    let minDist = Infinity;
    for (const sent of docCloud) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += qV[off + i] * sent[off + i]; na += qV[off + i] * qV[off + i]; nb += sent[off + i] * sent[off + i]; }
      const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
      if (d < minDist) minDist = d;
    }
    totalDist += minDist;
    // 反向：每个句子的第 s 个子空间到 query（query 只有 1 点），取平均
    let sumReverse = 0;
    for (const sent of docCloud) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += sent[off + i] * qV[off + i]; na += sent[off + i] * sent[off + i]; nb += qV[off + i] * qV[off + i]; }
      sumReverse += 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
    }
    totalDist += sumReverse / docCloud.length;
  }
  return totalDist / NS;
}

// vt_unaligned: 非对齐虚拟 token — 展开为 64d token 做全 Chamfer
function vtUnalignedDist(qV, docCloud) {
  const NS = 64, SD = 64;
  // query 展开为 64 个 64d 虚拟 token
  const qTokens = [];
  for (let s = 0; s < NS; s++) qTokens.push(qV.subarray ? qV.subarray(s * SD, (s + 1) * SD) : new Float32Array(qV.buffer || qV, s * SD * 4, SD));
  // doc 展开：限制最多 5 个句子（防止 O(64*320) 太慢）
  const maxSent = 5;
  const dTokens = [];
  const useSents = docCloud.length <= maxSent ? docCloud : docCloud.slice(0, maxSent);
  for (const sent of useSents) {
    for (let s = 0; s < NS; s++) {
      dTokens.push(sent.subarray ? sent.subarray(s * SD, (s + 1) * SD) : new Float32Array(sent.buffer || sent, s * SD * 4, SD));
    }
  }
  // Chamfer: query→doc
  let sAB = 0;
  for (const qt of qTokens) {
    let mn = Infinity;
    for (const dt of dTokens) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += qt[i] * dt[i]; na += qt[i] * qt[i]; nb += dt[i] * dt[i]; }
      const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
      if (d < mn) mn = d;
    }
    sAB += mn;
  }
  // Chamfer: doc→query
  let sBA = 0;
  for (const dt of dTokens) {
    let mn = Infinity;
    for (const qt of qTokens) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < SD; i++) { dot += qt[i] * dt[i]; na += qt[i] * qt[i]; nb += dt[i] * dt[i]; }
      const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
      if (d < mn) mn = d;
    }
    sBA += mn;
  }
  return sAB / qTokens.length + sBA / dTokens.length;
}

// vt_aligned doc-doc 距离：两个 doc 点云之间的对齐虚拟 token Chamfer
function vtAlignedChamfer(clA, clB) {
  const NS = 64, SD = 64;
  let totalDist = 0;
  for (let s = 0; s < NS; s++) {
    const off = s * SD;
    // 正向：clA 每个句子的第 s 个子空间在 clB 中找最近
    let sAB = 0;
    for (const a of clA) {
      let mn = Infinity;
      for (const b of clB) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < SD; i++) { dot += a[off + i] * b[off + i]; na += a[off + i] * a[off + i]; nb += b[off + i] * b[off + i]; }
        const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        if (d < mn) mn = d;
      }
      sAB += mn;
    }
    // 反向
    let sBA = 0;
    for (const b of clB) {
      let mn = Infinity;
      for (const a of clA) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < SD; i++) { dot += a[off + i] * b[off + i]; na += a[off + i] * a[off + i]; nb += b[off + i] * b[off + i]; }
        const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
        if (d < mn) mn = d;
      }
      sBA += mn;
    }
    totalDist += sAB / clA.length + sBA / clB.length;
  }
  return totalDist / NS;
}

// vt_unaligned doc-doc 距离：两个 doc 点云展开为 64d token 做全 Chamfer
function vtUnalignedChamfer(clA, clB) {
  const NS = 64, SD = 64, maxSent = 5;
  const useA = clA.length <= maxSent ? clA : clA.slice(0, maxSent);
  const useB = clB.length <= maxSent ? clB : clB.slice(0, maxSent);
  const tokA = [], tokB = [];
  for (const v of useA) for (let s = 0; s < NS; s++) tokA.push(v.subarray ? v.subarray(s * SD, (s + 1) * SD) : new Float32Array(v.buffer || v, s * SD * 4, SD));
  for (const v of useB) for (let s = 0; s < NS; s++) tokB.push(v.subarray ? v.subarray(s * SD, (s + 1) * SD) : new Float32Array(v.buffer || v, s * SD * 4, SD));
  let sAB = 0;
  for (const a of tokA) { let mn = Infinity; for (const b of tokB) { let dot = 0, na = 0, nb = 0; for (let i = 0; i < SD; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8); if (d < mn) mn = d; } sAB += mn; }
  let sBA = 0;
  for (const b of tokB) { let mn = Infinity; for (const a of tokA) { let dot = 0, na = 0, nb = 0; for (let i = 0; i < SD; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } const d = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8); if (d < mn) mn = d; } sBA += mn; }
  return sAB / tokA.length + sBA / tokB.length;
}

// shapeVtAligned: 对齐虚拟 token KNN 图 + PDE（流程同 shapePq64）
function shapeVtAligned(qV, poolIds) {
  const N = poolIds.length; if (N === 0) return [];
  const clouds = poolIds.map(id => getCloud(id)); const knn = 3;
  // doc-doc 距离矩阵（用 vtAlignedChamfer）
  const dist = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const d = vtAlignedChamfer(clouds[i], clouds[j]); dist[i * N + j] = d; dist[j * N + i] = d; }
  // KNN 图
  const ek = Math.min(knn, N - 1); const adj = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) { const nb = []; for (let j = 0; j < N; j++) if (j !== i) nb.push({ j, d: dist[i * N + j] }); nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) }); }
  for (let i = 0; i < N; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });
  // 对流系数
  let qN = 0; for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d]; const iqn = 1 / (Math.sqrt(qN) + 1e-8);
  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (const e of adj[i]) { const j = e.j; if (U[i * N + j] || U[j * N + i]) continue; let en = 0, dvv = 0; for (let d = 0; d < DIM; d++) { const df = dv[poolIds[j]][d] - dv[poolIds[i]][d]; en += df * df; dvv += df * qV[d] * iqn; } const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * 0.3; U[i * N + j] = u0; U[j * N + i] = -u0; }
  // 初始浓度：用 vtAlignedDist
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
  const Cf = pde4(C0, adj, U, N);
  return poolIds.map((id, i) => ({ id, s: Cf[i] })).sort((a, b) => b.s - a.s).map(x => x.id);
}

// shapeVtUnaligned: 非对齐虚拟 token KNN 图 + PDE（流程同 shapePq64）
function shapeVtUnaligned(qV, poolIds) {
  const N = poolIds.length; if (N === 0) return [];
  const clouds = poolIds.map(id => getCloud(id)); const knn = 3;
  // doc-doc 距离矩阵（用 vtUnalignedChamfer）
  const dist = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const d = vtUnalignedChamfer(clouds[i], clouds[j]); dist[i * N + j] = d; dist[j * N + i] = d; }
  // KNN 图
  const ek = Math.min(knn, N - 1); const adj = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) { const nb = []; for (let j = 0; j < N; j++) if (j !== i) nb.push({ j, d: dist[i * N + j] }); nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) }); }
  for (let i = 0; i < N; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });
  // 对流系数
  let qN = 0; for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d]; const iqn = 1 / (Math.sqrt(qN) + 1e-8);
  const U = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (const e of adj[i]) { const j = e.j; if (U[i * N + j] || U[j * N + i]) continue; let en = 0, dvv = 0; for (let d = 0; d < DIM; d++) { const df = dv[poolIds[j]][d] - dv[poolIds[i]][d]; en += df * df; dvv += df * qV[d] * iqn; } const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * 0.3; U[i * N + j] = u0; U[j * N + i] = -u0; }
  // 初始浓度：用 vtUnalignedDist
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * vtUnalignedDist(qV, clouds[i]));
  const Cf = pde4(C0, adj, U, N);
  return poolIds.map((id, i) => ({ id, s: Cf[i] })).sort((a, b) => b.s - a.s).map(x => x.id);
}

// shape_pq64: PQ-Chamfer KNN 图 + PDE
function shapePq64(qV,poolIds){
  const N=poolIds.length;if(N===0)return[];
  const clouds=poolIds.map(id=>getCloud(id));const knn=3;
  const dist=new Float64Array(N*N);
  for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){const d=pqChamfer(clouds[i],clouds[j]);dist[i*N+j]=d;dist[j*N+i]=d;}
  const ek=Math.min(knn,N-1);const adj=Array.from({length:N},()=>[]);
  for(let i=0;i<N;i++){const nb=[];for(let j=0;j<N;j++)if(j!==i)nb.push({j,d:dist[i*N+j]});nb.sort((a,b)=>a.d-b.d);for(let t=0;t<ek;t++)adj[i].push({j:nb[t].j,w:Math.exp(-2*nb[t].d)});}
  for(let i=0;i<N;i++)for(const e of adj[i])if(!adj[e.j].some(x=>x.j===i))adj[e.j].push({j:i,w:e.w});
  let qN=0;for(let d=0;d<DIM;d++)qN+=qV[d]*qV[d];const iqn=1/(Math.sqrt(qN)+1e-8);
  const U=new Float64Array(N*N);
  for(let i=0;i<N;i++)for(const e of adj[i]){const j=e.j;if(U[i*N+j]||U[j*N+i])continue;let en=0,dvv=0;for(let d=0;d<DIM;d++){const df=dv[poolIds[j]][d]-dv[poolIds[i]][d];en+=df*df;dvv+=df*qV[d]*iqn;}const u0=(dvv/(Math.sqrt(en)+1e-8))*0.3;U[i*N+j]=u0;U[j*N+i]=-u0;}
  const C0=new Float64Array(N);for(let i=0;i<N;i++)C0[i]=Math.exp(-2*qdChamfer(qV,poolIds[i]));
  const Cf=pde4(C0,adj,U,N);
  return poolIds.map((id,i)=>({id,s:Cf[i]})).sort((a,b)=>b.s-a.s).map(x=>x.id);
}

// stefan_rust: Stefan 预取 + Rust 探针
function stefanRust(qV,initCandidates){
  const initPool=topN,maxRounds=3,poolBudget=60,knn=3,probeCount=8,uStr=0.3,D=0.15;
  let qN=0;for(let d=0;d<DIM;d++)qN+=qV[d]*qV[d];const iqn=1/(Math.sqrt(qN)+1e-8);
  const poolSet=new Set();const poolIds=[];const limit=Math.min(initPool,initCandidates.length);
  for(let i=0;i<limit;i++){poolIds.push(initCandidates[i].did);poolSet.add(initCandidates[i].did);}
  const maxN=poolBudget;const distCache=new Float64Array(maxN*maxN);
  const clouds=poolIds.map(id=>getCloud(id));let N=poolIds.length;
  for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){const d=pqChamfer(clouds[i],clouds[j]);distCache[i*maxN+j]=d;distCache[j*maxN+i]=d;}

  function buildAdj(n,k){const ek=Math.min(k,n-1);const adj=Array.from({length:n},()=>[]);
    for(let i=0;i<n;i++){const nb=[];for(let j=0;j<n;j++)if(j!==i)nb.push({j,d:distCache[i*maxN+j]});nb.sort((a,b)=>a.d-b.d);for(let t=0;t<ek;t++)adj[i].push({j:nb[t].j,w:Math.exp(-2*nb[t].d)});}
    for(let i=0;i<n;i++)for(const e of adj[i])if(!adj[e.j].some(x=>x.j===i))adj[e.j].push({j:i,w:e.w});return adj;}
  function buildU(adj,n,centroids){const U=new Float64Array(n*n);for(let i=0;i<n;i++)for(const e of adj[i]){const j=e.j;if(U[i*n+j]||U[j*n+i])continue;let en=0,dvv=0;for(let d=0;d<DIM;d++){const df=centroids[j][d]-centroids[i][d];en+=df*df;dvv+=df*qV[d]*iqn;}const u0=(dvv/(Math.sqrt(en)+1e-8))*uStr;U[i*n+j]=u0;U[j*n+i]=-u0;}return U;}

  let centroids=poolIds.map(id=>getCentroid(id));
  let adj=buildAdj(N,knn);let U=buildU(adj,N,centroids);
  const C0=new Float64Array(N);for(let i=0;i<N;i++)C0[i]=Math.exp(-2*qdChamfer(qV,poolIds[i]));
  let C=pde4(C0,adj,U,N);

  // Stefan 预取轮次
  for(let r=1;r<=maxRounds;r++){
    if(N>=poolBudget||!vexus)break;
    const cVals=Array.from(C).sort((a,b)=>a-b);const median=cVals[cVals.length>>1];
    const fluxes=[];
    for(let i=0;i<N;i++){if(adj[i].length>=knn*2||C[i]<=median)continue;
      const ci=centroids[i];let dot=0;for(let d=0;d<DIM;d++)dot+=(qV[d]*iqn-ci[d])*(qV[d]*iqn);
      const flux=C[i]*Math.max(0,dot);if(flux>0)fluxes.push({idx:i,flux});}
    if(fluxes.length===0)break;fluxes.sort((a,b)=>b.flux-a.flux);
    const m=Math.min(probeCount,fluxes.length);const probes=fluxes.slice(0,m);const budget=poolBudget-N;

    const excludeFileIds=[];for(const did of poolSet){const fid=idMap[did];if(fid!==undefined)excludeFileIds.push(fid);}
    const newDids=[];const perProbe=Math.max(1,Math.ceil(budget/m));
    for(const p of probes){if(newDids.length>=budget)break;
      const probeCloud=getCloud(poolIds[p.idx]).map(v=>Buffer.from(v.buffer,v.byteOffset,v.byteLength));
      try{const hits=vexus.probePqChamfer(probeCloud,perProbe+excludeFileIds.length,excludeFileIds);
        let taken=0;for(const h of hits){if(taken>=perProbe||newDids.length>=budget)break;const did=reverseMap[h.id];if(!did||poolSet.has(did))continue;newDids.push(did);poolSet.add(did);taken++;}
      }catch(e){break;}}
    if(newDids.length===0)break;
    const oldN=N;for(const did of newDids){poolIds.push(did);clouds.push(getCloud(did));}N=poolIds.length;
    for(let i=oldN;i<N;i++)for(let j=0;j<N;j++){if(i===j)continue;const d=pqChamfer(clouds[i],clouds[j]);distCache[i*maxN+j]=d;distCache[j*maxN+i]=d;}
    centroids=poolIds.map(id=>getCentroid(id));adj=buildAdj(N,knn);U=buildU(adj,N,centroids);
    const C0n=new Float64Array(N);for(let i=0;i<oldN;i++)C0n[i]=C[i];for(let i=oldN;i<N;i++)C0n[i]=Math.exp(-2*qdChamfer(qV,poolIds[i]));
    C=pde4(C0n,adj,U,N);
  }
  return poolIds.map((id,i)=>({did:id,s:C[i]})).sort((a,b)=>b.s-a.s);
}

// stefan_full: Round 0 用 Rust fullscanPqChamfer 全库扫描替代 cosine 初始池
function stefanFull(qV,initCandidates){
  const initPool=topN,maxRounds=3,poolBudget=60,knn=3,probeCount=8,uStr=0.3,D=0.15;
  let qN=0;for(let d=0;d<DIM;d++)qN+=qV[d]*qV[d];const iqn=1/(Math.sqrt(qN)+1e-8);
  // Round 0: 用 Rust PQ-Chamfer 全库扫描获取初始 top-k
  const poolSet=new Set();const poolIds=[];
  if(vexus){
    const qBuf=[Buffer.from(qV.buffer,qV.byteOffset,qV.byteLength)];
    const rustHits=vexus.fullscanPqChamfer(qBuf,initPool);
    for(const h of rustHits){const did=reverseMap[h.id];if(did&&dv[did]){poolIds.push(did);poolSet.add(did);}}
  }else{
    // fallback: cosine
    const limit=Math.min(initPool,initCandidates.length);
    for(let i=0;i<limit;i++){poolIds.push(initCandidates[i].did);poolSet.add(initCandidates[i].did);}
  }
  const maxN=poolBudget;const distCache=new Float64Array(maxN*maxN);
  const clouds=poolIds.map(id=>getCloud(id));let N=poolIds.length;
  for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){const d=pqChamfer(clouds[i],clouds[j]);distCache[i*maxN+j]=d;distCache[j*maxN+i]=d;}

  function buildAdj(n,k){const ek=Math.min(k,n-1);const adj=Array.from({length:n},()=>[]);
    for(let i=0;i<n;i++){const nb=[];for(let j=0;j<n;j++)if(j!==i)nb.push({j,d:distCache[i*maxN+j]});nb.sort((a,b)=>a.d-b.d);for(let t=0;t<ek;t++)adj[i].push({j:nb[t].j,w:Math.exp(-2*nb[t].d)});}
    for(let i=0;i<n;i++)for(const e of adj[i])if(!adj[e.j].some(x=>x.j===i))adj[e.j].push({j:i,w:e.w});return adj;}
  function buildU(adj,n,centroids){const U=new Float64Array(n*n);for(let i=0;i<n;i++)for(const e of adj[i]){const j=e.j;if(U[i*n+j]||U[j*n+i])continue;let en=0,dvv=0;for(let d=0;d<DIM;d++){const df=centroids[j][d]-centroids[i][d];en+=df*df;dvv+=df*qV[d]*iqn;}const u0=(dvv/(Math.sqrt(en)+1e-8))*uStr;U[i*n+j]=u0;U[j*n+i]=-u0;}return U;}

  let centroids=poolIds.map(id=>getCentroid(id));
  let adj=buildAdj(N,knn);let U=buildU(adj,N,centroids);
  const C0=new Float64Array(N);for(let i=0;i<N;i++)C0[i]=Math.exp(-2*qdChamfer(qV,poolIds[i]));
  let C=pde4(C0,adj,U,N);

  // Stefan 预取轮次
  for(let r=1;r<=maxRounds;r++){
    if(N>=poolBudget||!vexus)break;
    const cVals=Array.from(C).sort((a,b)=>a-b);const median=cVals[cVals.length>>1];
    const fluxes=[];
    for(let i=0;i<N;i++){if(adj[i].length>=knn*2||C[i]<=median)continue;
      const ci=centroids[i];let dot=0;for(let d=0;d<DIM;d++)dot+=(qV[d]*iqn-ci[d])*(qV[d]*iqn);
      const flux=C[i]*Math.max(0,dot);if(flux>0)fluxes.push({idx:i,flux});}
    if(fluxes.length===0)break;fluxes.sort((a,b)=>b.flux-a.flux);
    const m=Math.min(probeCount,fluxes.length);const probes=fluxes.slice(0,m);const budget=poolBudget-N;

    const excludeFileIds=[];for(const did of poolSet){const fid=idMap[did];if(fid!==undefined)excludeFileIds.push(fid);}
    const newDids=[];const perProbe=Math.max(1,Math.ceil(budget/m));
    for(const p of probes){if(newDids.length>=budget)break;
      const probeCloud=getCloud(poolIds[p.idx]).map(v=>Buffer.from(v.buffer,v.byteOffset,v.byteLength));
      try{const hits=vexus.probePqChamfer(probeCloud,perProbe+excludeFileIds.length,excludeFileIds);
        let taken=0;for(const h of hits){if(taken>=perProbe||newDids.length>=budget)break;const did=reverseMap[h.id];if(!did||poolSet.has(did))continue;newDids.push(did);poolSet.add(did);taken++;}
      }catch(e){break;}}
    if(newDids.length===0)break;
    const oldN=N;for(const did of newDids){poolIds.push(did);clouds.push(getCloud(did));}N=poolIds.length;
    for(let i=oldN;i<N;i++)for(let j=0;j<N;j++){if(i===j)continue;const d=pqChamfer(clouds[i],clouds[j]);distCache[i*maxN+j]=d;distCache[j*maxN+i]=d;}
    centroids=poolIds.map(id=>getCentroid(id));adj=buildAdj(N,knn);U=buildU(adj,N,centroids);
    const C0n=new Float64Array(N);for(let i=0;i<oldN;i++)C0n[i]=C[i];for(let i=oldN;i<N;i++)C0n[i]=Math.exp(-2*qdChamfer(qV,poolIds[i]));
    C=pde4(C0n,adj,U,N);
  }
  return poolIds.map((id,i)=>({did:id,s:C[i]})).sort((a,b)=>b.s-a.s);
}

// v7_adjoint: 原版 V7.1 伴随状态预取算法（单轮，cosine 探针，不依赖 Rust）
function v7Adjoint(qV, initCandidates) {
  const initPool = 30, maxRounds = 1, poolBudget = 45, knn = 3, probeCount = 8, uStr = 0.3, D = 0.15;
  let qN = 0; for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);
  // query 单位向量（用于探针插值）
  const qHat = new Float32Array(DIM);
  for (let d = 0; d < DIM; d++) qHat[d] = qV[d] * iqn;

  // Round 0: cosine top-30 作为初始池
  const poolSet = new Set(); const poolIds = [];
  const limit = Math.min(initPool, initCandidates.length);
  for (let i = 0; i < limit; i++) { poolIds.push(initCandidates[i].did); poolSet.add(initCandidates[i].did); }
  const maxN = poolBudget; const distCache = new Float64Array(maxN * maxN);
  const clouds = poolIds.map(id => getCloud(id)); let N = poolIds.length;
  // 计算初始 PQ-Chamfer 距离矩阵
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const d = pqChamfer(clouds[i], clouds[j]); distCache[i * maxN + j] = d; distCache[j * maxN + i] = d;
  }

  // 构建 KNN 图
  function buildAdj(n, k) {
    const ek = Math.min(k, n - 1); const adj = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) {
      const nb = []; for (let j = 0; j < n; j++) if (j !== i) nb.push({ j, d: distCache[i * maxN + j] });
      nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) });
    }
    for (let i = 0; i < n; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });
    return adj;
  }
  // 构建对流系数矩阵
  function buildU(adj, n, centroids) {
    const U = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (const e of adj[i]) {
      const j = e.j; if (U[i * n + j] || U[j * n + i]) continue;
      let en = 0, dvv = 0;
      for (let d = 0; d < DIM; d++) { const df = centroids[j][d] - centroids[i][d]; en += df * df; dvv += df * qV[d] * iqn; }
      const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * uStr; U[i * n + j] = u0; U[j * n + i] = -u0;
    }
    return U;
  }

  let centroids = poolIds.map(id => getCentroid(id));
  let adj = buildAdj(N, knn); let U = buildU(adj, N, centroids);
  // 初始浓度 C_0 = exp(-2 * qdChamfer)
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * qdChamfer(qV, poolIds[i]));
  let C = pde4(C0, adj, U, N);

  // Round 1: 伴随状态预取（单轮）
  for (let r = 1; r <= maxRounds; r++) {
    if (N >= poolBudget) break;
    // 计算浓度中位数
    const cVals = Array.from(C).sort((a, b) => a - b); const median = cVals[cVals.length >> 1];

    // === 原版 V7.1 伴随通量 ===
    // 对节点 i 计算向外对流通量
    function outwardFlux(i) {
      let flux = 0;
      for (const e of adj[i]) {
        const j = e.j;
        // U[i*N+j] > 0 表示对流从 i 流向 j（向外）
        // 通量 = 对流系数 * 边权 * 邻居浓度
        const u_ij = U[i * N + j];
        if (u_ij > 0) flux += u_ij * e.w * C[j];
      }
      return flux;
    }

    // 伴随选择准则：边界节点 + 浓度 > 中位数 + 正向外通量
    const fluxes = [];
    for (let i = 0; i < N; i++) {
      if (adj[i].length >= knn * 2 || C[i] <= median) continue;
      const f = C[i] * Math.max(0, outwardFlux(i));
      if (f > 0) fluxes.push({ idx: i, flux: f });
    }
    if (fluxes.length === 0) break;
    fluxes.sort((a, b) => b.flux - a.flux);

    // 取 top-probeCount 个最高伴随通量的边界节点
    const m = Math.min(probeCount, fluxes.length);
    const probes = fluxes.slice(0, m);
    const budget = poolBudget - N;

    // === 原版 V7 探针：单向量插值 + cosine 全库检索（不用 PQ-Chamfer / Rust） ===
    const newDids = [];
    const perProbe = Math.max(1, Math.ceil(budget / m));
    for (const p of probes) {
      if (newDids.length >= budget) break;
      // probe_i = lambda * query_centroid + (1-lambda) * doc_centroid_i
      const lambda = 0.7;
      const docCentroid = centroids[p.idx];
      const probe = new Float32Array(DIM);
      for (let d = 0; d < DIM; d++) probe[d] = lambda * qHat[d] + (1 - lambda) * docCentroid[d];

      // cosine 全库扫描，排除已在池中的
      const hits = [];
      for (const did of allDids) {
        if (poolSet.has(did)) continue;
        const s = cosSim(probe, dv[did]);
        hits.push({ did, s });
      }
      hits.sort((a, b) => b.s - a.s);
      let taken = 0;
      for (const h of hits) {
        if (taken >= perProbe || newDids.length >= budget) break;
        newDids.push(h.did); poolSet.add(h.did); taken++;
      }
    }
    if (newDids.length === 0) break;

    // 新文档加入池（只增不踢）
    const oldN = N;
    for (const did of newDids) { poolIds.push(did); clouds.push(getCloud(did)); }
    N = poolIds.length;

    // 增量更新 PQ-Chamfer 距离矩阵
    for (let i = oldN; i < N; i++) for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const d = pqChamfer(clouds[i], clouds[j]); distCache[i * maxN + j] = d; distCache[j * maxN + i] = d;
    }

    // 重建 KNN 图 + 对流系数
    centroids = poolIds.map(id => getCentroid(id));
    adj = buildAdj(N, knn); U = buildU(adj, N, centroids);

    // 旧节点保留浓度，新节点用 exp(-2*qdChamfer) 初始化
    const C0n = new Float64Array(N);
    for (let i = 0; i < oldN; i++) C0n[i] = C[i];
    for (let i = oldN; i < N; i++) C0n[i] = Math.exp(-2 * qdChamfer(qV, poolIds[i]));
    C = pde4(C0n, adj, U, N);
  }

  // 最终按浓度排序
  return poolIds.map((id, i) => ({ did: id, s: C[i] })).sort((a, b) => b.s - a.s);
}

// vtV7Combined: vt_aligned 距离函数 + v7_adjoint 伴随通量预取策略的叠加
function vtV7Combined(qV, initCandidates) {
  const initPool = 30, maxRounds = 1, poolBudget = 45, knn = 3, probeCount = 8, uStr = 0.3, D = 0.15;
  let qN = 0; for (let d = 0; d < DIM; d++) qN += qV[d] * qV[d];
  const iqn = 1 / (Math.sqrt(qN) + 1e-8);
  // query 单位向量（用于 v7 cosine 插值探针）
  const qHat = new Float32Array(DIM);
  for (let d = 0; d < DIM; d++) qHat[d] = qV[d] * iqn;

  // ── Round 0: 和 shapeVtAligned 完全相同 ──
  // cosine top-30 初始池
  const poolSet = new Set(); const poolIds = [];
  const limit = Math.min(initPool, initCandidates.length);
  for (let i = 0; i < limit; i++) { poolIds.push(initCandidates[i].did); poolSet.add(initCandidates[i].did); }
  const maxN = poolBudget; const distCache = new Float64Array(maxN * maxN);
  const clouds = poolIds.map(id => getCloud(id)); let N = poolIds.length;
  // 用 vtAlignedChamfer 建 doc-doc 距离矩阵
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const d = vtAlignedChamfer(clouds[i], clouds[j]); distCache[i * maxN + j] = d; distCache[j * maxN + i] = d;
  }

  // 构建 KNN 图
  function buildAdj(n, k) {
    const ek = Math.min(k, n - 1); const adj = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++) {
      const nb = []; for (let j = 0; j < n; j++) if (j !== i) nb.push({ j, d: distCache[i * maxN + j] });
      nb.sort((a, b) => a.d - b.d); for (let t = 0; t < ek; t++) adj[i].push({ j: nb[t].j, w: Math.exp(-2 * nb[t].d) });
    }
    for (let i = 0; i < n; i++) for (const e of adj[i]) if (!adj[e.j].some(x => x.j === i)) adj[e.j].push({ j: i, w: e.w });
    return adj;
  }
  // 构建对流系数矩阵
  function buildU(adj, n, centroids) {
    const U = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (const e of adj[i]) {
      const j = e.j; if (U[i * n + j] || U[j * n + i]) continue;
      let en = 0, dvv = 0;
      for (let d = 0; d < DIM; d++) { const df = centroids[j][d] - centroids[i][d]; en += df * df; dvv += df * qV[d] * iqn; }
      const u0 = (dvv / (Math.sqrt(en) + 1e-8)) * uStr; U[i * n + j] = u0; U[j * n + i] = -u0;
    }
    return U;
  }

  let centroids = poolIds.map(id => getCentroid(id));
  let adj = buildAdj(N, knn); let U = buildU(adj, N, centroids);
  // 用 vtAlignedDist 计算 query-doc 初始浓度
  const C0 = new Float64Array(N); for (let i = 0; i < N; i++) C0[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
  let C = pde4(C0, adj, U, N);

  // ── Round 1: v7_adjoint 伴随通量预取 ──
  for (let r = 1; r <= maxRounds; r++) {
    if (N >= poolBudget) break;
    const cVals = Array.from(C).sort((a, b) => a - b); const median = cVals[cVals.length >> 1];

    // v7 伴随通量公式：outwardFlux(i) = sum_{j where U[i*N+j]>0} U[i*N+j] * w_ij * C[j]
    function outwardFlux(i) {
      let flux = 0;
      for (const e of adj[i]) {
        const j = e.j;
        const u_ij = U[i * N + j];
        if (u_ij > 0) flux += u_ij * e.w * C[j];
      }
      return flux;
    }

    // 伴随选择准则：边界节点 + 浓度 > 中位数 + 正向外通量
    const fluxes = [];
    for (let i = 0; i < N; i++) {
      if (adj[i].length >= knn * 2 || C[i] <= median) continue;
      const f = C[i] * Math.max(0, outwardFlux(i));
      if (f > 0) fluxes.push({ idx: i, flux: f });
    }
    if (fluxes.length === 0) break;
    fluxes.sort((a, b) => b.flux - a.flux);

    // 取 top-probeCount 个最高伴随通量的边界节点
    const m = Math.min(probeCount, fluxes.length);
    const probes = fluxes.slice(0, m);
    const budget = poolBudget - N;

    // v7 cosine 插值探针：probe = 0.7*qHat + 0.3*docCentroid，cosine 全库检索（不用 Rust）
    const newDids = [];
    const perProbe = Math.max(1, Math.ceil(budget / m));
    for (const p of probes) {
      if (newDids.length >= budget) break;
      const lambda = 0.7;
      const docCentroid = centroids[p.idx];
      const probe = new Float32Array(DIM);
      for (let d = 0; d < DIM; d++) probe[d] = lambda * qHat[d] + (1 - lambda) * docCentroid[d];

      // cosine 全库扫描，排除已在池中的
      const hits = [];
      for (const did of allDids) {
        if (poolSet.has(did)) continue;
        const s = cosSim(probe, dv[did]);
        hits.push({ did, s });
      }
      hits.sort((a, b) => b.s - a.s);
      let taken = 0;
      for (const h of hits) {
        if (taken >= perProbe || newDids.length >= budget) break;
        newDids.push(h.did); poolSet.add(h.did); taken++;
      }
    }
    if (newDids.length === 0) break;

    // 新文档加入池（只增不踢）
    const oldN = N;
    for (const did of newDids) { poolIds.push(did); clouds.push(getCloud(did)); }
    N = poolIds.length;

    // 用 vtAlignedChamfer 增量更新距离矩阵
    for (let i = oldN; i < N; i++) for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const d = vtAlignedChamfer(clouds[i], clouds[j]); distCache[i * maxN + j] = d; distCache[j * maxN + i] = d;
    }

    // 重建 KNN 图 + 对流系数
    centroids = poolIds.map(id => getCentroid(id));
    adj = buildAdj(N, knn); U = buildU(adj, N, centroids);

    // 旧节点保留浓度，新节点用 vtAlignedDist 初始化
    const C0n = new Float64Array(N);
    for (let i = 0; i < oldN; i++) C0n[i] = C[i];
    for (let i = oldN; i < N; i++) C0n[i] = Math.exp(-2 * vtAlignedDist(qV, clouds[i]));
    C = pde4(C0n, adj, U, N);
  }

  // 最终按浓度排序
  return poolIds.map((id, i) => ({ did: id, s: C[i] })).sort((a, b) => b.s - a.s);
}

// Worker 主循环
const R={cosine:0,shape_pq64:0,vt_aligned:0,vt_unaligned:0,stefan_rust:0,stefan_full:0,v7_adjoint:0,vt_v7:0};
for(const q of queries){
  const qV=new Float32Array(q.vec),qr=q.qrels;
  const cs=allDids.map(did=>({did,s:cosSim(qV,dv[did])}));cs.sort((a,b)=>b.s-a.s);
  R.cosine+=ndcg(cs.slice(0,10).map(d=>d.did),qr);
  const pool=cs.slice(0,topN).map(x=>x.did);
  R.shape_pq64+=ndcg(shapePq64(qV,pool),qr);
  R.vt_aligned+=ndcg(shapeVtAligned(qV,pool),qr);
  R.vt_unaligned+=ndcg(shapeVtUnaligned(qV,pool),qr);
  const sr=stefanRust(qV,cs.slice(0,topN));
  R.stefan_rust+=ndcg(sr.slice(0,10).map(x=>x.did),qr);
  // stefan_full: Round 0 也用 Rust PQ-Chamfer 全库扫描
  const sf=stefanFull(qV,cs.slice(0,topN));
  R.stefan_full+=ndcg(sf.slice(0,10).map(x=>x.did),qr);
  // v7_adjoint: 原版 V7.1 伴随状态预取（cosine 探针，单轮）
  const v7=v7Adjoint(qV,cs.slice(0,topN));
  R.v7_adjoint+=ndcg(v7.slice(0,10).map(x=>x.did),qr);
  // vt_v7: vt_aligned 距离 + v7_adjoint 预取
  const vtv7=vtV7Combined(qV,cs.slice(0,topN));
  R.vt_v7+=ndcg(vtv7.slice(0,10).map(x=>x.did),qr);
  parentPort.postMessage({type:'progress'});
}
parentPort.postMessage({type:'result',data:R});process.exit(0);
}

// ════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════
const DATA_DIR=path.join(__dirname,'beir_data','nfcorpus');
const SQLITE_PATH=path.join(DATA_DIR,'clouds.sqlite');
const ID_MAP_PATH=path.join(DATA_DIR,'id_map.json');

function loadJsonl(fp){return new Promise((r,j)=>{const a=[];const rl=readline.createInterface({input:fs.createReadStream(fp,{encoding:'utf-8'}),crlfDelay:Infinity});rl.on('line',l=>{if(l.trim())try{a.push(JSON.parse(l));}catch(e){}});rl.on('close',()=>r(a));rl.on('error',j);});}
function loadQrels(fp){const q={};const ls=fs.readFileSync(fp,'utf-8').trim().split('\n');for(let i=1;i<ls.length;i++){const[qi,di,s]=ls[i].split('\t');if(!q[qi])q[qi]={};q[qi][di]=parseInt(s);}return q;}

(async()=>{
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`  BEIR NFCorpus — 64-Worker 并行 Benchmark`);
  console.log(`  cosine | shape_pq64 | vt_aligned | vt_unaligned | stefan_rust | stefan_full | v7_adjoint | vt_v7`);
  console.log(`${'═'.repeat(60)}\n`);

  const MAX_Q=parseInt(process.env.MAX_Q||'0');

  // 加载数据
  const idMap=JSON.parse(fs.readFileSync(ID_MAP_PATH,'utf-8'));
  const reverseMap={};for(const[k,v]of Object.entries(idMap))reverseMap[v]=k;
  const cV={},cS={};
  for(const o of await loadJsonl(path.join(DATA_DIR,'corpus_vectors.jsonl'))){cV[o._id]=Array.from(new Float32Array(o.vector));if(o.sentences?.length>1)cS[o._id]=o.sentences;}
  const qV={};for(const o of await loadJsonl(path.join(DATA_DIR,'query_vectors.jsonl')))qV[o._id]=Array.from(new Float32Array(o.vector));
  const qrels=loadQrels(path.join(DATA_DIR,'qrels.tsv'));
  let qids=Object.keys(qrels).filter(q=>qV[q]);
  if(MAX_Q>0)qids=qids.slice(0,MAX_Q);
  const allDids=Object.keys(cV);
  const topN=30;

  console.log(`  Q:${qids.length} C:${allDids.length} W:${NW}\n`);

  // 分配 query 到 worker
  const batches=Array.from({length:NW},()=>[]);
  for(let i=0;i<qids.length;i++){const qi=qids[i];batches[i%NW].push({vec:qV[qi],qrels:qrels[qi]});}

  let completed=0;const t0=performance.now();
  const progressTimer=setInterval(()=>{
    const sec=(performance.now()-t0)/1000;const qps=completed/sec||0;const eta=qps>0?((qids.length-completed)/qps):0;
    process.stdout.write(`\r  ${completed}/${qids.length} | ${qps.toFixed(1)} q/s | ${sec.toFixed(0)}s | ETA ${eta.toFixed(0)}s  `);
  },3000);

  const wr=await Promise.all(batches.map((b,i)=>{
    if(!b.length)return Promise.resolve({cosine:0,shape_pq64:0,vt_aligned:0,vt_unaligned:0,stefan_rust:0,stefan_full:0,v7_adjoint:0,vt_v7:0});
    return new Promise((r,j)=>{
      const w=new Worker(__filename,{workerData:{queries:b,corpusVecs:cV,corpusSents:cS,allDids,topN,idMap,reverseMap,sqlitePath:SQLITE_PATH}});
      w.on('message',msg=>{if(msg.type==='progress')completed++;else if(msg.type==='result')r(msg.data);});
      w.on('error',j);w.on('exit',c=>{if(c!==0)j(new Error(`W${i}:${c}`));});
    });
  }));

  clearInterval(progressTimer);
  const sec=(performance.now()-t0)/1000;
  process.stdout.write('\r'+' '.repeat(80)+'\r');

  const methods=['cosine','shape_pq64','vt_aligned','vt_unaligned','stefan_rust','stefan_full','v7_adjoint','vt_v7'];
  const nd={};for(const m of methods)nd[m]=0;for(const w of wr)for(const m of methods)nd[m]+=(w[m]||0);
  const nQ=qids.length;

  console.log(`\n  ✅ ${sec.toFixed(1)}s (${(nQ/sec).toFixed(1)} q/s)\n`);
  console.log('  ┌───────────────────┬──────────┬─────────────┬─────────────┐');
  console.log('  │ Method            │ NDCG@10  │ vs cosine   │ vs pq64     │');
  console.log('  ├───────────────────┼──────────┼─────────────┼─────────────┤');
  const cos=nd.cosine/nQ,pq=nd.shape_pq64/nQ,vta=nd.vt_aligned/nQ,vtu=nd.vt_unaligned/nQ,st=nd.stefan_rust/nQ,sf=nd.stefan_full/nQ,v7a=nd.v7_adjoint/nQ,vtv7=nd.vt_v7/nQ;
  const fmt=(v,b)=>{const p=((v-b)/b*100);return `${p>=0?'+':''}${p.toFixed(1)}%`;};
  console.log(`  │ cosine            │ ${cos.toFixed(4).padStart(8)} │    baseline │ ${fmt(cos,pq).padStart(11)} │`);
  console.log(`  │ shape_pq64        │ ${pq.toFixed(4).padStart(8)} │ ${fmt(pq,cos).padStart(11)} │    baseline │`);
  console.log(`  │ vt_aligned        │ ${vta.toFixed(4).padStart(8)} │ ${fmt(vta,cos).padStart(11)} │ ${fmt(vta,pq).padStart(11)} │`);
  console.log(`  │ vt_unaligned      │ ${vtu.toFixed(4).padStart(8)} │ ${fmt(vtu,cos).padStart(11)} │ ${fmt(vtu,pq).padStart(11)} │`);
  console.log(`  │ stefan_rust       │ ${st.toFixed(4).padStart(8)} │ ${fmt(st,cos).padStart(11)} │ ${fmt(st,pq).padStart(11)} │`);
  console.log(`  │ stefan_full       │ ${sf.toFixed(4).padStart(8)} │ ${fmt(sf,cos).padStart(11)} │ ${fmt(sf,pq).padStart(11)} │`);
  console.log(`  │ v7_adjoint        │ ${v7a.toFixed(4).padStart(8)} │ ${fmt(v7a,cos).padStart(11)} │ ${fmt(v7a,pq).padStart(11)} │`);
  console.log(`  │ vt_v7             │ ${vtv7.toFixed(4).padStart(8)} │ ${fmt(vtv7,cos).padStart(11)} │ ${fmt(vtv7,pq).padStart(11)} │`);
  console.log('  └───────────────────┴──────────┴─────────────┴─────────────┘');

  fs.writeFileSync(path.join(DATA_DIR,'rust_parallel_results.json'),JSON.stringify({cosine:+cos.toFixed(4),shape_pq64:+pq.toFixed(4),vt_aligned:+vta.toFixed(4),vt_unaligned:+vtu.toFixed(4),stefan_rust:+st.toFixed(4),stefan_full:+sf.toFixed(4),v7_adjoint:+v7a.toFixed(4),vt_v7:+vtv7.toFixed(4),queries:nQ,workers:NW,seconds:+sec.toFixed(1)},null,2));
  console.log(`\n  💾 rust_parallel_results.json`);
})().catch(e=>{console.error('❌',e);process.exit(1);});
