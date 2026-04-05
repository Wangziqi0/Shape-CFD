#!/usr/bin/env python3
"""
BEIR Benchmark — 4 种重排序方法对比（内存安全版）

方法:
  A. Cosine 直排 (baseline)
  B. AD-Rank v2 (对流-扩散)
  C. Shape-CFD (Chamfer + 多点云 + 对流-扩散)
  D. BM25 (词法基线)

用法: python beir_benchmark_safe.py --dataset scifact
"""

import json, argparse, math, re, sys, time
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── 向量工具 ──
def cosine_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb + 1e-8))

def cosine_dist(a, b):
    return 1.0 - cosine_sim(a, b)

# ── 方法 A: Cosine 直排 ──
def cosine_rerank(query_vec, doc_vecs, doc_ids):
    scores = [(did, cosine_sim(query_vec, dv)) for did, dv in zip(doc_ids, doc_vecs)]
    scores.sort(key=lambda x: -x[1])
    return scores

# ── 方法 B: AD-Rank v2 ──
def ad_rank_v2(query_vec, doc_vecs, doc_ids, D=0.15, u_strength=0.1, max_iter=50, eps=1e-3, knn=3):
    N = len(doc_vecs)
    if N == 0: return []
    
    # 相似度矩阵
    sim_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            s = cosine_sim(doc_vecs[i], doc_vecs[j])
            sim_mat[i,j] = sim_mat[j,i] = s
    
    # KNN 图
    beta = 2.0
    ek = min(knn, N-1)
    adj = [[] for _ in range(N)]
    for i in range(N):
        nbs = sorted(range(N), key=lambda j: -sim_mat[i,j] if j != i else -99)
        for t in range(ek):
            j = nbs[t]
            adj[i].append((j, math.exp(-beta * (1 - sim_mat[i,j]))))
    
    # 对称化
    for i in range(N):
        for j, w in adj[i]:
            if not any(jj == i for jj, _ in adj[j]):
                adj[j].append((i, w))
    
    # 初始浓度
    C = np.array([cosine_sim(query_vec, dv) for dv in doc_vecs])
    
    # 对流系数
    dim = len(query_vec)
    q_norm = np.linalg.norm(query_vec) + 1e-8
    q_unit = query_vec / q_norm
    
    U = np.zeros((N, N))
    for i in range(N):
        for j, _ in adj[i]:
            if U[i,j] != 0: continue
            diff = doc_vecs[j] - doc_vecs[i]
            ed_norm = np.linalg.norm(diff) + 1e-8
            dot_val = np.dot(diff, q_unit)
            u_ij = (dot_val / ed_norm) * u_strength
            U[i,j] = u_ij
            U[j,i] = -u_ij
    
    # CFL
    max_deg = max(len(a) for a in adj) if adj else 1
    dt = min(0.1, 0.8 / max_deg if max_deg > 0 else 0.1)
    
    # 迭代
    for _ in range(max_iter):
        C_new = C.copy()
        max_delta = 0
        for i in range(N):
            diffusion = advection = 0
            for j, w in adj[i]:
                diffusion += D * w * (C[j] - C[i])
                u_ij, u_ji = U[i,j], U[j,i]
                advection += w * (max(u_ji, 0) * C[j] - max(u_ij, 0) * C[i])
            c_new = max(0, C[i] + dt * (diffusion + advection))
            C_new[i] = c_new
            max_delta = max(max_delta, abs(c_new - C[i]))
        C = C_new
        if max_delta < eps: break
    
    return sorted(zip(doc_ids, C.tolist()), key=lambda x: -x[1])

# ── 方法 C: Shape-CFD (调优版) ──
def chamfer_distance(cloud_a, cloud_b):
    """Chamfer distance between two point clouds (对称)"""
    sum_ab = 0
    for a in cloud_a:
        min_d = min(cosine_dist(a, b) for b in cloud_b)
        sum_ab += min_d
    sum_ba = 0
    for b in cloud_b:
        min_d = min(cosine_dist(a, b) for a in cloud_a)
        sum_ba += min_d
    return sum_ab / len(cloud_a) + sum_ba / len(cloud_b)

def shape_cfd(query_vec, doc_vecs, doc_clouds, doc_ids, alpha=0.4):
    """
    Shape-CFD v4 — Shape-Boost:
    不做 CFD 迭代（扩散在 reranking 中天然有害）
    
    核心思想: 文档中最相关的句子比整体文档 cosine 更精确
    score = α * doc_cosine + (1-α) * max_sentence_cosine
    
    当 α=1 时退化为 Cosine 基线
    当 α=0 时为纯 max-sentence matching
    """
    N = len(doc_clouds)
    if N == 0: return []
    
    scores = []
    for i in range(N):
        # 文档级 cosine
        doc_cos = cosine_sim(query_vec, doc_vecs[i])
        
        # 句子级 max cosine（找到与 query 最相关的句子）
        cloud = doc_clouds[i]
        if len(cloud) > 1:
            sent_sims = [cosine_sim(query_vec, s) for s in cloud]
            max_sent = max(sent_sims)
            # 也考虑 top-k 句子的平均（更鲁棒）
            top_k = sorted(sent_sims, reverse=True)[:3]
            avg_top_k = sum(top_k) / len(top_k)
            
            # 组合: doc_cos + max_sent + avg_top3
            shape_score = 0.5 * max_sent + 0.5 * avg_top_k
        else:
            shape_score = doc_cos
        
        final = alpha * doc_cos + (1 - alpha) * shape_score
        scores.append((doc_ids[i], final))
    
    scores.sort(key=lambda x: -x[1])
    return scores

# ── 方法 D: Graph Reaction-Diffusion Ranker ──
def reaction_diffusion(query_vec, doc_vecs, doc_ids,
                       D=0.20, gamma=0.5, tau=0.2,
                       knn=5, max_iter=20, eps=1e-3):
    """
    Graph Reaction-Diffusion Ranker (Fokker-Planck + Allen-Cahn)
    
    核心设计（来自 Gemini 理论诊断）：
    1. 非对称转移矩阵：T_ij = A_ij * exp((Φ_j - Φ_i) / τ)
       - 浓度自然沿势能梯度流向更像 query 的节点
       - 完全避开 4096 维向量内积的方差坍缩
    2. Allen-Cahn 反应项：R(C) = γ * C * (1-C) * (C - θ)
       - 相分离/极化：高 C 节点自催化推向 1，低 C 被压制向 0
       - θ 自适应取中位数
    """
    N = len(doc_ids)
    if N == 0: return []
    
    # 1. 势能场：Φ_i = cos(q, v_i)
    Phi = np.array([cosine_sim(query_vec, dv) for dv in doc_vecs])
    
    # 2. 文档间相似度矩阵
    sim_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            s = cosine_sim(doc_vecs[i], doc_vecs[j])
            sim_mat[i,j] = sim_mat[j,i] = s
    
    # 3. KNN 图 + 对称化（得到基础邻接权重 A_ij）
    beta = 2.0
    ek = min(knn, N-1)
    adj = [[] for _ in range(N)]
    for i in range(N):
        nbs = sorted(range(N), key=lambda j: -sim_mat[i,j] if j != i else -99)
        for t in range(ek):
            j = nbs[t]
            adj[i].append((j, math.exp(-beta * (1 - sim_mat[i,j]))))
    for i in range(N):
        for j, w in adj[i]:
            if not any(jj == i for jj, _ in adj[j]):
                adj[j].append((i, w))
    
    # 4. 构造非对称转移权重 T_ij = A_ij * exp((Φ_j - Φ_i) / τ)
    # T_ij 表示从 i 到 j 的转移倾向
    # 当 Φ_j > Φ_i 时，exp > 1，浓度更容易从 i 流向 j（高势能方向）
    T = {}  # T[(i,j)] = 非对称权重（从 i 看 j 的影响）
    for i in range(N):
        for j, a_ij in adj[i]:
            t_ij = a_ij * math.exp((Phi[j] - Phi[i]) / tau)
            T[(i, j)] = t_ij
    
    # 5. 初始浓度 C_0 = cos(q, v_i)，归一化到 [0,1]
    C_min, C_max = Phi.min(), Phi.max()
    if C_max - C_min > 1e-8:
        C = (Phi - C_min) / (C_max - C_min)
    else:
        C = np.ones(N) * 0.5
    
    # 自适应阈值 θ：取中位数
    theta = float(np.median(C))
    
    # 6. CFL 稳定性
    max_out = 0
    for i in range(N):
        out_i = 0
        for j, _ in adj[i]:
            out_i += T.get((i, j), 0)
        max_out = max(max_out, out_i)
    dt = 0.9 / (D * max_out + gamma + 1e-8)
    
    # 7. 迭代求解
    for _ in range(max_iter):
        C_new = C.copy()
        max_delta = 0
        for i in range(N):
            # 有向扩散：用非对称权重
            flux = 0
            for j, _ in adj[i]:
                t_ji = T.get((j, i), 0)  # j→i 方向的权重
                t_ij = T.get((i, j), 0)  # i→j 方向的权重
                # 净流入 = 从 j 流入 i 的量 - 从 i 流出到 j 的量
                flux += D * (t_ji * C[j] - t_ij * C[i])
            
            # 反应项（Allen-Cahn 双稳态）
            reaction = gamma * C[i] * (1 - C[i]) * (C[i] - theta)
            
            c_new = C[i] + dt * (flux + reaction)
            c_new = max(0, min(1, c_new))
            C_new[i] = c_new
            max_delta = max(max_delta, abs(c_new - C[i]))
        C = C_new
        if max_delta < eps: break
    
    return sorted(zip(doc_ids, C.tolist()), key=lambda x: -x[1])

# ── 方法 E: BM25 ──
def bm25_rerank(query_text, doc_texts, doc_ids, k1=1.2, b=0.75):
    tokenize = lambda t: re.sub(r'[^a-z0-9\u4e00-\u9fff]', ' ', t.lower()).split()
    
    q_tokens = tokenize(query_text)
    N = len(doc_texts)
    doc_tokens = [tokenize(t) for t in doc_texts]
    avg_dl = sum(len(d) for d in doc_tokens) / max(N, 1)
    
    df = {}
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    
    scores = []
    for i, did in enumerate(doc_ids):
        dl = len(doc_tokens[i])
        tf = {}
        for t in doc_tokens[i]: tf[t] = tf.get(t, 0) + 1
        
        score = 0
        for qt in q_tokens:
            if qt not in tf: continue
            idf = math.log((N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1)
            tf_norm = (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm
        scores.append((did, score))
    
    scores.sort(key=lambda x: -x[1])
    return scores

# ── 句子向量按需加载 ──
def load_sentences_for_docs(big_file, sent_index, doc_ids):
    """按需从大文件加载指定文档的句子向量（只读需要的行）"""
    clouds = {}
    
    with open(big_file, 'r') as f:
        for did in doc_ids:
            info = sent_index.get(did)
            if not info:
                clouds[did] = None
                continue
            
            f.seek(info["offset"])
            line = f.read(info["line_len"])
            try:
                obj = json.loads(line)
                sents = obj.get("sentences", [])
                if sents:
                    clouds[did] = [np.array(s, dtype=np.float32) for s in sents]
                else:
                    clouds[did] = None
            except:
                clouds[did] = None
    
    return clouds

# ── 主函数 ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--data-dir", default="./beir_data")
    parser.add_argument("--topn", type=int, default=30)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) / args.dataset
    topN = args.topn
    topK = args.topk
    
    print(f"\n{'='*60}")
    print(f"  BEIR Benchmark — {args.dataset} (top-{topN} → rerank → top-{topK})")
    print(f"  Python 内存安全版 + 多点云 Shape-CFD")
    print(f"{'='*60}\n")
    
    # 1. 加载小文件
    print("📂 加载数据...")
    t0 = time.time()
    
    # 文档级向量（小文件 ~331MB）
    doc_vectors = {}
    with open(data_dir / "doc_vectors.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            doc_vectors[obj["_id"]] = np.array(obj["vector"], dtype=np.float32)
    
    # 句子索引（极小 ~0.25MB）
    with open(data_dir / "sentence_index.json") as f:
        sent_index = json.load(f)
    
    # Query 向量
    query_vectors = {}
    query_texts = {}
    with open(data_dir / "query_vectors.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            query_vectors[obj["_id"]] = np.array(obj["vector"], dtype=np.float32)
            query_texts[obj["_id"]] = obj.get("text", "")
    
    # Corpus 文本（用于 BM25）
    corpus_texts = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            corpus_texts[obj["_id"]] = obj.get("text", "")
    
    # Qrels
    qrels = {}
    with open(data_dir / "qrels.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(parts[2])
                if qid not in qrels: qrels[qid] = {}
                qrels[qid][did] = score
    
    load_time = time.time() - t0
    print(f"  Corpus vectors: {len(doc_vectors)}")
    print(f"  Query vectors: {len(query_vectors)}")
    print(f"  Corpus texts: {len(corpus_texts)}")
    print(f"  Qrels: {len(qrels)} queries")
    print(f"  加载时间: {load_time:.1f}s")
    
    # 内存估算
    mem_mb = len(doc_vectors) * 4096 * 4 / 1024 / 1024  # Float32
    print(f"  估算内存: ~{mem_mb:.0f} MB (文档向量)\n")
    
    # 2. Benchmark
    big_file = data_dir / "corpus_vectors.jsonl"
    query_ids = [qid for qid in qrels if qid in query_vectors]
    
    print(f"🔄 Running {len(query_ids)} queries × 5 methods...\n")
    
    results = []
    all_doc_ids = list(doc_vectors.keys())
    
    # 预计算所有文档的 cosine 相似度不做（太慢），改用 numpy 批量
    doc_matrix = np.array([doc_vectors[did] for did in all_doc_ids], dtype=np.float32)
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8
    doc_matrix_normed = doc_matrix / doc_norms
    
    for qi, qid in enumerate(tqdm(query_ids, desc="Benchmarking")):
        q_vec = query_vectors[qid]
        q_text = query_texts.get(qid, "")
        q_normed = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        
        # 一阶段检索：numpy 批量 cosine
        cos_scores = doc_matrix_normed @ q_normed
        top_indices = np.argsort(-cos_scores)[:topN]
        
        cand_ids = [all_doc_ids[i] for i in top_indices]
        cand_vecs = [doc_vectors[did] for did in cand_ids]
        cand_scores = [float(cos_scores[i]) for i in top_indices]
        
        # A. Cosine
        cosine_result = [(did, s) for did, s in zip(cand_ids, cand_scores)]
        results.append({"query_id": qid, "method": "cosine",
                        "rankings": [{"doc_id": d, "score": s} for d, s in cosine_result[:topK]]})
        
        # B. AD-Rank v2
        v2_result = ad_rank_v2(q_vec, cand_vecs, cand_ids)
        results.append({"query_id": qid, "method": "ad_rank_v2",
                        "rankings": [{"doc_id": d, "score": s} for d, s in v2_result[:topK]]})
        
        # C. Shape-CFD（按需加载句子向量）
        sent_clouds = load_sentences_for_docs(big_file, sent_index, cand_ids)
        cand_clouds = []
        for did in cand_ids:
            cloud = sent_clouds.get(did)
            if cloud and len(cloud) > 1:
                cand_clouds.append(cloud)
            else:
                cand_clouds.append([doc_vectors[did]])
        
        shape_result = shape_cfd(q_vec, cand_vecs, cand_clouds, cand_ids)
        results.append({"query_id": qid, "method": "shape_cfd",
                        "rankings": [{"doc_id": d, "score": s} for d, s in shape_result[:topK]]})

        
        # D. Reaction-Diffusion
        rd_result = reaction_diffusion(q_vec, cand_vecs, cand_ids)
        results.append({"query_id": qid, "method": "reaction_diffusion",
                        "rankings": [{"doc_id": d, "score": s} for d, s in rd_result[:topK]]})
        
        # E. BM25
        cand_texts = [corpus_texts.get(did, "") for did in cand_ids]
        bm25_result = bm25_rerank(q_text, cand_texts, cand_ids)
        results.append({"query_id": qid, "method": "bm25",
                        "rankings": [{"doc_id": d, "score": s} for d, s in bm25_result[:topK]]})
    
    # 3. 保存
    output_path = data_dir / "results.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n💾 结果保存到 {output_path}")
    print(f"   共 {len(results)} 条 ({len(query_ids)} queries × 5 methods)")
    print(f"\n✅ 完成！运行 beir_evaluate.py 计算指标。")

if __name__ == "__main__":
    main()
