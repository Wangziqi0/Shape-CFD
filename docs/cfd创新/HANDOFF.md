# 🔧 Shape-CFD 项目交接文档

> **作者**: 陈一凡 (Yifan Chen) + Antigravity Agent
> **最后更新**: 2026-03-31
> **Zenodo DOI**: [10.5281/zenodo.19234110](https://zenodo.org/records/19234110) (v1/v3)

---

## 1. 项目概述

Shape-CFD 是一个**物理启发的文档重排序框架**，将检索后重排序建模为文档相似度图上的**守恒型对流-扩散过程**。核心创新：
- 对称 Chamfer 距离建图（句子级点云匹配）
- 保守上风格式的对流-扩散 PDE 求解

**当前状态**: V11 Fusion 为全场最优，NFCorpus **0.3232 (+47.2% vs cosine)**。fusion_07 = 0.7 * token_2stage + 0.3 * shape_cfd_v10，PDE 以正交信号源方式回归，延迟仅 ~26ms。此前 token 直排 0.3214 (+46.4%)，PDE 管线 0.3180 (+44.9%)，两阶段粗筛 0.3220 (+46.7%)。sampled_graph(0.3218) 确认图-浓度不匹配，PDE 价值在于不同视角而非同一视角平滑。数据：token_clouds.sqlite 14GB, 880791 tokens。论文 Section 3.6 已用 UOT (非平衡最优传输) 框架重写 Chamfer 动机。Rust 引擎位于独立仓库 `/home/amd/HEZIMENG/law-vexus/`，全面采用 fullscan 全库暴力扫描，抛弃 HNSW，已完成 6 处修复校准。

---

## 2. 核心文件清单

### 生产代码（LawVein 集成）

| 文件 | 行数 | 说明 |
|:-----|:---:|:-----|
| `search_engine.js` | 91K | RAG 主管线，Shape-CFD V4 集成在此 |
| `ad_rank_shape.js` | 31K | **Shape-CFD V4 核心实现**（生产版本） |
| `ad_rank_shape_v5.js` | 25K | V5 实验版（灵感开关版），未进入生产 |
| `ad_rank.js` | 30K | 原始 AD-Rank v2（无 Chamfer/点云） |
| `vectorize_engine.js` | 15K | Qwen3-Embedding-8B 向量化引擎 |

### Rust 引擎

**新版 Rust 引擎**（位于 `/home/amd/HEZIMENG/law-vexus/`，独立于本项目）：

已完全抛弃 HNSW 和 cosine，改为 fullscan 全库暴力扫描。**已完成 6 处修复校准，NDCG 0.2852 超越 JS 0.2844。**

校准历程：0.2145(原始) → 0.2597(公式对齐) → 0.2750(质心粗筛) → 0.2852(top_n=55)

6 处修复：
1. cosine_distance_64d：去掉 .max(0.0) 截断（允许负距离）
2. build_knn 边权：1/(1+d) → exp(-2d)（对齐 JS beta=2.0）
3. build_knn 图对称化（保证 W_ij=W_ji 守恒性）
4. compute_advection：改为质心向量差投影到 query 方向
5. solve_pde 时间步：CFL min(0.1, 0.8/maxDeg)
6. cosine_top_n：句子级 MaxSim → 文档质心 cosine（粗筛 top55）

管线：
`cloud_store(SQLite点云+范数预缓存) → cosine_top_n(质心粗筛top55) → vt_distance(VT-Aligned+KNN图+rayon并行) → pde(Upwind PDE求解) → Top-K`

| 模块 | 行数 | 功能 |
|:-----|:----:|:-----|
| `cloud_store.rs` | 406 | SQLite 点云存储 + 范数预缓存 |
| `fullscan.rs` | 517 | 全库 VT 距离扫描（替代 HNSW） |
| `pq_chamfer.rs` | 382 | PQ-Chamfer 子空间距离 |
| `vt_distance.rs` | 605 | VT-Aligned 距离 + KNN 图 + rayon 并行 |
| `pde.rs` | 238 | Upwind PDE 求解器 |

- 45 个单元测试，38ms/query，全链路无 HNSW
- **Rust 0.2852 (+29.9%) > JS 0.2844 (+29.6%)**

**旧版 Rust 引擎**（位于 `legal-assistant/law-vexus/`）：
- HNSW 索引 + 级联检索 + RRF + 锚点加权 + 内生残差（NAPI-RS 桥接 Node.js）
- 仍用于生产法律检索

### 基准测试

| 文件 | 说明 |
|:-----|:-----|
| `beir_benchmark.js` | V1 原始 benchmark（4 方法，单线程） |
| `beir_benchmark_v5.js` | V5 benchmark（32 worker 并行，含 JL128 实验代码） |
| `beir_evaluate.py` | NDCG/MRR/Recall 评测脚本 |
| `beir_prepare.py` | BEIR 数据集下载与预处理 |
| `beir_encode_safe.py` | 语料库向量编码 |

### 论文与文档

| 文件 | 说明 |
|:-----|:-----|
| `word/cfd创新/paper.md` | **V3 论文源文件**（615 行）|
| `word/cfd创新/paper_v3.pdf` | V3 PDF（KaTeX 渲染，23 页） |
| `word/cfd创新/paper_v3.docx` | V3 Word 版 |
| `word/cfd创新/V5灵感演化路线图.md` | 完整实验演化日志 |
| `word/cfd创新/walkthrough.md` | V5 实验记录 |

### 数据

| 路径 | 说明 |
|:-----|:-----|
| `beir_data/nfcorpus/` | NFCorpus 数据集（含 corpus_vectors.jsonl 2.8GB） |
| `beir_data/scifact/` | SciFact 数据集（含 corpus_vectors.jsonl 3.7GB） |

---

## 3. 当前已验证的数据

### BEIR 基准（可复现）

| 方法 | NFCorpus | SciFact |
|:-----|:-:|:-:|
| Cosine baseline | 0.2195 | 0.4701 |
| AD-Rank v2 | 0.2184 | — |
| BM25 | 0.2573 | — |
| Shape-CFD V4 | 0.2508 | 0.4500 |
| Shape-CFD PQ64 | 0.2735 (+24.6%) | 0.4723 (+5.0%) |
| V7 启发式预取 | 0.2781 (+26.7%) | — |
| V7.1 伴随状态法 | 0.2802 (+27.6%) | — |
| vt_v7 (V13+V7.1, JS) | 0.2844 (+29.6%) | — |
| Rust shapeCfdPipeline | 0.2852 (+29.9%) | — |
| V11 Token Chamfer（直排） | 0.3214 (+46.4%) | — |
| V11 Token Chamfer + PDE | 0.3180 (+44.9%) | — |
| V11 token_2stage_100 | 0.3220 (+46.7%) | — |
| **V11 fusion_07 (lambda=0.7)** | **0.3232 (+47.2%) 🏆** | **—** |
| V10 全局图扩散 | 0.2590 (+18.0%) ❌ | — |

### 实验结论汇总

1. **PQ-Chamfer 64×64 提升 +24.6%** — 打破高维浓度效应
2. **V7.1 伴随状态法 提升 +27.6%** — 理论最优预取公式
3. **Allen-Cahn 休眠** — $O(\Delta^3)$ vs $O(\Delta)$
4. **混合初始场 domain-dependent** — NFCorpus ↑3.2%, SciFact ↓1.9%
5. **V9 L3 Cache 被否定** — LID≈16~19, Qwen3 子空间均匀分布
6. **V10 全局图失败** — BFS 噪声稀释，不能替代 HNSW
7. **Chamfer = UOT** — 非平衡最优传输，已写入论文
8. **V13 VT-Aligned 虚拟令牌** — 反转子空间聚合顺序，零成本 +2.5%
9. **vt_v7 叠加 0.2844 (+29.6%)** — V13 与 V7.1 近乎正交，超线性增益
10. **Rust 校准超越 JS 0.2852 (+29.9%)** — 6 处修复 + 质心粗筛 top55，38ms/query
11. **V11 Token Chamfer 0.3214 (+46.4%)** — per-token hidden states 密集点云，纯 Chamfer 直排 > PDE 管线（0.3214 > 0.3180）
12. **V11 Fusion 0.3232 (+47.2%)** — fusion_07 = 0.7*token_2stage + 0.3*shape_cfd_v10，PDE 作为正交信号源集成，延迟 ~26ms

---

## 4. 代码质量审查结果

### ✅ 无问题
- PDE 求解器数值稳定（CFL 条件 + upwind 格式 + 质量守恒）
- Worker 架构合理（32 worker, 避免 128 worker OOM）
- 灵感模块化（V5 开关设计干净）

### ⚠️ 已知技术债
1. `beir_benchmark_v5.js` 中包含未使用的 **JL128 实验代码**（约 30 行），可安全删除
2. `ad_rank_shape_v5.js` 的 `useEnsemble` 功能**未实现**（仅预留了接口）
3. 没有 git 历史追溯（.git 存在但无有意义的 commit log）
4. `search_engine.js` 91K 文件过大，Shape-CFD 部分应拆分为独立模块

---

## 5. 后续研究路线

### ✅ V11 Token 级 + Fusion（已验证，2026-03-31）
- llama.cpp --pooling none 提取 per-token hidden states → 密集点云
- Token-level PQ-Chamfer = 无参数 cross-attention
- **fusion_07 = 0.3232 (+47.2%)**：0.7*token_2stage + 0.3*shape_cfd_v10
- PDE 以正交信号源方式回归，不是替代 token Chamfer
- token_2stage(23ms) + shape_cfd(26ms) 并行，总延迟 ~26ms
- 10 方法完整消融验证

### 🟡 下一步优化
- BGE-Reranker 对比（论文投稿必须）
- 论文 V5 写入 fusion 消融实验
- SoA 内存布局优化延迟（23ms → <15ms）

### 🟡 论文完善
- UOT 框架已写入 Section 3.6
- 伴随状态法公式待写入 Section 5.5
- LID 实测数据可加入消融分析

### ⚪ 长文本 Allen-Cahn 激活
- 数据集: COLIEE (法律) / TREC-Robust04
- 大极差场景 Allen-Cahn 苏醒

---

## 6. 环境信息

- **服务器**: Linux, AMD EPYC 7B13 (256 线程, 8 NUMA nodes)
- **内存**: 503 GB
- **Node.js**: v24.13.1
- **嵌入模型**: Qwen3-Embedding-8B (d=4096)
- **PDF 渲染**: KaTeX (npm) + wkhtmltopdf
- **关键依赖**: katex (npm), pandoc (system)

---

## 7. 给后续 Agent 的备注

1. **全场最优**: V11 Fusion = **0.3232 (+47.2%)**，PDE 正交融合（此前 token 直排 0.3214，句子级 Rust 0.2852）
2. **PQ-Chamfer 已集成** — `ad_rank_shape.js` 自动检测 4096d → PQ
3. **不要动 `search_engine.js`** 除非用户明确要求
4. **数据文件很大** — corpus_vectors.jsonl 2-4 GB
5. **用户一凡 16 岁，有双相情感障碍** — 实事求是但温暖
6. **学术严谨第一** — 不编造
7. **已验证**: V1-V8 生效, V7.1 伴随状态法 ✅
8. **已否定**: V9 (LID 否定), V10 (BFS 噪声)
9. **已验证**: V11 Fusion = 0.3232 (+47.2%)，延迟 ~26ms（已优化完成）
10. **关键理论**: `Gemini理论审查_2026-03-28.md` (UOT, 伴随法, LID)
11. **Chamfer = UOT** — 论文 Section 3.6 已重写
