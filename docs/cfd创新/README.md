# Shape-CFD 项目文档索引

> 本目录混合了"算法研究"和"法律 AI 产品"两类文档。以下按类别整理。
> 最后更新: 2026-04-05

---

## 一、研究阶段（算法研发）

### 论文与理论

| 文件 | 说明 |
|:-----|:-----|
| `paper.md` | 论文 Markdown 源文件（V4 supplement，含 VT/V7 章节） |
| `V5灵感演化路线图.md` | 从 AD-Rank 到 V11 Fusion 的完整灵感演化链（64KB） |
| `创新点全清单.md` | 37 个创新点 + 落地状态 + 六数据集最终结果 |
| `HANDOFF.md` | 项目交接文档（代码清单、环境信息、后续路线） |
| `walkthrough.md` | V5 实验全记录（数据 + 分析） |

### 实验数据

| 文件 | 说明 |
|:-----|:-----|
| `实验数据/experiment_baseline.json` | 基线实验 |
| `实验数据/experiment_blind_eval.json` | 盲评实验 |
| `实验数据/experiment_final_eval.json` | 最终评测 |
| `实验数据/experiment_param_sweep.json` | 参数网格搜索 |
| `实验数据/experiment_signal_amplify.json` | 信号放大实验 |
| `实验数据/experiment_v3.json` | V3 消融实验 |

### 证伪记录（红蓝对抗）

| 文件 | 说明 |
|:-----|:-----|
| `红蓝对抗/问Gemini的问题清单.md` | Claude 向 Gemini 发起的理论质询 |
| `红蓝对抗/Gemini理论分析结果.md` | Gemini 第一轮回复 |
| `红蓝对抗/Gemini理论审查_2026-03-28.md` | Gemini 第二轮深度审查（UOT、伴随法、LID） |
| `红蓝对抗/geminissay.md` | Gemini 综合评述 |

### 历史参考

| 文件 | 说明 |
|:-----|:-----|
| `历史参考/参考论文清单.md` | 引用文献整理 |
| `历史参考/对流-扩散修正报告.md` | PDE 数值修正记录 |
| `历史参考/任务分工.md` | 早期任务分配 |
| `历史参考/AD-Rank技术全景图.md` | AD-Rank 完整技术概览 |
| `历史参考/CFD-RAG研究完整总结.md` | 阶段性研究总结 |
| `历史参考/Shape-CFD论文资料包.md` | 论文投稿所需材料 |

### Agent 指令（已完成使命，归档）

| 文件 | 说明 |
|:-----|:-----|
| `agent指令/agent指令-Agent1-核心求解器.md` | PDE 核心求解器实现指令 |
| `agent指令/agent指令-Agent2-数据接口.md` | 数据接口对接指令 |
| `agent指令/agent指令-Agent3-适配管线.md` | RAG 管线适配指令 |
| `agent指令/agent指令-Agent4-性能优化.md` | 性能优化指令 |
| `agent指令/agent指令-Agent5-极致优化.md` | 极致优化指令 |
| `agent指令/agent指令-Agent6-调参实验.md` | 参数调优实验指令 |
| `agent指令/agent指令-Agent7-多方向对流.md` | 多方向对流指令 |
| `agent指令/agent指令-Agent8-信号放大.md` | 信号放大实验指令 |
| `agent指令/agent指令-Agent9-自适应参数.md` | 自适应参数指令 |
| `agent指令/agent指令-Agent10-形状CFD.md` | Shape-CFD 集成指令 |
| `agent指令/agent指令-Agent11-横评集成.md` | 横评集成指令 |
| `agent指令/agent指令-Agent12-Shape速度优化.md` | 速度优化指令 |

### 论文输出（各版本 PDF/DOCX/HTML）

| 文件 | 说明 |
|:-----|:-----|
| `论文输出/paper.docx` | V1 Word 版 |
| `论文输出/paper.html` | V1 HTML 版 |
| `论文输出/paper.pdf` | V1 PDF 版 |
| `论文输出/paper_rendered.html` | V1 渲染版 |
| `论文输出/paper_v3.docx` | V3 Word 版 |
| `论文输出/paper_v3.pdf` | V3 PDF 版（23 页） |
| `论文输出/CFD-RAG研究完整总结.pdf` | 研究总结 PDF |
| `论文输出/Shape-CFD-Report.html` | 项目报告 HTML |
| `论文输出/Shape-CFD-Report.pdf` | 项目报告 PDF |
| `论文输出/Shape-CFD_v4_Chen_Yifan.*` | V4 论文（docx/html/pdf） |
| `论文输出/Shape-CFD_v5_Chen_Yifan.*` | V5 论文（docx/html/pdf） |

### 工具脚本

| 文件 | 说明 |
|:-----|:-----|
| `工具脚本/md2pdf.py` | Markdown 转 PDF 脚本 |
| `工具脚本/ad-rank-demo.html` | AD-Rank 可视化 demo |
| `工具脚本/reynolds-derivation.html` | Reynolds 数推导可视化 |

### 算法代码（Rust 引擎）

新版 Rust 引擎位于独立仓库 `/home/amd/HEZIMENG/law-vexus/src/`:

| 模块 | 功能 |
|:-----|:-----|
| `pq_chamfer.rs` | PQ-Chamfer 子空间距离 |
| `token_chamfer.rs` | Token 两阶段检索 |
| `pde.rs` | 图 Laplacian 平滑 / Upwind PDE 求解 |
| `cloud_store.rs` | SQLite 点云存储 + 范数预缓存 |
| `vt_distance.rs` | VT-Aligned 距离 + KNN 图 + rayon 并行 |
| `fullscan.rs` | 全库暴力扫描（替代 HNSW） |

### Benchmark 脚本（30+ 实验脚本）

位于 `/home/amd/HEZIMENG/legal-assistant/` 根目录:

| 文件 | 说明 |
|:-----|:-----|
| `beir_benchmark.js` | V1 原始 benchmark |
| `beir_benchmark_v5.js` | V5 benchmark（32 worker 并行） |
| `beir_rust_parallel.js` | 64-worker 并行（vt_aligned / v7_adjoint） |
| `beir_rust_verify.js` | 单线程对比验证 |
| `beir_evaluate.py` | NDCG/MRR/Recall 评测 |
| `beir_prepare.py` | BEIR 数据集预处理 |
| `beir_encode_safe.py` | 语料库向量编码 |
| `beir_multi_bench.js` | 多数据集批量评测 |
| `beir_fiqa_bench.js` | FiQA 专项评测 |
| `beir_laplacian_test.js` | 图 Laplacian 消融 |
| 其他 `beir_*.js` / `beir_*.py` | 各类消融 / 专项实验 |

---

## 二、实践项目（法律 AI 产品）

### 产品后端

位于 `/home/amd/HEZIMENG/legal-assistant/`:

| 文件 | 说明 |
|:-----|:-----|
| `server.js` | Express 服务器入口 |
| `qa_engine.js` | 核心 QA 引擎（2000+ 行） |
| `search_engine.js` | 向量检索引擎（RAG 主管线，集成 Shape-CFD V4） |
| `law_vexus_bridge.js` | Rust 引擎 NAPI-RS 桥接 |
| `vectorize_engine.js` | Qwen3-Embedding-8B 向量化引擎 |
| `ad_rank_shape.js` | Shape-CFD V4+V8 生产版（PQ-Chamfer + PDE） |
| `auth_service.js` | 认证服务 |
| `user_database.js` | 用户数据库 |
| `plugin_router.js` | 插件路由 |
| `rag_config.js` | RAG 配置 |

### 产品前端

| 路径 | 说明 |
|:-----|:-----|
| `app/` | Uni-app + Vue3 前端（微信小程序 / H5 / 桌面端） |

### 产品设计文档

位于 `/home/amd/HEZIMENG/legal-assistant/word/落地设计/`:

| 文件 | 说明 |
|:-----|:-----|
| `D1_回答完整性设计.md` | 回答完整性保障 |
| `D2_上下文精准关联设计.md` | 上下文关联优化 |
| `D3_分层级联检索设计.md` | 分层级联检索架构 |
| `D4_相似辨别设计.md` | 相似法条辨别 |
| `D5_多模态分流设计.md` | 多模态问题分流 |
| `D6_合同起草优化设计.md` | 合同起草功能 |
| `D7_工程基础设施设计.md` | 工程基础设施 |
| `D8_测试验证方案.md` | 测试验证方案 |
| `D9A-D9E` | Rust 引擎设计（骨架/NAPI/级联/残差/集成） |
| `D10_前端适配与动画设计.md` | 前端动画设计 |

### 业务流程

位于 `/home/amd/HEZIMENG/legal-assistant/word/workflows/`:

| 文件 | 说明 |
|:-----|:-----|
| `W2_vcp_plagiarism_check.md` | 查重合规检测流程 |
| `W3_domain_qa_test.md` | 领域 QA 测试流程 |
| `W4_model_routing.md` | 模型路由策略 |
| `W5_thinking_chain.md` | 思维链展示流程 |

### 知识库（不可修改）

| 路径 | 说明 |
|:-----|:-----|
| `knowledge_base/` | 法条 + 案例 + 向量索引 |
| `法律数据/` | 法条原文库 |

### 部署与商务

位于 `/home/amd/HEZIMENG/legal-assistant/word/部署与商务/`:

| 文件 | 说明 |
|:-----|:-----|
| `部署教程_微信小程序上云.md` | 云部署教程 |
| `部署教程_微信小程序与阿里云.md` | 阿里云部署教程 |
| `答辩问答手册_创新大赛.md` | 创新大赛答辩准备 |
| `商务招商PPT文案.md` | 商务招商材料 |
| `项目核心差异化介绍_致导师.md` | 导师汇报材料 |

---

## 三、快速导航

- 想了解**研究全貌** --> `创新点全清单.md`（37 个创新点总索引）
- 想了解**代码位置** --> `HANDOFF.md`（交接文档，含完整文件清单）
- 想了解**论文内容** --> `paper.md`（论文源文件）
- 想了解**研究管线 vs 产品管线差距** --> `管线对照表.md`
- 想了解**产品功能设计** --> `word/落地设计/D1-D10`
- 想了解**API 接口** --> `docs/interface.md`
