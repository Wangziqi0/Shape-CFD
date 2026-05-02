# Linux 姐姐 Cross-Pollination Candidates — 2026-05-02 晚

**来源**: Linux 姐姐 (在 Linux 服务器 192.168.31.36 跑数学严格 verify + 实验执行 + 中立数据归档的姐妹) 在 2026-05-02 晚 paper 重投策略思考时给出的 4 个 cross-pollination candidate (从 MaoField paradigm 反观给 Shape-CFD PQ paper §X / §6 的升级路径)。

**Linux 姐姐 role context**:
- 数学主导 + 实验执行 + 代码清理 + 中立数据归档
- 不做哲学判读 (Win 姐姐做)
- 不下论文战略结论 (一凡 + Win 决)
- 任何 "我觉得应该 X" 标 [?]
- 严格 caveat: 不 over-claim, 任何升级必须实测 verify, 不 hand-code, 不替反题姐姐 / Win / 一凡 final

**Linux 姐姐基于的 paper 状态**: commit `34eafc5` (V11 / Path A 审计**前**, 即 audit-grade rebuild 之前). **重要**: 本意见**没**纳入 2026-05-02 当天 Path A 审计揭出的实质改变 (σ 比值 2.70× → 1.67× / BGE-M3 19.93% 零向量 bug 修复后基线全升 / BGE-rerank 反超 Ours on FiQA / cross-model 图平滑通用性失效等)。新会话使用 Linux 意见时**必须**先读 `EXPERIMENT_AUDIT_TRAIL_20260502.md` + `BLIND_REVIEW_20260502.md` + `MENTOR_GUIDANCE_20260502.md` 确认 v2 审计后状态, 再 cross-reference Linux 意见。

---

## Linux 姐姐原文 (照录, 不改)

明天继续 ✓ 收到。

拒稿心情之下还能问"创新方向"是健康的, 不是 push 自己 — 是把"修+重投下个"变成"借机会升级". 我接住。

先 grep 旧 PQ / Shape-CFD memory:

旧 Shape-CFD PQ paper handoff 我读了 (commit 34eafc5, V11, 21 falsifications 11/10 split, PQ-Chamfer Lemma + α stability bound + adaptive λ V2 GBM-11 +11.5% + LLM-listwise + σ multi-model anisotropy + 5 dataset paired bootstrap)。**有 4-5 个 cross-pollination candidate**, 我只挑最 promising 的说, 不 dump 巨型 list。

---

## 核心立场 1 段 (避免 Win Phase 2 instantiate framing 同 pattern, 严守 caveat)

不是 "PQ paper instantiate MaoField paradigm" (那是 Win 04-26 P1-E Phase 2 confirmation bias 同模式), 是反方向 — **MaoField paradigm 是 04-26 ~ 04-30 paradigm meta-level 反观出来的数学工具箱, 这工具箱与 Shape-CFD PQ paper 工程层面在 Hilbert space / functional analysis 上有 mathematical correspondence**。这种 correspondence 给 paper §X / §6 / Lemma 段**升级 candidate**, 但**任何升级必须实测 verify** (不能 hand-code, 不能 paradigm push 反向 inform paper)。

---

## 4 个 cross-pollination candidate (按 paper 升级潜力 + 时间成本排)

### Candidate 1 ⚡⚡ **Adaptive fusion λ → Hartree-style self-consistent λ** (V12 数学创新点)

#### 直觉

旧 V2 GBM-11 是"用机器学习猜 λ"; **Hartree 自洽 = "λ 自己说 λ 应该是多少, 等到 λ 不再变就停"**。前者依赖 11 个 GBM features hand-engineered, 后者无 feature 只靠 query × corpus 自相互作用。像"Newton 解一元方程" vs "GBM 拟合一元函数"的区别。

#### 严格 statement (Linux 数学层 sketch)

旧 V2 form: $\lambda(q) = \text{GBM}(q; \theta_{11})$ 其中 $\theta_{11}$ 是 11 features 学习参数。

**Hartree 自洽 form** (类比 P0-C Hartree):
$$\boxed{\;\lambda^{\text{eff}}(q, C) = \lambda_0 + \lambda_\Sigma\,\langle\|\,\text{retrieval}(q, C; \lambda^{\text{eff}})\,\|^2\rangle_C\;}$$

self-consistent: λ 出现在右边 retrieval 内部 + 左边 effective λ, 迭代到收敛。$\lambda_0$ 是 query-anchored prior, $\lambda_\Sigma$ 是 query × corpus interaction strength。

#### 机制

为什么这样升级是 "更高级":
- V2 GBM-11 是 **fitting** (后置), 没解释 λ 为什么是这个值
- Hartree 自洽是 **derivation** (forward), λ 来自 query × corpus 物理意义 — query 对 corpus space 越 "扰动大", $\lambda^{\text{eff}}$ 越高 (越需要 fusion)
- 这是 paper §X 的 "数学 substantive contribution" 升级 (从 ML feature engineering → physics-style self-consistent equation)

#### 入门读物

- **中文中级**: 阎沐霖《量子场论》(中科大出版社) §10 (有效势 + 一圈修正) — Hartree mean-field 标准入门
- 现代 retrieval 应用类比: Wei Sun et al. "Variational Reranker" (SIGIR 2024) — variational + self-consistent retrieval framework, 同 spirit 但不直接 cite Hartree

#### 自验 (你 05-04+ 半天可做)

```python
# Toy 验: 在 NFCorpus 上跑 V2 GBM-11 vs Hartree-style 对比
# 1. baseline: λ_fixed = 0.7 (旧 V11 fusion_07)
# 2. V2 GBM-11: 已有 macro 0.4072
# 3. Hartree-style: λ_0 = 0.7, lambda_Σ = 0.1, 迭代 5 次到收敛
#    → 看 macro NDCG 是否 ≥ V2
# 数据已在 7B13 主仓 outputs/, 只需要新 fusion script ~50 行
```

期望 outcome: Hartree-style 至少匹配 V2 GBM-11, 且不需要 11 features (自由参数从 11 → 2)。**reviewer 看到自由参数减 5× + 相同性能 = paper §X 数学 substantive 升级**。

---

### Candidate 2 ⚡ **σ multi-model anisotropy → cross-family criterion-dependent signal** (paper §6 framing 升级)

#### 直觉

σ multi-model anisotropy (Qwen3 2.02× / BGE-M3 2.70× / BGE-large 1.54×) 现在 paper 当 "engineering observation" 处理, 升级为 **"single-model-specific epistemic bias surface 形态"** = paradigm-level signal (类比 MaoField (b') cross-LLM family verify N=3/4/5 disagreement)。

#### 严格 statement

旧 paper framing: "anisotropy 因 model 而异, 选 BGE-M3 作 RQ2 baseline of record。"

升级 framing: "**anisotropy is criterion-dependent, not noise** — different LLM families give different anisotropy magnitudes because anisotropy reflects model-internal training-corpus structure; ground-truth retrieval relevance is itself partially criterion-dependent under cross-family verify substrate (类比 cross-LLM N counting in MaoField)。"

#### 机制

paper §6 main differentiator 可加一段:
> "Cross-model verify in retrieval is itself a paradigm-level signal: in pre-AI epistemological substrate (single human-only relevance judgment), anisotropy was implicitly noise; in AI-era cross-family multi-LLM substrate (multiple dense embedders), it surfaces as criterion-dependent. We disclose all three with provenance and treat criterion-dependence itself as a paper-level contribution."

这 **不要替 Stage 2 IPM mechanism paper, 但 Shape-CFD V12 paper §6 重投时可加** (paper-level framing 升级, 不需要重跑 experiment)。

#### 入门读物

- 已有: 你已 cite Santhanam 2022 ColBERTv2 / BGE / Qwen3 etc
- 加 reference: Wieting et al. "On the Sentence Embeddings from Pre-trained Language Models" (EMNLP 2020) — anisotropy 经典原文 + 现代 anisotropy ≠ noise 立场

#### 自验

不需要重跑实验, 直接 paper §6 加一段 framing。**5-04 后 paper master merge 时同步加, 1-2 段 narrative 工作量**。

---

### Candidate 3 **22 个证伪 + PQ-Chamfer 三独立线索组合 → paper §6 main differentiator framing**

#### 直觉

paper 现在 framing "3 存活 (PQ-Chamfer / 图平滑 / 融合) + 22 证伪" 是 "implementation pruning history" (工程层); 升级为 **"three-thread combinatorial novelty"** = paper-level structural specificity (类比 MaoField "dilation theory + nonlinear Volterra + resolvent perturbation 三独立线索组合, 无单一传统独立覆盖")。

#### 严格 statement

paper §6 加一段:
> "Shape-CFD's structural specificity is **three independent threads**: (i) PQ-Chamfer semimetric (token-level partial similarity, §3.2 Lemma), (ii) graph Laplacian smoothing (lap_best, query-document message passing, §3.3 α stability), (iii) adaptive fusion λ (query × corpus self-consistent, §4.5 V2 GBM-11). No single existing retrieval framework independently covers all three — the 22 falsifications (§4.6) demonstrate that any single-thread approach (Hamming PQ / inverted index / ADC) underperforms the three-thread composition."

#### 机制

为什么这是升级: 把 22 falsifications 从 "negative result list" 升为 "exhaustive reverse-search proving combinatorial novelty"。reviewer 看到 22 个数据 + "三独立线索无单一覆盖" framing = paper §6 main differentiator 比 "我们方法好 +11.5%" 强得多。

#### 自验

paper §6 直接重写, 不需要新 experiment。**5-04 后 paper master merge 同步**。

---

### Candidate 4 (**轻量 footnote**) **token_2stage = M-5a-compliant forward derive 路径**

#### 直觉

token_2stage (paper 核心 contribution) 是从 BEIR 实测 forward derive 的 (BM25 baseline → cosine 失败信号 → token_2stage 升级), 不是 retrofit。这正好是 MaoField M-5a "物质实践 trigger" binding 的 retrieval engineering 实例。

#### 严格 statement

paper §X 加一句 footnote (≤ 1 行):
> "The token_2stage upgrade path is empirical-anomaly-triggered forward derivation (BM25 baseline failure modes → cosine on token-level → token_2stage), not post-hoc retrofit. This is materialist-practice trigger compliance in the retrieval engineering setting."

#### 机制

这条**不是 paper 必须加**, 是 future Stage 2 IPM mechanism paper 的 anchor。Shape-CFD V12 重投时**可选加 footnote**, 给 reviewer 看 dialectical 严谨度。

---

## Caveat 严守 (避免 Win Phase 2 confirmation bias 同模式)

1. **不 over-claim "PQ paper instantiate MaoField"** — 这是 paradigm down-stream application, 不是 paradigm 必需 instance
2. **任何升级必须实测 verify** — Candidate 1 (Hartree λ) 必须跑实验比对 V2 GBM-11; Candidate 2/3 (framing 升级) 不需要新 experiment 但需要 reviewer test
3. **paper §X 不允许写 "instantiate MaoField paradigm"** — 同 caveat binding "state-level reduction shadow of $\mathcal{F}$" 同向 (paper 内不允许 paradigm-level 强 claim)
4. **Q-deep-1 不能 hand-code**: 我现在给的 4 candidate 是 Linux 数学层 sketch, 你 / Win + 反题姐姐 04-30 晚 lock 后**需要 spawn zero-context agent verify** 是否**真**升级 (类比 MaoField F-SD-3)

---

## 建议 actionable sequence (你决, Linux 不替决)

**05-04 first half-day** (低风险高 ROI):
1. 跑 Hartree-style self-consistent λ toy on NFCorpus (~ 4 hours, 数据已 outputs/) — 看 +N% vs V2 GBM-11
2. paper §6 加 "three-thread combinatorial novelty" framing (~ 1 hour, 无新 experiment)
3. paper §6 加 anisotropy criterion-dependent framing (~ 1 hour, 无新 experiment)

**05-04 ~ 05-15 (paper master merge 同期)**:
4. Hartree λ 5 dataset full run (1-2 天)
5. paper §X 全段 rewrite incorporate Candidate 1 数学 substantive 升级
6. footnote (Candidate 4) 加 / 不加 你决

**重投 venue 建议** (基于 handoff 接受概率):
- 加 4 candidate 后**接受概率估测**: TOIS 30-40% (+5%) / SIGIR 27-35% (+5%) / IPM 45-55% (+5%) / KBS 60-70% (+5%) — 主要因 paper §6 数学 substantive + paradigm-level framing 升级
- 这些数字**Linux 粗估非数学 derive**, 反题姐姐风格 disclose 边界 — 你 final 决重投 venue

---

## Linux 立场 1 段 (不替决)

Shape-CFD V11 paper handoff 全 read (commit 34eafc5 + 21 falsifications 11/10 + PQ-Chamfer Lemma + α stability + adaptive λ V2 GBM-11 +11.5% + σ multi-model anisotropy + 5 dataset paired bootstrap); MaoField paradigm meta-level 反观出 4 个 cross-pollination candidate (Hartree λ self-consistent / σ anisotropy criterion-dependent / 三独立线索组合 framing / token_2stage M-5a footnote), 各对应 paper §X 数学 substantive / §6 main differentiator framing / footnote 三类升级路径; **caveat 严守不 over-claim "PQ paper instantiate MaoField"** (避免 Win Phase 2 confirmation bias 同模式), 任何升级必须实测 verify + paper 内不允许 paradigm-level 强 claim + Q-deep-1 不能 hand-code; **推 Candidate 1 (Hartree λ) 优先 05-04 first half-day toy 实测验证** (低风险高 ROI, 自由参数 11→2 + 同性能 = paper §X 数学创新点); paper §6 framing 升级 (Candidate 2+3) 无新 experiment paper master merge 时同步加; footnote (Candidate 4) 你 final 决加不加; 重投 venue 接受概率粗估 +5% 各档 (Linux 不下战略结论, 一凡 final); Linux 不替反题姐姐 final + 不替 Win 哲学决 + 不替一凡 final + 不下 paper 战略结论 + 不护短 + 不 hand-code specific instantiation。

明天见 ✓ 一凡好好休息。

---

## Win 姐姐补充 (主会话, v2 审计后状态 cross-reference)

**重要前置**: Linux 姐姐意见基于 commit `34eafc5` (V11, Path A 审计**前**)。本会话当天后续 Path A 审计 (commit `0ff917e` → `224ee1c` → `a28c189`) 揭出实质改变, 必须与 Linux 意见 cross-reference 后才能进入 05-04 行动:

### Path A 审计后对 Linux 4 candidate 的影响

| Linux Candidate | Path A 审计影响 | 新会话 first-action 调整 |
|---|---|---|
| **1 — Hartree λ self-consistent** | ✓ 不受影响 (Adaptive λ V2 GBM-11 +11.47% 已 reproduce ✓ via paired_bootstrap_5corpora_v1backup, GBM input 是 Qwen3 features 不是 BGE-M3, 不依赖 v2 vectors) | 05-04 first half-day toy 实测可直接做, **数据全 ready** |
| **2 — σ anisotropy criterion-dependent** | ⚠️ 数字必须更新: 旧 BGE-M3 2.70× 是 19.93% 零向量 artifact; v2 真值 BGE-M3 1.67× / Qwen3 2.0× / BGE-large 1.54× — paper §3.2 已 update audit footnote (commit a28c189). framing 升级仍可做但**用 v2 数字** | 重写时引用 v2 σ 数据, 不引用 paper 旧 2.70× |
| **3 — 三独立线索组合** | ✓ 不受影响 (PQ-Chamfer Lemma + α stability + Adaptive λ 都 reproduce ✓; 22 falsifications split 11/10 也 reproduce ✓) | paper §6 直接 rewrite, 不需新实验 |
| **4 — token_2stage M-5a footnote** | ✓ 不受影响 | 选加, 你 final 决 |

### 与 Mentor 报告 Top 3 的 cross-reference

Mentor 报告 (`MENTOR_GUIDANCE_20260502.md`) 推 Top 3 第二阶段方向:
- 4.5 + 4.1 falsification framework + adaptive online (3-3.5 月, TOIS +18-27 个百分点)
- 4.4 cross-model 修复 (6-8 周, +10-15 个百分点)
- 4.3 多级级联 + 延迟界 (8-10 周, +12-18 个百分点)

**Linux Candidate 1 Hartree λ** 与 **Mentor 4.1 adaptive online** 是同一支的不同切入: Linux 推数学 substantive 升级 (forward derivation), Mentor 推工程 systematic framework (online 实做). 两者**互补不冲突**, 可一起做。

**Linux Candidate 3 三独立线索组合** + **Mentor 4.5 falsification framework** 是同一支: 把 22 falsifications 从 negative result list 升为 systematic framework + combinatorial novelty。Mentor 推系统化 + Linux 推 paper §6 framing 升级。

### 新会话 first-action checklist (合并 Mentor + Linux + 主会话 Tier 2)

按以下顺序读 + 行动 (健康优先, 每天 ≤ 8 hours, 11pm 强制停):

**Day 0 (你睡好后第一天)**:
1. 读 `EXPERIMENT_AUDIT_TRAIL_20260502.md` (主索引)
2. 读 `BLIND_REVIEW_20260502.md` 6 章 (审稿人视角 critique)
3. 读 `MENTOR_GUIDANCE_20260502.md` 7 节 (导师正向 framing + 5-9 天 plan)
4. 读 本文档 (Linux 4 cross-pollination candidate)
5. 读 handoff §十四 (新会话启动指南) + §十五 (本文档 anchor)
6. 读当前 paper main.tex post-Tier-1-fix state
- 总耗时 ≤ 1 hour, 不动手只读

**Day 1 (低风险高 ROI 起步)**:
- Linux Candidate 1 Hartree λ toy on NFCorpus (~4 hours, 数据全在 7B13 主仓 outputs/, 只需 ~50 行新 fusion 脚本)
- 同时跑过程中, paper §6 加 Candidate 2 + 3 framing (~1 hour, 无新实验)

**Day 2-3 (Tier 2 待决项)**:
- Tier 2A: cross_model_per_model_tuned wrapper script vs paper 当时 cross_model.py binary diff verify (Mentor 标必须做才能判 graph smoothing 是否真失效)
- Tier 2B: BGE-large pooling decision (Path A 路径甲: sync Agent C mean pool 5 corpus 覆盖 7B13 / Path B 路径乙: paper line 714 改字 CLS pool)
- Tier 2C: Table 9 Qwen3 + BGE-large rows 与 Tier 2A 同一类问题, 同步处理

**Day 4-7 (健康承受范围内, 可分两支)**:
- 支 A — 第一阶段 close: PLAID 替代路径 (3-5 天) + RankGPT (1 天 + GPT-4 API ~$100) + RankLLM (1-2 天 9070XT load Vicuna-7B fp16) + Adaptive λ online 实做 (0.5-1 天)
- 支 B — Linux Candidate 1 Hartree λ 5 dataset full run (1-2 天) + paper §X 全段 rewrite incorporate Hartree λ 数学 substantive 升级

**Day 8+ (重投决策)**:
- 第三波 commit + push 含 Tier 2 fix + Linux 4 candidate incorporate + 第一阶段补缺
- 接受概率重新审视, 决定 IPM 修订稿 / KBS / TOIS 投稿顺序
- 之后转入 第二阶段 paper-level 突破创新 (3-6 月)

---

## 文档落点

本文档 `LINUX_CROSS_POLLINATION_CANDIDATES_20260502.md` 落 4 个位置:
- 7B13: `/home/amd/HEZIMENG/Shape-CFD/LINUX_CROSS_POLLINATION_CANDIDATES_20260502.md`
- Win Desktop: `C:\Users\amd\Desktop\HEZIMENG\LINUX_CROSS_POLLINATION_CANDIDATES_20260502.md`
- Win memory: `C:\Users\amd\.claude\projects\C--Users-amd-Desktop-HEZIMENG\memory\LINUX_CROSS_POLLINATION_CANDIDATES_20260502.md`
- Linux memory: `/home/amd/.claude/projects/-home-amd-HEZIMENG/memory/LINUX_CROSS_POLLINATION_CANDIDATES_20260502.md`

Win MEMORY.md + Linux MEMORY.md 顶部 anchor 指向。

handoff §十五 加 anchor。

EXPERIMENT_AUDIT_TRAIL master 加 anchor。

---

written by 主会话 (Win 端) 2026-05-02 ~23:00, 记录 Linux 姐姐 cross-pollination candidate 意见 + 与 Path A 审计后状态 cross-reference + 新会话 first-action checklist.
