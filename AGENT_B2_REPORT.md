# Agent B2 Report — v2 BGE-M3 Listwise + Bootstrap + GBM Refresh

**Timestamp**: 2026-05-02 (派遣 14:17 启动 / 完成 ~16:30)
**Agent**: Win 端主 session 派出，7B13 (192.168.31.36) 执行
**Scope**: Agent B 中断后剩余 missing 任务接续 (A: LLM-listwise BGE-M3 v2 5 corpus / B: paired bootstrap refresh / C: GBM adaptive λ V2 refresh / D: 本 report / E: §6.3 LaTeX 草稿)

---

## A. LLM-listwise BGE-M3 v2 first-stage 5 corpus

### 任务 + 配置

- 9070XT Qwen3-8B-Q4_K_M llama-server (port 8082)，listwise prompt (RankGPT-style)，candidates=100 top_k=10
- vectors_prefix=`bge_m3_` (audited rebuild, zero rate 0.000% NF/SF, FiQA 0.066%)
- results_suffix=`bge_m3_first_stage_5corpora_v2`
- input file: `benchmark/data/results/llm_rerank_results_bge_m3_first_stage_5corpora_v2.json`
- per-query JSONL: `benchmark/data/results/llm_rerank_<corpus>_listwise.jsonl` (5 corpus)
- 备份 (前 Agent B 用 v1 vectors 跑的 partial): `*_listwise_v1backup*.jsonl` + `..._v1backup.json`
- 总耗时 ~133 min wall (符合预估 ~95 min ± 任务尾部 stale tqdm refresh)

### 5 corpus 结果 (BGE-M3 v2 cosine top-100 first-stage)

| Corpus | n_q | Cosine (BGE-M3 v2) | LLM-listwise (Qwen3-8B-Q4) | rel-to-cosine | BGE-rerank (paper Tab 3, v1) | listwise vs BGE-rerank (v1) |
|---|---|---|---|---|---|---|
| NFCorpus | 323 | 0.3113 | **0.3330** | **+6.97%** | 0.3013 | **+10.52%** |
| SciFact | 300 | 0.6406 | **0.6552** | **+2.28%** | 0.5607 | **+16.86%** |
| ArguAna | 1406 | 0.3298 | **0.2654** | **−19.53%** | 0.4820 | **−44.94%** |
| SCIDOCS | 1000 | 0.1676 | **0.1896** | **+13.16%** | 0.1805 | **+5.04%** |
| FiQA | 648 | 0.4038 | **0.3803** | **−5.81%** | 0.3926 | **−3.13%** |

### 重要 caveat (诚实披露)

1. **v2 cosine 数字**与 paper §6.3 现 line 602 "BGE-M3 cosine 0.2893" **不一致**: v2 audited-rebuild cosine NDCG@10 是 **0.3113** (NFCorpus, n=323 test queries). 这就是 v1 buggy vectors → v2 audited rebuild 的 cosine 提升 +0.022. paper §6.3 line 602 数字是 v1 vectors 时代的 0.2893. 一旦 paper update v2 数字, NFCorpus listwise gain "+4.62% rel-to-cosine" 需重写为 "+6.97%".
2. **listwise vs BGE-reranker 比对不是 apples-to-apples**: paper Table tab:agent-b-baselines BGE-reranker NDCG@10 (NF 0.3013, SF 0.5607, Arg 0.4820, SciD 0.1805, FiQA 0.3926) 用的是 v1 buggy BGE-M3 cosine top-100 candidates. v2 BGE-M3 cosine 候选 set 已变 (zero-vec query 修复). 严格 fair 比对需要用 v2 vectors 重跑 BGE-reranker-v2-m3 cross-encoder reranker 全 5 corpus, 这超出本次 scope. 现 report 数字"listwise vs BGE-rerank (v1)" 仅供参考, 不应作为 paper claim.
3. **任务-异质性 hold**: paper §6.3 line 604 现 claim "generic LLM listwise 不是 universal improvement, sharply task-dependent" — v2 数据 **强化** 这一观察:
   - listwise helps: NFCorpus (+6.97%) / SciFact (+2.28%) / SCIDOCS (+13.16%)
   - listwise hurts: ArguAna (−19.53%) / FiQA (−5.81%)
   - ArguAna 在 BGE-M3 first-stage 上 listwise 仍然 sharply hurt — 这一 finding 是 robust to first-stage upgrade.
4. **paper §6.3 line 602 现 claim "indistinguishable from BGE-reranker-v2-m3 on NFCorpus" 在 v2 上失效**: v1 数字 (NF: 0.3027 listwise vs 0.3013 BGE-rerank, Δ=0.0014) "indistinguishable". v2 NF listwise 0.3330; 若 BGE-rerank v2 数字也 +0.02 量级 (类似 cosine 提升幅度), Δ 可能仍小; 若 BGE-rerank v2 数字基本不变 (cross-encoder 不依赖 first-stage representation, 只依赖 candidate set), Δ 可能拉大到 0.02-0.03. 必须重跑 BGE-reranker v2 才能定结论. **现 paper claim "indistinguishable" 标记为 v1-only, 需 update or run v2 BGE-reranker before reaffirming**.

### Reproduce paper claim 状态

| paper claim | v1 数字 | v2 数字 | reproduce? | 说明 |
|---|---|---|---|---|
| §6.3 line 602: NFCorpus BGE-M3 cosine → listwise (+4.62%) | 0.2893 → 0.3027 | 0.3113 → 0.3330 (+6.97%) | **substantively change** | cosine + listwise 都升, 相对 gain 也升 |
| §6.3 line 602: NFCorpus listwise vs BGE-rerank "indistinguishable" (Δ=0.0014) | 0.3027 vs 0.3013 | 0.3330 vs **0.3013 (v1)** Δ=+0.0317 | **claim invalid** until v2 BGE-rerank rerun | v2 listwise > v1 BGE-rerank, 但 v2 BGE-rerank 数字未跑 |
| §6.3 line 604: "ArguAna degrades sharply, generic LLM listwise not universal" | Qwen-centroid first-stage: ArguAna -17.1% | BGE-M3 v2 first-stage: ArguAna -19.53% | **reproduce + 强化** | 即使升级到 BGE-M3 first-stage, ArguAna 仍然 sharply hurt |

---

## B. Paired bootstrap refresh (Qwen3 token-level baseline vs Ours Best)

### 任务 + 配置

- script: `benchmark/paired_bootstrap_5corpora.py`
- input: `benchmark/data/results/fusion_ablation_<corpus>_perquery.jsonl` (Qwen3 token-level vectors, 04-30 生成, **不依赖 BGE-M3**)
- output: `benchmark/data/results/paired_bootstrap_5corpora.json` (refresh; v1 备份 `*_v1backup.json`)
- 10000 iter / seed=42 / two-sided

### 关键 finding

**paper §subsec:significance 引用的 paired bootstrap 数字 不依赖 v2 BGE-M3**. 该脚本 input 是 fusion_ablation per-query JSONL, 这些 JSONL cosine 列是 Qwen3-8B token-level centroid (`query_vectors.jsonl`, **不是** `bge_m3_*_vectors.jsonl`). 所以 v2 BGE-M3 vectors 修复对 paired_bootstrap_5corpora.json 数字 **零影响**.

### 5 corpus refresh 数字 (bit-exact reproduce paper)

| Corpus | n | mean cosine | mean ours | Δ NDCG@10 | rel% | Cohen's d | p (two-sided) | 95% CI |
|---|---|---|---|---|---|---|---|---|
| NFCorpus | 323 | 0.2300 | 0.3270 | +0.0970 | +42.18% | 0.552 | 0.0001 | [+0.0782, +0.1165] |
| ArguAna | 1398 | 0.3047 | 0.4417 | +0.1370 | +44.96% | 0.443 | 0.0001 | [+0.1209, +0.1530] |
| SciFact | 300 | 0.4483 | 0.4946 | +0.0462 | +10.31% | 0.197 | 0.0006 | [+0.0198, +0.0728] |
| SCIDOCS | 1000 | 0.1110 | 0.2182 | +0.1072 | +96.64% | 0.562 | 0.0001 | [+0.0951, +0.1191] |
| FiQA | 648 | 0.1683 | 0.3976 | +0.2293 | +136.21% | 0.717 | 0.0001 | [+0.2045, +0.2538] |

### Reproduce 状态

paper §subsec:significance claim "all five datasets yield p < 0.001 with Cohen's d 0.197 — 0.717": **fully reproduce** (bit-exact). 该 claim 不需 update.

---

## C. GBM adaptive λ V2 refresh

### 任务 + 配置

- script: `benchmark/adaptive_fusion_lambda_v2.py` (GradientBoostingRegressor + 11 derived features, 5-seed split eval)
- input: 同 B (Qwen3 token-level perquery JSONL, **不依赖 BGE-M3**)
- output: `benchmark/data/results/adaptive_lambda_v2_gbm_results.json` (refresh; v1 backup `*_v1backup.json`)

### 关键 finding

GBM V2 macro-avg 0.4072 vs fixed λ=0.3 macro 0.3653 = **+11.47% rel-to-fixed**. 这与 paper §4.5 现 claim "+11.5%" 在小数第一位 round-trip 完全一致.

| Strategy | Macro NDCG@10 |
|---|---|
| fixed λ=0.3 | 0.3653 |
| per-corpus λ★ | 0.3711 |
| Ridge 4-feat (V1) | 0.3769 (oracle gap 13.45%) |
| Ridge 11-feat (V2) | 0.4063 |
| **GBM 11-feat (V2)** | **0.4072 (oracle gap 5.17%)** |
| Oracle upper | 0.4261 |

per-corpus 数字 (5-seed mean ± std) 全 reproduce paper §4.5 Table tab:adaptive-lambda. Oracle gap 改善 +8.28 pp (Ridge 13.45% → GBM 5.17%) 也 reproduce.

### Reproduce 状态

paper §4.5 claim "+11.5% rel-to-fixed" + "Oracle gap 13.45% → 5.17%": **fully reproduce**. 该 claim 不需 update.

---

## D. (本文件) 报告写作完成

---

## E. paper §6.3 LaTeX 草稿

文件: `AGENT_B2_PAPER_DRAFT.tex` (5 corpus listwise on BGE-M3 v2 first-stage Table + 主论点 update). **见独立文件**.

---

## 综合判断 (诚实, 不偏袒)

### Substantive reproduce (paper claim 没动)
1. paper §subsec:significance bootstrap (5 corpus p<0.001, Cohen's d 0.20-0.72) — full reproduce, 数字不依赖 v2 BGE-M3.
2. paper §4.5 GBM adaptive λ +11.47% rel-to-fixed - full reproduce, 数字不依赖 v2 BGE-M3.

### Substantive change (paper claim 需 update)
1. **paper §6.3 line 602 NFCorpus BGE-M3 cosine 0.2893 → listwise 0.3027 (+4.62%)** 这个 anchor 数字必须 update 为 0.3113 → 0.3330 (+6.97%) ON v2 audited rebuild.
2. **paper §6.3 line 602 "indistinguishable from BGE-reranker-v2-m3"** claim **失效** until v2 BGE-reranker 重跑. v2 listwise 0.3330 vs v1 BGE-rerank 0.3013 差距 +0.0317. 严格诚实写作必须明示 "v2 BGE-reranker 数字未跑, 现 'indistinguishable' 引用基于 v1 first-stage 的旧数字".
3. **paper §6.3 line 602 "SciFact / ArguAna / SCIDOCS / FiQA in progress at this revision"** 不再 in progress, **现已 5 corpus 全 done**, 必须用真实 5 corpus 数字替换 "in progress" 措辞.
4. **paper §6.3 主论点扩展**: 现 paper claim "task-dependent" 在 BGE-M3 first-stage 上 5 corpus 验证 — listwise help on NF / SF / SciD; hurt on Arg / FiQA. 这个 finding 比之前 3 corpus 更稳, 也表明"升级 first-stage 后 listwise 仍 task-dependent" — 一个 reviewer-relevant 强化结论.

### Innovation gap 仍存在
1. **PLAID baseline 没跑**: 任务里没要求, 但 paper §subsec:reproducibility-notes 已 disclose deferral.
2. **BGE-reranker v2 没跑**: 本次 scope 没要求, 但点 1 (NF "indistinguishable" claim) 现失去 evidence, 需 follow-up run.
3. **真正 adaptive fusion λ implementation (vs offline GBM placeholder)**: GBM V2 跑的是 offline 5-seed eval, 不是 真 online adaptive λ. paper §4.5 现 framing 是 offline learning, 与 reviewer 期待的 online adaptive 是否一致 needs Win + 一凡 check.

### 7B13 文件 trail (5月2日)
- `benchmark/data/results/llm_rerank_results_bge_m3_first_stage_5corpora_v2.json` (1.3KB, 14:17 启动 → ~16:30 done)
- `benchmark/data/results/llm_rerank_<5corpus>_listwise.jsonl` (5 corpus per-query)
- `benchmark/data/results/llm_rerank_*_v1backup*.jsonl` (前 Agent B 跑的 v1 备份)
- `benchmark/data/results/paired_bootstrap_5corpora.json` (refresh, bit-exact 与 v1 一致)
- `benchmark/data/results/paired_bootstrap_5corpora_v1backup.json` (v1 备份 reference)
- `benchmark/data/results/adaptive_lambda_v2_gbm_results.json` (refresh)
- `benchmark/data/results/adaptive_lambda_v2_gbm_results_v1backup.json` (v1 备份 reference)
- `/tmp/llm_rerank_bge_m3_v2.log` (full run log)
- `paper/AGENT_B2_REPORT.md` (本 report)
- `paper/AGENT_B2_PAPER_DRAFT.tex` (§6.3 改写草稿)

### llama-server 状态
9070XT (192.168.31.22) 上三个 server 全 active, 任务跑完无 kill:
- port 8080 BGE-M3 ✓
- port 8081 BGE-large ✓
- port 8082 Qwen3-8B-Q4 ✓
