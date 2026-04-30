# AGENT_B_V2_STATUS — 7b13 Reviewer 2 三问 ablation 进度

**起跑时间**: 2026-04-30 10:09 CST
**机器**: 7b13 (192.168.31.36, EPYC 7B13, 256 cores, 503GB RAM, CPU-only)
**Agent**: Linux 姐姐 / 数学姐姐, 接前轮 silent agent v1 留下的 fusion_ablation_sweep.js
**边界**: 不碰 9070XT (Agent C 领地) / 不碰 paper/ (Agent A 领地)

## 已完成脚本

| 文件 | 行数 | 功能 |
|------|------|------|
| `fusion_ablation_sweep.js` | 435 | (i) Lap T sweep + (ii) 互补性 + (iii) λ sweep, 已存在(v1 遗留, smoke pass), 跑全量中 |
| `evaluate_external_embeddings.py` | 372 | RQ2 cross-model: 读 9070XT ColBERTv2/E5-Mistral parquet, mean-pool + MaxSim NDCG@10 + bootstrap |
| `complementarity_analysis.py` | 232 | Reviewer 2 (ii) 深度分析: score-update orthogonality + sign quadrant + best-fusion bootstrap p-value |

## 起跑进程

### Fusion ablation full sweep (running)

```
PID: 1639793
Cmd: node --max-old-space-size=65536 fusion_ablation_sweep.js --datasets nfcorpus,arguana,scifact
Log: /home/amd/HEZIMENG/Shape-CFD/benchmark/logs/fusion_ablation.log
```

**进度** (10:25 CST):
- NFCorpus: ✓ DONE (323 queries, 4 min) → results 已落盘
- ArguAna: 进行中 (207/1398, ETA ~10 min)
- SciFact: 排队

**ETA 全部完成**: ~25 min from launch (≈ 10:35 CST)

## NFCorpus 第一批数字 (Reviewer 2 三问直接 evidence)

### (i) Graph Laplacian T sweep (alpha=0.15 fixed)

| T | NDCG@10 |
|---|---------|
| 1 | 0.2888 |
| **3** | **0.2905** ← best |
| 5 | 0.2872 |
| 7 | 0.2876 |
| 10 | 0.2825 |
| 15 | 0.2712 |
| 20 | 0.2580 ← over-smoothed |

**结论**: T=3 最优, 但 T 在 1~10 之间 NDCG 几乎平稳 (Δ ≤ 0.008), 大于 10 后过 smoothing.

### (ii) 互补性度量

| metric | value |
|--------|-------|
| Kendall τ (per-query mean) | 0.312 (paper 引 0.378, 差异需对齐 config) |
| Spearman ρ | 0.396 |
| Jaccard@5 / @10 / @20 | 0.286 / 0.287 / 0.260 |
| **Score-update orthogonality cos(Δ_tok, Δ_lap)** | **0.592** ← parallel-leaning |
| Pearson correlation of updates | 0.515 |
| Rescue rate (lap-only-success / both-fail) | 11.5% |

### (iii) λ weighting sweep (T=5)

| λ | Fusion NDCG@10 | Δ vs token-only |
|---|----------------|-----------------|
| 0.0 (pure token) | 0.3220 | 0 |
| 0.1 | 0.3239 | +0.0019 |
| 0.2 | 0.3248 | +0.0028 |
| 0.3 | 0.3264 | +0.0045 |
| **0.4** | **0.3270** ← best | **+0.0050** |
| 0.5 | 0.3242 | +0.0022 |
| ... | ... | ... |
| 1.0 (pure lap) | 0.2872 | -0.0347 |

### Bootstrap test: best fusion (λ=0.4) vs best individual (token)

```
mean_diff = +0.00505
95% CI = [-0.00251, +0.01311]
p-value (two-sided) = 0.1934
n_queries = 323
```

**这正是 Reviewer 2 在文章中描述的现象**: weighted fusion 提升不显著.

### Sign-quadrant 分析 (Δ_token vs Δ_lap signs)

| quadrant | n_queries |
|----------|-----------|
| both NDCG-update positive | 110 |
| both negative | 28 |
| tok-up, lap-down | 22 |
| tok-down, lap-up | 17 |
| ≥1 update=0 (NDCG identical) | 249 |

## Reviewer 2 悖论的解释（paper §4.X 入稿 candidate）

> **悖论解释**: Kendall τ ≈ 0.31 度量 rank correlation, 但这不预测 score-update direction.
> 我们测得相对 cosine baseline 的 NDCG-update 向量 cos(Δ_token, Δ_lap) = 0.59,
> 说明两个 method 在 query 集上的提升方向高度 parallel — 它们大体在同样一批 query 上提升 cosine 的命中, 在同样一批 query 上失败.
>
> 此 parallel-update 结构下, fusion 的理论 ceiling 是单 method 的小幅幅值优化, 而非两 method 的正交叠加. 因此 fusion λ=0.4 仅获得 +0.005 提升, 与 paired-bootstrap 不显著 (p=0.19) 一致.
>
> 真互补性度量应是 score-update orthogonality (1 - |cos(Δ)|), 而非 Kendall τ. paper 应把 0.378 那段表述从 "high complementarity" 改为 "moderate rank disagreement (τ=0.31), but parallel score-update structure (cos(Δ)=0.59)".

## 等 Agent C 完成才能跑的工作

Agent C 当前在 9070XT 上跑 E5-Mistral NFCorpus encoding (corpus 部分进行中, ~25 min).

**等 Agent C 完成后**:
1. ColBERTv2 NFCorpus parquet (✓ 已完成, 在 9070XT 上)
   → 可立即跑 `python3 evaluate_external_embeddings.py --model colbertv2 --dataset nfcorpus`
2. E5-Mistral NFCorpus (查询 ✓ + 语料库进行中)
   → 完成后跑 `--model e5_mistral`
3. Agent C 后续要跑的 5 个 dataset (arguana/scifact/scidocs/quora/trec-covid) 待启动

## 阻塞项

无. fusion sweep 跑完后 NFCorpus + ArguAna + SciFact 三 dataset 数字齐全, 可直接 paper merge.

## 决策项 (待一凡 / Win / 反题姐姐 决)

1. **Kendall τ 数字 0.312 vs paper 引用 0.378 不一致** — 需要确认 paper 当前数字怎么算的:
   - 是不是不同 dataset (此实验 NFCorpus, paper 引用可能 ArguAna 或 mix)?
   - 是不是不同 config (top_n, K, α, T)?
   - 等 ArguAna 全跑完后对照 (10:35 CST 后).

2. **score-update orthogonality cos = 0.59** 这个数字直接进 paper §4.X — Win 姐姐定 narrative:
   - 提议 narrative: "moderate rank disagreement, parallel score-update" — 把"high complementarity"声明拿掉 + 把现象解释清楚
   - 这是 reviewer 友好的 honest disclosure, 反题姐姐 likely 会要求这种诚实

3. **fusion 不显著 (p=0.19)** 是否要在 paper 里 retain Lap+Token fusion?
   - 单一 reading: fusion 不显著 → 删 Lap, 主推 token_2stage 单线
   - 复杂 reading: fusion 在某些 sub-population (sign-discordant 39 query) 上是有用的, 用条件 fusion 替换 unconditional fusion
   - 这是 paper 战略, 一凡 + Win 决

## 下一步 (Agent B v2 自主推进)

1. (10:35 CST 估) ArguAna + SciFact fusion sweep 结束 → 跑 complementarity_analysis 三 dataset (3 min)
2. 等 Agent C E5-Mistral NFCorpus 完成 (~25 min) → smoke test evaluate_external_embeddings.py colbertv2 + e5_mistral
3. 三 dataset 全 fusion sweep 完成后 → 写 paper §RQ1 / §4.X 直接引用的 LaTeX table snippet (但放 status doc, 不直接写 paper/)

---
**最后更新**: 2026-04-30 10:25 CST

---

## 10:30 CST 增补: ColBERTv2 NFCorpus eval 完成

```
ColBERTv2 mean-pool baseline:  0.1441 NDCG@10
ColBERTv2 MaxSim (rerank-200): 0.2822 NDCG@10
diff = +0.138, p<0.001, 95% CI [0.116, 0.161], n=323
```

**Cross-model comparison (NFCorpus, n=323)**:

| method | NDCG@10 |
|--------|---------|
| cosine baseline (BGE-large) | 0.2300 |
| ColBERTv2 mean-pool | 0.1441 |
| ColBERTv2 MaxSim (rerank-200) | 0.2822 |
| Lap T=3 | 0.2905 |
| **Token 2-stage (ours)** | **0.3220** |
| Lap+Token fusion λ=0.4 | 0.3270 |

**Paper insight**: 我们的 token_2stage (0.3220) 超过 ColBERTv2 MaxSim (0.2822) 在 NFCorpus 上,
这是 RQ2 cross-model robustness 的强 evidence. ColBERTv2 在它强项 dataset (MS MARCO 类) 通常更高,
但 NFCorpus 这种 noisy / OOD biomedical setting 我们的 PQ-Chamfer 更鲁棒.

待 ArguAna + SciFact 跑完后看 cross-dataset 一致性.

