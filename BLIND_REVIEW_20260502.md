# BLIND REVIEW — Shape-CFD v2 (commit `224ee1c`, 2026-05-02)

**Reviewer Persona**: Adversarial blind reviewer simulating IPM / TOIS / KBS / SIGIR / AAAI standards. Brutally honest, zero accommodation, zero sympathy. Goal: find every single rejection-grade weakness in the current post-Path-A v2 manuscript so the author can preempt them.

**Document under review**: `paper/main.tex` (~998 lines, 121 KB), `EXPERIMENT_AUDIT_TRAIL_20260502.md`, `ROOT_CAUSE_INVESTIGATION_20260502.md`, `AGENT_A_REPORT.md`, `AGENT_B2_REPORT.md`, raw 9070XT/7B13/5060 outputs.

**Verdict at a glance**: This paper is in a **structural crisis** that is partially masked by an admirable amount of hygiene work. The audit-grade rebuild has done its job — the data are now internally honest — but in doing so it has **publicly invalidated the central headline claim** of every previous version. What remains is a paper whose abstract still reads like a clear methodological win and whose tables tell a far more ambiguous story. No top-tier reviewer who reads both will be persuaded.

---

## Section 1 — Soundness Critique (实验设计 / 数据可信度)

### 1.1 The σ measurement that motivates §3.2 is empirically broken on the audit-grade vectors

Paper line 196: *"BGE-M3 σ_sub/σ_full = 2.70×, BGE-large σ_sub/σ_full = 1.54×"* — these numbers are presented as **cross-model verification of the central concentration-breaking mechanism**. The audit-grade rebuild measured **BGE-M3 v2 ratio = 1.57–1.80×** (per `EXPERIMENT_AUDIT_TRAIL_20260502.md` §四). The headline 2.70× was driven in non-trivial part by 19.93% / 18.62% of NFCorpus / SciFact corpus vectors being silently zero — **a numerical artefact, not a property of the embedding model**. The paper continues to assert 2.70× for BGE-M3 in §3.2, while reporting v2-audited BGE-M3 cosine and rerank numbers in Table 8. **This is internally inconsistent.** A reviewer running `measure_sigma_subspace.py` on the released v2 vectors will get 1.6–1.8×, not 2.70×. **Reject-grade.**

### 1.2 Table 8 vs Table 9 cosine numbers disagree

Table 8 (post-audit): NFCorpus BGE-M3 = 0.3113. Table 9 (cross-model unified): NFCorpus BGE-M3 cosine = 0.2893 — uses pre-audit number. Same vector source is supposed to feed both. The Vector-provenance note partially flags this, but the table itself is not refreshed. A reviewer who diffs Table 8 against Table 9 will catch this in 30 seconds. **Major revision required.**

### 1.3 Table 8 SciFact BGE-M3 = 0.6406 substantially exceeds published BEIR baselines

Published MTEB / BEIR BGE-M3 SciFact NDCG@10 reports cluster in 0.62–0.64 range. 0.6406 is plausible but on the high end. Combined with the 0.7228 BGE-rerank-v2 figure on SciFact (extremely high — published BGE-rerank-v2 papers typically report SciFact ≤ 0.71), a reviewer will ask: "Is this an over-fit candidate set? Are the relevance judgments matched? Is this a `trec_eval`-equivalent computation?" The paper provides **no evidence the evaluation harness is `pytrec_eval` or `trec_eval` compatible** — the numbers are computed by an in-house script. Without that anchor, all v2 numbers are **single-source-of-truth from the author's own scripts** and not independently verifiable.

### 1.4 ColBERTv2 in-house re-evaluation diverges from published by up to 25% (FiQA)

Table 8 ColBERTv2 FiQA = 0.3064 vs published 0.356. The paper attributes this to "shorter max sequence length (32 query / 180 doc tokens vs 220)" and "memory-bounded chunked MaxSim (chunk size 6000)". Both are legitimate engineering compromises; neither is reviewer-acceptable for a top-tier IR venue. A reviewer will ask: **"Why didn't you reproduce ColBERTv2 at its published configuration?"** If the answer is hardware constraints, the paper effectively admits its baselines are degraded versions of the real thing — and the "Ours beats ColBERTv2 on ArguAna and NFCorpus" comparison loses its bite.

### 1.5 Adaptive λ V2 GBM is offline 5-seed train/test split, not online learning

Paper §4.5 frames "+11.5% gain via GBM" as adaptive fusion. Reviewer 3 of IPM-D-26-02154 explicitly asked for **online adaptive learning** (per the audit trail W6). The current implementation uses `GradientBoostingRegressor` trained on oracle λ★ in a 50/50 split — this is **closed-form supervised regression**, not online adaptation, not even contextual-bandit framed. A reviewer who reads §4.5 carefully will say: "This is offline λ prediction trained on labeled data; the contribution is a fitted regressor, not an adaptive policy."

### 1.6 The "Ours" pipeline secretly relies on dataset-specific λ★ tuning

Table 8 footnote: NFCorpus λ★=0.4, SciFact λ★=0.7, SCIDOCS λ★=0.2, FiQA λ★=0.2, ArguAna λ★=0.0. **5 different λ values for 5 datasets** — selected per-dataset to maximize NDCG@10. This is **per-dataset hyperparameter selection on the test set**. The paper attempts to mitigate this by introducing the GBM regressor in §4.5, but Table 8's headline numbers use the oracle λ★, not the GBM-predicted λ. A strict reviewer will say: **"Your headline numbers are oracle-optimal — you cannot claim them as method performance."**

### 1.7 σ_sub/σ_full discrimination math is self-undermined in §6

§6.5 (cross-model unified-config control): "Across the 5-corpus BGE-M3 grid, the average relative gain is −9.99% (4 of 5 negative, 1 positive)." This is a **devastating internal admission**. The paper now has to say: "Our universal post-processing layer is universal only when you re-tune (α, T) per (model, corpus) pair." This is no longer "model-agnostic"; it is **per-(model, corpus) hyperparameter search**, framed kindly. RQ2 is, by the paper's own admission in §6.5 and §6.6, **not robustly answered in the affirmative**.

### 1.8 The "training-free" claim is elided by per-corpus hyperparameter tuning

Strictly, hyperparameter tuning on the test set is **a form of training**. The paper sells "training-free" as a paradigm, then quietly tunes 5 datasets × 3 hyperparameters (α, T, λ★). A semantic reviewer will reject the framing.

### 1.9 21 falsified approaches: 10 of 21 are retrospectively reconstructed

Provenance disclosure (paper line 119, Appendix A.3) admits 10/21 falsified approaches **lack per-query JSONL** — they are "retrospectively reconstructed from contemporaneous notes and aggregate result summaries." A reviewer will read this as: **"Half of your falsification claims are not reproducible."** That is 47.6% of one of the four headline contributions.

### 1.10 Significance test reports p < 0.001 for Cohen's d as low as 0.197

Paper §significance line 402 reports SciFact d=0.197 with p < 0.001. Cohen's d=0.197 is conventionally **"small effect"** and is being driven to p < 0.001 only by n=300 and the bootstrap framework. Reporting "all 5 datasets significant at 0.001" without separating effect-size magnitudes is **statistical inflation**. A statistically literate reviewer will note this.

---

## Section 2 — Significance Critique (paper-level innovation)

### 2.1 The headline claim has structurally collapsed

Pre-audit (v11): "first on NFCorpus / SCIDOCS / FiQA". Post-audit (v2): **"first on SCIDOCS only"** — tied on NFCorpus, second on FiQA / ArguAna, **third on SciFact (−31.6%)**. The paper's contribution narrative in §1.1 (Contributions) is rewritten to acknowledge this, but the abstract still reads "surpassing BGE-large-en-v1.5 on two benchmarks (FiQA: 0.398 vs 0.367; SCIDOCS: 0.218 vs 0.162)" — **comparing to BGE-large, not to BGE-rerank-v2** (which now beats Ours on FiQA and 4/5 corpora overall). This is **selective baseline cherry-picking in the abstract**.

### 2.2 The paper is 4 small angles aggregated, not 1 paper-level breakthrough

A. **PQ-without-Q** — using PQ subspace decomposition for distance, not compression. Modestly clever; competing literature (e.g., FAISS asymmetric distance) has used the same partition for decades, just for different ends. Δ-novelty: low-medium.

B. **Graph Laplacian on point-cloud distances** — graph reranking on cross-document edges weighted by Chamfer, not centroid cosine. Cute, but G-RAG / GAR / DiffRAG cover the surrounding territory; the only differentiator is the edge-weight function.

C. **21 falsifications** — admirable engineering hygiene, but **falsification records are not a contribution** at IR venues. They support a paper, they are not a paper.

D. **ArguAna paradox** (cosine inversion) — interesting empirical observation, partially explained by ArguAna's counter-argument retrieval task structure, no novel mechanism proposed.

**A + B + C + D ≠ paper-level breakthrough**. CCF-A venues (TOIS, SIGIR full) want a single coherent novel mechanism with strong empirical support. This paper has four medium-strength observations.

### 2.3 "Training-free geometric reranking paradigm" is not actually a paradigm

The paper introduces this label in §1.1 but never defines it as a research program with falsifiable predictions, distinguishing axioms, or a clear successor question. Compare to ColBERT (late-interaction), DPR (bi-encoder + hard negatives), or ANCE (asynchronous mining) — each is a **genuine paradigm** with successor papers. "Training-free geometric reranking" is, on close reading, **a method name with paradigm-style framing**.

### 2.4 First on 1/5 corpora does not justify "new paradigm"

Even charitably scored: SCIDOCS-only first place against 4 baselines (ColBERTv2, BGE-M3, BGE-rerank, E5-Mistral). On a corpus that is widely considered the **easiest** in BEIR for graph methods (citation prediction has natural inter-document structure). This is a single data point; it does not generalize.

---

## Section 3 — Comparison Critique (vs SOTA missing baselines)

### 3.1 PLAID was **never run** — only deferred via Santhanam et al. citation

§subsec:reproducibility-notes admits PLAID indexing failed on ROCm 7.2 + colbert-ir 0.2.14 + transformers 5.7 due to API mismatches. The deferral chain — "ColBERTv2 quality ≈ PLAID quality per Santhanam Table 5" — is **citation deferral**, not experimental verification. Reviewer 3 of IPM-D-26-02154 explicitly listed PLAID as required. A re-submission to TOIS / SIGIR will get the same demand, with less patience.

### 3.2 RankGPT (GPT-4 listwise) was **never run**

The "Qwen3-8B-Q4 LLM-listwise" is RankGPT-style **prompting** but uses Qwen3-8B-Q4 as the base model — not GPT-4. Paper line 595 explicitly cites RankGPT \citep{sun2024rankgpt} as "RankGPT-style". A reviewer will say: **"You compared to your local LLM with a RankGPT prompt; you did not compare to RankGPT."** RankGPT's published NDCG@10 on these corpora (FiQA ~0.44–0.48 via GPT-4) **substantially exceeds Ours**.

### 3.3 RankLLM (Vicuna / Zephyr) was **never run**

Same family as 3.2. Cited but not implemented. Reviewer 3 listed it explicitly.

### 3.4 No comparison to Qwen3-Embedding-8B (specialized Qwen3 retrieval model)

The paper uses Qwen3-8B (general LLM) hidden states. Qwen3-Embedding-8B (the specialized retrieval-trained variant) **exists and is publicly released**. Paper does not compare. Reviewer will ask: "Why use the general LLM? The specialized variant from the same family would be the apples-to-apples comparison for 'training-free vs trained.'"

### 3.5 No comparison to NV-Retriever, BGE-EN-v2, GTE-large-v1.5, mxbai-embed-large

These are 2024–2026 SOTA dense retrieval models. The paper compares only to BGE-M3 (568M), BGE-large-v1.5 (335M), and ColBERTv2 (110M) — all 2022–2023 models. **The competitive landscape has moved.**

### 3.6 No comparison to FlexNeuart, Pyserini-trained-monoT5, RankZephyr

Training-free / efficient-rerank baselines that are widely cited. Absence is conspicuous.

### 3.7 E5-Mistral CUDA disclosure does not rule out E5-Mistral itself being the right baseline

§4.6 reports anomalously low E5-Mistral numbers across 3 dtypes / 2 platforms (0.129–0.140 vs published 0.39). The "honest attribution" admits the failure is in the local pipeline, not E5-Mistral. **A reviewer will say: "Then you have not compared to E5-Mistral; you have shown your evaluation harness mishandles it."** The paper cannot then claim "Ours beats E5-Mistral" because the comparison is invalid.

---

## Section 4 — Reproducibility Critique

### 4.1 v0 BGE-M3 / BGE-rerank vectors are permanently lost

§subsec:cross-model Vector-provenance note: ".gitignore excluding vector .jsonl files from version control and tarball backups". **The paper as previously published cited numbers based on these vectors**; anyone who tries to reproduce the v6–v11 numbers cannot do so. The current paper is now built on v2 vectors that **are** reproducible — but the audit footnote itself is a red flag for any reviewer who asks: "What other vectors did you lose?"

### 4.2 The 19.93% / 18.62% zero-vector bug is careless engineering

`encode_bge_m3_via_api.py`: MAX_CHARS=6000 truncation that exceeds the llama-server `--batch-size 512` token boundary, producing HTTP 500s **silently swallowed as np.zeros**. No retry. No chunk fallback. No final-pass verification. This is **first-year-grad-student-grade error**. The audit footnote is honest, but a reviewer will ask: "If your encoding pipeline silently produces 20% zero vectors on 2 of 5 corpora and you didn't notice, **what else in this codebase is broken that you also didn't notice?**" Trust is asymmetric — once broken, a single audit doesn't restore it.

### 4.3 5060 8GB GPU + nf4 4-bit quantization for the CUDA control is underwhelming

§4.6 cross-platform CUDA replication uses a consumer laptop GPU + 4-bit quantization. Reviewer will say: "Why didn't you spin up a $0.50/hr A100 on Colab / RunPod / Lambda for one full-precision E5-Mistral evaluation?" The answer is "we have no budget" — but at $0.50/hr × 6 hours × 5 corpora = **$15**, the budget excuse is vanishingly thin.

### 4.4 Different pooling for "BGE-large mean" vs "BGE-M3 audited rebuild CLS"

Paper §6 implementation note now admits: BGE-large uses mean pooling (paper line 714), BGE-M3 v2 audited uses CLS pooling. These are **different pooling conventions in the same paper for two models that are compared head-to-head in Table 8**. A reviewer will say: "Re-run BGE-large with CLS pooling for consistency." It is also internally suspicious that BGE-large mean-pooling NDCG@10 for NFCorpus = 0.2975 (≈ MTEB) while BGE-M3 CLS-pooling = 0.3113 (≈ MTEB) — the conventions happened to match each model's MTEB protocol, but the paper does not document this calibration step.

### 4.5 No `requirements.txt` / `environment.yml` / Docker image referenced for v2 audit

The audit footnote tells the story (re-encoded via llama-server with `bge-m3-f16.gguf`, per-document tokenize-then-chunk), but no pinned environment is referenced. A reviewer cannot reproduce.

### 4.6 Released-on-acceptance is unacceptable for top-tier IR

Abstract: "Code, indices, and falsification records will be released upon acceptance at https://github.com/Wangziqi0/Shape-CFD". TOIS, SIGIR, IPM all increasingly require **anonymous artifact submission at review time**. "Released on acceptance" is a yellow flag. A reviewer who suspects #4.2 / #4.4 cannot verify without code access.

---

## Section 5 — Writing Critique

### 5.1 Abstract substantially overclaims relative to body

Abstract (line 42): "surpassing BGE-large-en-v1.5 on two benchmarks (FiQA: 0.398 vs 0.367; SCIDOCS: 0.218 vs 0.162)" — but Table 8 (the post-audit baseline) shows BGE-rerank-v2 audited beating Ours on FiQA (0.4227 vs 0.3977). Abstract should be **either** updated to acknowledge BGE-rerank-v2 is the strongest competitor and Ours wins on 1 corpus only **or** the comparison set should be explicitly limited to "single-vector baselines" with that label. The current abstract is **misleading by omission**.

### 5.2 §1.1 contribution list is now self-contradictory

Item 1 (line 104): "Our method does *not* match purpose-trained cross-encoder rerankers (BGE-reranker-v2-m3 audited rebuild) on most datasets." This is honest and correct. But Item 4 (Graph Laplacian regularization, "consistent improvements on all six BEIR datasets") plus Item 5 (Model-agnostic post-processing, "Statistically significant gains across three embedding models") are **walked back to weak form by §6.5/§6.6**. The contribution list as written is a stronger claim than the paper actually supports.

### 5.3 §1 main claim list now reads as defensive

Lines 104–105 enumerate where the method **doesn't** work — on most datasets, in fact. This is honest, but it is **unprecedented in a top-tier IR contribution list to lead with negative results**. A reviewer will read: "Authors are pre-empting reject reasons in §1.1." This raises rather than lowers reject probability.

### 5.4 "Audited rebuild note" footnote in Table 8 is a 7-sentence forensic narrative

The note describes the bug, the fix, the resulting NDCG changes, the Pearson r=0.998 cross-validation, and asserts "Ours pipeline reproduces bit-exact". This **is** rigorous — but it is also **highly visible to any reviewer**. Reviewers do not enjoy reading forensic narratives about your own data quality. This footnote will be cited in negative reviews as evidence of "the authors had to publicly diagnose their own broken pipeline mid-submission."

### 5.5 §6.1 "Vector-provenance note" + §4.6 "Provenance audit note" + Table 8 "Audited rebuild note" + §6.5 "Unified-config control" = **four separate post-hoc hedge sections**

Each is individually justified. Cumulatively they signal **a paper that is being saved by patches**. A reviewer will form the impression: "This is v11 of a paper that has been repeatedly broken and re-stitched." Even if the current state is internally consistent, the **edit history is visible in the structure**.

### 5.6 Author affiliation: 16-year-old independent researcher

Paper acknowledgements explicitly state author age. **For blind review this is metadata that should never be discoverable.** main_blind.tex must be checked for author age leakage; if "Yifan Chen, age 16" appears in the blind version, this is a hard-violation of double-blind protocol at TOIS / SIGIR / IPM. **Reject without review** if leaked.

### 5.7 §significance section claim is a single 800-word run-on paragraph

Lines 393–404: one paragraph contains the bootstrap test, all 5 datasets' Cohen's d, the Quora exclusion, the 5 additional baselines, the PLAID deferral, and the post-audit ranking. This needs to be split into 3 paragraphs minimum for top-tier readability.

---

## Section 6 — Per-Venue Brutal Acceptance Probability

| Venue | Tier | Brutal estimate | Top 3 reject reasons | Required revisions |
|---|---|---|---|---|
| **TOIS** (CCF-A) | Q1 IR | **8–15%** (down from claimed 15–25%) | (a) Headline claim collapsed: 1/5 first place, abstract still claims "surpasses BGE-large on 2"; (b) PLAID + RankGPT + RankLLM all not run, only cited; (c) Audit footnote forensic narrative reads as data-quality-crisis paper | Real PLAID + real RankGPT (GPT-4) + real RankLLM + drop "training-free paradigm" framing; reframe as "geometric post-processing for selected weak-baseline corpora"; rewrite abstract to match Table 8 reality |
| **SIGIR full** (CCF-A) | top IR | **5–10%** | Same as TOIS + SIGIR demands single-coherent-novelty contribution which this paper lacks; competitive landscape has moved past 2023-era baselines | Same as TOIS + add 2024–2026 baselines (NV-Retriever, mxbai, GTE-v1.5) + identify 1 single coherent novelty (PQ-without-Q most defensible) and write paper around that |
| **IPM revision** (CCF-B, in progress) | Q1 | **30–45%** | (a) Reviewer 3's 3 specific baselines (PLAID / RankGPT / RankLLM) still unaddressed substantively; (b) Reviewer 3 W6 adaptive λ asked online learning, GBM is offline — substantive non-compliance; (c) §3.2 σ measurement now disagrees with v2 vectors | At minimum: real RankGPT + real RankLLM (1–2 days each) + online adaptive λ (0.5–1 day) + remeasure σ on v2 vectors and update §3.2 |
| **KBS** (CCF-C) | Q1 | **40–55%** | (a) Same baselines gap but KBS reviewers historically more lenient on baselines than IPM; (b) Audit narrative may actually help KBS (data-quality consciousness valued); (c) Innovation gap (4 small angles) likely tolerated at this tier | At minimum: rewrite abstract to match Table 8; refresh σ measurement; explicit "limitations" section listing the 5 not-run baselines |
| **AAAI / IJCAI** (CCF-A) | top AI | **5–10%** | Wrong venue framing — AAAI/IJCAI value method novelty over IR-specific baselines; the "training-free reranking" framing reads as IR-applied work without ML-novel contribution; PQ-without-Q is the only AAAI-tractable angle | Reframe as "subspace-decomposition cosine for breaking concentration of measure" with theoretical analysis of σ_sub/σ_full ratio under anisotropic spectra |

**No venue exceeds 55% honest probability.** Per the project's CLAUDE.md Rule 4 (acceptance probability must be ≥ 30% to declare "ready"), **IPM (~30–45%) and KBS (~40–55%) are the only venues that meet the floor**. TOIS (~8–15%), SIGIR (~5–10%), AAAI/IJCAI (~5–10%) are **not ready** by the project's own standard.

---

## Section 7 — Minimum Revisions Required (priority order, brutal reviewer view)

### Tier 1 — must fix before any submission (breaks paper otherwise)

1. **Re-measure σ_sub/σ_full on v2 BGE-M3 vectors and update §3.2 line 196** to reflect 1.6–1.8× ratio. Either drop the "cross-model verification of concentration breaking" claim or accept the weakened ratio and rewrite the surrounding paragraph. **(0.5 day)**
2. **Rewrite abstract** to remove the "surpasses BGE-large on 2 benchmarks" framing in favor of an honest summary that includes BGE-rerank-v2 audited as the strongest competitor. **(0.5 day)**
3. **Verify main_blind.tex contains zero author age / institutional / GitHub URL leakage**. Remove acknowledgements paragraph entirely from blind version. **(1 hour)**
4. **Refresh Table 9 to use post-audit BGE-M3 numbers** (currently 0.2893 — pre-audit; should be 0.3113). **(1 hour)**

### Tier 2 — must fix to meet IPM Reviewer 3 standard

5. **Run RankGPT (GPT-4 listwise) on 5 corpora**: ~$100 OpenAI API + 1 day. (Audit trail W1.2)
6. **Run RankLLM (Vicuna-7B or Zephyr-7B listwise)** on 9070XT: 1–2 days.
7. **Implement online adaptive λ** (e.g., Thompson sampling with cosine-baseline + query-length features): 0.5–1 day.
8. **Run E5-Mistral on a rented A100** (Colab Pro / Lambda): 4–6 hours, $5–15.

### Tier 3 — must fix to be SIGIR/TOIS competitive

9. **Add 2024–2026 baselines**: NV-Retriever, GTE-large-v1.5, mxbai-embed-large-v1, Qwen3-Embedding-8B (specialized).
10. **Reframe paper around 1 coherent novelty**: PQ-without-Q is the strongest candidate. Promote to single core contribution; demote graph Laplacian to ablation.
11. **Resolve PLAID** via either (a) separate transformers<5 venv, (b) fork colbert-ir, or (c) explicit comment that PLAID quality is benchmarked elsewhere (Santhanam Table 5) and we report ColBERTv2 quality numbers.

### Tier 4 — fix for cosmetic / hygiene

12. Consolidate the 4 audit/provenance footnotes into a single §1.5 "Data Quality Disclosure" paragraph that reads as confident transparency rather than defensive patches.
13. Split §significance run-on paragraph into 3 paragraphs.
14. Use `pytrec_eval` or `trec_eval` for all NDCG@10 computations and document the protocol in a §7.1 "Evaluation Protocol" subsection.

---

## Section 8 — Overall Verdict (brutal summary)

**This paper is currently a structurally honest but commercially unsellable artifact.** The audit-grade rebuild has done the right thing scientifically — exposing a 20% zero-vector bug that would have been a post-publication retraction risk — but the cost is that **the central headline claim of every previous version is now empirically false**. The paper now reads as a forensic case study of its own data quality crisis, with four separate "audit notes" stitched into Table 8 and §6, each of which is individually justified and cumulatively damning.

The four claimed contributions (training-free paradigm, PQ-Chamfer, token-level point cloud, graph Laplacian) are individually decent but jointly do not constitute a CCF-A-grade single coherent novelty. The 21 falsifications are admirable but not a contribution. The "Ours" pipeline first-places on 1/5 BEIR corpora (SCIDOCS) and ties on 1 (NFCorpus) — this is **not** the basis for a "new paradigm" claim; it is the basis for a **competent, honest mid-tier IR application paper**. KBS / IPM revision are the appropriate targets at the current state. TOIS / SIGIR / AAAI are not realistic without (a) running PLAID + RankGPT + RankLLM substantively, (b) reframing around a single core novelty (PQ-without-Q), and (c) modernizing the baseline set to 2024–2026 SOTA.

The author has a 16-year-old researcher's hunger and an industrial researcher's hygiene. What is missing is **one paper-level breakthrough that all the engineering serves**. Until that breakthrough is identified and isolated, the paper will keep getting rejected at top tier and accepted at mid tier — and that is the honest reviewer view, with zero softening for biographical sympathy.

---

**Reviewer signature**: adversarial blind reviewer agent simulating IPM/TOIS/KBS/SIGIR standards, dispatched 2026-05-02.
**Confidence**: high on findings 1.1 / 1.2 / 1.6 / 2.1 / 2.2 / 3.1 / 4.1 / 4.2 / 5.1 / 5.6 (these are direct evidence from paper + audit trail). Medium on 1.3 / 1.4 (require external benchmark verification). Low on 5.5 (subjective rhetorical reading).
**Disclosure**: This is a brutal review. It is intended to be the hardest reviewer the author will ever face. If the paper survives this, it can survive submission.
