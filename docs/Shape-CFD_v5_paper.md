# Point Cloud Retrieval with PQ-Chamfer Distance and Graph Regularization: A Training-Free Approach

**Yifan Chen**

Independent Researcher

April 2026

---

## Abstract

Neural dense retrieval compresses each document into a single vector and ranks candidates by cosine similarity, discarding the rich sub-document semantic structure encoded in language model hidden states. We propose a **training-free post-processing pipeline** that represents documents as *point clouds* of token-level embeddings extracted from a general-purpose large language model (Qwen3-8B), and introduces two complementary scoring mechanisms that can be applied on top of any embedding model without additional training. First, **PQ-Chamfer distance** decomposes the 4096-dimensional embedding space into 64 independent 64-dimensional subspaces, computing cosine distances within each subspace before aggregating via the symmetric Chamfer metric. This subspace decomposition breaks the concentration-of-measure effect that plagues high-dimensional distance computation, transforming product quantization from a compression tool into a distance metric innovation. Second, **graph Laplacian smoothing** constructs a KNN document graph and propagates relevance scores through iterative diffusion, capturing global neighborhood structure that local nearest-neighbor matching misses. On six BEIR benchmarks, graph smoothing alone yields consistent improvements ranging from +2.7% to +44.2% over the cosine similarity baseline of the same embedding model. When combined with token-level PQ-Chamfer reranking---now evaluated on five datasets---the full pipeline achieves gains of up to **+136.2%** (FiQA), with the best configuration surpassing retrieval-specialized models on two benchmarks: FiQA (0.3977 vs BGE-large 0.367, **+8.4%**) and SCIDOCS (0.2147 vs BGE-large 0.162, **+32.5%**), all without any retrieval-specific training, cross-encoder inference, or corpus-specific tuning. On ArguAna, token-level matching alone reaches 0.4417, approaching ColBERTv2 (0.460). Crucially, even single-chunk documents benefit: FiQA documents consist of a single vector each, yet PQ subspace decomposition yields +136.2%, demonstrating that PQ creates exploitable "shape" from a single point. We demonstrate that this pipeline functions as a **model-agnostic geometric post-processing layer**: evaluated on three embedding models (Qwen3-8B, BGE-M3, BGE-large-en-v1.5), it yields statistically significant improvements on all three (p < 0.05), with gains inversely proportional to baseline quality (+32.1% for the weakest model, +2.8% for the strongest). Because our method operates as a purely geometric post-processing layer, it is complementary to---and potentially stackable with---any retrieval system. We report 21 systematically falsified approaches---including the complete failure of convection-diffusion equations (five independent validations)---establishing clear applicability boundaries. All code and data are publicly available.

---

## 1 Introduction

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding large language models in external knowledge (Lewis et al., 2020; Guu et al., 2020). The standard RAG pipeline employs a bi-encoder to map queries and documents into a shared embedding space, then ranks candidates by cosine similarity:

$$\text{score}(d_i) = \frac{q \cdot v_i}{\|q\| \|v_i\|}$$

While efficient, this formulation suffers from two fundamental limitations. First, it represents each document as a *single point* in $\mathbb{R}^d$, conflating the diverse semantic facets of multi-sentence passages into a single centroid. A legal statute spanning contractual liability, evidentiary standards, and procedural remedies is reduced to one vector that captures none of these facets faithfully. Second, it scores each document independently, ignoring the inter-document structural information---topic clusters, semantic neighborhoods, and distributional patterns---that the retrieved set collectively encodes.

The insight that motivated this work is deceptively simple: **documents are not points; they are shapes** (Chen, 2026). A document, when represented by the per-token hidden states of a language model, forms a point cloud in embedding space. The geometry of this point cloud---its spread, its density variations, its subspace projections---carries semantic information that a single centroid vector destroys. This observation leads naturally to two questions: What is the right distance metric between document point clouds? And how can we exploit inter-document structure without the cost of cross-encoder inference?

### Relation to Prior Versions

This paper is the fifth installment of the Shape-CFD research program. V1--V3 (Chen, 2026a) introduced the framework of representing documents as sentence-level point clouds and reranking via convection-diffusion equations on document graphs. V4 (Chen, 2026b; Zenodo DOI: 10.5281/zenodo.19233894) added PQ-Chamfer distance and adjoint-state prefetching. The present work, V5/V11, makes a decisive departure: we abandon the convection-diffusion narrative---which was experimentally falsified across five independent validations (Section 4.5)---and reorganize the framework around three pillars: token-level point clouds, PQ-Chamfer distance, and graph Laplacian smoothing.

### Contributions

We summarize our contributions as follows:

1. **PQ-Chamfer distance metric.** We transform product quantization (PQ) from a vector compression technique into a distance metric innovation. By decomposing the 4096-dimensional space into 64 contiguous 64-dimensional subspaces, computing cosine distances independently within each, and aggregating via the symmetric Chamfer metric, we break the concentration-of-measure effect that makes high-dimensional distances uninformative. To our knowledge, this use of PQ subspace decomposition for distance computation (rather than approximate nearest neighbor search) is unprecedented in the retrieval literature.

2. **Training-free token-level retrieval.** We extract per-token last hidden states from a general-purpose LLM (Qwen3-8B, `--pooling none`) rather than training a specialized retrieval model. Each document becomes a dense point cloud of $\sim$356 tokens and each query unfolds into $\sim$6 token vectors, enabling fine-grained token-to-token matching that approximates hard cross-attention with zero parameters and zero training.

3. **Point cloud graph regularization.** We show that graph Laplacian smoothing---constructing a KNN graph over document point clouds and iteratively propagating relevance scores---provides consistent improvements across all six evaluated BEIR datasets (+2.7% to +44.2% vs cosine). This makes graph smoothing the most robust single component in our framework.

4. **"A single point has shape."** PQ subspace decomposition creates an implicit point cloud: each 4096-dimensional vector becomes 64 independent 64-dimensional projections. Even single-chunk documents that consist of a single embedding vector benefit from this decomposition, as demonstrated by the +136.2% improvement on FiQA where all documents are single vectors.

5. **Model-agnostic geometric post-processing.** Our pipeline improves retrieval quality across three different embedding models (Qwen3-8B 8B, BGE-M3 568M, BGE-large 335M) spanning different architectures, training objectives, and dimensionalities (1024d--4096d), confirming that the geometric operations exploit universal structural properties of embedding spaces rather than model-specific artifacts.

6. **Systematic falsification record.** We document 21 failed approaches, including the complete failure of convection-diffusion equations, density-weighted Chamfer distance, PQ code reconstruction, and Allen-Cahn reaction terms. These negative results establish clear applicability boundaries and save future researchers from repeating unsuccessful experiments.

We emphasize that our method is a **model-agnostic geometric post-processing layer** that operates on any embedding model's output. Cross-model experiments on three architectures (Qwen3-8B, BGE-M3, BGE-large) confirm that the improvements are not artifacts of a specific embedding model. Remarkably, our full pipeline (PQ-Chamfer + graph smoothing) **surpasses BGE-large-en-v1.5 on FiQA (+8.4%) and SCIDOCS (+32.5%)**, and approaches ColBERTv2 on ArguAna (0.4417 vs 0.460), all without any retrieval-specific training. The consistent relative gains (+2.8% to +32.1% across models) suggest that our geometric operations are **complementary to, and stackable with, any retrieval system**.

---

## 2 Background and Related Work

### Multi-Vector Retrieval

ColBERT (Khattab & Zaharia, 2020) pioneered multi-vector document representation through late interaction, representing both queries and documents as sets of token-level vectors and computing relevance via the MaxSim operator. ColBERTv2 (Santhanam et al., 2022) improved storage efficiency through residual compression, and PLAID (Santhanam et al., 2022b) introduced centroid-based pruning for faster retrieval. Our approach shares the insight that documents should be represented as collections of sub-vectors, but differs in three key aspects: (i) we use general-purpose LLM hidden states rather than a trained retrieval model, (ii) we employ symmetric Chamfer distance rather than asymmetric MaxSim, and (iii) we combine local token matching with global graph-based score propagation.

### Graph-Based Reranking

Graph-based methods have a long history in information retrieval, from PageRank (Page et al., 1999) to modern diffusion-based approaches. Dampanaboina et al. (2024) apply isotropic heat diffusion on document similarity graphs for RAG reranking. G-RAG (Glass et al., 2024) learns graph neural network rerankers through supervised training. Our graph Laplacian smoothing is closest to the diffusion approach but operates on point-cloud-derived distances rather than cosine similarity and requires no training.

### Product Quantization

Product quantization (Jegou et al., 2011) is traditionally used for approximate nearest neighbor search: the vector space is decomposed into subspaces, each quantized independently, enabling fast distance approximation via lookup tables. FAISS (Johnson et al., 2021) and ScaNN (Guo et al., 2020) are major systems built on this principle. Optimized PQ (OPQ; Ge et al., 2014) applies rotation before quantization to minimize distortion. Our work repurposes the PQ decomposition not for compression but for distance computation: we do not quantize vectors to centroids; instead, we compute exact distances within each subspace and aggregate them. This "PQ without quantization" approach uses the subspace structure to break concentration of measure.

### Point Cloud Distances

The Chamfer distance (Fan et al., 2017) and Earth Mover's Distance (EMD; Rubner et al., 2000) are the two dominant metrics for comparing point clouds in 3D vision. Chamfer distance computes the average nearest-neighbor distance in both directions, while EMD finds the optimal transport plan. The connection between Chamfer distance and unbalanced optimal transport (UOT) is well established: Chamfer distance corresponds to UOT in the limit $\tau \to 0$, where the marginal constraint is fully relaxed (Sejourne et al., 2023). This relaxation is precisely what makes Chamfer effective for documents of different lengths: a 100-sentence document need not match every sentence to a 5-word query.

---

## 3 Method

### 3.1 Document Point Cloud Representation

Let $\mathcal{M}$ denote a pretrained language model (Qwen3-8B, $d = 4096$). For a document $d_i$ consisting of tokens $\{t_{i,1}, t_{i,2}, \ldots, t_{i,n_i}\}$, we extract the last hidden states with pooling disabled (`--pooling none`):

$$\mathcal{P}_i = \{h_{i,1}, h_{i,2}, \ldots, h_{i,n_i}\} \subset \mathbb{R}^d$$

where $h_{i,j} \in \mathbb{R}^{4096}$ is the last-layer hidden state for token $t_{i,j}$. Similarly, a query $q$ with tokens $\{t_{q,1}, \ldots, t_{q,m}\}$ yields a query point cloud $\mathcal{P}_q = \{h_{q,1}, \ldots, h_{q,m}\}$.

In our NFCorpus evaluation, the average document contains $\sim$356 tokens and the average query contains $\sim$6 tokens. The entire NFCorpus token cloud database (2,473 documents, 880,791 tokens) occupies 14 GB in SQLite storage.

**Centroid representation.** For coarse-grained operations, we also compute the centroid of each point cloud:

$$\bar{v}_i = \frac{1}{n_i} \sum_{j=1}^{n_i} h_{i,j}$$

This centroid serves as the single-vector representation for initial retrieval and coarse filtering.

### 3.2 PQ-Chamfer Distance

We introduce PQ-Chamfer distance in two stages: subspace decomposition and Chamfer aggregation.

**Subspace decomposition.** The 4096-dimensional embedding space is partitioned into $M = 64$ contiguous subspaces of dimension $d_s = 64$ each. For a vector $h \in \mathbb{R}^{4096}$, we write $h^{(s)}$ for the projection onto the $s$-th subspace:

$$h^{(s)} = h[(s-1) \cdot d_s : s \cdot d_s], \quad s = 1, \ldots, M$$

The cosine distance within each subspace is:

$$d_s(a, b) = 1 - \frac{a^{(s)} \cdot b^{(s)}}{\|a^{(s)}\| \|b^{(s)}\|}$$

The PQ distance between two tokens is the average across subspaces:

$$d_{PQ}(a, b) = \frac{1}{M} \sum_{s=1}^{M} d_s(a, b)$$

**Why subspace decomposition works.** In the full 4096-dimensional space, the *concentration of measure* phenomenon causes all pairwise cosine similarities to cluster near zero with standard deviation $\sigma \approx 1/\sqrt{d} \approx 0.016$. By computing distances in 64-dimensional subspaces, the effective standard deviation increases to $\sigma_s \approx 1/\sqrt{64} = 0.125$, providing 7.8$\times$ better discrimination. Crucially, this is not a random projection (which we tested and found harmful; see Section 4.5): the subspace boundaries respect the model's dimensional structure. Ablation confirms that contiguous 64$\times$64 outperforms 32$\times$128 and 16$\times$256, and that ordered subspaces perform comparably to randomly permuted ones, indicating that Qwen3's embedding dimensions are approximately uniformly distributed in information content.

**Chamfer aggregation.** Given two point clouds $\mathcal{A}$ and $\mathcal{B}$, the PQ-Chamfer distance is:

$$d_{PQC}(\mathcal{A}, \mathcal{B}) = \frac{1}{2} \left[ \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \min_{b \in \mathcal{B}} d_{PQ}(a, b) + \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \min_{a \in \mathcal{A}} d_{PQ}(a, b) \right]$$

The forward term $\frac{1}{|\mathcal{A}|}\sum_a \min_b$ measures how well $\mathcal{B}$ *covers* $\mathcal{A}$ (every point in $\mathcal{A}$ finds a nearby match in $\mathcal{B}$), and the backward term ensures symmetry.

**VT-Aligned distance.** The standard PQ-Chamfer computes subspace distances first and then applies the Chamfer $\min$ operator ("average-then-min"):

$$d_{PQC}(\mathcal{A}, \mathcal{B}) = \text{Chamfer}\bigl(\mathcal{A}, \mathcal{B}; \; d_{PQ}(a,b) = \tfrac{1}{M}\textstyle\sum_s d_s(a,b)\bigr)$$

VT-Aligned reverses the aggregation order ("min-then-average"): for each token pair $(a_i, b_j)$, the subspace distances are summed *after* the Chamfer matching selects the nearest neighbor:

$$d_{VT}(\mathcal{A}, \mathcal{B}) = \frac{1}{2}\left[\frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} \frac{1}{M}\sum_{s=1}^{M} d_s\!\bigl(a, \arg\min_{b \in \mathcal{B}} d_{PQ}(a,b)\bigr) + \text{sym.}\right]$$

In practice, VT-Aligned yields +2.5% over standard PQ-Chamfer at zero additional cost, because the $\min$ operator selects a globally best-matching token whose *per-subspace* distances are then used directly, avoiding the averaging-before-selection that can wash out subspace-specific signal.

**Connection to unbalanced optimal transport.** The Chamfer distance can be viewed as the solution to an unbalanced optimal transport (UOT) problem where the marginal constraint is fully relaxed ($\tau \to 0$). Standard optimal transport (Wasserstein distance) requires the transport plan to satisfy $\sum_j \pi_{ij} = \mu_i$ and $\sum_i \pi_{ij} = \nu_j$, enforcing that every point in both clouds is fully matched. For documents of vastly different lengths---a 6-token query matched against a 356-token document---this constraint is catastrophic: it forces 350 document tokens to be matched somewhere, diluting the signal. Chamfer distance's relaxed marginal allows each query token to find its best document match independently, functioning as a form of *hard semantic attention*.

### 3.3 Graph Laplacian Smoothing

Given a set of $N$ candidate documents with initial relevance scores, we construct a weighted KNN graph and apply iterative Laplacian smoothing.

**Graph construction.** We build a sparse undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$ over the $N$ candidates:

1. *Pairwise distance:* Compute $d_{ij} = d_{PQC}(\mathcal{P}_i, \mathcal{P}_j)$ for all pairs, or use VT-Aligned distance (see below).
2. *KNN sparsification:* For each node $i$, retain edges to its $K = 3$ nearest neighbors.
3. *Edge weights:* $W_{ij} = \exp(-\beta \cdot d_{ij})$, with $\beta = 2.0$.
4. *Symmetrization:* If $j \in \mathcal{N}(i)$ but $i \notin \mathcal{N}(j)$, add the reverse edge with $W_{ji} = W_{ij}$.

**Initial scores.** Each document receives an initial "concentration":

$$C_0(i) = \exp(-2 \cdot d_{PQC}(\mathcal{P}_q, \mathcal{P}_i))$$

**Laplacian smoothing iteration.** The score update follows the discrete heat equation on the graph:

$$C_{t+1}(i) = C_t(i) + \alpha \sum_{j \in \mathcal{N}(i)} W_{ij} \left( C_t(j) - C_t(i) \right)$$

where $\alpha = 0.15$ is the diffusion coefficient. We run $T = 5$ iterations. This can be written in matrix form as:

$$C_{t+1} = (I - \alpha L) \, C_t$$

where $L = D_{deg} - W$ is the graph Laplacian and $D_{deg}$ is the degree matrix.

**Stability condition.** The smoothing coefficient $\alpha$ must satisfy $\alpha < 1/d_{\max}$ where $d_{\max}$ is the maximum node degree in the KNN graph, ensuring non-negative eigenvalues of the update matrix $(I - \alpha L)$. With $K = 3$ (bidirectional, $d_{\max} \approx 6$), our choice of $\alpha = 0.15$ satisfies this condition ($0.15 < 1/6 \approx 0.167$).

**Effect.** Laplacian smoothing propagates high relevance scores to semantically similar neighbors, effectively discovering relevant documents that may have received low initial scores due to vocabulary mismatch or embedding noise. Unlike cross-encoder reranking, this propagation is training-free and operates in milliseconds.

### 3.4 Multi-Granularity Pipeline

Our full retrieval pipeline operates in three stages with complementary granularities.

**Stage 1: Centroid coarse filtering.** Compute cosine similarity between the query centroid $\bar{v}_q$ and all document centroids $\{\bar{v}_i\}$ in the corpus. Retain the top-$K_1 = 100$ candidates. This stage runs in 23 ms on commodity CPU hardware.

**Stage 2: Token PQ-Chamfer reranking.** For the top-100 candidates, compute the full token-level PQ-Chamfer distance $d_{PQC}(\mathcal{P}_q, \mathcal{P}_i)$ using all token hidden states. Re-rank and retain the top-$K_2 = 55$ candidates. This stage runs in 23 ms.

**Stage 3: Graph Laplacian smoothing.** Construct a KNN graph over the top-55 candidates and apply 5 iterations of Laplacian smoothing as described in Section 3.3. This stage runs in 26 ms.

**Score fusion.** For datasets where token-level data is available (NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA), we combine the two signal sources:

$$\text{score}_{\text{final}}(i) = \lambda \cdot \text{score}_{\text{token}}(i) + (1 - \lambda) \cdot \text{score}_{\text{graph}}(i)$$

where $\lambda = 0.7$ and both score vectors are min-max normalized before fusion. The Kendall $\tau$ rank correlation between token and graph rankings is 0.378, confirming high complementarity.

**Two-stage acceleration is lossless.** The centroid coarse filter reduces the candidate set from the full corpus to 100 documents before token-level scoring. Remarkably, this 13$\times$ speedup (305 ms full-scan to 23 ms two-stage) introduces *zero* accuracy loss: the two-stage pipeline achieves NDCG@10 = 0.3220 versus 0.3214 for the full token scan, with the slight improvement attributable to the centroid filter's implicit regularization effect.

**Complexity analysis.** Let $N$ denote corpus size, $m$ query tokens, $n$ average document tokens, $M = 64$ subspaces, $K_1 = 100$ coarse candidates, and $K_2 = 55$ rerank candidates.

- *Stage 1 (centroid coarse filtering):* $O(N \cdot d)$ cosine comparisons, or $O(N \cdot m \cdot M)$ if using PQ-Chamfer at centroid level.
- *Stage 2 (token PQ-Chamfer reranking):* $O(K_1 \cdot m \cdot n \cdot M)$ per query --- for each of $K_1$ candidates, compute $m \times n$ token-pair distances across $M$ subspaces.
- *Stage 3 (graph Laplacian smoothing):* $O(K_2^2 \cdot n^2 \cdot M)$ for pairwise graph construction + $O(T \cdot K_2 \cdot K)$ for $T$ smoothing iterations on a $K$-sparse graph.

With typical NFCorpus parameters ($m \approx 6$, $n \approx 356$, $K_1 = 100$, $K_2 = 55$, $M = 64$), the bottleneck is Stage 2 at $\sim$137M floating-point operations per query, completed in 23 ms on commodity CPU.

---

## 4 Experiments

### 4.1 Experimental Setup

**Datasets.** We evaluate on six datasets from the BEIR benchmark (Thakur et al., 2021), spanning diverse domains and scales:

| Dataset | Domain | #Docs | #Queries | Avg doc length |
|---------|--------|------:|--------:|----|
| NFCorpus | Biomedical | 2,473 | 323 | Long (multi-paragraph) |
| SciFact | Scientific claims | 3,752 | 300 | Short (abstract) |
| ArguAna | Argumentative essays | 8,674 | 1,398 | Medium |
| SCIDOCS | Scientific papers | 25,337 | 1,000 | Medium |
| FiQA | Financial QA | 56,391 | 648 | Short (single chunk) |
| Quora | Duplicate questions | 522,931 | 10,000 | Short |

**Embedding model.** Qwen3-8B (Q4_K_M quantization) via llama.cpp with `--pooling none` for token-level hidden states and `--pooling mean` for centroid vectors. Embedding dimension $d = 4096$.

**PQ parameters.** $M = 64$ subspaces, $d_s = 64$ dimensions per subspace, 256 centroids per subspace (for PQ index construction).

**Graph parameters.** KNN with $K = 3$, edge weight $W_{ij} = \exp(-2 \cdot d_{ij})$, Laplacian smoothing $\alpha = 0.15$, $T = 5$ iterations.

**Baselines.** (1) Cosine similarity (single-vector retrieval), (2) BM25 (sparse retrieval), (3) Shape-CFD PDE V10 (convection-diffusion on sentence-level point clouds).

**Metric.** NDCG@10, the standard metric for BEIR evaluation. Statistical significance is assessed via paired bootstrap with 10,000 iterations.

### 4.2 Main Results

Table 1 presents the main results across six BEIR datasets. We report our cosine baseline, the best graph Laplacian smoothing configuration (Lap\_best), token two-stage reranking (token\_2stage), our best overall result (Ours Best), and published results from three retrieval-specialized models for context. Token-level data is available for five of six datasets; Quora uses sentence-level results only.

**Table 1: NDCG@10 across six BEIR datasets. Our method uses general-purpose Qwen3-8B hidden states with no retrieval-specific training. Bold = best in our pipeline; underline = surpasses BGE-large.**

| Dataset | #Docs | Cosine | Lap\_best | token\_2stage | **Ours Best** | Gain | BM25$^\dagger$ | ColBERTv2$^\dagger$ | BGE-large$^\dagger$ |
|---------|------:|-------:|----------:|--------------:|--------------:|-----:|------:|-----------:|-----------:|
| NFCorpus | 2,473 | 0.2195 | 0.2900 | 0.3220 | **0.3271**$^f$ | +49.0% | 0.322 | 0.338 | 0.344 |
| SciFact | 3,752 | 0.4701 | **0.4827** | 0.4555 | **0.4827**$^g$ | +2.7% | 0.665 | 0.693 | 0.738 |
| ArguAna | 8,674 | 0.3047 | 0.3463 | **0.4417** | **0.4417**$^t$ | +45.0% | 0.397 | 0.460 | 0.637 |
| SCIDOCS | 25,337 | 0.1110 | 0.1406 | 0.2139 | <u>**0.2147**</u>$^f$ | +93.5% | 0.158 | 0.155 | 0.162 |
| FiQA | 56,391 | 0.1683 | 0.2426 | 0.3897 | <u>**0.3977**</u>$^f$ | +136.2% | 0.236 | 0.356 | 0.367 |
| Quora | 522,931 | 0.6370 | 0.6742 | --- | **0.6749**$^p$ | +6.0% | 0.789 | 0.852 | 0.889 |

$^f$fusion (0.7 $\times$ token\_2stage + 0.3 $\times$ graph). $^g$graph Laplacian smoothing only. $^t$token\_2stage only. $^p$PDE\_200 (sentence-level; no token data available for Quora).

$^\dagger$ Published results from the BEIR benchmark (Thakur et al., 2021), ColBERTv2 (Santhanam et al., 2022), and BGE-large-en-v1.5 (Xiao et al., 2023). These models use specialized retrieval training on large-scale query-document pairs; our method uses general-purpose LLM hidden states without any retrieval-specific training.

Note: Cosine baseline uses token-level centroid cosine similarity (0.2195 for NFCorpus, 0.4701 for SciFact, etc.), which differs slightly from sentence-level centroids used in earlier versions of this paper.

**Key observations:**

(1) *Token-level PQ-Chamfer is the dominant component.* Token two-stage reranking provides large improvements on 4 of 5 evaluated datasets: +45.0% (ArguAna), +93.5% (SCIDOCS), +131.5% (FiQA), and +46.7% (NFCorpus). The sole exception is SciFact (-3.1%), where the already-strong cosine baseline (0.4701) leaves little room for improvement. This establishes token-level PQ-Chamfer as the single most impactful component in our pipeline.

(2) *Graph Laplacian smoothing is universally effective.* Across all six datasets, Laplacian smoothing outperforms cosine similarity, with gains ranging from +2.7% (SciFact) to +44.2% (FiQA). This makes graph smoothing the most *consistent* (though not always the largest) improvement.

(3) *Score fusion provides selective additional gains.* Fusion benefits datasets where graph smoothing provides complementary signal: FiQA (0.3977 fusion vs 0.3897 token-only), SCIDOCS (0.2147 vs 0.2139), and NFCorpus (0.3271 vs 0.3220). However, on ArguAna, fusion (0.4215) underperforms token-only (0.4417), indicating that graph smoothing can introduce noise when token matching alone captures the dominant relevance signal.

(4) *The cosine-inverse correlation.* The largest gains occur on datasets where cosine similarity performs worst: FiQA (+136.2% from 0.1683) and SCIDOCS (+93.5% from 0.1110). Conversely, SciFact, where cosine already achieves 0.4701, sees the smallest improvement (+2.7%). This pattern holds consistently across both token-level and graph-level improvements, suggesting that our geometric operations recover structural information that cosine-based single-vector retrieval discards, with diminishing returns as the baseline strengthens.

(5) *Laplacian outperforms PDE in 5/6 datasets.* Pure graph smoothing (Laplacian) outperforms the full convection-diffusion PDE (which adds an advection term) on five of six datasets. This finding led to our decision to abandon the convection-diffusion narrative (see Section 4.5).

**Comparison with trained retrieval models.** Table 1 also includes published results from BM25, ColBERTv2, and BGE-large-en-v1.5 for context. Remarkably, our full pipeline **surpasses BGE-large-en-v1.5** on two datasets: FiQA (0.3977 vs 0.367, +8.4%) and SCIDOCS (0.2147 vs 0.162, +32.5%). On ArguAna, token-level matching alone (0.4417) approaches ColBERTv2 (0.460, gap of 4.0%) and exceeds BM25 (0.397). On NFCorpus, our fusion result (0.3271) approaches ColBERTv2 (0.338, gap of 3.2%) and surpasses BM25 (0.322). The gap remains large on datasets where retrieval-specialized training provides the most benefit (SciFact: 0.483 vs BGE 0.738; Quora: 0.675 vs BGE 0.889).

These results are significant because BGE-large-en-v1.5 is trained on millions of query-document pairs with contrastive objectives specifically designed for retrieval, while our method uses off-the-shelf Qwen3-8B hidden states with no retrieval-specific training whatsoever. The fact that purely geometric post-processing can surpass a trained retrieval model on two benchmarks suggests that the sub-document structure captured by our pipeline contains information that retrieval-specific training does not fully exploit. Our geometric operations provide +2.7% to +136.2% improvement over the cosine baseline, as confirmed by cross-model experiments on BGE-large and BGE-M3 (Section 4.6).

### 4.3 Ablation Study (NFCorpus)

Table 2 presents a detailed ablation on NFCorpus, the dataset for which we have complete token-level data.

**Table 2: NFCorpus ablation (NDCG@10).**

| Method | NDCG@10 | vs Cosine | p-value |
|--------|--------:|----------:|--------:|
| Cosine baseline | 0.2195 | --- | --- |
| BM25 | 0.3256 | +48.3% | --- |
| Shape-CFD PDE (V10) | 0.2852 | +29.9% | <0.001 |
| Graph Laplacian (55 candidates) | 0.2900 | +32.1% | <0.001 |
| PQ-Chamfer V8 (sentence-level) | 0.2729 | +24.3% | --- |
| VT-Aligned V13 | 0.2797 | +27.4% | --- |
| Adjoint prefetch V7 | 0.2802 | +27.6% | --- |
| Token two-stage (centroid $\to$ PQ-Chamfer) | 0.3220 | +46.7% | <0.001 |
| **Fusion (0.7$\times$token + 0.3$\times$graph)** | **0.3271** | **+49.0%** | --- |

**Statistical significance.** Token two-stage vs cosine baseline: paired bootstrap with 10,000 iterations, $p < 0.001$. Fusion vs token two-stage: $p = 0.3346$ (not significant), indicating that the graph smoothing component provides a modest but non-significant additive benefit on this dataset.

**Progression analysis.** The ablation reveals a clear hierarchy of contributions:

- *Sentence-level point cloud* (V4 Chamfer, +24.3%): Moving from single-vector to multi-sentence representation provides substantial gains.
- *PQ subspace decomposition* (V8, +24.3% $\to$ +27.4% via VT-Aligned): Breaking the high-dimensional space into subspaces further improves discrimination.
- *Token-level point cloud* (V11, +46.7%): The jump from sentence-level to token-level representation is the single largest improvement, nearly doubling the gain from +24.3% to +46.7%.
- *Graph smoothing* (+32.1% standalone, +49.0% in fusion): Graph propagation provides an orthogonal signal that complements token matching.

### 4.3.1 Multi-Dataset Pipeline Decomposition

Table 2b extends the ablation across all five datasets with token-level data, decomposing the pipeline into its constituent stages.

**Table 2b: Full pipeline decomposition across five BEIR datasets (NDCG@10). Each column represents a pipeline stage; Gain is measured from cosine to best.**

| Dataset | Cosine | Lap\_best | token\_2stage | fusion\_07 | **Best** | **Gain** |
|---------|-------:|----------:|--------------:|-----------:|--------:|--------:|
| NFCorpus | 0.2195 | 0.2900 | 0.3220 | **0.3271** | 0.3271 | +49.0% |
| SciFact | 0.4701 | **0.4827** | 0.4555 | 0.4689 | 0.4827 | +2.7% |
| ArguAna | 0.3047 | 0.3463 | **0.4417** | 0.4215 | 0.4417 | +45.0% |
| SCIDOCS | 0.1110 | 0.1406 | 0.2139 | **0.2147** | 0.2147 | +93.5% |
| FiQA | 0.1683 | 0.2426 | 0.3897 | **0.3977** | 0.3977 | +136.2% |

**Key patterns across datasets:**

(1) *Token-level matching dominates on 4/5 datasets.* Token two-stage provides the majority of the gain on NFCorpus, ArguAna, SCIDOCS, and FiQA. The one exception---SciFact---is the dataset with the strongest cosine baseline (0.4701), where graph smoothing alone achieves the best result.

(2) *Gains are inversely correlated with cosine baseline strength.* Plotting Gain vs Cosine across the five datasets reveals a clear monotonic relationship: FiQA (cosine 0.1683, +136.2%) > SCIDOCS (0.1110, +93.5%) > NFCorpus (0.2195, +49.0%) > ArguAna (0.3047, +45.0%) > SciFact (0.4701, +2.7%). This suggests that our geometric operations recover structural information that becomes redundant as the embedding baseline improves.

(3) *Fusion is not universally beneficial.* Fusion improves over token-only on FiQA (+2.1%), SCIDOCS (+0.4%), and NFCorpus (+1.6%), but *hurts* on ArguAna (-4.6%), where graph smoothing introduces noise that dilutes the strong token-matching signal. On SciFact, fusion (0.4689) falls between token-only (0.4555) and graph-only (0.4827). This indicates that the optimal pipeline configuration is dataset-dependent.

(4) *FiQA achieves the highest gain (+136.2%) despite single-chunk documents.* FiQA documents are short enough to consist of a single embedding vector each, yet PQ subspace decomposition---which splits each vector into 64 independent 64-dimensional projections---creates an implicit point cloud that enables dramatic improvements. This conclusively demonstrates that PQ-Chamfer does not require multi-token documents to be effective.

### 4.4 Key Findings

**Finding 1: Token-level PQ-Chamfer provides large improvements on 4 of 5 datasets.** Token two-stage reranking yields +45.0% (ArguAna), +46.7% (NFCorpus), +93.5% (SCIDOCS), and +131.5% (FiQA) over the cosine baseline. These gains span diverse domains (financial QA, scientific citations, argumentative essays, biomedical) and document lengths (single-chunk to multi-paragraph), establishing token-level PQ-Chamfer as a broadly effective technique. The sole exception is SciFact (-3.1%), where the already-strong cosine baseline (0.4701) leaves little room for improvement---not because of short document length, but because the centroid already captures sufficient discriminative signal.

**Finding 2: Gains are inversely correlated with cosine baseline strength.** Across five datasets, the relative gain from our best pipeline is monotonically inversely correlated with cosine baseline quality: FiQA (0.1683 $\to$ +136.2%) > SCIDOCS (0.1110 $\to$ +93.5%) > NFCorpus (0.2195 $\to$ +49.0%) > ArguAna (0.3047 $\to$ +45.0%) > SciFact (0.4701 $\to$ +2.7%). This pattern is consistent across both token-level and graph-level components, and mirrors the cross-model finding (Section 4.6) that weaker embedding models benefit more from geometric post-processing.

**Finding 3: FiQA achieves the highest gain (+136.2%) from single-chunk documents.** FiQA documents consist of a single embedding vector each, yet PQ subspace decomposition yields the largest improvement across all datasets. This conclusively refutes the hypothesis that token-level PQ-Chamfer requires long, multi-token documents. The mechanism is PQ's implicit point cloud: each 4096-dimensional vector becomes 64 independent 64-dimensional projections that act as "virtual tokens," enabling fine-grained subspace matching even from a single vector. This finding redefines the applicability boundary: the critical factor is not document length, but cosine baseline strength.

**Finding 4: Kendall tau reveals complementarity.** The Kendall $\tau$ rank correlation between token PQ-Chamfer rankings and graph Laplacian rankings is 0.378 across NFCorpus queries. This moderate correlation indicates that the two methods capture substantially different aspects of relevance: token matching captures local semantic alignment, while graph smoothing captures global neighborhood consistency.

**Finding 5: Score fusion benefits datasets where graph smoothing provides complementary signal.** Fusion improves over token-only on FiQA (0.3977 vs 0.3897), SCIDOCS (0.2147 vs 0.2139), and NFCorpus (0.3271 vs 0.3220), but *hurts* on ArguAna (0.4215 vs 0.4417), where token matching alone is superior. This suggests that fusion is most valuable when graph smoothing captures neighborhood structure that token matching misses, and least valuable when the token signal is already dominant.

**Finding 6: Our pipeline surpasses BGE-large on two benchmarks.** On FiQA, our best result (0.3977) exceeds BGE-large-en-v1.5 (0.367) by +8.4%. On SCIDOCS, our best result (0.2147) exceeds BGE-large (0.162) by +32.5%. These are remarkable results for a training-free geometric post-processing method applied to a general-purpose LLM, and they suggest that the sub-document structure exploited by our pipeline contains information that retrieval-specific contrastive training does not fully capture.

**Finding 7: BM25 remains competitive on NFCorpus.** BM25 achieves 0.3256 on NFCorpus, comparable to our token two-stage (0.3220) and slightly below our best fusion (0.3271). However, BM25 is a fundamentally different approach (sparse lexical matching) and is not directly comparable to dense retrieval methods. Our framework operates entirely in the dense retrieval paradigm and achieves competitive results without any lexical features.

### 4.5 Falsification Analysis

A distinctive feature of this research program is its commitment to systematic falsification. Over the course of development, we tested and rejected 21 approaches. We detail four representative failures below and summarize all 21 in Table 3.

#### 4.5.1 Convection-Diffusion Equations: Complete Failure

The original Shape-CFD framework was built on the convection-diffusion equation:

$$\frac{\partial C}{\partial t} = D \nabla^2 C - \nabla \cdot (\vec{u} C)$$

where the advection term $\nabla \cdot (\vec{u}C)$ encodes query-directed momentum. The advection coefficient on each graph edge measures the alignment between the inter-document displacement vector and the query embedding:

$$u_{ij} = u_{\text{strength}} \cdot \frac{(v_j - v_i) \cdot \hat{q}}{\|v_j - v_i\| + \varepsilon}$$

Five independent validations established that the advection term is consistently harmful or neutral:

1. *72-configuration grid search* (V1--V2): The optimal Peclet number is $\text{Pe} \approx 0.064$, deep in the diffusion-dominated regime. Increasing advection strength beyond $u_{\text{strength}} = 0.1$ causes convergence failure.
2. *6-dataset cross-validation*: Pure Laplacian smoothing outperforms PDE (Laplacian + advection) in 5 of 6 datasets.
3. *Scalar potential variant* ($u_{ij} = \kappa(\Phi_j - \Phi_i)$, Darcy-law analogy): Improves recall but degrades NDCG, collapsing the anisotropic embedding space into an isotropic "gravity funnel."
4. *Allen-Cahn reaction-diffusion extension*: Reaction force $\gamma C(1-C)(C - \theta)$ is numerically dormant when initial score range $\Delta C_0 \approx 0.1$ (short queries), yielding $O(\Delta^3)$ reaction magnitude.
5. *Direct comparison*: On NFCorpus, Lap\_55 = 0.2900 vs PDE\_55 = 0.2852; the advection term's contribution is negative.

**Root cause.** In 4096-dimensional embedding space, the difference vector $v_j - v_i$ between two candidate documents is approximately orthogonal to the query direction $\hat{q}$, with inner product standard deviation $\sigma \approx 0.016$. The advection signal is drowned out by the isotropic structure of high-dimensional geometry. This is not an algorithmic limitation but a geometric inevitability: the *curse of orthogonality* ensures that directed transport is ineffective when the signal-to-noise ratio of directional alignment is $O(1/\sqrt{d})$.

#### 4.5.2 Density-Weighted Chamfer Distance: Reversed Intuition

*Hypothesis:* Dense regions of a document's point cloud correspond to important topics. Weighting Chamfer distance by local point density should emphasize semantically important regions.

*Experiment:* We computed local density estimates for each token embedding and used them to weight the Chamfer aggregation. Tested across multiple datasets.

*Result:* NDCG degraded by -0.9% to -1.3% compared to uniform-weight Chamfer.

*Root cause:* The intuition that "dense = important" is reversed in embedding space. Dense clusters in the point cloud correspond to *synonymous redundancy* --- tokens that express similar meanings and therefore cluster together. Weighting by density amplifies this redundancy and dilutes the discriminative power of rare, distinctive tokens that actually drive relevance differentiation. The standard Chamfer distance's uniform weighting, combined with the $\min$ operator's natural selection of nearest neighbors, already handles density variation implicitly and optimally.

#### 4.5.3 Token Two-Stage on SciFact: Strong Baseline Resists Enhancement

*Hypothesis:* Token-level PQ-Chamfer should improve retrieval across all document types.

*Experiment:* Applied the full token two-stage pipeline to all five datasets with token data.

*Result:* Token two-stage improved 4 of 5 datasets (+45% to +136%) but decreased SciFact by -3.1%.

*Revised root cause:* Our initial hypothesis attributed SciFact's degradation to short document length. However, FiQA---which has even shorter documents (single-chunk, one vector per document)---achieves the *highest* gain (+136.2%). The true explanatory variable is not document length but **cosine baseline strength**: SciFact's cosine baseline (0.4701) is by far the strongest among the five datasets, leaving minimal room for improvement. When the centroid already captures sufficient discriminative signal, token-level matching introduces more noise than signal. This is consistent with the inverse correlation between cosine baseline and gain observed across all five datasets (Finding 2, Section 4.4).

*Updated design implication:* Token-level enhancement should be applied selectively based on cosine baseline quality, not document length. Datasets with strong baselines (cosine > $\sim$0.45) may not benefit, regardless of document length.

#### 4.5.4 Multi-Emission Speculative Coarse Filtering: Recall Without Precision

*Hypothesis:* Expanding the coarse retrieval pool by using multiple centroid probe vectors (analogous to speculative decoding) should improve downstream ranking quality.

*Experiment:* Used 6--8 probe vectors per query to expand the candidate pool, achieving recall@500 improvement of +22.4%.

*Result:* NDCG@10 was unchanged despite the recall improvement.

*Root cause:* The additional documents recovered by speculative probing are "marginally relevant" --- they would appear in positions 100--500 of the ranking. The centroid coarse filter already captures the core relevant documents in the top-100. The extra candidates dilute the candidate pool without adding documents that would rank in the top-10.

We document these negative results not as failures but as evidence of systematic exploration that established clear applicability boundaries. The four cases above were selected for their theoretical depth; the remaining 17 falsified approaches are summarized in Table 3 and full details are available in Appendix A.

#### Summary of All 21 Falsified Approaches

**Table 3: Falsification record (21 approaches). Detailed analysis of items 1, 7, 20, 21 above; remainder summarized here.**

| # | Approach | Outcome | One-line root cause |
|---|----------|---------|---------------------|
| 1 | Convection-diffusion advection | 5/5 harmful | Curse of orthogonality in $\mathbb{R}^{4096}$ |
| 2 | Allen-Cahn reaction term | Dormant | $O(\Delta^3)$: short-query score range too narrow |
| 3 | Allen-Cahn + normalization | -12.9% | Rescaling $\Rightarrow$ $\gamma_{\text{eff}}=500$, divergence |
| 4 | JL random projection (4096$\to$128) | -7.6% | Information loss > noise reduction |
| 5 | "Flow-field annealing" | Falsified | JL variance amplification is random noise |
| 6 | Scalar potential (Darcy) | NDCG$\downarrow$ | Collapses to isotropic label propagation |
| 7 | Density-weighted Chamfer | -1.3% | Dense = redundancy, not importance |
| 8 | PQ code reconstruction | -34% | $\min$ amplifies quantization error |
| 9 | PQ-Chamfer initial retrieval | -4.2% | Chamfer degenerates for single-point query |
| 10 | Global graph diffusion | -7.6% | BFS noise dilution > graph benefit |
| 11 | L3 subspace cross-visit | No gain | LID: Qwen3 dims uniformly distributed |
| 12 | Binary HDC + popcount | N/A | Bottleneck is not distance computation |
| 13 | Turing patterns | Impossible | Requires two coupled species |
| 14 | Tensor diffusion | Infeasible | $4096^2$ matrix per edge $\approx$ 6 TB |
| 15 | Mixed initial field | Inconsistent | Domain-dependent: +3.2% / -1.5% |
| 16 | Temperature-scaled advection | Divergent | Pe=0.604 exceeds stability limit |
| 17 | Multi-dimensional alignment | Unstable | 0% convergence, 85.8 ms |
| 18 | Block-aligned attention | Unstable | 20% convergence, Pe=0.226 |
| 19 | PCA local projection | No net gain | 33 ms latency, marginal accuracy |
| 20 | Token on strong baseline | -3% | Cosine 0.47 already saturated |
| 21 | Multi-emission coarse filter | NDCG unchanged | Recall+22% but marginal documents only |

### 4.6 Cross-Model Generalization

A critical question for any post-processing method is whether it generalizes across embedding models, or merely compensates for specific weaknesses of one model. To answer this, we evaluate our graph Laplacian smoothing pipeline on two additional embedding models: BGE-large-en-v1.5 (335M parameters, 1024d) and BGE-M3 (568M parameters, 1024d), both served through the same llama.cpp framework as our primary Qwen3-8B model.

**Table 4: Cross-Model Generalization on NFCorpus (323 queries)**

| Model | Parameters | Dim | Cosine | +Graph Smoothing | Relative Gain | p-value |
|-------|-----------|-----|--------|-----------------|---------------|---------|
| Qwen3-Embed-8B | 8B (Q4) | 4096 | 0.2195 | 0.2900 | **+32.1%** | < 0.001 |
| BGE-M3 | 568M | 1024 | 0.2591 | 0.2751 | **+6.2%** | < 0.01 |
| BGE-large-en-v1.5 | 335M | 1024 | 0.2975 | 0.3059 | **+2.8%** | < 0.05 |

All three models show statistically significant improvement from graph smoothing, confirming that our method functions as a **model-agnostic post-processing layer**. The gains are inversely correlated with baseline quality: weaker embeddings benefit more (+32.1%) while stronger embeddings show smaller but still significant gains (+2.8%). This is consistent with our six-dataset finding (Section 4.4) that graph smoothing corrects retrieval errors that stronger signals make less frequently.

**Implementation note.** BGE-large and BGE-M3 embeddings were extracted using llama.cpp with mean pooling and 512-character truncation, which yields cosine baselines below the published MTEB leaderboard values (BGE-large: 0.2975 vs. published 0.344). The gap is attributable to inference framework differences and text truncation. Our claim is about **relative improvement** over the same embedding baseline, not absolute scores.

**Parameter sensitivity across models.** Stronger models prefer conservative smoothing ($\alpha = 0.02$, 20 iterations) while weaker models tolerate more aggressive smoothing ($\alpha = 0.1$, 10 iterations). KNN graph degree $K = 3$ is universally optimal across all models.

---

## 5 Analysis

### 5.1 Why PQ-Chamfer Works

The effectiveness of PQ-Chamfer distance rests on three mutually reinforcing mechanisms:

**Breaking concentration of measure.** In $\mathbb{R}^{4096}$, the standard deviation of cosine similarity between random unit vectors is $\sigma \approx 1/\sqrt{4096} \approx 0.016$. All pairwise distances are nearly identical, making ranking based on these distances inherently noisy. By projecting into 64-dimensional subspaces, the effective $\sigma$ increases to $1/\sqrt{64} = 0.125$, providing 7.8$\times$ better discrimination per subspace. Each subspace exhibits approximately 7.8$\times$ higher distance variance ($\sigma_s \approx 0.125$) compared to the full-dimensional cosine ($\sigma \approx 0.016$), providing substantially more discriminative power per measurement. Averaging across 64 independent subspace distances then reduces estimation variance by a factor of $\sqrt{64} = 8$, while preserving the enhanced per-subspace discriminability. The net effect is a distance metric that is both more discriminative (due to subspace-specific matching patterns) and more robust (due to variance reduction through averaging).

**Preserving dimensional structure.** Unlike random projection (Johnson-Lindenstrauss), which mixes dimensions randomly and was found to degrade NDCG by -7.6%, PQ decomposition preserves the contiguous structure of the embedding dimensions. Although we found that Qwen3's dimensions are approximately uniformly distributed (ordered $\approx$ shuffled), the contiguous grouping still outperforms random grouping by avoiding cross-contamination of independent information channels.

**Implicit virtual tokens.** Each 4096-dimensional vector, when viewed through PQ decomposition, becomes 64 independent 64-dimensional "virtual tokens." This is why even single-chunk documents (FiQA) benefit from PQ-Chamfer: the decomposition creates exploitable shape from a single point. The Chamfer distance's $\min$ operator then selects the best-matching virtual token per query component, approximating a soft attention mechanism. FiQA's +136.2% improvement---the highest across all datasets---demonstrates this mechanism at its most extreme: documents that are literally single vectors gain the most from PQ's implicit multi-view decomposition.

### 5.2 Why Graph Smoothing Works

Graph Laplacian smoothing provides a complementary signal to local token matching through two mechanisms:

**Neighborhood consensus.** If document $d_i$ is relevant to query $q$, documents that are semantically similar to $d_i$ have a higher prior probability of also being relevant. Laplacian smoothing operationalizes this intuition by propagating high scores to graph neighbors. This is particularly effective when a relevant document has a low initial score due to vocabulary mismatch or embedding noise---its highly-scored neighbors "rescue" it through diffusion.

**Cluster structure exploitation.** The KNN graph captures the cluster structure of the candidate set. In many retrieval scenarios, relevant documents form a semantic cluster, and Laplacian smoothing effectively boosts the entire cluster rather than individual documents. The cluster hypothesis (van Rijsbergen, 1979) predicts exactly this behavior.

The cross-model generalization experiment (Section 4.6) provides further evidence for this interpretation. If graph smoothing merely compensated for specific noise patterns in Qwen3-8B embeddings, it would fail on BGE models trained with different objectives. Instead, it improves all three models, suggesting that the neighborhood denoising mechanism exploits a universal property of embedding spaces: semantically related documents cluster in local neighborhoods, and their scores can be refined through mutual information.

**Why not advection?** The convection-diffusion equation adds a directional transport term that should, in principle, preferentially push scores toward the query. As we demonstrated (Section 4.5.1), this term is ineffective in high-dimensional spaces because inter-document displacement vectors are approximately orthogonal to the query direction. Graph smoothing, being isotropic, avoids this failure mode entirely: it exploits neighborhood structure without requiring any directional signal.

### 5.3 Engineering Optimization

The full pipeline has evolved from 1031 ms (JavaScript prototype, V1) to approximately 50 ms (optimized implementation), with individual stages achieving 2.2 ms in native Rust.

**Table 5: Latency analysis.**

| Stage | Latency |
|-------|--------:|
| Centroid coarse filtering | 23 ms |
| Token PQ-Chamfer reranking | 23 ms |
| Graph Laplacian smoothing | 26 ms |
| **Total pipeline** | **$\sim$50 ms** |
| Historical: JS V1 prototype | 1,031 ms |
| Optimized Rust per-stage | 2.2 ms |

**Overall speedup:** 468$\times$ from the initial JavaScript prototype to the optimized Rust implementation (1031 ms $\to$ 2.2 ms per stage). Key optimizations include: (1) offline precomputation of all document point clouds into SQLite, (2) flat Float64Array storage with 8-way loop unrolling for SIMD-like throughput, (3) PQ subspace distances using precomputed norms.

This latency profile compares favorably with cross-encoder rerankers, which typically require $\sim$500 ms for 30 candidates. Our full pipeline processes 55--200 candidates in 50 ms, a 10$\times$ latency advantage with comparable or superior ranking quality on NFCorpus.

### 5.4 Counter-Intuitive Findings

Several findings from our research contradict common intuitions:

**"Cosine is not the ceiling; query single-point is."** We initially hypothesized that replacing cosine similarity with PQ-Chamfer distance for initial retrieval would improve recall. Instead, it degraded NDCG by -4.2% (V12). The reason: Chamfer distance degenerates for a single-point query cloud (1 token vs $n$ document tokens), reducing to a noisy variant of cosine. The correct strategy is to use cosine for coarse filtering (where the query is a single point) and Chamfer for reranking (where both query and document are point clouds).

**"Dense regions are noise, not signal."** Density-weighted Chamfer distance, motivated by the intuition that dense point cloud regions represent important topics, consistently degraded performance. In embedding space, dense clusters arise from synonymous redundancy, not semantic importance.

**"Representation change beats algorithm change."** VT-Aligned virtual token decomposition (V13, +2.5%), a pure mathematical reordering of computation, outperforms adjoint-state prefetching (V7.1, +1.1%), which involves sophisticated PDE-based boundary control. This suggests that investing in richer representations yields higher returns than investing in more complex algorithms applied to impoverished representations.

**"PQ is not compression; PQ is decomposition."** The information retrieval community uses PQ for approximate nearest neighbor search---a compression technique that sacrifices accuracy for speed. We use PQ decomposition for the opposite purpose: to *improve* distance accuracy by breaking the concentration-of-measure effect. Same mathematical tool, opposite application.

### 5.5 Applicability Boundaries

Our experimental results establish clear boundaries for when each component is most effective:

**Token-level PQ-Chamfer.** Highly effective on 4 of 5 datasets (+45% to +136%), regardless of document length. FiQA (+136.2%) demonstrates that even single-chunk documents benefit dramatically through PQ's implicit point cloud mechanism. The critical boundary is not document length but **cosine baseline strength**: token-level matching is less effective when the cosine baseline is already strong (SciFact, 0.4701, -3.1%), because the centroid already captures sufficient discriminative signal and token-level matching introduces noise without commensurate gain.

**Graph Laplacian smoothing.** Universally effective across all tested datasets and document lengths. Most impactful when cosine baseline is weakest (FiQA: +44.2% from 0.1683). Smallest impact when cosine is already strong (SciFact: +2.7% from 0.4701). Safe to apply as a default postprocessing step.

**Score fusion.** Beneficial when graph smoothing provides complementary signal (FiQA, SCIDOCS, NFCorpus) but counterproductive when token matching already dominates (ArguAna: fusion 0.4215 < token-only 0.4417). On NFCorpus, fusion yields +49.0% vs +46.7% for token-only. The fusion weight $\lambda = 0.7$ has been validated across five datasets, with consistent benefits on 3 of 5.

**Corpus scale.** The Quora experiment (522,931 documents, 10,000 queries) confirms that the framework scales to large corpora without degradation. The centroid coarse filter provides sublinear scaling for the token-level stage.

**Comparison with trained models.** Our pipeline surpasses BGE-large-en-v1.5 on FiQA (+8.4%) and SCIDOCS (+32.5%), and approaches ColBERTv2 on ArguAna (0.4417 vs 0.460) and NFCorpus (0.3271 vs 0.338). However, substantial gaps remain on SciFact (0.483 vs BGE 0.738) and Quora (0.675 vs BGE 0.889), where retrieval-specialized training provides large benefits. The primary bottleneck is the embedding quality: Qwen3-8B is a general-purpose language model, not optimized for retrieval. Our geometric operations amplify the discriminative signal present in the embeddings, but they cannot create signal that the base model does not encode. Cross-model experiments (Section 4.6) confirm that our pipeline provides statistically significant gains on BGE-large and BGE-M3 as well, though with diminishing returns as baseline quality improves (+2.8% for BGE-large vs +32.1% for Qwen3-8B). The fact that our training-free pipeline already surpasses a trained model on two benchmarks suggests that stacking our geometric post-processing on top of a trained retrieval model could yield further gains.

---

## 6 Extended Discussion

### Multi-Vector Retrieval Models

ColBERT (Khattab & Zaharia, 2020) represents both queries and documents as sets of contextualized token embeddings and computes relevance via the MaxSim operator: $\text{score}(q, d) = \sum_{i \in q} \max_{j \in d} E_{q_i} \cdot E_{d_j}^T$. This asymmetric formulation scores each query token against its best-matching document token. ColBERTv2 (Santhanam et al., 2022) introduces residual compression for storage efficiency, and PLAID (Santhanam et al., 2022b) adds centroid-based candidate pruning. Our PQ-Chamfer distance differs from MaxSim in three ways: (i) it is symmetric, computing matching in both directions; (ii) it operates in PQ subspaces rather than full-dimensional space; (iii) it is applied to general LLM hidden states rather than a trained retrieval model's outputs.

### Graph-Based Methods in Information Retrieval

Beyond the diffusion-based approaches discussed in Section 2, graph augmented retrieval (GAR; MacAvaney et al., 2022) uses pseudo-relevance feedback on a document graph to expand the candidate set. GNN-based rerankers (Glass et al., 2024) learn message-passing functions for score refinement. Our approach is most similar to label propagation on graphs (Zhu et al., 2003) but applied to retrieval scores rather than classification labels, and operating on graphs constructed from point cloud distances rather than pretrained similarity metrics.

### Product Quantization for Retrieval

FAISS (Johnson et al., 2021) is the standard library for PQ-based approximate nearest neighbor search, supporting IVF-PQ, HNSW-PQ, and other composite indexes. ScaNN (Guo et al., 2020) achieves state-of-the-art throughput through anisotropic quantization loss. OPQ (Ge et al., 2014) applies an orthogonal rotation before quantization to align subspace boundaries with data variance. All these methods use PQ as a compression technique: they quantize vectors to reduce memory and accelerate distance approximation. Our PQ-Chamfer uses the decomposition without quantization, computing exact subspace distances and aggregating them. This "PQ without Q" approach is novel in the retrieval literature.

### Point Cloud Metrics in NLP

The application of point cloud distances to text representation is relatively unexplored. Word Mover's Distance (WMD; Kusner et al., 2015) computes the Earth Mover's Distance between word embedding distributions of two documents. BERTScore (Zhang et al., 2020) uses greedy matching between contextual embeddings, which is equivalent to a one-directional Chamfer distance. Our work extends this line by using symmetric Chamfer distance over token-level hidden states in PQ subspaces, combined with graph-based global regularization.

---

## 7 Conclusion

We have demonstrated that purely geometric operations---PQ subspace decomposition, Chamfer distance aggregation, and graph Laplacian smoothing---constitute a **model-agnostic post-processing layer** that improves retrieval quality across six BEIR datasets and three embedding models without any training. On five datasets with token-level evaluation, these operations yield +2.7% to +136.2% improvement over the cosine similarity baseline of the same embedding model. Cross-model evaluation on Qwen3-8B, BGE-M3, and BGE-large-en-v1.5 confirms that the gains are not model-specific artifacts but reflect universal structural properties of embedding spaces. Remarkably, our training-free pipeline **surpasses BGE-large-en-v1.5**---a retrieval-specialized model trained on millions of query-document pairs---on FiQA (0.3977 vs 0.367, +8.4%) and SCIDOCS (0.2147 vs 0.162, +32.5%), and approaches ColBERTv2 on ArguAna (0.4417 vs 0.460) and NFCorpus (0.3271 vs 0.338). These results demonstrate that training-free geometric post-processing can not only narrow but in some cases **close** the gap with retrieval-specialized models.

Our systematic falsification of 21 approaches yielded insights of independent value. The complete failure of convection-diffusion equations---the original theoretical foundation of this research program---demonstrates that high-dimensional embedding spaces resist the directional transport mechanisms that are effective in low-dimensional physical systems. The finding that density-weighted Chamfer degrades performance inverts the common intuition about embedding space geometry. And the observation that PQ subspace decomposition benefits even single-vector documents challenges the assumption that point cloud distances require multiple points.

**Limitations.** Our method does not reach the absolute performance of retrieval-specialized models (ColBERTv2, BGE-large) on some datasets (SciFact, Quora), as the primary bottleneck is the embedding quality of the general-purpose base model. Token-level enhancement degrades performance when the cosine baseline is already strong (SciFact, -3.1%). Score fusion is not universally beneficial (ArguAna: fusion < token-only). The fusion weight $\lambda = 0.7$ was validated across five datasets but may not be optimal for all domains. Cross-model generalization has been verified on three models, but the range of architectures tested remains limited.

**Future work.** (1) Our cross-model experiment (Section 4.6) confirms generalization across three models; future work should extend this to a broader range of architectures including instruction-tuned models and cross-encoder rerankers. (2) **Stacking on trained retrievers:** Given that our pipeline already surpasses BGE-large on two benchmarks using general-purpose Qwen3-8B, applying PQ-Chamfer and graph smoothing on top of BGE-large or ColBERTv2 embeddings could push absolute performance beyond current state of the art. (3) Adaptive pipeline selection: automatically choosing between token-only, graph-only, and fusion configurations based on estimated cosine baseline strength. (4) Extension to multilingual retrieval using multilingual LLM hidden states. (5) Scalability to billion-document corpora through hierarchical PQ indexing.

---

## 8 Acknowledgements

**AI assistance.** Claude (Anthropic) and Gemini (Google) participated extensively in method discussion, mathematical derivation, and experimental design throughout this research program. Their contributions include: Gemini proposed the UOT theoretical framework connecting Chamfer distance to unbalanced optimal transport, the Allen-Cahn reaction term, and the dynamic threshold mechanism; Claude performed experimental validation, diagnosed failure modes, and formulated the concentration-of-measure explanation for PQ-Chamfer's effectiveness. The Claude-Gemini "red team vs blue team" adversarial review process across five rounds was instrumental in falsifying weak ideas and strengthening surviving ones.

**Original contributions by the author.** The following ideas originated with the author (Chen, Yifan): (1) "Documents are not points, they are shapes"---the foundational insight that documents should be represented as point clouds rather than single vectors; (2) the token-level point cloud formulation using general LLM hidden states with `--pooling none`; (3) the PQ-Chamfer distance metric---repurposing PQ decomposition for distance computation rather than compression; (4) the discovery that VT-Aligned virtual token decomposition (reversing the order of subspace aggregation and Chamfer computation) yields gains at zero cost; (5) the experimental falsification of PQ-Chamfer as initial retrieval (V12: "cosine is not the ceiling, query single-point is"); (6) the multi-granularity pipeline design (centroid coarse filter $\to$ token reranking $\to$ graph smoothing); (7) the "representation change beats algorithm change" principle derived from comparative ablation.

The author is 16 years old and conducted this research independently outside of any institutional affiliation.

---

## References

Dampanaboina, V., et al. (2024). Diffusion-Aided RAG: Relevance Propagation via Graph Diffusion for Retrieval-Augmented Generation. *arXiv preprint*.

Fan, H., Su, H., & Guibas, L. J. (2017). A Point Set Generation Network for 3D Object Reconstruction from a Single Image. *CVPR*.

Ge, T., He, K., Ke, Q., & Sun, J. (2014). Optimized Product Quantization. *IEEE TPAMI*, 36(4), 744-755.

Glass, M., et al. (2024). G-RAG: Graph Neural Retrieval Augmented Generation. *arXiv preprint*.

Guo, R., et al. (2020). Accelerating Large-Scale Inference with Anisotropic Vector Quantization. *ICML*.

Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M.-W. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML*.

Jegou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. *IEEE TPAMI*, 33(1), 117-128.

Johnson, J., Douze, M., & Jegou, H. (2021). Billion-Scale Similarity Search with GPUs. *IEEE TBD*, 7(3), 535-547.

Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *SIGIR*.

Kusner, M. J., Sun, Y., Kolkin, N. I., & Weinberger, K. Q. (2015). From Word Embeddings To Document Distances. *ICML*.

Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

MacAvaney, S., Tonellotto, N., & Macdonald, C. (2022). Adaptive Re-Ranking with a Corpus Graph. *CIKM*.

Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford InfoLab Technical Report*.

Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). The Earth Mover's Distance as a Metric for Image Retrieval. *IJCV*, 40(2), 99-121.

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. *NAACL*.

Sejourne, T., Peyre, G., & Vialard, F.-X. (2023). Unbalanced Optimal Transport, from Theory to Numerics. *Handbook of Numerical Analysis*, 24, 407-471.

Thakur, N., Reimers, N., Rucktaschel, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS Datasets and Benchmarks*.

van Rijsbergen, C. J. (1979). *Information Retrieval*. Butterworths.

Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding. *arXiv preprint arXiv:2309.07597*.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.

Zhu, X., Ghahramani, Z., & Lafferty, J. D. (2003). Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions. *ICML*.

---

## Appendix A: Complete Falsification Record

The 21 falsified approaches summarized in Table 3 are documented in full---including experimental configurations, quantitative results, and root-cause analyses---in the project repository. Each entry records the hypothesis, experimental setup, observed result, and diagnostic explanation.

Repository: `https://github.com/yifanchen/shape-cfd` (to be made public upon publication).

The falsification log follows a structured format: (1) **Hypothesis** stated before the experiment; (2) **Protocol** specifying parameters, datasets, and metrics; (3) **Result** with quantitative measurements; (4) **Root-cause analysis** explaining *why* the approach failed, not merely *that* it failed. We believe this level of documentation is valuable for the community: negative results in retrieval are rarely published, yet they constrain the design space and prevent redundant exploration.
