# 01 — RAG 核心三件套研究报告

> **Agent 1** | 研究目标：EPAModule.js · ResidualPyramid.js · ResultDeduplicator.js  
> 文件体量：~43KB | 报告日期：2026-03-13

---

## 一、整体设计理念

VCPToolBox 的 RAG 核心由三个模块组成，形成一条 **"语义理解 → 逐层分解 → 多样性保障"** 的流水线：

| 模块 | 隐喻 | 核心职责 |
|------|------|---------|
| **EPAModule** | 光谱仪 | 把高维向量投射到正交语义坐标系，得出"语义能量谱" |
| **ResidualPyramid** | 洋葱剥皮 | 逐层剥离已被 Tag 解释的能量，暴露出未被覆盖的语义残差 |
| **ResultDeduplicator** | 策展人 | 从候选结果中挑选出覆盖度最大、冗余最小的精选子集 |

三者的设计哲学统一于 **"能量/信息论"**：将向量模的平方视为"能量"，用能量的分解、传递、耗散来描述语义匹配的质量。

---

## 二、EPAModule — 嵌入投影分析

### 2.1 核心数据结构

```
orthoBasis: Float32Array[]   // 正交基向量（语义坐标轴）
basisMean: Float32Array      // 全局加权平均向量（用于中心化）
basisEnergies: Float32Array  // 各基底的特征值（方差贡献）
basisLabels: string[]        // 基底标签（最近 Tag 名称）
```

### 2.2 算法流程

**初始化阶段（构建语义坐标系）**：

1. **K-Means 聚类**：将数据库中所有 Tag 向量聚为 K=32 个簇，得到加权质心。使用 Forgy 初始化 + 归一化质心 + 收敛检测（tolerance=1e-4，max=50 轮）。用点积代替欧氏距离（假设向量已归一化），提速明显。
2. **加权 PCA**：
   - 计算加权平均向量 → 中心化 → 构建 **Gram 矩阵**（n×n 而非 dim×dim，避免构建巨大协方差矩阵）
   - **Power Iteration + Deflation** 提取特征向量
   - 关键优化：每次迭代后做 **Gram-Schmidt 再正交化**，防止收敛到已提取主成分
   - 特征向量从 Gram 空间映射回原始高维空间：`U_pca = X^T * v / sqrt(λ)`
3. **主成分选择**：累计方差贡献达 95% 时截断，最少保留 8 个基底

**查询阶段（语义投影）**：

1. 去中心化：`v' = v - mean`
2. 投影到各主成分轴，计算能量分布
3. 计算 **归一化熵** → `logicDepth = 1 - entropy`（熵低 = 聚焦特定领域）
4. 提取主导轴（能量 > 5% 的基底）

**跨域共振检测**：检测 Query 是否同时强激活了多个正交语义轴。用几何平均 `sqrt(E1 * E2)` 评估共激活强度，阈值 0.15 以上视为"共振"（跨领域查询）。

### 2.3 可借鉴点

- **Gram 矩阵技巧**：当样本数 n << 维度 dim 时，n×n 矩阵替代 dim×dim 协方差矩阵，计算效率极高
- **Power Iteration + 再正交化**：纯 JS 实现 SVD 的实用方案，避免引入大型线代库
- **Rust/JS 双路径**：优先调 Rust native addon，失败后回退 JS，保证可用性

---

## 三、ResidualPyramid — 残差金字塔

### 3.1 核心思想

把查询向量 Q 看作一个"能量体"，逐层用最相似的 Tag 向量去"吸收"它的能量：

```
Level 0: Q₀ → 搜索最近 Tag → 正交投影 → 残差 R₁ = Q₀ - Projection₀
Level 1: R₁ → 搜索最近 Tag → 正交投影 → 残差 R₂ = R₁ - Projection₁
...
停止条件: 残差能量 / 原始能量 < 10%
```

### 3.2 关键算法

**Modified Gram-Schmidt 正交投影**：

对每层检索到的 Top-K Tag 向量，先做 Gram-Schmidt 正交化构建基底，然后将当前残差投影到该子空间。利用正交投影性质：`||R_old||² = ||Projection||² + ||R_new||²`，精确计算每层解释的能量比例。

**握手分析（Handshake Analysis）**：

计算 Query 与每个 Tag 的差向量（Delta），分析：
- **方向一致性 (Coherence)**：所有 Delta 是否指向同一方向 → 高则说明 Query 偏向未知领域
- **模式强度 (Pattern Strength)**：Delta 之间两两相似度 → 高则说明 Tag 在同一个簇
- **新颖度公式**：`novelty = residualRatio × 0.7 + directionalNovelty × 0.3`

### 3.3 输出特征

```js
{
  depth,              // 金字塔层数
  coverage,           // 总解释能量（0~1）
  novelty,            // 新颖度（未被解释的能量 + 偏移方向性）
  coherence,          // Tag 聚类一致性
  tagMemoActivation,  // 综合决策：是否激活 TagMemo 增强？
  expansionSignal     // 是否需要搜索新 Tag？
}
```

### 3.4 可借鉴点

- **残差迭代思想**：不满足于单次检索，逐层"剥皮"发现隐含语义，对法律场景"一问多点"极有价值
- **能量阈值截断**：自适应决定检索深度，避免过拟合噪声
- **tagMemoActivation 决策信号**：通过 `coverage × coherence × (1 - noise)` 自动调节后续策略强度

---

## 四、ResultDeduplicator — 智能去重

### 4.1 设计目标

从 N 个候选结果中选出最多 maxResults（默认 20）个，**最大化信息覆盖、最小化冗余**。

### 4.2 算法流程

1. **SVD 主题提取**：对当前结果集（而非预训练 Tag 库）执行加权 PCA，发现本次检索结果的潜在主题分布。累计能量 95% 作为显著主题筛选线。

2. **贪心残差选择**：
   - 第一步：选择与 Query 余弦相似度最高的结果作为 Anchor
   - 迭代步：对每个未选候选，计算其在已选集合正交基下的残差能量（= 新信息量），乘以原始分数作为综合得分
   - 每轮选综合得分最高者加入集合
   - 终止条件：残差贡献 < 0.01（新信息可忽略）或达到 maxResults

3. **关键公式**：`score = noveltyEnergy × (originalScore + 0.5)`

### 4.3 可借鉴点

- **贪心正交选择法**：经典的 **MaxVol / Diversity-promoting** 策略，比简单余弦阈值去重更优雅
- **在候选集上临时做 SVD**：不依赖预训练，每次查询实时分析，适应性强
- **复用已有模块**：直接调用 EPAModule 和 ResidualPyramid 的方法，体现了模块化设计

---

## 五、三模块协作关系

```
Query 进入
  ↓
EPAModule.project(query)  →  得到语义能量谱、logicDepth、共振信息
  ↓                            ↓（决策参数）
ResidualPyramid.analyze(query) → 逐层检索 Tag，计算 coverage/novelty
  ↓                               ↓（tagMemoActivation 等信号）
[TagMemo 浪潮检索]             ←  根据信号调整策略（检索量、阈值等）
  ↓
候选结果集
  ↓
ResultDeduplicator.deduplicate(candidates, query)  → 精选去重后的最终结果
  ↓
送入 LLM
```

**协作本质**：EPA 做"宏观诊断"（这个问题属于什么领域、是否跨域）→ ResidualPyramid 做"微观分析"（现有知识覆盖了多少、还差什么）→ Deduplicator 做"质量控制"（最终结果不冗余、有多样性）。

---

## 六、对 legal-assistant 的借鉴价值

| 借鉴点 | 原模块 | 法律场景应用 |
|--------|--------|-------------|
| Gram 矩阵 PCA 降维 | EPA | 法律概念空间的快速语义分析，判断查询属于劳动法/合同法/刑法等领域 |
| 跨域共振检测 | EPA | 检测"劳动合同中的知识产权条款"等跨领域问题 |
| 残差金字塔逐层分解 | Pyramid | 法律问题"一问多点"的逐层拆解（主请求→从属请求→补充证据） |
| 能量阈值自适应 | Pyramid | 自动决定检索深度，简单问题浅搜、复杂问题深搜 |
| tagMemoActivation 信号 | Pyramid | 动态调节检索策略强度（DRC 模块的灵感来源） |
| 贪心正交去重 | Deduplicator | 法规检索结果的多样性保障，避免返回高度相似的法条 |
| 模块化 + Rust 加速回退 | 三者共有 | 核心计算 Rust 加速、JS 兜底的双轨设计模式 |

---

> **结论**：三件套的核心价值在于用线性代数（PCA/Gram-Schmidt/SVD）构建了一套 **"语义能量分析"框架**，使得 RAG 系统不再是简单的"搜索 → 拼接"，而能够理解查询的语义深度、知识覆盖度和结果多样性。这对法律 RAG 系统的精度提升具有直接参考意义。
