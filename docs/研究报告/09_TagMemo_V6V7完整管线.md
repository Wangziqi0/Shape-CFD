# 09 — TagMemo V6/V7 完整管线深挖报告

> **Agent B1** | 深挖对象：`KnowledgeBaseManager.js` + `rag_params.json`  
> 关联第一轮报告：01（RAG核心三件套）| 报告日期：2026-03-13

---

## 一、V6 七步管线逐步拆解

入口函数：`_applyTagBoostV6(vector, baseTagBoost, coreTags, coreBoostFactor)`

### Step 1：EPA 分析 — "你在哪个世界"

```js
const epaResult = this.epa.project(originalFloat32);           // 语义投影
const resonance = this.epa.detectCrossDomainResonance(vector); // 跨域共振
const queryWorld = epaResult.dominantAxes[0]?.label || 'Unknown';
```

**输出**：`logicDepth`（0~1，高=聚焦）、`entropy`（0~1，高=散乱）、`resonance`（跨域共振强度）、`queryWorld`（主导世界标签）

### Step 2：残差金字塔分析 — "现有知识覆盖了多少"

```js
const pyramid = this.residualPyramid.analyze(originalFloat32);
const features = pyramid.features; // { coverage, novelty, tagMemoActivation, ... }
```

**输出**：`coverage`（能量覆盖率）、`novelty`（新颖度）、`tagMemoActivation`（是否需要增强记忆）

### Step 3：动态 Boost 计算 — 核心公式

```
activationMultiplier = actRange[0] + tagMemoActivation × (actRange[1] - actRange[0])
                     = 0.5 + tagMemoActivation × 1.0    // 范围 [0.5, 1.5]

dynamicBoostFactor = logicDepth × (1 + ln(1 + resonance)) / (1 + entropy × 0.5) × activationMultiplier

effectiveTagBoost = baseTagBoost × clamp(dynamicBoostFactor, 0.3, 2.0)
```

**动态核心加权**：
```
coreMetric = logicDepth × 0.5 + (1 - coverage) × 0.5
dynamicCoreBoostFactor = 1.20 + coreMetric × 0.20    // 范围 [1.20, 1.40]
```
> 逻辑深度越高或覆盖率越低时，核心标签权重越高（20%~40% 加成）

### Step 4：Tag 收集 + 三重门控

遍历金字塔每层的 Tag，对每个 Tag 计算 `adjustedWeight`：

```
adjustedWeight = contribution × layerDecay × langPenalty × coreBoost
```

**4A. 语言置信度门控**（Language Confidence Gating）：
- 检测纯英文技术词（正则：非中文 + 字母数字 + 长度>3）
- 非技术语境下施加惩罚：未知世界 `penaltyUnknown=0.05`，跨域 `penaltyCrossDomain=0.1`
- 社会/政治世界观下惩罚软化为 `sqrt(penalty)`，保护 Trump/Musk 等英文实体

**4B. 世界观门控**（Worldview Gating）：
- 简化实现：`layerDecay = 0.7^level`（更深层级权重递减）

**4C. 核心 Tag Spotlight**：
- 核心标签额外乘以 `dynamicCoreBoostFactor × (0.95 + similarity × 0.1)`

### Step 4.5：LIF 脉冲扩散 — "仿脑认知涌现"

模拟 Leaky Integrate-and-Fire (LIF) 神经元网络扩散：

| 参数 | 值 | 含义 |
|------|---|------|
| `MAX_HOPS` | 2 | 最大扩散跳数 |
| `FIRING_THRESHOLD` | 0.10 | 触发门槛，低于此不放电 |
| `DECAY_FACTOR` | 0.3 | 突触衰减因子 |
| `MAX_EMERGENT_NODES` | 50 | 涌现节点总数上限 |
| `MAX_NEIGHBORS_PER_NODE` | 20 | 单节点扇出限制 |

**脉冲传递公式（V7 增强）**：
```
injectedCurrent = energy × coocWeight × DECAY_FACTOR × nodeResidual
```
> V7 新增 `nodeResidual`：来自 `tagIntrinsicResiduals`（内生残差增益，范围 [0.5, 2.0]）

**关键工程决策**：
- 种子节点不被循环共现膨胀（取 `max` 而非累加）
- 涌现节点按能量排序只保留 Top-50
- 微电流 < 0.01 直接丢弃，极大缩减 Map 规模

### Step 4.6：核心 Tag 补全

如果核心标签未被金字塔和脉冲扩散召回，从数据库中强制补入，权重为 `maxBaseWeight × dynamicCoreBoostFactor`。

### Step 5：语义去重

- 按 `adjustedWeight` 降序排列
- 逐个与已选 Tag 计算余弦相似度，阈值 `0.88`（可调）
- 冗余 Tag 的 20% 能量转移给代表性 Tag
- 核心属性可继承（冗余 Tag 是 Core 则代表也标为 Core）

### Step 6-7：向量融合与输出

```
contextVec = Σ(tagVec[i] × adjustedWeight[i]) / totalWeight → 归一化
alpha = min(1.0, effectiveTagBoost)
fusedVec = (1 - alpha) × originalVec + alpha × contextVec → 归一化
```

**输出过滤**（返回 matchedTags 时的门槛）：
- Core Tags 必须包含（不被过滤）
- 英文技术词：权重 > maxWeight × 0.08
- 普通词：权重 > maxWeight × 0.015

---

## 二、V7 有向共现矩阵构建

函数：`_buildDirectedCooccurrenceMatrix()`

### 核心概念：序位势能 PHI

每个 Tag 在文件中有 `position` 字段（1-indexed，表示标签在文件中的出现顺序）。位置越靠前，势能越高：

```
PHI(pos, n) = PHI_MAX - (PHI_MAX - PHI_MIN) × (pos - 1) / (n - 1)
            = 0.9 - 0.4 × (pos - 1) / (n - 1)     // 范围 [0.5, 0.9]
```

### 有向边权重计算

```sql
-- 只查询 position 有效（>0）且 pos1 < pos2 的配对（有向：前 → 后）
SELECT ft1.tag_id as source, ft2.tag_id as target, ft1.position, ft2.position
FROM file_tags ft1 JOIN file_tags ft2 
  ON ft1.file_id = ft2.file_id AND ft1.position < ft2.position
```

```
edgeWeight(source → target) += PHI(pos1) × PHI(pos2)
```

> 这意味着：文件中靠前的两个 Tag 共现权重最高（0.9×0.9=0.81），末尾两个最低（0.5×0.5=0.25）

### 旧数据兼容

对于 `position=0` 的旧数据，退化为无向等权重：
```
weight = count × LEGACY_PHI² = count × 0.49
```

---

## 三、V7 内生残差（OrdinalSpike）

### 计算方式

由 Rust 引擎 `VexusIndex.computeIntrinsicResiduals(dbPath)` 预计算：
- 对每个 Tag，在其共现邻居的向量子空间中做正交投影
- 计算残差能量 `residual_energy`
- 结果写入 `tag_intrinsic_residuals` 表

### 加载与应用

```js
// 加载时 clamp 到 [0.5, 2.0]
clamped = Math.max(0.5, Math.min(2.0, row.residual_energy))
```

- **残差高（→2.0）**：该 Tag 向量与其邻居差异大，是"独特节点"，脉冲扩散时增益 ×2
- **残差低（→0.5）**：该 Tag 可被邻居线性表示，是"冗余节点"，脉冲扩散时衰减 ×0.5

### 重建调度（V7.7 混合防抖）

```
累积变更数 += 每次 Tag 变更
当 累积数 >= 动态阈值(Tag总数 × 1%, clamp [10, 200]):
  → 启动 5 分钟冷却防抖计时器
  → 超时后执行: 重建共现矩阵 + Rust 重算内生残差
```

---

## 四、rag_params.json 完整参数表

### KnowledgeBaseManager 参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `activationMultiplier` | [0.5, 1.5] | tagMemoActivation 映射到 boost 乘数的范围 |
| `dynamicBoostRange` | [0.3, 2.0] | effectiveTagBoost 的最终 clamp 范围 |
| `coreBoostRange` | [1.20, 1.40] | 核心标签动态加权的范围（20%~40%加成） |
| `deduplicationThreshold` | 0.88 | 语义去重的余弦相似度阈值 |
| `techTagThreshold` | 0.08 | 英文技术词的输出过滤门槛（相对最大权重） |
| `normalTagThreshold` | 0.015 | 普通词的输出过滤门槛 |
| `languageCompensator.penaltyUnknown` | 0.05 | 未知世界观下英文实体的惩罚系数 |
| `languageCompensator.penaltyCrossDomain` | 0.1 | 跨域世界观下英文实体的惩罚系数 |

### RAGDiaryPlugin 参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `noise_penalty` | 0.05 | 噪音标签的惩罚因子 |
| `tagWeightRange` | [0.05, 0.45] | Tag 权重的有效范围 |
| `tagTruncationBase` | 0.6 | Tag 截断基准线 |
| `tagTruncationRange` | [0.5, 0.9] | Tag 截断范围 |
| `timeDecay.halfLifeDays` | 30 | 时间衰减半衰期（天） |
| `timeDecay.minScore` | 0.5 | 时间衰减最低得分 |

---

## 五、与第一轮报告的关联补充

| 第一轮报告 | 补充内容 |
|-----------|---------|
| 01（RAG三件套） | EPA/ResidualPyramid 被 V6 管线在 Step 1-2 调用，它们的输出直接驱动 Step 3 的动态 boost 公式 |
| 01（ResultDeduplicator） | V6 管线内 Step 5 的语义去重是 Tag 级别的余弦去重（阈值0.88），而 ResultDeduplicator 是检索结果级别的正交去重，两者在不同层级工作 |
| 02（知识库） | 共现矩阵基于 `file_tags` 表的 `position` 字段构建，这是第一轮未涉及的数据库字段 |

---

## 六、对 legal-assistant 的深度借鉴

| 借鉴点 | 具体建议 |
|--------|---------|
| **动态 Boost 公式** | 法律 RAG 可根据查询的法律领域确定性（类似 logicDepth）和知识库覆盖率动态调节检索强度 |
| **LIF 脉冲扩散** | 法律概念的关联扩散（如"劳动合同" → "经济补偿金" → "N+1赔偿"），但需将 DECAY_FACTOR 调高（法律概念关联更强） |
| **有向序位势能** | 法条中的概念有自然顺序（如"犯罪构成要件"中主体→客体→主观→客观），可利用序位势能建模 |
| **内生残差增益** | 法律概念中的"独特节点"（如特殊法条）在检索扩散时应获得更高权重 |
| **热调参数 + chokidar 监听** | 允许运行时修改 RAG 参数无需重启，对法律 RAG 调优非常实用 |
| **混合防抖调度** | 法律知识库更新时的索引重建策略：累积变更达阈值后等待冷却期再重建 |

---

> **结论**：TagMemo V6 管线是一个精密的 7 步向量增强流水线，核心创新在于将 EPA 语义分析 + 残差金字塔覆盖率 + LIF 神经脉冲扩散三者有机结合。V7 的有向序位势能和内生残差增益进一步引入了 Tag 的位置语义和独特性信息。整套系统通过 rag_params.json 实现热调控，且带有完善的防抖/重建调度机制。
