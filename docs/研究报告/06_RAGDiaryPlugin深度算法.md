# 06 RAGDiaryPlugin 深度算法挖掘

> Agent 6 研究报告 | 研究范围：~247KB（主文件157KB + 5个子模块）

---

## 一、设计理念

RAGDiaryPlugin 是 VCPToolBox 中最大的单插件（3242行），其核心理念是**将日记转变为可检索的长期记忆系统**，解决传统 RAG 的三大痛点：

1. **时间维度模糊** → 时间感知检索（TimeAware RAG）
2. **逻辑关联模糊** → 语义组增强检索（Semantic Group）
3. **上下文漂移** → 衰减聚合向量 + 动态K值

**插件类型为 `hybridservice`**，采用 `direct` 通信协议，通过替换 system prompt 中的占位符（`[[]]`/`<<>>`/`《《》》`/`{{}}`）将检索结果注入上下文。

---

## 二、核心数据结构

### 2.1 多级缓存体系

```
queryResultCache (Map)    — 查询结果缓存，TTL 1h，最大200条
embeddingCache (Map)      — 向量缓存，TTL 2h，最大500条
aiMemoCache (Map)         — AIMemo结果缓存，TTL 30min，最大50条
enhancedVectorCache (Obj) — 日记本标签增强向量（磁盘持久化）
groupVectorCache (Map)    — 语义组预计算向量
metaChainThemeVectors (Obj) — 元思考链主题向量
```

所有缓存均基于 SHA-256 哈希键，配合配置文件哈希变更检测自动失效。

### 2.2 上下文向量映射 (ContextVectorManager)

```
vectorMap: Map<normalizedHash, {vector, role, originalText, timestamp}>
historyAssistantVectors: Array<vector>  // 按时间排序的AI向量
historyUserVectors: Array<vector>       // 按时间排序的用户向量
```

核心参数：`fuzzyThreshold=0.85`（Dice系数模糊匹配），`decayRate=0.75`，`maxContextWindow=10`。

---

## 三、算法流程

### 3.1 主管线 (`processMessages`)

```
消息输入 → 上下文向量映射更新
        → 识别四种占位符（[[]], <<>>, 《《》》, {{}}）
        → 文本预处理（HTML/emoji/工具标记/系统通知 清洗）
        → 统一向量化（合并 AI+User 内容）
        → 动态参数计算 (L, R, S → K, β, TagTruncation)
        → 上下文语义分段 (cosine阈值=0.70)
        → 时间表达式解析
        → 上下文日记去重前缀提取
        → 并行处理各占位符 → 替换注入
```

### 3.2 动态参数计算（V3 三指标体系）

这是最精华的算法之一，基于三个向量空间指标动态调控检索行为：

**逻辑深度指数 L ∈ [0,1]**：衡量向量能量集中度。取Top-K维度能量占比，归一化后得出。L越高 → 意图越明确。

```js
concentration = topKEnergy / totalEnergy
L = (concentration - expectedUniform) / (1 - expectedUniform)
```

**共振指数 R**：由 EPA 模块提供，反映查询与知识库的"共振程度"。

**语义宽度指数 S ∈ [0,1]**：将 L2 归一化向量的 v_i² 视为概率分布，计算归一化熵。S≈1 → 语义宽泛，S≈0 → 语义精准。

```js
S = -Σ(p * log(p)) / log(dim)   // p = v_i²
```

三指标融合计算动态参数：

```
β = sigmoid(L * log(1 + R + 1) - S * noise_penalty)
TagWeight = weightRange[0] + β * (weightRange[1] - weightRange[0])  // [0.05, 0.45]
K = clamp(k_base + round(L*3 + log1p(R)*2), 3, 10)
TagTruncation = clamp(0.6 + L*0.3 - S*0.2 + min(R,1)*0.1, 0.5, 0.9)
```

### 3.3 语义组增强检索 (SemanticGroupManager)

**激活检测**：遍历所有语义组，对组内词元（含自学习词元）进行 `flexibleMatch`（大小写不敏感的 includes），计算激活强度 = 命中词元数 / 总词元数。

**向量增强**：将查询向量与命中语义组的预计算向量做加权平均。权重 = 组全局权重 × 激活强度。

```js
vectors = [queryVector, ...groupVectors]
weights = [1.0, groupWeight1 * strength1, ...]
enhancedVector = weightedAverage(vectors, weights)
```

语义组向量通过 `"组名相关主题：词1, 词2, ..."` 文本 embedding 得到，基于词元哈希变更检测增量更新。

### 3.4 时间感知检索 (TimeExpressionParser)

**双层匹配架构**：
1. **硬编码表达式**（从长到短排序避免歧义）：`"昨天"→减1天`，`"上周"→上周边界`
2. **动态正则**：`"N天前"`、`"上周N"`、`"N个月前"` 等模式

特色：支持中文数字转换（`"二十三"→23`）、支持多时间点查询（`"上周和三个月前"`）、自动去重重叠时间范围。

### 3.5 元思考递归推理链 (MetaThinkingManager)

**核心概念**：定义多个"思维簇"（知识库分区），配合K值序列，进行多阶段递归检索。

**流程**：
```
Auto模式：queryVector vs 各主题向量 → 选择最匹配的推理链
Stage 1: search(cluster1, queryVector, k1) → results1
         向量融合: currentQuery = 0.8*original + 0.2*avg(results1)
Stage 2: search(cluster2, currentQuery, k2) → results2
         向量融合: ...
Stage N: search(clusterN, currentQuery, kN) → resultsN
```

关键设计：每阶段的检索结果向量与原始查询向量按 **0.8:0.2** 比例融合，形成"渐进式语义漂移"，让后续阶段能发现更深层的关联。

### 3.6 Rerank + RRF 融合排序

**标准 Rerank**：超量获取（K × multiplier）→ 外部 Reranker 模型精排 → 取 top-K。

**Rerank+ (RRF 融合)**：
```
RRF(d) = α × 1/(60 + rerank_rank) + (1-α) × 1/(60 + retrieval_rank)
```
α 可通过 `::Rerank+0.7` 语法控制（默认0.5），K=60 是业界标准平滑常数。

**容错机制**：断路器模式（1分钟内5次失败则跳过）、查询截断（预留70%给文档）、批次失败率>50%提前终止。

### 3.7 时间衰减重排 (TimeDecay)

```
Score_new = Score_original × 0.5^(days / halfLife)
```

支持精准打击：通过 `::TimeDecay30|0.5|Wiki,技巧` 语法指定半衰期、最低分数阈值和目标标签白名单。

### 3.8 AI 驱动记忆召回 (AIMemoHandler)

将日记全文交给独立的 AI 模型（低温度 0.3），让 AI 判断哪些记忆片段与当前对话相关。支持：
- **跨库联合检索**：多日记本聚合
- **贪心分批**：按 token 限制切分文件批次
- **重复检测**：发现 AI 输出中的循环重复文本块并截断
- **三级降级提取**：标准格式 → 提取所有 `[[]]` 块 → 包装全文

---

## 四、关键代码精要

### 上下文分段算法（霰弹枪查询阵列基础）

```js
// 连续消息的 cosine > 0.70 → 合入同一段落
// 否则断开，每段计算平均向量并L2归一化
segmentContext(messages, threshold = 0.70) {
    // 遍历有序消息序列
    for (let i = 1; i < sequence.length; i++) {
        const sim = cosineSimilarity(prev.vector, curr.vector);
        sim >= threshold ? merge() : split();
    }
    // 每段输出: { vector(均值归一化), range, count }
}
```

### 上下文日记去重

```js
// 扫描 assistant 消息中的 DailyNote create 工具调用
// 提取 Content 前80字符作为前缀索引
// 在 RAG 结果中过滤掉前缀匹配的条目（避免重复召回刚写入的日记）
```

---

## 五、可借鉴到 legal-assistant 的设计点

1. **L/R/S 三指标动态参数体系**：可直接适配到律脉的检索参数调控（DRC模块），用逻辑深度和语义宽度动态调节 K 值和标签权重。

2. **上下文衰减聚合**：`decayRate=0.75` 的指数衰减聚合可用于法律咨询的多轮对话场景，自动降低早期对话的影响权重。

3. **语义组增强检索**：可为法律领域创建"劳动法争议"、"合同纠纷"等语义组，将分散的法律概念关联起来增强检索精度。

4. **标签截断策略**：动态截断尾部噪音标签（保留率由 L/S 决定），可防止法条标签过多导致检索漂移。

5. **Rerank+RRF 融合**：结合向量检索排位和语义精排排位，对法律文本这类精确性要求高的场景特别有价值。

6. **断路器容错模式**：Rerank API 的断路器设计可直接应用于律脉的外部 API 调用保护。

7. **上下文日记去重**：前缀匹配去重思路可适配到法律知识库的增量更新场景，避免重复召回刚入库的内容。
