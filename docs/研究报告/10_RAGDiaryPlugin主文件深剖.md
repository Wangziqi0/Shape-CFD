# 10_RAGDiaryPlugin 主文件深度剖析

> Agent B2 第二轮深挖报告 | 文件：`RAGDiaryPlugin.js` (157KB, 3242行)
> 补充第一轮报告 06 未覆盖的核心算法实现细节

---

## 一、整体架构

RAGDiaryPlugin 是 VCPToolBox 中最庞大的单个插件，作为 `MessagePreprocessor` 挂载在消息管线中。其核心职责是：**在 LLM 收到消息前，扫描 system prompt 中的占位符，按语义检索注入相关日记记忆片段**。

### 数据流全景

```
用户消息 → processMessages() 入口
  ↓
[1] 上下文向量更新 (ContextVectorManager.updateContext)
[2] 扫描所有 system 消息识别四种占位符
[3] 提取最后 user/assistant 消息 → 净化(HTML/Emoji/工具标记/系统通知)
[4] 组合上下文统一向量化 → queryVector
[5] 动态参数计算 (_calculateDynamicParams) → K, TagWeight, TruncationRatio
[6] 历史分段提取 (segmentContext) → historySegments（用于霰弹枪）
[7] 时间表达式解析 (TimeExpressionParser)
[8] 上下文日记去重前缀提取
[9] 循环处理每个 system 消息中的占位符（并行 Promise.all）
  ↓
返回注入记忆后的消息数组
```

---

## 二、四种占位符模式

| 占位符 | 格式示例 | 行为 | 阈值判断 |
|--------|---------|------|---------|
| `[[]]` | `[[物理日记本::TagMemo::Rerank]]` | **RAG 向量检索**——核心模式 | ❌ 无阈值，直接检索 |
| `<<>>` | `<<物理日记本>>` | **全文召回**——整个日记本内容注入 | ✅ 余弦相似度 ≥ threshold |
| `《《》》` | `《《物理日记本::TagMemo》》` | **混合模式**——阈值门控的 RAG 检索 | ✅ 余弦相似度 ≥ threshold |
| `{{}}` | `{{物理日记本}}` | **直接引入**——无检索、无阈值 | ❌ 无条件注入全文 |

### 聚合语法

所有占位符均支持管道符 `|` 实现**多日记本聚合检索**：

```
[[物理|政治|python日记本:1.2::TagMemo::Rerank+0.7]]
```

解析为：`diaryNames=['物理','政治','python']`，`kMultiplier=1.2`。

### 修饰符完整列表

| 修饰符 | 功能 | 参数 |
|--------|------|------|
| `::TagMemo` / `::TagMemo0.3` | 启用 Tag 增强检索 | 可选权重(默认从动态参数获取) |
| `::Time` | 启用时间感知双路召回 | 自动从文本解析时间范围 |
| `::Group` | 启用语义组增强 | 关键词激活语义组 |
| `::Rerank` | 启用 Reranker 精排 | 自动扩大检索量(×multiplier) |
| `::Rerank+` / `::Rerank+0.7` | RRF 融合排序 | α值(默认0.5) |
| `::TimeDecay30\|0.5\|Wiki` | 时间衰减重排 | halfLife\|minScore\|targetTags |
| `::AIMemo` / `::AIMemo:preset` | AIMemo 语义推理 | 需 `[[AIMemo=True]]` 许可证 |
| `::RoleValve:...` | 角色门控 | 条件表达式 |
| `:1.5` (前缀) | K 值倍率 | 浮点数 |

---

## 三、核心算法详解

### 3.1 动态参数计算（V3）

**函数**: `_calculateDynamicParams(queryVector, userText, aiText)` (L431)

从 EPA 和上下文向量管理器获取三个信号，通过 Sigmoid 映射为 RAG 参数：

```
输入:
  L = EPA.logicDepth       (0~1，逻辑深度，高=意图明确)
  R = EPA.resonance        (0~∞，跨域共振，高=多领域关联)
  S = SemanticWidth        (0~∞，语义宽度，高=话题发散)

→ Beta(TagWeight):
  β_input = L × log(1 + R + 1) - S × noise_penalty(默认0.05)
  β = σ(β_input)  // Sigmoid
  TagWeight = range[0] + β × (range[1] - range[0])  // 默认 [0.05, 0.45]

→ Dynamic K:
  k_adj = round(L × 3 + log1p(R) × 2)
  K = clamp(k_base + k_adj, 3, 10)

→ TruncationRatio (Tag截断比例):
  ratio = base(0.6) + L×0.3 - S×0.2 + min(R,1)×0.1
  ratio = clamp(ratio, 0.5, 0.9)
```

**设计思路**：逻辑深度高→意图明确→更多Tag/更多K覆盖；语义宽度大→噪音多→收紧Tag权重和截断。

### 3.2 霰弹枪查询阵列（Shotgun Query, V4）

**位置**: `_processRAGPlaceholder` L2170-2253

**核心思想**：不再仅用当前查询向量检索，而是同时用最近 N 个历史对话分段的向量并行检索，实现上下文连续性。

```
Step 1: 构建搜索向量集
  searchVectors = [
    { vector: currentQueryVector,     type: 'current',   weight: 1.0 },
    { vector: segment[-1].vector,     type: 'history_2', weight: 0.85^1 = 0.85 },
    { vector: segment[-2].vector,     type: 'history_1', weight: 0.85^2 = 0.72 },
    { vector: segment[-3].vector,     type: 'history_0', weight: 0.85^3 = 0.61 }
  ]
  // 最多取最近 3 个历史分段

Step 2: 并行检索 (Promise.all)
  current → search(dbName, vector, kForSearch, tagWeight, coreTags)
  history → search(dbName, vector, max(2, kForSearch/2), tagWeight, coreTags)
  // 历史分段检索量减半，节约资源

Step 3: 合并 + 衰减得分
  history 结果的 score *= weight  (近因效应：越远越弱)

Step 4: 上下文日记去重 → SVD 智能去重 → Rerank(可选) → 截断
```

### 3.3 聚合检索与 Softmax K 值分配

**函数**: `_processAggregateRetrieval` (L1747)

当占位符包含 `|` 管道符聚合多个日记本时：

```
Step 1: 获取每个日记本向量 → 计算与 queryVector 的余弦相似度
Step 2: Softmax 温度分配 K 值
  exp_i = exp(similarity_i × temperature)  // temperature 默认 3.0
  weight_i = exp_i / Σ(exp)
  k_i = minK + round(distributableK × weight_i)  // minK 默认 1
Step 3: 并行调用各子日记本的 _processRAGPlaceholder
Step 4: 拼接所有日记本的检索结果
```

**温度参数的意义**：temperature 越高 → 相似度差异被放大 → K 分配越集中在高相似度日记本。

### 3.4 Rerank + RRF 融合排序

**函数**: `_rerankDocuments(query, documents, k, rrfOptions)` (L2601)

```
Step 1: Token 预算管理
  maxQueryTokens = maxTokens × 0.3 (预留70%给文档)
  查询过长 → 按比例截断

Step 2: 按 Token 预算切分 Batch
  每批不超过 maxTokens，至少1个文档

Step 3: 顺序调用 Rerank API (v1/rerank)
  断路器: 1分钟内累计5次失败 → 自动熔断，跳过 Rerank
  高失败率: >50%失败且>2批 → 提前终止

Step 4a (标准 Rerank):
  按 rerank_score 降序排列 → 取前 K

Step 4b (Rerank+ RRF 融合):
  RRF_K = 60 (平滑常数)
  rrf_score(d) = α × 1/(60 + rerank_rank)
               + (1-α) × 1/(60 + retrieval_rank)
  按 rrf_score 降序排列 → 取前 K
```

### 3.5 时间衰减重排（TimeDecay）

启用 `::TimeDecay` 修饰符后（可选参数 `halfLife|minScore|targetTags`）：

```
对每条检索结果:
  1. 检查是否在目标标签名单内（精准打击）
  2. 从文本/文件名提取日期 [YYYY-MM-DD]
  3. diffDays = now - entryDate
  4. decayFactor = 0.5^(diffDays / halfLife)  // 默认 halfLife=30
  5. newScore = originalScore × decayFactor
  6. 按 newScore 重排序，过滤 < minScore 的结果
```

### 3.6 平衡双路召回（Time 模式, V5）

启用 `::Time` 修饰符时，语义和时间各占比例：

```
kSemantic = ceil(K × 0.6)   // 语义路 60%
kTime     = K - kSemantic   // 时间路 40%

语义路: vectorDBManager.search(dbName, vector, kSemantic)
时间路: 
  1. 扫描日记文件首行提取日期 → 筛选时间范围内的文件路径
  2. 获取这些文件的所有分块及向量
  3. 计算每个分块与 queryVector 的余弦相似度
  4. 按相似度排序取前 kTime (时间路也进行相关性排序!)

合并: 语义路优先入 Map → 时间路补充（文本去重）
```

### 3.7 上下文日记去重（V4.1）

**问题**：AI 在当前对话中刚写入的日记，下一轮可能被 RAG 召回造成重复。

```
Step 1: 扫描所有 assistant 消息中的 TOOL_REQUEST 块
  → 提取 tool_name=DailyNote, command=create 的 Content 字段
  → 取前 80 字符作为前缀索引，存入 Set

Step 2: 对检索结果执行前缀匹配过滤
  → 跳过日记头 "[YYYY-MM-DD] - name\n"
  → 取 body 前 80 字符与 Set 中所有前缀比较
  → compareLen > 10 且前缀完全匹配 → 过滤掉
```

---

## 四、三级缓存系统

| 缓存层 | 键生成 | 容量 | TTL | 淘汰策略 |
|--------|--------|------|-----|---------|
| 查询结果缓存 | SHA256(user+ai+db+modifiers+k+date) | 200 (可配置) | 1h | LRU(删Map首元素) |
| 向量缓存 | SHA256(text.trim()) | 500 | 2h | LRU |
| AIMemo缓存 | (由AIMemoHandler管理) | 50 | 30min | 定期清理 |

**缓存失效条件**：
1. TTL 过期（定期扫描 + 读取时验证）
2. `rag_tags.json` 配置文件哈希变化 → 清空全部查询缓存
3. Time 模式的缓存键包含当前日期 → 次日自动失效

---

## 五、文本净化管线

在向量化前对 userContent/aiContent 执行 4 层净化：

```
原始文本
  → _stripSystemNotification  // 移除 [系统通知]...[系统通知结束]
  → _stripHtml               // cheerio 解析，移除 style/script，提取纯文本
  → _stripEmoji              // 正则移除 15 类 Unicode emoji 范围
  → _stripToolMarkers        // 提取工具调用块中的有效语义内容，过滤技术标记
```

`_stripToolMarkers` 的精妙之处：不是简单删除工具调用，而是**提取有语义价值的字段内容**（过滤 tool_name/command 等技术键），保留用户真正想表达的信息。

---

## 六、与第一轮报告的关联补充

| 第一轮报告 | 本轮补充 |
|-----------|---------|
| 06 提到 RAGDiaryPlugin 架构 | 逐函数级剖析了八大核心算法的完整实现 |
| 02 提到 KnowledgeBaseManager 的 search 接口 | 本文揭示了调用侧的完整策略（霰弹枪、聚合、双路召回） |
| 01 提到 ResultDeduplicator | 本文展示了去重在管线中的确切位置（霰弹枪合并后、Rerank之前） |
| 06 提到 ContextVectorManager | PSR偏振语义舵的具体实现在 ContextVectorManager 子模块中，本文涉及其接口调用 |

---

## 七、可借鉴点（for legal-assistant）

| # | 借鉴点 | legal-assistant 适用场景 |
|---|--------|------------------------|
| 1 | **霰弹枪查询+衰减权重** | 多轮法律咨询中保持上下文连续性 |
| 2 | **Softmax 温度分配 K 值** | 跨法律领域聚合检索时的资源分配 |
| 3 | **RRF 融合排序** | 结合向量检索排位和精排模型，比单一排序更鲁棒 |
| 4 | **TimeDecay 精准打击** | 法律时效性场景（如诉讼时效、法规更新） |
| 5 | **四种占位符模式** | 灵活的记忆注入策略（全文/RAG/混合/直接），适配不同法律场景 |
| 6 | **三级 LRU 缓存** | 减少重复向量化开销，SHA256 键保证稳定性 |
| 7 | **上下文日记去重（前缀匹配）** | 防止刚生成的法律分析被重复召回 |
| 8 | **文本净化管线** | 法律文书中的 HTML/特殊格式清理 |
| 9 | **断路器模式** | Rerank API 异常时自动降级，保证系统可用性 |
| 10 | **动态参数计算 EPA+Sigmoid** | 自适应调节检索策略，替代硬编码参数 |
