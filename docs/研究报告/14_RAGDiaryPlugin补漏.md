# 14 — RAGDiaryPlugin 逐行补漏报告

> **Agent C1** | 补漏对象：`RAGDiaryPlugin.js`（3242行, 157KB）  
> 对照报告：06、10 | 报告日期：2026-03-13

---

## 一、方法覆盖对照表

| 已覆盖（06/10）| 本轮新发现 |
|---|---|
| `_calculateDynamicParams` 概要 | ✅ 完整公式拆解 |
| `_processRAGPlaceholder` 概要 | ✅ Shotgun Query + TimeDecay + 双路召回  |
| `_rerankDocuments` 提及 | ✅ RRF 融合公式 + 断路器模式 |
| 霰弹枪查询概要 | ✅ 衰减权重 + 并行策略详解 |
| — | ✅ `_evaluateRoleValve` 角色门控表达式引擎 |
| — | ✅ `_processAggregateRetrieval` Softmax K 分配 |
| — | ✅ 四种占位符完整分发逻辑 |
| — | ✅ `_truncateCoreTags` 尾部噪音截断 |
| — | ✅ `_filterContextDuplicates` 上下文去重 |
| — | ✅ `_stripToolMarkers` 工具调用净化器 |
| — | ✅ `_jsonToMarkdown` 嵌套 JSON 递归转换 |
| — | ✅ LRU 缓存系统（SHA-256 键 + TTL + 命中率统计）|
| — | ✅ `refreshRagBlock` 区块刷新（U:0.5, A:0.35, T:0.15） |

---

## 二、新发现算法详解

### 2.1 V3 动态参数计算（`_calculateDynamicParams`）

**核心公式**：
```
β = sigmoid(L × log(1 + R + 1) - S × noise_penalty)
tagWeight = 0.05 + β × 0.40                    // 范围 [0.05, 0.45]
K = clamp(k_base + round(L×3 + log1p(R)×2), 3, 10)
truncationRatio = clamp(0.6 + L×0.3 - S×0.2 + min(R,1)×0.1, 0.5, 0.9)
```

其中 L=logicDepth, R=resonance, S=semanticWidth（来自 ContextVectorManager）。

### 2.2 RoleValve 角色门控（`_evaluateRoleValve`）

语法：`::RoleValve@User>3&Assistant>=2|System`

- 统计消息中各角色（User/Assistant/System）的数量
- 支持比较运算符：`<`, `>`, `<=`, `>=`, `=`
- 支持逻辑组合：`&`（AND）、`|`（OR）
- 优先级：单条件 > AND > OR
- 用于控制"对话初期不激活某些重型检索"等场景

### 2.3 四种占位符模式的完整语义

| 占位符 | 阈值门控 | 检索方式 | 支持聚合 | 用途 |
|--------|---------|---------|---------|------|
| `[[X日记本]]` | **无** | 向量 RAG | ✅ `\|`分隔 | 标准语义检索 |
| `<<X日记本>>` | **余弦阈值** | 全文召回 | ❌ | 相关时注入全部日记内容 |
| `《《X日记本》》` | **余弦阈值** | 混合（先门控再RAG）| ✅ | 按需激活的语义检索 |
| `{{X日记本}}` | **无** | 直接注入全文 | ❌ | 强制注入（如角色设定） |

所有模式均支持 `::RoleValve`、`::AIMemo`、循环引用检测。

### 2.4 聚合检索 Softmax K 分配（`_processAggregateRetrieval`）

```
expScore[i] = exp(similarity[i] × temperature)    // temperature = 3.0
weight[i] = expScore[i] / Σ(expScore)
K[i] = minKPerDiary + round(distributableK × weight[i])
```

> 每个子日记本**并行独立检索**，结果拼接返回。

### 2.5 Shotgun Query 多向量并行检索

```
searchVectors = [当前查询向量(weight=1.0)]
                + 最近3个历史分段向量(weight = 0.85^distance)
```

- 当前查询用完整 K 检索，历史分段用 `K/2`
- 结果 score × weight 实现**近因效应**
- 合并后执行 SVD 去重（TagMemo V4），再 Rerank

### 2.6 平衡双路召回（Time 模式）

```
kSemantic = ceil(K × 0.6)   // 语义路 60%
kTime = K - kSemantic         // 时间路 40%
```

时间路：先获取时间范围内的文件，提取所有 chunk → 计算余弦相似度 → 按分数排序取 TopK。

### 2.7 TimeDecay 时间衰减重排

```
decayFactor = 0.5 ^ (diffDays / halfLife)
newScore = originalScore × decayFactor
```

- 支持修饰符语法：`::TimeDecay30|0.5|Wiki,技巧`
- 支持精准打击目标标签（只衰减特定标签的条目）
- 日期提取：优先从文本 `[YYYY-MM-DD]` 提取，回退到文件路径

### 2.8 Rerank+ RRF 融合（`_rerankDocuments`）

```
RRF(d) = α × 1/(60 + rerank_rank) + (1-α) × 1/(60 + retrieval_rank)
```

- K=60 为业界标准平滑常数
- α 默认 0.5，可通过 `::Rerank+0.7` 调节
- **断路器模式**：1分钟内 5 次失败 → 跳过 Rerank
- **批次处理**：按 token 预算分批调用 Rerank API
- **查询截断**：超过 `maxTokens×30%` 时自动截断

### 2.9 上下文日记去重（V4.1）

`_extractContextDiaryPrefixes`：扫描 assistant 消息中的 DailyNote create 工具调用，提取 Content 前 80 字符作为去重索引。`_filterContextDuplicates`：召回结果中如果前 80 字符匹配上下文已写入的日记，则过滤。

### 2.10 工具调用净化器（`_stripToolMarkers`）

从 AI 回复中移除 `<<<[TOOL_REQUEST]>>>` 标记块，提取自然语言内容，过滤技术键名（tool_name/command 等），防止英文偏好噪音干扰向量搜索。

### 2.11 LRU 缓存系统

- **缓存键**：SHA-256(JSON(userContent, aiContent, dbName, modifiers, dynamicK, ...))
- **TTL**：可配置过期时间，定期清理
- **LRU 淘汰**：超过 maxCacheSize 时删除最早条目
- **命中率统计**：`cacheHits / (cacheHits + cacheMisses)`
- **Time 模式**：缓存键包含当前日期，保证每日刷新

### 2.12 RAG 区块刷新（`refreshRagBlock`）

当 LLM 进入工具循环时，可以刷新之前注入的 RAG 区块：
- 用新的上下文重新构建查询向量
- 权重分配：User 0.5, AI 0.35, Tool 0.15

---

## 三、已确认无遗漏的区域

| 区域 | 状态 |
|------|------|
| `cosineSimilarity` / `_getWeightedAverageVector` / `_getAverageVector` | 标准数学工具，无算法 |
| `getDiaryContent` | 简单文件读取 |
| `_stripHtml` / `_stripEmoji` / `_stripSystemNotification` | 文本清洗，无算法 |
| `_isLikelyBase64` | 启发式检测，无复杂算法 |
| `formatStandardResults` / `formatCombinedTimeAwareResults` / `formatGroupRAGResults` | 纯格式化 |
| `_estimateTokens` | 简单 token 估算 |

---

## 四、对 legal-assistant 的补充借鉴

| 新发现 | 法律 RAG 应用 |
|--------|-------------|
| RoleValve 角色门控 | 控制对话早期不激活重型法规检索，节省 token |
| 四种占位符模式 | 法律 RAG 可设计类似：`[[法规库]]`（语义检索）、`{{判例库}}`（强制注入）、`《《法条库》》`（按需激活） |
| Softmax K 分配 | 多知识库（刑法+民法+行政法）查询时自动分配检索配额 |
| TimeDecay | 法律法规的时效性处理——新法优先于旧法 |
| RRF 融合 | 综合向量检索排序和交叉编码器精排的结果 |
| 断路器模式 | Rerank API 故障时优雅降级 |
| 上下文去重 | 避免重复注入已在对话中出现的法条 |

---

> **结论**：前两轮报告覆盖了 RAGDiaryPlugin 的核心框架，但遗漏了大量辅助算法。本轮补漏新发现 **12 个独立算法/机制**，其中 V3 动态参数计算、RoleValve 角色门控、Softmax K 分配、RRF 融合和 TimeDecay 时间衰减是最有借鉴价值的新发现。
