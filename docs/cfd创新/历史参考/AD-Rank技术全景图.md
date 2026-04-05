# AD-Rank 技术全景图

> 2026-03-24 | 一凡 + Antigravity 研究讨论产出
> ⚠️ **注意**：此文档为早期设计阶段产物。当前最优方案已演化为 Shape-CFD V7.1 (PQ-Chamfer 64×64 + 伴随状态法预取，NDCG@10 = 0.2802)。最新信息请参考 `V5灵感演化路线图.md` 和 `HANDOFF.md`。

---

## 一、总体架构

```mermaid
graph TB
    subgraph INPUT["输入层"]
        Q["query向量 (d维)"]
        D["候选文档向量 (N×d维)"]
    end

    subgraph GRAPH["图构建层"]
        G1["两两 cosine similarity (原始d维)"]
        G2["KNN 取 k=5-10 近邻"]
        G3["稀疏邻接矩阵 W"]
        G4["图拉普拉斯 L = D_deg - W"]
    end

    subgraph INIT["场量初始化"]
        I1["浓度 C₀ = cosine(query, doc_i)"]
        I2["对流方向 u = f(query, graph)"]
        I3["扩散系数 D = 常数 or 自适应"]
    end

    subgraph SOLVE["求解器"]
        S1["迭代法: C_{t+1} = C_t + Δt(-u·∇C + D·L·C)"]
        S2["OR 谱方法: ĉ = H(Λ,u)·Φᵀ·C₀"]
    end

    subgraph ANALYSIS["流场分析"]
        A1["汇聚点检测"]
        A2["发散点检测"]
        A3["停滞点检测"]
        A4["Re = 对流/扩散 比值"]
    end

    subgraph OUTPUT["输出层"]
        O1["排序: 按 C_final 排序 → Top-k"]
        O2["拓扑标签: 汇聚/发散/停滞"]
        O3["Re 值 → 模型选择/Prompt 策略"]
    end

    Q --> G1
    D --> G1
    G1 --> G2 --> G3 --> G4

    Q --> I1
    Q --> I2
    G4 --> I3

    I1 --> S1
    I2 --> S1
    I3 --> S1
    G4 --> S1
    I1 --> S2
    I2 --> S2
    G4 --> S2

    S1 --> A1 & A2 & A3 & A4
    S2 --> A1 & A2 & A3 & A4

    A1 & A2 & A3 --> O2
    A4 --> O3
    S1 --> O1
    S2 --> O1

    style S1 fill:#ffd93d,stroke:#333
    style S2 fill:#ffd93d,stroke:#333
    style I2 fill:#ff6b6b,stroke:#333
    style I3 fill:#ffb347,stroke:#333
```

---

## 二、每个模块的技术细节与开放问题

### 🟢 已确定

```mermaid
graph LR
    subgraph DECIDED["已确定的设计决策"]
        D1["不降维: 在原始d维计算相似度"]
        D2["不用N-S: 用对流-扩散方程"]
        D3["不用网格: 用图上离散化"]
        D4["通用化: 适用于任何embedding"]
        D5["核心方程: ∂C/∂t + u·∇C = D∇²C"]
    end
    style DECIDED fill:#d4edda,stroke:#333
```

### 🔴 开放问题（需要灵感）

```mermaid
graph TB
    subgraph OPEN["开放问题"]
        A["A: 对流方向u怎么定义?"]
        B["B: 扩散系数D怎么设?"]
        C["C: 谱方法 vs 迭代法?"]
        DD["D: 多轮对话怎么处理?"]
        E["E: 对流项真的有用吗?"]
        F["F: 能不能学习参数?"]
        G["G: 量化分块思路怎么融入?"]
    end

    A -->|决定| CORE["算法效果的上限"]
    E -->|验证| CORE
    F -->|演进| FUTURE["下一篇论文"]
    DD -->|演进| FUTURE
    G -->|增强| A

    style A fill:#ff6b6b,stroke:#333
    style E fill:#ff6b6b,stroke:#333
    style F fill:#ff6b6b,stroke:#333
    style G fill:#ff6b6b,stroke:#333
    style B fill:#ffb347,stroke:#333
    style C fill:#ffb347,stroke:#333
    style DD fill:#ffb347,stroke:#333
```

---

## 三、问题 A 展开：对流方向 u 的 4 种方案

```mermaid
graph TB
    subgraph METHODS["u 的定义方式"]
        M1["① query投影<br>u_ij = proj(doc_j - doc_i, query)<br>✅简单 ❌只管方向"]
        M2["② 梯度场<br>u = ∇sim(query, x)<br>✅每点独立 ❌需连续假设"]
        M3["③ 注意力加权<br>u = attention(query, i→j)<br>✅最智能 ❌要训练"]
        M4["④ 量化分块<br>u按维度块加权<br>✅query-adaptive ❌设计复杂"]
    end

    M1 -->|v1 先实现| MVP["最小可行版本"]
    M2 -->|v2 如果①不够好| MVP
    M4 -->|一凡的灵感| NEW["新方向"]
    M3 -->|需要数据| FUTURE2["远期"]

    style M1 fill:#d4edda
    style M4 fill:#ffd93d
    style NEW fill:#ffd93d
```

---

## 四、问题 G 展开：量化分块思路

```mermaid
graph LR
    subgraph QUANT["FP8→FP4 量化类比"]
        Q1["4096维向量"]
        Q2["分成64块 × 64维"]
        Q3["每块算query重要性 w_i"]
        Q4["加权cosine:<br>sim = Σ w_i × cos(a_i, b_i)"]
        Q5["图的结构随query变化"]
    end

    Q1 --> Q2 --> Q3 --> Q4 --> Q5

    Q5 -->|影响| GRAPH2["建图: 相似度矩阵变了"]
    Q5 -->|影响| CONV["对流: 方向也变了"]

    style Q3 fill:#ffd93d
    style Q5 fill:#ff6b6b
```

---

## 五、竞品对比总览

```mermaid
graph TB
    subgraph US["AD-Rank (我们)"]
        U1["对流-扩散方程"]
        U2["方向性检索"]
        U3["流场拓扑分析"]
        U4["雷诺数调参"]
        U5["通用向量检索"]
    end

    subgraph COMP["竞品"]
        C1["Diffusion-Aided RAG<br>纯扩散, 无方向"]
        C2["G-RAG<br>GNN重排, 非物理方程"]
        C3["SLoD<br>热核+庞加莱, 非检索"]
        C4["Cross-Encoder<br>SOTA精排, 计算贵"]
        C5["Cosine Similarity<br>基线, 无图结构"]
    end

    C5 -->|"+图结构"| C1
    C1 -->|"+对流项"| U1
    U1 -->|"+拓扑分析"| U3
    U3 -->|"+Re调参"| U4

    C2 -.->|"不同方法论"| U1
    C3 -.->|"理论参照"| U1
    C4 -.->|"效果对标"| U5

    style U1 fill:#6bcb77
    style U2 fill:#6bcb77
    style U3 fill:#6bcb77
    style C1 fill:#ffd93d
```

---

## 六、实现路线图

```mermaid
gantt
    title AD-Rank 实现路线
    dateFormat  YYYY-MM-DD
    section 第一阶段: MVP
    核心算法(建图+迭代+排序)     :a1, 2026-03-25, 2d
    u方案①实现                  :a2, after a1, 1d
    section 第二阶段: 验证
    三组对比实验(A/B/C)          :b1, after a2, 3d
    消融实验(u/D/k/步数)        :b2, after b1, 2d
    section 第三阶段: 增强
    流场拓扑分析                 :c1, after b2, 3d
    可视化demo                  :c2, after c1, 2d
    section 第四阶段: 演进
    量化分块加权(灵感G)          :d1, after c2, 3d
    多轮对话(问题D)              :d2, after d1, 3d
    PINN参数学习(问题F)          :d3, after d2, 5d
```

---

## 七、一句话速查表

| 编号 | 问题 | 当前状态 | 你的灵感可以改变什么 |
|:---:|:---|:---:|:---|
| **A** | 对流方向 u | 🔴 待定 | 量化分块思路可以让 u 变成 query-adaptive |
| **B** | 扩散系数 D | ✅ 已实现 | Fiedler value 自适应 D (Agent 9) |
| **C** | 求解方法 | 🟡 先迭代法 | 谱方法是性能升级路径 |
| **D** | 多轮对话 | 🟡 语义动量 | 流场记忆效应 |
| **E** | 有效性验证 | ✅ 已完成 | DeepSeek 盲评 + 全策略横评确认 v2 最优 |
| **F** | 参数学习 | 🔴 远期 | PINN 式训练 |
| **G** | 量化分块 | 🔴 新灵感 | 和 A 结合，改变建图方式 |
