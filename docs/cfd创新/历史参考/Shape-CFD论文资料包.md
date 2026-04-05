# Shape-CFD：基于形状感知对流-扩散方程的向量检索重排序算法

## 论文资料包 — 完整数学推导 · 实验数据 · 核心流程

> 作者：陈一凡 (Yifan Chen)
> 日期：2026-03-29
> 版本：v2.0
> 用途：论文撰写参考资料，不直接作为投稿稿件

---

# 第一部分：数学理论推导

## 1. 问题形式化

### 1.1 检索重排序问题定义

给定：
- 查询向量 $q \in \mathbb{R}^d$（由预训练语言模型编码，$d=4096$）
- 候选文档集 $\mathcal{D} = \{d_1, d_2, ..., d_N\}$，其中每个文档已编码为 $v_i \in \mathbb{R}^d$
- 候选集由 HNSW 近邻检索的 top-$N$ 结果构成（$N=30$）

目标：学习一个排序函数 $f: \mathcal{D} \times q \to \mathbb{R}^N$，使得排序结果优于原始 cosine similarity 排序。

### 1.2 现有方法的局限

**Cosine 直排**：

$$\text{score}(d_i) = \frac{q \cdot v_i}{\|q\| \|v_i\|}$$

每个文档独立打分，忽略文档间的语义关联。当 query 存在多义性或候选文档形成语义聚类时，孤立打分无法利用文档间的互信息。

**Cross-Encoder 重排序**：

$$\text{score}(d_i) = \text{BERT}([q; d_i])$$

质量高但计算开销 $O(N \cdot L^2 \cdot H)$（$L$ 为序列长度，$H$ 为隐藏维度），$N=30$ 时约 500ms。

---

## 2. AD-Rank 核心数学框架

### 2.1 图构建

**Step 1: 相似度矩阵**

$$S_{ij} = \cos(v_i, v_j) = \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}, \quad \forall i, j \in \{1, ..., N\}$$

**Step 2: KNN 稀疏化**

对每个节点 $i$，仅保留相似度最高的 $k$ 个邻居：

$$\mathcal{N}(i) = \text{argTopK}_{j \neq i}(S_{ij}, k), \quad k=3$$

**Step 3: 边权重**

$$W_{ij} = \exp(-\beta \cdot (1 - S_{ij})), \quad \beta = 2.0$$

权重随相似度指数衰减，$\beta$ 控制衰减速率。

**Step 4: 对称化**

$$\text{若 } j \in \mathcal{N}(i) \text{ 而 } i \notin \mathcal{N}(j)，\text{则添加 } W_{ji} = W_{ij}$$

确保无向图性质，为扩散算子的对称正定性提供保证。

### 2.2 初始浓度场

$$C_0(i) = \cos(q, v_i), \quad i = 1, ..., N$$

**物理解释**：初始浓度等于文档与 query 的原始相关度，作为对流-扩散过程的初始条件。

> **注意**：cosine 值域为 $[-1, 1]$，可能产生负浓度，在物理上不自洽。Shape CFD 扩展（见 3.4 节）通过 $C_0(i) = \exp(-2 \cdot d_{\text{Chamfer}})$ 解决此问题——指数函数天然非负，满足 PDE 浓度场 $C \geq 0$ 的边界条件。

### 2.3 对流方向推导

对每条边 $(i, j)$，对流系数定义为：

$$u_{ij} = \left\langle \frac{v_j - v_i}{\|v_j - v_i\|_2 + \varepsilon}, \frac{q}{\|q\|} \right\rangle \cdot u_{\text{strength}}, \quad \varepsilon = 10^{-8}$$

其中 $\varepsilon$ 防止同质法条（$v_i \approx v_j$）导致的除零异常。

**推导过程**：

1. 边方向向量：$\vec{e}_{ij} = v_j - v_i$（从 $i$ 指向 $j$）
2. 归一化：$\hat{e}_{ij} = \frac{\vec{e}_{ij}}{\|\vec{e}_{ij}\| + \varepsilon}$（$\varepsilon = 10^{-8}$，防除零）
3. Query 方向：$\hat{q} = \frac{q}{\|q\|}$
4. 投影：$u_{ij} = (\hat{e}_{ij} \cdot \hat{q}) \cdot u_{\text{strength}}$

**物理含义**：$u_{ij} > 0$ 表示从 $i$ 到 $j$ 的方向与 query 方向一致（"顺风"），$u_{ij} < 0$ 表示逆向（"逆风"）。

**反对称性证明**：

$$u_{ji} = \left\langle \frac{v_i - v_j}{\|v_i - v_j\|}, \hat{q} \right\rangle \cdot u_{\text{strength}} = \left\langle -\frac{v_j - v_i}{\|v_j - v_i\|}, \hat{q} \right\rangle \cdot u_{\text{strength}} = -u_{ij}$$

因为 $v_i - v_j = -(v_j - v_i)$，且 $\|v_i - v_j\| = \|v_j - v_i\|$。$\blacksquare$

### 2.4 守恒型 Upwind 格式

**连续形式（对流-扩散方程）**：

$$\frac{\partial C}{\partial t} = D \nabla^2 C - \nabla \cdot (\vec{u} C)$$

**图离散化**：

$$C_{t+1}(i) = C_t(i) + \Delta t \cdot \left[ \underbrace{D \sum_{j \in \mathcal{N}(i)} W_{ij} (C_t(j) - C_t(i))}_{\text{扩散通量}} + \underbrace{\sum_{j \in \mathcal{N}(i)} W_{ij} [\max(u_{ji}, 0) \cdot C_t(j) - \max(u_{ij}, 0) \cdot C_t(i)]}_{\text{守恒型 upwind 对流通量}} \right]$$

**为什么必须用 Upwind 而非中心差分？**

中心差分格式 $u_{ij} W_{ij} (C_j - C_i)$ 在 $\text{Pe} > 2$ 时产生数值振荡。Upwind 格式通过选取上游节点的浓度值来保证传输方向的因果性：

- $u_{ij} > 0$（$i \to j$ 方向）：从 $i$ 的浓度出发 → $\max(u_{ij}, 0) \cdot C_i$
- $u_{ji} > 0$（$j \to i$ 方向）：从 $j$ 的浓度出发 → $\max(u_{ji}, 0) \cdot C_j$

**全局守恒证明**：

$$\sum_{i=1}^{N} F_i^{\text{adv}} = \sum_{i=1}^{N} \sum_{j \in \mathcal{N}(i)} W_{ij} [\max(u_{ji}, 0) C_j - \max(u_{ij}, 0) C_i]$$

由于 $u_{ji} = -u_{ij}$，每对 $(i,j)$ 的贡献互为相反数，因此：

$$\sum_{i=1}^{N} F_i^{\text{adv}} = 0 \quad \blacksquare$$

系统总"浓度"（总分数）守恒，不会凭空产生或消失。

### 2.5 CFL 稳定性条件

为保证显式时间积分的绝对稳定性（不产生负浓度或数值振荡），从 §2.4 更新方程中提取 $C_t(i)$ 的系数，令其 $\geq 0$：

$$1 - \Delta t \sum_{j \in \mathcal{N}(i)} W_{ij} \left[ D + \max(u_{ij}, 0) \right] \geq 0$$

由此推导出严格 CFL 条件：

$$\Delta t_{\text{CFL}} = \frac{1}{\max_{i} \sum_{j \in \mathcal{N}(i)} W_{ij} \left( D + \max(u_{ij}, 0) \right)}$$

实际取 $\Delta t = C_{\text{safe}} \cdot \Delta t_{\text{CFL}}$，安全系数 $C_{\text{safe}} = 0.9$。此公式正确反映了 $D$ 和 $u$ 对步长的联合约束——增大扩散或对流强度时，$\Delta t$ 必须相应减小。

### 2.6 收敛条件与收敛性证明

**终止条件**：

$$\max_{i \in \{1,...,N\}} |C_{t+1}(i) - C_t(i)| < \epsilon, \quad \epsilon = 10^{-3}$$

**定理（Shape-CFD 全局收敛性）**：给定连通图 $\mathcal{G}$，且 $\Delta t$ 满足 §2.5 的严格 CFL 条件，守恒型 Upwind 格式迭代 $\{C_t\}$ 收敛于唯一稳态分布 $C^*$，且系统总浓度守恒。

**证明**：

将浓度更新写为矩阵形式 $C_{t+1} = \mathbf{M} C_t$，其中：

- 对角元素：$\mathbf{M}_{ii} = 1 - \Delta t \sum_{k \in \mathcal{N}(i)} W_{ik} \left[ D + \max(u_{ik}, 0) \right]$
- 非对角元素（$j \in \mathcal{N}(i)$）：$\mathbf{M}_{ij} = \Delta t \cdot W_{ij} \left[ D + \max(u_{ji}, 0) \right]$

**Step 1（非负性）**：CFL 条件保证 $\mathbf{M}_{ii} \geq 0$；$D \geq 0, W \geq 0$ 保证 $\mathbf{M}_{ij} \geq 0$。故 $\mathbf{M}$ 为非负矩阵。

**Step 2（列随机矩阵）**：$\mathbf{M}$ 第 $j$ 列列和：

$$\sum_i \mathbf{M}_{ij} = 1 - \Delta t \sum_{k \in \mathcal{N}(j)} W_{jk}\left[D + \max(u_{jk}, 0)\right] + \Delta t \sum_{i \in \mathcal{N}(j)} W_{ij}\left[D + \max(u_{ji}, 0)\right]$$

由图对称化 $W_{ij} = W_{ji}$，且 $\mathbf{M}_{ij}$ 中的 $\max(u_{ji}, 0)$ 表示"从 $j$ 到 $i$ 方向的对流系数"，与 $\mathbf{M}_{jj}$ 中的 $\max(u_{jk}, 0)$（$k=i$ 时）是完全相同的量。因此两个求和逐项抵消，$\sum_i \mathbf{M}_{ij} = 1, \forall j$。即 $\mathbf{M}$ 是列随机矩阵（column-stochastic），系统总浓度守恒。

**Step 3（Perron-Frobenius 定理）**：$\mathbf{M}$ 满足：(a) 非负；(b) 列随机（$\lambda_1 = 1$）；(c) 不可约（图连通）；(d) 非周期（$\mathbf{M}_{ii} > 0$ 提供自环）。由 Perron-Frobenius 定理，主特征值 $\lambda_1 = 1$ 严格唯一，其余 $|\lambda_k| < 1$，迭代 $C_t = \mathbf{M}^t C_0$ 以指数速率收敛于唯一稳态 $C^*$。$\blacksquare$

**Early Stopping 的作用**：完全收敛后 $C^*$ 仅由图结构和对流场决定，与初始浓度 $C_0$ 无关。实际算法设 $\epsilon = 10^{-3}$ early stopping（通常 30-50 步），此时 $C_0$（cosine 相关度）信号尚未被完全抹去。这正是设计意图：**有限步迭代在保留初始检索信号的同时，通过文档间信息传播改善排序**。

### 2.7 双物理指标

**动态 Reynolds 数**（运行时指标）：

$$\text{Re} = \frac{\sum_{i=1}^{N} |F_i^{\text{adv}}|}{\sum_{i=1}^{N} |F_i^{\text{diff}}|}$$

- $\text{Re} > 1$：对流主导
- $\text{Re} < 1$：扩散主导

**结构 Péclet 数**（静态指标）：

$$\text{Pe} = \frac{\overline{|u_{ij}|}}{D} = \frac{\frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} |u_{ij}|}{D}$$

- 不依赖浓度场 $C$，仅由图结构和 query 决定
- 零运行时开销（预计算即可）

### 2.8 自适应参数

**Fiedler Value 自适应扩散系数**：

拉普拉斯矩阵 $L = D_{\text{deg}} - W$ 的第二小特征值 $\lambda_2$（Fiedler value，代数连通度）：

$$D_{\text{adaptive}} = \frac{D_{\text{base}}}{1 + \lambda_2}$$

- $\lambda_2$ 大（图紧密）→ $D$ 减小，防止过度平滑
- $\lambda_2$ 小（图分散）→ $D$ 增大，增强信号传播

**跨域检测自适应对流强度**：

$$u_{\text{adaptive}} = u_{\min} + (u_{\max} - u_{\min}) \cdot \min\left(\frac{\sigma^2_{\text{sim}}}{0.04}, 1\right)$$

其中 $\sigma^2_{\text{sim}}$ 为候选文档间相似度方差。

### 2.9 流场拓扑分析

对每个节点计算速度场的离散散度（不依赖浓度场，仅反映流场拓扑结构）：

$$\text{div}(\vec{u})_i = \sum_{j \in \mathcal{N}(i)} W_{ij} \cdot u_{ij}$$

分类：
- $\text{div}(\vec{u})_i > \theta$：**发散点**（divergence / source，局部净流出区）
- $\text{div}(\vec{u})_i < -\theta$：**汇聚点**（convergence / sink，高相关度吸引区）
- $|\text{div}(\vec{u})_i| < \theta$：**停滞点**（stagnation，平衡区）

**物理含义**：散度为正表示该节点的对流场净流出（浓度倾向减少），散度为负表示净流入（浓度倾向积累）。汇聚点对应查询方向上的“语义引力井”——高相关文档自然聚集处。

---

## 3. Shape CFD 扩展（核心创新）

### 3.1 动机

**维度诅咒导致对流信号极弱**：

在 $d$ 维空间中，两个随机单位向量的内积满足：

$$\mathbb{E}[\langle x, y \rangle] = 0, \quad \text{Var}(\langle x, y \rangle) = \frac{1}{d}, \quad \sigma = \frac{1}{\sqrt{d}}$$

对 $d=4096$：$\sigma = \frac{1}{\sqrt{4096}} \approx 0.0156$

实验验证：$\overline{|u_{ij}|} \approx 0.01$，与理论标准差高度吻合。意味着对流系数几乎为零——不是算法的问题，是高维几何的必然。

### 3.2 "向量不是点，是形状" 假设

将文档从单一向量升级为**句子级点云**：

$$d_i \to \mathcal{P}_i = \{s_{i,1}, s_{i,2}, ..., s_{i,K_i}\} \subset \mathbb{R}^d$$

其中 $s_{i,k}$ 是文档 $d_i$ 的第 $k$ 个句子的 embedding 向量。

### 3.3 Chamfer 距离

两个点云之间的距离：

$$d_{\text{Chamfer}}(\mathcal{P}_A, \mathcal{P}_B) = \frac{1}{|\mathcal{P}_A|} \sum_{a \in \mathcal{P}_A} \min_{b \in \mathcal{P}_B} d_{\cos}(a, b) + \frac{1}{|\mathcal{P}_B|} \sum_{b \in \mathcal{P}_B} \min_{a \in \mathcal{P}_A} d_{\cos}(a, b)$$

其中 $d_{\cos}(a, b) = 1 - \cos(a, b)$ 为余弦距离。

**性质**：
- 对称性：$d_{\text{Chamfer}}(A, B) = d_{\text{Chamfer}}(B, A)$
- 非负性：$d_{\text{Chamfer}}(A, B) \geq 0$，等号当且仅当 $A = B$
- 自距离为零：$d_{\text{Chamfer}}(A, A) = 0$
- 捕获局部语义匹配：即使两个文档整体语义不同，只要部分句子高度相似，Chamfer 距离就会较小

### 3.4 Shape KNN 图

边权重用 Chamfer 距离替代 cosine：

$$W_{ij}^{\text{shape}} = \exp(-\beta \cdot d_{\text{Chamfer}}(\mathcal{P}_i, \mathcal{P}_j))$$

初始浓度：

$$C_0(i) = \exp(-2 \cdot d_{\text{Chamfer}}(\mathcal{P}_q, \mathcal{P}_i))$$

对流方向用**点云质心**：

$$\bar{v}_i = \frac{1}{K_i} \sum_{k=1}^{K_i} s_{i,k}, \quad u_{ij}^{\text{shape}} = \left\langle \frac{\bar{v}_j - \bar{v}_i}{\|\bar{v}_j - \bar{v}_i\|}, \frac{\bar{v}_q}{\|\bar{v}_q\|} \right\rangle \cdot u_{\text{strength}}$$

### 3.5 PQ-Chamfer 子空间距离 (V8 灵感, 2026-03-28)

为解决高维浓度效应对 Chamfer 距离区分度的压制，引入子空间切分：

$$d_{\text{PQ}}(a, b) = \frac{1}{K} \sum_{k=1}^{K} d_{\cos}(a^{(k)}, b^{(k)})$$

其中 $a^{(k)} = a[(k-1) \cdot m : k \cdot m]$ 为第 $k$ 个子空间切片，$K \times m = d$。

**物理解释**：在 $d$ 维空间中，cosine 距离的标准差 $\sigma \propto 1/\sqrt{d}$。切分为 $m$ 维子空间后，$\sigma \propto 1/\sqrt{m}$，方差放大 $d/m$ 倍。当 $d=4096, m=64$ 时，方差放大 64 倍，显著提升细粒度语义区分能力。

**与随机投影 (JL) 的对比**：

| 方法 | 原理 | NFCorpus NDCG@10 | 变化 |
|:---|:---|:-:|:-:|
| JL128 (随机投影 4096→1280d) | 破坏维度结构 | 0.2316 | -7.6% |
| **PQ 64×64 (结构化切分)** | **保持维度结构** | **0.2735** | **+9.1%** |

总摆差 16.7 个百分点，证明关键因素是**保持还是破坏维度结构**。

**消融实验结论**：
- 64×64 > 32×128 > 16×256（单调提升）
- 均匀加权 > 方差加权（方差加权 -3.7%）
- 顺序切分 ≈ 打乱切分（Qwen3 维度无局部语义结构）
- 纯 PQ > PQ+混合初始场（两者解决同一问题，不正交）

实现时自动回退：若 $d \neq K \times m$，自动使用全维度 cosine 距离。

加速比：无加速，与全维 cosine 计算量相同，但信息增益 +9.1%。

### 3.6 句子向量预缓存

- 离线阶段：对 23,701 个文档逐句 embed 并存入 SQLite（一次性成本）
- 在线阶段：查询时直接从 SQLite 读取 128 维投影向量
- 缓存命中后延迟：~3ms（对比无缓存的 1785ms embed API 调用）

---

# 第二部分：实验数据

## 4. 实验设置

### 4.1 数据集

| 属性 | 值 |
|:---|:---|
| 数据集 | 中国法律法规库 |
| 文档数 | 23,701 个法条 chunk |
| 维度 | 4096 维 (Qwen3-Embedding-8B) |
| 索引 | USearch HNSW (M=32, ef=128) |
| 候选数 N | 30 |

### 4.2 测试 Query 集

| # | Query | 类型 |
|:---|:---|:---|
| 1 | 劳动合同解除赔偿标准 | 单一法域 |
| 2 | 醉驾量刑标准 | 单一法域 |
| 3 | 离婚财产分割 | 跨法域 (民法+诉讼法) |
| 4 | 借款合同利息上限 | 单一法域 |
| 5 | 房屋租赁合同纠纷 | 单一法域 |
| 6 | 交通事故责任划分 | 跨法域 |
| 7 | 知识产权侵权赔偿 | 单一法域 |
| 8 | 商标侵权赔偿 | 单一法域 |
| 9 | 行政处罚听证程序 | 单一法域 |
| 10 | 未成年人犯罪处罚 | 跨法域 |

### 4.3 评估方法

**DeepSeek V3.2 盲评**：

- A/B 匿名化：随机分配 cosine 排序和 Shape-CFD 排序为"方案 A"或"方案 B"
- 评分标准：1-5 分（相关性、覆盖度、排序合理性）
- 每次评审独立（无上下文记忆）

## 5. 核心实验结果

### 5.1 参数搜索（72 组网格搜索）

| D | uStrength | knn | 平均 Re | 平均 Pe | 收敛率 | 最优 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 0.1 | 3 | 0.892 | 0.195 | 80% | |
| 0.10 | 0.1 | 3 | 0.756 | 0.098 | 90% | |
| **0.15** | **0.1** | **3** | **0.695** | **0.065** | **100%** | **⭐** |
| 0.20 | 0.1 | 3 | 0.623 | 0.049 | 100% | |
| 0.15 | 0.3 | 3 | — | — | 40% | |
| 0.15 | 0.5 | 3 | — | — | 10% | |

最优配置：$D=0.15, u_{\text{strength}}=0.1, k=3$

### 5.2 AD-Rank v2 vs Cosine 盲评

| Query | v2 得分 | Cosine 得分 | 胜方 |
|:---|:---:|:---:|:---:|
| 劳动合同解除赔偿 | 3 | 3 | 平 |
| 醉驾量刑标准 | 2 | 3 | cosine |
| 离婚财产分割 | 4 | 3 | **v2** |
| 借款合同利息上限 | 4 | 2 | **v2** |
| 房屋租赁合同纠纷 | 3 | 3 | 平 |
| 交通事故责任划分 | 4 | 2 | **v2** |
| 知识产权侵权赔偿 | 3 | 3 | 平 |
| 商标侵权赔偿 | 4 | 2 | **v2** |
| 行政处罚听证程序 | 5 | 3 | **v2** |
| 未成年人犯罪处罚 | 2 | 4 | cosine |
| **平均** | **3.4** | **2.8** | **v2 胜 7:2 (1平)** |

### 5.3 全策略横评（7 组 × 10 query × 60 次盲评）

| Rank | 方案 | 盲评分 | Pe | 速度 | 收敛率 | 胜/负 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **AD-Rank v2** | **3.4/5** | 0.065 | 14.8ms | **100%** | **7/2** |
| 🥈 | BAA 分块 | 3.3/5 | 0.226 | 7ms | 20% | 7/3 |
| 🥉 | v2+PCA | 3.2/5 | 0.112 | 33ms | 90% | 6/4 |
| 4 | v2+自适应 | 3.0/5 | 0.122 | 3.3ms | 100% | 5/4 |
| 5 | v2+温度 | 2.9/5 | 0.604 | 3.5ms | 10% | 5/5 |
| 6 | MDA | 2.7/5 | 0.351 | 85.8ms | 0% | 3/7 |

**关键发现**：Pe 高 ≠ 排序好。弱对流 + 强扩散是最佳平衡。

### 5.4 Shape CFD 独立实验

| 维度 | AD-Rank v2 | Shape CFD | 改进 |
|:---|:---:|:---:|:---:|
| 盲评分 (vs cosine) | 3.3/5 | **3.4/5** | +3% |
| 胜率 (vs cosine) | 6:4 | **8:2** | +33% |
| Pe (Péclet 数) | 0.065 | **0.281** | +332% |
| 收敛率 | 100% | 100% | — |
| 速度 (缓存加速后) | 14.8ms | **22ms** | +49% 可接受 |

### 5.5 速度优化历程

| 版本 | 延迟 | 加速比 | 手段 |
|:---|:---:|:---:|:---|
| v0 原始 | 1031ms | 1× | JavaScript 朴素实现 |
| v1 缓存 | 29.7ms | 35× | SQLite 向量缓存 |
| v2 极致 | 2.2ms | 468× | 平铺 Float64Array + 8 路展开 |
| Shape v0 | 1837ms | — | 每次调 API embed 句子 |
| **Shape v1** | **22ms** | **84×** | 句子预缓存 + 128 维投影 |

---

# 第三部分：核心流程

## 6. 算法流程图

```
输入: query 文本 + 知识库

Step 1: Query Embedding
  query → Qwen3-Embedding → q ∈ R^4096

Step 2: HNSW 粗筛
  q → USearch HNSW → top-30 候选 {v_1, ..., v_30}

Step 3: 点云构建 (Shape CFD)
  对每个候选文档:
    拆句 → 读 SQLite 缓存 → 128 维投影点云
  对 query:
    实时 embed → 128 维投影

Step 4: Shape KNN 图构建
  30×30 PQ-Chamfer 距离 (64×64 子空间) → KNN 稀疏化 (k=3)
  边权重: W_ij = exp(-2 × d_PQ-Chamfer)

Step 5: 初始浓度 + 对流方向
  C_0(i) = exp(-2 × d_PQ-Chamfer(q, d_i))  [纯 Chamfer, 不混合]
  u_ij = dot(centroid_j - centroid_i, query_dir) × u_strength

Step 6: 对流-扩散迭代
  while (maxDelta > ε and t < maxIter):
    C_{t+1} = C_t + dt × [扩散 + upwind 对流]
    检查 CFL 条件

Step 7: 拓扑分析
  计算每个节点散度 → 标记汇聚/发散/停滞点

Step 8: 排序输出
  按 C 降序 → top-K → 返回给 LLM

输出: 重排序后的 top-K 法条 + 物理指标 (Re, Pe)
```

## 7. 代码架构

| 文件 | 行数 | 职责 |
|:---|:---:|:---|
| `ad_rank.js` | 885 | 核心求解器 v2 (含自适应参数) |
| `ad_rank_shape.js` | 869 | Shape CFD (Chamfer + 点云 + 缓存加速) |
| `ad_rank_data.js` | 414 | 数据接口 (USearch + SQLite) |
| `ad_rank_shape_cache.js` | 329 | 句子向量预热脚本 |
| `ad_rank_v3.js` | ~400 | MDA / BAA 策略实现 |
| `ad_rank_final_eval.js` | ~480 | 全策略横评 |

---

# 第四部分：与已有工作对比

## 8. 相关工作定位

| 方法 | 年份 | 扩散 | 对流 | 形状 | 自适应 | 可解释 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Cosine Similarity | — | ❌ | ❌ | ❌ | ❌ | ❌ |
| PageRank (Brin 1998) | 1998 | 类似 | ❌ | ❌ | ❌ | ❌ |
| Diffusion-Aided RAG | 2025 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Gaussian Embeddings | 2014 | ❌ | ❌ | ✅(分布) | ❌ | ❌ |
| GELD | 2024 | ❌ | ❌ | ✅(高斯) | ❌ | ❌ |
| Cross-Encoder | 2019 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **AD-Rank (Ours)** | **2026** | **✅** | **✅** | ❌ | **✅** | **✅** |
| **Shape-CFD (Ours)** | **2026** | **✅** | **✅** | **✅** | **✅** | **✅** |

**独创组合**：形状表示 + 对流-扩散方程 + 检索重排序 + 自适应参数 + 物理可解释性

---

# 第五部分：待补充项（论文投稿前必须完成）

## 9. 短板清单

| # | 待补充 | 重要性 | 状态 |
|:---|:---|:---:|:---:|
| 1 | ~~公开数据集实验~~ (NFCorpus / SciFact) | 🆗 已完成 | ✅ |
| 2 | ~~NDCG@10 标准指标~~ | 🆗 已完成 | ✅ |
| 3 | ~~PQ-Chamfer 消融实验~~ | 🆗 已完成 | ✅ (Section 4.3.5) |
| 4 | ~~V7.1 伴随状态法预取~~ | 🆗 已完成 | ✅ 0.2802 (+27.6%) |
| 5 | ~~LID 实测~~ (Qwen3 内蓴维度) | 🆗 已完成 | ✅ LID≈16~19 |
| 6 | ~~UOT 理论框架~~ (Chamfer 动机) | 🆗 已完成 | ✅ Section 3.6 |
| 7 | 与 Neural Reranker 对比 (BGE-Reranker) | 🔴 必须 | 待实验 |
| 8 | V11 Token 级点云 | 🔴 最高优先 | 待实验 |
| 9 | 人工标注评估 | 🟡 建议 | 视资源 |

---

# 第六部分：参考文献

1. Vilnis & McCallum. "Word Representations via Gaussian Embedding." ICLR 2015.
2. Yang et al. "Diffusion-Aided RAG." ACL 2025.
3. Noack et al. "Advection-Augmented CNN." arXiv 2024.
4. Page et al. "The PageRank Citation Ranking." Stanford InfoLab, 1998.
5. Bruna et al. "Spectral Networks and Deep Locally Connected Networks on Graphs." ICLR 2014.
6. Kipf & Welling. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
7. Johnson & Lindenstrauss. "Extensions of Lipschitz mappings into a Hilbert space." AMS, 1984.
8. Rubner et al. "The Earth Mover's Distance as a Metric for Image Retrieval." IJCV, 2000.
9. Fan et al. "A Point Set Generation Network for 3D Object Reconstruction." CVPR 2017.
10. Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.
11. Nogueira & Cho. "Passage Re-ranking with BERT." arXiv:1901.04085, 2019.
12. Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS 2013.
13. Zhao et al. "GRAND: Graph Neural Diffusion." ICML 2021.
14. Chamberlain et al. "BLEND: A Framework for Graph Neural Diffusion." ICLR 2021.
15. Li et al. "DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models." ICLR 2023.
