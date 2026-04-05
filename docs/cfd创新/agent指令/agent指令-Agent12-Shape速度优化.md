# Agent 12 指令：Shape CFD 速度优化（句子缓存 + Rust Chamfer + 随机投影）

> 目标：将 Shape CFD 从 1837ms 降到 < 30ms
> 瓶颈分析：embed 1785ms (97%) + solve 52ms (3%)
> 攻克路线：预缓存消灭 embed + Rust/随机投影加速 Chamfer

## 速度瓶颈拆解

```
当前 Shape CFD 延迟 = 1837ms：
  句子 embedding   1785ms  ← 每次查询都调 API，最大瓶颈
  Chamfer 距离计算    40ms  ← N² × K² 的暴力匹配
  AD-Rank 求解        12ms  ← 已优化过
```

## 攻克方案（三层并进）

### 层1：句子向量预缓存（消灭 1785ms）

在 SQLite 中预存每个法条的句子级 embedding。

修改文件：`/home/amd/HEZIMENG/legal-assistant/ad_rank_data.js`

```js
// 新增表：sentence_vectors
// CREATE TABLE IF NOT EXISTS sentence_vectors (
//   chunk_id TEXT NOT NULL,     -- 对应 metadata 的 chunk ID
//   sent_idx INTEGER NOT NULL,  -- 句子序号
//   vector BLOB NOT NULL,       -- Float32Array → Buffer
//   PRIMARY KEY (chunk_id, sent_idx)
// )
```

**预热脚本** `ad_rank_shape_cache.js`：
1. 遍历 23,701 个 chunk
2. 按句号/分号拆句
3. 批量调 Qwen3-Embedding API（每批 32 句）
4. 存入 SQLite

**查询时**：直接从 SQLite 读句子向量（~3ms），零 API 调用。

预热只需跑一次（预计 30-60 分钟），之后永久缓存。

### 层2：Chamfer 距离加速（两个方案）

#### 方案 A：随机投影降维

Chamfer 距离计算是 O(|A|×|B|×d)。d=4096 太大。

```js
// 用固定随机投影将 4096 → 128 维
// Johnson-Lindenstrauss: 128 维足以保持距离结构（误差 < 10%）
const PROJ_DIM = 128;
const projMatrix = generateFixedProjection(4096, PROJ_DIM, seed=42);

// 预计算：缓存时就存 128 维投影版本
// 查询时：Chamfer 在 128 维上算

// 加速比：4096/128 = 32×
// 原 40ms → ~1.2ms
```

#### 方案 B：Rust 原生模块（napi-rs）

如果方案 A 精度不够，用 Rust 实现 Chamfer 距离：

```rust
// ad_rank_chamfer.rs
use napi::bindgen_prelude::*;
use napi_derive::macro_rules_attribute;

#[napi]
pub fn chamfer_distance_batch(
    clouds_a: Vec<Float32Array>,  // 文档 i 的句子向量
    clouds_b: Vec<Float32Array>,  // 文档 j 的句子向量
    dim: u32,
) -> f64 {
    // SIMD 加速的 Chamfer 距离
    // 用 packed_simd 或手写 AVX2
}
```

**Rust 的优势**：
- SIMD 并行（AVX2 一次处理 8 个 f32）
- 零 GC 停顿
- napi-rs 可直接被 Node.js require

### 层3：分块 Chamfer（结合 BAA 思想）

将 Chamfer 距离按维度分块计算，可并行且信号更强：

```js
// 不在 4096 维整体算 Chamfer，而是分 B=8 块
// 每块 512 维的 Chamfer 距离，加权求和
function blockChamfer(cloudA, cloudB, B = 8) {
  const dim = cloudA[0].length;
  const blockSize = dim / B;
  let totalDist = 0;
  
  for (let b = 0; b < B; b++) {
    const start = b * blockSize;
    const end = start + blockSize;
    // 提取每个句子的第 b 块
    const blockA = cloudA.map(v => v.slice(start, end));
    const blockB = cloudB.map(v => v.slice(start, end));
    totalDist += chamferDistance(blockA, blockB);
  }
  return totalDist / B;
}
```

## 实现计划

### Step 1：预缓存脚本（消灭 embedding 瓶颈）

创建 `ad_rank_shape_cache.js`：
- 读 metadata.json，遍历所有 chunk
- 拆句 → 批量 embed → 存 SQLite
- 同时存 128 维随机投影版本

### Step 2：优化 ad_rank_shape.js

修改 `getCandidatesClouds()` 方法：
- 从 SQLite 直接读句子向量（不调 API）
- Chamfer 用 128 维投影版本计算
- 对流方向仍用 4096 维质心

### Step 3：验证

```
目标延迟：
  句子向量读取:   3ms (SQLite 缓存)
  Chamfer 距离:   2ms (128 维投影)
  AD-Rank 求解:  12ms (现有)
  总计:          ~17ms ← 从 1837ms 优化到 17ms = 108× 加速
```

对 10 个 query 做 DeepSeek 盲评，确认 128 维投影版的 Chamfer 排序质量不低于 4096 维版本。

## 文件清单

| 文件 | 操作 |
|:---|:---|
| `ad_rank_shape_cache.js` | 新建：预热脚本（句子 embed + SQLite 存储） |
| `ad_rank_shape.js` | 修改：集成缓存层 + 128 维投影 Chamfer |
| `ad_rank_data.js` | 修改：新增 sentence_vectors 表 |

## ⚠️ 约束

- 预热脚本可能跑 30-60 分钟（23,701 × ~3 句 ≈ 70,000 次 embed）
- 用批量 API 减少调用次数（每批 32 句）
- 随机投影矩阵用 seed=42，存在代码中（确保可复现）
- **不要动 ad_rank.js**（v2 基线保留）
- Rust 方案是 Plan B，先试随机投影
