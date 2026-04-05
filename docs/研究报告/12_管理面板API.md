# 12 — 管理面板 API 与数据可视化

> Agent B4 第二轮深挖报告 | 研究范围：`routes/adminPanelRoutes.js`（2546行 / 124KB）、`AdminPanel/` 目录结构

---

## 一、算法目标

VCPToolBox 的管理面板是一个**全功能运维控制中心**，通过 Express Router 提供 RESTful API，覆盖系统监控、插件管理、Agent 人格编辑、RAG 参数调优、知识库管理、梦境审批等全部管理功能。前端为纯 HTML/CSS/JS 的 SPA（`AdminPanel/index.html` 约 68KB）。

---

## 二、API 端点分类总览

按功能域分为 **8 大模块、25+ 端点**：

### 2.1 系统运维模块

| 端点 | 方法 | 功能 |
|------|------|------|
| `/system-monitor/pm2/processes` | GET | PM2 进程列表（CPU/内存/重启次数/运行时间） |
| `/system-monitor/system/resources` | GET | 系统级CPU/内存，跨平台兼容（Win/macOS/Linux） |
| `/server-log` | GET | **增量日志读取**（含 inode 轮转检测） |
| `/server-log/clear` | POST | 清空日志 |
| `/server/restart` | POST | PM2 重启（清除 TextChunker/VectorDBManager 模块缓存） |
| `/vectordb/status` | GET | VectorDB 健康状态 |

**关键算法**：日志增量读取——通过 `offset + inode` 双重机制检测日志轮转，首次加载限制 2MB 最大读取并跳过首行不完整行。

### 2.2 插件管理模块

| 端点 | 方法 | 功能 |
|------|------|------|
| `/plugins` | GET | 列出全部插件（启用+禁用+分布式） |
| `/plugins/:name/toggle` | POST | 启用/禁用（rename manifest ↔ manifest.block） |
| `/plugins/:name/config` | POST | 保存插件 config.env + 热重载 |
| `/plugins/:name/description` | POST | 编辑插件描述 |
| `/plugins/:name/commands/:id/description` | POST | 编辑工具调用指令描述 |
| `/preprocessors/order` | GET/POST | 预处理器执行顺序管理 + 热重载 |

**关键设计**：插件启用/禁用通过**文件重命名**实现（`plugin-manifest.json` ↔ `plugin-manifest.json.block`），无需数据库，简单可靠。

### 2.3 Agent 人格管理模块

| 端点 | 方法 | 功能 |
|------|------|------|
| `/agents` | GET | Agent 文件列表（支持文件夹结构） |
| `/agents/map` | GET/POST | Agent 别名映射表（model → Agent 文件） |
| `/agents/new-file` | POST | 创建新 Agent 提示词文件（.txt/.md） |
| `/agents/:fileName` | GET/POST | 读写 Agent 提示词内容 |
| `/agent-assistant/config` | GET/POST | AgentAssistant 多 Agent 配置 |
| `/agent-assistant/scores` | GET | Agent 评分数据 |

**关键算法**：AgentAssistant 配置解析器——从 `config.env` 中用正则 `AGENT_{BASENAME}_{FIELD}` 提取多 Agent 定义（含 model_id、system_prompt、temperature 等），写入时保留原文件的注释和非 Agent 行。

### 2.4 RAG 与知识库管理模块

| 端点 | 方法 | 功能 |
|------|------|------|
| `/rag-tags` | GET/POST | RAG 标签配置（rag_tags.json） |
| `/rag-params` | GET/POST | RAG 12 参数调优（rag_params.json） |
| `/semantic-groups` | GET/POST | 语义分组配置（支持 .edit.json → 重启生效） |
| `/thinking-chains` | GET/POST | 元思维链配置 |
| `/available-clusters` | GET | 可用簇列表（以"簇"结尾的文件夹） |

### 2.5 占位符查看器（数据聚合算法）

这是最复杂的单个端点，聚合 **9 种占位符类型**进行统一展示：

```
1. agent          — Agent 人格文件内容
2. env_tar_var    — 环境变量（Tar/Var 前缀）
3. env_sar        — SarPrompt 占位符
4. fixed          — 固定值（Date/Time/Today/Festival/Port）
5. static_plugin  — 静态插件输出值
6. tool_description — 工具描述（{{VCPPluginName}}）
7. vcp_all_tools  — 聚合所有工具描述
8. diary          — 日记数据（角色级别）
9. async_placeholder — 异步结果占位符
```

**数据聚合逻辑**：
- 固定值包含**农历日期**计算（`chinese-lunar-calendar` 库 → 节气/生肖）
- TVS 变量支持**间接引用**：env 值以 `.txt` 结尾时自动读取文件内容
- 表情包占位符特殊处理：`xxx表情包.txt` → 从缓存 Map 获取
- 每项展示：`{type, name, preview(截断180字), charCount}`

### 2.6 多媒体与缓存管理

| 端点 | 方法 | 功能 |
|------|------|------|
| `/multimodal-cache` | GET/POST | 多模态缓存（图片/视频描述） |
| `/multimodal-cache/reidentify` | POST | 重新识别媒体（调用 ImageProcessor） |
| `/image-cache` | GET/POST | 旧版图片缓存（向后兼容） |

### 2.7 梦境审批系统（AgentDream）

| 端点 | 方法 | 功能 |
|------|------|------|
| `/dream-logs` | GET | 梦日志列表（含 pending 操作计数） |
| `/dream-logs/:filename` | GET | 梦日志详情 |
| `/dream-logs/:filename/operations/:opId` | POST | 审批/拒绝梦操作 |

**关键算法**：审批执行器支持 3 种操作类型：
- `merge`：合并多篇日记 → 调用 DailyNoteWrite 创建新日记 → 删除源文件 → 更新向量库
- `delete`：删除指定日记 → 更新向量库
- `insight`：AI 梦感悟 → 创建新日记到 `[角色名的梦]` 前缀目录

### 2.8 工具与配置编辑器

| 端点 | 方法 | 功能 |
|------|------|------|
| `/tool-list-editor/tools` | GET | 所有可用工具列表 |
| `/tool-list-editor/configs` | GET | 配置方案列表 |
| `/tool-list-editor/config/:name` | GET/POST/DELETE | 配置方案 CRUD |
| `/tool-list-editor/export/:name` | POST | 导出工具列表为 TXT |
| `/config/main` | GET/POST | 主 config.env 编辑 |
| `/tvsvars` | GET/POST | TVS 变量文件 CRUD |
| `/verify-login` / `/logout` / `/check-auth` | POST/GET | Cookie 认证（24h HttpOnly） |

---

## 三、安全与权限设计

1. **认证方式**：HTTP Basic Auth → 通过 `adminAuth` 中间件前置拦截
2. **Cookie 认证**：登录后设置 `admin_auth` Cookie（HttpOnly + SameSite=Strict + 24h 有效期）
3. **HTTPS 自适应**：检测 `x-forwarded-proto` 后自动添加 `Secure` 标志
4. **敏感数据保护**：Image_Key 在占位符列表中显示为 `******`
5. **文件安全**：Agent 文件名验证只允许 `.txt/.md`，Dream 文件名禁止 `..` 路径穿越

---

## 四、前端结构速览

```
AdminPanel/
├── index.html (68KB)           — 主 SPA 页面
├── login.html (14KB)           — 登录页
├── script.js (13KB)            — 通用逻辑
├── style.css (92KB)            — 全局样式（含丰富的暗色主题）
├── tool_list_editor.html/js    — 工具列表编辑器
├── rag_tags_editor.html/css    — RAG 标签编辑器
├── image_cache_editor.html/css — 图片缓存编辑器
├── vcptavern_editor.html/js/css — VCPTavern 编辑器
└── js/                         — 子模块 JS
```

前端为**无框架 SPA**，通过 hash 路由切换面板，无构建步骤部署。

---

## 五、与第一轮报告的关联补充

- **补充报告04**（插件系统）：管理面板提供了插件的**启用/禁用/配置/描述编辑**的完整 GUI，通过文件 rename 实现状态切换，每次操作后自动调用 `pluginManager.loadPlugins()` 热重载。
- **补充报告05**（分布式通信）：管理面板的系统监控直接调用 PM2 API 获取进程信息，插件列表同时展示本地和分布式（`isDistributed`）插件。
- **新发现**：AgentDream 梦境审批是第一轮未覆盖的重要功能——AI 可以在"做梦"时提出日记合并/删除/感悟操作，需要管理员通过面板审批后才执行，体现了**人类在环**的安全设计。

---

## 六、可借鉴点

### 1. 占位符聚合查看器
将 9 种不同来源的占位符（Agent、环境变量、静态插件、工具描述、日记等）统一聚合展示，是非常有价值的**调试工具**。legal-assistant 的 RAG 系统也有多种上下文注入源，可借鉴此设计构建诊断面板。

### 2. 日志增量读取 + inode 轮转检测
`offset + inode` 双机制实现高效日志流式查看，比全量读取高效得多。可直接用于 legal-assistant 的运维日志。

### 3. 配置分层管理
`主 config.env → 插件 config.env（覆盖） → config.env.example（模板回退）` 的三层配置机制，提供良好的默认值和灵活性。

### 4. 插件启停的文件 rename 模式
不需要数据库，只需 `manifest.json ↔ manifest.json.block` 的重命名即可实现启用/禁用，极其轻量且可靠。

### 5. 梦境审批的人类在环设计
AI 提出的关键操作（删除/合并日记）需要管理员审批才执行，这种 HITL（Human-In-The-Loop）模式对 legal-assistant 的敏感法律操作审批很有参考价值。

---

> 报告完成 | Agent B4 | 2026-03-13
