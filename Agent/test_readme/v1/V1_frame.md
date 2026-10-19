# Agent Evaluation V1 运行框架

> 文档版本：V1  
> 记录日期：2026-10-19  
> 适用范围：当前 `Agent` 目录下的 AIME、GSM8K、BFCL Single Turn、BFCL Multi Turn 和 LoCoMo 评测流程

## 1. 框架概述

当前框架可以概括为：

> 配置驱动的数据集运行器 + LangChain 模型适配层 + 数据集专属 Agent 循环 + 独立评分器。

它不是让所有数据集共用同一种 ReAct 流程，而是先读取统一 YAML 配置和统一 JSONL 数据，再根据数据集及任务类型选择不同执行路径。

```text
YAML 配置
  ↓
加载并校验统一 JSONL 数据
  ↓
选择模型 profile 并创建客户端
  ↓
根据数据集进入专属 Agent 运行路径
  ↓
并发执行样本并持续保存 checkpoint
  ↓
使用数据集专属 evaluator 评分
  ↓
生成逐样本结果和聚合指标
```

## 2. 运行入口

统一入口为：

```bash
python3 -m Agent.cli.evaluate \
  --config Agent/configs/evaluations/<dataset>.yaml \
  --profile <model_profile>
```

当前数据集配置包括：

- `aime.yaml`
- `gsm8k.yaml`
- `bfcl_single_turn.yaml`
- `bfcl_multi_turn.yaml`
- `locomo.yaml`

当前模型 profile 包括：

- `qwen3_5_9b`：本地 Ollama 的 `qwen3.5:9b`
- `v4flash`：OpenAI-compatible API 的 `DeepSeek-V4-Flash`

例如：

```bash
python3 -m Agent.cli.evaluate \
  --config Agent/configs/evaluations/locomo.yaml \
  --profile qwen3_5_9b

python3 -m Agent.cli.evaluate \
  --config Agent/configs/evaluations/locomo.yaml \
  --profile v4flash
```

运行参数：

| 参数 | 作用 |
|---|---|
| `--profile` | 覆盖 YAML 中的默认模型 profile |
| `--limit N` | 只运行前 N 条记录，用于冒烟测试 |
| `--resume` | 读取 checkpoint，只继续未完成样本 |
| `--rescore` | 不调用模型，复用已有 AgentOutput 重新评分 |

## 3. 配置解析与模型初始化

### 3.1 YAML 配置

每个数据集的 YAML 定义：

- 数据文件路径
- `single_turn` 或 `multiturn`
- 最大 Agent step
- 单样本超时时间
- 工具是否启用
- 工具是否真实执行
- System Prompt
- 模型 profile
- Context Window
- Max Tokens
- Temperature
- Reasoning/Thinking 开关

`Agent.evaluation.runner.execute_run` 只接受 `schema_version: "2.0"`。最终解析配置、实际 profile 和数据文件路径会写入 `resolved_config.json`。

### 3.2 模型客户端

`DirectLangChainRuntime` 根据 profile 创建模型客户端。

#### qwen3.5_9b

- 客户端：`ChatOllama`
- 模型：`qwen3.5:9b`
- 地址：优先使用 `OLLAMA_HOST`，否则使用 `http://127.0.0.1:11434`
- 上下文长度：通过 `num_ctx` 设置
- 最大生成长度：通过 `num_predict` 设置
- 显式传递 `reasoning=false`

#### deepseekv4flash

- 客户端：`ChatOpenAI`
- 模型：`DeepSeek-V4-Flash`
- API 地址：`FIRE_AGENT_BASE_URL`
- API 密钥：`FIRE_AGENT_API_KEY`
- 最大生成长度：通过 `max_tokens` 设置
- 通过请求参数显式关闭 thinking

当前实验中两个模型均使用 `temperature=0`。

## 4. 数据加载与隔离

数据统一采用：

```json
{
  "input": {},
  "reference": {}
}
```

- `input` 交给 Agent。
- `reference` 不进入模型上下文，只供评分器使用。

加载时检查：

- JSON Schema 是否合规
- 样本 ID 是否唯一
- 数据集名称是否与 YAML 一致
- `task_family` 是否一致
- 工具配置是否符合数据集要求
- LoCoMo 是否包含问题和长期记忆后端字段
- BFCL Multi Turn 是否包含初始环境及工具类

模型在运行过程中无法看到标准答案。

## 5. 样本并发与断点恢复

Runner 在加载全部记录后，通过 `asyncio.Semaphore(4)` 将并发样本数限制为 4。

```text
全部样本
  ↓
读取已有 checkpoint
  ↓
过滤已完成样本 ID
  ↓
最多并发执行 4 个样本
  ↓
每完成一条立即追加 checkpoint
```

Checkpoint 位于：

```text
runs/<dataset>-<profile>/checkpoints/agent_outputs.jsonl
```

使用 `--resume` 时，已有 ID 不会重复执行，只调度缺失样本。

## 6. 普通 Agent 循环

除 LoCoMo 外，数据集共享基础 Agent 循环。

### 6.1 初始化

每个样本构造：

```text
System Prompt
+ 用户消息
+ 当前样本的工具定义
```

存在工具 schema 时，通过 `model.bind_tools(...)` 将工具绑定为 OpenAI function-calling 格式。

### 6.2 单个 Step

```text
调用模型
  ↓
序列化并保存模型消息
  ↓
检查 finish_reason/done_reason
  ↓
解析结构化 tool calls
  ├─ 没有调用：当前 turn 结束
  ├─ execution=disabled：只记录调用，不执行
  └─ execution=enabled：执行工具，将结果加入上下文，继续下一 step
```

运行时保存：

- 模型消息
- 最终文本
- 工具名称和参数
- 工具返回结果
- 运行状态
- 样本耗时
- 尝试次数
- XML 重试次数
- 数据集专属元数据

模型达到输出长度上限时，样本状态记为 `limit_exceeded`。

## 7. 各数据集执行路径

### 7.1 AIME 与 GSM8K

- `task_family=single_turn`
- 不绑定工具
- 每条样本只执行一次模型调用
- 最终文本交给数学答案提取器
- AIME 提取整数，GSM8K 提取最后数值

### 7.2 BFCL Single Turn

- `task_family=single_turn`
- 向模型提供函数 schema
- `execution=disabled`
- 模型生成工具名称和参数，但运行器不执行工具
- 结构化调用交给 BFCL AST checker

该路径测量函数选择和参数生成能力，不测量工具执行后的任务完成状态。

### 7.3 BFCL Multi Turn

- `task_family=multiturn`
- 每个用户 turn 最多执行 20 个 Agent step
- 工具在 BFCL 环境中真实执行
- 工具结果作为 tool message 返回模型
- 多个用户 turn 之间保留消息上下文和环境状态

```text
用户 turn
  ↓
模型生成工具调用
  ↓
BFCL 后端执行
  ↓
工具结果返回模型
  ↓
模型继续调用或结束当前 turn
  ↓
进入下一用户 turn
```

每轮调用保存到：

```text
metadata.bfcl_calls_by_turn
```

若样本环境在某一 turn 提供 `missed_function`，运行器会动态把新增函数加入工具集合。`long_context` 类别会启用 BFCL 长上下文模式。

最终评分使用完整调用轨迹和环境状态，而不是只检查最后文本。

## 8. LoCoMo 长对话记忆路径

LoCoMo 使用专用运行分支，不采用普通多轮对话上下文累积方式。其核心是：

> 分 session 摄取原始对话，将信息写入外部长期记忆；评测问题阶段不提供原始对话，只允许从长期记忆检索。

### 8.1 Conversation 隔离

每个 conversation 使用独立 SQLite 数据库：

```text
runs/locomo-<profile>/locomo_memory/<conversation-id>.sqlite3
```

不同 conversation 不共享记忆，防止样本间信息污染。

### 8.2 数据库结构

主表 `memories` 包含：

| 字段 | 含义 |
|---|---|
| `id` | 记忆主键 |
| `content` | 自包含记忆文本 |
| `date` | 相关日期 |
| `source_ids` | 记忆来源的对话 ID |

数据库还包含：

- `memories_fts` FTS5 全文索引
- insert/update/delete 同步 trigger
- `UNIQUE(content, date, source_ids)` 去重约束

每次尝试先创建：

```text
<conversation-id>.attempt-<attempt>.sqlite3
```

整条样本成功后，才通过原子替换发布为正式数据库。失败尝试不会覆盖正式记忆文件。

### 8.3 摄取阶段

原始对话按 session 依次处理。每个 session 都创建独立的短期消息上下文：

```text
LoCoMo ingestion system prompt
+ 当前 session 原始对话
```

System Prompt 要求模型：

- 不回答对话参与者
- 提取事实、事件、日期、关系和偏好
- 识别后续纠正信息
- 将信息写入持久记忆

摄取阶段允许使用：

| 工具 | 作用 |
|---|---|
| `memory_add` | 添加新记忆 |
| `memory_update` | 修改或合并已有记忆 |
| `memory_delete` | 删除过时或错误记忆 |
| `memory_search` | 查找已有记忆，辅助去重和修正 |

每个 session 最多执行 20 个 step。工具结果会重新加入当前 session 的消息上下文，模型可连续检索、写入、更新或删除。

不同 session 不重复携带此前全部原始对话；它们通过同一个 SQLite 数据库延续长期信息。

### 8.4 问答阶段

全部 session 摄取完成后，逐题回答 held-out questions。

每道问题创建全新上下文：

```text
LoCoMo question system prompt
+ 当前评测问题
```

此阶段：

- 不提供原始长对话
- 只向模型暴露 `memory_search`
- 禁止调用 `memory_add`、`memory_update` 和 `memory_delete`
- 每道问题最多执行 20 个 step
- 检索结果作为 tool message 返回模型
- 最终短答案保存到 `metadata.locomo_answers`
- 记忆不足时要求回答 `No information available`

即使模型尝试调用写工具，运行时也会返回：

```text
only memory_search is allowed during evaluation
```

因此评测问题不能污染已经形成的长期记忆。

### 8.5 检索算法

`memory_search` 的执行过程为：

1. Query 小写化。
2. 使用 Unicode `\w+` 规则分词。
3. 将每个 token 转为 FTS5 前缀查询。
4. Token 之间使用 OR。
5. 使用 `bm25(memories_fts)` 排序。
6. 默认返回 5 条。
7. `limit` 最小为 1、最大为 20。

返回字段包括：

- `id`
- `content`
- `date`
- `source_ids`

如果 Query 分词后为空，则按记忆 ID 倒序返回最新记录。

当前检索是词法 FTS5/BM25 检索，不是向量或 embedding 语义检索。同义改写可能降低召回率。

## 9. 运行时自动重试

单样本出现以下问题时可以自动重试：

- Timeout
- API 连接错误
- 502、503、504
- 部分瞬态服务端错误

最多额外重试 2 次，即单样本最多进行 3 次完整尝试：

```text
第一次失败 → 等待 1 秒
第二次失败 → 等待 2 秒
第三次失败 → 返回最终失败状态
```

XML 工具调用响应解析错误使用 step 内重试：

- 最多重试 2 次
- 在当前 step 的工具真正执行前重试
- 避免解析失败导致有副作用的工具被重复执行

可能的最终状态包括：

- `success`
- `timeout`
- `runtime_error`
- `limit_exceeded`

## 10. 运行后评分

模型执行与评分相互分离：

```text
Agent 执行
  ↓
AgentOutput
  ↓
数据集专属 scorer
  ↓
SampleResult
  ↓
聚合 Metrics
```

| 数据集 | 评分方式 |
|---|---|
| AIME | 抽取整数并与标准答案精确匹配 |
| GSM8K | 抽取最后数值并精确匹配 |
| BFCL Single Turn | AST/function-call checker |
| BFCL Multi Turn | multi-turn checker 检查完整调用轨迹与任务状态 |
| LoCoMo | 问题级 Token F1 |

由于执行和评分分离，可以使用 `--rescore` 修改或重新运行评分，而不重新调用模型。

## 11. 运行产物

| 文件 | 内容 |
|---|---|
| `checkpoints/agent_outputs.jsonl` | 边运行边追加的恢复点 |
| `agent_outputs.jsonl` | 模型消息、状态、耗时、工具调用和元数据 |
| `sample_results.jsonl` | 逐样本评分结果 |
| `metrics.json` | 数据集及类别聚合指标 |
| `resolved_config.json` | 完整解析配置、profile 和数据路径 |
| `bfcl_results.jsonl` | BFCL 标准调用结果 |
| `locomo_memory/*.sqlite3` | LoCoMo conversation 级长期记忆数据库 |

## 12. 运行结束后的失败补跑

除样本内部自动重试外，框架还提供独立的失败补跑工具：

```bash
python3 -m Agent.cli.retry_transient \
  --run runs/<run-directory> \
  --output-root runs_retry
```

其逻辑为：

1. 读取已完成 run。
2. 筛选 timeout 或瞬态连接/服务端错误。
3. 只重新执行这些最终失败的样本。
4. 未重试样本沿用原结果。
5. 将重试输出合并回完整结果集。
6. 重新运行评分和聚合。
7. 写入 `retry_audit.jsonl`，保留重试前后结果和尝试次数。

因此当前框架包含两层重试：

| 重试层级 | 目的 |
|---|---|
| 单样本运行中的自动重试 | 立即处理当前请求的瞬态故障 |
| 运行结束后的 `retry_transient` | 批量补跑最终仍失败的样本 |

## 13. 框架特征与边界

当前运行框架的主要特征是：

- YAML 配置驱动
- 统一输入结构与 reference 隔离
- 数据集专属执行路径
- 工具调用循环
- BFCL 环境真实执行
- LoCoMo 外部持久记忆
- 异步并发与断点恢复
- 运行和评分分离
- 单样本重试与批量补跑两级容错

当前边界包括：

- LoCoMo 使用词法全文检索，不是语义向量检索。
- LoCoMo 的记忆质量依赖模型主动抽取，漏写或错误更新会影响后续回答。
- 不同模型的运行后端不同，本地 Ollama 与远程 API 的耗时不能直接解释为纯模型推理速度。
- Token 使用字段当前没有形成可靠统计，不能据此比较成本。
- BFCL 当前使用抽样子集，不能直接视为完整 BFCL 榜单成绩。
