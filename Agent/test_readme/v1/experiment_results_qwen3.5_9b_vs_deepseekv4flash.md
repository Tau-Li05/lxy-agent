# qwen3.5_9b 与 deepseekv4flash 全数据集实验结果对比报告-v1

> 对比模型：`qwen3.5_9b`、`deepseekv4flash`  
> 报告日期：2026-10-19  
> 差值定义：除特别说明外，差值均为 `qwen3.5_9b - deepseekv4flash`  
> 数值约定：准确率/F1 保留 4 位小数，百分点差保留 2 位，耗时单位为秒

## 1. 实验结论摘要

本次实验覆盖 AIME、GSM8K、BFCL Single Turn、BFCL Multi Turn 和 LoCoMo 五个测评部分。整体结果不是单一“总分”，因为各数据集的任务性质、样本单位和评分尺度不同，不应把准确率与 F1 直接加总。

| 数据集 | 样本规模 | 核心指标 | qwen3.5_9b | deepseekv4flash | 差值 | 结论 |
|---|---:|---|---:|---:|---:|---|
| AIME | 90 题 | Accuracy | 25.56% | 55.56% | -30.00 pp | deepseekv4flash 明显占优 |
| GSM8K | 1,319 题 | Accuracy | 95.07% | 92.65% | +2.43 pp | qwen3.5_9b 占优 |
| BFCL Single Turn | 547 条 | Accuracy | 72.03% | 77.33% | -5.30 pp | deepseekv4flash 占优 |
| BFCL Multi Turn（原始） | 120 条 | Accuracy | 18.33% | 55.00% | -36.67 pp | deepseekv4flash 明显占优 |
| BFCL Multi Turn（retry 后） | 120 条 | Accuracy | 18.33% | 61.67% | -43.33 pp | deepseekv4flash 优势扩大 |
| LoCoMo | 10 段对话 / 199 题 | Mean F1 | 0.3701 | 0.4455 | -0.0753 | deepseekv4flash 占优 |

主要发现：

- `deepseekv4flash` 在 AIME、BFCL Single Turn、BFCL Multi Turn 和 LoCoMo 上领先。
- `qwen3.5_9b` 在 GSM8K 上以 32 道净胜题、2.43 个百分点的优势领先。
- AIME 上 `qwen3.5_9b` 的主要问题不仅是答案质量，还包括大量 `limit_exceeded` 和 `timeout`；其执行成功率仅 27.78%。
- BFCL Multi Turn 重试修复了 `deepseekv4flash` 的 8 条失败样本，使准确率由 55.00% 提升到 61.67%；`qwen3.5_9b` 的两条重试没有带来正确率提升。
- LoCoMo 中 `deepseekv4flash` 在类别 3、4、5 上领先，特别是类别 4；`qwen3.5_9b` 在类别 1、2 上小幅领先。
- 当前 Token 字段全部为 0，不能据此比较 Token 效率或成本。

## 2. 当前实验框架

### 2.1 数据流与运行流程

实验采用 Agent Evaluation v2 直接运行框架：

1. `dataset_processing.build_evaluation_data` 确定性构建统一 JSONL 数据。
2. 每条数据采用 `{input, reference}` 结构；Agent 仅接收 `input`，评测器保留 `reference`。
3. `load_evaluation_dataset` 使用 JSON Schema 校验结构，并校验任务类型、工具配置、数据集名称与 ID 唯一性。
4. `DirectLangChainRuntime` 按模型 profile 执行任务。
5. Runner 使用并发度 4 执行样本，并把中间结果写入 checkpoint。
6. 每个数据集使用专属 evaluator 评分。
7. 输出 `agent_outputs.jsonl`、`sample_results.jsonl`、`metrics.json`；BFCL 额外输出 `bfcl_results.jsonl`。
8. BFCL Multi Turn 的失败任务通过独立 retry 流程重跑，并在 `retry_audit.jsonl` 中保存前后结果与尝试次数。

### 2.2 Agent 的运行方式

#### 2.2.1 启动入口

Agent 通过单文件 v2 YAML 配置启动，统一入口为：

```bash
python3 -m Agent.cli.evaluate \
  --config Agent/configs/evaluations/<dataset>.yaml \
  --profile <model_profile>
```

本次实验中的 `<dataset>` 为 `aime`、`gsm8k`、`bfcl_single_turn`、`bfcl_multi_turn` 或 `locomo`，模型 profile 为 `qwen3_5_9b` 或 `v4flash`。例如：

```bash
python3 -m Agent.cli.evaluate \
  --config Agent/configs/evaluations/locomo.yaml \
  --profile qwen3_5_9b

python3 -m Agent.cli.evaluate \
  --config Agent/configs/evaluations/locomo.yaml \
  --profile v4flash
```

运行器还支持：

- `--limit N`：只运行数据集前 N 条，用于冒烟测试。
- `--resume`：从 `checkpoints/agent_outputs.jsonl` 继续未完成的运行。
- `--rescore`：不重新调用模型，复用已有 `agent_outputs.jsonl` 重新执行评分。
- `--profile`：覆盖 YAML 中的 `model.default_profile`。

瞬态失败可通过以下入口筛选并重试：

```bash
python3 -m Agent.cli.retry_transient \
  --run runs/<run-directory> \
  --output-root runs_retry
```

可先添加 `--dry-run` 查看将被重试的样本；中断后可使用 `--resume` 继续。

#### 2.2.2 配置解析与模型初始化

`Agent.evaluation.runner.execute_run` 读取 YAML，并要求 `schema_version: "2.0"`。解析后的配置连同实际 profile 和数据文件绝对路径写入 `resolved_config.json`，便于复现实验。

`DirectLangChainRuntime` 根据 profile 创建模型客户端：

- `qwen3.5_9b` 使用 `ChatOllama`，连接 `OLLAMA_HOST`；若环境变量未设置，则使用配置中的 `http://127.0.0.1:11434`。
- `deepseekv4flash` 使用 `ChatOpenAI` 兼容客户端，端点和密钥分别来自 `FIRE_AGENT_BASE_URL` 与 `FIRE_AGENT_API_KEY`。
- 两个 profile 均使用 `temperature=0`，并显式关闭 thinking/reasoning。
- Ollama 通过 `num_ctx`、`num_predict` 设置上下文和生成长度；API profile 通过 `max_tokens` 设置生成长度。

#### 2.2.3 样本调度、Agent 循环与工具执行

Runner 首先加载并校验全部评测记录，再使用 `asyncio.Semaphore(4)` 控制最多 4 个样本并发执行。每完成一个样本便追加写入 checkpoint，因此异常中断后能够从已有 ID 继续。

普通任务的 Agent 循环如下：

1. 加载数据集 system prompt、当前用户 turn 和该样本的工具 schema。
2. 有工具时通过 `model.bind_tools(...)` 把 OpenAI function-calling 形式的工具绑定到模型。
3. 调用模型并保存完整响应、最终文本及结构化工具调用。
4. 若模型没有工具调用，则结束当前 turn。
5. 若工具 execution 为 `disabled`，只记录模型提出的调用，不实际执行；这是 BFCL Single Turn 的运行方式。
6. 若工具 execution 为 `enabled`，执行工具，把工具返回值作为 tool message 加回上下文，再进入下一步；BFCL Multi Turn 和 LoCoMo 使用此方式。
7. Single Turn 每个 turn 最多执行 1 个模型 step；Multi Turn 每个 turn 最多执行配置中的 20 个 step。
8. 模型返回 `finish_reason/done_reason=length` 时记录为 `limit_exceeded`；超过单样本 timeout 时记录为 `timeout`。

BFCL Multi Turn 会把每轮调用保存为 `bfcl_calls_by_turn`，并通过 BFCL 的 `execute_multi_turn_func_call` 在任务初始环境中真实执行。`long_context` 类别会显式启用 BFCL 长上下文模式。

#### 2.2.4 自动重试、输出与评分

运行时对 timeout、连接错误、502/503/504 和部分服务端错误最多额外重试 2 次，即单样本最多 3 次完整尝试；等待时间依次为 1 秒和 2 秒。XML 响应解析错误在当前 step 执行任何工具前最多重试 2 次，避免重复产生工具副作用。

每个运行目录包含：

| 文件 | 内容 |
|---|---|
| `checkpoints/agent_outputs.jsonl` | 边运行边追加的恢复点 |
| `agent_outputs.jsonl` | 最终文本、消息、状态、耗时、工具调用和运行元数据 |
| `sample_results.jsonl` | 专属 evaluator 生成的逐样本得分 |
| `metrics.json` | 数据集级和类别级聚合指标 |
| `resolved_config.json` | 完整解析配置、profile 和数据路径 |
| `bfcl_results.jsonl` | BFCL 标准调用结果，非 BFCL 不生成 |

评分在模型执行结束后进行。模型始终看不到 `reference`；AIME/GSM8K 使用答案抽取与精确匹配，BFCL 使用 AST 或 multi-turn checker，LoCoMo 使用问题级 F1。

### 2.3 长对话记忆方式

LoCoMo 不把完整历史对话长期塞在模型上下文中，而是采用“分会话摄取 → SQLite 持久化 → 问题期只读检索”的外部长期记忆架构。每个 conversation 使用独立数据库，避免不同样本间记忆污染。

#### 2.3.1 记忆存储结构

每个 conversation 的记忆保存在：

```text
runs/locomo-<profile>/locomo_memory/<conversation-id>.sqlite3
```

SQLite 中包含：

- `memories` 主表：`id`、`content`、`date`、`source_ids`。
- `memories_fts`：SQLite FTS5 全文检索虚表。
- insert/update/delete trigger：主表变化时同步更新 FTS 索引。
- `UNIQUE(content, date, source_ids)`：阻止完全相同记忆的重复写入。

正式运行时先创建 `<conversation-id>.attempt-<n>.sqlite3` 临时数据库。整条样本成功后使用原子替换将其发布为 `<conversation-id>.sqlite3`；失败尝试不会覆盖最终记忆文件。

#### 2.3.2 摄取阶段：将长对话压缩为持久记忆

LoCoMo 的每个 session 独立进入摄取阶段。模型会收到该 session 的对话和专门的 ingestion system prompt，要求提取自包含的事实、事件、日期、关系、偏好与纠正信息，而不是回答对话参与者。

摄取阶段允许模型使用：

| 工具 | 作用 |
|---|---|
| `memory_add` | 写入新的记忆内容、日期和来源 ID |
| `memory_update` | 根据 memory ID 修正或合并已有记忆 |
| `memory_delete` | 删除过时或错误的记忆 |
| `memory_search` | 查找已存记忆，辅助去重、更新和关联 |

每个 session 最多执行 20 个 Agent step。工具返回值重新加入当前 session 的消息上下文，模型可以连续检索、写入和修正，直到不再产生工具调用。不同 session 共享同一个 conversation 数据库，但不会把前面所有原始对话文本重新放入上下文；跨 session 信息通过数据库延续。

#### 2.3.3 问答阶段：隔离原始对话并只读检索

所有 session 摄取完成后，评测问题逐题独立执行：

1. 为每道 held-out question 新建消息上下文。
2. 原始长对话不放入该问题的上下文。
3. 只向模型暴露 `memory_search`，不暴露 add/update/delete。
4. 模型必须先检索持久记忆，再给出简短答案。
5. 问答阶段即使尝试调用写工具，运行时也会返回 `only memory_search is allowed during evaluation`，防止测试问题污染记忆。
6. 每道问题最多执行 20 个 step；最终文本和该问题的检索调用保存到 `metadata.locomo_answers`。
7. 若记忆不足，system prompt 要求严格回答 `No information available`。

这种隔离确保 LoCoMo 测量的是“写入后能否从外部长期记忆检索并回答”，而不是模型能否直接从长上下文复制答案。

#### 2.3.4 检索算法

`memory_search` 对 query 执行 Unicode `\w+` 分词和小写化，并将各 token 转为带前缀匹配的 FTS5 查询，token 之间使用 OR。结果通过 SQLite `bm25(memories_fts)` 排序。默认返回 5 条，调用方可设置 `limit`，但运行时限制在 1～20 条之间。若 query 分词后为空，则按最新写入顺序返回记忆。

检索结果包含 `id`、`content`、`date` 和 `source_ids`，使模型既能读取事实，也能利用时间及来源信息判断答案。

#### 2.3.5 记忆方式的边界

- 这是词法 FTS5/BM25 检索，不是向量检索或语义 embedding 检索；同义改写可能降低召回率。
- 记忆内容由模型主动抽取，漏写、错误合并或未处理纠正信息都会影响后续答案。
- 每道问题的上下文相互独立，但共享摄取完成后的只读数据库。
- 当前没有跨 conversation 的全局记忆；每个 LoCoMo conversation 完全隔离。
- 数据库记录 `source_ids`，但当前评分依据最终答案，不单独评分引用是否完整。
- 最终 `AgentOutput.metadata.memory_db` 记录数据库路径，便于复核记忆内容和检索行为。

### 2.4 模型配置

| 文档名称 | 实际 profile | 实际模型名 | Provider | Context Window | Temperature | Reasoning |
|---|---|---|---|---:|---:|---|
| qwen3.5_9b | `qwen3_5_9b` | `qwen3.5:9b` | Ollama | 65,536 | 0 | disabled |
| deepseekv4flash | `v4flash` | `DeepSeek-V4-Flash` | 配置的 OpenAI-compatible API | 65,536 | 0 | disabled |

### 2.5 数据集与运行限制

| 数据集 | Task family | 工具模式 | Max steps | Timeout/样本 | Max output tokens |
|---|---|---|---:|---:|---:|
| AIME | single turn | 无工具 | 1 | 300 | 4,096 |
| GSM8K | single turn | 无工具 | 1 | 300 | 4,096 |
| BFCL Single Turn | single turn | 传入 schema、不执行工具 | 1 | 300 | 4,096 |
| BFCL Multi Turn | multi turn | BFCL 后端真实执行 | 20 | 600 | 8,192 |
| LoCoMo | multi turn | SQLite FTS5 长期记忆工具 | 20 | 3,600 | 4,096 |

### 2.6 数据规模与抽样

| 数据集 | 当前规模 | 数据构建方式 |
|---|---:|---|
| GSM8K | 1,319 题 | 完整 test split |
| AIME | 90 题 | AIME 2024、2025、2026，各 30 题 |
| BFCL Single Turn | 547 条 | 13 个类别分别抽取 15% |
| BFCL Multi Turn | 120 条 | 4 个 multi-turn 类别分别抽取 15%，各 30 条 |
| LoCoMo | 10 段对话 / 199 个问题 | 五类 QA 分层抽取 10% |

首版实验未纳入数据构建清单中的独立 `memory`、`web_search` 和 `format_sensitivity` 扩展类别；这里的独立 `memory` 类别不等同于 LoCoMo 已启用的 SQLite 长期记忆机制。BFCL 的聚合结果带有 `partial_evaluation=true`，因此本文将其视为当前抽样子集结果，而非完整 BFCL 榜单成绩。

## 3. 统一指标与计算方法

| 指标 | 计算方式 | 方向 |
|---|---|---|
| 正确数 | 单题/单样本判分为 1 的数量 | 越高越好 |
| Accuracy | `正确数 / 总样本数` | 越高越好 |
| 错误率 | `1 - Accuracy` | 越低越好 |
| 绝对差值 | `qwen3.5_9b - deepseekv4flash` | 依指标解释 |
| 相对差值 | `(qwen3.5_9b - deepseekv4flash) / deepseekv4flash` | 依指标解释；分母为 0 时 N/A |
| 微平均 | 汇总全部样本后计算 | 受大类别权重影响 |
| 宏平均 | 各类别指标等权平均 | 每个类别同权 |
| 双方均正确 | 两模型在同一 ID 上均正确 | 描述一致成功 |
| 仅单方正确 | 同一 ID 上仅一个模型正确 | 描述配对优势 |
| 净胜样本数 | `仅 qwen 正确 - 仅 deepseek 正确` | 正数有利于 qwen |
| 结果一致率 | `(双方均正确 + 双方均错误) / 配对数` | 描述行为一致性 |
| 平均耗时 | `总 duration_seconds / 样本数` | 越低越好 |
| P90/P95/P99 | 对应分位点的样本耗时 | 越低越好 |
| 执行成功率 | `agent_outputs.status=success / 总样本数` | 越高越好 |
| 评测正确率 | 通过专属 evaluator 的样本数占比 | 越高越好 |

“执行成功”和“评测正确”严格区分：前者只代表程序正常产生输出，后者才表示答案通过评测器。

## 4. AIME

### 4.1 评测方式

Runner 按以下优先级从最终回答抽取整数：

1. `#### <1～3 位整数>`；
2. `\boxed{<1～3 位整数>}`；
3. 回答中最后一个独立的 1～3 位非负整数。

抽取值转为整数后与参考答案严格相等得 1 分，否则得 0 分。无法抽取、超时或达到运行限制均计 0 分。

### 4.2 总体与年份结果

| 范围 | 题数 | qwen3.5_9b 正确数 | qwen3.5_9b Accuracy | deepseekv4flash 正确数 | deepseekv4flash Accuracy | 差值 | 优胜模型 |
|---|---:|---:|---:|---:|---:|---:|---|
| AIME 2024 | 30 | 9 | 30.00% | 17 | 56.67% | -26.67 pp | deepseekv4flash |
| AIME 2025 | 30 | 8 | 26.67% | 13 | 43.33% | -16.67 pp | deepseekv4flash |
| AIME 2026 | 30 | 6 | 20.00% | 20 | 66.67% | -46.67 pp | deepseekv4flash |
| 总体 | 90 | 23 | 25.56% | 50 | 55.56% | -30.00 pp | deepseekv4flash |

两个模型各年份样本数相等，因此年份宏平均与总体微平均相同。`deepseekv4flash` 三个年份均领先，最大差距出现在 2026。

### 4.3 配对结果

| 配对类型 | 数量 | 占 90 题比例 |
|---|---:|---:|
| 两模型均正确 | 23 | 25.56% |
| 仅 qwen3.5_9b 正确 | 0 | 0.00% |
| 仅 deepseekv4flash 正确 | 27 | 30.00% |
| 两模型均错误 | 40 | 44.44% |
| 结果一致 | 63 | 70.00% |

净胜题数为 `0 - 27 = -27`。本次运行中，qwen3.5_9b 答对的 23 题全部也被 deepseekv4flash 答对。

### 4.4 运行状态与耗时

| 指标 | qwen3.5_9b | deepseekv4flash | 解释 |
|---|---:|---:|---|
| 执行成功 | 25（27.78%） | 73（81.11%） | status=`success` |
| limit_exceeded | 38（42.22%） | 17（18.89%） | 达到运行/生成限制 |
| timeout | 27（30.00%） | 0 | 超过单样本限制 |
| 总耗时 | 50,723.906 | 3,076.872 | 全部样本 duration 求和 |
| 平均耗时 | 563.599 | 34.187 | 总耗时/90 |
| 中位耗时 | 552.929 | 25.532 | P50 |
| P90 | 903.009 | 78.936 | 90% 样本不超过该值 |
| P95 | 903.010 | 81.363 | 95% 样本不超过该值 |
| P99 | 903.014 | 86.475 | 99% 样本不超过该值 |
| 最短/最长 | 94.509 / 903.016 | 7.902 / 100.970 | 极值 |

注意：配置中的 AIME timeout 为 300 秒，但部分记录耗时接近 903 秒，表明外层执行/终止链路可能叠加了额外等待或重试时间；耗时仍按记录中的 `duration_seconds` 如实报告。

## 5. GSM8K

### 5.1 评测方式

模型答案优先取 `####` 后的最后一个数值；如果没有该标记，则取回答中最后一个数值。解析支持正负号、小数和千位逗号，并移除逗号后与参考答案抽取值进行字符串精确匹配。该方法不是符号数学等价判断。

### 5.2 准确性与配对结果

| 指标 | qwen3.5_9b | deepseekv4flash | 差值 |
|---|---:|---:|---:|
| 总题数 | 1,319 | 1,319 | 0 |
| 正确数 | 1,254 | 1,222 | +32 |
| 错误数 | 65 | 97 | -32 |
| Accuracy | 95.0720% | 92.6459% | +2.4261 pp |
| Error rate | 4.9280% | 7.3541% | -2.4261 pp |

| 配对类型 | 数量 | 比例 |
|---|---:|---:|
| 两模型均正确 | 1,188 | 90.07% |
| 仅 qwen3.5_9b 正确 | 66 | 5.00% |
| 仅 deepseekv4flash 正确 | 34 | 2.58% |
| 两模型均错误 | 31 | 2.35% |
| 结果一致 | 1,219 | 92.42% |

qwen3.5_9b 净胜 32 题。若只看两模型判定不同的 100 题，qwen3.5_9b 赢得 66 题，配对胜率为 66.00%。

### 5.3 运行状态与耗时

| 指标 | qwen3.5_9b | deepseekv4flash | 优胜模型 |
|---|---:|---:|---|
| 执行成功 | 1,317（99.85%） | 1,319（100.00%） | deepseekv4flash |
| limit_exceeded | 2 | 0 | deepseekv4flash |
| 总耗时 | 32,261.005 | 6,717.553 | deepseekv4flash |
| 平均耗时 | 24.459 | 5.093 | deepseekv4flash |
| 中位耗时 | 20.431 | 4.709 | deepseekv4flash |
| 标准差 | 15.473 | 2.004 | deepseekv4flash 更稳定 |
| P90 | 35.789 | 7.606 | deepseekv4flash |
| P95 | 55.118 | 8.562 | deepseekv4flash |
| P99 | 98.751 | 11.843 | deepseekv4flash |
| 最短/最长 | 1.611 / 138.151 | 1.514 / 23.183 | deepseekv4flash |

qwen3.5_9b 准确率更高，但 deepseekv4flash 的平均耗时约为其 20.82%，形成明显的准确率—速度权衡。

## 6. BFCL Single Turn

### 6.1 评测方式

- `irrelevance` 和 `live_irrelevance`：不发起任何工具调用才判正确。
- 无 AST ground truth 的 relevance 样本：至少产生一个工具调用判正确。
- 其他类别：将函数名与参数解析为结构化调用，交给 BFCL AST checker；Java、JavaScript、Python 使用对应语言模式。
- `parallel` 类别允许并行调用无序匹配。
- 若 BFCL checker 依赖不可用，则使用仓库内 dependency-free BFCL-compatible fallback，核对函数名称、参数集合与可接受值。
- checker 返回 `valid=true` 得 1 分，否则得 0 分。

### 6.2 各类别结果

| 类别 | N | qwen3.5_9b 正确/Accuracy | deepseekv4flash 正确/Accuracy | 差值 | 优胜模型 |
|---|---:|---:|---:|---:|---|
| irrelevance | 36 | 25 / 69.44% | 23 / 63.89% | +5.56 pp | qwen3.5_9b |
| live_irrelevance | 133 | 88 / 66.17% | 95 / 71.43% | -5.26 pp | deepseekv4flash |
| live_multiple | 158 | 116 / 73.42% | 117 / 74.05% | -0.63 pp | deepseekv4flash |
| live_parallel | 2 | 1 / 50.00% | 1 / 50.00% | 0.00 pp | 持平 |
| live_parallel_multiple | 4 | 3 / 75.00% | 1 / 25.00% | +50.00 pp | qwen3.5_9b |
| live_relevance | 2 | 2 / 100.00% | 2 / 100.00% | 0.00 pp | 持平 |
| live_simple | 39 | 24 / 61.54% | 33 / 84.62% | -23.08 pp | deepseekv4flash |
| multiple | 30 | 27 / 90.00% | 29 / 96.67% | -6.67 pp | deepseekv4flash |
| parallel | 30 | 20 / 66.67% | 25 / 83.33% | -16.67 pp | deepseekv4flash |
| parallel_multiple | 30 | 18 / 60.00% | 25 / 83.33% | -23.33 pp | deepseekv4flash |
| simple_java | 15 | 10 / 66.67% | 8 / 53.33% | +13.33 pp | qwen3.5_9b |
| simple_javascript | 8 | 6 / 75.00% | 6 / 75.00% | 0.00 pp | 持平 |
| simple_python | 60 | 54 / 90.00% | 58 / 96.67% | -6.67 pp | deepseekv4flash |
| **总体（微平均）** | **547** | **394 / 72.03%** | **423 / 77.33%** | **-5.30 pp** | **deepseekv4flash** |

类别胜负为：qwen3.5_9b 领先 3 类，deepseekv4flash 领先 7 类，3 类持平。样本极少的 `live_parallel`、`live_parallel_multiple` 和 `live_relevance` 波动很大，不宜仅凭百分比作强结论。

### 6.3 配对、调用与耗时

| 配对类型 | 数量 | 比例 |
|---|---:|---:|
| 两模型均正确 | 360 | 65.81% |
| 仅 qwen3.5_9b 正确 | 34 | 6.22% |
| 仅 deepseekv4flash 正确 | 63 | 11.52% |
| 两模型均错误 | 90 | 16.45% |
| 结果一致 | 450 | 82.27% |

| 指标 | qwen3.5_9b | deepseekv4flash | 说明 |
|---|---:|---:|---|
| 工具调用总数 | 549 | 565 | `tool_calls` 长度求和 |
| 每样本平均调用 | 1.004 | 1.033 | 总调用/547 |
| 中位调用数 | 1 | 1 | P50 |
| 最大调用数 | 6 | 6 | 单样本最大值 |
| 零调用样本 | 113 | 121 | 包含应零调用的 irrelevance 样本 |
| 执行成功率 | 100% | 100% | 两模型均无运行态失败 |
| 平均耗时 | 11.613 | 12.826 | qwen3.5_9b 更快 |
| 中位耗时 | 10.120 | 9.350 | deepseekv4flash 更快 |
| P90 | 17.851 | 23.254 | qwen3.5_9b 更低 |
| P95 | 21.405 | 33.439 | qwen3.5_9b 更低 |
| P99 | 28.318 | 53.770 | qwen3.5_9b 更低 |
| 最大耗时 | 101.202 | 107.968 | qwen3.5_9b 更低 |

工具选择正确率与参数正确率无法从所有样本的现有 `details` 字段中稳定拆分，因此不从最终 `valid` 结果反推这两个指标。

## 7. BFCL Multi Turn

### 7.1 评测方式

使用 BFCL 官方 `multi_turn_checker`，联合检查每轮工具调用、调用顺序、函数选择、参数、多轮状态变化及环境执行结果。只有最终返回 `valid=true` 才得 1 分；局部步骤正确但任务整体未完成仍为 0 分。

`metrics.json` 中部分运行失败样本被记入 `main` 类别。本文类别表根据稳定的样本 ID 前缀恢复真实类别，因此四类均为 30 条；总体正确数与官方聚合保持一致。

### 7.2 原始 runs 结果

| 类别 | N | qwen3.5_9b 正确/Accuracy | deepseekv4flash 正确/Accuracy | 差值 | 优胜模型 |
|---|---:|---:|---:|---:|---|
| multi_turn_base | 30 | 9 / 30.00% | 24 / 80.00% | -50.00 pp | deepseekv4flash |
| multi_turn_long_context | 30 | 7 / 23.33% | 19 / 63.33% | -40.00 pp | deepseekv4flash |
| multi_turn_miss_func | 30 | 3 / 10.00% | 8 / 26.67% | -16.67 pp | deepseekv4flash |
| multi_turn_miss_param | 30 | 3 / 10.00% | 15 / 50.00% | -40.00 pp | deepseekv4flash |
| **总体** | **120** | **22 / 18.33%** | **66 / 55.00%** | **-36.67 pp** | **deepseekv4flash** |

配对结果：双方均正确 18 条，仅 qwen3.5_9b 正确 4 条，仅 deepseekv4flash 正确 48 条，双方均错误 50 条；一致率 56.67%，qwen3.5_9b 净胜数为 -44。

### 7.3 Retry 后结果

| 类别 | qwen 原始 | qwen retry | qwen 提升 | deepseek 原始 | deepseek retry | deepseek 提升 |
|---|---:|---:|---:|---:|---:|---:|
| multi_turn_base | 30.00% | 30.00% | 0.00 pp | 80.00% | 83.33% | +3.33 pp |
| multi_turn_long_context | 23.33% | 23.33% | 0.00 pp | 63.33% | 73.33% | +10.00 pp |
| multi_turn_miss_func | 10.00% | 10.00% | 0.00 pp | 26.67% | 40.00% | +13.33 pp |
| multi_turn_miss_param | 10.00% | 10.00% | 0.00 pp | 50.00% | 50.00% | 0.00 pp |
| **总体** | **18.33%** | **18.33%** | **0.00 pp** | **55.00%** | **61.67%** | **+6.67 pp** |

| 状态迁移 | qwen3.5_9b | deepseekv4flash |
|---|---:|---:|
| 错误 → 正确 | 0 | 8 |
| 正确 → 错误 | 0 | 0 |
| 正确 → 正确 | 22 | 66 |
| 错误 → 错误 | 98 | 46 |
| 实际 retry audit 样本数 | 2 | 15 |
| 原始尝试数合计 | 2 | 15 |
| retry 尝试数合计 | 2 | 15 |
| script rerun 数合计 | 2 | 15 |
| 总尝试数合计 | 4 | 30 |

修复率按“原始错误且实际进入 retry 的样本”计算：qwen3.5_9b 为 `0/2=0%`；deepseekv4flash 为 `8/15=53.33%`。两者均无正确转错误，净修复数分别为 0 和 8。

### 7.4 运行状态、调用与耗时

| 指标 | qwen 原始 | qwen retry 后 | deepseek 原始 | deepseek retry 后 |
|---|---:|---:|---:|---:|
| 执行成功 | 116 | 118 | 100 | 112 |
| timeout | 2 | 0 | 6 | 0 |
| runtime_error | 0 | 0 | 9 | 0 |
| limit_exceeded | 2 | 2 | 5 | 8 |
| 平均耗时 | 245.041 | 237.667 | 190.581 | 151.760 |
| 中位耗时 | 223.987 | 219.723 | 124.684 | 107.092 |
| P90 | 414.616 | 408.588 | 489.788 | 339.839 |
| P95 | 463.435 | 452.376 | 589.190 | 415.856 |
| P99 | 598.626 | 497.725 | 600.002 | 523.198 |
| 工具调用总数 | 1,139 | 1,204 | 1,457 | 2,168 |
| 每样本平均调用 | 9.492 | 10.033 | 12.142 | 18.067 |
| 中位调用数 | 9 | 9 | 11 | 12 |
| 最大调用数 | 40 | 41 | 65 | 446 |
| 零调用样本 | 2 | 0 | 15 | 0 |

deepseekv4flash retry 后的单样本最大调用数达到 446，说明 retry 结果中至少存在极端长调用轨迹；平均调用数因此不宜脱离中位数单独解释。

## 8. LoCoMo

### 8.1 评测方式

LoCoMo 以问题级 Token F1 为核心，而不是 conversation 全对率。

标准化步骤：转小写、删除逗号、删除标点、空格分词、去除 `a/an/the/and`，再进行 Porter stemming。普通题目的重合词元数用于计算：

`Precision = overlap / predicted_tokens`

`Recall = overlap / reference_tokens`

`F1 = 2 × Precision × Recall / (Precision + Recall)`

无重合时 F1=0；两边标准化后均为空时 F1=1，仅一边为空时 F1=0。

特殊类别规则：

- 类别 1：预测与参考按逗号拆分；每个参考答案取与所有预测候选之间的最高 F1，再对参考答案求平均。
- 类别 3：参考答案只使用第一个分号之前的内容。
- 类别 5：预测包含 `no information available` 或 `not mentioned` 得 1，否则得 0。
- 类别 2、4：普通 Token F1。
- conversation 运行失败时，其下所有问题计 0 分。

### 8.2 总体 F1 分布

| 指标 | qwen3.5_9b | deepseekv4flash | 差值 |
|---|---:|---:|---:|
| 问题数 | 199 | 199 | 0 |
| F1 总和 | 73.6549 | 88.6481 | -14.9932 |
| Mean F1 | 0.3701 | 0.4455 | -0.0753 |
| Median F1 | 0.2353 | 0.3333 | -0.0980 |
| F1 标准差 | 0.3653 | 0.3855 | -0.0201 |
| P25 | 0.1026 | 0.1250 | -0.0224 |
| P75 | 0.5455 | 1.0000 | -0.4545 |
| P90 | 1.0000 | 1.0000 | 0.0000 |
| F1=1 | 43（21.61%） | 55（27.64%） | -12 题 |
| F1=0 | 44（22.11%） | 29（14.57%） | +15 题 |
| 0<F1<1 | 112（56.28%） | 115（57.79%） | -3 题 |

题目级配对中，qwen3.5_9b 得分更高 54 题，deepseekv4flash 得分更高 76 题，69 题相同；平均题目 F1 差为 -0.0753。

### 8.3 各类别结果

| 类别 | N | qwen F1 和 / Mean F1 | deepseek F1 和 / Mean F1 | Mean F1 差 | 优胜模型 |
|---|---:|---:|---:|---:|---|
| 1 | 28 | 6.4699 / 0.2311 | 5.1036 / 0.1823 | +0.0488 | qwen3.5_9b |
| 2 | 32 | 8.5781 / 0.2681 | 8.4632 / 0.2645 | +0.0036 | qwen3.5_9b |
| 3 | 10 | 0.3485 / 0.0348 | 1.1624 / 0.1162 | -0.0814 | deepseekv4flash |
| 4 | 84 | 19.2584 / 0.2293 | 31.9188 / 0.3800 | -0.1507 | deepseekv4flash |
| 5 | 45 | 39.0000 / 0.8667 | 42.0000 / 0.9333 | -0.0667 | deepseekv4flash |

类别 4 有 84 题，是规模最大的类别，也是总体差距的主要来源。

### 8.4 Conversation 级结果

| Conversation | qwen3.5_9b Mean F1 | deepseekv4flash Mean F1 | 差值 | 优胜模型 |
|---|---:|---:|---:|---|
| conv-26 | 0.3382 | 0.4109 | -0.0726 | deepseekv4flash |
| conv-30 | 0.4348 | 0.5125 | -0.0777 | deepseekv4flash |
| conv-41 | 0.3446 | 0.4502 | -0.1056 | deepseekv4flash |
| conv-42 | 0.3948 | 0.5188 | -0.1240 | deepseekv4flash |
| conv-43 | 0.3829 | 0.4178 | -0.0350 | deepseekv4flash |
| conv-44 | 0.3166 | 0.4680 | -0.1515 | deepseekv4flash |
| conv-47 | 0.3986 | 0.4510 | -0.0524 | deepseekv4flash |
| conv-48 | 0.4570 | 0.5816 | -0.1247 | deepseekv4flash |
| conv-49 | 0.3281 | 0.2789 | +0.0492 | qwen3.5_9b |
| conv-50 | 0.3374 | 0.4346 | -0.0972 | deepseekv4flash |

deepseekv4flash 在 10 段 conversation 中赢 9 段，qwen3.5_9b 赢 1 段。

### 8.5 记忆工具调用与耗时

| 指标 | qwen3.5_9b | deepseekv4flash | 解释 |
|---|---:|---:|---|
| Conversation 执行成功 | 10/10 | 10/10 | 均无运行失败 |
| 工具调用总数 | 1,285 | 2,037 | 包含记忆写入和检索调用 |
| 每 conversation 平均调用 | 128.5 | 203.7 | deepseek 调用更多 |
| 中位调用数 | 135.0 | 209.5 | P50 |
| 最大调用数 | 151 | 252 | 单 conversation 最大值 |
| 总耗时 | 14,850.122 | 7,881.170 | deepseek 更低 |
| 平均耗时 | 1,485.012 | 788.117 | deepseek 更低 |
| 中位耗时 | 1,516.295 | 822.154 | deepseek 更低 |
| P90 | 1,883.128 | 979.477 | deepseek 更低 |
| P95 | 1,979.661 | 985.589 | deepseek 更低 |
| P99 | 2,056.886 | 990.479 | deepseek 更低 |
| 最短/最长 | 671.677 / 2,076.193 | 422.457 / 991.702 | deepseek 更低 |

deepseekv4flash 虽然平均记忆工具调用更多，但总耗时约为 qwen3.5_9b 的 53.07%，且 Mean F1 更高。

## 9. 跨数据集综合分析

### 9.1 准确性与能力侧重点

- 数学：qwen3.5_9b 在 GSM8K 的基础数学文字题上更准确；deepseekv4flash 在难度更高的 AIME 上显著领先。
- 单轮函数调用：deepseekv4flash 总体领先，优势集中在 `live_simple`、`parallel` 和 `parallel_multiple`；qwen3.5_9b 在 `simple_java` 和小样本的 `live_parallel_multiple` 上领先。
- 多轮函数调用：deepseekv4flash 四个类别全部领先，特别是基础多轮和缺失参数场景。
- 长期记忆：deepseekv4flash 在 LoCoMo 总体、9/10 conversation 及 3/5 类别上领先。

### 9.2 效率

- deepseekv4flash 在 AIME、GSM8K、BFCL Multi Turn 和 LoCoMo 的平均耗时更低。
- BFCL Single Turn 上 qwen3.5_9b 的平均耗时、P90、P95、P99 更低，但 deepseekv4flash 的中位耗时略低。
- 两模型的运行环境不同：qwen3.5_9b 通过本地 Ollama，deepseekv4flash 通过 API。耗时包含各自端到端链路，不能解释为纯模型推理速度。

### 9.3 稳定性与重试

- AIME 中 qwen3.5_9b 的 timeout 和 limit_exceeded 比例很高，是当前最主要的稳定性问题。
- GSM8K 和 BFCL Single Turn 的运行完成度较高。
- BFCL Multi Turn retry 对 deepseekv4flash 有显著收益，对 qwen3.5_9b 无正确率收益。
- retry 后 deepseekv4flash 的准确率提升，但工具调用出现极端离群值，需要结合样本轨迹人工检查。

## 10. 数据质量、限制与不可计算指标

1. **Token 与成本不可计算**：`prompt_tokens`、`completion_tokens`、`total_tokens` 字段存在，但当前记录均为 0；不能据此报告零消耗或推算成本。
2. **跨运行环境耗时不可完全归因于模型**：本地 Ollama 与远程 API 的硬件、排队、网络和服务端实现不同。
3. **BFCL 是部分评测**：当前为按类别 15% 抽样，且 `partial_evaluation=true`。
4. **LoCoMo 是 10% 抽样**：只有 10 段 conversation、199 个问题，结论仅代表当前固定样本。
5. **BFCL 类别恢复**：Multi Turn 的部分非成功样本在聚合文件中落入 `main`；本文按样本 ID 恢复真实类别。总体分数不受影响。
6. **LoCoMo conversation 的 `correct` 不适合作为准确率**：只有该 conversation 下所有问题 F1 都为 1 时才为 true；因此本文使用问题级 Mean F1，而不是 0/10 的 conversation 全对率。
7. **BFCL 参数级错误不可完整拆分**：现有 `details` 并非所有样本都提供统一的函数、参数、顺序错误标签，故未伪造参数正确率。
8. **统计显著性未计算**：当前报告提供描述性统计和逐样本配对计数，未进行 bootstrap 置信区间、McNemar 检验或其他显著性检验。
9. **成本不可计算**：缺少可用 Token 数据及统一价格口径。

## 11. 最终结论

按当前实验数据，deepseekv4flash 的综合覆盖面更强：在高难数学、单轮与多轮函数调用、长期记忆四类核心能力上取得更高主指标，并且在多数数据集上拥有更低端到端耗时。qwen3.5_9b 的突出优势是 GSM8K，其准确率高出 2.43 个百分点；在 BFCL Single Turn 的少数类别上也有局部优势。

最需要优先排查的是 qwen3.5_9b 的 AIME 运行稳定性，以及 BFCL Multi Turn 中较低的任务完成率。deepseekv4flash 的 retry 策略有效修复了 8 条多轮失败样本，但 retry 后出现 446 次工具调用的极端轨迹，说明提高成功率的同时可能引入调用冗余。

## 附录 A：指标方向与适用范围

| 指标 | AIME | GSM8K | BFCL ST | BFCL MT | LoCoMo |
|---|---:|---:|---:|---:|---:|
| Accuracy | ✓ | ✓ | ✓ | ✓ | — |
| 年份 Accuracy | ✓ | — | — | — | — |
| 类别 Accuracy | — | — | ✓ | ✓ | — |
| Mean F1 | — | — | — | — | ✓ |
| 类别 Mean F1 | — | — | — | — | ✓ |
| Conversation Mean F1 | — | — | — | — | ✓ |
| 样本配对胜负 | ✓ | ✓ | ✓ | ✓ | ✓（按 F1） |
| 工具调用统计 | — | — | ✓ | ✓ | ✓ |
| Retry 修复率 | — | — | — | ✓ | — |
| 耗时分布 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Token/成本 | 不可用 | 不可用 | 不可用 | 不可用 | 不可用 |

## 附录 B：结果文件

- 聚合指标：`runs/*/metrics.json`、`runs_retry/*/metrics.json`
- 样本评分：`runs/*/sample_results.jsonl`、`runs_retry/*/sample_results.jsonl`
- Agent 原始输出、状态、耗时和调用：`runs/*/agent_outputs.jsonl`、`runs_retry/*/agent_outputs.jsonl`
- BFCL 标准提交格式：`runs/*/bfcl_results.jsonl`、`runs_retry/*/bfcl_results.jsonl`
- Retry 前后审计：`runs_retry/*/retry_audit.jsonl`
- 解析后配置：`runs/*/resolved_config.json`、`runs_retry/*/resolved_config.json`

## 附录 C：复核规则

- 所有模型比较按相同样本 ID 配对。
- Accuracy 差值使用百分点，不把百分点误写成百分比增长率。
- 耗时分位数采用线性插值分位数。
- LoCoMo 的总体 Mean F1 按 199 个问题微平均；类别表按类别内问题平均。
- BFCL Multi Turn 类别由样本 ID 的稳定前缀恢复；总体正确数直接与 `metrics.json` 交叉核对。
- Retry 后结果表示用 retry 输出替换对应失败输出后的完整 120 条结果，而不是只对 retry 子集单独计算准确率。
