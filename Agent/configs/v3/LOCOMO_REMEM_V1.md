# REMem 启发的 LoCoMo Ollama 适配版

这是基于 `REMem-main` 的 **gist + fact 情景记忆、实体关联、检索扩展与原文回溯** 思路实现的简化研究版本。它复用 Agent 的 `qwen_harness`、`data/evaluation/locomo_10pct.jsonl` 和现有 LoCoMo 评分器；不执行论文原版的 PPR 图排序、DSPy 事实重排、vLLM 离线批处理或多工具自由选择，因此**不能称为 REMem 论文结果的复现**。

## 运行

本机需运行 Ollama，并安装 `qwen3.5:4b` 与 `qwen3-embedding:0.6b`。先用不依赖 embedding 的版本检查一段对话（`--limit 1` 是一整段对话及其所有问题）：

```bash
python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_remem_v1_bm25_only_qwen4b.json \
  --limit 1 --run-name locomo-remem-v1-bm25-smoke
```

完整版本：

```bash
python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_remem_v1_full_qwen4b.json \
  --run-name locomo-remem-v1-full

  caffeinate -i python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_remem_v1_facts_only_qwen4b.json \
  --run-name locomo_remem_v1_facts_only_qwen4b && \
  caffeinate -i python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_remem_v1_no_graph_qwen4b.json \
  --run-name locomo_remem_v1_no_graph_only_qwen4b && \
  caffeinate -i python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_remem_v1_no_raw_qwen4b.json \
  --run-name locomo_remem_v1_no_raw_only_qwen4b && \
  caffeinate -i python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_remem_v1_no_time_qwen4b.json \
  --run-name locomo_remem_v1_no_raw_only_qwen4b
```

所有配置使用同一读取预算 `max_chars=12000`、首轮 `top_k=5`。`use_graph=true` 时允许从首轮节点沿实体关联增加最多 `graph_extra=5` 个节点；预算不足时跳过超长节点。当前是**一次图邻居扩展**，不是原版 REMem 的多步自由工具 Agent。Qwen 写入每个 session 调用一次；生成的事实必须引用该 session 中真实存在的 `dia_id`，否则丢弃。若该 session 没有可用事实且 `use_raw_backtrace=true`，保留原始 session 作为回退节点。

| 配置文件 | 与完整 v1 的唯一主要差异 | 检验问题 |
|---|---|---|
| `locomo_remem_v1_full_qwen4b.json` | 全功能 | 主结果 |
| `locomo_remem_v1_no_time_qwen4b.json` | 关闭日期元数据和结构化时间 | 时间索引的贡献 |
| `locomo_remem_v1_no_graph_qwen4b.json` | 关闭实体邻接扩展 | 图关联的贡献 |
| `locomo_remem_v1_no_raw_qwen4b.json` | 不返回原始轮次，也不使用原文回退节点 | 来源原文的贡献 |
| `locomo_remem_v1_facts_only_qwen4b.json` | 不存 gist | gist 的贡献 |
| `locomo_remem_v1_bm25_only_qwen4b.json` | 关闭向量分数 | 向量检索的增量价值 |

输出仍是 `runs/<run-name>/` 中的原有格式。`agent_outputs.jsonl` 的 `metadata` 额外记录 `memory_write_calls`、写入阶段输入/输出 tokens、`memory_invalid_extractions` 和节点数；逐题 `locomo_answers[].memory_retrieval` 记录检索节点数、图扩展数和原文回溯数。现有总 `usage` 仍是回答阶段；比较成本时把写入阶段与 Ollama embedding 计算另计。

## 公平比较与限制

- 问题、参考答案和 `qa.evidence` 不参与写入与检索，只在 Agent 原评分器中使用。
- 每段对话独立构建一次记忆；每道题独立回答，不把前一道题的答案加入记忆。
- 消融改变检索和可见信息，写入的模型输出也可能随提示内容变化；要严格隔离单一模块贡献，应进一步冻结同一份提取结果再重放各消融。
- Qwen4B 的 JSON 提取与来源 ID 准确性尚未经过本机服务端到端验证。若事实没有有效来源 ID，该事实会被丢弃；`no_raw` 可能因此出现空记忆，这是可观察的写入失败而非模型回答错误。
- 当前图边由标准化实体字符串相同产生，尚无 REMem 原版的复杂关联、PPR 和动态多工具检索。时间仅用于记忆节点和检索文本；没有独立的时间约束求解器。
