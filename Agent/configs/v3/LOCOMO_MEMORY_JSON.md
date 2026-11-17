# LoCoMo Agent 长对话基线

五个 JSON 配置均由现有 `Agent/cli/evaluate.py` 读取，使用本机 Ollama `qwen3.5:4b` 和同一份 `data/evaluation/locomo_10pct.jsonl`。评测器按 session 读取对话，再逐个回答 held-out 问题；每道问题使用独立的回答上下文，避免问题间泄漏。

| 文件 | `features.memory.mode` |
|---|---|
| `locomo_full_context_qwen4b.json` | 全部历史 |
| `locomo_recent_window_qwen4b.json` | 最近窗口 |
| `locomo_bm25_rag_qwen4b.json` | BM25 检索原文 |
| `locomo_bm25_rag_high_recall_qwen4b.json` | BM25 较高召回设置（5 轮/块、top 10） |
| `locomo_vector_rag_qwen4b.json` | 向量检索原文 |
| `locomo_session_summary_qwen4b.json` | session 摘要后 BM25 检索 |

运行示例：

```bash
python -m Agent.cli.evaluate \
  --config Agent/configs/v3/locomo_bm25_rag_qwen4b.json \
  --limit 1 --run-name locomo-qwen4b-bm25-smoke
```

`--limit 1` 是 **1 段对话及其中全部问题**，不是 1 道题。去掉它会处理全部 10 段对话。每种配置使用不同的 `--run-name`，避免覆盖结果。当前向量配置使用本机 Ollama 的 `qwen3-embedding:0.6b`；`embedding_provider` 也可设为 `sentence_transformers`，此时需安装 `sentence-transformers`，并把 `embedding_model` 设为 Hugging Face 模型名或本地目录。

在用户提供的 7 段对话上，按“检索上下文包含问题标注的全部证据 ID”计算，BM25 3 轮/块 top 5 为 624/1033（60.4%，平均约 2776 字符/题）；5 轮/块 top 10 为 753/1033（72.9%，约 8863 字符/题）。后者是额外上下文换召回的配置，不应与前者当作同一读取预算比较。原始 BM25 配置保留作为基础基线。

所有配置可编辑 `features.memory`：

- `mode`：五种策略之一；`top_k` 控制 RAG 或摘要检索条数；`chunk_turns` 控制原文检索块包含的对话轮数；`max_chars` 控制最近窗口字符数。
- `include_dates`、`include_speakers`、`include_image_captions`：控制存入检索上下文的信息。设置为 `false` 做消融。
- `embedding_provider`、`embedding_model`：向量 RAG 的提供方及模型。当前为 `ollama`、`qwen3-embedding:0.6b`；建议记录 Ollama 模型 digest 以便复现。
- `features.compaction.enabled`：控制 Agent 的上下文压缩。基线默认关闭；打开后测试的是“记忆策略 + Agent 压缩”组合。
- `model.reasoning.enabled`：控制 Ollama 思考模式，基线默认关闭。

摘要方法会对每段对话的每个 session 调用一次同一个 Ollama 模型，在一次评测运行内缓存，后续问题复用；摘要检索仍使用 BM25。当前 Agent 结果会记录摘要生成调用次数，但现有 usage 汇总仅覆盖回答阶段，因此比较成本时需另计摘要写入调用。`features.memory.enabled` 对这五种策略应保持 `true`；工具开关保持关闭，因为这些是 Agent 自动构造上下文的基线，不是可由模型调用的记忆工具。
