# Agent Evaluation

Agent 只消费预处理后的统一 JSONL，不读取或转换 benchmark 原始数据。数据转换位于项目根目录的
`dataset_processing/`，生成文件位于 `data/evaluation/`。

BFCL 的多轮后端从仓库内源码直接加载，不需要对 `benchmark/BFCL-main` 执行 `pip install -e`。

## 数据构建

```bash
python -m dataset_processing.build_evaluation_data
```

构建结果：GSM8K test 1,319 条；AIME 2024–2026 共 90 条；BFCL 的 13 个 single-turn
类别各抽取 15%，共 547 条；四个 `multi_turn_*` 类别各确定性抽取 15%，共 120 条。
LoCoMo 保留全部 10 段长期对话，按 QA 类别确定性抽取 10%，共 199 个问题，
使用 LLM 自主管理的本地 SQLite FTS5 记忆。
BFCL 的 `memory`、`web_search` 与 `format_sensitivity` 暂不纳入，详情见
`data/evaluation/manifest.json`。

## 单 YAML 运行

- `Agent/configs/evaluations/gsm8k.yaml`
- `Agent/configs/evaluations/bfcl_single_turn.yaml`
- `Agent/configs/evaluations/bfcl_multi_turn.yaml`
- `Agent/configs/evaluations/aime.yaml`
- `Agent/configs/evaluations/locomo.yaml`

```bash
export FIRE_AGENT_BASE_URL="http://127.0.0.1:8000/v1"
export FIRE_AGENT_API_KEY="..."
export FIRE_AGENT_BASE_URL="https://openapi.seu.edu.cn/v1"
python -m Agent.cli.evaluate --config Agent/configs/evaluations/gsm8k.yaml --limit 5
python -m Agent.cli.evaluate --config Agent/configs/evaluations/bfcl_multi_turn.yaml --profile v4flash --limit 5
python -m Agent.cli.evaluate --config Agent/configs/evaluations/gsm8k.yaml --profile v4flash
python -m Agent.cli.evaluate --config Agent/configs/evaluations/aime.yaml --profile v4flash
python -m Agent.cli.evaluate --config Agent/configs/evaluations/locomo.yaml --profile v4flash
```

本地 Ollama 可使用内置的 `qwen3_5_9b` profile：

```bash
ollama pull qwen3.5:9b
python -m Agent.cli.evaluate --config Agent/configs/evaluations/gsm8k.yaml --profile qwen3_5_9b --limit 5

python -m Agent.cli.evaluate --config Agent/configs/evaluations/locomo.yaml --profile qwen3_5_9b --resume

```

默认连接 `http://127.0.0.1:11434`；如 Ollama 位于其他地址，可设置 `OLLAMA_HOST`。
每个模型 profile 都显式声明 `context_window`。对 Ollama，该字段会作为 `num_ctx`
实际传给本地服务；`max_tokens` 则映射为 `num_predict`。

运行目录固定为 `runs/<yaml-name>-<profile>/`。`--resume` 断点续跑，`--rescore` 只重新评分。

LoCoMo 每段对话只构建一次记忆。运行器逐 session 输入对话并在 session 之间清空即时消息历史；
问答阶段也逐题清空上下文，只开放 `memory_search`。SQLite 数据库位于
`runs/locomo-<profile>/locomo_memory/`。

## 校验和测试

```bash
python Agent/tools/validate_configs.py
python -m unittest dataset_processing.tests.test_build Agent.tests.test_evaluation_v2
```
