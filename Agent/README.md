# Agent v3 评测说明

本目录实现统一的 v3 评测入口。Agent 只读取规范化 JSONL，原始数据处理由项目根目录的 `dataset_processing/` 完成。当前只保留 v3 schema 和三类 Runtime，不维护 v2/v1 兼容路径。

## 1. 数据处理

```bash
python -m dataset_processing.build_evaluation_data
python Agent/tools/validate_configs.py
```

输出位于 `data/evaluation/`：`gsm8k_test.jsonl`、`aime.jsonl`、`competition_math.jsonl`、`gaia_validation.jsonl`、`bfcl_single_turn.jsonl`、`bfcl_multi_turn.jsonl`、`bfcl_web_search.jsonl`、`bfcl_memory.jsonl`、`locomo_10pct.jsonl` 和 `manifest.json`。抽样使用带 seed 的 SHA-256 稳定抽样，并按题型/年份/类型/level/category/backend 分层。实际比例、seed、输入与输出 hash 以 `data/evaluation/manifest.json` 为准；BFCL prerequisite 保留但不计入 held-out 分数。每行遵循 `Agent/schemas/evaluation-record-v3.schema.json`。

## 2. 三类 Runtime

模型身份由 `--profile` 决定，Runtime 不包含模型名：

| Runtime | 用途 | 工具执行 | 增强能力 |
|---|---|---:|---|
| `llm_direct` | 单次模型生成；BFCL single-turn 只生成调用 | 否 | 无 |
| `bfcl_protocol` | BFCL 必需的最小多轮协议循环 | 是 | 禁用 planning、verification、compaction、自修正 |
| `qwen_harness` | Qwen 的 State 驱动 Harness 增益组 | 是 | feature flag 可独立开关 |

实现位于 `Agent/runtime/model_direct_runtime.py`、`protocol_runtime.py`、`qwen_harness.py`；状态在 `Agent/engine/state.py`，BFCL 执行器在 `Agent/runtime/bfcl_executor.py`。配置位于 `Agent/configs/v3/`。Direct/Protocol 提供 `v4flash`、`qwen3_5_4b`、`qwen3_5_9b`；Harness 用于 Qwen 4B/9B。LoCoMo 当前仅有 Qwen Harness 配置，尚无 DeepSeek LoCoMo YAML。

## 3. 配置与工具 schema

配置必须符合 `Agent/schemas/run-config-v3.schema.json`。数学工具 schema 在 `Agent/tools/math_tools.py`（Python/SymPy）；BFCL schema 来自样本 `input.tools`，由共享 `BFCLProtocolExecutor` 执行并以 `trajectory_id + tool_call_id` 幂等；GAIA schema 在 `Agent/tools/gaia_tools.py`；LoCoMo memory schema 由数据构建器写入；Harness 的 `create_plan` 为运行时内部工具。调用分别记录格式、选择、schema/语义参数、执行、结果引用和最终成功。

## 4. 运行

```bash
export FIRE_AGENT_BASE_URL=""
export FIRE_AGENT_API_KEY=""


python -m Agent.cli.evaluate --config Agent/configs/v3/aime_direct.yaml --profile v4flash --run-name aime-deepseek-direct
python -m Agent.cli.evaluate --config Agent/configs/v3/aime_direct.yaml --profile qwen3_5_9b --run-name aime-qwen9b-direct
python -m Agent.cli.evaluate --config Agent/configs/v3/aime_qwen_harness.yaml --profile qwen3_5_4b --run-name aime-qwen4b-harness
python -m Agent.cli.evaluate --config Agent/configs/v3/aime_qwen_harness.yaml --profile qwen3_5_9b --run-name aime-qwen9b-harness
python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_multi_protocol.yaml --profile v4flash --run-name bfcl-multi-deepseek-protocol
python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_multi_protocol.yaml --profile qwen3_5_9b --run-name bfcl-multi-qwen9b-protocol
python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_multi_qwen_harness.yaml --profile qwen3_5_9b --run-name bfcl-multi-qwen9b-harness
python -m Agent.cli.evaluate_passk --config Agent/configs/v3/aime_qwen_harness.yaml --profile qwen3_5_4b
```

`--limit N` 做 smoke test，`--resume` 继续运行，`--rescore` 只重算评分。Web search 需要后端环境变量；BFCL 后端从仓库源码加载，无需 `pip install -e`。

## 5. Harness 流程

`QwenHarnessRuntime` 为每条 trajectory 创建 `AgentState`，执行 system prompt → 模型响应 → 解析调用 → 工具 → 写回结果 → 可选验证/压缩/自修正，最后生成 `AgentOutput`。状态包括消息、计划、步骤、工具结果、验证、重试和最终答案。关键阶段写 checkpoint，其余写 delta、前后 hash 和事件；system prompt 由 `Agent/prompting/system_prompt.py` 组合。

`features.parallel_tools` 控制同一模型回合中的外部工具并发，`max_concurrency` 设置上限。Planning 工具保持顺序执行；数学与 GAIA 工具并行运行并实时产生 `tool_started/tool_completed` 事件，结果按原始调用顺序写回。BFCL 官方 batch 因共享状态保持原子执行。`QueryEngine.stream()` 可消费这些事件，并以 `query_completed` 返回最终输出。

## 6. 日志与指标

每个 run 在 `runs/<run-name>/`，包含 `manifest.json`、`resolved_config.json`、`samples.jsonl`、`trajectories.jsonl`、`events.jsonl`、`checkpoints.jsonl`、`metric_details.jsonl`、`sample_results.jsonl`、`agent_outputs.jsonl`、`metrics.json` 和 `artifacts/`。manifest 记录模型/provider、配置和数据 hash、feature flags、环境；大文本和网页正文脱敏压缩到 artifacts。缺少 usage、相关性金标或最短路径标注时输出 `unavailable`、原因和 coverage。

```bash
python -m Agent.cli.compare_runs MODEL_DELTA <deepseek_run> <qwen_protocol_run>
python -m Agent.cli.compare_runs HARNESS_GAIN <qwen_protocol_run> <qwen_harness_run>
python -m Agent.cli.compare_runs FINAL_DELTA <deepseek_protocol_run> <qwen_harness_run>
```

RE 区分 provider/visible/output token；跨模型报告 normalized token 与时间效率。Pass@k 使用独立 trajectory，retry 不增加 k；Step Accuracy 只计可验证步骤并报告 coverage；LoCoMo Context/Memory 与 BFCL Memory 分开报告。

## 7. 测试

```bash
python -m unittest discover -s Agent/tests -p 'test_*.py'
python -m unittest discover -s dataset_processing/tests -p 'test_*.py'
python -m compileall -q Agent dataset_processing
git diff --check
```

公平比较必须匹配数据、seed、prompt、预算和模型版本；先用 `--limit` 确认 provider、模型 tag、tool-call 能力和输出目录，再运行完整样本。

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/aime_direct.yaml --profile v4flash --run-name aime-deepseek-direct


python -m Agent.cli.evaluate --config Agent/configs/v3/gsm8k_direct.yaml --profile v4flash --run-name gsm8k-deepseek-direct

python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_single_direct.yaml --profile v4flash --run-name bfcl_single-deepseek-direct

python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_multi_protocol.yaml --profile v4flash --run-name bfcl_multi-deepseek-protocol


python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_web_protocol.yaml --profile v4flash --run-name bfcl_web-deepseek-protocol


python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_memory_protocol.yaml --profile v4flash --run-name bfcl_memory-deepseek-protocol


python -m Agent.cli.evaluate --config Agent/configs/v3/gsm8k_direct.yaml --profile qwen3_5_9b --run-name gsm8k-qwen9b-direct

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_single_direct.yaml --profile qwen3_5_9b --run-name bfcl_single-qwen9b-direct

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_multi_protocol.yaml --profile qwen3_5_9b --run-name bfcl_multi-qwen9b-protocol

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_web_protocol.yaml --profile qwen3_5_9b --run-name bfcl_web-qwen9b-protocol

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/gsm8k_qwen_harness.yaml --profile qwen3_5_9b --run-name gsm8k-qwen9b-harness

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/aime_qwen_harness.yaml --profile qwen3_5_9b --run-name aime-qwen9b-harness



caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/aime_qwen_harness.yaml --profile v4flash --run-name aime-v4flash-harness


caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/aime_qwen_harness.yaml --profile qwen3_5_9b --run-name aime-qwen9b-harness

caffeinate -i python -m Agent.cli.evaluate --config Agent/configs/v3/bfcl_multi_qwen_harness.yaml --profile qwen3_5_9b --run-name bfcl_multi-qwen9b-harness
