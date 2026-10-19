# Evaluation YAML v2

每个 YAML 描述一次可运行评测，只保留六类高价值配置：`dataset`、`task_family`、
`agent`、`tool`、`model` 和 `prompt`。完整结构由 `schemas/run-config-v2.schema.json` 约束。

- `dataset.name` 选择评分协议，`dataset.path` 指向统一 JSONL。
- `task_family` 只能是 `single_turn` 或 `multiturn`。
- `agent` 只暴露 runtime、最大步骤和样本超时。
- `tool` 选择工具来源及是否执行；BFCL 单轮只传 schema，多轮调用官方后端。
- `model.profiles` 定义可切换端点，`default_profile` 选择默认模型。
- 每个 profile 的 `context_window` 表示输入、历史和输出共享的总上下文预算；
  `max_tokens` 仅表示单次最大输出，并且必须小于 `context_window`。
- `prompt.system` 是唯一运行时 prompt 配置；用户题面不在 Agent 内加工。

JSONL 每行严格包含 `{input, reference}`。运行器只把 `input` 交给 Agent；`reference`
始终留在评分侧。相对路径相对于 YAML 文件解析。
