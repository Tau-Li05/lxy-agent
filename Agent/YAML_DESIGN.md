# Agent Benchmark YAML 设计方案

## 1. 目标

本方案用于配置一套可扩展的 Agent Benchmark 运行与评测框架，使不同数据集、Agent Runtime 和官方评测器可以通过统一协议协作。

整体采用两层 YAML：

1. **Suite YAML**：定义一次评测需要运行哪些 Benchmark、Suite 级默认参数、局部覆盖、指标汇总和报告输出。
2. **Benchmark YAML**：定义单个 Benchmark 的数据加载、输入适配、Runtime 约束、输出适配、Submission 生成和官方评测协议。

当多个 Benchmark 需要复用同一数据源时，可再将 `dataset` 抽离成第三层 Dataset YAML；第一版不需要提前拆分。

## 2. 架构与数据流

```text
配置层
Suite YAML
    ↓
Benchmark YAML

样本执行层
Dataset Loader
    ↓
Raw Sample
    ├──→ Input Adapter → Unified AgentInput
    │                         ↓
    │                    Agent Runtime
    │                         ↓
    │                 Unified AgentOutput
    │                         ↓
    │                   Output Adapter
    │                         ↓
    │                      Prediction
    │                         │
    └──→ Reference Adapter → Reference
                              │
评测协议层                    ↓
Prediction[] ─→ Submission Builder ─→ BenchmarkSubmission
                                              │
Reference/Official Ground Truth ───────────┘
                                              ↓
                                      Official Evaluator
                                              ↓
                                      Per-sample Results

聚合报告层
Per-sample Results
    ↓
Benchmark Metric Aggregator
    ↓
Benchmark Metrics
    ↓
Suite Report Aggregator
    ↓
Suite Report
```

`Reference Adapter` 是可选节点。如果官方 Evaluator 自行读取标准答案，或使用隐藏测试集，框架只向 Evaluator 传递 `BenchmarkSubmission`。

## 3. 目录结构

```text
Agent/
├── configs/
│   ├── suites/
│   │   └── math_core.yaml
│   └── benchmarks/
│       ├── gsm8k.yaml
│       ├── aime24.yaml
│       └── gaia.yaml
├── schemas/
│   ├── suite-v1.schema.json
│   ├── benchmark-v1.schema.json
│   ├── agent-input-v1.schema.json
│   └── agent-output-v1.schema.json
└── YAML_DESIGN.md
```

YAML 用于声明组件和参数，JSON Schema/Pydantic 用于约束字段类型和做启动前校验。

## 4. 核心规则

### 4.1 事实来源

- Suite YAML 只做选择、组合和覆盖，不复制 Benchmark 的完整定义。
- Benchmark YAML 是评测协议的事实来源。
- Dataset 不等于 Benchmark；Benchmark 由 Dataset、Prompt/输入协议、运行约束、答案提取和 Evaluator 共同决定。
- 每次运行必须保存解析和覆盖后的完整配置。

### 4.2 配置覆盖优先级

从低到高为：

```text
框架内置默认值
< Benchmark YAML
< Suite YAML defaults
< Suite benchmarks[].overrides
< CLI 显式参数
```

覆盖采用递归 map merge；列表默认整体替换，不隐式追加。

### 4.3 路径解析

- YAML 内的相对路径一律相对于当前 YAML 文件所在目录。
- 可使用 `${DATA_ROOT}` 等环境变量指向机器本地的数据根目录。
- 可共享配置中不应写死用户专属绝对路径。
- 环境变量缺失、路径不存在或 checksum 不匹配时，应在启动前失败。

### 4.4 版本与复现

- 所有 YAML 必须声明 `schema_version`。
- 数据尽量记录 checksum、样本数或上游 revision。
- Evaluator 记录实现版本、代码 revision 或镜像 digest。
- Prompt、Output Adapter 或 Evaluator 发生变化时，应更新 `benchmark.protocol_version`。

## 5. Suite YAML

### 5.1 完整示例

```yaml
schema_version: "1.0"
kind: suite

suite:
  id: math-core
  version: "1.0"
  description: Core mathematical reasoning evaluation suite

defaults:
  runtime:
    limits:
      max_turns: 1
      max_steps: 20
      max_tool_calls: 10
      tokens:
        max_total: 16384
        max_output_per_call: 4096
      time:
        sample_timeout_seconds: 120
        tool_call_timeout_seconds: 30

  execution:
    retries: 1

benchmarks:
  - config: ../benchmarks/gsm8k.yaml
    enabled: true
    splits: [test]
    limit: null
    weight: 1.0

  - config: ../benchmarks/aime24.yaml
    enabled: true
    splits: [test]
    limit: null
    weight: 1.0
    overrides:
      runtime:
        limits:
          time:
            sample_timeout_seconds: 300

execution:
  concurrency: 4
  fail_fast: false
  random_seed: 42

aggregation:
  # 默认只展示各 Benchmark 指标，不强制生成综合总分。
  composite_score: null
  missing_result_policy: error

report:
  output_directory: ../../runs/${RUN_ID}
  formats: [json, jsonl]
  include_per_sample_results: true
  include_resolved_config: true
```

### 5.2 字段说明

| 字段 | 必填 | 作用 |
|---|---:|---|
| `schema_version` | 是 | YAML Schema 版本，用于校验和迁移 |
| `kind` | 是 | 固定为 `suite`，防止加载错误文件 |
| `suite.id` | 是 | Suite 全局唯一标识 |
| `suite.version` | 是 | Suite 组合与聚合规则的版本 |
| `defaults` | 否 | 对所有 Benchmark 生效的默认值 |
| `benchmarks[].config` | 是 | Benchmark YAML 路径 |
| `enabled` | 否 | 是否运行，默认 `true` |
| `splits` | 否 | 本次运行的数据切分 |
| `limit` | 否 | 调试或 smoke test 的样本上限 |
| `weight` | 否 | 显式启用综合分时的权重 |
| `overrides` | 否 | 当前 Suite 对单个 Benchmark 的覆盖 |
| `execution` | 否 | Suite 的并发、失败策略和随机种子 |
| `aggregation` | 否 | Benchmark 间的指标聚合规则 |
| `report` | 是 | 报告目录、格式和保留粒度 |

## 6. Benchmark YAML

### 6.1 基于本地 GSM8K Parquet 的示例

以项目中已经存在的 `benchmark/gsm8k/main/test-00000-of-00001.parquet` 为例：

```yaml
schema_version: "1.0"
kind: benchmark

benchmark:
  id: gsm8k
  version: "main"
  protocol_version: "exact-match-v1"
  description: Grade-school mathematical reasoning
  task_type: question_answering
  tags: [math, reasoning, text]

dataset:
  loader:
    type: parquet
    paths:
      train: ../../../benchmark/gsm8k/main/train-00000-of-00001.parquet
      test: ../../../benchmark/gsm8k/main/test-00000-of-00001.parquet

  integrity:
    # 建议在配置稳定后填入真实值。
    expected_samples:
      test: 1319
    sha256: {}

  sample_id:
    field: null
    strategy: content_hash

  fields:
    question: question
    reference_answer: answer

  filters: []

input_adapter:
  type: template
  input_fields: [question]
  template: |-
    Solve the following problem. Explain your reasoning, then provide the final
    answer in the form: #### <answer>

    {{ question }}

  output:
    messages:
      - role: user
        content: "{{ rendered_input }}"

agent_contract:
  input_schema: agent-input/v1
  output_schema: agent-output/v1
  required_capabilities: []
  allowed_tools: []

execution_environment:
  dependencies:
    type: venv
    python: "3.12"
    requirements: ../../requirements/gsm8k.lock
  sandbox:
    type: none

runtime:
  limits:
    max_turns: 1
    max_steps: 20
    max_tool_calls: 0
    tokens:
      max_total: 16384
      max_output_per_call: 4096
    time:
      sample_timeout_seconds: 120
      tool_call_timeout_seconds: 30
  accounting:
    failed_tool_calls: count
    parallel_tool_calls: per_call
    cached_model_calls: count

output_adapter:
  type: regex_extract
  source: final_answer
  pattern: '####\s*([-+]?[0-9,]+(?:\.[0-9]+)?)'
  on_no_match: invalid_prediction
  output:
    id_field: sample_id
    prediction_field: answer
  normalization:
    - strip
    - remove_commas

reference_adapter:
  type: regex_extract
  source_field: reference_answer
  pattern: '####\s*([-+]?[0-9,]+(?:\.[0-9]+)?)'
  output_field: reference
  normalization:
    - strip
    - remove_commas

submission:
  builder:
    type: jsonl
    sort_by: sample_id
    required_fields: [sample_id, answer]
  output:
    filename: predictions.jsonl

evaluator:
  mode: sample
  type: builtin
  implementation: exact_match
  prediction_field: answer
  reference_field: reference
  metrics:
    - id: exact_match
      primary: true
      direction: maximize
      range: [0, 1]

reproducibility:
  random_seed: 42
  shuffle: false

provenance:
  homepage: https://github.com/openai/grade-school-math
  license: MIT
  citation: null
```

### 6.2 顶层字段说明

| 字段 | 必填 | 作用 |
|---|---:|---|
| `schema_version` | 是 | Benchmark YAML Schema 版本 |
| `kind` | 是 | 固定为 `benchmark` |
| `benchmark` | 是 | Benchmark 身份、任务类型和评测协议版本 |
| `dataset` | 是 | 本地/远程数据的加载、切分、字段映射和完整性 |
| `input_adapter` | 是 | `Raw Sample → Unified AgentInput` |
| `agent_contract` | 是 | Agent 输入输出 Schema 和所需能力 |
| `execution_environment` | 是 | 依赖环境和安全沙箱 |
| `runtime` | 是 | 单样本执行限制与计数规则 |
| `output_adapter` | 是 | `Unified AgentOutput → Prediction` |
| `reference_adapter` | 否 | `Raw Reference → Reference` |
| `submission` | 是 | `Prediction[] → BenchmarkSubmission` |
| `evaluator` | 是 | 官方或内置评测器的调用协议 |
| `reproducibility` | 否 | 随机种子和数据顺序 |
| `provenance` | 否 | 上游来源、License 和引用信息 |

## 7. Dataset Loader

Loader 通过 `type` 进行分派。第一版建议支持：

### 7.1 本地 Parquet

```yaml
dataset:
  loader:
    type: parquet
    paths:
      test: ${DATA_ROOT}/gsm8k/main/test-00000-of-00001.parquet
```

### 7.2 本地 JSONL

```yaml
dataset:
  loader:
    type: jsonl
    paths:
      test: ${DATA_ROOT}/aime25/aime25.jsonl
    encoding: utf-8
```

### 7.3 Hugging Face `save_to_disk` 本地目录

```yaml
dataset:
  loader:
    type: huggingface_disk
    path: ${DATA_ROOT}/gsm8k-arrow
```

### 7.4 自定义 Python Loader

```yaml
dataset:
  loader:
    type: python
    entrypoint: agent_benchmarks.gaia.dataset:load
    kwargs:
      data_dir: ${DATA_ROOT}/GAIA/2023
```

`python` Loader 只允许加载框架注册表中已授权的 entrypoint，不应把 YAML 当作任意代码执行入口。

## 8. Unified AgentInput 和 AgentOutput

YAML 只引用 Schema 版本，实际字段由 JSON Schema/Pydantic 统一定义。

### 8.1 Unified AgentInput

```yaml
id: string
messages:
  - role: system | user | assistant | tool
    content: string
tools: []
attachments: []
environment: {}
metadata: {}
```

### 8.2 Unified AgentOutput

```yaml
id: string
status: success | timeout | limit_exceeded | error
final_answer: string | null
messages: []
tool_calls: []
artifacts: []
usage:
  prompt_tokens: 0
  completion_tokens: 0
  total_tokens: 0
timing:
  duration_seconds: 0.0
error: null
metadata: {}
```

Runtime 必须保留原始 AgentOutput。Output Adapter 或 Evaluator 变更后，可以基于原始输出重新评测，无需再调用模型。

## 9. Runtime 语义

```yaml
runtime:
  limits:
    max_turns: 1
    max_steps: 20
    max_tool_calls: 10
    tokens:
      max_total: 16384
      max_output_per_call: 4096
    time:
      sample_timeout_seconds: 120
      tool_call_timeout_seconds: 30
```

| 字段 | 精确语义 |
|---|---|
| `max_turns` | Agent 与用户或外部任务环境的最大高层交互轮数 |
| `max_steps` | Runtime 内部的最大决策循环次数；模型调用、工具结果处理和最终回答均需遵循统一计数规则 |
| `max_tool_calls` | 单样本实际发出的工具调用总数 |
| `tokens.max_total` | 单样本所有模型调用的 prompt + completion token 总上限 |
| `tokens.max_output_per_call` | 单次模型调用的最大输出 token |
| `sample_timeout_seconds` | 单样本从开始到结束的墙钟时间限制 |
| `tool_call_timeout_seconds` | 单次工具调用的墙钟时间限制 |

计数规则：

- 一次并行调用 3 个工具，计为 3 次 tool call。
- 失败、超时和被拒绝的工具调用也计数。
- 缓存命中的模型调用默认仍纳入 step 和 token 记录，并额外标记 `cached: true`。
- 达到任意一项限制时，Runtime 应停止执行并返回 `status: limit_exceeded`，不应抛弃已有轨迹。

## 10. 依赖环境与安全沙箱

`execution_environment` 明确分为两层：

```yaml
execution_environment:
  dependencies:  # 解决包和运行时版本
    type: venv
  sandbox:       # 限制代码可以访问什么
    type: none
```

### 10.1 venv：默认依赖环境

```yaml
execution_environment:
  dependencies:
    type: venv
    python: "3.12"
    requirements: ../../requirements/gsm8k.lock
  sandbox:
    type: none
```

`venv` 可隔离 Python 包，但不隔离文件系统、网络、进程、资源和密钥。它适用于 GSM8K、MMLU 等不执行不可信代码的 Benchmark。

配置应记录重建环境所需的 requirements/lock 文件，而不是把某台机器上的 venv 绝对路径作为唯一事实来源。

### 10.2 Conda：复杂依赖可选方案

```yaml
execution_environment:
  dependencies:
    type: conda
    lock_file: ../../environments/gaia-linux-64.lock
  sandbox:
    type: none
```

Conda 更适合 CUDA、非 Python 库或复杂二进制依赖，但同样不是安全沙箱。

### 10.3 Docker：执行不可信代码时的可选后端

```yaml
execution_environment:
  dependencies:
    type: image
  sandbox:
    type: docker
    image: humaneval-runner@sha256:...
    network: none
    filesystem:
      host: denied
      workspace: read_write
    resources:
      cpus: 2
      memory_mb: 2048
```

Docker 不是框架的全局强制依赖。只有声明 `sandbox.type: docker` 的 Benchmark 才要求运行机安装并启动 Docker。启动该 Benchmark 前应检查 Docker daemon 和镜像；不可用时应明确报错或按 Suite 策略跳过。

## 11. Output Adapter、Submission 和 Evaluator

### 11.1 Output Adapter

Output Adapter 只处理单样本，生成统一 Prediction：

```yaml
sample_id: test_001
answer: "42"
status: success
```

答案无法提取时，应返回结构化状态，不要静默转成空字符串：

```yaml
sample_id: test_001
answer: null
status: invalid_prediction
error:
  code: OUTPUT_PARSE_FAILED
  message: Final-answer pattern was not found
```

### 11.2 Submission Builder

Submission Builder 聚合 `Prediction[]`，负责官方提交格式、排序、必填字段和缺失预测策略。

```yaml
submission:
  builder:
    type: jsonl
    sort_by: sample_id
    required_fields: [sample_id, answer]
    missing_prediction_policy: error
  output:
    filename: predictions.jsonl
```

### 11.3 Evaluator 的三种模式

**显式 Reference 的逐样本评测：**

```yaml
evaluator:
  mode: sample
  type: builtin
  implementation: exact_match
```

**官方评测器自行读取答案：**

```yaml
reference_adapter:
  type: none

evaluator:
  mode: submission
  type: command
  command:
    - python
    - official_eval.py
    - --submission
    - "{{ submission_path }}"
    - --dataset
    - "${DATA_ROOT}/benchmark"
  version: "1.4.2"
```

**隐藏测试集/远程评测：**

```yaml
reference_adapter:
  type: none

evaluator:
  mode: remote_submission
  type: http
  endpoint_env: BENCHMARK_EVALUATION_ENDPOINT
  credential_env: BENCHMARK_API_KEY
```

YAML 只保存环境变量名，不保存 token、API key 等真实密钥。

## 12. Metrics 与报告

Benchmark Metrics 应同时报告指标和执行质量：

```yaml
benchmark_id: gsm8k
benchmark_protocol_version: exact-match-v1
metrics:
  exact_match: 0.823
counts:
  total: 1319
  evaluated: 1319
  success: 1300
  invalid_prediction: 4
  timeout: 15
  runtime_error: 0
```

不同 Benchmark 的指标含义不同，Suite 默认只分别展示，不自动求平均。如果确实需要综合分，必须显式配置：

```yaml
aggregation:
  composite_score:
    method: weighted_mean
    components:
      - benchmark: gsm8k
        metric: exact_match
        weight: 0.5
      - benchmark: aime24
        metric: accuracy
        weight: 0.5
```

## 13. 错误与失败策略

建议统一样本状态：

```text
success
invalid_input
runtime_error
timeout
limit_exceeded
invalid_prediction
evaluator_error
skipped
```

原则：

- 数据路径、Schema、必需能力、依赖环境在执行前校验。
- 单样本失败不应默认中止整个 Suite，由 `fail_fast` 控制。
- 重试只用于暂时性错误；输入无效、输出解析失败不自动重试。
- 所有失败保留结构化 error code、message 和已有轨迹。

## 14. 实现顺序

第一版建议按以下顺序落地：

1. 定义 Suite/Benchmark JSON Schema 和 Pydantic 模型。
2. 实现 YAML 加载、环境变量展开、相对路径解析、配置覆盖与校验。
3. 实现本地 `parquet` 和 `jsonl` Loader。
4. 实现 Unified AgentInput/AgentOutput 和基本 Runtime limits。
5. 实现 template Input Adapter、regex Output/Reference Adapter。
6. 实现 JSONL Submission Builder 和 builtin exact-match Evaluator。
7. 实现 Benchmark Metrics 和 Suite Report，保存 resolved config。
8. 实现 `venv` 能力检查；需要执行不可信代码时再增加 Docker backend。

## 15. 最小可行范围

第一版只需要支持：

- Suite YAML + Benchmark YAML；
- 本地 Parquet/JSONL；
- template Input Adapter；
- 单轮 Agent Runtime，但保留 turn/step/tool/token/time 限制字段；
- `venv + sandbox: none`；
- regex Output/Reference Adapter；
- JSONL Submission Builder；
- builtin exact match 和 command official evaluator；
- JSON/JSONL 报告与 resolved config 快照。

该范围可先覆盖 GSM8K、AIME 等本地数学评测，同时为 GAIA、BFCL、HumanEval 和 SWE-bench 等复杂任务保留扩展点。
