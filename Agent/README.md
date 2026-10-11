# Agent Benchmark 配置

本目录按 `YAML_DESIGN.md` 实现配置层与协议 Schema。Benchmark 原始数据仍保留在项目根目录的 `benchmark/` 下，Agent 相关声明全部位于本目录。

## 目录

```text
Agent/
├── configs/
│   ├── benchmarks/   # 单个 Benchmark 的完整协议事实来源
│   └── suites/       # Benchmark 选择、默认值、覆盖和报告配置
├── docs/             # 面向使用者的中文字段字典
├── schemas/          # Suite、Benchmark、Canonical Record、统一输入输出 Schema
├── tools/            # 配置静态校验工具
└── YAML_DESIGN.md
```

`configs/suites/core_benchmarks.yaml` 包含当前要求的 6 个 Benchmark。默认只运行 GAIA validation，因为 test 的答案是隐藏的；如需提交 GAIA test，可在 Suite 中把 split 改为 `test`，并提供远程评测所需环境变量。

BFCL 使用其官方 evaluator。运行前需安装 `benchmark/BFCL-main/berkeley-function-call-leaderboard`，设置 `BFCL_MODEL_NAME`，并让 Submission Builder 按 BFCL 官方目录格式写出模型结果。`all_scoring` 不包含非计分类别 `format_sensitivity`。

## 校验

```bash
python3 Agent/tools/validate_configs.py
```

校验器检查 JSON/YAML 语法、JSON Schema、Suite 引用、相对路径、样本数和已声明的 SHA-256。它不安装依赖，也不会运行模型或官方评测器。

`schemas/canonical-record-v1.schema.json` 定义原始数据标准化后的中间记录。训练数据转换应先生成 Canonical Record，再按 `usage` 分别导出 SFT、RL 和 TEST 视图，避免让各训练框架直接依赖不同 Benchmark 的原始字段。

Conversion YAML 的完整中文字段说明见 [docs/conversion-config-fields.zh-CN.json](docs/conversion-config-fields.zh-CN.json)。

## 确定性数据转换

完整校验但不发布：

```bash
python3 -m Agent.data_pipeline build --all --output-root data --verify-only
```

完整构建并发布：

```bash
python3 -m Agent.data_pipeline build --all --output-root data --publish
```

产物先完整写入 `data/.staging/<build-id>`，通过 Schema、数量、隔离性和双次确定性校验后移动到 `data/releases/<build-id>`，最后原子切换 `data/current`。单数据集构建只能用于开发检查，不能发布。

`validation` 是开发验证集，只能用于调参和模型选择；最终测试报告只能读取 `data/current/test/`。RL 的模型输入位于 `rl/policy/`，奖励参考位于 `rl/rewards/`，两者通过 ID 关联，不允许使用同一个 loader 加载。

运行测试：

```bash
python3 -m unittest Agent.tests.test_data_pipeline
```

## Competition MATH DEV Agent 评测

首版评测器读取 `data/current/dev/competition_math`，使用 LangChain
`create_agent` 调用 OpenAI-compatible 服务，并用 Math-Verify 评分。输入和参考答案由不同
Loader 路径读取；Runtime 的构造函数和 `run()` 接口都不会收到参考答案。

安装 Python 3.11 环境依赖：

```bash
python -m pip install -r requirements-agent-eval.txt
```

配置模型端点（变量名由 Agent YAML 声明）：

```bash
export MATH_AGENT_MODEL="your-model-name"
export MATH_AGENT_BASE_URL="http://127.0.0.1:8000/v1"
export MATH_AGENT_API_KEY="your-api-key"
```

先运行全部离线测试：

```bash
python -m unittest Agent.tests.test_data_pipeline Agent.tests.test_agent_evaluation
```

执行 5 条真实 DEV 输入的端点 smoke run，确认输出后用同一 `run_id` 断点续跑剩余样本：

```bash
python -m Agent.cli.evaluate --config Agent/configs/evaluations/competition_math_dev.yaml --limit 5
python -m Agent.cli.evaluate --config Agent/configs/evaluations/competition_math_dev.yaml --resume
```

同一 `run_id` 的中断任务使用 `--resume`；只修改答案解析或评分逻辑时使用 `--rescore`。
默认禁止覆盖已有 run。报告固定写入 `dev_metrics.json`，其中只含
`evaluation_scope=development` 的 validation 指标，不能作为最终测试成绩。

