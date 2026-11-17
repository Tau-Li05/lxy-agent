# 数学工具与 AIME 运行说明

这份文档说明当前 `qwen_harness` 的数学工具、反馈和收尾流程。AIME 的基准配置是 `Agent/configs/v3/aime_qwen_harness.yaml`。AIME 题内 memory 实验已移除；`features.memory.enabled` 在该配置中为 `false`。`compaction` 是独立的上下文压缩功能。

## 工具接口

| 工具 | 用途 | 输入 |
|---|---|---|
| `python_calculate` | 有界枚举、动态规划、符号计算、约束回代 | Python `code`，可选 `purpose` 和 `constraints` |
| `exact_calculate` | 短的精确整数、分数和取模计算 | `expression` |

`python_calculate` 可导入 `math`、`fractions`、`decimal`、`itertools`、`functools`、`collections`、`sympy` 和 `json`；不允许文件或网络访问。代码最多 10,000 字符，默认工具超时由配置控制（AIME 基准配置为 5 秒）。`exact_calculate` 支持 `+ - * / // % **`、括号和 `abs/gcd/lcm/factorial/comb/isqrt`，表达式最多 2,000 字符；变量、赋值和任意 Python 代码应改用 `python_calculate`。

开启 `features.math_tools.candidate_protocol` 后，Python 工具才向模型公开以下可选参数；AIME 基准配置中该开关为 `false`，公开的参数只有 `code`：

- `purpose`：`calculate`（默认）、`enumerate` 或 `verify`。
- `constraints`：最多 16 条非空字符串，记录调用者认为需要检查的条件。

当调用明确指定 `purpose=enumerate` 时，必须提供 `constraints`，代码中必须有 `assert`，且运行时至少执行一个断言。断言是否真的覆盖原题条件仍需核对；工具无法自动判断建模是否正确。

## 怎样读工具反馈

反馈是 JSON。重点查看 `success`、`execution_succeeded`、`usable_result`、`error_type`、`stdout`、`stderr`、`timed_out`、`truncated` 和 `assertions_passed`。

- Python 正常退出但没有可用 stdout：`empty_output`。此时 `execution_succeeded=true`，但 `success=false`、`usable_result=false`。
- 输出过长被截断：`output_truncated`。缩短调试输出，只打印检查数量和相关结果。
- 超时：`timeout`。先从题目条件推出有限范围，再缩小搜索或换算法。
- 相同失败调用再次出现：`duplicate_failed_call`；修改代码或方法后才能重新执行。
- 连续修改代码却遇到同类 `TypeError` 或 `AttributeError`：反馈可带 `same_error_diagnostic`，提示先检查报错行的变量类型和值。

代码运行成功只说明代码产生了观察值，**不证明它检查了原题全部条件，也不证明搜索范围完整**。例如枚举应明确说明范围、检查了多少候选，以及接受候选时执行了哪些条件检查。

## 候选答案与证据

数学工作流通过 `features.math_tools.workflow_enabled` 开启；AIME 基准配置已开启。它把中间计算结果与最终答案候选分开记录。程序若要明确提交候选，可在最后打印一行 JSON：

```python
import json

# 先完成计算，并检查自己声明的约束。
print(json.dumps({
    "candidate_answer": 279,
    "method": "bounded_enumeration",
    "checked_count": 1000,
    "search_bounds": "1 <= n <= 1000",
    "search_complete": True,
}))
```

`candidate_answer` 是候选，不是正确性证明。`checked_count`、`search_bounds` 和 `search_complete` 是程序报告；运行器不能自动证明范围覆盖了原题。裸数字 stdout、`exact_calculate` 的计算值和失败前的部分输出都不会自动成为已验证的最终答案。

证据记录在 `AgentState.math_evidence` 和运行轨迹中。常见事件包括 `math_evidence_updated`、`finalization_evidence` 与 `finalization_context`。只有启用相应开关时才会有缓存复用或提前停止：`cache_successes`、`stop_on_repetition`、`candidate_protocol` 默认均关闭。第二次等价失败后安排无工具修复轮的 `repair_turn_enabled` 也默认关闭。

## 最终回答与实验开关

AIME 基准配置开启 `finalization`，预留最终回答上限 **2048 tokens**。最终轮不提供工具。`adaptive` 策略只在有合格候选时使用更紧凑的证据；没有候选时仍需从保留的历史继续求解，所以最终轮仍可能触及上限。`finalization_started` 事件的 `reason` 可帮助区分输出触顶、token 预留和轮次限制。

以下开关用于单独实验，默认关闭，不能单凭开启就认定解题质量提高：

| 开关 | 作用 |
|---|---|
| `pre_final_audit` | 首轮提醒核对题目目标，首次工具反馈后再提醒核查条件与答案形式 |
| `concise_progress` | 非最终轮临时加入简短进展提示，不增加历史消息 |
| `bounded_turns` | 限制非最终轮输出，并允许有限次数续解；最终 2048 上限不变 |

比较实验应固定题目、模型、工具和最终回答上限，只改变一个开关，并使用新的 `--run-name`。不要用 `--resume` 把不同配置或实现混入旧运行目录。结果应同时统计**正确、错误、截断、工具错误与 token 用量**；减少截断但输出错误答案不算正确率改善。
