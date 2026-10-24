from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any

from Agent.engine import AgentPhase, AgentState, ContextManager, Transition, TransitionKind, execute_parallel
from Agent.evaluation.models import AgentOutput
from Agent.runtime.direct_runtime import (
    DirectLangChainRuntime, _accumulate_usage, _call_string, _content_text, _done_reason,
    _empty_usage, _invoke_with_xml_retry, _malformed_retry_limit, _openai_tool, _response_usage,
    _serialize_message, _tool_calls,
)
from Agent.runtime.bfcl_executor import ToolExecutionResult
from Agent.telemetry import RunRecorder
from Agent.tools.math_tools import execute_math_tool, math_tool_schemas
from Agent.prompting import SystemPromptBuilder
from Agent.tools.gaia_tools import GAIA_TOOLS, execute_gaia_tool


_PLANNING_TOOL_NAMES = {"create_plan", "task_list", "task_update"}


def _planning_tools() -> list[dict[str, Any]]:
    task = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["analysis", "execution", "verification", "synthesis"]},
            "blocked_by": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "subject", "description", "kind", "blocked_by"],
        "additionalProperties": False,
    }
    return [
        {
            "name": "create_plan",
            "description": (
                "Create or revise a structured task plan for a genuinely complex problem. "
                "Tasks form a dependency graph and should include verification when useful."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "tasks": {"type": "array", "items": task},
                    "reason": {"type": "string"},
                },
                "required": ["goal", "tasks"],
                "additionalProperties": False,
            },
        },
        {
            "name": "task_list",
            "description": "List the current plan, task states, dependencies, and runnable tasks.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "task_update",
            "description": (
                "Move one plan task through pending, in_progress, completed, failed, or skipped. "
                "Complete a task only after its success criteria are satisfied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed", "skipped"]},
                    "result": {"type": "string"},
                    "error": {"type": "string"},
                },
                "required": ["task_id", "status"],
                "additionalProperties": False,
            },
        },
    ]


def _runnable_task_ids(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return []
    tasks = {task["id"]: task for task in plan.get("tasks", [])}
    return [
        task["id"] for task in tasks.values()
        if task["status"] == "pending" and all(
            tasks.get(dependency, {}).get("status") in {"completed", "skipped"}
            for dependency in task.get("blocked_by", [])
        )
    ]


def _build_plan(args: dict[str, Any], previous: dict[str, Any] | None,
                planning: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    tasks = args.get("tasks") or []
    max_tasks = int(planning.get("max_tasks", 8))
    if not tasks or len(tasks) > max_tasks:
        return None, f"Plan must contain between 1 and {max_tasks} tasks."
    ids = [str(task.get("id", "")).strip() for task in tasks]
    if any(not task_id for task_id in ids) or len(set(ids)) != len(ids):
        return None, "Every task must have a unique non-empty id."
    known = set(ids)
    if any(set(task.get("blocked_by", [])) - known for task in tasks):
        return None, "Every blocked_by dependency must reference a task in this plan."
    graph = {task_id: list(task.get("blocked_by", [])) for task_id, task in zip(ids, tasks)}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return False
        if task_id in visited:
            return True
        visiting.add(task_id)
        if not all(visit(dependency) for dependency in graph[task_id]):
            return False
        visiting.remove(task_id)
        visited.add(task_id)
        return True

    if not all(visit(task_id) for task_id in ids):
        return None, "Task dependencies must not contain a cycle."
    if planning.get("require_verification_task") and len(tasks) >= 3 and not any(
        task.get("kind") == "verification" for task in tasks
    ):
        return None, "Plans with three or more tasks must include a verification task."
    version = int((previous or {}).get("version", 0)) + 1
    if version > int(planning.get("max_replans", 1)) + 1:
        return None, "Plan revision limit reached."
    normalized = []
    for task_id, task in zip(ids, tasks):
        normalized.append({
            "id": task_id, "subject": task["subject"], "description": task["description"],
            "kind": task["kind"], "blocked_by": list(task.get("blocked_by", [])),
            "status": "pending", "result": None, "error": None,
        })
    return {
        "version": version, "goal": args.get("goal", ""), "reason": args.get("reason", ""),
        "tasks": normalized, "active_task_id": None,
    }, None


def _update_plan_task(plan: dict[str, Any] | None, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if not plan:
        return plan, "No plan exists. Call create_plan first."
    task = next((row for row in plan["tasks"] if row["id"] == args.get("task_id")), None)
    if task is None:
        return plan, f"Unknown task id: {args.get('task_id')}"
    status = args.get("status")
    tasks = {row["id"]: row for row in plan["tasks"]}
    if status == "in_progress":
        blockers = [dep for dep in task["blocked_by"] if tasks[dep]["status"] not in {"completed", "skipped"}]
        if blockers:
            return plan, "Task is blocked by incomplete tasks: " + ", ".join(blockers)
        active = plan.get("active_task_id")
        if active and active != task["id"]:
            return plan, f"Task {active} is already in progress."
        if task["status"] not in {"pending", "in_progress"}:
            return plan, f"Cannot start a task with status {task['status']}."
        task["status"] = "in_progress"
        plan["active_task_id"] = task["id"]
    elif status in {"completed", "failed", "skipped"}:
        if task["status"] != "in_progress":
            return plan, "Only an in_progress task can be resolved."
        if status == "completed" and not str(args.get("result", "")).strip():
            return plan, "Completing a task requires a non-empty result."
        task["status"] = status
        task["result"] = args.get("result")
        task["error"] = args.get("error")
        plan["active_task_id"] = None
    elif status == "pending":
        if task["status"] not in {"failed", "in_progress"}:
            return plan, f"Cannot reset a task with status {task['status']}."
        task["status"], task["result"], task["error"] = "pending", None, None
        if plan.get("active_task_id") == task["id"]:
            plan["active_task_id"] = None
    else:
        return plan, f"Unsupported task status: {status}"
    return plan, json.dumps({"task": task, "runnable_task_ids": _runnable_task_ids(plan)}, ensure_ascii=False)


def _plan_complete(plan: dict[str, Any] | None) -> bool:
    return bool(plan and all(task["status"] in {"completed", "skipped"} for task in plan["tasks"]))


class QwenHarnessRuntime(DirectLangChainRuntime):
    """State-driven harness loop for Ollama and OpenAI-compatible profiles."""

    runtime_kind = "qwen_harness"

    def __init__(self, config: dict[str, Any], project_root: Path, profile_name: str | None = None,
                 model: Any | None = None) -> None:
        super().__init__(config, project_root, profile_name, model)
        self.features = config.get("features", {})
        self._bfcl_batch_lock = asyncio.Lock()

    async def _run(self, item: dict[str, Any], started: float, attempt: int = 1) -> AgentOutput:
        if item["metadata"]["benchmark"] == "locomo":
            memory_feature = self.features.get("memory", {})
            memory_mode = memory_feature.get("mode", "persistent" if memory_feature.get("enabled") else "full_context")
            if memory_mode == "persistent":
                output = await super()._run_locomo(item, started, attempt)
                output.metadata["runtime"] = self.runtime_kind
                output.metadata["features"] = self.features
                return output
            return await self._run_locomo_context(item, started, attempt, include_conversation=memory_mode == "full_context")
        trajectory_id = str(uuid.uuid4())
        recorder = RunRecorder(
            self.run_state_root, experiment_id=self.run_state_root.name,
            runtime_kind=self.runtime_kind, model_profile=self.profile_name,
            sample_id=item["id"], trajectory_id=trajectory_id,
        )
        router = self.features.get("difficulty_router", {})
        route = "agent"
        router_usage = _empty_usage()
        router_raw: str | None = None
        if router.get("enabled"):
            question = "\n".join(
                str(message.get("content", "")) for message in (item.get("turns") or [[]])[0]
                if isinstance(message, dict) and message.get("role") == "user"
            )
            router_messages = [
                {
                    "role": "system",
                    "content": (
                        "Classify the math problem without solving it. Reply with exactly DIRECT or AGENT. "
                        "Choose DIRECT for a clear problem solvable by a short arithmetic chain. Choose AGENT "
                        "only when the problem has multiple dependent states, genuine interpretive ambiguity, "
                        "several linked derivations, or needs explicit calculation and verification."
                    ),
                },
                {"role": "user", "content": question},
            ]
            recorder.event("router_request", {"messages": recorder.bounded_payload(router_messages)}, attempt_id=attempt)
            router_response = await self.model.ainvoke(router_messages)
            router_value = _serialize_message(router_response)
            _accumulate_usage(router_usage, router_value)
            router_raw = _content_text(router_value.get("content")) or ""
            labels = re.findall(r"\b(DIRECT|AGENT)\b", router_raw.upper())
            if len(set(labels)) == 1:
                route = labels[0].lower()
            else:
                route = str(router.get("fallback", "direct"))
            recorder.event("router_decision", {
                "route": route, "raw_output": router_raw,
                "method": router.get("method", "model_judgment"),
                "fallback_used": len(set(labels)) != 1,
            }, attempt_id=attempt)
        messages: list[Any] = []
        base_system = self.config["prompt"]["system"]
        turns = deepcopy(item["turns"])
        permissions = item.get("environment", {}).get("_query_permissions") or {}
        query_event_sink = item.get("environment", {}).get("_query_event_sink")

        async def emit_query_event(event: dict[str, Any]) -> None:
            if query_event_sink is None:
                return
            pending = query_event_sink(event)
            if hasattr(pending, "__await__"):
                await pending
        agent_route = route == "agent"
        tools_enabled = agent_route and bool(
            self.features.get("tools", {}).get("enabled", self.config["tool"]["enabled"])
        ) and permissions.get("allow_tools", True)
        execute_tools = bool(self.features.get("tools", {}).get(
            "execution", self.config["tool"]["execution"] == "enabled")) and permissions.get("allow_tool_execution", True)
        tools = deepcopy(item["tools"]) if tools_enabled else []
        math_mode = self.features.get("math_tools", {}).get("mode", "none")
        if tools_enabled and self.config["tool"]["provider"] == "math":
            tools = math_tool_schemas(math_mode)
        if tools_enabled and self.config["tool"]["provider"] == "gaia":
            tools = deepcopy(GAIA_TOOLS)
        planning = self.features.get("planning", {}) if agent_route else {"enabled": False}
        if planning.get("enabled"):
            tools = [*_planning_tools(), *tools]
            base_system += (
                "\nFor genuinely complex tasks, create a structured plan before external actions. "
                "Use task_update to mark exactly one runnable task in_progress, perform that task, then "
                "mark it completed with a concrete result. Respect dependencies and finish verification "
                "tasks before the final answer. Simple tasks may be answered directly in adaptive mode."
            )
        initial_additions = item["environment"].get("missed_function", {}).get("0", [])
        if initial_additions:
            tools.extend(initial_additions)
        allowed_names = permissions.get("allowed_tool_names")
        if allowed_names is not None:
            tools = [tool for tool in tools if tool["name"] in set(allowed_names)]
        system = SystemPromptBuilder(self.project_root).build(
            base_system, self.config, tool_names=[tool["name"] for tool in tools]
        )
        if system:
            messages.append({"role": "system", "content": system})
        turn_index = 0
        current_turn_calls: list[list[str]] = []
        if turns:
            messages.extend(turns[0])
        state = AgentState(messages=messages, phase=AgentPhase.RUNNING,
                           transition=Transition(TransitionKind.CALL_MODEL), usage=_empty_usage())
        for key, value in router_usage.items():
            if value is not None:
                state.usage[key] = int(state.usage.get(key) or 0) + int(value)
        max_turns = int(self.config.get("limits", {}).get("max_model_turns", self.config["agent"]["max_steps"]))
        max_tool_calls = int(self.config.get("limits", {}).get("max_tool_calls", max_turns * 4))
        serialized: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        calls_by_turn: list[list[list[str]]] = []
        verification_active = False
        answer_before_verification: str | None = None
        compaction_count = 0
        finalization_started = False
        transition_count = 0
        compaction_checked_turn = -1
        context_manager = ContextManager.from_config(
            self.features.get("compaction", {}), self.profile,
            self.config.get("limits", {}).get("context_window_override"),
        )

        def set_transition(kind: TransitionKind, reason: str, payload: dict[str, Any] | None = None) -> None:
            nonlocal transition_count
            previous = state.transition.kind if state.transition else None
            state.transition = Transition(kind, reason=reason, payload=payload or {})
            transition_count += 1
            recorder.event("transition", {
                "from": previous, "to": kind, "reason": reason, "payload": payload or {},
            }, attempt_id=attempt)

        def advance_turn_or_finish() -> None:
            nonlocal turn_index, current_turn_calls, verification_active, answer_before_verification
            calls_by_turn.append(current_turn_calls)
            if turn_index + 1 >= len(turns):
                set_transition(TransitionKind.FINISH, "all user turns completed")
                return
            turn_index += 1
            additions = item["environment"].get("missed_function", {}).get(str(turn_index), [])
            if additions:
                tools.extend(additions)
            next_turn = turns[turn_index] or [
                {"role": "user", "content": "I have updated some more functions you can choose from. What about now?"}
            ]
            state.messages.extend(next_turn)
            current_turn_calls = []
            verification_active = False
            answer_before_verification = None
            recorder.event("user_turn_started", {"turn_index": turn_index, "messages": next_turn}, attempt_id=attempt)
            set_transition(TransitionKind.CALL_MODEL, "next user turn")

        while state.phase == AgentPhase.RUNNING:
            transition = state.transition or Transition(TransitionKind.FAIL, reason="missing transition")

            if transition.kind == TransitionKind.FINISH:
                state.phase = AgentPhase.FINISHED
                state.transition = None
                break

            if transition.kind == TransitionKind.FAIL:
                if state.error is None:
                    state.error = {"type": "invalid_state", "message": transition.reason or "state machine failed"}
                state.phase = AgentPhase.FAILED
                state.transition = None
                break

            if transition.kind == TransitionKind.COMPACT:
                compaction_checked_turn = state.model_turn_count
                result = context_manager.compact(state.messages, artifact_writer=recorder.artifact)
                state.messages = result.messages
                if result.changed:
                    compaction_count += 1
                    recorder.event("context_compacted", {
                        "original_characters": result.original_characters,
                        "compressed_characters": result.compacted_characters,
                        "original_tokens_estimate": result.original_tokens,
                        "compacted_tokens_estimate": result.compacted_tokens,
                        "compression_ratio": (
                            result.compacted_characters / result.original_characters
                            if result.original_characters else 1.0
                        ),
                        "actions": result.actions,
                    }, attempt_id=attempt)
                    recorder.checkpoint(state, kind="compaction", attempt_id=attempt)
                set_transition(TransitionKind.CALL_MODEL, "compaction completed")
                continue

            if transition.kind == TransitionKind.RECOVER:
                recovery = self.features.get("output_recovery", {})
                if recovery.get("enabled") and state.output_recovery_count < int(recovery.get("max_retries", 0)):
                    state.output_recovery_count += 1
                    state.messages.append({
                        "role": "user",
                        "content": "Continue concisely from the partial response and finish with the required final answer.",
                    })
                    recorder.event("output_recovery", {"recovery_count": state.output_recovery_count}, attempt_id=attempt)
                    set_transition(TransitionKind.CALL_MODEL, "output recovery requested")
                else:
                    state.error = {"type": "output_token_limit", "message": "model output reached max_tokens"}
                    set_transition(TransitionKind.FAIL, "output recovery unavailable")
                continue

            if transition.kind == TransitionKind.VERIFY:
                verification = self.features.get("verification", {})
                if transition.payload.get("action") == "start":
                    verification_active = True
                    answer_before_verification = state.final_answer
                    method = verification.get("method", "self")
                    state.messages.append({
                        "role": "user",
                        "content": (
                            f"Verify the proposed answer carefully using {method}. "
                            "Begin with exactly VERIFICATION: CORRECT or VERIFICATION: INCORRECT. "
                            "If it is wrong, correct it. Return the complete final answer in the required format."
                        ),
                    })
                    recorder.event("verification_started", {
                        "method": method, "answer_before": answer_before_verification,
                    }, attempt_id=attempt)
                    set_transition(TransitionKind.CALL_MODEL, "verification prompt added")
                else:
                    state.verification_result = {
                        "method": verification.get("method", "self"),
                        "answer_before": answer_before_verification,
                        "answer_after": state.final_answer,
                        "answer_changed": answer_before_verification != state.final_answer,
                        "error_detected": "VERIFICATION: INCORRECT" in (state.final_answer or "").upper(),
                    }
                    recorder.event("verification_finished", state.verification_result, attempt_id=attempt)
                    recorder.checkpoint(state, kind="verification", attempt_id=attempt)
                    advance_turn_or_finish()
                continue

            if transition.kind == TransitionKind.EXECUTE_TOOLS:
                calls = list(state.pending_tool_calls)
                if state.tool_call_count + len(calls) > max_tool_calls:
                    state.error = {"type": "max_tool_calls", "message": "tool call budget exhausted"}
                    set_transition(TransitionKind.FAIL, "tool call budget exhausted")
                    continue
                call_strings = [_call_string(call) for call in calls]
                external_calls = [call for call in calls if call["name"] not in _PLANNING_TOOL_NAMES]
                internal_calls = [call for call in calls if call["name"] in _PLANNING_TOOL_NAMES]
                for call in internal_calls:
                    old_plan = deepcopy(state.current_plan)
                    if call["name"] == "create_plan":
                        new_plan, error = _build_plan(call["args"], state.current_plan, planning)
                        if error:
                            content = "Plan rejected: " + error
                        else:
                            state.current_plan = new_plan
                            content = json.dumps({"plan": state.current_plan,
                                                  "runnable_task_ids": _runnable_task_ids(state.current_plan)}, ensure_ascii=False)
                            recorder.event("plan_created" if old_plan is None else "plan_revised",
                                           {"plan": state.current_plan, "previous_plan": old_plan}, attempt_id=attempt)
                    elif call["name"] == "task_list":
                        content = json.dumps({"plan": state.current_plan,
                                              "runnable_task_ids": _runnable_task_ids(state.current_plan)}, ensure_ascii=False)
                    else:
                        state.current_plan, content = _update_plan_task(state.current_plan, call["args"])
                        updated_task = next((task for task in (state.current_plan or {}).get("tasks", [])
                                             if task["id"] == call["args"].get("task_id")), None)
                        recorder.event("plan_task_updated", {
                            "plan_version": (state.current_plan or {}).get("version"),
                            "task_id": call["args"].get("task_id"), "status": call["args"].get("status"),
                            "accepted": bool(updated_task and updated_task["status"] == call["args"].get("status")),
                        }, attempt_id=attempt)
                    message = {"role": "tool", "tool_call_id": call["id"], "content": content}
                    state.messages.append(message)
                    serialized.append(message)
                unplanned_calls: list[dict[str, Any]] = []
                if state.current_plan and not state.current_plan.get("active_task_id"):
                    unplanned_calls, external_calls = external_calls, []
                for call in unplanned_calls:
                    content = "External action rejected: mark one runnable plan task in_progress first."
                    message = {"role": "tool", "tool_call_id": call["id"], "content": content}
                    state.messages.append(message)
                    state.tool_results.append({"call": call, "result": content, "success": False})
                    serialized.append(message)
                    recorder.event("unplanned_tool_call", {
                        "call": call, "plan_version": state.current_plan.get("version"),
                    }, attempt_id=attempt)
                call_strings = [_call_string(call) for call in external_calls]
                for call in external_calls:
                    recorder.event("external_action", {
                        "call": call, "plan_version": (state.current_plan or {}).get("version"),
                        "planned": state.current_plan is not None,
                        "task_id": (state.current_plan or {}).get("active_task_id"),
                    }, attempt_id=attempt)
                parallel = self.features.get("parallel_tools", {})
                max_concurrency = int(parallel.get("max_concurrency", 4)) if parallel.get("enabled", False) else 1
                if self.config["tool"]["provider"] in {"math", "gaia"}:
                    async def execute_external(call: dict[str, Any]) -> Any:
                        if self.config["tool"]["provider"] == "math":
                            return await asyncio.to_thread(execute_math_tool, call["name"], call["args"])
                        return await asyncio.to_thread(
                            execute_gaia_tool, call["name"], call["args"], item["environment"]
                        )

                    completed = await execute_parallel(
                        external_calls, execute_external, max_concurrency=max_concurrency,
                        event_sink=query_event_sink,
                    )
                    results = [
                        ToolExecutionResult(
                            f"{trajectory_id}:{entry.call['id']}",
                            json.dumps({"error": str(entry.value)}, ensure_ascii=False), False, False,
                            {"message": str(entry.value)},
                        ) if isinstance(entry.value, Exception) else entry.value
                        for entry in completed
                    ]
                else:
                    if query_event_sink is not None:
                        for index, call in enumerate(external_calls):
                            await emit_query_event({"event": "tool_started", "index": index,
                                                    "tool_call_id": call["id"], "name": call["name"]})
                    # The official BFCL backend mutates shared prerequisite state.
                    # Keep each official batch atomic while moving it off the loop.
                    async with self._bfcl_batch_lock:
                        results = await asyncio.to_thread(
                            self.bfcl_executor.execute_batch, call_strings, item=item,
                            trajectory_id=trajectory_id,
                            tool_call_ids=[call["id"] for call in external_calls], attempt=attempt,
                        )
                    if query_event_sink is not None:
                        for index, (call, result) in enumerate(zip(external_calls, results)):
                            await emit_query_event({"event": "tool_completed", "index": index,
                                                    "tool_call_id": call["id"], "name": call["name"],
                                                    "success": result.success})
                all_calls.extend(calls)
                current_turn_calls.append(call_strings)
                state.tool_call_count += len(calls)
                for call, result in zip(external_calls, results):
                    result_payload: Any = result
                    if len(result.output) > 2000:
                        artifact = recorder.artifact({"tool": call["name"], "output": result.output})
                        result_payload = {"success": result.success, "error": result.error,
                                          "artifact": artifact, "preview": result.output[:1000]}
                    event_type = "search_result" if call["name"] in {"search_engine_query", "fetch_url_content"} else "tool_result"
                    event_payload = {"call": call, "result": result_payload}
                    if event_type == "search_result":
                        event_payload["capture"] = {
                            "mode": self.features.get("web_search", {}).get("mode", "live"),
                            "backend": os.getenv("BFCL_SEARCH_BACKEND", "ddgs"),
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        }
                    if call["name"] == "search_engine_query":
                        try:
                            search_rows = json.loads(result.output)
                            event_payload["ranked_results"] = [
                                {"rank": index + 1, "url": row.get("href"), "title": row.get("title"),
                                 "snippet": row.get("body")}
                                for index, row in enumerate(search_rows) if isinstance(row, dict)
                            ]
                        except (ValueError, TypeError):
                            event_payload["ranked_results"] = []
                    recorder.event(event_type, event_payload, attempt_id=attempt)
                    tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result.output}
                    state.messages.append(tool_message)
                    state.tool_results.append({"call": call, "result": result.output, "success": result.success})
                    serialized.append(tool_message)
                    if not result.success and self.features.get("self_correction", {}).get("enabled"):
                        feedback = {
                            "role": "user",
                            "content": "The last tool action failed. Inspect the error, revise the approach, and retry only if useful.",
                        }
                        state.messages.append(feedback)
                        recorder.event("feedback_applied", {"source": "tool_failure", "call_id": call["id"],
                                                             "error": result.error}, attempt_id=attempt)
                state.pending_tool_calls = []
                recorder.checkpoint(state, kind="tool_batch", attempt_id=attempt)
                set_transition(TransitionKind.CALL_MODEL, "tool batch completed")
                continue

            if transition.kind != TransitionKind.CALL_MODEL:
                state.error = {"type": "invalid_transition", "message": f"unsupported transition: {transition.kind}"}
                set_transition(TransitionKind.FAIL, "unsupported transition")
                continue

            max_total_tokens = int(self.config.get("limits", {}).get("max_total_tokens", 65536))
            if int(state.usage.get("total_tokens") or state.usage.get("normalized_completion_tokens") or 0) >= max_total_tokens:
                state.error = {"type": "token_budget", "message": "total token budget exhausted"}
                set_transition(TransitionKind.FAIL, "token budget exhausted")
                continue
            if state.model_turn_count >= max_turns:
                state.error = {"type": "max_turns", "message": "model turn budget exhausted"}
                set_transition(TransitionKind.FAIL, "model turn budget exhausted")
                continue
            if compaction_checked_turn != state.model_turn_count and context_manager.needs_compaction(state.messages):
                set_transition(TransitionKind.COMPACT, "context budget threshold exceeded")
                continue
            finalization_enabled = bool(self.features.get("finalization", {}).get("enabled", False))
            finalization_active = finalization_enabled and state.model_turn_count == max_turns - 1
            if finalization_active and not finalization_started:
                finalization_started = True
                plan_summary = ""
                if state.current_plan:
                    plan_summary = " Current plan state: " + json.dumps(state.current_plan, ensure_ascii=False)
                state.messages.append({
                    "role": "user",
                    "content": (
                        "This is the final model turn. Do not call any tools. Use the information and tool "
                        "results already available to provide the complete final answer in the required format."
                        + plan_summary
                    ),
                })
                recorder.event("finalization_started", {"model_turn_count": state.model_turn_count}, attempt_id=attempt)
            before = state.digest()
            bound = self.model if finalization_active else (
                self.model.bind_tools([_openai_tool(tool) for tool in tools]) if tools else self.model
            )
            recorder.event("model_request", {"messages": recorder.bounded_payload(state.messages), "tools": tools},
                           attempt_id=attempt, state_before_hash=before)
            response, malformed_retries = await _invoke_with_xml_retry(
                bound, state.messages, _malformed_retry_limit(self.config)
            )
            state.malformed_retry_count += malformed_retries
            value = _serialize_message(response)
            serialized.append(value)
            state.messages.append(response)
            state.model_turn_count += 1
            _accumulate_usage(state.usage, value)
            calls = _tool_calls(response, value)
            if finalization_active and calls:
                state.error = {"type": "finalization_failed", "message": "model requested tools during the reserved final turn"}
                recorder.event("finalization_failed", {"tool_calls": calls}, attempt_id=attempt)
                set_transition(TransitionKind.FAIL, "tool requested during finalization")
                continue
            state.pending_tool_calls = calls
            text = _content_text(value.get("content"))
            if text:
                state.final_answer = text
            after = state.digest()
            recorder.event("model_response", {"response": value, "tool_calls": calls,
                                                "stop_reason": _done_reason(value), "usage": state.usage},
                           attempt_id=attempt, state_before_hash=before, state_after_hash=after)
            recorder.checkpoint(state, kind="model_response", attempt_id=attempt)
            if _done_reason(value) == "length":
                set_transition(TransitionKind.RECOVER, "model output reached max_tokens")
                continue
            if calls and not execute_tools:
                all_calls.extend(calls)
                current_turn_calls.append([_call_string(call) for call in calls])
                calls_by_turn.append(current_turn_calls)
                set_transition(TransitionKind.FINISH, "tool calls returned without execution")
                continue
            if calls:
                set_transition(TransitionKind.EXECUTE_TOOLS, "model requested tools",
                               {"tool_call_count": len(calls)})
                continue
            if state.current_plan and not _plan_complete(state.current_plan) and not finalization_active:
                state.messages.append({
                    "role": "user",
                    "content": (
                        "The plan is not complete. Continue with a runnable task using task_update, or "
                        "revise the plan if it can no longer be completed."
                    ),
                })
                recorder.event("premature_final_answer", {
                    "plan_version": state.current_plan.get("version"),
                    "runnable_task_ids": _runnable_task_ids(state.current_plan),
                }, attempt_id=attempt)
                set_transition(TransitionKind.CALL_MODEL, "plan incomplete")
                continue
            verification = self.features.get("verification", {})
            if verification.get("enabled") and not verification_active and state.final_answer:
                set_transition(TransitionKind.VERIFY, "final answer requires verification", {"action": "start"})
            elif verification_active:
                set_transition(TransitionKind.VERIFY, "verification response received", {"action": "finish"})
            else:
                advance_turn_or_finish()

        if state.current_plan:
            recorder.event("plan_finished", {
                "plan_version": state.current_plan.get("version"),
                "completed": _plan_complete(state.current_plan),
                "tasks": state.current_plan.get("tasks", []),
            }, attempt_id=attempt)
        recorder.checkpoint(state, kind="finish", attempt_id=attempt)
        status = "success" if state.phase == AgentPhase.FINISHED else (
            "limit_exceeded" if state.error and state.error["type"] in {"max_turns", "max_tool_calls", "output_token_limit"}
            else "runtime_error"
        )
        return AgentOutput(
            id=item["id"], status=status, final_answer=state.final_answer,
            messages=serialized, tool_calls=all_calls, usage=state.usage,
            timing={"duration_seconds": time.perf_counter() - started}, error=state.error,
            metadata={"runtime": self.runtime_kind, "profile": self.profile_name,
                      "trajectory_id": trajectory_id, "bfcl_calls_by_turn": calls_by_turn,
                      "model_turn_count": state.model_turn_count,
                      "tool_call_count": state.tool_call_count, "features": self.features,
                      "verification_result": state.verification_result,
                      "compaction_count": compaction_count,
                      "transition_count": transition_count,
                      "route": route,
                      "router_method": router.get("method", "model_judgment") if router.get("enabled") else None,
                      "router_raw_output": router_raw,
                      "finalization_started": finalization_started,
                      "plan": state.current_plan,
                      "plan_completed": _plan_complete(state.current_plan) if state.current_plan else None},
        )

    async def _run_locomo_context(self, item: dict[str, Any], started: float, attempt: int,
                                  *, include_conversation: bool) -> AgentOutput:
        trajectory_id = str(uuid.uuid4())
        recorder = RunRecorder(
            self.run_state_root, experiment_id=self.run_state_root.name,
            runtime_kind=self.runtime_kind, model_profile=self.profile_name,
            sample_id=item["id"], trajectory_id=trajectory_id,
        )
        conversation = ([message for turn in item["turns"] for message in deepcopy(turn)]
                        if include_conversation else [])
        answers: list[dict[str, Any]] = []
        serialized: list[dict[str, Any]] = []
        usage = _empty_usage()
        system = SystemPromptBuilder(self.project_root).build(self.config["prompt"]["system"], self.config)
        context_manager = ContextManager.from_config(
            self.features.get("compaction", {}), self.profile,
            self.config.get("limits", {}).get("context_window_override"),
        )
        compaction_count = 0
        for question in item["environment"]["questions"]:
            messages: list[Any] = ([{"role": "system", "content": system}] if system else []) + conversation + [
                {"role": "user", "content": question["question"]}
            ]
            if context_manager.needs_compaction(messages):
                context_result = context_manager.compact(messages, artifact_writer=recorder.artifact)
                messages = context_result.messages
                if context_result.changed:
                    compaction_count += 1
                    recorder.event("context_compacted", {
                        "phase": "context_qa", "question_id": question["id"],
                        "original_characters": context_result.original_characters,
                        "compressed_characters": context_result.compacted_characters,
                        "original_tokens_estimate": context_result.original_tokens,
                        "compacted_tokens_estimate": context_result.compacted_tokens,
                        "compression_ratio": (
                            context_result.compacted_characters / context_result.original_characters
                            if context_result.original_characters else 1.0
                        ),
                        "actions": context_result.actions,
                    }, attempt_id=attempt)
            recorder.event("model_request", {"messages": recorder.bounded_payload(messages), "phase": "context_qa"}, attempt_id=attempt)
            response = await self.model.ainvoke(messages)
            value = _serialize_message(response)
            _accumulate_usage(usage, value)
            text = _content_text(value.get("content")) or ""
            serialized.append(value)
            answers.append({"id": question["id"], "category": question["category"],
                            "answer": text, "tool_calls": []})
            recorder.event("model_response", {"response": value, "phase": "context_qa", "usage": _response_usage(value)},
                           attempt_id=attempt)
        recorder.checkpoint({"answers": answers, "usage": usage}, kind="finish", attempt_id=attempt)
        return AgentOutput(
            id=item["id"], status="success", final_answer=answers[-1]["answer"] if answers else None,
            messages=serialized, usage=usage, timing={"duration_seconds": time.perf_counter() - started},
            metadata={"runtime": self.runtime_kind, "profile": self.profile_name,
                      "trajectory_id": trajectory_id, "locomo_answers": answers,
                      "locomo_mode": "full_context" if include_conversation else "no_memory",
                      "compaction_count": compaction_count,
                      "features": self.features},
        )
