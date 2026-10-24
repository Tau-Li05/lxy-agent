from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ToolExecutionResult:
    execution_key: str
    output: str
    success: bool
    replayed: bool
    error: dict[str, Any] | None = None


class BFCLProtocolExecutor:
    """Shared BFCL backend with an execution ledger for recovery-safe replay."""

    def __init__(self, project_root: Path, backend: Callable[..., tuple[list[str], Any]] | None = None) -> None:
        self.project_root = project_root
        self._ledger: dict[str, ToolExecutionResult] = {}
        self._backend = backend
        self.result_root = project_root / "runs" / "bfcl_memory_state"

    def execute_batch(
        self, calls: list[str], *, item: dict[str, Any], trajectory_id: str,
        tool_call_ids: list[str], attempt: int,
    ) -> list[ToolExecutionResult]:
        keys = [f"{trajectory_id}:{call_id}" for call_id in tool_call_ids]
        if all(key in self._ledger for key in keys):
            return [ToolExecutionResult(key, self._ledger[key].output, self._ledger[key].success, True,
                                        self._ledger[key].error) for key in keys]

        if self._backend is None:
            package = self.project_root / "benchmark/BFCL-main/berkeley-function-call-leaderboard"
            if str(package) not in sys.path:
                sys.path.insert(0, str(package))
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
            backend = execute_multi_turn_func_call
        else:
            backend = self._backend

        environment = item["environment"]
        initial_config = json.loads(json.dumps(environment["initial_config"]))
        for class_name, settings in initial_config.items():
            if class_name.startswith("MemoryAPI_"):
                settings["model_result_dir"] = self.result_root
        outputs, involved_instances = backend(
            calls, initial_config, environment["involved_classes"],
            f"protocol_attempt_{attempt}", item["id"],
            long_context="long_context" in item["metadata"].get("category", ""), is_evaL_run=False,
        )
        # The official BFCL handler flushes memory only after a complete
        # conversation. The Agent protocol calls the executor one model turn at
        # a time, so persist prerequisite state after every successful batch.
        # This makes snapshots durable across retries and process restarts.
        if "_prereq" in item["id"]:
            for instance in involved_instances.values():
                flush = getattr(instance, "_flush_memory_to_local_file", None)
                if flush is not None:
                    flush()
        results: list[ToolExecutionResult] = []
        for key, output in zip(keys, outputs):
            parsed_error = None
            try:
                parsed = json.loads(output) if isinstance(output, str) else output
                if isinstance(parsed, dict) and "error" in parsed:
                    parsed_error = {"message": parsed["error"]}
            except (ValueError, TypeError):
                pass
            result = ToolExecutionResult(key, str(output), parsed_error is None, False, parsed_error)
            self._ledger[key] = result
            results.append(result)
        return results

    def flush_memory_prerequisite(self, *, item: dict[str, Any], attempt: int) -> None:
        """Persist BFCL memory state even when the model made no tool call.

        The official evaluator flushes at end-of-conversation.  Protocol runs
        may finish a prerequisite with a plain ``stop`` response, so there is
        no execute_batch call at which to flush the backend instance.
        """
        if "_prereq" not in item.get("id", ""):
            return
        environment = item["environment"]
        initial_config = json.loads(json.dumps(environment["initial_config"]))
        for class_name, settings in initial_config.items():
            if class_name.startswith("MemoryAPI_"):
                settings["model_result_dir"] = self.result_root
        package = self.project_root / "benchmark/BFCL-main/berkeley-function-call-leaderboard"
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
        _, instances = execute_multi_turn_func_call(
            [], initial_config, environment["involved_classes"],
            f"protocol_attempt_{attempt}", item["id"],
            long_context="long_context" in item["metadata"].get("category", ""), is_evaL_run=False,
        )
        for instance in instances.values():
            flush = getattr(instance, "_flush_memory_to_local_file", None)
            if flush is not None:
                flush()
