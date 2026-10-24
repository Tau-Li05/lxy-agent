from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from Agent.evaluation.io import stable_json


SCHEMA_VERSION = "1.0"
NORMALIZED_TOKENIZER = "unicode-wordpunct-v1"
_SECRET_KEYS = re.compile(r"(?:api[_-]?key|authorization|token|secret|password|cookie)", re.I)
_FILE_LOCK = threading.Lock()


def normalized_token_count(value: str) -> int:
    """Provider-independent, deterministic comparison unit (not a model token count)."""
    return len(re.findall(r"[\w]+|[^\w\s]", value, flags=re.UNICODE))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEYS.search(str(key)) else redact(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact(child) for child in value]
    return value


def content_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(_jsonable(value)).encode("utf-8")).hexdigest()


def state_delta(before: Any, after: Any) -> dict[str, Any]:
    """Return a replayable top-level state delta with explicit deletions."""
    old = _jsonable(before)
    new = _jsonable(after)
    if not isinstance(old, dict) or not isinstance(new, dict):
        return {"$replace": new}
    return {
        "set": {key: value for key, value in new.items() if key not in old or old[key] != value},
        "delete": sorted(key for key in old if key not in new),
    }


def apply_state_delta(state: Any, delta: dict[str, Any]) -> Any:
    if "$replace" in delta:
        return delta["$replace"]
    result = dict(_jsonable(state))
    for key in delta.get("delete", []):
        result.pop(key, None)
    result.update(delta.get("set", {}))
    return result


class RunRecorder:
    """Append-only telemetry, durable state checkpoints, and content-addressed artifacts."""

    def __init__(
        self, run_dir: Path, *, experiment_id: str, runtime_kind: str,
        model_profile: str, sample_id: str, trajectory_id: str,
    ) -> None:
        self.run_dir = run_dir
        self.experiment_id = experiment_id
        self.runtime_kind = runtime_kind
        self.model_profile = model_profile
        self.sample_id = sample_id
        self.trajectory_id = trajectory_id
        self.events_path = run_dir / "events.jsonl"
        self.checkpoints_path = run_dir / "checkpoints.jsonl"
        self.artifact_dir = run_dir / "artifacts"
        self._lock = threading.Lock()
        self._sequence = 0
        self._last_event_id: str | None = None
        self._previous_checkpoint_hash: str | None = None
        run_dir.mkdir(parents=True, exist_ok=True)

    def event(
        self, event_type: str, payload: dict[str, Any], *, attempt_id: int = 1,
        state_before_hash: str | None = None, state_after_hash: str | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event_id = str(uuid.uuid4())
            record = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "runtime_kind": self.runtime_kind,
                "model_profile": self.model_profile,
                "sample_id": self.sample_id,
                "trajectory_id": self.trajectory_id,
                "attempt_id": attempt_id,
                "event_id": event_id,
                "parent_event_id": parent_event_id or self._last_event_id,
                "sequence_number": self._sequence,
                "event_type": event_type,
                "timestamp": time.time(),
                "monotonic_time": time.monotonic(),
                "payload": redact(_jsonable(payload)),
                "state_before_hash": state_before_hash,
                "state_after_hash": state_after_hash,
            }
            self._append(self.events_path, record)
            self._last_event_id = event_id
            return record

    def checkpoint(self, state: Any, *, kind: str, attempt_id: int = 1) -> dict[str, Any]:
        snapshot = redact(_jsonable(state))
        state_hash = content_hash(snapshot)
        record = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "runtime_kind": self.runtime_kind,
            "model_profile": self.model_profile,
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "attempt_id": attempt_id,
            "sequence_number": self._sequence,
            "checkpoint_kind": kind,
            "previous_hash": self._previous_checkpoint_hash,
            "state_hash": state_hash,
            "atomic_write": True,
            "state": snapshot,
        }
        if len(stable_json(snapshot).encode("utf-8")) > 100000:
            record["state_artifact"] = self.artifact(snapshot)
            record["state"] = None
        self._append(self.checkpoints_path, record)
        self._previous_checkpoint_hash = state_hash
        return record

    def artifact(self, value: Any, *, media_type: str = "application/json", suffix: str = ".json.gz") -> dict[str, Any]:
        raw = stable_json(redact(_jsonable(value))).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("artifacts") / f"{digest}{suffix}"
        path = self.run_dir / relative
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            with gzip.open(temporary, "wb") as stream:
                stream.write(raw)
            os.replace(temporary, path)
        return {"path": str(relative), "media_type": media_type, "byte_size": len(raw), "sha256": digest}

    def bounded_payload(self, value: Any, *, threshold: int = 20000) -> Any:
        normalized = redact(_jsonable(value))
        encoded = stable_json(normalized)
        if len(encoded.encode("utf-8")) <= threshold:
            return normalized
        return {"artifact": self.artifact(normalized), "preview": encoded[:2000], "truncated": True}

    @staticmethod
    def _append(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _FILE_LOCK:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(stable_json(value) + "\n")
                stream.flush()


def load_checkpoint_chain(path: Path, *, trajectory_id: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if trajectory_id is not None:
        records = [row for row in records if row.get("trajectory_id") == trajectory_id]
    previous: str | None = None
    for index, record in enumerate(records, 1):
        if record.get("previous_hash") != previous:
            raise ValueError(f"{path}:{index}: broken checkpoint previous_hash chain")
        snapshot = _checkpoint_state(path, record)
        actual = content_hash(snapshot)
        if actual != record.get("state_hash"):
            raise ValueError(f"{path}:{index}: checkpoint state hash mismatch")
        previous = actual
    return records


def latest_checkpoint(path: Path, *, trajectory_id: str) -> dict[str, Any] | None:
    records = load_checkpoint_chain(path, trajectory_id=trajectory_id)
    return _checkpoint_state(path, records[-1]) if records else None


def reconstruct_state(checkpoints_path: Path, events_path: Path, *, trajectory_id: str) -> Any:
    """Restore the latest durable checkpoint and replay later state deltas."""
    records = load_checkpoint_chain(checkpoints_path, trajectory_id=trajectory_id)
    if not records:
        return None
    checkpoint = records[-1]
    state = _checkpoint_state(checkpoints_path, checkpoint)
    sequence = int(checkpoint.get("sequence_number", 0))
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("trajectory_id") != trajectory_id or int(event.get("sequence_number", 0)) <= sequence:
                continue
            delta = (event.get("payload") or {}).get("state_delta")
            if delta is not None:
                if event.get("state_before_hash") and content_hash(state) != event["state_before_hash"]:
                    raise ValueError("state delta before-hash mismatch")
                state = apply_state_delta(state, delta)
                if event.get("state_after_hash") and content_hash(state) != event["state_after_hash"]:
                    raise ValueError("state delta after-hash mismatch")
    return state


def _checkpoint_state(checkpoint_path: Path, record: dict[str, Any]) -> Any:
    if record.get("state") is not None:
        return record["state"]
    artifact = record.get("state_artifact") or {}
    artifact_path = checkpoint_path.parent / artifact["path"]
    with gzip.open(artifact_path, "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))

