from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_json(value: Any) -> str:
    return json.dumps(_json_compatible(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_compatible(value: Any) -> Any:
    """Normalize mapping keys before deterministic JSON sorting.

    BFCL evaluator details may contain both integer and string keys. Python's
    json encoder cannot compare those key types when sort_keys=True, although
    JSON object keys are strings by definition.
    """
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            text_key = str(key)
            if text_key in normalized:
                raise ValueError(f"duplicate JSON key after string conversion: {text_key!r}")
            normalized[text_key] = _json_compatible(child)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_compatible(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted((_json_compatible(child) for child in value), key=repr)
    return repr(value)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(stable_json(record) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(value) + "\n", encoding="utf-8")
