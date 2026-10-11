from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read by physical LF-delimited lines; str.splitlines() breaks on U+2028."""
    with path.open(encoding="utf-8", newline="") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{index + 1}: JSONL record must be an object")
            yield index, value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(stable_json(record))
            stream.write("\n")
            count += 1
    return count, file_hash(path)


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(stable_json(value) + "\n")
    return file_hash(path)

