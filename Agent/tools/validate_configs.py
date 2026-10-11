#!/usr/bin/env python3
"""Static validation for Agent benchmark YAML files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import jsonschema
except ImportError as exc:
    raise SystemExit("缺少 jsonschema：请先安装 jsonschema") from exc


AGENT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCHEMA = AGENT_ROOT / "schemas/benchmark-v1.schema.json"
SUITE_SCHEMA = AGENT_ROOT / "schemas/suite-v1.schema.json"
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("YAML 顶层必须是对象")
    return value


def expand_known_env(value: str) -> str:
    """Expand set variables; preserve unset placeholders for static validation."""
    return ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def resolve_path(config_path: Path, value: str) -> Path | None:
    expanded = expand_known_env(value)
    if ENV_PATTERN.search(expanded):
        return None
    candidate = Path(expanded)
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path, loader_type: str) -> int:
    if loader_type == "jsonl":
        with path.open(encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)
    if loader_type == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("校验 Parquet 样本数需要 pyarrow") from exc
        return pq.ParquetFile(path).metadata.num_rows
    raise ValueError(f"不支持计算 {loader_type} 的样本数")


def validate_benchmark(path: Path, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        config = load_yaml(path)
        jsonschema.Draft202012Validator(schema).validate(config)
    except Exception as exc:
        return [f"{path}: {exc}"]

    loader = config["dataset"]["loader"]
    loader_type = loader["type"]
    paths = loader.get("paths", {})
    expected = config["dataset"]["integrity"]["expected_samples"]
    hashes = config["dataset"]["integrity"]["sha256"]

    for split, raw_path in paths.items():
        resolved = resolve_path(path, raw_path)
        if resolved is None:
            continue
        if not resolved.is_file():
            errors.append(f"{path}: dataset.loader.paths.{split} 不存在: {resolved}")
            continue
        if split in expected:
            actual_count = row_count(resolved, loader_type)
            if actual_count != expected[split]:
                errors.append(f"{path}: {split} 样本数 {actual_count} != {expected[split]}")
        if split in hashes:
            actual_hash = sha256(resolved)
            if actual_hash != hashes[split]:
                errors.append(f"{path}: {split} SHA-256 不匹配")

    for key in ("root", "attachment_root"):
        if key in loader:
            resolved = resolve_path(path, loader[key])
            if resolved is not None and not resolved.exists():
                errors.append(f"{path}: dataset.loader.{key} 不存在: {resolved}")
    return errors


def validate_suite(path: Path, schema: dict[str, Any], benchmark_schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        config = load_yaml(path)
        jsonschema.Draft202012Validator(schema).validate(config)
    except Exception as exc:
        return [f"{path}: {exc}"]
    for item in config["benchmarks"]:
        benchmark_path = resolve_path(path, item["config"])
        if benchmark_path is None or not benchmark_path.is_file():
            errors.append(f"{path}: Benchmark 配置不存在: {item['config']}")
            continue
        errors.extend(validate_benchmark(benchmark_path, benchmark_schema))
        if not errors:
            benchmark = load_yaml(benchmark_path)
            available = set(benchmark["dataset"]["integrity"]["expected_samples"])
            missing = set(item.get("splits", [])) - available
            if missing:
                errors.append(f"{path}: {benchmark_path.name} 缺少 split: {sorted(missing)}")
    return errors


def main() -> int:
    benchmark_schema = json.loads(BENCHMARK_SCHEMA.read_text(encoding="utf-8"))
    suite_schema = json.loads(SUITE_SCHEMA.read_text(encoding="utf-8"))
    errors: list[str] = []
    benchmarks = sorted((AGENT_ROOT / "configs/benchmarks").glob("*.yaml"))
    suites = sorted((AGENT_ROOT / "configs/suites").glob("*.yaml"))
    for path in benchmarks:
        errors.extend(validate_benchmark(path, benchmark_schema))
    for path in suites:
        errors.extend(validate_suite(path, suite_schema, benchmark_schema))
    if errors:
        print("配置校验失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"配置校验通过：{len(benchmarks)} 个 Benchmark，{len(suites)} 个 Suite。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

