from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from . import PROCESSOR_VERSION
from .adapters import ADAPTERS, ConversionResult
from .io import file_hash, read_jsonl, stable_json, write_json, write_jsonl
from .validation import (
    artifact_hashes,
    compare_builds,
    ensure_policy_isolated,
    ensure_same_ids,
    ensure_unique_ids,
    load_validators,
    validate_jsonl,
    validate_records,
)


EXPECTED_CONFIGS = {"competition_math", "aime24", "aime25", "aime26", "gsm8k", "livebench_math"}


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError(f"Invalid conversion config: {path}")
    schema_path = path.parents[2] / "schemas" / "conversion-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"{path}: conversion config schema error: {errors[0].message}")
    return value


def discover_configs(config_dir: Path, selected: str | None) -> list[tuple[Path, dict[str, Any]]]:
    configs = [(path, _load_config(path)) for path in sorted(config_dir.glob("*.yaml"))]
    if selected:
        configs = [(path, cfg) for path, cfg in configs if cfg["benchmark"] == selected]
        if not configs:
            raise ValueError(f"Unknown dataset: {selected}")
    return configs


def _input_paths(config: dict[str, Any], config_path: Path) -> list[Path]:
    return [(config_path.parent / source["path"]).resolve() for source in config["sources"]]


def _verify_sources(config: dict[str, Any], config_path: Path) -> None:
    paths = _input_paths(config, config_path)
    counts = []
    hashes = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Source does not exist: {path}")
        counts.append(sum(1 for _ in read_jsonl(path)))
        hashes.append(file_hash(path))
    if sum(counts) != config["expected_source_samples"]:
        raise ValueError(f"{config['benchmark']}: source count {sum(counts)} != {config['expected_source_samples']}")
    expected_hash = config["source_sha256"]
    if isinstance(expected_hash, str):
        if len(hashes) != 1 or hashes[0] != expected_hash:
            raise ValueError(f"{config['benchmark']}: source SHA-256 mismatch")
    else:
        for source, actual in zip(config["sources"], hashes):
            key = source["split"]
            if expected_hash[key] != actual:
                raise ValueError(f"{config['benchmark']}:{key}: source SHA-256 mismatch")


def compute_build_id(agent_root: Path, configs: list[tuple[Path, dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    digest.update(PROCESSOR_VERSION.encode())
    for module in sorted((agent_root / "data_pipeline").glob("*.py")):
        digest.update(module.name.encode())
        digest.update(file_hash(module).encode())
    for path, config in configs:
        digest.update(path.name.encode())
        digest.update(stable_json(config).encode())
        for source_path in _input_paths(config, path):
            digest.update(file_hash(source_path).encode())
    for schema in sorted((agent_root / "schemas").glob("*-v1.schema.json")):
        digest.update(schema.name.encode())
        digest.update(file_hash(schema).encode())
    return digest.hexdigest()[:20]


def _sft(record: dict[str, Any]) -> dict[str, Any]:
    response = record["target"]["response"]
    if not isinstance(response, str):
        raise ValueError(f"{record['id']}: SFT record lacks response")
    return {
        "schema_version": "1.0", "id": record["id"],
        "messages": record["input"]["messages"] + [{"role": "assistant", "content": response}],
        "metadata": {"benchmark": record["benchmark"], "split": record["split"], **record["metadata"]},
    }


def _policy(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "id": record["id"], "input": record["input"],
        "metadata": {"benchmark": record["benchmark"], "split": record["split"], **record["metadata"]},
    }


def _reward(record: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "1.0", "id": record["id"], "reward": record["reward"], "metadata": {"benchmark": record["benchmark"], "split": record["split"]}}


def _eval_input(record: dict[str, Any], evaluation_scope: str) -> dict[str, Any]:
    return {"schema_version": "1.0", "id": record["id"], "evaluation_scope": evaluation_scope, "input": record["input"], "metadata": {"benchmark": record["benchmark"], "split": record["split"], **record["metadata"]}}


def _eval_reference(record: dict[str, Any], evaluation_scope: str) -> dict[str, Any]:
    return {"schema_version": "1.0", "id": record["id"], "evaluation_scope": evaluation_scope, "target": record["target"], "reward": record["reward"], "metadata": {"benchmark": record["benchmark"], "split": record["split"]}}


def build_tree(agent_root: Path, workspace_root: Path, configs: list[tuple[Path, dict[str, Any]]], output: Path, build_id: str, complete: bool) -> dict[str, Any]:
    validators = load_validators(agent_root)
    results: list[tuple[Path, dict[str, Any], ConversionResult]] = []
    for config_path, config in configs:
        _verify_sources(config, config_path)
        adapter = ADAPTERS.get(config["adapter"])
        if adapter is None:
            raise ValueError(f"Unknown adapter: {config['adapter']}")
        result = adapter(config, config_path, workspace_root)
        validate_records(result.records, validators["canonical"], config["benchmark"])
        ensure_unique_ids(result.records, config["benchmark"])
        results.append((config_path, config, result))

    all_records = [record for _, _, result in results for record in result.records]
    ensure_unique_ids(all_records, "all canonical records")
    prompt_keys: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        prompt = stable_json(record["input"]["messages"])
        if prompt in prompt_keys[record["benchmark"]] and record["benchmark"] == "competition_math":
            raise ValueError("competition_math prompt leakage across splits")
        prompt_keys[record["benchmark"]].add(prompt)

    sft: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    policies: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rewards: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    test_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dev_records: list[dict[str, Any]] = []
    dataset_manifests = []

    for config_path, config, result in results:
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in result.records:
            by_split[record["split"]].append(record)
            if record["split"] in sft and "sft" in record["usage"]:
                sft[record["split"]].append(_sft(record))
            if record["split"] in policies and "rl" in record["usage"]:
                policies[record["split"]].append(_policy(record)); rewards[record["split"]].append(_reward(record))
            if record["split"] == "validation":
                dev_records.append(record)
            elif record["split"] == "test":
                test_groups[(record["benchmark"], "test")].append(record)
        output_files = {}
        for split, records in sorted(by_split.items()):
            path = output / "canonical" / result.benchmark / f"{split}.jsonl"
            count, digest = write_jsonl(path, sorted(records, key=lambda item: item["id"]))
            output_files[path.relative_to(output).as_posix()] = {"count": count, "sha256": digest}
        rejected_path = output / "rejected" / f"{result.benchmark}.jsonl"
        count, digest = write_jsonl(rejected_path, result.rejected)
        output_files[rejected_path.relative_to(output).as_posix()] = {"count": count, "sha256": digest}
        manifest = {
            "schema_version": "1.0", "processor_version": PROCESSOR_VERSION, "build_id": build_id,
            "benchmark": result.benchmark, "config_sha256": file_hash(config_path),
            "sources": [{"path": p.relative_to(workspace_root).as_posix(), "sha256": file_hash(p)} for p in result.source_files],
            "split_policy": result.split_policy, "counts": dict(sorted(Counter(x["split"] for x in result.records).items())),
            "rejected": dict(sorted(Counter(x["reason"] for x in result.rejected).items())), "outputs": output_files,
        }
        manifest_path = output / "manifests" / f"{result.benchmark}.json"
        write_json(manifest_path, manifest)
        dataset_manifests.append(manifest)

    for split in ("train", "validation"):
        sft[split].sort(key=lambda item: item["id"]); policies[split].sort(key=lambda item: item["id"]); rewards[split].sort(key=lambda item: item["id"])
        validate_records(sft[split], validators["sft"], f"sft/{split}")
        validate_records(policies[split], validators["rl_policy"], f"rl/policy/{split}")
        validate_records(rewards[split], validators["rl_reward"], f"rl/rewards/{split}")
        ensure_policy_isolated(policies[split]); ensure_same_ids(policies[split], rewards[split], f"rl/{split}")
        write_jsonl(output / "sft" / f"{split}.jsonl", sft[split])
        write_jsonl(output / "rl" / "policy" / f"{split}.jsonl", policies[split])
        write_jsonl(output / "rl" / "rewards" / f"{split}.jsonl", rewards[split])

    dev_inputs = [_eval_input(record, "development") for record in sorted(dev_records, key=lambda item: item["id"])]
    dev_refs = [_eval_reference(record, "development") for record in sorted(dev_records, key=lambda item: item["id"])]
    validate_records(dev_inputs, validators["eval_input"], "dev inputs"); validate_records(dev_refs, validators["eval_reference"], "dev references")
    ensure_same_ids(dev_inputs, dev_refs, "dev")
    write_jsonl(output / "dev" / "competition_math" / "inputs.jsonl", dev_inputs)
    write_jsonl(output / "dev" / "competition_math" / "references.jsonl", dev_refs)

    for (benchmark, split), records in sorted(test_groups.items()):
        inputs = [_eval_input(record, "final_test") for record in sorted(records, key=lambda item: item["id"])]
        refs = [_eval_reference(record, "final_test") for record in sorted(records, key=lambda item: item["id"])]
        validate_records(inputs, validators["eval_input"], f"test/{benchmark} inputs"); validate_records(refs, validators["eval_reference"], f"test/{benchmark} references")
        ensure_same_ids(inputs, refs, f"test/{benchmark}")
        write_jsonl(output / "test" / benchmark / split / "inputs.jsonl", inputs)
        write_jsonl(output / "test" / benchmark / split / "references.jsonl", refs)

    counts = {
        "canonical": len(all_records), "sft_train": len(sft["train"]), "sft_validation": len(sft["validation"]),
        "rl_policy_train": len(policies["train"]), "rl_policy_validation": len(policies["validation"]),
        "dev": len(dev_inputs), "final_test": sum(len(records) for records in test_groups.values()),
    }
    if complete:
        expected = {"canonical": 21744, "sft_train": 18717, "sft_validation": 625, "rl_policy_train": 18717, "rl_policy_validation": 625, "dev": 625, "final_test": 2402}
        if counts != expected:
            raise ValueError(f"Complete build counts mismatch: {counts} != {expected}")
    manifest = {
        "schema_version": "1.0", "processor_version": PROCESSOR_VERSION, "build_id": build_id,
        "complete": complete, "validation_semantics": "development_only_not_final_test", "counts": counts,
        "benchmarks": [item["benchmark"] for item in dataset_manifests],
        "artifacts": artifact_hashes(output),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_tree(agent_root: Path, root: Path) -> None:
    validators = load_validators(agent_root)
    for path in sorted((root / "canonical").glob("*/*.jsonl")): validate_jsonl(path, validators["canonical"])
    for path in sorted((root / "sft").glob("*.jsonl")): validate_jsonl(path, validators["sft"])
    for path in sorted((root / "rl/policy").glob("*.jsonl")): ensure_policy_isolated(validate_jsonl(path, validators["rl_policy"]))
    for path in sorted((root / "rl/rewards").glob("*.jsonl")): validate_jsonl(path, validators["rl_reward"])
    for path in sorted(root.glob("dev/*/inputs.jsonl")) + sorted(root.glob("test/*/*/inputs.jsonl")): validate_jsonl(path, validators["eval_input"])
    for path in sorted(root.glob("dev/*/references.jsonl")) + sorted(root.glob("test/*/*/references.jsonl")): validate_jsonl(path, validators["eval_reference"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    current = artifact_hashes(root)
    recorded = manifest["artifacts"]
    if current != recorded:
        raise ValueError("Artifact hashes differ from manifest")


def publish(output_root: Path, staging: Path, build_id: str) -> Path:
    releases = output_root / "releases"; releases.mkdir(parents=True, exist_ok=True)
    release = releases / build_id
    if release.exists():
        shutil.rmtree(staging)
    else:
        staging.rename(release)
    temporary_link = output_root / f".current-{build_id}.tmp"
    if temporary_link.exists() or temporary_link.is_symlink(): temporary_link.unlink()
    temporary_link.symlink_to(Path("releases") / build_id)
    os.replace(temporary_link, output_root / "current")
    return release


def run_build(agent_root: Path, workspace_root: Path, output_root: Path, selected: str | None, verify_only: bool, do_publish: bool, force: bool) -> tuple[str, Path]:
    config_dir = agent_root / "configs" / "conversions"
    configs = discover_configs(config_dir, selected)
    complete = {cfg["benchmark"] for _, cfg in configs} == EXPECTED_CONFIGS
    if do_publish and not complete:
        raise ValueError("Publishing requires a complete --all build")
    for path, config in configs: _verify_sources(config, path)
    build_id = compute_build_id(agent_root, configs)
    staging_root = output_root / ".staging"; staging = staging_root / build_id
    mirror = staging_root / f"{build_id}.determinism"
    release = output_root / "releases" / build_id
    if release.exists() and not force:
        validate_tree(agent_root, release)
        if do_publish:
            return build_id, publish(output_root, staging, build_id) if staging.exists() else _switch_existing(output_root, release, build_id)
        return build_id, release
    for target in (staging, mirror):
        if target.exists(): shutil.rmtree(target)
    staging.mkdir(parents=True, exist_ok=False)
    mirror.mkdir(parents=True, exist_ok=False)
    try:
        build_tree(agent_root, workspace_root, configs, staging, build_id, complete)
        validate_tree(agent_root, staging)
        build_tree(agent_root, workspace_root, configs, mirror, build_id, complete)
        validate_tree(agent_root, mirror)
        compare_builds(staging, mirror)
        shutil.rmtree(mirror)
        if verify_only or not do_publish:
            return build_id, staging
        return build_id, publish(output_root, staging, build_id)
    except Exception:
        failed = staging_root / f"{build_id}.failed"
        if failed.exists(): shutil.rmtree(failed)
        if staging.exists(): staging.rename(failed)
        if mirror.exists(): shutil.rmtree(mirror)
        raise


def _switch_existing(output_root: Path, release: Path, build_id: str) -> Path:
    temporary_link = output_root / f".current-{build_id}.tmp"
    if temporary_link.exists() or temporary_link.is_symlink(): temporary_link.unlink()
    temporary_link.symlink_to(Path("releases") / build_id)
    os.replace(temporary_link, output_root / "current")
    return release

