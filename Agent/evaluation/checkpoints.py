from __future__ import annotations

import json
from pathlib import Path

from Agent.data_pipeline.io import stable_json

from .models import AgentOutput


def append_checkpoint(path: Path, output: AgentOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(stable_json(output.to_dict()) + "\n")
        stream.flush()


def load_checkpoints(path: Path) -> dict[str, AgentOutput]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    outputs: dict[str, AgentOutput] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"{path}:{index + 1}: corrupt checkpoint record")
        output = AgentOutput.from_dict(value)
        if output.id in outputs:
            raise ValueError(f"{path}:{index + 1}: duplicate checkpoint ID {output.id}")
        outputs[output.id] = output
    return outputs
