"""REMem-inspired episodic memory for the existing LoCoMo Agent harness.

This is an Ollama-friendly adaptation, not a reproduction of REMem's graph/PPR
or DSPy components. Only conversation turns are used during construction.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from Agent.runtime.locomo_context_methods import _sample


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group())
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


class EpisodicMemoryV1:
    def __init__(self, item: dict[str, Any], feature: dict[str, Any], model: Any, encoder: Any | None = None):
        self.item, self.feature, self.model, self.encoder = item, feature, model, encoder
        self.sample = _sample(item, feature)
        self.nodes: list[dict[str, Any]] = []
        self.edges: dict[int, set[int]] = defaultdict(set)
        self.raw: dict[str, str] = {}
        self.write_calls = 0
        self.invalid_extractions = 0
        self.write_prompt_tokens = 0
        self.write_completion_tokens = 0
        self.vectors = None

    async def build(self) -> None:
        conv = self.sample["conversation"]
        for key in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                          key=lambda k: int(k.split("_")[1])):
            rows = conv[key]
            date = conv.get(key + "_date_time", "")
            for row in rows:
                self.raw[row["dia_id"]] = f"[{row['dia_id']}] {row['speaker']}: {row['text']}"
            if not rows:
                continue
            allowed_ids = [row["dia_id"] for row in rows]
            source = "\n".join(self.raw[d] for d in allowed_ids)
            prompt = (
                "Extract episodic memory from this conversation session. Return only JSON with "
                "keys 'gists' and 'facts'. 'gists' is a list of brief objects with 'text' and "
                "'source_ids'. 'facts' is a list of objects with 'subject', 'predicate', "
                "'object', 'time', and 'source_ids'. Use only dialogue IDs from this session. "
                "Do not invent facts. Keep dates and changes. Example: "
                '{"gists":[{"text":"A started pottery","source_ids":["D1:2"]}],'
                '"facts":[{"subject":"A","predicate":"started","object":"pottery",'
                '"time":"2023-05","source_ids":["D1:2"]}]}\n\n'
                f"Session date: {date}\n{source}"
            )
            response = await self.model.ainvoke([{"role": "user", "content": prompt}])
            self.write_calls += 1
            usage = getattr(response, "usage_metadata", None) or {}
            self.write_prompt_tokens += int(usage.get("input_tokens", 0))
            self.write_completion_tokens += int(usage.get("output_tokens", 0))
            extracted = _parse_json(str(response.content))
            if not extracted:
                self.invalid_extractions += 1
            start_count = len(self.nodes)
            if self.feature.get("use_gists", True):
                for gist in extracted.get("gists", []):
                    if isinstance(gist, dict) and gist.get("text"):
                        self._add(str(gist["text"]), gist.get("source_ids"), allowed_ids,
                                  date, "gist", [])
            for fact in extracted.get("facts", []):
                if not isinstance(fact, dict):
                    continue
                parts = [str(fact.get(k, "")).strip() for k in ("subject", "predicate", "object")]
                if not any(parts):
                    continue
                entities = [parts[0], parts[2]]
                time = str(fact.get("time") or date) if self.feature.get("use_time", True) else ""
                self._add(" ".join(x for x in parts if x), fact.get("source_ids"), allowed_ids,
                          time, "fact", entities)
            # If extraction failed, retain the session as a searchable raw episode.
            if len(self.nodes) == start_count:
                if self.feature.get("use_raw_backtrace", True):
                    self._add(source, allowed_ids, allowed_ids, date, "raw_fallback", [])
            for node in self.nodes[start_count:]:
                node["session"] = key
        if self.feature.get("use_graph", True):
            by_entity: dict[str, list[int]] = defaultdict(list)
            for i, node in enumerate(self.nodes):
                for entity in node["entities"]:
                    label = entity.lower().strip()
                    if len(label) >= 3:
                        by_entity[label].append(i)
            for ids in by_entity.values():
                for i in ids:
                    self.edges[i].update(j for j in ids if j != i)
        if self.encoder and self.feature.get("use_vector", True):
            self.vectors = self.encoder.encode([n["search_text"] for n in self.nodes],
                                               normalize_embeddings=True)

    def _add(self, text: str, source_ids: Any, allowed: list[str], time: str,
             kind: str, entities: list[str]) -> None:
        ids = [s for s in (source_ids if isinstance(source_ids, list) else []) if s in allowed]
        if not ids:
            return
        if not self.feature.get("use_time", True):
            time = ""
        search_text = f"{text} {time}" if self.feature.get("use_time", True) else text
        self.nodes.append({"text": text, "search_text": search_text, "time": time,
                           "kind": kind, "source_ids": ids, "entities": entities,
                           "session": ""})

    def retrieve(self, question: str) -> tuple[str, dict[str, int]]:
        if not self.nodes:
            return "", {"retrieved_nodes": 0, "graph_expanded": 0}
        terms = [Counter(_words(n["search_text"])) for n in self.nodes]
        df = Counter(t for row in terms for t in row)
        avg = sum(sum(row.values()) for row in terms) / len(terms)
        query = set(_words(question))
        lexical = [sum(math.log(1 + (len(terms) - df[t] + .5) / (df[t] + .5))
                       * row[t] * 2.2 / (row[t] + 1.2 * (.25 + .75 * sum(row.values()) / max(avg, 1)))
                       for t in query if row[t]) for row in terms]
        scores = lexical[:]
        if self.vectors is not None:
            import numpy as np
            q = self.encoder.encode([question], normalize_embeddings=True)[0]
            semantic = np.asarray(self.vectors) @ np.asarray(q)
            lo, hi = min(lexical), max(lexical)
            scores = [(lexical[i] - lo) / max(hi - lo, 1e-9) + float(semantic[i])
                      for i in range(len(lexical))]
        top_k = int(self.feature.get("top_k", 5))
        selected = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        expanded = 0
        if self.feature.get("use_graph", True) and self.feature.get("graph_steps", 1) > 0:
            candidates = set(j for i in selected for j in self.edges.get(i, ())) - set(selected)
            limit = int(self.feature.get("graph_extra", 5))
            extras = sorted(candidates, key=lambda i: (-scores[i], i))[:limit]
            selected.extend(extras)
            expanded = len(extras)
        pieces = []
        seen_raw: set[str] = set()
        budget = int(self.feature.get("max_chars", 12000))
        for i in selected:
            node = self.nodes[i]
            line = f"[{node['kind']} | {node['time']} | sources: {', '.join(node['source_ids'])}] {node['text']}"
            if len("\n\n".join(pieces + [line])) > budget:
                continue
            pieces.append(line)
            if self.feature.get("use_raw_backtrace", True):
                for source_id in node["source_ids"]:
                    if source_id in seen_raw:
                        continue
                    raw = self.raw.get(source_id, "")
                    if raw and len("\n\n".join(pieces + [raw])) <= budget:
                        pieces.append(raw)
                        seen_raw.add(source_id)
        return "\n\n".join(pieces), {"retrieved_nodes": len(selected),
                                      "graph_expanded": expanded,
                                      "raw_backtraced": len(seen_raw)}
