"""Build query-specific LoCoMo context from the existing Agent evaluation record."""
from __future__ import annotations

import re
import json
import urllib.request
from typing import Any

from baselines.locomo_rag import bm25_rag, full_context, recent_window, vector_rag


class OllamaEncoder:
    """SentenceTransformer-compatible adapter for Ollama /api/embed."""

    def __init__(self, host: str, model: str, batch_size: int = 32):
        self.host, self.model, self.batch_size = host.rstrip("/"), model, batch_size
        self.cache: dict[str, Any] = {}

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        import numpy as np
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch_texts = missing[start:start + self.batch_size]
            payload = json.dumps({"model": self.model,
                                  "input": batch_texts}).encode()
            req = urllib.request.Request(self.host + "/api/embed", data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as response:
                batch = json.load(response)["embeddings"]
            if len(batch) != len(batch_texts):
                raise RuntimeError("Ollama returned the wrong number of embeddings")
            self.cache.update(zip(batch_texts, batch))
        array = np.asarray([self.cache[text] for text in texts], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            array = array / np.maximum(norms, 1e-12)
        return array


def _sample(item: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    conv: dict[str, Any] = {}
    for index, batch in enumerate(item["turns"], 1):
        key = f"session_{index}"
        lines = "\n".join(message["content"] for message in batch).splitlines()
        conv[key + "_date_time"] = (next((line[5:].strip() for line in lines if line.startswith("DATE:")), "")
                                            if feature.get("include_dates", True) else "")
        rows = []
        for line in lines:
            match = re.match(r"^\[([^]]+)\]\s*([^:]+):\s*(.*)$", line)
            if match:
                content = match[3]
                if not feature.get("include_image_captions", True):
                    content = re.sub(r"\s*\[shared image:.*?\]", "", content)
                rows.append({"dia_id": match[1],
                             "speaker": match[2] if feature.get("include_speakers", True) else "participant",
                             "text": content})
            elif rows and line.strip():
                rows[-1]["text"] += "\n" + line
        conv[key] = rows
    return {"sample_id": item["metadata"]["conversation_id"], "conversation": conv}


async def build_context(item: dict[str, Any], question: str, feature: dict[str, Any], model: Any,
                        encoder: Any | None = None, summary_cache: dict[str, str] | None = None) -> tuple[str, int]:
    sample = _sample(item, feature)
    mode = feature["mode"]
    kwargs = {"top_k": feature.get("top_k", 5), "chunk_turns": feature.get("chunk_turns", 3)}
    if mode == "full_context":
        return full_context(sample), 0
    if mode == "recent_window":
        return recent_window(sample, max_chars=feature.get("max_chars", 16000)), 0
    if mode == "bm25_rag":
        return bm25_rag(sample, question, **kwargs), 0
    if mode == "vector_rag":
        if encoder is None:
            raise RuntimeError("vector_rag requires sentence-transformers")
        return vector_rag(sample, question, encoder, **kwargs), 0
    if mode == "session_summary":
        summary_docs = []
        calls = 0
        for key in sorted((k for k in sample["conversation"] if re.fullmatch(r"session_\d+", k)),
                          key=lambda x: int(x.split("_")[1])):
            rows = sample["conversation"][key]
            source = "\n".join(f"[{row['dia_id']}] {row['speaker']}: {row['text']}" for row in rows)
            prompt = ("Summarize this conversation session as concise factual notes. Preserve names, "
                      "dates, changes and dialogue IDs. Do not invent facts.\n\n"
                      + sample["conversation"].get(key + "_date_time", "") + "\n" + source)
            if summary_cache is not None and key in summary_cache:
                summary = summary_cache[key]
            else:
                response = await model.ainvoke([{"role": "user", "content": prompt}])
                summary = str(response.content)
                if summary_cache is not None:
                    summary_cache[key] = summary
                calls += 1
            summary_docs.append({"dia_id": key, "speaker": "summary", "text": summary})
        fake = {"conversation": {"session_1": summary_docs}}
        return bm25_rag(fake, question, top_k=kwargs["top_k"], chunk_turns=1), calls
    raise ValueError(f"unsupported LoCoMo context mode: {mode}")
