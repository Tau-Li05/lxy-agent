from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Return the last balanced \boxed{...}, or the last unbraced token."""
    marker = r"\boxed"
    positions = [m.start() for m in re.finditer(re.escape(marker), text)]
    for start in reversed(positions):
        pos = start + len(marker)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            continue
        if text[pos] != "{":
            match = re.match(r"([^\s.$,;]+)", text[pos:])
            if match:
                return match.group(1).strip()
            continue
        depth = 0
        for end in range(pos, len(text)):
            char = text[end]
            if char == "{" and (end == 0 or text[end - 1] != "\\"):
                depth += 1
            elif char == "}" and (end == 0 or text[end - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    value = text[pos + 1 : end].strip()
                    return value or None
    return None


def extract_gsm8k(text: str) -> str | None:
    matches = re.findall(r"####\s*([^\r\n]+)", text)
    return matches[-1].strip() if matches else None


def integer_string(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return str(int(value.strip()))
    return None


def comma_integer_list(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not re.fullmatch(r"\d+", part) for part in parts):
        return None
    return ",".join(str(int(part)) for part in parts)


def livebench_answer(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized_list = comma_integer_list(value)
    return normalized_list if normalized_list is not None else value.strip()


EXTRACTORS = {
    "boxed_latex": extract_boxed,
    "gsm8k_hash": extract_gsm8k,
    "integer": integer_string,
    "comma_integer_list": comma_integer_list,
    "livebench_answer": livebench_answer,
}

