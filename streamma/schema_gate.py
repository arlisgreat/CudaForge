from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
    JSONSCHEMA_IMPORT_ERROR = exc
else:
    JSONSCHEMA_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "experiments/streamma/schemas"
SEMANTIC_SCHEMA_PATH = SCHEMA_DIR / "semantic_frame.schema.json"
VERDICT_SCHEMA_PATH = SCHEMA_DIR / "frame_verdict.schema.json"

CODE_LEAK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"```",
        r"\b__global__\b",
        r"\b__device__\b",
        r"\b__host__\b",
        r"\bextern\s+\"C\"",
        r"\bload_inline\s*\(",
        r"torch\.utils\.cpp_extension",
        r"\bsource\s*=",
        r"\bcpp_src\s*=",
        r"#include\s*<",
        r"\btemplate\s*<",
        r"<<<\s*[^>]+>>>",
    )
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validator(path: Path):
    if jsonschema is None:  # pragma: no cover
        raise RuntimeError("jsonschema is required for stream_json_gate") from JSONSCHEMA_IMPORT_ERROR
    schema = _load_json(path)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def iter_strings(value: Any, path: str = "$"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from iter_strings(item, f"{path}[{idx}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(item, f"{path}.{key}")


def code_leak_errors(frame: dict[str, Any]) -> list[str]:
    if frame.get("frame_type") == "FinalCode":
        return []

    errors: list[str] = []
    for path, text in iter_strings(frame):
        for pattern in CODE_LEAK_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path}: matched {pattern.pattern!r}")
    return errors


def validate_semantic_frame(frame: dict[str, Any], *, gate_mode: str = "balanced") -> list[str]:
    errors: list[str] = []
    validator = _validator(SEMANTIC_SCHEMA_PATH)
    for err in sorted(validator.iter_errors(frame), key=lambda item: list(item.path)):
        loc = "$" + "".join(f".{part}" for part in err.path)
        errors.append(f"schema error at {loc}: {err.message}")

    if gate_mode in {"balanced", "strict"}:
        errors.extend(f"code leak error at {err}" for err in code_leak_errors(frame))

    if gate_mode == "strict" and frame.get("confidence", 1.0) < 0.4:
        errors.append("confidence below strict threshold 0.4")

    return errors


def validate_verdict(verdict: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    validator = _validator(VERDICT_SCHEMA_PATH)
    for err in sorted(validator.iter_errors(verdict), key=lambda item: list(item.path)):
        loc = "$" + "".join(f".{part}" for part in err.path)
        errors.append(f"schema error at {loc}: {err.message}")
    return errors

