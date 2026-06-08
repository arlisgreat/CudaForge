#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover
    raise SystemExit("jsonschema is required for StreamMA frame validation") from exc


ROOT = Path(__file__).resolve().parent
SCHEMA_DIR = ROOT / "schemas"

SCHEMAS = {
    "semantic": SCHEMA_DIR / "semantic_frame.schema.json",
    "verdict": SCHEMA_DIR / "frame_verdict.schema.json",
}

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


def _iter_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _iter_strings(item, f"{path}[{idx}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, f"{path}.{key}")


def _check_no_code_leak(frame: dict[str, Any]) -> list[str]:
    if frame.get("frame_type") == "FinalCode":
        return []

    errors: list[str] = []
    for path, text in _iter_strings(frame):
        for pattern in CODE_LEAK_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path}: matched {pattern.pattern!r}")
    return errors


def validate(kind: str, payload_path: Path) -> None:
    schema = _load_json(SCHEMAS[kind])
    payload = _load_json(payload_path)

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    if errors:
        for err in errors:
            loc = "$" + "".join(f".{part}" for part in err.path)
            print(f"schema error at {loc}: {err.message}", file=sys.stderr)
        raise SystemExit(1)

    if kind == "semantic":
        leak_errors = _check_no_code_leak(payload)
        if leak_errors:
            for err in leak_errors:
                print(f"code leak error at {err}", file=sys.stderr)
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser("Validate StreamMA JSON frames")
    parser.add_argument("kind", choices=sorted(SCHEMAS))
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()

    validate(args.kind, args.payload)
    print(f"{args.kind} frame valid: {args.payload}")


if __name__ == "__main__":
    main()
