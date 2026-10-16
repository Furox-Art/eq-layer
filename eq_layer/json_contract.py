"""Unambiguous finite UTF-8 JSON for externally supplied transcripts and cases."""

from __future__ import annotations

import json
import math
from typing import Any


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def loads(text: str) -> Any:
    data = json.loads(
        text,
        object_pairs_hook=_unique_keys,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    try:
        json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("JSON text must contain valid UTF-8") from exc
    return data
