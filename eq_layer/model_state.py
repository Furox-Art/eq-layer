"""Schema validation for persisted trained-model state (audit H-2).

A trained adapter is restored from a joblib artifact. Before this validation
the loaders indexed the payload directly, so a malformed artifact surfaced as a
raw ``KeyError`` for a missing key, or a ``TypeError``/``ValueError`` raised
from inside the dataclass constructor for the wrong field type. Either way the
caller got an unhandled path with no hint about which field was at fault.

Every loader in ``eq_layer/trained_*.py`` now validates the payload schema
before constructing the adapter, and raises one typed error naming the offending
field.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# A field that must be present and non-None but whose value is an opaque fitted
# estimator, so its concrete type is deliberately not pinned here.
FITTED = object()

_FLOAT_TYPES = (int, float)


class ModelStateError(ValueError):
    """A persisted model artifact failed schema validation at load time."""

    def __init__(
        self,
        model: str,
        field: str,
        problem: str,
        path: str | Path | None = None,
    ) -> None:
        location = f" in {path}" if path is not None else ""
        super().__init__(
            f"{model} artifact{location} has an invalid {field!r} field: {problem}"
        )
        self.model = model
        self.field = field
        self.problem = problem


def _type_names(types: tuple[type, ...]) -> str:
    return " or ".join(entry.__name__ for entry in types)


def _check(
    model: str,
    field: str,
    value: Any,
    accepted: Any,
    path: str | Path | None,
) -> None:
    if accepted is FITTED:
        if value is None:
            raise ModelStateError(model, field, "must not be None", path)
        return
    if not isinstance(value, accepted):
        raise ModelStateError(
            model,
            field,
            f"expected {_type_names(accepted)}, got {type(value).__name__}",
            path,
        )


def validate_state(
    payload: Any,
    *,
    model: str,
    path: str | Path | None = None,
    required: Mapping[str, Any] | None = None,
    optional: Mapping[str, Any] | None = None,
) -> None:
    """Validate a persisted model artifact before it is used.

    ``required``/``optional`` map a field name to an accepted type tuple, or to
    ``FITTED`` for an opaque fitted estimator. A missing required key, an empty
    payload and a wrongly typed field each raise ``ModelStateError`` naming the
    offending field.
    """
    if not isinstance(payload, Mapping):
        raise ModelStateError(
            model,
            "payload",
            f"expected a mapping of fields, got {type(payload).__name__}",
            path,
        )

    for spec, must_exist in ((required or {}, True), (optional or {}, False)):
        for field, accepted in spec.items():
            if field not in payload:
                if must_exist:
                    raise ModelStateError(model, field, "missing required key", path)
                continue
            _check(model, field, payload[field], accepted, path)
