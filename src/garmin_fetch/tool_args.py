"""Lenient argument annotations for the agent's typed tools.

Models do not always send a nested argument the way its schema says: a list
or object arrives JSON-encoded as a string, or a one-item list as the bare
item. These ``BeforeValidator``s accept both forms before pydantic validates
the real type, so a cosmetic slip costs no retry.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BeforeValidator


def _decode_json_text(value: Any) -> Any:
    """A JSON-encoded list/object string -> the decoded value."""
    if isinstance(value, str) and value.strip()[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> Any:
    """Like :func:`_decode_json_text`, and a bare scalar becomes a one-item list."""
    value = _decode_json_text(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


#: For an object/list argument: also accepts it JSON-encoded.
Loose = BeforeValidator(_decode_json_text)

#: For a list-of-strings argument: also accepts JSON, a bare string or a comma list.
AsList = BeforeValidator(_as_list)


def compact_schema(schema: Any) -> Any:
    """A tool's JSON schema without pydantic's per-field optional noise.

    Every ``X | None = None`` field renders as ``anyOf: [X, {type: null}]``
    plus ``default: null`` — about half of a nested schema's tokens, re-sent
    on every request. Optional is already said by the field's absence from
    ``required``, so both are dropped (``X`` is kept). Validation is unchanged:
    pydantic still accepts an explicit null.
    """
    if isinstance(schema, list):
        return [compact_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: compact_schema(v) for k, v in schema.items()}
    variants = out.get("anyOf")
    if isinstance(variants, list) and {"type": "null"} in variants:
        rest = [v for v in variants if v != {"type": "null"}]
        if len(rest) == 1 and isinstance(rest[0], dict):
            del out["anyOf"]
            out = {**rest[0], **out}
        else:
            out["anyOf"] = rest
    if "default" in out and out["default"] is None:
        del out["default"]
    return out
