"""Parameter-profile parsing with inert ``#`` provenance fields."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping

from _linear_probe.common import ContractError, candidate_id, read_json


ALLOWED_PARAMETER_SOURCES = ("publication", "grid", "default")


def strip_note_fields(value: Any) -> Any:
    """Recursively discard JSON object keys beginning with ``#``.

    JSON has no comment syntax.  The project therefore uses explicit keys such
    as ``"# source (not executable)"``.  They remain visible to readers and in
    the profile SHA-256, but can never become executable hyperparameters.
    """

    if isinstance(value, Mapping):
        return {
            str(key): strip_note_fields(item)
            for key, item in value.items()
            if not str(key).lstrip().startswith("#")
        }
    if isinstance(value, list):
        return [strip_note_fields(item) for item in value]
    return value


def load_profile(
    path: str | Path,
    *,
    expected_model: str,
    expected_method: str,
    expected_source: str,
) -> tuple[dict, dict]:
    raw = read_json(path)
    profile = strip_note_fields(raw)
    if profile.get("model") != expected_model:
        raise ContractError(
            f"Profile model must be {expected_model!r}, got {profile.get('model')!r}"
        )
    if profile.get("method") != expected_method:
        raise ContractError(
            f"Profile method must be {expected_method!r}, got {profile.get('method')!r}"
        )
    if profile.get("parameter_source") != expected_source:
        raise ContractError(
            "Profile parameter_source does not match the requested source: "
            f"expected {expected_source!r}, got {profile.get('parameter_source')!r}"
        )
    if expected_source not in ALLOWED_PARAMETER_SOURCES:
        raise ContractError(f"Unsupported parameter source: {expected_source!r}")
    return profile, raw


def build_candidates(profile: Mapping[str, Any]) -> list[dict]:
    """Expand a fixed profile, explicit candidate list, or Cartesian grid."""

    fixed = dict(profile.get("fixed_parameters", {}))
    explicit = profile.get("candidates")
    search = profile.get("search_space")
    if isinstance(explicit, list):
        candidates = [{**fixed, **dict(item)} for item in explicit]
    elif isinstance(search, Mapping):
        dimensions = {
            str(key): list(value)
            for key, value in search.items()
            if key != "candidate_count" and not str(key).lstrip().startswith("#")
        }
        if any(not values for values in dimensions.values()):
            raise ContractError("Every grid-search dimension must be non-empty")
        keys = list(dimensions)
        candidates = [
            {**fixed, **dict(zip(keys, values, strict=True))}
            for values in itertools.product(*(dimensions[key] for key in keys))
        ]
    else:
        parameters = profile.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = fixed if fixed else None
        if not isinstance(parameters, Mapping) or not parameters:
            raise ContractError("Parameter profile contains no executable parameters")
        candidates = [dict(parameters)]

    declared = None
    if isinstance(search, Mapping):
        declared = search.get("candidate_count")
    if declared is None:
        declared = profile.get("candidate_count")
    if declared is not None and int(declared) != len(candidates):
        raise ContractError(
            f"Profile declares {declared} candidates but expands to {len(candidates)}"
        )

    result: list[dict] = []
    seen: set[str] = set()
    for raw_parameters in candidates:
        parameters = strip_note_fields(dict(raw_parameters))
        parameters.pop("candidate_id", None)
        identifier = candidate_id(parameters)
        if identifier in seen:
            raise ContractError(f"Duplicate canonical candidate {identifier}")
        seen.add(identifier)
        result.append({**parameters, "candidate_id": identifier})
    return result


__all__ = [
    "ALLOWED_PARAMETER_SOURCES",
    "build_candidates",
    "load_profile",
    "strip_note_fields",
]
