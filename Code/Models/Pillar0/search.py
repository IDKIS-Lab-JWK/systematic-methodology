"""Curated, budget-aware parameter search for Pillar-0 adaptations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def canonical_parameters(parameters: dict) -> dict:
    return json.loads(json.dumps(parameters, sort_keys=True))


def candidate_id(parameters: dict) -> str:
    payload = json.dumps(
        canonical_parameters(parameters), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def load_search_definition(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        definition = json.load(handle)
    if definition.get("strategy") != "curated_successive_halving":
        raise ValueError(
            "Pillar-0 search files must declare strategy=curated_successive_halving; "
            "Cartesian product grids are intentionally unsupported."
        )
    candidates = definition.get("candidates")
    rounds = definition.get("rounds")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("Search definition requires at least two explicit candidates.")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Search definition requires at least one budget round.")
    identities = [candidate_id(candidate) for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Search definition contains duplicate candidates.")
    previous_budget = -math.inf
    previous_keep = len(candidates) + 1
    for index, round_definition in enumerate(rounds, start=1):
        budget = int(round_definition["budget"])
        keep = int(round_definition.get("keep", 1))
        folds = [int(value) for value in round_definition.get("folds", range(1, 6))]
        if budget <= previous_budget:
            raise ValueError("Search budgets must increase monotonically.")
        if keep <= 0 or keep >= previous_keep:
            raise ValueError("Each search round must retain fewer positive candidates.")
        if not folds or any(value not in range(1, 6) for value in folds):
            raise ValueError(f"Invalid folds in search round {index}: {folds}")
        previous_budget, previous_keep = budget, keep
    return definition


def search_candidates(definition: dict, maximum: int | None = None) -> list[dict]:
    candidates = [canonical_parameters(item) for item in definition["candidates"]]
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("maximum candidates must be positive.")
        candidates = candidates[:maximum]
    return candidates


def rank_records(records: list[dict]) -> list[dict]:
    """Rank complete records using Merlin-compatible robust fold scoring."""

    required = {"mean_f1_macro", "std_f1_macro", "mean_balanced_accuracy", "std_balanced_accuracy"}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Search record is missing {sorted(missing)}")
        record["selection_score"] = 0.7 * (
            float(record["mean_f1_macro"]) - float(record["std_f1_macro"])
        ) + 0.3 * (
            float(record["mean_balanced_accuracy"])
            - float(record["std_balanced_accuracy"])
        )
    return sorted(
        records,
        key=lambda item: (
            -float(item["selection_score"]),
            -float(item.get("minority_perforation_recall", -1.0)),
            str(item.get("candidate_id", "")),
        ),
    )


def survivors(records: list[dict], keep: int) -> list[str]:
    if keep <= 0:
        raise ValueError("keep must be positive.")
    return [str(record["candidate_id"]) for record in rank_records(records)[:keep]]
