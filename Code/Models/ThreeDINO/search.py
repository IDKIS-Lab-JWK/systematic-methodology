"""Deterministic candidate construction and development-only ranking for 3DINO."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

try:
    from .metrics import grid_selection_score
except ImportError:  # pragma: no cover - direct script/test imports
    from metrics import grid_selection_score


SUPPORTED_STRATEGIES = {
    "coupled_candidates",
    "curated_successive_halving",
    "publication_multi_head",
}
SUPPORTED_LR_SCALE_RULES = {"none", "nominal", "global_batch_over_256"}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported parameter value: {type(value).__name__}")


def canonical_parameters(parameters: Mapping) -> dict:
    """Return a JSON-native, recursively key-sorted parameter dictionary."""

    if not isinstance(parameters, Mapping):
        raise TypeError("Parameters must be a mapping.")
    serialized = json.dumps(
        dict(parameters),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    value = json.loads(serialized)
    if not isinstance(value, dict):  # defensive; the input check already ensures it
        raise TypeError("Canonical parameters must be a dictionary.")
    return value


def parameter_fingerprint(parameters: Mapping, length: int = 64) -> str:
    """SHA-256 fingerprint used to reject incompatible checkpoint resumes."""

    if not 8 <= int(length) <= 64:
        raise ValueError("Fingerprint length must be between 8 and 64 characters.")
    payload = json.dumps(
        canonical_parameters(parameters),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[: int(length)]


def candidate_id(parameters: Mapping) -> str:
    """Short, key-order-independent identity used in result folder metadata."""

    identity_parameters = {
        key: value
        for key, value in canonical_parameters(parameters).items()
        if key not in {"candidate_id", "selection_score", "rank"}
    }
    return parameter_fingerprint(identity_parameters, length=12)


def _feature_recipes(definition: Mapping) -> list[dict]:
    search_space = definition.get("search_space", definition)
    recipes = search_space.get("feature_recipes", search_space.get("representations"))
    if recipes is None:
        n_values = search_space.get("n_last_blocks", [1, 4])
        mean_values = search_space.get("use_mean_patch_token", [False, True])
        recipes = [
            {
                "n_last_blocks": int(n_last_blocks),
                "use_mean_patch_token": bool(use_mean),
            }
            for n_last_blocks in n_values
            for use_mean in mean_values
        ]
    result = []
    for recipe in recipes:
        value = canonical_parameters(recipe)
        n_last_blocks = int(value["n_last_blocks"])
        if n_last_blocks not in (1, 4):
            raise ValueError(
                "Publication linear probing supports n_last_blocks in {1, 4}."
            )
        value["n_last_blocks"] = n_last_blocks
        value["use_mean_patch_token"] = bool(
            value.get("use_mean_patch_token", value.get("use_avgpool", False))
        )
        value.pop("use_avgpool", None)
        result.append(value)
    identities = [
        (value["n_last_blocks"], value["use_mean_patch_token"])
        for value in result
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Feature recipes contain duplicates.")
    return result


def effective_learning_rate(
    nominal_lr: float,
    *,
    batch_size: int,
    world_size: int = 1,
    scale_rule: str = "global_batch_over_256",
) -> float:
    """Reproduce the official code's optional global-batch LR scaling."""

    nominal_lr = float(nominal_lr)
    batch_size, world_size = int(batch_size), int(world_size)
    if not math.isfinite(nominal_lr) or nominal_lr <= 0:
        raise ValueError("nominal_lr must be finite and positive.")
    if batch_size <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive.")
    scale_rule = str(scale_rule).lower()
    if scale_rule not in SUPPORTED_LR_SCALE_RULES:
        raise ValueError(f"Unsupported learning-rate scale rule: {scale_rule}")
    if scale_rule == "global_batch_over_256":
        return nominal_lr * (batch_size * world_size) / 256.0
    return nominal_lr


def build_probe_candidates(
    definition: Mapping,
    *,
    world_size: int = 1,
) -> list[dict]:
    """Expand explicit candidates or the publication 13 x 4 head search."""

    definition = canonical_parameters(definition)
    fixed = canonical_parameters(definition.get("fixed_parameters", {}))
    if isinstance(definition.get("candidates"), list):
        raw_candidates = definition["candidates"]
    else:
        search_space = definition.get("search_space", definition)
        learning_rates = search_space.get(
            "learning_rates_nominal",
            search_space.get("learning_rates", definition.get("learning_rates")),
        )
        if not isinstance(learning_rates, list) or not learning_rates:
            # A resolved single-candidate profile is also accepted.
            scalar_lr = definition.get("nominal_lr", definition.get("lr"))
            if scalar_lr is None:
                raise ValueError("No linear-probe learning rate was supplied.")
            learning_rates = [scalar_lr]
        raw_candidates = [
            {**recipe, "nominal_lr": float(nominal_lr)}
            for recipe in _feature_recipes(definition)
            for nominal_lr in learning_rates
        ]

    batch_size = int(fixed.get("batch_size", definition.get("batch_size", 32)))
    scale_rule = str(
        fixed.get(
            "learning_rate_scale_rule",
            definition.get(
                "learning_rate_scale_rule",
                "global_batch_over_256"
                if definition.get("learning_rate_mode") == "linear_scale_batch_256"
                else "none",
            ),
        )
    )
    candidates: list[dict] = []
    for raw in raw_candidates:
        candidate = {**fixed, **canonical_parameters(raw)}
        candidate["n_last_blocks"] = int(candidate.get("n_last_blocks", 1))
        candidate["use_mean_patch_token"] = bool(
            candidate.get("use_mean_patch_token", candidate.get("use_avgpool", False))
        )
        candidate.pop("use_avgpool", None)
        if candidate["n_last_blocks"] not in (1, 4):
            raise ValueError("n_last_blocks must be 1 or 4.")
        nominal_lr = float(candidate.get("nominal_lr", candidate.get("lr")))
        supplied_effective_lr = candidate.get("lr")
        effective_lr = effective_learning_rate(
            nominal_lr,
            batch_size=int(candidate.get("batch_size", batch_size)),
            world_size=world_size,
            scale_rule=str(candidate.get("learning_rate_scale_rule", scale_rule)),
        )
        if supplied_effective_lr is not None and "nominal_lr" in candidate:
            if not math.isclose(
                float(supplied_effective_lr), effective_lr, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    "Candidate lr disagrees with nominal_lr and the declared scale rule."
                )
        candidate["nominal_lr"] = nominal_lr
        candidate["lr"] = effective_lr
        candidate["learning_rate_scale_rule"] = str(
            candidate.get("learning_rate_scale_rule", scale_rule)
        )
        candidate["candidate_id"] = candidate_id(candidate)
        candidates.append(candidate)
    identities = [value["candidate_id"] for value in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Linear-probe definition contains duplicate candidates.")
    return candidates


def build_search_candidates(
    definition: Mapping, *, world_size: int = 1
) -> list[dict]:
    """Resolve method-aware candidates without imposing probe fields on AdamW runs."""

    method = str(definition.get("method", "linear_probe")).lower()
    if method == "linear_probe":
        return build_probe_candidates(definition, world_size=world_size)
    if method != "full_finetune":
        raise ValueError(f"Unsupported 3DINO search method: {method}")
    fixed = canonical_parameters(definition.get("fixed_parameters", {}))
    raw_candidates = definition.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        parameters = definition.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("Full-fine-tune search requires explicit candidates.")
        raw_candidates = [parameters]
    candidates: list[dict] = []
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise TypeError("Every full-fine-tune candidate must be a mapping.")
        candidate = {**fixed, **canonical_parameters(raw)}
        candidate["candidate_id"] = candidate_id(candidate)
        candidates.append(candidate)
    identities = [value["candidate_id"] for value in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Full-fine-tune search contains duplicate candidates.")
    return candidates


def load_search_definition(path: Path) -> dict:
    """Load and validate either a curated or publication multi-head profile."""

    with Path(path).open("r", encoding="utf-8") as handle:
        definition = json.load(handle)
    strategy = definition.get("strategy")
    is_publication_grid = isinstance(definition.get("search_space"), dict)
    if strategy is not None and strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported 3DINO search strategy: {strategy}")
    if strategy is None and not is_publication_grid:
        raise ValueError(
            "3DINO searches require explicit coupled candidates, curated "
            "successive halving, or a publication search_space."
        )
    candidates = build_search_candidates(definition)
    if not candidates:
        raise ValueError("Search definition contains no candidates.")

    rounds = definition.get("rounds")
    if rounds is not None:
        if not isinstance(rounds, list) or not rounds:
            raise ValueError("rounds must be a non-empty list when supplied.")
        previous_budget = -math.inf
        previous_keep = len(candidates) + 1
        for index, round_definition in enumerate(rounds, start=1):
            budget = int(
                round_definition.get("budget", round_definition.get("max_iterations", 0))
            )
            keep = int(round_definition.get("keep", 1))
            folds = [int(value) for value in round_definition.get("folds", range(1, 6))]
            if budget <= previous_budget:
                raise ValueError("Search budgets must increase monotonically.")
            if keep <= 0 or keep >= previous_keep:
                raise ValueError("Each round must retain fewer positive candidates.")
            if not folds or any(value not in range(1, 6) for value in folds):
                raise ValueError(f"Invalid folds in search round {index}: {folds}")
            previous_budget, previous_keep = budget, keep
    definition["resolved_candidates"] = candidates
    return definition


def search_candidates(definition: Mapping, maximum: int | None = None) -> list[dict]:
    candidates = definition.get("resolved_candidates")
    if candidates is None:
        candidates = build_search_candidates(definition)
    candidates = [canonical_parameters(value) for value in candidates]
    if maximum is not None:
        if int(maximum) <= 0:
            raise ValueError("maximum candidates must be positive.")
        candidates = candidates[: int(maximum)]
    return candidates


def rank_records(records: list[dict]) -> list[dict]:
    """Rank complete CV records by robust score and prespecified tie-breakers."""

    required = {
        "mean_f1_macro",
        "std_f1_macro",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
    }
    ranked: list[dict] = []
    for original in records:
        record = dict(original)
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Search record is missing {sorted(missing)}")
        record["selection_score"] = grid_selection_score(
            record["mean_f1_macro"],
            record["std_f1_macro"],
            record["mean_balanced_accuracy"],
            record["std_balanced_accuracy"],
        )
        ranked.append(record)
    def learning_rate_tie_breaker(item: Mapping) -> float:
        value = item.get("nominal_lr")
        if value is None:
            value = item.get("lr")
        return math.inf if value is None else float(value)

    ranked.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            -float(item.get("minority_perforation_recall", -1.0)),
            learning_rate_tie_breaker(item),
            str(item.get("candidate_id", "")),
        )
    )
    for index, record in enumerate(ranked, start=1):
        record["rank"] = index
    return ranked


def aggregate_fold_records(
    fold_records: Iterable[Mapping], *, minority_class: str = "Perforation"
) -> list[dict]:
    """Aggregate per-fold metrics into the records consumed by ``rank_records``."""

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for record in fold_records:
        candidate = str(record.get("candidate_id", ""))
        if not candidate:
            raise ValueError("Every fold record requires candidate_id.")
        grouped[candidate].append(record)
    aggregated: list[dict] = []
    for identity, values in grouped.items():
        f1_values = np_values(values, "f1_macro")
        ba_values = np_values(values, "balanced_accuracy")
        recalls = []
        for value in values:
            per_class = value.get("per_class", {})
            recall = per_class.get(minority_class, {}).get(
                "recall", per_class.get(minority_class, {}).get("sensitivity")
            )
            if recall is not None and math.isfinite(float(recall)):
                recalls.append(float(recall))
        aggregate = {
                "candidate_id": identity,
                "fold_count": len(values),
                "mean_f1_macro": sum(f1_values) / len(f1_values),
                "std_f1_macro": population_std(f1_values),
                "mean_balanced_accuracy": sum(ba_values) / len(ba_values),
                "std_balanced_accuracy": population_std(ba_values),
                "minority_perforation_recall": (
                    sum(recalls) / len(recalls) if recalls else -1.0
                ),
            }
        nominal_lr = values[0].get("nominal_lr")
        if nominal_lr is not None:
            aggregate["nominal_lr"] = nominal_lr
        aggregated.append(aggregate)
    return rank_records(aggregated)


def np_values(records: Iterable[Mapping], key: str) -> list[float]:
    values = [float(record[key]) for record in records]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{key} values must be non-empty and finite.")
    return values


def population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def survivors(records: list[dict], keep: int) -> list[str]:
    if int(keep) <= 0:
        raise ValueError("keep must be positive.")
    return [
        str(record["candidate_id"])
        for record in rank_records(records)[: int(keep)]
    ]
