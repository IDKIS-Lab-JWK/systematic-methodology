"""Exact, recorded code compatibility for migrated supervised experiments."""

from pathlib import Path

from _linear_probe.common import ContractError, fingerprint, read_json, sha256_file
from .trainer import AdaptationBackend, _implementation_identity

MANIFEST_NAME = "FIXED_PUBLICATION_MIGRATION.json"


def read_migration(method_root):
    path = Path(method_root) / MANIFEST_NAME
    if not path.is_file():
        return None
    doc = read_json(path)
    digest = doc.pop("manifest_sha256", None)
    if doc.get("schema_version") != 1 or digest != fingerprint(doc):
        raise ContractError(f"Invalid fixed-publication migration manifest: {path}")
    return doc


def migrated_model_identity(current, args):
    method_root = Path(args.results_root) / current["method"]
    migration = read_migration(method_root)
    if migration is None:
        return current
    entries = [entry for entry in migration["entries"]
               if entry["percentage"] == int(args.percentage)
               and entry["seed"] == int(args.seed)
               and entry["parameter_source"] == args.parameter_source]
    if not entries:
        return current
    if len(entries) != 1:
        raise ContractError(f"Ambiguous migrated experiment: {method_root}")
    entry = entries[0]
    old = entry["legacy_model_identity"]
    if ({k: v for k, v in current.items() if k != "implementation"}
        != {k: v for k, v in old.items() if k != "implementation"}):
        raise ContractError("Migrated experiment checkpoint or representation changed")
    if current["implementation"] != entry["accepted_implementation"]:
        raise ContractError("Implementation differs from the exact reviewed migration version")
    if sha256_file(entry["profile"]) != entry["profile_sha256"]:
        raise ContractError("Migrated experiment profile changed")
    return dict(old)


class CompatibleAdaptationBackend(AdaptationBackend):
    """Keep historical tensor identity only for explicitly migrated runs.

    All checkpoint checks in the original trainer remain exact, including batch,
    precision, optimizer/RNG state, runtime, input inventory, and patient splits.
    """

    def feature_identity(self, args):
        current = super().feature_identity(args)
        current["implementation"] = _implementation_identity(
            [Path(r["path"]) for r in current["implementation"]["files"]]
            + [Path(__file__)]
        )
        return migrated_model_identity(current, args)


def resolve_historical_artifact(method_root, historical_path):
    """Resolve preserved grid-origin references to canonical CV or the archive."""
    original = Path(historical_path).resolve()
    if original.is_file():
        return original
    document = read_migration(method_root)
    if document is None:
        raise FileNotFoundError(original)
    for mapping in document["path_mappings"]:
        grid = Path(mapping["historical_grid"]).resolve()
        if not original.is_relative_to(grid):
            continue
        relative = original.relative_to(grid)
        if relative.parts and relative.parts[0] == f"candidate_{mapping['candidate_id']}":
            canonical_root = Path(mapping["canonical_cv"]).resolve()
            target = (canonical_root / Path(*relative.parts[1:])).resolve()
            if target.is_relative_to(canonical_root) and target.is_file():
                return target
        archive_root = Path(mapping["archive_grid"]).resolve()
        target = (archive_root / relative).resolve()
        if target.is_relative_to(archive_root) and target.is_file():
            return target
    raise FileNotFoundError(original)
