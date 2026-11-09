"""Archive redundant fixed-publication grids and rebind CV/Test provenance.

Run prepare BEFORE editing the packaged profiles/workflow, then apply/verify.
Only metadata is rewritten; trained tensors, predictions, and fold seals remain
byte-identical. Archives live outside Results so result discovery cannot double count.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

MODELS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS))

from _linear_probe.common import (
    ContractError, atomic_write_json, completion_seal, fingerprint,
    load_manifest, read_json, sha256_file, tree_completion_seal,
    validate_completion, validate_tree_completion, utc_now,
)
from _supervised_adaptation.profiles import build_candidates
from _supervised_adaptation.trainer import AdaptationBackend, _implementation_identity
from _supervised_adaptation.workflow import (
    _cv_completion_context, _cv_identity, _cv_provenance_identity,
    _validate_cv, _validate_grid, _validate_test,
)

ARCHIVE_NAME = "FixedPublication_20260906"
MANIFEST_NAME = "FIXED_PUBLICATION_MIGRATION.json"
TARGETS = {
    "FinalBlockFineTune": "publication_final_block_finetune.json",
    "FrozenSingleHeadAttentionReadout": "publication_attention_readout.json",
}


def contained(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root):
        raise ContractError(f"Target must be strictly inside {root}: {path}")
    return path


def snapshot_tree(source: Path, destination: Path) -> dict:
    records = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        target = destination / relative
        digest = sha256_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != digest:
                raise ContractError(f"Archive collision: {target}")
        elif path.suffix in (".pt", ".pth", ".safetensors"):
            # These archived tensor files are immutable; metadata uses independent copies.
            os.link(path, target)
        else:
            shutil.copy2(path, target)
        if sha256_file(target) != digest:
            raise ContractError(f"Archive verification failed: {target}")
        records[relative] = digest
    return records


def inventory(source: Path) -> dict:
    return {p.relative_to(source).as_posix(): sha256_file(p)
            for p in sorted(source.rglob("*")) if p.is_file()}


def sealed_document(document: dict) -> dict:
    return {**document, "manifest_sha256": fingerprint(document)}


def read_sealed(path: Path) -> dict:
    document = read_json(path)
    digest = document.pop("manifest_sha256", None)
    if digest != fingerprint(document):
        raise ContractError(f"Migration manifest changed: {path}")
    return document


def validate_cv_tree(cv: Path, dataset: Path) -> None:
    context = read_json(cv / "cv_run_context.json")
    manifest = load_manifest(dataset, int(context["percentage"]))
    _validate_cv(
        cv, expected_context=context, backend=AdaptationBackend,
        args=SimpleNamespace(dataset_root=dataset), manifest=manifest,
    )


def prepare(root: Path, archive: Path) -> dict:
    plan_path = archive / "plan.json"
    if plan_path.exists():
        return read_sealed(plan_path)
    methods = []
    for name, profile_name in TARGETS.items():
        method = root / "Results" / "FlexiCT" / name
        grids = method / "Publication_Grid"
        if not grids.exists():
            continue
        profile_path = root / "Code" / "Models" / "FlexiCT" / profile_name
        candidates = build_candidates(read_json(profile_path))
        if len(candidates) != 1:
            raise ContractError(f"Cannot migrate genuine parameter search: {profile_path}")
        registry = read_json(method / "CV" / "cv_registry.json")
        covered = []
        for grid in sorted(grids.iterdir()):
            if not grid.is_dir():
                raise ContractError(f"Unexpected grid artifact: {grid}")
            selected = _validate_grid(grid)
            gc = read_json(grid / "grid_context.json")
            if gc["candidate_count"] != 1 or selected["parameters"] != candidates[0]:
                raise ContractError(f"Grid does not match fixed publication recipe: {grid}")
            matches = []
            for entry in registry["runs"]:
                cv = method / "CV" / entry["cv_id"] / grid.name
                if not (cv / "cv_run_context.json").is_file():
                    continue
                context = read_json(cv / "cv_run_context.json")
                if (context["parameters"] == candidates[0]
                    and context["model_identity"] == gc["model_identity"]
                    and context["parameter_provenance"].get("grid_root") == str(grid.resolve())):
                    validate_cv_tree(cv, root / "Dataset")
                    for fold in range(1, 6):
                        source = grid / f"candidate_{selected['candidate_id']}" / f"fold_{fold:02d}"
                        dest = cv / f"fold_{fold:02d}"
                        seal = validate_completion(source, "completion.json")
                        for record in seal["artifacts"]:
                            if sha256_file(dest / record["path"]) != record["sha256"]:
                                raise ContractError(f"CV materialization differs from original grid: {dest}")
                    matches.append(entry["cv_id"])
            if len(matches) != 1:
                raise ContractError(f"Expected one complete CV copy for {grid}, found {matches}")
            covered.append({"percentage": gc["percentage"], "cv_id": matches[0]})
        # Archive CV/Test headers and tensors before any rewrite. Grid is moved
        # intact only after canonical CV/Test verification during apply.
        snapshots = {}
        for branch in ("CV", "Test"):
            if (method / branch).exists():
                snapshots[branch] = snapshot_tree(
                    method / branch, archive / "Results" / "FlexiCT" / name / branch
                )
        methods.append({
            "model": "FlexiCT", "method": name, "profile": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "parameters": candidates[0], "covered": covered,
            "snapshots": snapshots, "grid_inventory": inventory(grids),
        })
        print(f"Prepared {name}: {len(covered)} completed percentages", flush=True)

    compatibility = []
    for registry_path in sorted((root / "Results").glob("*/*/CV/cv_registry.json")):
        registry = read_json(registry_path)
        entries = []
        for run in registry.get("runs", []):
            identity = run.get("identity", {})
            model_identity = identity.get("model_identity", {})
            if model_identity.get("schema_version") != "supervised-adaptation-v1":
                continue
            for context_path in sorted((registry_path.parent / run["cv_id"]).glob("*_percent/cv_run_context.json")):
                context = read_json(context_path)
                # Only workflow and CLI changes are approved. Any model/trainer/
                # preprocessing difference requires its own reviewed migration.
                for record in model_identity["implementation"]["files"]:
                    p = Path(record["path"])
                    if sha256_file(p) != record["sha256"] and p not in (
                        MODELS / "_supervised_adaptation" / "workflow.py",
                        MODELS / "_supervised_adaptation" / "cli.py",
                    ):
                        raise ContractError(f"Unreviewed implementation difference: {p}")
                entries.append({
                    "cv_id": run["cv_id"], "percentage": context["percentage"],
                    "seed": context["seed"],
                    "parameter_source": context["parameter_provenance"]["parameter_source"],
                    "legacy_model_identity": model_identity,
                    "profile": context["parameter_provenance"]["profile"],
                    "partial": not (context_path.parent / "cv_completion.json").exists(),
                })
        if entries:
            compatibility.append({"method_root": str(registry_path.parent.parent), "entries": entries})
    sources = archive / "OriginalCode"
    for directory in (MODELS / "_supervised_adaptation", MODELS / "FlexiCT"):
        for p in directory.iterdir():
            if p.is_file() and p.suffix in (".py", ".json"):
                target = sources / directory.name / p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256_file(target) != sha256_file(p):
                    raise ContractError(f"Original code snapshot differs: {target}")
                shutil.copy2(p, target)
    plan = {
        "schema_version": 1, "created_at": utc_now(), "root": str(root),
        "archive": str(archive), "methods": methods, "compatibility": compatibility,
    }
    atomic_write_json(plan_path, sealed_document(plan))
    return plan


def checked_write(path: Path, document: dict, original: Path) -> None:
    """Idempotent metadata replacement, rejecting changes after preparation."""
    existing = read_json(path)
    if existing == document:
        return
    if existing != read_json(original):
        raise ContractError(f"Metadata changed after preparation: {path}")
    atomic_write_json(path, document)


def provenance_for(item: dict) -> dict:
    return {"parameter_source": "publication", "profile": item["profile"],
            "profile_sha256": sha256_file(item["profile"])}


def apply(root: Path, archive: Path, plan: dict) -> dict:
    for item in plan["methods"]:
        method = contained(root / "Results" / item["model"] / item["method"], root)
        saved = archive / "Results" / item["model"] / item["method"]
        profile = read_json(item["profile"])
        if profile.get("parameter_selection") != "fixed" or build_candidates(profile) != [item["parameters"]]:
            raise ContractError("Migration may change profile routing only, never executable parameters")
        provenance = provenance_for(item)
        registry = read_json(saved / "CV" / "cv_registry.json")
        for entry in registry["runs"]:
            entry["identity"]["parameter_provenance_identity"] = _cv_provenance_identity(provenance)
            entry["identity_sha256"] = fingerprint(entry["identity"])
            path = method / "CV" / entry["cv_id"] / "cv_parameters.json"
            original = saved / "CV" / entry["cv_id"] / "cv_parameters.json"
            document = read_json(original)
            document["identity"] = entry["identity"]
            checked_write(path, document, original)
        checked_write(method / "CV" / "cv_registry.json", registry, saved / "CV" / "cv_registry.json")
        for run in item["covered"]:
            relative = Path("CV") / run["cv_id"] / f"{run['percentage']}_percent"
            cv, original_cv = method / relative, saved / relative
            context = read_json(original_cv / "cv_run_context.json")
            context["parameter_provenance"] = provenance
            context["parameter_provenance_identity"] = _cv_provenance_identity(provenance)
            checked_write(cv / "cv_run_context.json", context, original_cv / "cv_run_context.json")
            # Fold/grid_reuse records and fold tensors are immutable historical
            # evidence. They are no longer used for parameter lookup/execution.
            original_seal = read_json(original_cv / "cv_completion.json")
            seal = tree_completion_seal(
                cv, [cv / r["path"] for r in original_seal["artifacts"]],
                context=_cv_completion_context(context, read_json(cv / "refit_selection.json")),
            )
            seal["completed_at"] = original_seal["completed_at"]
            checked_write(cv / "cv_completion.json", seal, original_cv / "cv_completion.json")
            validate_cv_tree(cv, root / "Dataset")
        for original_selection in sorted((saved / "Test").glob("CV_*/*/*_percent/model_selection.json")):
            relative = original_selection.parent.relative_to(saved)
            test, original_test = method / relative, saved / relative
            old_seal = validate_completion(original_test, "test_completion.json")
            selection = read_json(original_selection)
            selection["parameter_provenance"] = provenance
            for filename in ("test_context.json", "model_selection.json"):
                checked_write(test / filename, selection, original_test / filename)
            seal = completion_seal(
                [test / r["path"] for r in old_seal["artifacts"]],
                context={"cv_id": selection["cv_id"], "percentage": selection["percentage"],
                         "model_source": selection["model_source"], "selection_sha256": fingerprint(selection)},
            )
            seal["completed_at"] = old_seal["completed_at"]
            checked_write(test / "test_completion.json", seal, original_test / "test_completion.json")
        print(f"Canonical metadata verified: {item['method']}", flush=True)

    # Compatibility permits only these exact reviewed implementation hashes.
    # The original checkpoint identity stays unchanged inside binary files.
    for group in plan["compatibility"]:
        method = Path(group["method_root"])
        entries = copy.deepcopy(group["entries"])
        for entry in entries:
            old = entry["legacy_model_identity"]
            paths = [Path(r["path"]) for r in old["implementation"]["files"]]
            paths.append(MODELS / "_supervised_adaptation" / "compatibility.py")
            entry["accepted_implementation"] = _implementation_identity(paths)
            entry["profile_sha256"] = sha256_file(entry["profile"])
        matching = [i for i in plan["methods"] if i["method"] == method.name]
        mappings = []
        for item in matching:
            for run in item["covered"]:
                old = method / "Publication_Grid" / f"{run['percentage']}_percent"
                canonical = method / "CV" / run["cv_id"] / old.name
                mappings.append({"historical_grid": str(old), "canonical_cv": str(canonical),
                                 "archive_grid": str(archive / old.relative_to(root)),
                                 "candidate_id": item["parameters"]["candidate_id"]})
        document = sealed_document({
            "schema_version": 1, "kind": "fixed_publication_routing_and_code_compatibility",
            "archive": str(archive), "entries": entries, "path_mappings": mappings,
        })
        manifest = method / MANIFEST_NAME
        if manifest.exists() and read_json(manifest) != document:
            raise ContractError(f"Compatibility manifest collision: {manifest}")
        atomic_write_json(manifest, document)

    # Move, rather than delete, original grid trees only after verification.
    verify(root, archive, plan, require_archived=False)
    for item in plan["methods"]:
        source = contained(root / "Results" / item["model"] / item["method"] / "Publication_Grid", root)
        destination = contained(archive / source.relative_to(root), archive)
        if source.exists():
            if destination.exists():
                raise ContractError(f"Both active and archived grids exist: {source}")
            if inventory(source) != item["grid_inventory"]:
                raise ContractError(f"Grid artifacts changed during migration: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        elif not destination.is_dir():
            raise ContractError(f"Missing grid archive: {destination}")
    return verify(root, archive, plan)


def verify(root: Path, archive: Path, plan: dict, require_archived=True) -> dict:
    from _supervised_adaptation.compatibility import migrated_model_identity
    reports = []
    for group in plan["compatibility"]:
        method = Path(group["method_root"])
        document = read_sealed(method / MANIFEST_NAME)
        for entry in document["entries"]:
            current = copy.deepcopy(entry["legacy_model_identity"])
            current["implementation"] = _implementation_identity(
                [Path(r["path"]) for r in entry["accepted_implementation"]["files"]]
            )
            args = SimpleNamespace(results_root=method.parent, percentage=entry["percentage"],
                                   seed=entry["seed"], parameter_source=entry["parameter_source"])
            if migrated_model_identity(current, args) != entry["legacy_model_identity"]:
                raise ContractError("Runtime model compatibility lookup failed")
            if entry["partial"]:
                from _supervised_adaptation.runtime import torch_load
                cv = method / "CV" / entry["cv_id"] / f"{entry['percentage']}_percent"
                for checkpoint in cv.glob("*/last_checkpoint.pt"):
                    context = read_json(checkpoint.parent / "training_context.json")
                    payload = torch_load(checkpoint, map_location="cpu")
                    if (payload["identity"] != context
                        or payload["identity_sha256"] != fingerprint(context)
                        or context["model_identity"] != entry["legacy_model_identity"]):
                        raise ContractError(f"Partial checkpoint identity mismatch: {checkpoint}")
                    print(f"Checkpoint preserved: {checkpoint.parent.name}, step {payload['completed_optimizer_steps']}", flush=True)
    for item in plan["methods"]:
        method = root / "Results" / item["model"] / item["method"]
        saved = archive / "Results" / item["model"] / item["method"]
        rewritable = {"cv_registry.json", "cv_parameters.json", "cv_run_context.json", "cv_completion.json",
                      "test_context.json", "model_selection.json", "test_completion.json"}
        unchanged = 0
        for branch, records in item["snapshots"].items():
            for relative, digest in records.items():
                if sha256_file(saved / branch / relative) != digest:
                    raise ContractError(f"Archive changed: {saved / branch / relative}")
                if Path(relative).name not in rewritable:
                    if sha256_file(method / branch / relative) != digest:
                        raise ContractError(f"Scientific result changed: {method / branch / relative}")
                    unchanged += 1
        registry = read_json(method / "CV" / "cv_registry.json")
        for run in item["covered"]:
            cv = method / "CV" / run["cv_id"] / f"{run['percentage']}_percent"
            validate_cv_tree(cv, root / "Dataset")
            context = read_json(cv / "cv_run_context.json")
            identity = _cv_identity(
                SimpleNamespace(name=item["model"], method_label=item["method"], method_id=context["method_id"]),
                SimpleNamespace(seed=context["seed"]), context["parameters"],
                provenance_for(item), context["model_identity"],
            )
            matches = [r for r in registry["runs"] if r["identity_sha256"] == fingerprint(identity)]
            if len(matches) != 1 or matches[0]["cv_id"] != run["cv_id"]:
                raise ContractError(f"Canonical fixed CV lookup failed: {cv}")
            # Exercise real CV/Test lookup without permitting any training or
            # inference. Concrete input inventories and all seals are checked.
            from _supervised_adaptation.workflow import run_cv, run_test
            from _supervised_adaptation.trainer import input_rows_identity as file_identity
            class VerifyBackend:
                name = item["model"]
                method_label = item["method"]
                method_id = context["method_id"]
                artifact_filename = "model.pt"
                publication_profile_path = Path(item["profile"])
                build_candidates = staticmethod(build_candidates)
                refit_budget = staticmethod(AdaptationBackend.refit_budget)
                input_rows_identity = staticmethod(file_identity)

                def ensure_feature_rows(self, args, rows):
                    pass

                def feature_identity(self, args):
                    return context["model_identity"]

                def fit_one(self, *args, **kwargs):
                    raise AssertionError("Verification must never train")

                refit = fit_one
                predict = fit_one

            runtime_args = SimpleNamespace(
                results_root=method.parent, dataset_root=root / "Dataset",
                percentage=run["percentage"], seed=context["seed"], resume=True,
                parameter_source="publication", parameter_config=None, grid_config=None,
                max_grid_candidates=None,
            )
            manifest = load_manifest(runtime_args.dataset_root, runtime_args.percentage)
            if run_cv(runtime_args, VerifyBackend(), manifest)["status"] != "reused":
                raise ContractError("Existing canonical CV was not reused")
            for test_seal in (method / "Test" / run["cv_id"]).glob(f"*/{run['percentage']}_percent/test_completion.json"):
                runtime_args.model_source = test_seal.parent.parent.name
                if run_test(runtime_args, VerifyBackend(), manifest)["status"] != "reused":
                    raise ContractError("Existing canonical test was not reused")
        tests = 0
        for path in (method / "Test").glob("CV_*/*/*_percent/test_completion.json"):
            selection = read_json(path.parent / "model_selection.json")
            from _linear_probe.common import heldout_mapping, rows_from_mapping
            manifest = load_manifest(root / "Dataset", selection["percentage"])
            rows = rows_from_mapping(root / "Dataset", heldout_mapping(manifest), require_files=False)
            _validate_test(path.parent, selection=selection, expected_rows=rows)
            if selection["parameter_provenance"] != provenance_for(item):
                raise ContractError(f"Test still requires a grid: {path}")
            for artifact in selection["artifacts"]:
                if sha256_file(artifact["path"]) != artifact["sha256"]:
                    raise ContractError(f"Test model artifact changed: {artifact['path']}")
            tests += 1
        if require_archived:
            if (method / "Publication_Grid").exists():
                raise ContractError(f"Redundant active grid remains: {method}")
            if inventory(saved / "Publication_Grid") != item["grid_inventory"]:
                raise ContractError(f"Archived grid differs: {saved}")
        reports.append({"method": item["method"], "completed_cv_percentages": len(item["covered"]),
                        "completed_tests": tests, "unchanged_result_files": unchanged})
    return {"status": "verified", "archive": str(archive), "methods": reports,
            "trained_or_inferred": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "apply", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.resolve()
    archive = contained(root / "Archives" / ARCHIVE_NAME, root)
    if args.stage == "prepare":
        plan = prepare(root, archive)
        print(json.dumps({"status": "prepared", "methods": len(plan["methods"]), "archive": str(archive)}, indent=2))
    else:
        plan = read_sealed(archive / "plan.json")
        if plan["root"] != str(root) or plan["archive"] != str(archive):
            raise ContractError("Migration root differs from prepared plan")
        result = apply(root, archive, plan) if args.stage == "apply" else verify(root, archive, plan)
        if args.stage == "apply":
            atomic_write_json(archive / "verification.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
