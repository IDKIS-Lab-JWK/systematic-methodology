"""Dependency, dataset, cache, and automatically provisioned asset preflight."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # Package imports.
    from . import (
        DEFAULT_CACHE_NAMESPACE,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REVISION,
    )
    from .cache import cache_directory, validate_cache
    from .data import collect_required_rows, load_manifest
    from .provisioning import provision_assets
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import (
        DEFAULT_CACHE_NAMESPACE,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_SIZE_BYTES,
        UPSTREAM_REVISION,
    )
    from cache import cache_directory, validate_cache
    from data import collect_required_rows, load_manifest
    from provisioning import provision_assets


SUPPORTED_RUNTIME = {
    "platform": "win32",
    "implementation": "CPython",
    "python_major_minor": [3, 10],
    "pointer_bits": 64,
}

# These are the exact binary-era pins inherited from the official 3DINO
# requirements.  A local build suffix such as ``+cu117`` is accepted when the
# public/base version is exact.
CORE_PACKAGE_POLICY = {
    "torch": {"distribution": "torch", "specifier": "==2.0.0", "native": "torch._C"},
    "torchvision": {"distribution": "torchvision", "specifier": "==0.15.0"},
    "xformers": {
        "distribution": "xformers",
        "specifier": "==0.0.18",
        "native": "xformers C++/CUDA extension",
    },
    "torchmetrics": {"distribution": "torchmetrics", "specifier": "==0.10.3"},
    "monai": {"distribution": "monai", "specifier": "==1.3.0"},
    "nibabel": {"distribution": "nibabel", "specifier": "==5.1.0"},
}

# Project constraints around the remaining required source, preprocessing, and
# systematic-evaluation imports.  They are checked as bounded/minimum ranges.
REQUIRED_PACKAGE_POLICY = {
    "omegaconf": {"distribution": "omegaconf", "specifier": ">=2.3,<3"},
    "fvcore": {"distribution": "fvcore", "specifier": ">=0.1.5"},
    "iopath": {"distribution": "iopath", "specifier": ">=0.1.10"},
    "submitit": {"distribution": "submitit", "specifier": ">=1.4"},
    "torchio": {"distribution": "torchio", "specifier": ">=0.18,<1"},
    "numpy": {"distribution": "numpy", "specifier": ">=1.24,<2"},
    "pandas": {"distribution": "pandas", "specifier": ">=1.5,<3"},
    "sklearn": {"distribution": "scikit-learn", "specifier": ">=1.2,<2"},
    "scipy": {"distribution": "scipy", "specifier": ">=1.10,<2"},
    "yaml": {"distribution": "PyYAML", "specifier": ">=6,<7"},
    "huggingface_hub": {
        "distribution": "huggingface-hub",
        "specifier": ">=0.30,<1",
    },
    "tqdm": {"distribution": "tqdm", "specifier": ">=4.65,<5"},
}


def _load_model_function(name: str) -> Any:
    """Import the torch-bearing model adapter only after runtime validation."""

    if __package__:
        module = importlib.import_module(".model", package=__package__)
    else:  # pragma: no cover - direct invocation from this folder.
        module = importlib.import_module("model")
    return getattr(module, name)


def validate_upstream_checkout(*args: Any, **kwargs: Any) -> dict:
    return _load_model_function("validate_upstream_checkout")(*args, **kwargs)


def resolve_checkpoint(*args: Any, **kwargs: Any) -> Any:
    return _load_model_function("resolve_checkpoint")(*args, **kwargs)


def load_official_model(*args: Any, **kwargs: Any) -> Any:
    return _load_model_function("load_official_model")(*args, **kwargs)


def methodology_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = methodology_root()
    parser = argparse.ArgumentParser(
        description="Validate the isolated 3DINO environment without training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=Path, default=root / "Dataset")
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument(
        "--upstream_root", type=Path, default=root / "tmp" / "upstream_3dino"
    )
    parser.add_argument("--upstream_revision", default=UPSTREAM_REVISION)
    parser.add_argument("--model_repo_id", default=MODEL_REPO_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Optional verified local checkpoint override; otherwise use Hugging Face.",
    )
    parser.add_argument("--checkpoint_sha256", default=MODEL_SHA256)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not clone or download; require existing source and checkpoint/cache.",
    )
    parser.add_argument(
        "--no_auto_provision",
        action="store_true",
        help="Inspect only explicit local asset paths instead of provisioning missing assets.",
    )
    parser.add_argument(
        "--provision_log",
        type=Path,
        default=root / "Results" / "3DINO" / "Setup" / "asset_provisioning.jsonl",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=root / "Results" / "3DINO" / "Cache",
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Also construct the official model on CPU/GPU and verify strict weight loading.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip_artifact_validation",
        action="store_true",
        help=(
            "Perform structural-only cache checks without reading, hashing, or "
            "deserializing tensors; this is never model-execution-ready."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Results" / "3DINO" / "Setup" / "setup_preflight.json",
    )
    return parser


def _base_version(value: str) -> str:
    """Return the public version while allowing wheel tags such as ``+cu117``."""

    return str(value).strip().split("+", 1)[0]


def _numeric_version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", _base_version(value))
    if match is None:
        raise ValueError(f"Unsupported package version format: {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def _compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    lhs = left + (0,) * (width - len(left))
    rhs = right + (0,) * (width - len(right))
    return (lhs > rhs) - (lhs < rhs)


def _version_satisfies(value: str, specifier: str) -> bool:
    """Evaluate the simple exact/range clauses used by requirements.txt."""

    public = _base_version(value)
    installed = _numeric_version(public)
    for clause in (part.strip() for part in specifier.split(",")):
        match = re.fullmatch(r"(==|>=|<=|>|<)(\d+(?:\.\d+)*)", clause)
        if match is None:
            raise ValueError(f"Unsupported version constraint: {clause!r}")
        operator, required_text = match.groups()
        if operator == "==":
            if public != required_text:
                return False
            continue
        comparison = _compare_versions(installed, _numeric_version(required_text))
        if operator == ">=" and comparison < 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
    return True


def inspect_runtime_environment(
    *,
    device: Any = "cpu",
    version_info: tuple[int, int, int] | None = None,
    implementation: str | None = None,
    platform_name: str | None = None,
    pointer_bits: int | None = None,
    import_module: Any = importlib.import_module,
    distribution_version: Any = importlib.metadata.version,
) -> dict:
    """Import and version-check the one supported Windows compatibility stack.

    Injection points are intentionally limited to read-only runtime discovery;
    they make this fail-closed policy testable without installing or importing
    heavyweight binaries in unit tests.
    """

    detected_version = tuple(version_info or sys.version_info[:3])
    detected_implementation = implementation or platform.python_implementation()
    detected_platform = platform_name or sys.platform
    detected_bits = int(pointer_bits or struct.calcsize("P") * 8)
    python_checks = {
        "platform": detected_platform == SUPPORTED_RUNTIME["platform"],
        "implementation": detected_implementation
        == SUPPORTED_RUNTIME["implementation"],
        "python_major_minor": list(detected_version[:2])
        == SUPPORTED_RUNTIME["python_major_minor"],
        "pointer_bits": detected_bits == SUPPORTED_RUNTIME["pointer_bits"],
    }

    package_policy: dict[str, dict[str, Any]] = {
        name: dict(policy) for name, policy in CORE_PACKAGE_POLICY.items()
    }
    package_policy.update(
        {name: dict(policy) for name, policy in REQUIRED_PACKAGE_POLICY.items()}
    )
    packages: dict[str, dict[str, Any]] = {}
    imported: dict[str, Any] = {}
    for module_name, policy in package_policy.items():
        expected = str(policy["specifier"])
        record: dict[str, Any] = {
            "distribution": policy["distribution"],
            "required_version": expected,
            "importable": False,
            "version_compatible": False,
            "native_extension_loaded": None,
        }
        try:
            installed = str(distribution_version(policy["distribution"]))
        except Exception as exc:
            record.update(
                {
                    "installed_version": None,
                    "version_error_type": type(exc).__name__,
                    "version_error": str(exc),
                }
            )
        else:
            record["installed_version"] = installed
            try:
                record["version_compatible"] = _version_satisfies(
                    installed, expected
                )
            except ValueError as exc:
                record["version_compatible"] = False
                record["version_error_type"] = type(exc).__name__
                record["version_error"] = str(exc)
        try:
            module = import_module(module_name)
        except Exception as exc:
            record.update(
                {
                    "import_error_type": type(exc).__name__,
                    "import_error": str(exc),
                }
            )
        else:
            record["importable"] = True
            imported[module_name] = module
        packages[module_name] = record

    torch_module = imported.get("torch")
    if torch_module is not None:
        torch_native = getattr(torch_module, "_C", None) is not None
        packages["torch"]["native_extension_loaded"] = torch_native
        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            packages["torch"]["cuda_probe_error_type"] = type(exc).__name__
            packages["torch"]["cuda_probe_error"] = str(exc)
        packages["torch"]["cuda_available"] = cuda_available
        packages["torch"]["compiled_cuda_version"] = getattr(
            getattr(torch_module, "version", None), "cuda", None
        )
    else:
        packages["torch"]["native_extension_loaded"] = False
        packages["torch"]["cuda_available"] = False

    xformers_module = imported.get("xformers")
    if xformers_module is not None:
        cpp_library = getattr(xformers_module, "_has_cpp_library", None)
        if cpp_library is None:
            cpp_state = getattr(xformers_module, "_cpp_lib", None)
            if cpp_state is not None:
                cpp_library = (
                    getattr(cpp_state, "_cpp_library_load_exception", object())
                    is None
                )
        packages["xformers"]["native_extension_loaded"] = cpp_library is True
        if cpp_library is not True:
            cpp_state = getattr(xformers_module, "_cpp_lib", None)
            error = getattr(cpp_state, "_cpp_library_load_exception", None)
            if error is not None:
                packages["xformers"]["native_error_type"] = type(error).__name__
                packages["xformers"]["native_error"] = str(error)
    else:
        packages["xformers"]["native_extension_loaded"] = False

    requested_device = str(device)
    device_compatible = not requested_device.lower().startswith("cuda") or bool(
        packages["torch"].get("cuda_available")
    )
    for name, record in packages.items():
        native_required = name in {"torch", "xformers"}
        record["compatible"] = bool(
            record["importable"]
            and record["version_compatible"]
            and (
                not native_required
                or record.get("native_extension_loaded") is True
            )
        )

    compatible = bool(
        all(python_checks.values())
        and all(record["compatible"] for record in packages.values())
        and device_compatible
    )
    return {
        "policy": {
            **SUPPORTED_RUNTIME,
            "core_packages": {
                name: policy["specifier"]
                for name, policy in CORE_PACKAGE_POLICY.items()
            },
            "required_package_ranges": {
                name: policy["specifier"]
                for name, policy in REQUIRED_PACKAGE_POLICY.items()
            },
            "local_build_suffixes_allowed": True,
            "xformers_native_extension_required": True,
        },
        "detected": {
            "platform": detected_platform,
            "implementation": detected_implementation,
            "python_version": ".".join(map(str, detected_version)),
            "pointer_bits": detected_bits,
            "requested_device": requested_device,
        },
        "python_checks": python_checks,
        "packages": packages,
        "device_compatible": device_compatible,
        "compatible": compatible,
    }


def inspect_setup(
    *,
    dataset_root: str | Path,
    percentage: int,
    upstream_root: str | Path,
    upstream_revision: str,
    checkpoint_path: str | Path | None,
    checkpoint_sha256: str | None = MODEL_SHA256,
    model_repo_id: str = MODEL_REPO_ID,
    model_revision: str = MODEL_REVISION,
    local_files_only: bool = False,
    auto_provision: bool = False,
    provision_log: str | Path | None = None,
    cache_root: str | Path | None = None,
    cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
    verify_cache_artifacts: bool = True,
    load_model: bool = False,
    device: Any = "cpu",
) -> dict:
    """Collect a fail-closed report, optionally provisioning pinned assets."""

    dataset_path = Path(dataset_root).resolve(strict=False)
    manifest = load_manifest(dataset_path, int(percentage))
    rows = collect_required_rows(dataset_path, manifest, require_files=True)
    runtime = inspect_runtime_environment(device=device)
    runtime_compatible = bool(runtime["compatible"])
    assets = None
    if runtime_compatible and auto_provision:
        assets = provision_assets(
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            checkpoint_path=checkpoint_path,
            repo_id=model_repo_id,
            model_revision=model_revision,
            expected_sha256=checkpoint_sha256,
            expected_size_bytes=MODEL_SIZE_BYTES,
            local_files_only=local_files_only,
            log_path=Path(provision_log) if provision_log is not None else (
                methodology_root()
                / "Results"
                / "3DINO"
                / "Setup"
                / "asset_provisioning.jsonl"
            ),
        )
        upstream_root = assets.upstream_root
        checkpoint_path = assets.checkpoint_path
    if runtime_compatible and checkpoint_path is None:
        raise FileNotFoundError(
            "No local 3DINO checkpoint was supplied and automatic provisioning "
            "is disabled. Remove --no_auto_provision or provide --checkpoint_path."
        )
    upstream = (
        dict(assets.upstream)
        if assets is not None
        else validate_upstream_checkout(upstream_root, upstream_revision)
        if runtime_compatible
        else {
            "status": "skipped_runtime_incompatible",
            "reason": "Runtime policy must pass before importing the model adapter.",
        }
    )
    source_scm_attested = bool(
        isinstance(upstream, Mapping)
        and upstream.get("revision_claim_verified_from_scm") is True
    )
    checkpoint = None
    dependencies = {
        name: bool(details["compatible"])
        for name, details in runtime["packages"].items()
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "passed" if runtime_compatible else "failed_runtime_compatibility",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "runtime": runtime,
        "dependencies": dependencies,
        "dataset": {
            "root": str(dataset_path),
            "percentage": int(percentage),
            "patient_count": len(rows),
            "class_order": [
                "Appendicitis",
                "Cholezystitis",
                "Normal",
                "Perforation",
                "Sigmoid",
            ],
            "manifest_validated": True,
            "patient_partitions_disjoint": True,
            "all_nifti_sources_present": True,
        },
        "upstream": upstream,
        "source_scm_attested": source_scm_attested,
        "checkpoint": None,
        "asset_provisioning": assets.to_dict() if assets is not None else None,
        "auto_provision_requested": bool(auto_provision),
        "model_load_requested": bool(load_model),
        "downloads_performed": bool(assets and assets.downloads_performed),
        "git_operations_performed": bool(
            assets and assets.git_operations_performed
        ),
        "production_training_run": False,
        "secrets_logged": False,
        "licence_notice": (
            "Official 3DINO code and weights are CC BY-NC-ND 4.0; eligibility "
            "and any adaptation permission remain the user's responsibility."
        ),
    }

    cache_requested = cache_root is not None
    cache_valid = not cache_requested
    if cache_root is not None:
        cache_dir = cache_directory(cache_root, cache_namespace)
        cache_manifest = cache_dir / "cache_manifest.json"
        report["cache"] = {
            "namespace": cache_namespace,
            "manifest": str(cache_manifest),
            "exists": cache_manifest.is_file(),
            "status": "missing",
            "required_for_model_execution": True,
        }
        if cache_manifest.is_file():
            try:
                validation = validate_cache(
                    cache_manifest, verify_artifacts=verify_cache_artifacts
                )
            except Exception as exc:
                report["cache"].update(
                    {
                        "status": "invalid",
                        "validation_error_type": type(exc).__name__,
                        "validation_error": str(exc),
                    }
                )
            else:
                report["cache"]["validation"] = validation
                cache_valid = validation.get("artifacts_verified") is True
                report["cache"]["status"] = (
                    "valid"
                    if cache_valid
                    else "structurally_valid_artifacts_unverified"
                )

    missing_required = [
        name
        for name, details in runtime["packages"].items()
        if not details["importable"] or details.get("installed_version") is None
    ]
    incompatible_required = [
        name
        for name, details in runtime["packages"].items()
        if name not in missing_required and not details["compatible"]
    ]
    report["missing_required_dependencies"] = missing_required
    report["incompatible_required_dependencies"] = incompatible_required

    if load_model and runtime_compatible and source_scm_attested:
        model = load_official_model(
            upstream_root=upstream_root,
            upstream_revision=upstream_revision,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=checkpoint_sha256,
            device=device,
        )
        provenance = getattr(model, "_threedino_provenance", {})
        report["checkpoint"] = (
            dict(assets.checkpoint)
            if assets is not None
            else provenance.get("checkpoint")
        )
        report["model"] = {
            "loaded": True,
            "device": str(device),
            "checkpoint_coverage": provenance.get("load_report", {}).get("coverage"),
            "random_fallback_used": provenance.get("load_report", {}).get(
                "random_fallback_used"
            ),
        }
        if report["model"]["random_fallback_used"] is not False:
            report["status"] = "failed_random_fallback_guard"
    elif runtime_compatible:
        checkpoint = resolve_checkpoint(
            checkpoint_path, expected_sha256=checkpoint_sha256
        )
        report["checkpoint"] = (
            dict(assets.checkpoint)
            if assets is not None
            else checkpoint.to_dict()
        )
        if load_model and not source_scm_attested:
            report["model"] = {
                "loaded": False,
                "status": "skipped_source_scm_unattested",
                "device": str(device),
            }
    else:
        report["checkpoint"] = {
            "status": "skipped_runtime_incompatible",
            "path": (
                str(Path(checkpoint_path).resolve(strict=False))
                if checkpoint_path is not None
                else f"hf://{model_repo_id}@{model_revision}/3dino_vit_weights.pth"
            ),
        }
        if load_model:
            report["model"] = {
                "loaded": False,
                "status": "skipped_runtime_incompatible",
                "device": str(device),
            }
    # A missing cache is an expected state before cache preparation, whereas it
    # is a hard stop for model execution.  Keep these two readiness claims
    # separate so a preflight can never advertise a runnable model while the
    # requested persistent cache is absent or invalid.
    prerequisites_ready = report["status"] == "passed"
    report["ready_for_cache_build"] = prerequisites_ready
    report["ready_for_model_execution"] = (
        prerequisites_ready and cache_valid and source_scm_attested
    )
    if prerequisites_ready and cache_requested and not cache_valid:
        report["status"] = (
            "failed_cache_missing"
            if report["cache"]["status"] == "missing"
            else (
                "failed_cache_artifacts_unverified"
                if report["cache"]["status"]
                == "structurally_valid_artifacts_unverified"
                else "failed_cache_invalid"
            )
        )
    elif prerequisites_ready and not source_scm_attested:
        report["status"] = "failed_source_scm_unattested"
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_setup(
            dataset_root=args.dataset_root,
            percentage=args.percentage,
            upstream_root=args.upstream_root,
            upstream_revision=args.upstream_revision,
            checkpoint_path=args.checkpoint_path,
            checkpoint_sha256=args.checkpoint_sha256,
            model_repo_id=args.model_repo_id,
            model_revision=args.model_revision,
            local_files_only=args.local_files_only,
            auto_provision=not args.no_auto_provision,
            provision_log=args.provision_log,
            cache_root=args.cache_root,
            cache_namespace=args.cache_namespace,
            verify_cache_artifacts=not args.skip_artifact_validation,
            load_model=args.load_model,
            device=args.device,
        )
        _write_json(args.output, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "passed" else 2


def _write_json(path: Path, value: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
