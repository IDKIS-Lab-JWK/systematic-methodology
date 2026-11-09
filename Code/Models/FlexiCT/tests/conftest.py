from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from FlexiCT import CLASS_NAMES, DEFAULT_CACHE_NAMESPACE, UPSTREAM_REVISION


def write_split_dataset(root: Path, *, create_sources: bool) -> Path:
    folds = []
    for fold in range(1, 6):
        mapping = {
            class_name: [f"dev_f{fold}_{label}"]
            for label, class_name in enumerate(CLASS_NAMES)
        }
        folds.append({"fold": fold, "patients_by_class": mapping})
    heldout = {
        class_name: [f"test_{label}"]
        for label, class_name in enumerate(CLASS_NAMES)
    }
    manifest = {
        "heldout_test": {"patients_by_class": heldout},
        "cross_validation": {"folds": folds},
    }
    splits = root / "Splits"
    splits.mkdir(parents=True, exist_ok=True)
    for percentage in (20, 40, 60, 80, 100):
        (splits / f"{percentage}_percent.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    if create_sources:
        for mapping in [*(item["patients_by_class"] for item in folds), heldout]:
            for class_name, patient_ids in mapping.items():
                directory = root / "NIFTI" / class_name
                directory.mkdir(parents=True, exist_ok=True)
                for patient_id in patient_ids:
                    (directory / f"{patient_id}.nii.gz").write_bytes(
                        f"synthetic-{patient_id}".encode("utf-8")
                    )
    return root


@pytest.fixture
def split_dataset(tmp_path: Path) -> Path:
    return write_split_dataset(tmp_path / "Dataset", create_sources=False)


@pytest.fixture
def split_dataset_with_sources(tmp_path: Path) -> Path:
    return write_split_dataset(tmp_path / "Dataset", create_sources=True)


def cache_args(dataset_root: Path, tmp_path: Path, checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        percentage=100,
        dataset_root=dataset_root,
        results_root=tmp_path / "Results",
        cache_root=tmp_path / "Cache",
        cache_namespace=DEFAULT_CACHE_NAMESPACE,
        upstream_root=tmp_path / "upstream",
        upstream_revision=UPSTREAM_REVISION,
        checkpoint=checkpoint,
        checkpoint_sha256=None,
        allow_trusted_legacy_checkpoint=False,
        provision_log=tmp_path / "provision.jsonl",
        local_files_only=True,
        device="cpu",
        feature_precision="fp32",
        source_fingerprint="sha256",
        num_workers=0,
    )
