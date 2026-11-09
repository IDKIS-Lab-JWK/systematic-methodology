"""Patient-safe NIfTI manifests and publication-faithful 3DINO transforms.

The deterministic prefix is deliberately separate from the post-cache
transform.  It is safe to persist the former once per source volume while the
training crop and augmentations remain stochastic on every access.
"""

from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Package import.
    from . import (
        CACHE_SCHEMA_VERSION,
        CLASS_NAMES,
        INPUT_SIZE,
        PREFIX_SIZE,
    )
except ImportError:  # pragma: no cover - direct invocation from this folder.
    from __init__ import CACHE_SCHEMA_VERSION, CLASS_NAMES, INPUT_SIZE, PREFIX_SIZE


NIFTI_SUFFIXES = (".nii", ".nii.gz")
_TRAIN_SPLITS = frozenset({"train", "training"})
_EVAL_SPLITS = frozenset(
    {"val", "valid", "validation", "test", "heldout", "heldout_test"}
)

# Internal DataLoader protocol used by the exact-resume training sampler. A
# request is a plain three-integer tuple so it remains trivially picklable with
# Windows' spawn multiprocessing context.
DRAW_INDEX_KEY = "_threedino_draw_index"
AUGMENTATION_SEED_KEY = "_threedino_augmentation_seed"
DRAW_REQUEST_FIELDS = ("sample_index", "draw_index", "augmentation_seed")
DRAW_SEED_PROTOCOL = "threedino-draw-seed-v2"
_UINT32_MODULUS = 1 << 32


def _index_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"3DINO draw-request {field} must be an integer, not bool.")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(
            f"3DINO draw-request {field} must be an integer, got {type(value).__name__}."
        ) from exc


def make_draw_request(
    sample_index: int, draw_index: int, augmentation_seed: int
) -> tuple[int, int, int]:
    """Return one validated, worker-independent stochastic draw request.

    The tuple layout is ``(sample_index, absolute_draw_index,
    augmentation_seed)``. The augmentation seed is constrained to the uint32
    domain accepted by MONAI's ``Randomizable`` transforms.
    """

    sample = _index_integer(sample_index, field="sample_index")
    draw = _index_integer(draw_index, field="draw_index")
    seed = _index_integer(augmentation_seed, field="augmentation_seed")
    if sample < 0:
        raise ValueError("3DINO draw-request sample_index must be non-negative.")
    if draw < 0:
        raise ValueError("3DINO draw-request draw_index must be non-negative.")
    if not 0 <= seed < _UINT32_MODULUS:
        raise ValueError(
            "3DINO draw-request augmentation_seed must be in the uint32 range."
        )
    return sample, draw, seed


def _foreground_above_minimum(value: Any) -> Any:
    """Picklable equivalent of the official ``lambda x: x > -1``."""

    return value > -1.0


def is_nifti_path(path: str | Path) -> bool:
    """Return whether *path* has an accepted NIfTI suffix."""

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def require_nifti_path(path: str | Path, *, must_exist: bool = True) -> Path:
    """Enforce the full-volume NIfTI-only 3DINO input contract."""

    candidate = Path(path)
    if not is_nifti_path(candidate):
        raise ValueError(
            "3DINO inputs must be full 3-D NIfTI volumes (.nii or .nii.gz), "
            f"not PNG/2.5-D/DICOM inputs: {candidate}"
        )
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"Missing NIfTI volume: {candidate}")
    return candidate


def nifti_path(dataset_root: Path, class_name: str, patient_id: str) -> Path:
    """Resolve one patient volume, preferring the repository's gzip form."""

    _require_class_name(class_name)
    patient = _normalise_patient_id(patient_id)
    base = Path(dataset_root) / "NIFTI" / class_name
    compressed = base / f"{patient}.nii.gz"
    if compressed.is_file():
        return compressed
    uncompressed = base / f"{patient}.nii"
    if uncompressed.is_file():
        return uncompressed
    return compressed


def split_manifest_path(dataset_root: Path, percentage: int) -> Path:
    percentage = int(percentage)
    if not 0 < percentage <= 100:
        raise ValueError(f"Percentage must be in [1, 100], got {percentage}")
    return Path(dataset_root) / "Splits" / f"{percentage}_percent.json"


def load_manifest(
    dataset_root: Path,
    percentage: int,
    *,
    expected_folds: int = 5,
) -> dict:
    """Load and fully validate an existing percentage split manifest."""

    return load_manifest_file(
        split_manifest_path(dataset_root, percentage),
        expected_percentage=int(percentage),
        expected_folds=expected_folds,
    )


def load_manifest_file(
    path: str | Path,
    *,
    expected_percentage: int | None = None,
    expected_folds: int = 5,
) -> dict:
    """Load either supported split layout and reject leakage or bad labels."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid split JSON in {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Split manifest must contain a JSON object: {manifest_path}")

    if expected_percentage is not None:
        recorded = manifest.get("metadata", {}).get("percentage")
        if recorded is not None and int(recorded) != int(expected_percentage):
            raise ValueError(
                f"Manifest percentage mismatch in {manifest_path}: expected "
                f"{expected_percentage}, found {recorded}"
            )
    validate_manifest(manifest, expected_folds=expected_folds)
    return manifest


def classes_from_manifest(manifest: Mapping) -> list[str]:
    """Validate labels and return the one canonical five-class order."""

    mappings: list[Mapping] = []
    heldout = manifest.get("heldout_test")
    if heldout is not None:
        mappings.append(_partition_mapping(heldout, "heldout_test"))
    cv = _cv_block(manifest)
    folds = cv.get("folds")
    if isinstance(folds, list) and folds:
        mappings.append(_partition_mapping(folds[0], "first CV fold"))
    if not mappings:
        raise ValueError("Manifest contains no class mappings")
    for mapping in mappings:
        _validate_class_keys(mapping)
    return list(CLASS_NAMES)


def fold_mapping(manifest: Mapping, fold_number: int) -> dict[str, list[str]]:
    """Return one fold's mapping in canonical class order."""

    requested = int(fold_number)
    for position, fold in enumerate(_folds(manifest), start=1):
        recorded = fold.get("fold", fold.get("fold_number", position))
        if int(recorded) == requested:
            return _normalise_mapping(
                _partition_mapping(fold, f"CV fold {requested}"),
                f"CV fold {requested}",
            )
    raise ValueError(f"Fold {requested} not found")


def merge_mappings(
    mappings: Iterable[Mapping[str, Iterable[str]]],
    classes: Sequence[str] = CLASS_NAMES,
) -> dict[str, list[str]]:
    """Merge mappings while preserving canonical class and source order."""

    if tuple(classes) != CLASS_NAMES:
        raise ValueError(
            f"3DINO class order must be {list(CLASS_NAMES)}, got {list(classes)}"
        )
    merged = {class_name: [] for class_name in CLASS_NAMES}
    for mapping in mappings:
        normalised = _normalise_mapping(mapping, "mapping")
        for class_name in CLASS_NAMES:
            merged[class_name].extend(normalised[class_name])
    return merged


def cross_validation_split(
    manifest: Mapping, validation_fold: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Use one fold for validation and the remaining folds for training."""

    fold_numbers = _fold_numbers(manifest)
    requested = int(validation_fold)
    if requested not in fold_numbers:
        raise ValueError(
            f"Validation fold {requested} not in available folds {fold_numbers}"
        )
    validation = fold_mapping(manifest, requested)
    training = merge_mappings(
        [fold_mapping(manifest, fold) for fold in fold_numbers if fold != requested]
    )
    return training, validation


def development_mapping(manifest: Mapping) -> dict[str, list[str]]:
    """Return all development patients across the five CV folds."""

    return merge_mappings(
        [fold_mapping(manifest, fold) for fold in _fold_numbers(manifest)]
    )


def heldout_mapping(manifest: Mapping) -> dict[str, list[str]]:
    """Return the untouched held-out mapping."""

    partition = manifest.get("heldout_test")
    if partition is None:
        raise ValueError("Manifest has no heldout_test partition")
    return _normalise_mapping(
        _partition_mapping(partition, "heldout_test"), "heldout_test"
    )


def all_patient_mapping(manifest: Mapping) -> dict[str, list[str]]:
    """Return development plus held-out patients for shared cache creation."""

    return merge_mappings([development_mapping(manifest), heldout_mapping(manifest)])


def rows_from_mapping(
    dataset_root: Path,
    mapping: Mapping[str, Iterable[str]],
    classes: Sequence[str] = CLASS_NAMES,
    *,
    require_files: bool = True,
) -> list[dict]:
    """Create model/cache rows from manifest patient identifiers."""

    if tuple(classes) != CLASS_NAMES:
        raise ValueError(
            f"3DINO class order must be {list(CLASS_NAMES)}, got {list(classes)}"
        )
    normalised = _normalise_mapping(mapping, "patient mapping")
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    rows: list[dict] = []
    for class_name in CLASS_NAMES:
        for patient_id in normalised[class_name]:
            source = nifti_path(Path(dataset_root), class_name, patient_id)
            require_nifti_path(source, must_exist=require_files)
            resolved = source.resolve(strict=False)
            row = {
                "image": str(resolved),
                "source_path": str(resolved),
                "label": class_to_index[class_name],
                "patient_id": patient_id,
                "label_str": class_name,
                "accession": f"{class_name}__{patient_id}",
            }
            if source.is_file():
                stat = source.stat()
                row["source_size_bytes"] = int(stat.st_size)
                row["source_mtime_ns"] = int(stat.st_mtime_ns)
            rows.append(row)
    return rows


def collect_required_rows(
    dataset_root: Path,
    manifest: Mapping,
    *,
    require_files: bool = True,
) -> list[dict]:
    """Collect all development and held-out rows for one isolated cache."""

    rows = rows_from_mapping(
        dataset_root,
        all_patient_mapping(manifest),
        require_files=require_files,
    )
    identifiers = [str(row["patient_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate patient IDs found while collecting cache inputs")
    return rows


def validate_no_overlap(*row_groups: Sequence[Mapping]) -> None:
    """Raise when any patient appears in more than one row partition."""

    seen: set[str] = set()
    for rows in row_groups:
        identifiers = {str(row["patient_id"]) for row in rows}
        if len(identifiers) != len(rows):
            raise RuntimeError("Duplicate patient IDs within a partition")
        overlap = seen.intersection(identifiers)
        if overlap:
            raise RuntimeError(
                f"Patient overlap across partitions: {sorted(overlap)[:5]}"
            )
        seen.update(identifiers)


def validate_manifest(manifest: Mapping, *, expected_folds: int = 5) -> None:
    """Validate classes, fold identifiers, counts, and patient isolation."""

    classes_from_manifest(manifest)
    folds = _folds(manifest)
    recorded_count = _cv_block(manifest).get("number_of_folds")
    if recorded_count is not None and int(recorded_count) != len(folds):
        raise ValueError(
            f"CV fold-count mismatch: metadata says {recorded_count}, found {len(folds)}"
        )
    if expected_folds is not None and len(folds) != int(expected_folds):
        raise ValueError(f"Expected {expected_folds} CV folds, found {len(folds)}")

    fold_numbers = _fold_numbers(manifest)
    if len(fold_numbers) != len(set(fold_numbers)):
        raise ValueError(f"Duplicate fold identifiers: {fold_numbers}")
    required_fold_ids = {1, 2, 3, 4, 5}
    if set(fold_numbers) != required_fold_ids:
        raise ValueError(
            "3DINO requires exact CV fold identifiers {1, 2, 3, 4, 5}; "
            f"found {sorted(fold_numbers)}"
        )

    groups: list[set[str]] = []
    for fold_number in fold_numbers:
        mapping = fold_mapping(manifest, fold_number)
        groups.append(_mapping_patient_set(mapping, f"CV fold {fold_number}"))
    heldout = heldout_mapping(manifest)
    groups.append(_mapping_patient_set(heldout, "heldout_test"))

    seen: set[str] = set()
    for group in groups:
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(
                "Patient leakage/duplication across folds or held-out test: "
                f"{sorted(overlap)[:5]}"
            )
        seen.update(group)
    _validate_recorded_counts(manifest, heldout)


def normalise_split_name(split: str) -> str:
    """Map accepted split aliases to ``train``, ``validation``, or ``test``."""

    value = str(split).strip().lower()
    if value in _TRAIN_SPLITS:
        return "train"
    if value in {"val", "valid", "validation"}:
        return "validation"
    if value in _EVAL_SPLITS:
        return "test"
    raise ValueError(
        f"Unknown split {split!r}; expected train, validation, or test"
    )


def preprocessing_contract() -> dict:
    """Return the exact deterministic and stochastic preprocessing contract."""

    return {
        "input": "single-channel NIfTI",
        "orientation": "RAS",
        "percentiles": [0.05, 99.95],
        "intensity_range": [-1.0, 1.0],
        "foreground_threshold": "> -1.0",
        "deterministic_resize_xyz": list(PREFIX_SIZE),
        "model_crop_xyz": list(INPUT_SIZE),
        "interpolation": "trilinear",
        "training": {
            "crop": "random_spatial_crop",
            "one_of": [
                "adjust_contrast",
                "gaussian_sharpen",
                "gaussian_smooth",
                "gaussian_noise",
            ],
            "axis_flip_probability": 0.5,
            "scale_intensity_factor": 0.1,
            "shift_intensity_offset": 0.1,
        },
        "validation": {"crop": "center_spatial_crop"},
        "test": {"crop": "center_spatial_crop"},
        "validation_test_crop": "center_spatial_crop",
        "source": "AICONSlab/3DINO dinov2/data/transforms.py COVID-CT-MD",
    }


def make_deterministic_transform() -> Any:
    """Build the cacheable official CT preprocessing prefix.

    MONAI is imported only when this factory is called so manifest-only tests
    and orchestration dry-runs do not require the imaging stack.
    """

    transforms = _import_monai_transforms()
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
            transforms.EnsureTyped(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.05,
                upper=99.95,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            ),
            transforms.CropForegroundd(
                keys=["image"],
                source_key="image",
                select_fn=_foreground_above_minimum,
            ),
            transforms.Resized(
                keys=["image"], spatial_size=PREFIX_SIZE, mode="trilinear"
            ),
        ]
    )


def make_post_cache_transform(split: str, *, seed: int | None = None) -> Any:
    """Build stochastic train or deterministic validation/test postprocessing."""

    transforms = _import_monai_transforms()
    canonical = normalise_split_name(split)
    if canonical == "train":
        transform = transforms.Compose(
            [
                transforms.RandSpatialCropd(
                    keys=["image"], roi_size=INPUT_SIZE, random_size=False
                ),
                transforms.OneOf(
                    transforms=[
                        transforms.RandAdjustContrastd(
                            keys=["image"], prob=0.3, gamma=(0.5, 2.0)
                        ),
                        transforms.RandGaussianSharpend(keys=["image"], prob=0.3),
                        transforms.RandGaussianSmoothd(keys=["image"], prob=0.3),
                        transforms.RandGaussianNoised(
                            keys=["image"], prob=0.3, std=0.002
                        ),
                    ]
                ),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                transforms.RandScaleIntensityd(
                    keys=["image"], factors=0.1, prob=1.0
                ),
                transforms.RandShiftIntensityd(
                    keys=["image"], offsets=0.1, prob=1.0
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.CenterSpatialCropd(
                    keys=["image"], roi_size=INPUT_SIZE
                )
            ]
        )
    if seed is not None and hasattr(transform, "set_random_state"):
        transform.set_random_state(seed=int(seed))
    return transform


def persistent_cache_key(row: Mapping) -> str:
    """Return the MONAI cache key for a deterministic-prefix source row."""

    source = str(row.get("source_path", row.get("image", "")))
    if not source:
        raise ValueError("Cache row needs image or source_path")
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_path": str(Path(source).resolve(strict=False)),
    }
    path = Path(source)
    if path.is_file():
        stat = path.stat()
        payload.update(
            {"source_size_bytes": int(stat.st_size), "source_mtime_ns": int(stat.st_mtime_ns)}
        )
    elif "source_size_bytes" in row or "source_mtime_ns" in row:
        payload.update(
            {
                "source_size_bytes": row.get("source_size_bytes"),
                "source_mtime_ns": row.get("source_mtime_ns"),
            }
        )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persistent_cache_hash(row: Mapping) -> bytes:
    """MONAI ``PersistentDataset`` hash callback."""

    return persistent_cache_key(row).encode("ascii")


class PostCacheTransformDataset:
    """Apply split-specific postprocessing to a deterministic base dataset.

    Ordinary integer indices retain the validation/test-compatible behavior.
    Training may instead supply a request created by :func:`make_draw_request`;
    its seed is applied immediately before the stochastic post-cache transform,
    making the result independent of the DataLoader worker that executes it.
    """

    _threedino_accepts_draw_request = True
    _threedino_draw_seed_protocol = DRAW_SEED_PROTOCOL

    def __init__(self, base_dataset: Any, post_transform: Any):
        self.base_dataset = base_dataset
        self.post_transform = post_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_random_state(self, seed: int | None = None, state: Any = None):
        """Delegate MONAI worker initialization to the post-cache transform."""

        setter = getattr(self.post_transform, "set_random_state", None)
        if not callable(setter):
            raise RuntimeError(
                "3DINO post-cache transform does not expose set_random_state."
            )
        setter(seed=seed, state=state)
        return self

    def __getitem__(self, index: int | tuple[int, int, int]) -> dict:
        draw_index: int | None = None
        augmentation_seed: int | None = None
        if isinstance(index, tuple):
            if len(index) != len(DRAW_REQUEST_FIELDS):
                raise TypeError(
                    "3DINO draw requests must be "
                    "(sample_index, draw_index, augmentation_seed) tuples."
                )
            sample_index, draw_index, augmentation_seed = make_draw_request(*index)
            if sample_index >= len(self.base_dataset):
                raise IndexError(
                    f"3DINO draw-request sample_index {sample_index} is outside "
                    f"dataset size {len(self.base_dataset)}."
                )
        else:
            sample_index = _index_integer(index, field="sample_index")

        base_value = self.base_dataset[sample_index]
        if not isinstance(base_value, Mapping):
            raise TypeError("3DINO post-cache datasets must return mapping samples.")
        transform_input = dict(base_value)

        if draw_index is not None:
            # Keep reseeding adjacent to the transform call: worker-init seeds,
            # scheduling, prefetch depth, and worker count cannot affect this draw.
            self.set_random_state(seed=augmentation_seed)
        transformed = self.post_transform(transform_input)
        if not isinstance(transformed, Mapping):
            raise TypeError("3DINO post-cache transforms must return mapping samples.")
        result = dict(transformed)
        result.pop(DRAW_INDEX_KEY, None)
        result.pop(AUGMENTATION_SEED_KEY, None)
        if draw_index is not None:
            result[DRAW_INDEX_KEY] = int(draw_index)
            result[AUGMENTATION_SEED_KEY] = int(augmentation_seed)
        return result


def make_prefix_dataset(
    rows: Sequence[Mapping], *, cache_dir: str | Path | None = None
) -> Any:
    """Create a dataset that stops exactly after the deterministic prefix."""

    Dataset, PersistentDataset = _import_monai_datasets()
    data = [dict(row) for row in rows]
    transform = make_deterministic_transform()
    if cache_dir is None:
        return Dataset(data=data, transform=transform)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return PersistentDataset(
        data=data,
        transform=transform,
        cache_dir=directory,
        hash_func=persistent_cache_hash,
    )


def make_dataset(
    rows: Sequence[Mapping],
    *,
    split: str,
    cache_dir: str | Path | None = None,
    seed: int | None = None,
) -> PostCacheTransformDataset:
    """Create a split dataset with a shared deterministic persistent cache."""

    base = make_prefix_dataset(rows, cache_dir=cache_dir)
    post = make_post_cache_transform(split, seed=seed)
    return PostCacheTransformDataset(base, post)


def _import_monai_transforms() -> Any:
    try:
        from monai import transforms
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "3DINO preprocessing requires MONAI. Install this model's "
            "requirements.txt in its isolated environment."
        ) from exc
    return transforms


def _import_monai_datasets() -> tuple[Any, Any]:
    try:
        from monai.data import Dataset, PersistentDataset
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "3DINO datasets require MONAI. Install this model's "
            "requirements.txt in its isolated environment."
        ) from exc
    return Dataset, PersistentDataset


def _cv_block(manifest: Mapping) -> Mapping:
    block = manifest.get("cross_validation") or manifest.get("nested_cv")
    if not isinstance(block, Mapping):
        raise ValueError("Manifest needs a cross_validation or nested_cv object")
    return block


def _folds(manifest: Mapping) -> list[Mapping]:
    folds = _cv_block(manifest).get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Cross-validation block needs a non-empty folds list")
    if not all(isinstance(fold, Mapping) for fold in folds):
        raise ValueError("Every CV fold must be a JSON object")
    return folds


def _fold_numbers(manifest: Mapping) -> list[int]:
    numbers: list[int] = []
    for position, fold in enumerate(_folds(manifest), start=1):
        if "fold" not in fold and "fold_number" not in fold:
            raise ValueError(f"CV fold at position {position} has no fold identifier")
        value = fold.get("fold", fold.get("fold_number"))
        try:
            numbers.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid fold identifier: {value!r}") from exc
    return numbers


def _partition_mapping(partition: Mapping, name: str) -> Mapping:
    mapping = partition.get("patients_by_class", partition)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} needs a patients_by_class mapping")
    return mapping


def _normalise_mapping(mapping: Mapping, name: str) -> dict[str, list[str]]:
    _validate_class_keys(mapping)
    result: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        values = mapping[class_name]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name}.{class_name} must be a patient list")
        patients = [_normalise_patient_id(value) for value in values]
        if len(patients) != len(set(patients)):
            raise ValueError(f"Duplicate patients within {name}.{class_name}")
        result[class_name] = patients
    return result


def _validate_class_keys(mapping: Mapping) -> None:
    expected = set(CLASS_NAMES)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            "Class mismatch: expected exactly "
            f"{list(CLASS_NAMES)}; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _mapping_patient_set(mapping: Mapping, name: str) -> set[str]:
    normalised = _normalise_mapping(mapping, name)
    ownership: dict[str, str] = {}
    for class_name in CLASS_NAMES:
        for patient_id in normalised[class_name]:
            previous = ownership.get(patient_id)
            if previous is not None:
                raise ValueError(
                    f"Patient {patient_id} occurs in both {previous} and "
                    f"{class_name} ({name})"
                )
            ownership[patient_id] = class_name
    return set(ownership)


def _validate_recorded_counts(manifest: Mapping, heldout: Mapping) -> None:
    heldout_block = manifest.get("heldout_test", {})
    recorded = heldout_block.get("counts_by_class")
    if recorded is not None:
        actual = {name: len(heldout[name]) for name in CLASS_NAMES}
        try:
            normalised_recorded = {
                name: int(recorded[name]) for name in CLASS_NAMES
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid held-out counts_by_class metadata") from exc
        if normalised_recorded != actual:
            raise ValueError(
                f"Held-out class-count mismatch: recorded={normalised_recorded}, "
                f"actual={actual}"
            )
    total = heldout_block.get("total")
    actual_total = sum(len(heldout[name]) for name in CLASS_NAMES)
    if total is not None and int(total) != actual_total:
        raise ValueError(
            f"Held-out total mismatch: recorded={total}, actual={actual_total}"
        )

    cv_block = _cv_block(manifest)
    development = development_mapping(manifest)
    development_actual = {
        name: len(development[name]) for name in CLASS_NAMES
    }
    development_recorded = cv_block.get("development_counts_by_class")
    if development_recorded is not None:
        if not isinstance(development_recorded, Mapping):
            raise ValueError("Invalid development_counts_by_class metadata")
        _validate_class_keys(development_recorded)
        try:
            development_normalised = {
                name: int(development_recorded[name]) for name in CLASS_NAMES
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid development_counts_by_class metadata") from exc
        if development_normalised != development_actual:
            raise ValueError(
                "Development class-count mismatch: "
                f"recorded={development_normalised}, actual={development_actual}"
            )

    development_total = cv_block.get("development_total")
    actual_development_total = sum(development_actual.values())
    if development_total is not None:
        try:
            normalised_total = int(development_total)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid development_total metadata") from exc
        if normalised_total != actual_development_total:
            raise ValueError(
                "Development total mismatch: "
                f"recorded={development_total}, actual={actual_development_total}"
            )


def _normalise_patient_id(value: object) -> str:
    patient_id = str(value).strip()
    if not patient_id:
        raise ValueError("Patient IDs cannot be empty")
    if patient_id in {".", ".."} or Path(patient_id).name != patient_id:
        raise ValueError(f"Unsafe patient ID: {patient_id!r}")
    return patient_id


def _require_class_name(class_name: str) -> None:
    if class_name not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class {class_name!r}; expected one of {list(CLASS_NAMES)}"
        )


__all__ = [
    "AUGMENTATION_SEED_KEY",
    "CLASS_NAMES",
    "DRAW_INDEX_KEY",
    "DRAW_REQUEST_FIELDS",
    "DRAW_SEED_PROTOCOL",
    "NIFTI_SUFFIXES",
    "PostCacheTransformDataset",
    "all_patient_mapping",
    "classes_from_manifest",
    "collect_required_rows",
    "cross_validation_split",
    "development_mapping",
    "fold_mapping",
    "heldout_mapping",
    "is_nifti_path",
    "load_manifest",
    "load_manifest_file",
    "make_dataset",
    "make_draw_request",
    "make_deterministic_transform",
    "make_post_cache_transform",
    "make_prefix_dataset",
    "merge_mappings",
    "nifti_path",
    "normalise_split_name",
    "persistent_cache_hash",
    "persistent_cache_key",
    "preprocessing_contract",
    "require_nifti_path",
    "rows_from_mapping",
    "split_manifest_path",
    "validate_manifest",
    "validate_no_overlap",
]
