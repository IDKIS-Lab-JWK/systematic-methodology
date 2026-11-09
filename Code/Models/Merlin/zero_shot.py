"""Frozen Merlin image-text zero-shot inference and result export.

This module is deliberately independent from the supervised grid/CV/refit
orchestrator.  It uses the pretrained 512-dimensional contrastive image and
text projections exposed by ``Merlin()``, with no optimizer or fitted head.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from data import (
    classes_from_manifest,
    heldout_mapping,
    load_manifest,
    rows_from_mapping,
)
from metrics import compute_metrics, fold_selection_score, format_predictions
from model import import_merlin
from orchestrator import _configure_shared_cache
from training import make_loader, save_json, seed_everything, setup_logger


HELDOUT_MANIFEST_PERCENTAGE = 100
EXPECTED_CONTRASTIVE_DIMENSION = 512
PINNED_MERLIN_VERSION = "0.0.7"
AVAILABLE_MODES = ("five_class_adapted", "author_binary")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_profile(
    path: Path, manifest_classes: list[str]
) -> tuple[dict, str]:
    """Load and strictly validate the fixed, versioned prompt profile."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Zero-shot prompt configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)

    if int(profile.get("schema_version", -1)) != 1:
        raise ValueError("Zero-shot prompt schema_version must be 1.")
    prompt_set_id = str(profile.get("prompt_set_id", ""))
    if not SAFE_IDENTIFIER.fullmatch(prompt_set_id):
        raise ValueError(
            "prompt_set_id must contain only letters, numbers, '.', '-', or '_'."
        )

    class_order = profile.get("class_order")
    if class_order != manifest_classes:
        raise ValueError(
            "Prompt class_order must exactly match the held-out manifest order: "
            f"expected {manifest_classes}, received {class_order}."
        )
    class_definitions = profile.get("classes")
    if not isinstance(class_definitions, dict):
        raise ValueError("Prompt configuration must contain a classes object.")
    if list(class_definitions) != manifest_classes:
        raise ValueError(
            "Prompt class keys and order must exactly match class_order."
        )

    for class_name in manifest_classes:
        definition = class_definitions[class_name]
        if not isinstance(definition, dict):
            raise ValueError(f"Prompt definition for {class_name} must be an object.")
        concept = definition.get("canonical_concept")
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError(f"{class_name} requires a canonical_concept.")
        for polarity in ("positive", "negative"):
            prompts = definition.get(polarity)
            if not isinstance(prompts, list) or not prompts:
                raise ValueError(
                    f"{class_name}.{polarity} must be a non-empty prompt list."
                )
            if any(
                not isinstance(prompt, str)
                or not prompt.strip()
                or prompt != prompt.strip()
                for prompt in prompts
            ):
                raise ValueError(
                    f"{class_name}.{polarity} contains a blank or untrimmed prompt."
                )
            folded = [prompt.casefold() for prompt in prompts]
            if len(folded) != len(set(folded)):
                raise ValueError(
                    f"{class_name}.{polarity} contains duplicate prompts."
                )

    binary_classes = profile.get("author_binary_classes")
    if not isinstance(binary_classes, list) or not binary_classes:
        raise ValueError("author_binary_classes must be a non-empty list.")
    if len(binary_classes) != len(set(binary_classes)):
        raise ValueError("author_binary_classes contains duplicates.")
    unknown_binary = [
        class_name
        for class_name in binary_classes
        if class_name not in manifest_classes
    ]
    if unknown_binary:
        raise ValueError(
            f"Unknown author_binary_classes: {unknown_binary}."
        )
    if "Normal" in binary_classes:
        raise ValueError(
            "Normal is not an independent finding in author_binary mode."
        )

    return profile, hashlib.sha256(_canonical_json_bytes(profile)).hexdigest()


def validate_fixed_heldout_manifests(
    dataset_root: Path, canonical_manifest: dict
) -> list[int]:
    """Verify that every available percentage manifest uses the same test set."""
    canonical_classes = classes_from_manifest(canonical_manifest)
    canonical = {
        name: [str(value) for value in heldout_mapping(canonical_manifest)[name]]
        for name in canonical_classes
    }
    checked: list[int] = []
    for percentage in (20, 40, 60, 80, 100):
        path = dataset_root / "Splits" / f"{percentage}_percent.json"
        if not path.is_file():
            continue
        manifest = load_manifest(dataset_root, percentage)
        classes = classes_from_manifest(manifest)
        candidate = {
            name: [str(value) for value in heldout_mapping(manifest)[name]]
            for name in classes
        }
        if classes != canonical_classes or candidate != canonical:
            raise RuntimeError(
                "Held-out test membership differs across percentage manifests; "
                f"mismatch found in {path}. Zero-shot evaluation stopped."
            )
        checked.append(percentage)
    return checked


def flatten_prompts(profile: dict) -> tuple[list[str], dict[str, dict[str, list[int]]]]:
    """Flatten prompts while retaining exact class/polarity column indices."""
    prompts: list[str] = []
    layout: dict[str, dict[str, list[int]]] = {}
    for class_name in profile["class_order"]:
        layout[class_name] = {}
        for polarity in ("positive", "negative"):
            indices = []
            for prompt in profile["classes"][class_name][polarity]:
                indices.append(len(prompts))
                prompts.append(prompt)
            layout[class_name][polarity] = indices
    return prompts, layout


def as_embedding_matrix(
    value: torch.Tensor,
    expected_rows: int,
    description: str,
) -> torch.Tensor:
    """Normalize known Merlin singleton layouts to a strict [rows, dim] matrix."""
    if not torch.is_tensor(value):
        raise TypeError(f"{description} must be a tensor, got {type(value).__name__}.")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    elif value.ndim == 3 and value.shape[1] == 1:
        value = value.squeeze(1)
    if value.ndim != 2:
        raise RuntimeError(
            f"Expected {description} with shape [batch, dim], got {tuple(value.shape)}."
        )
    if value.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} {description} rows, got {value.shape[0]}."
        )
    if value.shape[1] != EXPECTED_CONTRASTIVE_DIMENSION:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTRASTIVE_DIMENSION}-D {description}, "
            f"got {value.shape[1]} dimensions."
        )
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{description} contains non-finite values.")
    return value.float()


def normalize_embeddings(value: torch.Tensor, description: str) -> torch.Tensor:
    norms = value.norm(dim=-1)
    if torch.any(norms <= 0):
        raise RuntimeError(f"{description} contains a zero-norm embedding.")
    normalized = F.normalize(value.float(), dim=-1)
    if not torch.isfinite(normalized).all():
        raise RuntimeError(f"Normalized {description} contains non-finite values.")
    return normalized


def mean_prompt_scores(
    similarities: torch.Tensor,
    layout: dict[str, dict[str, list[int]]],
    classes: list[str],
    polarity: str,
) -> torch.Tensor:
    """Mean per-prompt cosine similarity, matching the publication procedure."""
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [patients, prompts].")
    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity must be positive or negative.")
    columns = []
    for class_name in classes:
        indices = layout[class_name][polarity]
        if not indices:
            raise ValueError(f"No {polarity} prompts configured for {class_name}.")
        columns.append(similarities[:, indices].mean(dim=1))
    return torch.stack(columns, dim=1)


def uncalibrated_softmax(scores: np.ndarray) -> np.ndarray:
    """Create derived confidences without changing cosine-score argmax."""
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    probabilities = values / values.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output.astype(np.float32)


def _batch_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def validate_merlin_package_version() -> str:
    installed = _package_version("merlin-vlm")
    if installed != PINNED_MERLIN_VERSION:
        raise RuntimeError(
            "Zero-shot inference depends on the pinned Merlin internal encoder "
            f"API. Expected merlin-vlm=={PINNED_MERLIN_VERSION}, found "
            f"{installed or 'no installed distribution metadata'}. Install "
            "Code\\Models\\Merlin\\requirements.txt in the active environment."
        )
    return installed


def _checkpoint_metadata(model, calculate_hash: bool) -> dict:
    config = getattr(model, "_config", {}) or {}
    filename = config.get("checkpoint")
    path = None
    if filename:
        try:
            module_path = Path(inspect.getfile(type(model))).resolve()
            candidate = module_path.parent / "checkpoints" / str(filename)
            if candidate.is_file():
                path = candidate
        except (OSError, TypeError):
            path = None
    return {
        "filename": filename,
        "path": str(path) if path else None,
        "size_bytes": path.stat().st_size if path else None,
        "sha256": sha256_file(path) if path and calculate_hash else None,
        "sha256_status": (
            "computed"
            if path and calculate_hash
            else "skipped_by_user"
            if path
            else "checkpoint_file_not_resolved"
        ),
    }


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _encode_text_prompts(model, prompts: list[str]) -> torch.Tensor:
    architecture = getattr(model, "model", None)
    encoder = getattr(architecture, "encode_text", None)
    if encoder is None or not callable(encoder):
        raise RuntimeError("Pinned Merlin model does not expose model.encode_text.")
    with torch.no_grad():
        embeddings = encoder(prompts)
    embeddings = as_embedding_matrix(
        embeddings, len(prompts), "Merlin text embeddings"
    )
    return normalize_embeddings(embeddings, "Merlin text embeddings").cpu()


def _encode_heldout_images(
    model,
    loader,
    row_count: int,
    text_embeddings: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, np.ndarray, list[str], list[str]]:
    architecture = getattr(model, "model", None)
    image_encoder = getattr(architecture, "encode_image", None)
    if image_encoder is None or not callable(image_encoder):
        raise RuntimeError("Pinned Merlin model does not expose model.encode_image.")
    # Only the image branch moves to the inference device. The text branch has
    # already run on CPU, so do not call the complete split-device wrapper from
    # this point onward.
    image_encoder = image_encoder.to(device).eval()
    text_on_device = text_embeddings.to(device=device, dtype=torch.float32)

    similarity_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    patient_ids: list[str] = []
    label_strings: list[str] = []
    for batch in loader:
        batch_patient_ids = _batch_strings(batch["patient_id"])
        images = batch["image"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        with torch.no_grad(), _autocast_context(device, amp):
            encoded = image_encoder(images)
        image_embeddings = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
        image_embeddings = as_embedding_matrix(
            image_embeddings,
            len(batch_patient_ids),
            "Merlin image embeddings",
        )
        image_embeddings = normalize_embeddings(
            image_embeddings, "Merlin image embeddings"
        )
        similarities = image_embeddings @ text_on_device.T
        if not torch.isfinite(similarities).all():
            raise RuntimeError("Image-text cosine similarities contain non-finite values.")
        similarity_batches.append(similarities.detach().cpu())
        label_batches.append(torch.as_tensor(batch["label"]).reshape(-1).cpu())
        patient_ids.extend(batch_patient_ids)
        label_strings.extend(_batch_strings(batch["label_str"]))

    if not similarity_batches:
        raise RuntimeError("Held-out loader produced no batches.")
    similarities = torch.cat(similarity_batches, dim=0).float()
    labels = torch.cat(label_batches).long().numpy()
    if similarities.shape[0] != row_count or labels.shape[0] != row_count:
        raise RuntimeError(
            "Held-out inference count mismatch: "
            f"expected {row_count}, got {similarities.shape[0]}."
        )
    if len(patient_ids) != row_count or len(label_strings) != row_count:
        raise RuntimeError("Held-out patient metadata count mismatch.")
    return similarities, labels, patient_ids, label_strings


def _binary_condition_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    average_precision = (
        float(average_precision_score(truth, score))
        if np.unique(truth).size == 2
        else None
    )
    return {
        "accuracy": float((tp + tn) / len(truth)),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1),
        "average_precision": average_precision,
        "support_positive": int(np.sum(truth == 1)),
        "support_negative": int(np.sum(truth == 0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "class_order": ["absent", "present"],
    }


def _mean_available(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _base_selection(
    mode: str,
    output_dir: Path,
    args,
    prompt_profile: dict,
    prompt_hash: str,
    prompt_file_hash: str,
    checkpoint: dict,
    shared_cache_dir: Path,
    device: torch.device,
    amp_enabled: bool,
) -> dict:
    derived_confidence = (
        "softmax_of_raw_class_cosines_temperature_1"
        if mode == "five_class_adapted"
        else "sigmoid_of_positive_minus_negative_cosine_margin"
    )
    parameters = {
        "training_percentage": 0,
        "manifest_percentage_for_fixed_test": HELDOUT_MANIFEST_PERCENTAGE,
        "prompt_set_id": prompt_profile["prompt_set_id"],
        "canonical_prompt_sha256": prompt_hash,
        "prompt_file_sha256": prompt_file_hash,
        "merlin_vlm_version": PINNED_MERLIN_VERSION,
        "checkpoint_filename": checkpoint.get("filename"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "checkpoint_size_bytes": checkpoint.get("size_bytes"),
        "prompt_aggregation": "mean_per_prompt_cosine_similarity",
        "similarity": "cosine_after_l2_normalization",
        "score_scale": "raw_cosine",
        "derived_confidence": derived_confidence,
        "embedding_dim": EXPECTED_CONTRASTIVE_DIMENSION,
        "image_inference_precision": (
            "cuda_amp_fp16" if amp_enabled else "fp32"
        ),
        "similarity_precision": "fp32",
        "batch_size": int(args.batch_size),
        "gradient_accumulation": None,
    }
    return {
        "cv_id": "Pretrained",
        "variant": "zero_shot",
        "test_results_directory": str(output_dir),
        "model_source": mode,
        "checkpoint_count": 1,
        "checkpoints": [checkpoint],
        "parameter_origin": "pretrained_zero_shot",
        "parameter_source": "fixed_prompt_profile",
        "parameter_provenance": {
            "parameter_origin": "pretrained_zero_shot",
            "parameter_source": "fixed_prompt_profile",
            "profile_name": prompt_profile["prompt_set_id"],
            "parameters": parameters,
        },
        "parameters": parameters,
        "shared_cache_dir": str(shared_cache_dir),
        "heldout_test_used_during_selection": False,
        "model_selection_performed": False,
        "development_labels_used": False,
        "training_performed": False,
        "runtime": {
            "device": str(device),
            "amp_enabled": bool(amp_enabled),
        },
    }


def _five_class_documents(
    similarities: torch.Tensor,
    layout: dict[str, dict[str, list[int]]],
    classes: list[str],
    labels: np.ndarray,
    patient_ids: list[str],
    label_strings: list[str],
    selection: dict,
    prompt_profile: dict,
    prompt_hash: str,
    prompt_file_hash: str,
) -> tuple[dict, list[dict], pd.DataFrame, pd.DataFrame]:
    score_tensor = mean_prompt_scores(similarities, layout, classes, "positive")
    scores = score_tensor.numpy()
    probabilities = uncalibrated_softmax(scores)
    predicted = scores.argmax(axis=1)
    # Use the same confidence matrix for AP that is exported in
    # test_probabilities.csv, matching the trained-run artifact contract.
    metrics = compute_metrics(labels, predicted, probabilities, classes)
    raw_score_metrics = compute_metrics(labels, predicted, scores, classes)
    for class_name in classes:
        metrics["per_class"][class_name]["raw_cosine_average_precision"] = (
            raw_score_metrics["per_class"][class_name]["average_precision"]
        )
    comparison_score = fold_selection_score(metrics)
    metrics.update(
        {
            "comparison_score": comparison_score,
            "average_precision_score_input": (
                "exported_uncalibrated_softmax_confidence"
            ),
            "raw_cosine_average_precision_macro": raw_score_metrics[
                "average_precision_macro"
            ],
            "per_class_raw_cosine_average_precision": raw_score_metrics[
                "per_class_average_precision"
            ],
            "prediction_rule": "argmax_mean_positive_prompt_cosine_similarity",
            "confidence_semantics": (
                "Uncalibrated softmax of raw class cosine scores; not a "
                "calibrated probability."
            ),
        }
    )
    prediction_rows, probability_frame = format_predictions(
        patient_ids, label_strings, probabilities, classes
    )
    for index, row in enumerate(prediction_rows):
        row["scores"] = {
            class_name: float(scores[index, class_index])
            for class_index, class_name in enumerate(classes)
        }
        row["confidence_semantics"] = "uncalibrated_softmax_of_raw_cosines"

    score_frame = pd.DataFrame(
        scores,
        columns=[f"cosine_{class_name}" for class_name in classes],
    )
    score_frame["true_label"] = label_strings
    score_frame["pred_label"] = [classes[int(index)] for index in predicted]
    score_frame["patient_id"] = patient_ids
    summary = {
        "model_name": "merlin_zero_shot",
        "model_family": "Merlin",
        "variant": "zero_shot",
        "percentage": 0,
        "training_percentage": 0,
        "cv_id": "Pretrained",
        "parameter_origin": "pretrained_zero_shot",
        "evaluation_task": "mutually_exclusive_five_class",
        "prompt_strategy": "five_class_adapted",
        "prompt_set_id": prompt_profile["prompt_set_id"],
        "canonical_prompt_sha256": prompt_hash,
        "prompt_file_sha256": prompt_file_hash,
        "publication_exact_reproduction": False,
        "class_semantics_status": prompt_profile.get(
            "class_semantics_status"
        ),
        "class_assumptions": prompt_profile.get("class_assumptions"),
        "benchmark_interpretation": (
            "exploratory_until_dataset_class_semantics_are_confirmed"
        ),
        "methodological_label": (
            "publication-method-based task-specific zero-shot classification"
        ),
        "model_selection": selection,
        "test_metrics": metrics,
    }
    return summary, prediction_rows, probability_frame, score_frame


def _author_binary_documents(
    similarities: torch.Tensor,
    layout: dict[str, dict[str, list[int]]],
    classes: list[str],
    labels: np.ndarray,
    patient_ids: list[str],
    label_strings: list[str],
    selection: dict,
    prompt_profile: dict,
    prompt_hash: str,
    prompt_file_hash: str,
) -> tuple[dict, list[dict], pd.DataFrame]:
    binary_classes = list(prompt_profile["author_binary_classes"])
    positive = mean_prompt_scores(similarities, layout, binary_classes, "positive").numpy()
    negative = mean_prompt_scores(similarities, layout, binary_classes, "negative").numpy()
    margins = positive - negative
    confidences = _sigmoid(margins)
    predicted = margins > 0
    label_to_index = {name: index for index, name in enumerate(classes)}

    per_class = {}
    for index, class_name in enumerate(binary_classes):
        truth = (labels == label_to_index[class_name]).astype(np.int64)
        per_class[class_name] = _binary_condition_metrics(
            truth,
            predicted[:, index].astype(np.int64),
            margins[:, index],
        )
    macro_metrics = {
        f"{metric_name}_macro": _mean_available(
            [values[metric_name] for values in per_class.values()]
        )
        for metric_name in (
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "average_precision",
        )
    }
    metrics = {
        **macro_metrics,
        "per_class": per_class,
        "class_order": binary_classes,
        "score_input": "positive_mean_cosine_minus_negative_mean_cosine",
        "prediction_rule": "present_when_positive_mean_cosine_exceeds_negative_mean_cosine",
        "confidence_semantics": (
            "Uncalibrated sigmoid of the positive-minus-negative cosine margin."
        ),
        "normal_handling": (
            "Normal is not forced from the independent findings; this mode can "
            "predict zero or multiple findings per CT."
        ),
        "ground_truth_caveat": (
            "One-vs-rest evaluation treats every other mutually exclusive "
            "dataset label as finding absent. Interpret these binary metrics "
            "only if the labels certify absence rather than only the selected "
            "primary diagnosis; findings can coexist clinically."
        ),
        "ground_truth_type": (
            "one_vs_rest_proxy_derived_from_mutually_exclusive_primary_labels"
        ),
    }

    prediction_rows = []
    table_rows = []
    for patient_index, patient_id in enumerate(patient_ids):
        findings = {}
        table_row: dict[str, Any] = {
            "patient_id": patient_id,
            "true_label": label_strings[patient_index],
        }
        for class_index, class_name in enumerate(binary_classes):
            finding = {
                "predicted_present": bool(predicted[patient_index, class_index]),
                "positive_mean_cosine": float(positive[patient_index, class_index]),
                "negative_mean_cosine": float(negative[patient_index, class_index]),
                "cosine_margin": float(margins[patient_index, class_index]),
                "presence_confidence": float(confidences[patient_index, class_index]),
            }
            findings[class_name] = finding
            prefix = class_name
            table_row[f"{prefix}_predicted_present"] = finding["predicted_present"]
            table_row[f"{prefix}_positive_mean_cosine"] = finding[
                "positive_mean_cosine"
            ]
            table_row[f"{prefix}_negative_mean_cosine"] = finding[
                "negative_mean_cosine"
            ]
            table_row[f"{prefix}_cosine_margin"] = finding["cosine_margin"]
            table_row[f"{prefix}_presence_confidence"] = finding[
                "presence_confidence"
            ]
        prediction_rows.append(
            {
                "patient_id": patient_id,
                "true_label": label_strings[patient_index],
                "findings": findings,
            }
        )
        table_rows.append(table_row)

    summary = {
        "model_name": "merlin_zero_shot",
        "model_family": "Merlin",
        "variant": "zero_shot",
        "percentage": 0,
        "training_percentage": 0,
        "cv_id": "Pretrained",
        "parameter_origin": "pretrained_zero_shot",
        "evaluation_task": (
            "one_vs_rest_binary_proxy_from_mutually_exclusive_labels"
        ),
        "prompt_strategy": "author_binary",
        "prompt_set_id": prompt_profile["prompt_set_id"],
        "canonical_prompt_sha256": prompt_hash,
        "prompt_file_sha256": prompt_file_hash,
        "publication_exact_reproduction": False,
        "class_semantics_status": prompt_profile.get(
            "class_semantics_status"
        ),
        "class_assumptions": prompt_profile.get("class_assumptions"),
        "benchmark_interpretation": (
            "exploratory_until_dataset_class_semantics_are_confirmed"
        ),
        "methodological_label": (
            "publication-scoring-inspired task-specific binary zero-shot "
            "analysis with one-vs-rest proxy ground truth"
        ),
        "requires_independent_finding_annotations_for_publication_task_match": True,
        "model_selection": selection,
        "test_metrics": metrics,
    }
    return summary, prediction_rows, pd.DataFrame(table_rows)


def _prompt_provenance(
    prompt_path: Path,
    prompt_profile: dict,
    prompt_hash: str,
    prompt_file_hash: str,
    prompt_layout: dict,
    checkpoint: dict,
    manifest_path: Path,
    manifest_hash: str,
    checked_percentages: list[int],
    row_count: int,
    model,
    device: torch.device,
    amp_enabled: bool,
) -> dict:
    architecture = model.model
    raw_logit_scale = float(architecture.logit_scale.detach().float().cpu())
    effective_logit_scale = float(
        architecture.logit_scale.detach().float().exp().cpu()
    )
    return {
        "created_at_utc": _utc_now(),
        "prompt_config_path": str(prompt_path.resolve()),
        "canonical_prompt_sha256": prompt_hash,
        "prompt_file_sha256": prompt_file_hash,
        "prompt_profile": prompt_profile,
        "prompt_layout": prompt_layout,
        "prompt_selection_used_development_labels": False,
        "prompt_selection_used_heldout_labels": False,
        "publication_exact_prompts": False,
        "model": {
            "package": "merlin-vlm",
            "package_version": _package_version("merlin-vlm"),
            "wrapper_task": getattr(model, "task", None),
            "checkpoint": checkpoint,
            "image_embedding_dimension": EXPECTED_CONTRASTIVE_DIMENSION,
            "text_embedding_dimension": EXPECTED_CONTRASTIVE_DIMENSION,
            "text_encoder": "yikuan8/Clinical-Longformer",
            "raw_learned_logit_scale_recorded_but_not_used": raw_logit_scale,
            "exp_learned_logit_scale_recorded_but_not_used": effective_logit_scale,
            "score_scale_used": "raw_cosine",
            "parameters_frozen": True,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "amp_requested_and_cuda_enabled": bool(amp_enabled),
        },
        "dataset": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "manifest_percentage_for_fixed_test": HELDOUT_MANIFEST_PERCENTAGE,
            "verified_identical_heldout_manifest_percentages": checked_percentages,
            "heldout_patient_count": row_count,
            "training_percentage": 0,
        },
        "scoring": {
            "embedding_normalization": "L2 per image and per prompt",
            "prompt_aggregation": "mean of individual prompt cosine similarities",
            "five_class_rule": "argmax of mean positive-prompt cosine similarity",
            "author_binary_rule": (
                "positive finding when mean positive-prompt cosine similarity "
                "exceeds mean negative-prompt cosine similarity"
            ),
            "confidence_values_are_calibrated_probabilities": False,
        },
    }


def _mode_output_directory(results_root: Path, mode: str, prompt_set_id: str) -> Path:
    return results_root / "ZeroShot" / "Test" / mode / prompt_set_id


def _requested_modes(mode: str) -> list[str]:
    return list(AVAILABLE_MODES) if mode == "both" else [mode]


def run_zero_shot(args) -> dict:
    """Validate, run, and export the requested frozen zero-shot evaluation."""
    if int(args.batch_size) < 1:
        raise ValueError("--batch_size must be at least 1.")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.dry_run and args.validate_setup:
        raise ValueError("--dry_run and --validate_setup are mutually exclusive.")

    dataset_root = Path(args.dataset_root).resolve()
    results_root = Path(args.results_root).resolve()
    prompt_path = Path(args.prompt_config).resolve()
    manifest_path = (
        dataset_root
        / "Splits"
        / f"{HELDOUT_MANIFEST_PERCENTAGE}_percent.json"
    )
    manifest = load_manifest(dataset_root, HELDOUT_MANIFEST_PERCENTAGE)
    classes = classes_from_manifest(manifest)
    checked_percentages = validate_fixed_heldout_manifests(dataset_root, manifest)
    prompt_profile, prompt_hash = load_prompt_profile(prompt_path, classes)
    prompt_file_hash = sha256_file(prompt_path)
    prompts, prompt_layout = flatten_prompts(prompt_profile)
    rows = rows_from_mapping(
        dataset_root, heldout_mapping(manifest), classes
    )
    modes = _requested_modes(args.mode)
    output_directories = {
        mode: _mode_output_directory(
            results_root, mode, prompt_profile["prompt_set_id"]
        )
        for mode in modes
    }
    cache_dir, cache_temp_dir = _configure_shared_cache(
        args, create=not args.dry_run
    )

    plan = {
        "status": "dry_run" if args.dry_run else "ready",
        "model": "Merlin() joint image-text checkpoint",
        "training_performed": False,
        "training_percentage": 0,
        "manifest": str(manifest_path),
        "manifest_percentage_for_fixed_test": HELDOUT_MANIFEST_PERCENTAGE,
        "verified_identical_heldout_manifest_percentages": checked_percentages,
        "heldout_patient_count": len(rows),
        "classes": classes,
        "prompt_config": str(prompt_path),
        "prompt_set_id": prompt_profile["prompt_set_id"],
        "canonical_prompt_sha256": prompt_hash,
        "prompt_file_sha256": prompt_file_hash,
        "prompt_count": len(prompts),
        "modes": modes,
        "output_directories": {
            mode: str(path) for mode, path in output_directories.items()
        },
        "shared_cache_dir": str(cache_dir),
        "shared_cache_temp_dir": str(cache_temp_dir),
    }
    if args.dry_run:
        return plan

    for mode, output_dir in output_directories.items():
        marker = output_dir / "metrics_summary.json"
        if marker.exists() and not args.force and not args.validate_setup:
            raise FileExistsError(
                f"Zero-shot results already exist for {mode}: {marker}. "
                "Use --force to replace the exported result files."
            )

    log_dir = results_root / "ZeroShot" / "Logs"
    logger = setup_logger(log_dir)
    seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    logger.info("Loading frozen Merlin joint image-text model on CPU.")
    installed_merlin_version = validate_merlin_package_version()
    logger.info("Verified merlin-vlm==%s.", installed_merlin_version)
    Merlin, _ = import_merlin()
    model = Merlin()
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Merlin zero-shot model unexpectedly has trainable parameters.")

    logger.info("Encoding %d fixed prompts in FP32.", len(prompts))
    text_embeddings = _encode_text_prompts(model, prompts)
    if not args.skip_checkpoint_hash:
        logger.info(
            "Computing the downloaded checkpoint SHA-256; this can take time. "
            "Use --skip_checkpoint_hash to omit it."
        )
    checkpoint = _checkpoint_metadata(model, not args.skip_checkpoint_hash)
    loader_rows = rows[:1] if args.validate_setup else rows
    loader = make_loader(
        loader_rows,
        cache_dir,
        int(args.batch_size),
        False,
        int(args.num_workers),
    )
    logger.info(
        "Encoding %d held-out CT volume(s) on %s (AMP=%s).",
        len(loader_rows),
        device,
        amp_enabled,
    )
    similarities, labels, patient_ids, label_strings = _encode_heldout_images(
        model,
        loader,
        len(loader_rows),
        text_embeddings,
        device,
        amp_enabled,
    )
    expected_patient_ids = [str(row["patient_id"]) for row in loader_rows]
    expected_label_strings = [str(row["label_str"]) for row in loader_rows]
    if patient_ids != expected_patient_ids or label_strings != expected_label_strings:
        raise RuntimeError(
            "Held-out loader order differs from the manifest row order despite "
            "shuffle=False."
        )
    if args.validate_setup:
        return {
            **plan,
            "status": "setup_validated",
            "validated_patient_id": patient_ids[0],
            "image_embedding_dimension": EXPECTED_CONTRASTIVE_DIMENSION,
            "text_embedding_shape": list(text_embeddings.shape),
            "similarity_shape": list(similarities.shape),
            "checkpoint": checkpoint,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "results_written": False,
        }

    manifest_hash = sha256_file(manifest_path)
    provenance = _prompt_provenance(
        prompt_path,
        prompt_profile,
        prompt_hash,
        prompt_file_hash,
        prompt_layout,
        checkpoint,
        manifest_path,
        manifest_hash,
        checked_percentages,
        len(rows),
        model,
        device,
        amp_enabled,
    )
    written = {}
    for mode in modes:
        output_dir = output_directories[mode]
        selection = _base_selection(
            mode,
            output_dir,
            args,
            prompt_profile,
            prompt_hash,
            prompt_file_hash,
            checkpoint,
            cache_dir,
            device,
            amp_enabled,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "prompt_provenance.json", provenance)
        save_json(output_dir / "model_selection.json", selection)
        if mode == "five_class_adapted":
            summary, predictions, probabilities, scores = _five_class_documents(
                similarities,
                prompt_layout,
                classes,
                labels,
                patient_ids,
                label_strings,
                selection,
                prompt_profile,
                prompt_hash,
                prompt_file_hash,
            )
            save_json(output_dir / "test_patient_predictions.json", predictions)
            probabilities.to_csv(
                output_dir / "test_probabilities.csv", index=False
            )
            scores.to_csv(output_dir / "test_similarity_scores.csv", index=False)
        elif mode == "author_binary":
            summary, predictions, scores = _author_binary_documents(
                similarities,
                prompt_layout,
                classes,
                labels,
                patient_ids,
                label_strings,
                selection,
                prompt_profile,
                prompt_hash,
                prompt_file_hash,
            )
            save_json(
                output_dir / "test_binary_patient_predictions.json", predictions
            )
            scores.to_csv(output_dir / "test_binary_scores.csv", index=False)
        else:  # guarded by argparse, retained for programmatic callers
            raise ValueError(f"Unknown zero-shot mode: {mode}")
        save_json(output_dir / "metrics_summary.json", summary)
        written[mode] = {
            "output_directory": str(output_dir),
            "metrics_summary": str(output_dir / "metrics_summary.json"),
            "test_metrics": summary["test_metrics"],
        }
        logger.info("Wrote %s zero-shot results to %s.", mode, output_dir)

    return {
        **plan,
        "status": "completed",
        "device": str(device),
        "amp_enabled": amp_enabled,
        "checkpoint": checkpoint,
        "results": written,
    }
