"""Capability-gated exploratory zero-shot evaluation for Pillar-0.

Pillar-0 did not demonstrate zero-shot classification. This module implements
one strict experimental five-class positive-minus-negative prompt-margin
protocol. It requires the released checkpoint's trained 4096->1152 text
projection and learned logit scale at runtime; missing components fail closed.
Raw Qwen and vision embeddings are never compared directly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .cache import load_pillar_volume
    from .data import CLASS_NAMES
    from .metrics import compute_metrics, format_predictions
    from .model import (
        FEATURE_DIM,
        MODEL_REPO_ID,
        MODEL_REVISION,
        ZeroShotCapabilityError,
        checkpoint_logit_scale,
        extract_vision_features,
        load_official_model,
        project_qwen_embeddings,
        zero_shot_capability_report,
    )
    from .training import FeatureStore, save_json
except ImportError:  # pragma: no cover - script entry point
    from cache import load_pillar_volume
    from data import CLASS_NAMES
    from metrics import compute_metrics, format_predictions
    from model import (
        FEATURE_DIM,
        MODEL_REPO_ID,
        MODEL_REVISION,
        ZeroShotCapabilityError,
        checkpoint_logit_scale,
        extract_vision_features,
        load_official_model,
        project_qwen_embeddings,
        zero_shot_capability_report,
    )
    from training import FeatureStore, save_json


QWEN_REPO_ID = "Qwen/Qwen3-Embedding-8B"
QWEN_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
ZERO_SHOT_VARIANT = "exploratory_five_class_prompt_similarity"
PROMPT_PROFILE_SCHEMA_VERSION = 2
CLASS_SCORE_RULE = (
    "mean_positive_prompt_cosine_minus_mean_negative_prompt_cosine"
)
PROBABILITY_RULE = "softmax(checkpoint_logit_scale * class_score)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_profile(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("schema_version") != PROMPT_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Pillar zero-shot requires the canonical schema-v2 margin profile; "
            "legacy profiles are immutable provenance and cannot be rerun through "
            "the current scorer."
        )
    if profile.get("zero_shot_variant") != ZERO_SHOT_VARIANT:
        raise ValueError(
            f"zero_shot_variant must be {ZERO_SHOT_VARIANT!r}."
        )
    if profile.get("class_order") != list(CLASS_NAMES):
        raise ValueError(
            f"Prompt class order must exactly match {list(CLASS_NAMES)}."
        )
    if profile.get("publication_demonstrated") is not False:
        raise ValueError("Pillar zero-shot profile must state publication_demonstrated=false.")
    if profile.get("strict_zero_shot") is not True:
        raise ValueError("Pillar zero-shot profile must state strict_zero_shot=true.")
    scoring = profile.get("scoring", {})
    if scoring.get("class_score") != CLASS_SCORE_RULE:
        raise ValueError(f"scoring.class_score must be {CLASS_SCORE_RULE!r}.")
    if scoring.get("probability") != PROBABILITY_RULE:
        raise ValueError(f"scoring.probability must be {PROBABILITY_RULE!r}.")
    if scoring.get("prediction") != "argmax_class_score":
        raise ValueError("scoring.prediction must be 'argmax_class_score'.")
    if "author_binary_classes" in profile:
        raise ValueError(
            "author_binary_classes is a retired legacy field and is forbidden "
            "in the canonical five-class margin profile."
        )
    classes = profile.get("classes", {})
    for class_name in CLASS_NAMES:
        definition = classes.get(class_name)
        if not isinstance(definition, dict):
            raise ValueError(f"Missing prompt definition for {class_name}.")
        for polarity in ("positive", "negative"):
            prompts = definition.get(polarity)
            if not isinstance(prompts, list) or not prompts or not all(
                isinstance(value, str) and value.strip() for value in prompts
            ):
                raise ValueError(f"{class_name}.{polarity} must be a non-empty string list.")
    return profile


def flatten_prompts(profile: dict) -> tuple[list[str], list[tuple[str, str]]]:
    prompts, mapping = [], []
    for class_name in CLASS_NAMES:
        for polarity in ("positive", "negative"):
            for prompt in profile["classes"][class_name][polarity]:
                prompts.append(prompt)
                mapping.append((class_name, polarity))
    return prompts, mapping


def encode_qwen_findings(
    prompts: list[str],
    *,
    device: torch.device,
    repo_id: str = QWEN_REPO_ID,
    revision: str = QWEN_REVISION,
    max_length: int = 512,
    batch_size: int = 8,
) -> torch.Tensor:
    """Reproduce Pillar pretraining's attention-mask mean pooling."""

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    text_model = AutoModel.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    text_model.eval()
    pooled = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            output = text_model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu())
    del text_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = F.normalize(torch.cat(pooled), dim=-1)
    if result.shape != (len(prompts), 4096):
        raise RuntimeError(f"Expected Qwen [N,4096] embeddings, got {tuple(result.shape)}.")
    return result


def aggregate_prompt_embeddings(
    embeddings: torch.Tensor,
    mapping: list[tuple[str, str]],
) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"positive": [], "negative": []} for name in CLASS_NAMES
    }
    for embedding, (class_name, polarity) in zip(embeddings, mapping):
        grouped[class_name][polarity].append(embedding)
    return {
        class_name: {
            # The dot product with this un-normalized mean is exactly the mean
            # of the individual normalized-prompt cosine similarities.
            polarity: torch.stack(values).mean(dim=0)
            for polarity, values in polarities.items()
        }
        for class_name, polarities in grouped.items()
    }


def _materialize_image_embeddings(
    rows: list[dict],
    store: FeatureStore,
    model,
    device: torch.device,
) -> torch.Tensor:
    features = []
    model.to(device).eval()
    for row in rows:
        feature = store.load(row)
        if feature is None:
            raw = load_pillar_volume(Path(row["cache_path"])).unsqueeze(0).to(device)
            with torch.inference_mode():
                feature = extract_vision_features(model, raw).squeeze(0)
            store.save(row, feature)
        features.append(F.normalize(feature.float(), dim=0))
    return torch.stack(features)


def score_five_class_prompt_margins(
    image_embeddings: torch.Tensor,
    prompts: dict[str, dict[str, torch.Tensor]],
    logit_scale: float,
) -> dict[str, torch.Tensor]:
    """Score the single strict experimental five-class prompt protocol.

    Each prompt embedding is individually L2-normalized before aggregation.
    The polarity means intentionally remain unnormalized so their dot products
    are exact means of the constituent prompt cosine similarities.
    """

    if image_embeddings.ndim != 2 or image_embeddings.shape[0] == 0:
        raise ValueError(
            "image_embeddings must be a non-empty [patients, features] tensor."
        )
    if not torch.isfinite(image_embeddings).all():
        raise ValueError("image_embeddings contains non-finite values.")
    scale = float(logit_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint logit scale must be finite and positive.")

    try:
        positive = torch.stack(
            [prompts[name]["positive"] for name in CLASS_NAMES]
        )
        negative = torch.stack(
            [prompts[name]["negative"] for name in CLASS_NAMES]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Every class must provide positive and negative prompt means."
        ) from exc
    expected_shape = (len(CLASS_NAMES), image_embeddings.shape[1])
    if tuple(positive.shape) != expected_shape or tuple(negative.shape) != expected_shape:
        raise ValueError(
            "Prompt means must match [classes, features] "
            f"{expected_shape}; got positive={tuple(positive.shape)}, "
            f"negative={tuple(negative.shape)}."
        )
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("Prompt means contain non-finite values.")

    positive_similarity = image_embeddings @ positive.T
    negative_similarity = image_embeddings @ negative.T
    class_score = positive_similarity - negative_similarity
    scaled_logit = scale * class_score
    probability = torch.softmax(scaled_logit, dim=1)
    return {
        "positive_similarity": positive_similarity,
        "negative_similarity": negative_similarity,
        "class_score": class_score,
        "scaled_logit": scaled_logit,
        "probability": probability,
    }


def five_class_margin_documents(
    rows: list[dict],
    image_embeddings: torch.Tensor,
    prompts: dict[str, dict[str, torch.Tensor]],
    logit_scale: float,
) -> tuple[dict, list[dict], pd.DataFrame, pd.DataFrame]:
    if len(rows) != image_embeddings.shape[0]:
        raise ValueError(
            f"Row/embedding count mismatch: {len(rows)} rows versus "
            f"{image_embeddings.shape[0]} embeddings."
        )
    scores = score_five_class_prompt_margins(
        image_embeddings, prompts, logit_scale
    )
    probabilities = scores["probability"].cpu().numpy()
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predicted = scores["class_score"].argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(truth, predicted, probabilities, list(CLASS_NAMES))
    metrics.update(
        {
            "methodological_label": ZERO_SHOT_VARIANT,
            "publication_demonstrated": False,
            "zero_training_examples": True,
            "class_score_rule": CLASS_SCORE_RULE,
            "probability_rule": PROBABILITY_RULE,
            "probability_interpretation": "uncalibrated_relative_score",
        }
    )
    predictions, probability_frame = format_predictions(
        [str(row["patient_id"]) for row in rows],
        [str(row["label_str"]) for row in rows],
        probabilities,
        list(CLASS_NAMES),
    )
    score_frame = pd.DataFrame(
        {
            f"{class_name}_{field}": scores[field][:, class_index].cpu().numpy()
            for class_index, class_name in enumerate(CLASS_NAMES)
            for field in (
                "positive_similarity",
                "negative_similarity",
                "class_score",
                "scaled_logit",
            )
        }
    )
    score_frame["true_label"] = [str(row["label_str"]) for row in rows]
    score_frame["pred_label"] = [
        CLASS_NAMES[int(index)] for index in predicted
    ]
    score_frame["patient_id"] = [str(row["patient_id"]) for row in rows]
    for row_index, prediction in enumerate(predictions):
        prediction["prompt_similarity_scores"] = {
            class_name: {
                field: float(scores[field][row_index, class_index])
                for field in (
                    "positive_similarity",
                    "negative_similarity",
                    "class_score",
                    "scaled_logit",
                )
            }
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
        prediction["probability_interpretation"] = "uncalibrated_relative_score"
    return metrics, predictions, probability_frame, score_frame


def run_zero_shot(
    rows: list[dict],
    prompt_path: Path,
    raw_cache_manifest: Path,
    feature_cache_root: Path,
    results_root: Path,
    *,
    device: torch.device,
    repo_id: str = MODEL_REPO_ID,
    revision: str = MODEL_REVISION,
    force: bool = False,
) -> dict:
    profile = load_prompt_profile(prompt_path)
    results_root = Path(results_root)
    output_dir = (
        results_root
        / "ZeroShot"
        / "Test"
        / ZERO_SHOT_VARIANT
        / profile["prompt_set_id"]
    )
    existing = output_dir / "metrics_summary.json"
    if existing.is_file() and not force:
        raise FileExistsError(
            f"Zero-shot result already exists; refusing to overwrite: {existing}. "
            "Pass --force only for an intentional replacement."
        )
    setup_dir = results_root / "Setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    model = load_official_model(repo_id, revision, device="cpu")
    capability = zero_shot_capability_report(model)
    try:
        # A dummy probe proves the checkpoint owns a working trained projection
        # before the 8B text model is downloaded or loaded.
        project_qwen_embeddings(model, torch.zeros(1, 4096))
        capability["runtime_projection_probe"] = "passed"
        scale = checkpoint_logit_scale(model)
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ZeroShotCapabilityError(
                "unsupported_missing_logit_scale: the released checkpoint did "
                "not expose a finite positive trained logit scale."
            )
        capability["checkpoint_logit_scale_probe"] = "passed"
        capability["resolved_checkpoint_logit_scale"] = float(scale)
    except Exception as exc:
        projection_passed = capability.get("runtime_projection_probe") == "passed"
        capability.update(
            {
                "runtime_projection_probe": (
                    "passed" if projection_passed else "failed"
                ),
                "checkpoint_logit_scale_probe": (
                    "failed" if projection_passed else "not_run"
                ),
                "status": (
                    "unsupported_missing_logit_scale"
                    if projection_passed
                    else "unsupported_missing_text_projection"
                ),
                "error": str(exc),
                "checked_at_utc": _utc_now(),
            }
        )
        save_json(setup_dir / "zero_shot_capability.json", capability)
        raise
    prompts, mapping = flatten_prompts(profile)
    raw_text = encode_qwen_findings(prompts, device=device)
    with torch.inference_mode():
        aligned_text = project_qwen_embeddings(model, raw_text.to(next(model.parameters()).device)).cpu()
    prompt_embeddings = aggregate_prompt_embeddings(aligned_text, mapping)
    store = FeatureStore(feature_cache_root, raw_cache_manifest, repo_id, revision)
    image_embeddings = _materialize_image_embeddings(rows, store, model, device).cpu()
    capability.update(
        {
            "runtime_projection_probe": "passed",
            "checkpoint_logit_scale_probe": "passed",
            "status": "experimental_supported",
            "zero_shot_variant": ZERO_SHOT_VARIANT,
            "checked_at_utc": _utc_now(),
        }
    )
    save_json(setup_dir / "zero_shot_capability.json", capability)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "prompt_profile_snapshot.json", profile)
    metrics, predictions, probability_frame, score_frame = five_class_margin_documents(
        rows, image_embeddings, prompt_embeddings, float(scale)
    )
    save_json(output_dir / "metrics_summary.json", metrics)
    save_json(output_dir / "test_patient_predictions.json", predictions)
    probability_frame.to_csv(output_dir / "test_probabilities.csv", index=False)
    score_frame.to_csv(output_dir / "test_similarity_scores.csv", index=False)
    selection = {
        "schema_version": 2,
        "model_family": "Pillar-0",
        "model_repo_id": repo_id,
        "model_revision": revision,
        "method": ZERO_SHOT_VARIANT,
        "methodological_status": "experimental_not_publication_demonstrated",
        "publication_demonstrated": False,
        "zero_training_examples": True,
        "strict_zero_shot": True,
        "model_selection_performed": False,
        "heldout_test_used_during_prompt_selection": False,
        "development_labels_used_during_prompt_selection": False,
        "prompt_or_threshold_tuning_performed": False,
        "prompt_profile": str(Path(prompt_path).resolve()),
        "prompt_profile_sha256": _sha256_file(prompt_path),
        "prompt_profile_snapshot": str(
            (output_dir / "prompt_profile_snapshot.json").resolve()
        ),
        "prompt_set_id": profile["prompt_set_id"],
        "prompt_count": len(prompts),
        "class_score_rule": CLASS_SCORE_RULE,
        "prediction_rule": "argmax_class_score",
        "probability_rule": PROBABILITY_RULE,
        "probability_interpretation": "uncalibrated_relative_score",
        "qwen_repo_id": QWEN_REPO_ID,
        "qwen_revision": QWEN_REVISION,
        "qwen_pooling": "attention_mask_mean_pooling",
        "pillar_text_projection_runtime_verified": True,
        "checkpoint_logit_scale_runtime_verified": True,
        "resolved_checkpoint_logit_scale": float(scale),
        "raw_cache_manifest": str(Path(raw_cache_manifest).resolve()),
        "raw_cache_manifest_sha256": _sha256_file(raw_cache_manifest),
        "class_semantics_status": profile.get("class_semantics_status"),
        "created_at_utc": _utc_now(),
    }
    save_json(output_dir / "model_selection.json", selection)
    return {
        "status": "completed",
        "outputs": {ZERO_SHOT_VARIANT: str(output_dir)},
        "capability": capability,
    }
