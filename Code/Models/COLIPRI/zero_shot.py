"""Publication-backed COLIPRI zero-shot held-out evaluation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import (
        CLASS_NAMES,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
    )
    from .model import preprocess_nifti, resolve_precision
    from .provisioning import ensure_upstream, resolve_checkpoint
except ImportError:  # pragma: no cover
    from __init__ import (
        CLASS_NAMES,
        MODEL_FILENAME,
        MODEL_REPO_ID,
        MODEL_REVISION,
        MODEL_SHA256,
    )
    from model import preprocess_nifti, resolve_precision
    from provisioning import ensure_upstream, resolve_checkpoint

from _linear_probe.common import (
    ContractError,
    atomic_write_csv,
    atomic_write_json,
    completion_seal,
    fingerprint,
    heldout_mapping,
    load_manifest,
    read_json,
    rows_fingerprint,
    rows_from_mapping,
    sha256_file,
    utc_now,
    validate_completion,
)
from _linear_probe.metrics import compute_metrics, prediction_documents
from _supervised_adaptation.profiles import strip_note_fields
from _supervised_adaptation.runtime import atomic_torch_save, autocast_context, require_torch, torch_load
from _supervised_adaptation.safe_io import same_or_write_csv, same_or_write_json
from _supervised_adaptation.trainer import input_rows_identity


ZERO_SHOT_SCHEMA = "colipri-zero-shot-v1"


def _implementation_identity() -> dict:
    package = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        package / "run_zero_shot.py",
        package / "model.py",
        package / "provisioning.py",
        package.parent / "_supervised_adaptation" / "profiles.py",
        package.parent / "_supervised_adaptation" / "runtime.py",
        package.parent / "_supervised_adaptation" / "safe_io.py",
    ]
    records = [
        {
            "relative_path": path.relative_to(package.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {"files": records, "sha256": fingerprint(records)}


def load_prompts(path: str | Path) -> tuple[dict, dict]:
    raw = read_json(path)
    profile = strip_note_fields(raw)
    if profile.get("schema_version") != 1:
        raise ContractError("COLIPRI zero-shot prompt schema_version must be 1")
    if profile.get("class_order") != list(CLASS_NAMES):
        raise ContractError("COLIPRI zero-shot class order differs from the dataset contract")
    classes = profile.get("classes")
    if not isinstance(classes, Mapping) or list(classes) != list(CLASS_NAMES):
        raise ContractError("COLIPRI zero-shot prompt classes are missing or reordered")
    for class_name in CLASS_NAMES:
        definition = classes[class_name]
        if not isinstance(definition, Mapping):
            raise ContractError(f"Prompt definition must be an object: {class_name}")
        for polarity in ("positive", "negative"):
            value = definition.get(polarity)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ContractError(f"Invalid {class_name}.{polarity} prompt")
    if float(profile.get("temperature", 0.0)) <= 0.0:
        raise ContractError("COLIPRI zero-shot temperature must be positive")
    return profile, raw


def _activate_source(root: Path) -> Any:
    source = str((root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("colipri")
    if root.resolve() not in Path(module.__file__).resolve().parents:
        raise ContractError(f"Imported colipri from an unexpected path: {module.__file__}")
    return module


def _validate_heldout_membership(dataset_root: Path, canonical: Mapping[str, Any]) -> list[int]:
    expected = heldout_mapping(canonical)
    checked = []
    for percentage in (20, 40, 60, 80, 100):
        path = dataset_root / "Splits" / f"{percentage}_percent.json"
        if not path.is_file():
            continue
        candidate = load_manifest(dataset_root, percentage)
        if heldout_mapping(candidate) != expected:
            raise ContractError(f"Held-out membership differs in {path}")
        checked.append(percentage)
    return checked


def _model_and_processor(args: Any, checkpoint: Mapping[str, Any]):
    torch = require_torch()
    colipri = _activate_source(Path(args.upstream_root))
    model = colipri.get_model(
        checkpoint_path=Path(checkpoint["path"]), pretrained=True, image_only=False
    ).to(str(args.device))
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("COLIPRI zero-shot model was not fully frozen")
    processor = colipri.get_processor(image_only=False)
    return model, processor


def _text_bank(model: Any, processor: Any, profile: Mapping[str, Any], device: str, dtype: Any | None):
    torch = require_torch()
    prompts = []
    layout = []
    for class_name in CLASS_NAMES:
        negative = profile["classes"][class_name]["negative"]
        positive = profile["classes"][class_name]["positive"]
        layout.append({"class_name": class_name, "negative_index": len(prompts), "positive_index": len(prompts) + 1})
        prompts.extend([negative, positive])
    token_ids, attention_mask = processor.process_text(prompts)
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode(), autocast_context(device, dtype):
        embeddings = model.encode_text(
            token_ids, attention_mask=attention_mask, pool=True, normalize=True
        )
    embeddings = embeddings.float()
    if tuple(embeddings.shape) != (len(prompts), 768) or not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError(f"Unexpected COLIPRI text-embedding shape: {tuple(embeddings.shape)}")
    return prompts, layout, embeddings


def _score_image(
    model: Any,
    processor: Any,
    path: str,
    text_embeddings: Any,
    temperature: float,
    device: str,
    dtype: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    image, _metadata = preprocess_nifti(processor, path)
    images = image.to(device)
    with torch.inference_mode(), autocast_context(device, dtype):
        image_embedding = model.encode_image(
            images, pool=True, project=True, normalize=True
        )
    image_embedding = image_embedding.float()
    if tuple(image_embedding.shape) != (1, 768):
        raise RuntimeError(f"Unexpected COLIPRI image embedding shape: {tuple(image_embedding.shape)}")
    similarities = (image_embedding @ text_embeddings.T)[0]
    pairs = similarities.reshape(len(CLASS_NAMES), 2)
    pairwise = torch.softmax(pairs / float(temperature), dim=1)[:, 1]
    five_class = pairwise / pairwise.sum()
    return (
        five_class.cpu().numpy().astype(np.float64),
        pairwise.cpu().numpy().astype(np.float64),
        pairs.cpu().numpy().astype(np.float64),
    )


def run_zero_shot(args: Any) -> dict:
    torch = require_torch()
    prompt_profile, raw_prompts = load_prompts(args.prompt_config)
    manifest = load_manifest(args.dataset_root, 100)
    checked_manifests = _validate_heldout_membership(Path(args.dataset_root), manifest)
    rows = rows_from_mapping(
        args.dataset_root, heldout_mapping(manifest), require_files=not args.dry_run
    )
    prompt_set_id = str(prompt_profile["prompt_set_id"])
    root = Path(args.results_root) / "ZeroShot" / "Test" / prompt_set_id
    planned = {
        "schema_version": ZERO_SHOT_SCHEMA,
        "model": "COLIPRI",
        "method": "ZeroShot",
        "parameter_source": "publication",
        "prompt_set_id": prompt_set_id,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_filename": MODEL_FILENAME,
        "prompt_profile": str(Path(args.prompt_config).resolve()),
        "prompt_profile_sha256": sha256_file(args.prompt_config),
        "temperature": float(prompt_profile["temperature"]),
        "heldout_manifest": manifest["_path"],
        "heldout_manifest_sha256": manifest["_sha256"],
        "heldout_rows_sha256": rows_fingerprint(rows),
        "heldout_count": len(rows),
        "checked_percentage_manifests": checked_manifests,
        "upstream_revision": str(args.upstream_revision),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "pinned Hub resolution",
        "checkpoint_sha256": MODEL_SHA256,
        "implementation": _implementation_identity(),
        "scoring": prompt_profile["scoring"],
    }
    if args.dry_run:
        return {
            "status": "dry_run",
            "side_effects": False,
            "output": str(root.resolve()),
            "context": planned,
        }
    source = ensure_upstream(
        args.upstream_root,
        revision=str(args.upstream_revision),
        local_files_only=bool(args.local_files_only),
        provision_log=args.provision_log,
    )
    checkpoint = resolve_checkpoint(
        args.checkpoint,
        upstream_root=args.upstream_root,
        local_files_only=bool(args.local_files_only),
        provision_log=args.provision_log,
    )
    precision_name, dtype = resolve_precision(str(args.device), str(args.precision))
    context = {
        **planned,
        "source": source,
        "checkpoint_identity": checkpoint,
        "precision_requested": str(args.precision),
        "precision_resolved": precision_name,
        "heldout_input_identity": input_rows_identity(rows),
    }
    if (root / "test_completion.json").is_file():
        if not args.resume:
            raise ContractError(
                f"Completed COLIPRI zero-shot output exists and resume is disabled: {root}"
            )
        seal = validate_completion(root, "test_completion.json")
        if read_json(root / "zero_shot_context.json") != context:
            raise ContractError(f"Completed COLIPRI zero-shot context changed: {root}")
        expected_completion_context = {
            "prompt_set_id": prompt_set_id,
            "context_sha256": fingerprint(context),
            "heldout_rows_sha256": rows_fingerprint(rows),
        }
        if seal.get("context") != expected_completion_context:
            raise ContractError(
                f"Completed COLIPRI zero-shot seal context changed: {root}"
            )
        return {"status": "reused", "output": str(root.resolve())}
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ContractError(f"Partial COLIPRI zero-shot output exists and resume is disabled: {root}")
    if root.exists():
        allowed_partial = {
            "zero_shot_context.json",
            "prompt_profile.json",
            "prompt_bank.json",
            "progress.pt",
            "metrics_summary.json",
            "test_patient_predictions.json",
            "test_probabilities.csv",
            "model_selection.json",
        }
        unknown = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and "Logs" not in path.relative_to(root).parts
            and path.relative_to(root).as_posix() not in allowed_partial
        ]
        if unknown:
            raise ContractError(
                f"Unknown unsealed COLIPRI zero-shot artifact; refusing resume: {unknown[0]}"
            )
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / "zero_shot_context.json"
    if context_path.is_file() and read_json(context_path) != context:
        raise ContractError(f"Partial COLIPRI zero-shot context changed: {root}")
    same_or_write_json(context_path, context)
    same_or_write_json(root / "prompt_profile.json", raw_prompts)

    model, processor = _model_and_processor(args, checkpoint)
    prompts, layout, text_embeddings = _text_bank(
        model, processor, prompt_profile, str(args.device), dtype
    )
    observed_temperature = float(model.softmax_temperature.detach().float().item())
    expected_temperature = float(prompt_profile["temperature"])
    if abs(observed_temperature - expected_temperature) > 1e-6:
        raise ContractError(
            f"Checkpoint softmax temperature changed: expected {expected_temperature}, got {observed_temperature}"
        )
    same_or_write_json(
        root / "prompt_bank.json",
        {
            "schema_version": 1,
            "prompts": prompts,
            "layout": layout,
            "embedding_shape": list(text_embeddings.shape),
            "checkpoint_temperature": observed_temperature,
        },
    )
    progress_path = root / "progress.pt"
    progress_identity = {
        "context_sha256": fingerprint(context),
        "prompt_bank_sha256": sha256_file(root / "prompt_bank.json"),
        "row_count": len(rows),
    }
    if progress_path.is_file():
        payload = torch_load(progress_path, map_location="cpu")
        if (
            payload.get("schema_version") != ZERO_SHOT_SCHEMA
            or payload.get("identity") != progress_identity
        ):
            raise ContractError(f"Incompatible COLIPRI zero-shot progress: {progress_path}")
        next_index = int(payload["next_index"])
        five_class_rows = [item.numpy() for item in payload["five_class_probabilities"]]
        pairwise_rows = [item.numpy() for item in payload["pairwise_positive_probabilities"]]
        similarity_rows = [item.numpy() for item in payload["paired_similarities"]]
    else:
        next_index = 0
        five_class_rows, pairwise_rows, similarity_rows = [], [], []
    lengths = (len(five_class_rows), len(pairwise_rows), len(similarity_rows))
    if not 0 <= next_index <= len(rows) or any(length != next_index for length in lengths):
        raise ContractError("Malformed COLIPRI zero-shot progress index")
    for five_class, pairwise, similarities in zip(
        five_class_rows, pairwise_rows, similarity_rows
    ):
        if (
            tuple(five_class.shape) != (len(CLASS_NAMES),)
            or tuple(pairwise.shape) != (len(CLASS_NAMES),)
            or tuple(similarities.shape) != (len(CLASS_NAMES), 2)
            or not np.isfinite(five_class).all()
            or not np.isfinite(pairwise).all()
            or not np.isfinite(similarities).all()
        ):
            raise ContractError("Malformed COLIPRI zero-shot progress payload")
    for index in range(next_index, len(rows)):
        five_class, pairwise, similarities = _score_image(
            model,
            processor,
            rows[index]["image_path"],
            text_embeddings,
            expected_temperature,
            str(args.device),
            dtype,
        )
        five_class_rows.append(five_class)
        pairwise_rows.append(pairwise)
        similarity_rows.append(similarities)
        atomic_torch_save(
            {
                "schema_version": ZERO_SHOT_SCHEMA,
                "identity": progress_identity,
                "next_index": index + 1,
                "five_class_probabilities": [torch.as_tensor(item) for item in five_class_rows],
                "pairwise_positive_probabilities": [torch.as_tensor(item) for item in pairwise_rows],
                "paired_similarities": [torch.as_tensor(item) for item in similarity_rows],
                "updated_at": utc_now(),
            },
            progress_path,
        )
    probabilities = np.stack(five_class_rows, axis=0)
    pairwise_probabilities = np.stack(pairwise_rows, axis=0)
    similarities = np.stack(similarity_rows, axis=0)
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    metrics = compute_metrics(
        truth, probabilities.argmax(axis=1), probabilities, CLASS_NAMES
    )
    json_rows, csv_rows = prediction_documents(rows, probabilities, CLASS_NAMES)
    for index, document in enumerate(json_rows):
        document["pairwise_positive_probabilities"] = {
            class_name: float(pairwise_probabilities[index, class_index])
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
        document["paired_cosine_similarities"] = {
            class_name: {
                "negative": float(similarities[index, class_index, 0]),
                "positive": float(similarities[index, class_index, 1]),
            }
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
    same_or_write_json(root / "metrics_summary.json", metrics)
    same_or_write_json(root / "test_patient_predictions.json", json_rows)
    same_or_write_csv(root / "test_probabilities.csv", csv_rows, list(csv_rows[0]))
    same_or_write_json(
        root / "model_selection.json",
        {
            "schema_version": 1,
            "model_source": "pinned_pretrained_checkpoint_zero_shot",
            "checkpoint": checkpoint,
            "source": source,
            "prompt_set_id": prompt_set_id,
            "prompt_profile_sha256": sha256_file(args.prompt_config),
            "temperature": expected_temperature,
            "parameter_source": "publication",
        },
    )
    atomic_write_json(
        root / "test_completion.json",
        completion_seal(
            [
                root / "zero_shot_context.json",
                root / "prompt_profile.json",
                root / "prompt_bank.json",
                root / "progress.pt",
                root / "metrics_summary.json",
                root / "test_patient_predictions.json",
                root / "test_probabilities.csv",
                root / "model_selection.json",
            ],
            context={
                "prompt_set_id": prompt_set_id,
                "context_sha256": fingerprint(context),
                "heldout_rows_sha256": rows_fingerprint(rows),
            },
        ),
    )
    return {"status": "complete", "output": str(root.resolve()), "heldout_count": len(rows)}


__all__ = ["ZERO_SHOT_SCHEMA", "load_prompts", "run_zero_shot"]
