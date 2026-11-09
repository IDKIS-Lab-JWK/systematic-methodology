"""Publication-method-backed FlexiCT zero-shot held-out evaluation."""

from __future__ import annotations

import gc
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import CLASS_NAMES, FEATURE_DIM, UPSTREAM_REVISION
    from .data import preprocess_nifti, preprocessing_contract
    from .model import extract_embedding, resolve_precision
    from .provisioning import ensure_upstream, resolve_checkpoint
    from .zero_shot_model import (
        OFFICIAL_CHECKPOINT_SHA256,
        TEXT_MODEL_REPO_ID,
        TEXT_MODEL_REVISION,
        build_image_encoder,
        build_text_encoder,
        resolve_text_snapshot,
    )
except ImportError:  # pragma: no cover
    from __init__ import CLASS_NAMES, FEATURE_DIM, UPSTREAM_REVISION
    from data import preprocess_nifti, preprocessing_contract
    from model import extract_embedding, resolve_precision
    from provisioning import ensure_upstream, resolve_checkpoint
    from zero_shot_model import (
        OFFICIAL_CHECKPOINT_SHA256,
        TEXT_MODEL_REPO_ID,
        TEXT_MODEL_REVISION,
        build_image_encoder,
        build_text_encoder,
        resolve_text_snapshot,
    )

from _linear_probe.common import (
    ContractError,
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
from _supervised_adaptation.runtime import (
    atomic_torch_save,
    autocast_context,
    require_torch,
    torch_load,
)
from _supervised_adaptation.safe_io import same_or_write_csv, same_or_write_json
from _supervised_adaptation.trainer import input_rows_identity


ZERO_SHOT_SCHEMA = "flexict-zero-shot-v1"


def _implementation_identity() -> dict:
    package = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        package / "run_zero_shot.py",
        package / "zero_shot_model.py",
        package / "data.py",
        package / "model.py",
        package / "provisioning.py",
        package.parent / "_supervised_adaptation" / "profiles.py",
        package.parent / "_supervised_adaptation" / "runtime.py",
        package.parent / "_supervised_adaptation" / "safe_io.py",
    ]
    records = [
        {
            "relative_path": path.relative_to(package.parent).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {"files": records, "sha256": fingerprint(records)}


def load_prompts(path: str | Path) -> tuple[dict, dict]:
    raw = read_json(path)
    profile = strip_note_fields(raw)
    if profile.get("schema_version") != 1:
        raise ContractError("FlexiCT zero-shot prompt schema_version must be 1")
    if profile.get("parameter_source") != "publication":
        raise ContractError("FlexiCT zero-shot parameter_source must remain publication")
    if profile.get("class_order") != list(CLASS_NAMES):
        raise ContractError("FlexiCT zero-shot class order differs from the dataset contract")
    text_model = profile.get("text_model")
    if not isinstance(text_model, Mapping):
        raise ContractError("FlexiCT zero-shot text_model definition is missing")
    if (
        text_model.get("repo_id") != TEXT_MODEL_REPO_ID
        or text_model.get("revision") != TEXT_MODEL_REVISION
        or text_model.get("pooling") != "last_non_padding_token"
    ):
        raise ContractError("FlexiCT zero-shot text-model identity differs from the pinned contract")
    classes = profile.get("classes")
    if not isinstance(classes, Mapping) or list(classes) != list(CLASS_NAMES):
        raise ContractError("FlexiCT zero-shot prompt classes are missing or reordered")
    for class_name in CLASS_NAMES:
        definition = classes[class_name]
        if not isinstance(definition, Mapping):
            raise ContractError(f"Prompt definition must be an object: {class_name}")
        for polarity in ("positive", "negative"):
            value = definition.get(polarity)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ContractError(f"Invalid {class_name}.{polarity} prompt")
    scoring = profile.get("scoring")
    if not isinstance(scoring, Mapping) or scoring.get("logit_scale") != "checkpoint_learned":
        raise ContractError("FlexiCT zero-shot must use the checkpoint-learned logit scale")
    return profile, raw


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


def _runtime_identity(device: str, precision_requested: str, precision_resolved: str) -> dict:
    torch = require_torch()
    identity = {
        "python": __import__("sys").version,
        "torch_version": str(torch.__version__),
        "transformers_version": importlib.metadata.version("transformers"),
        "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
        "device_requested": str(device),
        "precision_requested": str(precision_requested),
        "precision_resolved": str(precision_resolved),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": (
            int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        ),
    }
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        identity.update(
            {
                "device_resolved": f"cuda:{index}",
                "device_name": torch.cuda.get_device_name(index),
                "device_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    else:
        identity["device_resolved"] = "cpu"
    return identity


def _text_bank(
    model: Any,
    tokenizer: Any,
    profile: Mapping[str, Any],
    device: str,
    dtype: Any | None,
) -> tuple[list[str], list[dict], Any]:
    torch = require_torch()
    prompts: list[str] = []
    layout: list[dict] = []
    for class_name in CLASS_NAMES:
        positive = profile["classes"][class_name]["positive"]
        negative = profile["classes"][class_name]["negative"]
        layout.append(
            {
                "class_name": class_name,
                "positive_index": len(prompts),
                "negative_index": len(prompts) + 1,
            }
        )
        prompts.extend([positive, negative])
    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    tokens = {key: value.to(str(device)) for key, value in tokens.items()}
    with torch.inference_mode(), autocast_context(str(device), dtype):
        embeddings = model(tokens)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    embeddings = embeddings.cpu().contiguous()
    if tuple(embeddings.shape) != (len(prompts), FEATURE_DIM):
        raise RuntimeError(f"Unexpected FlexiCT text-embedding shape: {tuple(embeddings.shape)}")
    if not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("FlexiCT text embeddings contain non-finite values")
    return prompts, layout, embeddings


def pairwise_five_class_probabilities(
    paired_similarities: np.ndarray, logit_multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the released pairwise softmax, then the declared five-class adaptation."""

    values = np.asarray(paired_similarities, dtype=np.float64)
    if values.shape != (len(CLASS_NAMES), 2) or not np.isfinite(values).all():
        raise ContractError("FlexiCT paired similarities must be a finite [5,2] matrix")
    if not np.isfinite(logit_multiplier) or float(logit_multiplier) <= 0.0:
        raise ContractError("FlexiCT logit multiplier must be finite and positive")
    logits = values * float(logit_multiplier)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    pairwise_positive = exp_logits[:, 0] / exp_logits.sum(axis=1)
    denominator = float(pairwise_positive.sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ContractError("FlexiCT pairwise probabilities cannot be normalized")
    return pairwise_positive / denominator, pairwise_positive


def _score_image(
    image_model: Any,
    path: str,
    text_embeddings: Any,
    logit_multiplier: float,
    device: str,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, _metadata = preprocess_nifti(path)
    image_embedding, _details = extract_embedding(
        image_model,
        image,
        device=device,
        precision=precision,
    )
    similarities = (
        image_embedding.float().cpu().numpy().astype(np.float64)
        @ text_embeddings.float().cpu().numpy().astype(np.float64).T
    ).reshape(len(CLASS_NAMES), 2)
    five_class, pairwise = pairwise_five_class_probabilities(
        similarities, logit_multiplier
    )
    return five_class, pairwise, similarities


def run_zero_shot(args: Any) -> dict:
    prompt_profile, raw_prompts = load_prompts(args.prompt_config)
    manifest = load_manifest(args.dataset_root, 100)
    checked_manifests = _validate_heldout_membership(Path(args.dataset_root), manifest)
    rows = rows_from_mapping(
        args.dataset_root,
        heldout_mapping(manifest),
        require_files=not args.dry_run,
    )
    prompt_set_id = str(prompt_profile["prompt_set_id"])
    root = Path(args.results_root) / "ZeroShot" / "Test" / prompt_set_id
    planned = {
        "schema_version": ZERO_SHOT_SCHEMA,
        "model": "FlexiCT",
        "method": "ZeroShot",
        "parameter_source": "publication",
        "publication_exact": False,
        "prompt_set_id": prompt_set_id,
        "prompt_profile": str(Path(args.prompt_config).resolve()),
        "prompt_profile_sha256": sha256_file(args.prompt_config),
        "heldout_manifest": manifest["_path"],
        "heldout_manifest_sha256": manifest["_sha256"],
        "heldout_rows_sha256": rows_fingerprint(rows),
        "heldout_count": len(rows),
        "checked_percentage_manifests": checked_manifests,
        "upstream_revision": str(args.upstream_revision),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": str(args.checkpoint_sha256),
        "text_model_repo_id": TEXT_MODEL_REPO_ID,
        "text_model_revision": str(args.text_model_revision),
        "preprocessing": preprocessing_contract(),
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

    torch = require_torch()
    source = ensure_upstream(
        args.upstream_root,
        revision=str(args.upstream_revision),
        local_files_only=bool(args.local_files_only),
        provision_log=args.provision_log,
    )
    checkpoint = resolve_checkpoint(
        args.checkpoint,
        expected_sha256=str(args.checkpoint_sha256),
    )
    text_snapshot = resolve_text_snapshot(
        cache_dir=args.hf_cache_root,
        local_files_only=bool(args.local_files_only),
        provision_log=args.provision_log,
    )
    precision_name, dtype = resolve_precision(str(args.device), str(args.precision))
    context = {
        **planned,
        "source": source,
        "checkpoint_identity": checkpoint,
        "text_snapshot_identity": text_snapshot,
        "runtime": _runtime_identity(str(args.device), str(args.precision), precision_name),
        "heldout_input_identity": input_rows_identity(rows),
    }
    if (root / "test_completion.json").is_file():
        if not args.resume:
            raise ContractError(
                f"Completed FlexiCT zero-shot output exists and resume is disabled: {root}"
            )
        seal = validate_completion(root, "test_completion.json")
        if read_json(root / "zero_shot_context.json") != context:
            raise ContractError(f"Completed FlexiCT zero-shot context changed: {root}")
        expected_completion_context = {
            "prompt_set_id": prompt_set_id,
            "context_sha256": fingerprint(context),
            "heldout_rows_sha256": rows_fingerprint(rows),
        }
        if seal.get("context") != expected_completion_context:
            raise ContractError(f"Completed FlexiCT zero-shot seal context changed: {root}")
        return {"status": "reused", "output": str(root.resolve())}
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ContractError(f"Partial FlexiCT zero-shot output exists and resume is disabled: {root}")
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
                f"Unknown unsealed FlexiCT zero-shot artifact; refusing resume: {unknown[0]}"
            )
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / "zero_shot_context.json"
    if context_path.is_file() and read_json(context_path) != context:
        raise ContractError(f"Partial FlexiCT zero-shot context changed: {root}")
    same_or_write_json(context_path, context)
    same_or_write_json(root / "prompt_profile.json", raw_prompts)

    text_model, tokenizer, logit_multiplier, text_details = build_text_encoder(
        upstream_root=args.upstream_root,
        checkpoint_path=checkpoint["path"],
        text_snapshot=text_snapshot,
        device=str(args.device),
        allow_trusted_legacy_checkpoint=bool(args.allow_trusted_legacy_checkpoint),
        trusted_checkpoint_sha256=str(args.checkpoint_sha256),
    )
    prompts, layout, text_embeddings = _text_bank(
        text_model,
        tokenizer,
        prompt_profile,
        str(args.device),
        dtype,
    )
    del text_model, tokenizer
    gc.collect()
    if str(args.device).lower().startswith("cuda"):
        torch.cuda.empty_cache()
    same_or_write_json(
        root / "prompt_bank.json",
        {
            "schema_version": 1,
            "prompts": prompts,
            "layout": layout,
            "embedding_shape": list(text_embeddings.shape),
            "logit_multiplier": logit_multiplier,
            "text_encoder": text_details,
        },
    )
    image_model, image_details = build_image_encoder(
        upstream_root=args.upstream_root,
        checkpoint_path=checkpoint["path"],
        device=str(args.device),
        allow_trusted_legacy_checkpoint=bool(args.allow_trusted_legacy_checkpoint),
        trusted_checkpoint_sha256=str(args.checkpoint_sha256),
    )

    progress_path = root / "progress.pt"
    progress_identity = {
        "context_sha256": fingerprint(context),
        "prompt_bank_sha256": sha256_file(root / "prompt_bank.json"),
        "row_count": len(rows),
    }
    if progress_path.is_file():
        payload = torch_load(progress_path, map_location="cpu")
        if payload.get("schema_version") != ZERO_SHOT_SCHEMA or payload.get(
            "identity"
        ) != progress_identity:
            raise ContractError(f"Incompatible FlexiCT zero-shot progress: {progress_path}")
        next_index = int(payload["next_index"])
        five_class_rows = [item.numpy() for item in payload["five_class_probabilities"]]
        pairwise_rows = [item.numpy() for item in payload["pairwise_positive_probabilities"]]
        similarity_rows = [item.numpy() for item in payload["paired_similarities"]]
    else:
        next_index = 0
        five_class_rows, pairwise_rows, similarity_rows = [], [], []
    lengths = (len(five_class_rows), len(pairwise_rows), len(similarity_rows))
    if not 0 <= next_index <= len(rows) or any(length != next_index for length in lengths):
        raise ContractError("Malformed FlexiCT zero-shot progress index")
    for five_class, pairwise, similarities in zip(
        five_class_rows,
        pairwise_rows,
        similarity_rows,
        strict=True,
    ):
        if (
            tuple(five_class.shape) != (len(CLASS_NAMES),)
            or tuple(pairwise.shape) != (len(CLASS_NAMES),)
            or tuple(similarities.shape) != (len(CLASS_NAMES), 2)
            or not np.isfinite(five_class).all()
            or not np.isfinite(pairwise).all()
            or not np.isfinite(similarities).all()
        ):
            raise ContractError("Malformed FlexiCT zero-shot progress payload")
    for index in range(next_index, len(rows)):
        five_class, pairwise, similarities = _score_image(
            image_model,
            rows[index]["image_path"],
            text_embeddings,
            logit_multiplier,
            str(args.device),
            precision_name,
        )
        five_class_rows.append(five_class)
        pairwise_rows.append(pairwise)
        similarity_rows.append(similarities)
        atomic_torch_save(
            {
                "schema_version": ZERO_SHOT_SCHEMA,
                "identity": progress_identity,
                "next_index": index + 1,
                "five_class_probabilities": [
                    torch.as_tensor(item) for item in five_class_rows
                ],
                "pairwise_positive_probabilities": [
                    torch.as_tensor(item) for item in pairwise_rows
                ],
                "paired_similarities": [
                    torch.as_tensor(item) for item in similarity_rows
                ],
                "updated_at": utc_now(),
            },
            progress_path,
        )

    probabilities = np.stack(five_class_rows, axis=0)
    pairwise_probabilities = np.stack(pairwise_rows, axis=0)
    similarities = np.stack(similarity_rows, axis=0)
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    metrics = compute_metrics(
        truth,
        probabilities.argmax(axis=1),
        probabilities,
        CLASS_NAMES,
    )
    json_rows, csv_rows = prediction_documents(rows, probabilities, CLASS_NAMES)
    for index, document in enumerate(json_rows):
        document["pairwise_positive_probabilities"] = {
            class_name: float(pairwise_probabilities[index, class_index])
            for class_index, class_name in enumerate(CLASS_NAMES)
        }
        document["paired_cosine_similarities"] = {
            class_name: {
                "positive": float(similarities[index, class_index, 0]),
                "negative": float(similarities[index, class_index, 1]),
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
            "text_snapshot": text_snapshot,
            "prompt_set_id": prompt_set_id,
            "prompt_profile_sha256": sha256_file(args.prompt_config),
            "logit_multiplier": logit_multiplier,
            "parameter_source": "publication",
            "publication_exact": False,
            "image_encoder": image_details,
            "text_encoder": text_details,
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
    return {
        "status": "complete",
        "output": str(root.resolve()),
        "heldout_count": len(rows),
    }


__all__ = [
    "ZERO_SHOT_SCHEMA",
    "load_prompts",
    "pairwise_five_class_probabilities",
    "run_zero_shot",
]
