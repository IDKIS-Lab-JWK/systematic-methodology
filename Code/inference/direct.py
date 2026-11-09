"""Portable held-out inference with explicit checkpoint and class identity.

No feature store, volume cache, or persistent test dataset is constructed.
Frozen model assets must be installed in the selected model environment.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import numpy as np

class InferenceError(RuntimeError):
    pass

@dataclass
class PredictionResult:
    probabilities: np.ndarray
    classes: list[str]
    sample_ids: list[str]
    provenance: dict

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_family(value):
    aliases = {'merlin': 'Merlin', 'pillar0': 'Pillar0', 'pillar-0': 'Pillar0',
               '3dino': 'ThreeDINO', 'threedino': 'ThreeDINO', 'flexict': 'FlexiCT',
               'colipri': 'COLIPRI', 'medicalnet': 'MedicalNet', 'densenet': 'DenseNet'}
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise InferenceError(f'Unknown model family: {value}') from exc

def canonical_method(family, value):
    normalized = ''.join(c for c in str(value).lower() if c.isalnum())
    # COLIPRI renamed its published pooling search without changing tensors.
    if family == 'COLIPRI' and normalized == 'linearprobe':
        return 'poolingoptimisedlinearprobe'
    return normalized


def activate_models():
    path = str(Path(__file__).resolve().parents[1] / 'Models')
    if path not in sys.path:
        sys.path.insert(0, path)

def _metadata(path, metadata_path):
    candidate = Path(metadata_path) if metadata_path else Path(str(path) + '.metadata.json')
    if not candidate.is_file():
        if metadata_path:
            raise FileNotFoundError(candidate)
        return {}
    value = json.loads(candidate.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise InferenceError(f'Inference sidecar must be a JSON object: {candidate}')
    if not isinstance(value.get('runtime_assets', {}), dict):
        raise InferenceError('runtime_assets must be a JSON object')
    runtime = dict(value.get('runtime_assets', {}))
    for name in ('upstream_root', 'checkpoint_path', 'rave_root', 'rave_config'):
        if isinstance(runtime.get(name), str):
            p = Path(runtime[name])
            runtime[name] = str(p if p.is_absolute() else (candidate.parent / p).resolve())
    value['runtime_assets'] = runtime
    return value

def _load(path, metadata, allow_trusted_pickle):
    expected = metadata.get('checkpoint_sha256')
    observed = sha256(path)
    if expected and observed.lower() != str(expected).lower():
        raise InferenceError(f'Checkpoint SHA-256 mismatch: {path}')
    if path.suffix.lower() in {'.joblib', '.pkl', '.pickle'}:
        if not allow_trusted_pickle or not expected:
            raise InferenceError('Legacy sklearn loading requires allow_trusted_pickle=True and a sidecar checkpoint_sha256')
        import joblib
        return {'sklearn_pipeline': joblib.load(path)}, observed
    import torch
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        if not allow_trusted_pickle or not expected:
            raise InferenceError('Checkpoint is not weights-only compatible. For an independently trusted legacy artifact supply its SHA-256 sidecar and allow_trusted_pickle=True.') from exc
        payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise InferenceError('Checkpoint must contain a named-state dictionary')
    return payload, observed

def _classes(payload, metadata):
    stored = payload.get('classes', payload.get('class_names'))
    supplied = metadata.get('classes', metadata.get('class_names'))
    if stored is not None and supplied is not None and list(stored) != list(supplied):
        raise InferenceError('Checkpoint and sidecar class ordering differ')
    classes = stored if stored is not None else supplied
    if not isinstance(classes, (list, tuple)) or len(classes) < 2 or any(not isinstance(x, str) or not x for x in classes) or len(set(classes)) != len(classes):
        raise InferenceError('Checkpoint or sidecar must declare at least two unique ordered class names')
    return list(classes)

def _records(records):
    result = []
    for source in records:
        path = source.get('nifti_path', source.get('image_path', source.get('image')))
        if path is None or not Path(path).is_file():
            raise InferenceError(f'Source image does not exist: {path}')
        identifier = str(source.get('sample_id', source.get('patient_id', '')))
        if not identifier:
            raise InferenceError('Every inference record requires sample_id or patient_id')
        result.append({**source, 'sample_id': identifier, 'patient_id': str(source.get('patient_id', identifier)),
                       'image_path': str(Path(path).resolve()), 'image': str(Path(path).resolve()),
                       'nifti_path': str(Path(path).resolve()), 'label': source.get('label', -1)})
    if not result or len({r['sample_id'] for r in result}) != len(result):
        raise InferenceError('Inference requires nonempty records with unique sample IDs')
    return result

def _checked_probabilities(value, rows, classes):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (len(rows), len(classes)) or not np.isfinite(value).all():
        raise InferenceError(f'Invalid probability shape or non-finite values: {value.shape}')
    if np.any(value < 0) or np.any(value > 1) or not np.allclose(value.sum(1), 1, atol=1e-5):
        raise InferenceError('Model output is not a normalized probability distribution')
    return value

def predict(checkpoints: Sequence[str | Path], records: Sequence[Mapping[str, Any]], *,
            model_family: str, device: str = 'auto', metadata_paths=None,
            model_overrides: Mapping[str, Any] | None = None,
            allow_trusted_pickle: bool = False) -> PredictionResult:
    """Predict from one checkpoint or a five-fold arithmetic probability mean.

    Sidecars default to <checkpoint>.metadata.json. Asset paths can be relocated
    with model_overrides; no runtime option enables a persistent test cache.
    """
    if isinstance(checkpoints, (str, Path)):
        checkpoints = [checkpoints]
    paths = [Path(p).resolve() for p in checkpoints]
    if len(paths) not in (1, 5) or len(set(paths)) != len(paths):
        raise InferenceError('Select one checkpoint or five distinct fold checkpoints')
    if metadata_paths is None:
        metadata_paths = [None] * len(paths)
    if len(metadata_paths) != len(paths):
        raise InferenceError('metadata_paths must match checkpoint count')
    family = canonical_family(model_family)
    rows = _records(records)
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    activate_models()
    from .legacy import predict_payload
    probabilities, identities = [], []
    classes, method = None, None
    for path, metadata_path in zip(paths, metadata_paths):
        metadata = _metadata(path, metadata_path)
        if metadata.get('model_family') and canonical_family(metadata['model_family']) != family:
            raise InferenceError('Selected family differs from checkpoint sidecar')
        payload, digest = _load(path, metadata, allow_trusted_pickle)
        # Sidecar identity supplies essential asset hashes for formats such as sklearn.
        for field in ('model_identity', 'parameters', 'preprocessing_plan'):
            if field not in payload and metadata.get(field) is not None:
                payload[field] = metadata[field]
        stored_family = payload.get('model_family', payload.get('model'))
        if isinstance(stored_family, str) and canonical_family(stored_family) != family:
            raise InferenceError('Selected family differs from checkpoint')
        ordered = _classes(payload, metadata)
        selected_method = payload.get('variant', payload.get('method', payload.get('method_label', metadata.get('method'))))
        if metadata.get('method') and selected_method:
            if canonical_method(family, metadata['method']) != canonical_method(family, selected_method):
                raise InferenceError('Checkpoint method differs from sidecar')
        if not selected_method:
            raise InferenceError('Checkpoint or sidecar must declare its training method')
        selected_method = canonical_method(family, selected_method)
        if classes is not None and (ordered != classes or selected_method != method):
            raise InferenceError('Ensemble checkpoint class order or method differs')
        classes, method = ordered, selected_method
        runtime = {**metadata.get('runtime_assets', {}), **dict(model_overrides or {})}
        if metadata.get('preprocessing_plan') is not None:
            runtime['preprocessing_plan'] = metadata['preprocessing_plan']
        if runtime.get('checkpoint') and not runtime.get('checkpoint_path'):
            runtime['checkpoint_path'] = runtime['checkpoint']
        if any(key in runtime for key in ('cache_dir', 'cache_root', 'feature_cache_root')):
            raise InferenceError('Persistent test cache options are not supported')
        explicit_precision = dict(model_overrides or {}).get('precision')
        if explicit_precision and explicit_precision not in ('auto', 'fp32', 'float32') and not str(device).startswith('cuda'):
            raise InferenceError('Explicit mixed-precision inference requires CUDA')
        runtime['allow_trusted_pickle'] = allow_trusted_pickle
        if payload.get('format') == 'systematic-framework-v1':
            from framework.training import predict_checkpoint
            # The framework owns reconstruction; release tensors before its read.
            payload = None
            values = predict_checkpoint(path, rows, device=device, overrides=dict(model_overrides or {}))['probabilities']
        else:
            values = predict_payload(family, selected_method, payload, rows, classes, runtime, device)
        probabilities.append(_checked_probabilities(values, rows, classes))
        identities.append({'path': str(path), 'sha256': digest, 'method': method,
                           'runtime': runtime})
        del payload
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return PredictionResult(np.mean(probabilities, axis=0), classes,
                            [r['sample_id'] for r in rows],
                            {'model_family': family, 'classes': classes, 'checkpoints': identities,
                             'ensemble': 'probability_mean' if len(paths) == 5 else 'single',
                             'device': str(device), 'persistent_test_cache': False,
                             'preprocessing': 'source_image_in_memory', 'samples': len(rows)})
