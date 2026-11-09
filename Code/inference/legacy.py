"""Exact adapters for the published classifier checkpoint formats.

Backbone weights are restored before partial tensors. Strict state-key checking
prevents accidentally evaluating a random or incompletely restored head.
"""
from __future__ import annotations
import contextlib
from pathlib import Path
import re
import numpy as np
from .direct import InferenceError, sha256


def _key(value):
    return re.sub('[^a-z0-9]', '', str(value).lower())


def _required(runtime, name):
    value = runtime.get(name)
    if not value:
        raise InferenceError(f'Missing model asset {name}; set it in runtime_assets or model_overrides')
    return value


def _find_hash(node):
    """Find recorded pretrained weight digests, excluding code/cache digests."""
    if not isinstance(node, dict):
        return None
    for key in ('checkpoint_sha256', 'pretrained_checkpoint_sha256', 'weights_sha256'):
        value = node.get(key)
        if isinstance(value, str) and re.fullmatch('[0-9a-fA-F]{64}', value):
            return value
    for key in ('checkpoint', 'encoder_checkpoint', 'pretrained_checkpoint'):
        child = node.get(key)
        if isinstance(child, dict):
            value = child.get('sha256')
            if isinstance(value, str) and re.fullmatch('[0-9a-fA-F]{64}', value):
                return value
            found = _find_hash(child)
            if found:
                return found
    for key in ('model_identity', 'feature_identity', 'source_provenance', 'source', 'model', 'asset_provisioning', 'model_details', 'frozen_encoder_provenance', 'pretrained_model'):
        found = _find_hash(node.get(key))
        if found:
            return found
    return None


def _asset_checkpoint(payload, runtime, expected=None):
    path = Path(_required(runtime, 'checkpoint_path'))
    known = expected or _find_hash(payload)
    requested = runtime.get('checkpoint_sha256')
    if known and requested and known.lower() != str(requested).lower():
        raise InferenceError('Backbone digest override differs from the saved training identity')
    digest = known or requested
    if not digest:
        raise InferenceError('External backbone requires a recorded checkpoint_sha256')
    if sha256(path).lower() != str(digest).lower():
        raise InferenceError(f'External backbone SHA-256 mismatch: {path}')
    return str(path), str(digest)


def _restore(model, payload):
    if 'trainable_state_dict' in payload:
        from _supervised_adaptation.runtime import load_trainable_state_dict
        load_trainable_state_dict(model, payload['trainable_state_dict'])
    elif 'model_state_dict' in payload:
        model.load_state_dict(payload['model_state_dict'], strict=True)
    elif 'state_dict' in payload:
        model.load_state_dict(payload['state_dict'], strict=True)
    else:
        raise InferenceError('Checkpoint has no compatible learned model state')
    return model.eval()


def _class_count(model, classes):
    """Adapt constructor defaults without changing saved state names."""
    import torch
    for attr in ('classifier', 'fc'):
        head = getattr(model, attr, None)
        if isinstance(head, torch.nn.Linear):
            if head.out_features != len(classes):
                setattr(model, attr, torch.nn.Linear(head.in_features, len(classes), bias=head.bias is not None))
            return model
        if isinstance(head, torch.nn.Sequential) and isinstance(head[-1], torch.nn.Linear):
            if head[-1].out_features != len(classes):
                head[-1] = torch.nn.Linear(head[-1].in_features, len(classes), bias=head[-1].bias is not None)
            return model
    layers = getattr(model, 'class_layers', None)
    head = getattr(layers, 'out', None)
    if isinstance(head, torch.nn.Linear):
        if head.out_features != len(classes):
            layers.out = torch.nn.Linear(head.in_features, len(classes), bias=head.bias is not None)
        return model
    raise InferenceError('Cannot locate classifier output layer')


def _autocast(device, precision):
    import torch
    name = str(precision).lower()
    if not str(device).startswith('cuda') or name in ('fp32', 'float32', 'none', 'false'):
        return contextlib.nullcontext()
    if name == 'auto':
        name = 'bf16' if torch.cuda.is_bf16_supported() else 'float16'
    mapping = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp16': torch.float16, 'float16': torch.float16}
    if name not in mapping:
        raise InferenceError(f'Unsupported inference precision {precision}')
    return torch.autocast('cuda', dtype=mapping[name])


def _precision(payload, runtime, default='fp32'):
    # An explicit override is recorded in output provenance. Published extraction
    # and head precision are otherwise taken from saved checkpoint contracts.
    if runtime.get('precision'):
        return runtime['precision']
    params = payload.get('parameters', {})
    if params.get('mixed_precision'):
        return params.get('autocast_dtype', 'auto')
    feature = payload.get('feature_identity', payload.get('model_identity', {}))
    def find_precision(node):
        if not isinstance(node, dict): return None
        for key in ('extraction_precision', 'training_precision_resolved', 'training_precision'):
            if isinstance(node.get(key), str): return node[key]
        for key in ('pretrained_model', 'runtime', 'frozen_cache', 'contract', 'training'):
            found = find_precision(node.get(key))
            if found: return found
        return None
    return find_precision(feature) or find_precision(payload.get('training', {})) or ('auto' if 'trainable_state_dict' in payload else default)


def _cnn(family, payload, classes, runtime, device):
    import torch
    from _cnn.data import preprocess
    plan = runtime.get('preprocessing_plan') or payload.get('preprocessing_plan') or payload.get('training', {}).get('preprocessing_plan')
    if not plan:
        raise InferenceError('CNN inference requires its fold-fitted preprocessing_plan in checkpoint or sidecar; it must never be fitted on test data')
    from _linear_probe.common import fingerprint
    if fingerprint(plan['numeric']) != plan.get('numeric_sha256'):
        raise InferenceError('CNN preprocessing plan fingerprint mismatch')
    if family == 'MedicalNet':
        from MedicalNet.model import MedicalNetClassifier
        model = MedicalNetClassifier(classes=len(classes))
    else:
        from DenseNet.model import build_model
        model = _class_count(build_model(payload.get('parameters', {}).get('dropout', 0.0)), classes)
    # Both published CNN families saved every parameter and running buffer.
    model = _restore(model, payload).to(device)
    return model, lambda path: torch.from_numpy(preprocess(path, plan)[0]), None


def _threedino(payload, classes, runtime, device):
    import torch
    from ThreeDINO.model import build_model, load_official_model, extract_representation_bank, feature_dimension, UPSTREAM_REVISION
    method = _key(payload.get('variant', payload.get('method')))
    path, digest = _asset_checkpoint(payload, runtime)
    assets = dict(upstream_root=_required(runtime, 'upstream_root'), checkpoint_path=path,
                  expected_checkpoint_sha256=digest, upstream_revision=runtime.get('upstream_revision', UPSTREAM_REVISION))
    if runtime.get('upstream_revision'):
        assets['upstream_revision'] = runtime['upstream_revision']
    if method == 'linearprobe':
        candidate = payload.get('candidate')
        if candidate is None:
            selected = str(payload.get('selected_candidate_id'))
            candidate = next((x for x in payload.get('candidates', []) if str(x.get('candidate_id')) == selected), None)
        if candidate is None:
            raise InferenceError('3DINO linear probe lacks its selected representation candidate')
        representation = (int(candidate['n_last_blocks']), bool(candidate['use_mean_patch_token']))
        state = payload.get('head_state_dict')
        if state is None:
            state = payload.get('candidate_head_state_dicts', {}).get(str(payload.get('selected_candidate_id')))
        if state is None:
            raise InferenceError('3DINO checkpoint lacks selected head weights')
        encoder = load_official_model(device=device, **assets).eval()
        head = torch.nn.Linear(feature_dimension(*representation), len(classes)).to(device)
        head.load_state_dict(state, strict=True)
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.head = encoder, head
            def forward(self, images):
                return self.head(extract_representation_bank(self.encoder, images, [representation])[representation])
        model = Probe().eval()
    elif method in ('lora', 'fullfinetune'):
        params = {**payload['parameters'], 'upstream_root': assets['upstream_root'],
                  'checkpoint_path': path, 'checkpoint_sha256': digest}
        if assets.get('upstream_revision'):
            params['upstream_revision'] = assets['upstream_revision']
        model, _ = build_model(variant='lora' if method == 'lora' else 'full_finetune',
                               num_classes=len(classes), parameters=params, device=device)
        if method == 'lora':
            if payload.get('checkpoint_scope') != 'lora_and_classifier' or list(model.target_modules) != payload.get('lora_target_modules'):
                raise InferenceError('3DINO LoRA target modules or state scope differ from checkpoint')
            model.load_adaptation_state_dict(payload['model_state_dict'])
        else:
            _restore(model, payload)
        model.eval()
    else:
        raise InferenceError(f'Unsupported 3DINO method: {method}')
    from ThreeDINO.data import make_dataset
    def preprocess(path):
        row = {'image': str(path), 'image_path': str(path), 'patient_id': 'inference', 'label': -1}
        return make_dataset([row], split='test', cache_dir=None)[0]['image']
    return model, preprocess, None


def _flexict(method, payload, classes, runtime, device):
    import torch
    from FlexiCT.model import build_vision_only_encoder
    from FlexiCT.data import preprocess_nifti
    path, digest = _asset_checkpoint(payload, runtime)
    encoder, _ = build_vision_only_encoder(upstream_root=_required(runtime, 'upstream_root'),
        checkpoint_path=path, device=device,
        allow_trusted_legacy_checkpoint=bool(runtime.get('allow_trusted_pickle', False)),
        trusted_checkpoint_sha256=digest)
    params = payload.get('parameters', {})
    if method in ('linearprobe',):
        pipeline = payload.get('sklearn_pipeline')
        if pipeline is None:
            raise InferenceError('FlexiCT linear probe requires its sklearn pipeline artifact')
        def predict_proba(image):
            # The released frozen VLM projection and L2 normalization belong to
            # the encoder, then the saved pipeline applies its own scaler/head.
            features = encoder(image).float().cpu().numpy()
            values = pipeline.predict_proba(features)
            order = list(pipeline.classes_)
            if order == list(range(len(classes))):
                return values
            if set(order) == set(classes):
                return values[:, [order.index(c) for c in classes]]
            raise InferenceError('FlexiCT sklearn classes do not match declared output classes')
        return encoder.eval(), lambda path: preprocess_nifti(path)[0], predict_proba
    if method in ('frozensingleheadattentionreadout', 'attentionreadout'):
        from FlexiCT.attention_readout import SingleHeadAttentionReadout
        head = _class_count(SingleHeadAttentionReadout(), classes)
        _restore(head, payload)
        class AttentionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.head = encoder, head
            def forward(self, image):
                patches = self.encoder.vision(image, is_training=True)['x_norm_patchtokens']
                if tuple(patches.shape[1:]) != (8000, 864):
                    raise InferenceError('Unexpected FlexiCT token grid')
                # Reproduce the training boundary's FP16 storage quantization
                # in memory; no shard is created.
                tokens = patches.float().reshape(-1, 20, 20, 20, 864).mean(dim=(2, 3)).half().float()
                return self.head(tokens)
        model = AttentionModel().to(device).eval()
    elif method in ('finalblockfinetune', 'finalquarterfinetune'):
        if method == 'finalblockfinetune':
            from FlexiCT.final_block_finetune import FlexiCTFinalBlockClassifier as Classifier
        else:
            from FlexiCT.final_quarter_finetune import FlexiCTFinalQuarterClassifier as Classifier
        model = _class_count(Classifier(encoder, float(params.get('dropout', 0.0))), classes).to(device)
        _restore(model, payload)
    else:
        raise InferenceError(f'Unsupported FlexiCT method: {method}')
    return model, lambda path: preprocess_nifti(path)[0], None


def _colipri(method, payload, classes, runtime, device):
    import torch
    from COLIPRI.model import build_image_encoder, preprocess_nifti
    from COLIPRI import MODEL_SHA256
    path, _ = _asset_checkpoint(payload, runtime, MODEL_SHA256)
    encoder, processor, _ = build_image_encoder(upstream_root=_required(runtime, 'upstream_root'), checkpoint_path=path, device=device)
    if method in ('poolingoptimisedlinearprobe', 'linearprobe'):
        from COLIPRI.pooling import COLIPRIPoolingOptimisedLinearProbe
        architecture = payload['architecture']
        head = COLIPRIPoolingOptimisedLinearProbe(pooling=architecture['pooling'],
            embed_dim=int(architecture['embed_dim']), num_heads=int(architecture['num_heads']), num_classes=len(classes)).to(device)
        head.load_state_dict(payload['state_dict'], strict=True)
        class PoolingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.head = encoder, head
            def forward(self, images):
                dense = self.encoder.encode_image(images, pool=False, project=False, normalize=False)
                return self.head(dense.float())
        model = PoolingModel().eval()
    elif method in ('primusmfinalblockfinetune', 'primusmfinalstagefinetune'):
        from COLIPRI.partial_finetune import PrimusMPartialClassifier
        scope = 'final_stage' if method == 'primusmfinalstagefinetune' else 'final_block'
        model = _class_count(PrimusMPartialClassifier(encoder, scope, float(payload['parameters'].get('dropout', 0.0))), classes).to(device)
        _restore(model, payload)
    else:
        raise InferenceError(f'Unsupported COLIPRI method: {method}')
    return model, lambda path: preprocess_nifti(processor, path)[0][0], None


def _merlin(method, payload, classes, runtime, device):
    import torch
    from Merlin.model import build_model
    variant = {'linearprobe': 'linear_probe', 'convlorallrd': 'conv_lora_llrd',
               'convlora': 'conv_lora', 'fullfinetune': 'full_finetune'}.get(method)
    if not variant:
        raise InferenceError(f'Unsupported Merlin method: {method}')
    import importlib, inspect
    loader = importlib.import_module('merlin.models.load')
    installed = Path(inspect.getfile(loader.Merlin)).resolve().parent / 'checkpoints' / loader.MODEL_CONFIGS['default']['checkpoint']
    expected = _find_hash(payload)
    if not installed.is_file():
        raise InferenceError('Install the official Merlin model asset before inference')
    if variant == 'linear_probe' and not expected:
        raise InferenceError('Frozen Merlin checkpoint lacks the pretrained encoder digest')
    if expected and sha256(installed) != expected:
        raise InferenceError('Installed Merlin encoder differs from the saved training identity')
    model, _ = build_model(variant, len(classes), payload['parameters'], torch.device(device))
    if variant == 'linear_probe':
        model.classifier.load_state_dict(payload['classifier_state_dict'], strict=True)
    else:
        model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device).eval()
    from framework.runtime import preprocess_one
    return model, lambda path: preprocess_one('Merlin', path, {'assets': runtime, 'parameters': payload['parameters']}), None


def _pillar(method, payload, classes, runtime, device):
    import torch
    from Pillar0.model import load_official_model, extract_vision_features, PillarClassifier, MODEL_REPO_ID, MODEL_REVISION, FEATURE_DIM
    repo_id = payload.get('model_repo_id', runtime.get('model_repo_id', MODEL_REPO_ID))
    revision = payload.get('model_revision', runtime.get('model_revision', MODEL_REVISION))
    encoder = load_official_model(repo_id=repo_id, revision=revision, device=device, local_files_only=runtime.get('local_files_only', True))
    if method == 'linearprobe':
        head = torch.nn.Linear(FEATURE_DIM, len(classes)).to(device)
        head.load_state_dict(payload['head_state_dict'], strict=True)
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.head = encoder, head
            def forward(self, images):
                return self.head(extract_vision_features(self.encoder, images))
        model = Probe().eval()
    elif method == 'fullfinetune':
        model = PillarClassifier(encoder, len(classes), frozen_encoder=False,
                                dropout=float(payload['parameters'].get('dropout', 0.0))).to(device)
        _restore(model, payload)
    elif method in ('atlasfinalblockfinetune', 'atlasfinalstagefinetune'):
        from Pillar0.partial_finetune import AtlasPartialClassifier
        scope = 'final_stage' if method == 'atlasfinalstagefinetune' else 'final_block'
        model = _class_count(AtlasPartialClassifier(encoder, scope, float(payload['parameters'].get('dropout', 0.0))), classes).to(device)
        _restore(model, payload)
    else:
        raise InferenceError(f'Unsupported Pillar0 method: {method}')
    from .pillar import preprocess_nifti
    return model, lambda path: preprocess_nifti(path, runtime), None


def predict_payload(family, method, payload, rows, classes, runtime, device):
    """Restore one classifier, preprocess each source and return probabilities."""
    import torch
    method = _key(method)
    if family in ('MedicalNet', 'DenseNet'):
        model, preprocess, probability_fn = _cnn(family, payload, classes, runtime, device)
    elif family == 'ThreeDINO':
        model, preprocess, probability_fn = _threedino(payload, classes, runtime, device)
    elif family == 'FlexiCT':
        model, preprocess, probability_fn = _flexict(method, payload, classes, runtime, device)
    elif family == 'COLIPRI':
        model, preprocess, probability_fn = _colipri(method, payload, classes, runtime, device)
    elif family == 'Merlin':
        model, preprocess, probability_fn = _merlin(method, payload, classes, runtime, device)
    elif family == 'Pillar0':
        model, preprocess, probability_fn = _pillar(method, payload, classes, runtime, device)
    else:
        raise InferenceError(f'Unsupported family {family}')
    precision = _precision(payload, runtime)
    runtime['precision'] = precision if str(device).startswith('cuda') else 'fp32'
    result = []
    with torch.inference_mode():
        for row in rows:
            tensor = preprocess(row['image_path'])
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or not bool(torch.isfinite(tensor).all()):
                raise InferenceError('Preprocessor must return a finite [C,D,H,W] tensor')
            images = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
            with _autocast(device, precision):
                values = probability_fn(images) if probability_fn else model(images).float().softmax(-1).cpu().numpy()
            result.append(np.asarray(values, dtype=np.float64))
    return np.concatenate(result, axis=0)

