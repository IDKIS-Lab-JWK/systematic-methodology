"""Lazy model construction and uncached image preprocessing for new development data."""
import sys
from pathlib import Path


def activate_models():
    folder = str(Path(__file__).resolve().parents[1] / 'Models')
    if folder not in sys.path:
        sys.path.insert(0, folder)


def resolve_device_precision(device, precision):
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if precision == 'auto':
        precision = 'bf16' if device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'fp32'
    if precision not in ('fp32','fp16','bf16'):
        raise ValueError('precision must be auto, fp32, fp16, or bf16')
    if precision != 'fp32' and not device.startswith('cuda'):
        raise ValueError('Mixed precision currently requires CUDA')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA device does not support bf16')
    return device, precision


def build_runtime(config, classes, train_rows=None):
    import torch
    from torch import nn
    activate_models()
    family, method, p = config['model'], config['method'], config['parameters']
    assets, device = config.get('assets', {}), config['device']
    runtime = {}
    if family in ('DenseNet','MedicalNet'):
        if train_rows is not None:
            from _cnn.geometry import fit_plan
            config['preprocessing_plan'] = fit_plan([dict(r, image_path=r['nifti_path'], class_name=r['label'], label=classes.index(r['label'])) for r in train_rows], p['canvas_xyz'])
        if 'preprocessing_plan' not in config:
            raise ValueError('CNN inference requires checkpoint preprocessing_plan')
        if family == 'DenseNet':
            from monai.networks.nets import DenseNet121
            model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=len(classes), dropout_prob=p['dropout'])
        else:
            from MedicalNet.model import MedicalNetClassifier
            model = MedicalNetClassifier(classes=len(classes))
            model.load_pretrained(assets['checkpoint'])
            if method != 'full':
                model.requires_grad_(False)
                model.fc.requires_grad_(True)
                if method == 'partial':
                    model.layer4.requires_grad_(True)
                runtime['train_modules'] = [model.fc] + ([model.layer4] if method == 'partial' else [])
    elif family == 'ThreeDINO':
        from ThreeDINO.model import build_model
        merged = dict(p, **assets, checkpoint_path=assets['checkpoint'], batch_size=config['batch_size'])
        merged.setdefault('lora_rank', 4)
        merged.setdefault('lora_alpha', 1.0)
        merged.setdefault('head_dropout', p['dropout'])
        model, runtime['model_details'] = build_model({'frozen':'linear_probe','partial':'lora','full':'full_finetune'}[method], len(classes), merged, device)
    elif family == 'Merlin':
        merlin_dir = str(Path(__file__).resolve().parents[1] / 'Models' / 'Merlin')
        if merlin_dir not in sys.path:
            sys.path.insert(0, merlin_dir)
        from Merlin.model import build_model
        merged = dict(p, dropout_prob=p['dropout'])
        merged.setdefault('conv_lora_rank', 4)
        merged.setdefault('conv_lora_alpha', 4)
        model, runtime['model_details'] = build_model({'frozen':'linear_probe','partial':'conv_lora','full':'full_finetune'}[method], len(classes), merged, torch.device(device))
    elif family in ('FlexiCT','COLIPRI'):
        if family == 'FlexiCT':
            from FlexiCT.model import build_vision_only_encoder
            encoder, runtime['model_details'] = build_vision_only_encoder(upstream_root=assets['upstream_root'], checkpoint_path=assets['checkpoint'], device=device, allow_trusted_legacy_checkpoint=bool(assets.get('allow_trusted_legacy_checkpoint',False)), trusted_checkpoint_sha256=assets.get('checkpoint_sha256'))
            if method == 'partial':
                from FlexiCT.final_block_finetune import FlexiCTFinalBlockClassifier
                from FlexiCT.final_quarter_finetune import FlexiCTFinalQuarterClassifier
                cls = FlexiCTFinalQuarterClassifier if p.get('partial_scope') == 'final_quarter' else FlexiCTFinalBlockClassifier
                model = cls(encoder, p['dropout'])
            dimension = 1024
        else:
            from COLIPRI.model import build_image_encoder
            encoder, runtime['processor'], runtime['model_details'] = build_image_encoder(upstream_root=assets['upstream_root'], checkpoint_path=assets['checkpoint'], device=device)
            if method == 'partial':
                from COLIPRI.partial_finetune import PrimusMPartialClassifier
                model = PrimusMPartialClassifier(encoder, p.get('partial_scope','final_block'), p['dropout'])
            dimension = 768
        if method == 'frozen':
            class FrozenClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = encoder.requires_grad_(False)
                    self.classifier = nn.Linear(dimension, len(classes))
                def train(self, mode=True):
                    super().train(mode)
                    self.encoder.eval()
                    return self
                def forward(self, x):
                    with torch.no_grad():
                        features = self.encoder(x) if family == 'FlexiCT' else self.encoder.encode_image(x, pool=True, project=True, normalize=False)
                    return self.classifier(features.float())
            model = FrozenClassifier()
        else:
            previous = model.classifier[-1]
            model.classifier[-1] = nn.Linear(previous.in_features, len(classes))
    elif family == 'Pillar0':
        from Pillar0.model import load_official_model, PillarClassifier
        kwargs = {'device':device, 'local_files_only':assets.get('local_files_only',True)}
        if assets.get('repo_id'): kwargs['repo_id'] = assets['repo_id']
        if assets.get('revision'): kwargs['revision'] = assets['revision']
        encoder = load_official_model(**kwargs)
        if method == 'partial':
            from Pillar0.partial_finetune import AtlasPartialClassifier
            model = AtlasPartialClassifier(encoder, p.get('partial_scope','final_block'), p['dropout'])
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
        else:
            model = PillarClassifier(encoder, num_classes=len(classes), frozen_encoder=method=='frozen', dropout=p['dropout'])
    else:
        raise ValueError(f'Unsupported model: {family}')
    return model.to(device), runtime


def set_training_mode(model, runtime, training):
    model.train(training)
    if training and 'train_modules' in runtime:
        model.eval()
        for module in runtime['train_modules']:
            module.train()


def preprocess_one(model_family, path, config, runtime=None):
    import torch
    activate_models()
    runtime = runtime if runtime is not None else {}
    if model_family in ('DenseNet','MedicalNet'):
        from _cnn.data import preprocess
        result = torch.from_numpy(preprocess(path, config['preprocessing_plan'])[0])
    elif model_family == 'FlexiCT':
        from FlexiCT.data import preprocess_nifti
        result = preprocess_nifti(path)[0]
    elif model_family == 'COLIPRI':
        from COLIPRI.model import preprocess_nifti
        if 'processor' not in runtime:
            raise ValueError('COLIPRI preprocessing needs runtime processor from build_runtime')
        result = preprocess_nifti(runtime['processor'], path)[0][0]
    elif model_family == 'ThreeDINO':
        from ThreeDINO.data import make_deterministic_transform, make_post_cache_transform
        if 'transform' not in runtime:
            prefix, suffix = make_deterministic_transform(), make_post_cache_transform('validation')
            runtime['transform'] = lambda value: suffix(prefix(value))
        result = runtime['transform']({'image':str(path)})['image']
    elif model_family in ('Merlin','Pillar0'):
        from inference.preprocessing import preprocess_direct
        result = preprocess_direct(model_family, path, config)
    else:
        raise ValueError(f'Unsupported model: {model_family}')
    result = torch.as_tensor(result, dtype=torch.float32)
    if result.ndim != 4 or not torch.isfinite(result).all():
        raise ValueError(f'Invalid preprocessed CDHW image: {path}, shape={tuple(result.shape)}')
    return result


