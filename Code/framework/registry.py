"""Dependency-light public method registry and generic training recipes."""
CAPABILITIES = {
    'DenseNet': {'scratch': '3D DenseNet121, random initialization'},
    'MedicalNet': {'frozen': 'linear fc head', 'partial': 'layer4 and fc', 'full': 'all image encoder parameters and fc'},
    'Merlin': {'frozen': 'linear probe', 'partial': 'ConvLoRA', 'full': 'full image encoder'},
    'ThreeDINO': {'frozen': 'linear probe', 'partial': 'attention LoRA', 'full': 'experimental full encoder (explicit config opt-in)'},
    'FlexiCT': {'frozen': 'projected image embedding linear head', 'partial': 'final block or final quarter'},
    'COLIPRI': {'frozen': 'official pooled projected embedding linear head', 'partial': 'final block or final stage'},
    'Pillar0': {'partial': 'Atlas final block or final stage', 'frozen': 'frozen image encoder and classifier', 'full': 'vision tower and classifier'},
}
PRESETS = {
    'default': {'epochs': 20, 'learning_rate': 0.0001, 'weight_decay': 0.0001, 'dropout': 0.0, 'canvas_xyz': [128,128,128], 'gradient_clip_norm': 1.0},
    'smoke': {'epochs': 1, 'learning_rate': 0.0001, 'weight_decay': 0.0, 'dropout': 0.0, 'canvas_xyz': [64,64,64], 'gradient_clip_norm': 1.0},
}


def validate_config(config):
    import math
    family, method = config['model'], config['method']
    if family not in CAPABILITIES or method not in CAPABILITIES[family]:
        raise ValueError(f'Unsupported model/method: {family}/{method}; use models for supported combinations')
    p, assets = config['parameters'], config.get('assets', {})
    for key in ('epochs',):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip_norm'):
        if not isinstance(p[key], (int, float)) or not math.isfinite(p[key]) or p[key] < 0 or (key == 'learning_rate' and p[key] == 0):
            raise ValueError(f'Invalid {key}')
    if not 0 <= p['dropout'] < 1:
        raise ValueError('dropout must lie in [0,1)')
    if config['batch_size'] < 1:
        raise ValueError('batch_size must be positive')
    if family in ('DenseNet','MedicalNet') and config['batch_size'] != 1:
        raise ValueError('CNN physical-space variable-shape preprocessing currently requires batch_size=1')
    if family in ('DenseNet','MedicalNet') and (len(p['canvas_xyz']) != 3 or any(type(n) is not int or n < 64 or n % 32 for n in p['canvas_xyz'])):
        raise ValueError('canvas_xyz requires three multiples of 32, each >=64 (BatchNorm with batch_size=1)')
    required = {'Pillar0': ['rave_root'], 'MedicalNet': ['checkpoint'], 'ThreeDINO': ['upstream_root','upstream_revision','checkpoint'], 'FlexiCT': ['upstream_root','checkpoint'], 'COLIPRI': ['upstream_root','checkpoint']}.get(family, [])
    missing = [key for key in required if not assets.get(key)]
    if missing:
        raise ValueError(f'{family} requires config.assets: {missing}')
    if family == 'ThreeDINO' and method == 'full' and not p.get('allow_experimental_full_finetune'):
        raise ValueError('ThreeDINO full requires parameters.allow_experimental_full_finetune=true')
    if family == 'FlexiCT' and p.get('partial_scope', 'final_block') not in ('final_block','final_quarter'):
        raise ValueError('FlexiCT partial_scope must be final_block or final_quarter')
    if family == 'COLIPRI' and p.get('partial_scope', 'final_block') not in ('final_block','final_stage'):
        raise ValueError('COLIPRI partial_scope must be final_block or final_stage')
    if family == 'Pillar0' and p.get('partial_scope', 'final_block') not in ('final_block','final_stage'):
        raise ValueError('Pillar0 partial_scope must be final_block or final_stage')
    if family == 'ThreeDINO' and method == 'partial':
        if config['batch_size'] < 2 or p['weight_decay'] != 0:
            raise ValueError('ThreeDINO LoRA requires batch_size >=2 and weight_decay=0')
        if p.get('gradient_checkpointing',False):
            raise ValueError('ThreeDINO LoRA gradient checkpointing is not validated')
    allowed = {'epochs','learning_rate','weight_decay','gradient_clip_norm','dropout','canvas_xyz'}
    allowed |= {'MedicalNet':set(), 'DenseNet':set(), 'Merlin':{'conv_lora_rank','conv_lora_alpha','train_image_projection','gradient_checkpointing'}, 'ThreeDINO':{'n_last_blocks','use_mean_patch_token','allow_experimental_full_finetune','gradient_checkpointing','lora_rank','lora_alpha','head_dropout'}, 'FlexiCT':{'partial_scope'}, 'COLIPRI':{'partial_scope'}, 'Pillar0':{'partial_scope'}}[family]
    if set(p)-allowed:
        raise ValueError(f'Unsupported parameters: {sorted(set(p)-allowed)}')

    if (family == 'MedicalNet' or (method == 'frozen' and family in ('Merlin','ThreeDINO','FlexiCT','COLIPRI'))) and p['dropout'] != 0:
        raise ValueError('This adapter uses a strict linear head; dropout must be 0')
    asset_keys = {'checkpoint','checkpoint_sha256','upstream_root','upstream_revision','rave_root','repo_id','revision','local_files_only','allow_trusted_legacy_checkpoint'}
    if set(assets)-asset_keys:
        raise ValueError(f'Unsupported assets keys: {sorted(set(assets)-asset_keys)}')
    if not 0 <= config['seed'] < 2**32-10000:
        raise ValueError('seed must lie in [0, 2**32-10000)')


def canonical_model(value):
    """Normalize public model spellings without importing model libraries."""
    aliases = {name.casefold():name for name in CAPABILITIES}
    aliases.update({'3dino':'ThreeDINO','pillar-0':'Pillar0'})
    try:
        return aliases[str(value).casefold()]
    except KeyError as exc:
        raise ValueError(f'Unknown model family: {value}') from exc
