"""Small end-to-end trainer for arbitrary classes and patient manifests.

This is a new generic AdamW/CE recipe, not a reproduction of study tuning.
All image preprocessing executes on demand; no persistent volume/feature cache.
"""
import copy
import hashlib
import secrets
import json
import platform
from importlib import metadata as package_metadata
import random
from pathlib import Path

from .data import digest, select
from .runtime import build_runtime, preprocess_one, resolve_device_precision, set_training_mode

FORMAT = 'systematic-framework-v1'


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def batches(rows, size, *, shuffle=False, seed=0, drop_last=False):
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), size):
        batch = [rows[i] for i in indices[start:start+size]]
        if drop_last and len(batch) < size:
            continue
        yield batch


def run_batches(model, runtime, config, rows, classes, *, optimizer=None, scaler=None, seed=0):
    import torch
    import torch.nn.functional as F
    training = optimizer is not None
    set_training_mode(model, runtime, training)
    size, device = config['batch_size'], config['device']
    total_loss, count, predictions, labels = 0.0, 0, [], []
    dtype = {'fp16':torch.float16, 'bf16':torch.bfloat16, 'fp32':None}[config['precision']]
    drop_last = training and config['model'] == 'ThreeDINO' and config['method'] == 'partial'
    for batch in batches(rows, size, shuffle=training, seed=seed, drop_last=drop_last):
        images = torch.stack([preprocess_one(config['model'], r['nifti_path'], config, runtime) for r in batch]).to(device)
        target = torch.tensor([classes.index(r['label']) for r in batch], dtype=torch.long, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype is not None):
            logits = model(images)
            if logits.shape != (len(batch),len(classes)) or not torch.isfinite(logits).all():
                raise ValueError('Model produced invalid class logits')
            loss = F.cross_entropy(logits.float(), target)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Fail if an advertised trainable parameter never participates.
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
                raise RuntimeError('No finite training gradients')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
                raise RuntimeError('Non-finite training gradients')
            if config['method'] in ('partial','full'):
                encoder_gradients = [p.grad for name,p in model.named_parameters() if p.requires_grad and not name.startswith(('classifier.','fc.')) and p.grad is not None]
                if not encoder_gradients:
                    raise RuntimeError('Requested encoder adaptation produced no encoder gradients')
            clip = config['parameters']['gradient_clip_norm']
            if clip:
                torch.nn.utils.clip_grad_norm_(trainable, clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach()) * len(batch)
        count += len(batch)
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
        labels.extend(target.cpu().tolist())
    if not count:
        raise ValueError('No usable batches (check sample count and batch_size)')
    recalls, f1s = [], []
    for i in range(len(classes)):
        tp = sum(a == b == i for a,b in zip(labels,predictions))
        fp = sum(a != i and b == i for a,b in zip(labels,predictions))
        fn = sum(a == i and b != i for a,b in zip(labels,predictions))
        if tp+fn: recalls.append(tp/(tp+fn))
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'loss':total_loss/count, 'accuracy':sum(a==b for a,b in zip(labels,predictions))/count, 'macro_f1':sum(f1s)/len(f1s), 'balanced_accuracy':sum(recalls)/len(recalls), 'samples':count}


def train(config, rows, splits, output):
    import numpy as np
    import torch
    output = Path(output)
    config = copy.deepcopy(config)
    config['device'], config['precision'] = resolve_device_precision(config['device'], config['precision'])
    development = select(rows, set(r['patient_id'] for r in rows)-set(splits['test']))
    classes = config.get('classes') or sorted({r['label'] for r in development})
    if not isinstance(classes,list) or len(set(classes)) != len(classes) or len(classes)<2 or any(not isinstance(c,str) or not c for c in classes):
        raise ValueError('classes must be a unique ordered list of >=2 nonempty label strings')
    if set(classes) != {r['label'] for r in development}:
        raise ValueError('Configured classes must exactly match development labels')
    config['classes'] = classes
    provenance = {'python':platform.python_version(), 'torch':str(torch.__version__), 'numpy':str(np.__version__), 'selection':'minimum validation cross entropy', 'cache_policy':'no persistent image/feature caches', 'data_sha256':{r['sample_id']:digest(r['nifti_path']) for r in development}, 'test_patients_excluded_from_training':splits['test']}
    provenance['packages'] = {}
    for name in ('monai','nibabel','scipy','SimpleITK','transformers','merlin-vlm','timm'):
        try: provenance['packages'][name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError: pass
    provenance['implementation_sha256'] = {str(p.relative_to(Path(__file__).parents[1])):digest(p) for p in Path(__file__).parent.glob('*.py')}
    assets = config.get('assets',{})
    if assets.get('checkpoint'):
        observed = digest(assets['checkpoint'])
        if assets.get('checkpoint_sha256') and assets['checkpoint_sha256'] != observed:
            raise ValueError('Pretrained checkpoint SHA-256 mismatch')
        assets['checkpoint_sha256'] = observed
    save_json(output/'resolved_config.json', config)
    save_json(output/'provenance.json', provenance)
    save_json(output/'splits.json', splits)
    results = []
    for number, fold in enumerate(splits['folds'], 1):
        seed = config['seed'] + number-1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        folder = output / f'fold_{number}'
        folder.mkdir()
        fold_config = copy.deepcopy(config)
        train_rows, val_rows = select(rows,fold['train']), select(rows,fold['validation'])
        if {r['label'] for r in train_rows} != set(classes):
            raise ValueError(f'Fold {number} training data does not contain every class')
        model, runtime = build_runtime(fold_config, classes, train_rows)
        trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise ValueError('Model has no trainable parameters')
        save_json(folder/'resolved_config.json', fold_config)
        save_json(folder/'trainable_parameters.json', {name:list(p.shape) for name,p in trainable})
        save_json(folder/'model_provenance.json',runtime.get('model_details',{}))
        optimizer = torch.optim.AdamW([p for _,p in trainable], lr=config['parameters']['learning_rate'], weight_decay=config['parameters']['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=config['precision']=='fp16')
        best, history = float('inf'), []
        patient_salt = secrets.token_hex(16)
        patient_fingerprints = [hashlib.sha256((patient_salt+p).encode()).hexdigest() for p in fold['train']+fold['validation']]
        checkpoint = folder/'best.pt'
        for epoch in range(config['parameters']['epochs']):
            train_metrics = run_batches(model,runtime,fold_config,train_rows,classes,optimizer=optimizer,scaler=scaler,seed=seed+epoch)
            val_metrics = run_batches(model,runtime,fold_config,val_rows,classes)
            history.append({'epoch':epoch+1,'train':train_metrics,'validation':val_metrics})
            save_json(folder/'history.json',history)
            if val_metrics['loss'] < best:
                best = val_metrics['loss']
                payload = {'format':FORMAT,'model_family':config['model'],'method':config['method'],'classes':classes,'config':fold_config,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch+1,'validation_metrics':val_metrics,'patient_fingerprint_salt':patient_salt,'development_patient_fingerprints':patient_fingerprints}
                temporary = folder/'best.pt.tmp'
                torch.save(payload,temporary)
                temporary.replace(checkpoint)
            print(f'fold={number} epoch={epoch+1} train_loss={train_metrics["loss"]:.6g} validation_loss={val_metrics["loss"]:.6g}',flush=True)
        sidecar = {'schema_version':1,'format':FORMAT,'model_family':config['model'],'method':config['method'],'classes':classes,'checkpoint_sha256':digest(checkpoint),'preprocessing_plan':fold_config.get('preprocessing_plan'),'runtime_assets':assets,'model_identity':{'recipe':'generic_adamw_cross_entropy','parameters':config['parameters']}}
        save_json(Path(str(checkpoint)+'.metadata.json'),sidecar)
        results.append({'fold':number,'checkpoint':str(checkpoint),'best_validation_loss':best})
        del model,optimizer,runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    save_json(output/'results.json',results)
    return results


def predict_checkpoint(checkpoint, records, *, device='auto', overrides=None):
    import torch
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Not a reusable framework checkpoint')
    config, classes = copy.deepcopy(payload['config']), payload['classes']
    if config['model'] != payload['model_family'] or config['method'] != payload['method'] or config['classes'] != classes:
        raise ValueError('Checkpoint model/method/class metadata disagree')
    overrides = dict(overrides or {})
    if 'checkpoint_path' in overrides:
        if 'checkpoint' in overrides and overrides['checkpoint'] != overrides['checkpoint_path']:
            raise ValueError('Conflicting checkpoint/checkpoint_path overrides')
        overrides['checkpoint'] = overrides.pop('checkpoint_path')
    requested_precision = overrides.pop('precision', None)
    if requested_precision is None:
        cpu_target = device == 'cpu' or (device == 'auto' and not torch.cuda.is_available())
        requested_precision = 'fp32' if cpu_target else config.get('precision','auto')
    config['device'],config['precision'] = resolve_device_precision(device,requested_precision)
    allowed = {'upstream_root','checkpoint','repo_id','local_files_only','rave_root'}
    if set(overrides)-allowed:
        raise ValueError(f'Only relocatable runtime asset overrides allowed: {sorted(allowed)}')
    config.setdefault('assets',{}).update(overrides)
    assets = config['assets']
    if assets.get('checkpoint') and assets.get('checkpoint_sha256') and digest(assets['checkpoint']) != assets['checkpoint_sha256']:
        raise ValueError('Pretrained asset SHA-256 differs from training checkpoint')
    seen_patients = set(payload.get('development_patient_fingerprints',[]))
    salt = payload.get('patient_fingerprint_salt','')
    if any(hashlib.sha256((salt+str(r.get('patient_id',''))).encode()).hexdigest() in seen_patients for r in records):
        raise ValueError('Independent test contains training or validation patients')
    model,runtime = build_runtime(config,classes)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for row in records:
            path = row.get('nifti_path') or row.get('image_path')
            image = preprocess_one(config['model'],path,config,runtime).unsqueeze(0).to(config['device'])
            dtype = {'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':None}[config['precision']]
            with torch.autocast(device_type=torch.device(config['device']).type, dtype=dtype, enabled=dtype is not None):
                values = torch.softmax(model(image).float(),dim=1)
            if values.shape != (1,len(classes)) or not torch.isfinite(values).all():
                raise ValueError('Invalid inference probabilities')
            probabilities.append(values[0].cpu().tolist())
    return {'probabilities':probabilities,'classes':classes,'sample_ids':[r['sample_id'] for r in records],'provenance':{'checkpoint_sha256':digest(checkpoint),'model_family':config['model'],'method':config['method'],'precision':config['precision'],'cache_policy':'no persistent image/feature caches'}}




