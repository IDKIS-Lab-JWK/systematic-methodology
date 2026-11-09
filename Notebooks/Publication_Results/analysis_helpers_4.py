"""Read-only result ingestion and statistical utilities for Publication_Results_4.

No model/training imports. All writes are to the caller's versioned export folder.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)

CLASSES = ['Appendicitis', 'Cholezystitis', 'Normal', 'Perforation', 'Sigmoid']
DISPLAY_CLASSES = ['Appendicitis', 'Cholecystitis', 'Normal', 'Perforation', 'Sigmoid']
PERCENTAGES = [20, 40, 60, 80, 100]
SOURCES = ['cv_ensemble', 'best_cv_fold', 'refit_full_development']
METHODS = {
    '3DINO': ['LinearProbe', 'LoRA'],
    'COLIPRI': ['PoolingOptimisedLinearProbe', 'PrimusMFinalBlockFineTune'],
    'FlexiCT': ['LinearProbe', 'FinalBlockFineTune'],
    'Merlin': ['LinearProbe', 'ConvLoRA_LLRD', 'FullFineTune'],
    'Pillar0': ['LinearProbe', 'AtlasFinalBlockFineTune'],
    'MedicalNet': ['FullFineTune'], 'DenseNet': ['FromScratch'],
}
FOUNDATION_MODELS = ('COLIPRI', 'FlexiCT', 'Merlin', 'Pillar0')
BASELINE_MODELS = ('3DINO', 'MedicalNet', 'DenseNet')
FROZEN_METHODS = {'3DINO': 'LinearProbe', 'COLIPRI': 'PoolingOptimisedLinearProbe', 'FlexiCT': 'LinearProbe',
                  'Merlin': 'LinearProbe', 'Pillar0': 'LinearProbe'}
ADAPTATION_METHODS = {'3DINO': 'LoRA', 'COLIPRI': 'PrimusMFinalBlockFineTune', 'FlexiCT': 'FinalBlockFineTune',
                      'Merlin': 'ConvLoRA_LLRD', 'Pillar0': 'AtlasFinalBlockFineTune'}
EVIDENCE_NOTES = {
    ('3DINO', 'LinearProbe'): 'Publication-derived frozen-encoder linear probe: native 52-candidate search over learning rate and feature readout; development-only selection; five-class endpoint is a task adaptation',
    ('Merlin', 'FullFineTune'): 'Retrospective publication-derived full-network addition, FullFineTune CV_2; distinct from corrected ConvLoRA_LLRD CV_2 and historical FullFineTune CV_1', 
    ('3DINO', 'LoRA'): 'Project-adapted 3DINO LoRA classification; pretrained backbone with adapters, not full-network fine-tuning',
    ('COLIPRI', 'PoolingOptimisedLinearProbe'): 'Publication-derived native pooling search; learned pooling is not uniformly a strict linear probe',
    ('COLIPRI', 'PrimusMFinalBlockFineTune'): 'Publication-anchored downstream schedule/LR; final-block scope and five-class head are task adaptations',
    ('FlexiCT', 'LinearProbe'): 'Publication-derived frozen projected features and model-native classifier search',
    ('FlexiCT', 'FinalBlockFineTune'): 'Publication-anchored optimizer/LR/scheduler; final-block classification scope and stopping are task adaptations',
    ('Merlin', 'LinearProbe'): 'Publication-derived frozen image encoder with a task-adapted five-class linear head',
    ('Merlin', 'ConvLoRA_LLRD'): 'Publication-inspired image-only ConvLoRA with layer-wise learning-rate decay; task-adapted supervised protocol',
    ('Pillar0', 'LinearProbe'): 'Publication-derived frozen-feature linear head; five-class loss and labels are task adaptations',
    ('Pillar0', 'AtlasFinalBlockFineTune'): 'Publication-anchored optimizer/training budget; final Atlas block and five-class head are task adaptations',
    ('MedicalNet', 'FullFineTune'): 'Full-network transfer from official pretrained weights; five-class task adaptation',
    ('DenseNet', 'FromScratch'): 'All parameters trained from random initialization; full-network training, not pretrained-weight fine-tuning',
}
METHOD_LABELS = {
    'LinearProbe': 'Frozen features', 'PoolingOptimisedLinearProbe': 'Frozen + native pooling',
    'FrozenSingleHeadAttentionReadout': 'Frozen + attention readout',
    'LoRA': 'LoRA', 'FullFineTune': 'Full fine-tuning', 'PrimusMFinalBlockFineTune': 'Final block',
    'PrimusMFinalStageFineTune': 'Final quarter', 'FinalBlockFineTune': 'Final block',
    'FinalQuarterFineTune': 'Final quarter', 'AtlasFinalBlockFineTune': 'Final block',
    'AtlasFinalStageFineTune': 'Final stage', 'ConvLoRA': 'ConvLoRA',
    'ConvLoRA_LLRD': 'ConvLoRA + LLRD', 'FromScratch': '3D DenseNet121 from scratch',
    'ZeroShot': 'Zero-shot (task-adapted)',
}
METRICS = ['f1_macro', 'balanced_accuracy', 'average_precision_macro', 'accuracy',
           'roc_auc_macro', 'brier_multiclass', 'log_loss', 'ece']
METRIC_LABELS = {'f1_macro': 'Macro F1', 'balanced_accuracy': 'Balanced accuracy',
    'average_precision_macro': 'Macro average precision', 'accuracy': 'Accuracy',
    'roc_auc_macro': 'Macro OvR AUROC', 'brier_multiclass': 'Multiclass Brier score',
    'log_loss': 'Log loss', 'ece': 'Top-label ECE'}
REQUIRED = {'metrics_summary.json', 'model_selection.json',
            'test_patient_predictions.json', 'test_probabilities.csv'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, default=str).encode()).hexdigest()


def slug(value):
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(value)).strip('_')


def load_splits(root):
    splits, rows, audits = {}, [], []
    reference = None
    previous = set()
    for pct in PERCENTAGES:
        p = Path(root) / 'Dataset' / 'Splits' / f'{pct}_percent.json'
        j = read_json(p)
        test = j['heldout_test']['patients_by_class']
        cv = j['cross_validation']
        mapping = {str(pid): c for c in CLASSES for pid in test[c]}
        assert len(mapping) == sum(map(len, test.values())), 'Duplicate test IDs'
        assert reference is None or mapping == reference, 'Test cohort changes across percentages'
        reference = mapping
        dev = set()
        dev_counts = {c: 0 for c in CLASSES}
        for fold in cv['folds']:
            groups = fold['patients_by_class']
            ids = [str(x) for c in CLASSES for x in groups[c]]
            assert len(ids) == len(set(ids)), 'Duplicate fold IDs'
            assert not dev.intersection(ids), 'Validation folds overlap'
            dev.update(ids)
            for c in CLASSES:
                dev_counts[c] += len(groups[c])
        assert not dev.intersection(mapping), 'Development/test patient overlap'
        assert previous <= dev, 'Development subsets are not nested'
        assert dev_counts == cv['development_counts_by_class']
        assert len(dev) == cv['development_total']
        assert len(cv['folds']) == 5
        previous = dev
        splits[pct] = j
        rows.append({'cohort': f'{pct}% development', 'percentage': pct,
                     **dev_counts, 'total': len(dev)})
        audits.append({'percentage': pct, 'manifest_sha256': digest(p), 'test_n': len(mapping),
                       'development_n': len(dev), 'folds_disjoint': True,
                       'test_disjoint': True, 'nested': True})
    rows.append({'cohort': 'Fixed held-out test', 'percentage': 0,
                 **{c: len(splits[100]['heldout_test']['patients_by_class'][c]) for c in CLASSES},
                 'total': len(reference)})
    ids = sorted(reference)
    truth = np.array([CLASSES.index(reference[i]) for i in ids], dtype=int)
    return splits, pd.DataFrame(rows), pd.DataFrame(audits), ids, truth


def validate_seal(directory, model, method, selection, pct, source, cv_id):
    path = directory / 'test_completion.json'
    if not path.exists():
        legacy = model in {'Merlin', 'Pillar0'} and method in {
            'LinearProbe', 'FullFineTune', 'ConvLoRA', 'ConvLoRA_LLRD', 'ZeroShot'}
        if not legacy:
            raise ValueError('Required test completion seal missing')
        return 'legacy_unsealed_content_validated'
    seal = read_json(path)
    records = seal.get('artifacts', {})
    names = set(records) if isinstance(records, dict) else {r['path'] for r in records}
    if not REQUIRED <= names:
        raise ValueError('Completion seal does not cover all canonical artifacts')
    entries = [{'path': k, 'sha256': v} for k, v in records.items()] if isinstance(records, dict) else records
    for item in entries:
        target = (directory / item['path']).resolve()
        if not target.is_relative_to(directory.resolve()):
            raise ValueError('Seal path escapes result folder')
        if 'bytes' in item and target.stat().st_size != item['bytes']:
            raise ValueError(f'Seal size mismatch: {item["path"]}')
        if digest(target) != item['sha256']:
            raise ValueError(f'Seal hash mismatch: {item["path"]}')
    context = seal.get('context', seal)
    for key, expected in [('percentage', pct), ('cv_id', cv_id), ('model_source', source)]:
        if method != 'ZeroShot' and key in context and context[key] != expected:
            raise ValueError(f'Seal {key} differs from path')
    if 'seal_sha256' in seal:
        if fingerprint({'context': seal['context'], 'artifacts': records}) != seal['seal_sha256']:
            raise ValueError('Completion seal fingerprint mismatch')
    if 'selection_sha256' in context and context['selection_sha256'] != fingerprint(selection):
        raise ValueError('Selection document binding mismatch')
    if isinstance(records, dict):
        if seal.get('complete') is not True:
            raise ValueError('Run not marked complete')
        if seal.get('test_fingerprint') != selection.get('test_fingerprint'):
            raise ValueError('Test fingerprint differs from selection')
    return 'hash_seal_verified'


def prediction_arrays(directory, ids, truth):
    records = read_json(directory / 'test_patient_predictions.json')
    if not isinstance(records, list):
        raise ValueError('Unknown prediction JSON layout')
    by_id = {str(r['patient_id']): r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(ids):
        raise ValueError('Prediction patients are duplicated, missing, or extra')
    probabilities, predictions = [], []
    for pid, yi in zip(ids, truth):
        row = by_id[pid]
        actual = row.get('true_label', row.get('true_class'))
        pred = row.get('pred_label', row.get('predicted_class'))
        probs = row.get('probs', row.get('probabilities'))
        if actual != CLASSES[yi] or set(probs) != set(CLASSES):
            raise ValueError('Truth or class probability keys differ from manifest')
        probabilities.append([probs[c] for c in CLASSES])
        predictions.append(CLASSES.index(pred))
        for k, expected in [('true_index', yi), ('predicted_index', CLASSES.index(pred))]:
            if k in row and row[k] != expected:
                raise ValueError('Index/label disagreement')
    p = np.asarray(probabilities, dtype=float)
    pred = np.asarray(predictions)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError('Probability rows do not sum to one')
    if not np.all(pred == p.argmax(axis=1)):
        raise ValueError('Saved predictions disagree with argmax probabilities')
    frame = pd.read_csv(directory/'test_probabilities.csv', dtype={'patient_id': str})
    if frame['patient_id'].duplicated().any() or set(frame.patient_id) != set(ids):
        raise ValueError('CSV patients differ from JSON/manifest')
    frame = frame.set_index('patient_id').loc[ids]
    cols = CLASSES if set(CLASSES) <= set(frame) else [f'probability_{c}' for c in CLASSES]
    if not np.allclose(frame[cols].to_numpy(float), p, atol=1e-7, rtol=1e-6):
        raise ValueError('CSV/JSON probabilities disagree')
    for aliases, expected in [(['true_label', 'true_class'], np.array(CLASSES)[truth]),
                              (['pred_label', 'predicted_class'], np.array(CLASSES)[pred])]:
        key = next((k for k in aliases if k in frame), None)
        if key is None or not np.array_equal(frame[key].to_numpy(), expected):
            raise ValueError('CSV labels disagree with JSON/manifest')
    return p, pred


def provenance_for(root, model, method, cv_id, pct, selection):
    prov = selection.get('parameter_provenance') or {}
    context_path = root/'Results'/model/method/'CV'/cv_id/f'{pct}_percent'/'cv_run_context.json'
    if not prov and context_path.exists():
        context = read_json(context_path)['context']
        if context['cv_id'] != cv_id or context['percentage'] != pct:
            raise ValueError('CV context identity mismatch')
        if context.get('parameters') != selection.get('parameters'):
            raise ValueError('CV/test parameter disagreement')
        prov = context.get('parameter_provenance', {})
    src = prov.get('parameter_source', selection.get('parameter_source', 'unknown'))
    if src in {'publication', 'publication_parameters', 'publication_derived_parameters'}:
        family = 'publication'
    elif src in {'grid', 'shared_grid', 'percentage_specific_grid'}:
        family = 'grid'
    elif src in {'default', 'default_parameters'}:
        family = 'default'
    else:
        family = src
    return prov, src, family, context_path if context_path.exists() else None


def discover(root, ids, truth):
    root = Path(root)
    paths = []
    for model, methods in METHODS.items():
        for method in methods:
            paths.extend((root/'Results'/model/method/'Test').glob('CV_*/*/*_percent/metrics_summary.json'))
    zero_patterns = {
        'Merlin': 'five_class_adapted/*/metrics_summary.json',
        'Pillar0': 'exploratory_five_class_prompt_similarity/*/metrics_summary.json',
        'COLIPRI': '*/metrics_summary.json', 'FlexiCT': '*/metrics_summary.json'}
    for model, pattern in zero_patterns.items():
        paths.extend((root/'Results'/model/'ZeroShot'/'Test').glob(pattern))
    metadata, arrays, audit, summaries, source_files = [], {}, [], {}, {}
    # Record intentional method exclusions without reading or scoring their predictions.
    for model, methods in METHODS.items():
        for candidate in (root/'Results'/model).glob('*/Test/CV_*/*/*_percent/metrics_summary.json'):
            method = candidate.relative_to(root/'Results'/model).parts[0]
            if method not in methods:
                audit.append({'run_id': candidate.parent.relative_to(root/'Results').as_posix(),
                              'status': 'scope_excluded', 'reason': 'Method excluded by version-4 reporting scope'})
    for path in sorted(set(paths)):
        directory = path.parent
        rel = path.relative_to(root/'Results')
        model, method = rel.parts[:2]
        run_id = directory.relative_to(root/'Results').as_posix()
        try:
            absent = [x for x in REQUIRED if not (directory/x).is_file()]
            if absent:
                raise ValueError(f'Missing canonical artifacts: {absent}')
            initial_hashes = {name: digest(directory/name) for name in sorted(REQUIRED)}
            summary = read_json(path)
            selection = read_json(directory/'model_selection.json')
            metrics = summary.get('test_metrics', summary)
            if model == 'Merlin' and method == 'ConvLoRA_LLRD' and 'CV_1' in rel.parts:
                audit.append({'run_id':run_id,'status':'scope_excluded','reason':'Legacy implementation before adapter-gradient fix; corrected CV_2 required'})
                continue
            zero = method == 'ZeroShot'
            cv_id, source, pct = ('Pretrained', 'zero_shot', 0) if zero else (
                rel.parts[3], rel.parts[4], int(rel.parts[5].split('_')[0]))
            if not zero and (source not in SOURCES or pct not in PERCENTAGES):
                raise ValueError('Out-of-scope source or percentage')
            checkpoints = selection.get('checkpoints') or selection.get('classifier_artifacts') or selection.get('artifacts') or []
            if source == 'cv_ensemble' and len(checkpoints) != 5:
                raise ValueError('CV ensemble must identify exactly five checkpoint artifacts')
            for key, expected in [('cv_id', cv_id), ('percentage', pct), ('model_source', source)]:
                if not zero and key in selection and selection[key] != expected:
                    raise ValueError(f'Selection {key} differs from result path')
            if any(selection.get(k) is True for k in ['heldout_test_used_during_selection',
                                                     'heldout_test_used_during_prompt_selection']):
                raise ValueError('Provenance declares held-out data used for selection')
            seal_status = validate_seal(directory, model, method, selection, pct, source, cv_id)
            p, pred = prediction_arrays(directory, ids, truth)
            if 'class_order' in metrics and metrics['class_order'] != CLASSES:
                raise ValueError('Stored class order mismatch')
            prov, raw_source, family, cv_path = provenance_for(root, model, method, cv_id, pct, selection)
            if not zero and model in FOUNDATION_MODELS and family != 'publication':
                audit.append({'run_id': run_id, 'status': 'scope_excluded',
                              'reason': 'Foundation-model supervised results require publication-derived parameter provenance in version 4'})
                continue
            if zero:
                family, raw_source = 'zero_shot', 'fixed_prompt_profile'
            params = selection.get('parameters', prov.get('parameters', {}))
            files = {name: digest(directory/name) for name in sorted(REQUIRED)}
            if files != initial_hashes:
                raise ValueError('Source artifacts changed during ingestion; rerun after completion')
            if (directory/'test_completion.json').exists():
                files['test_completion.json'] = digest(directory/'test_completion.json')
            if cv_path:
                files['cv_run_context'] = digest(cv_path)
            label = f'{model} | {METHOD_LABELS[method]}'
            if zero and model == 'Pillar0':
                label = 'Pillar0 | Zero-shot (experimental)'
            profile = prov.get('profile_name') or prov.get('profile') or prov.get('profile_path') or raw_source
            profile = str(profile).replace('\\', '/').split('/')[-1]
            record = {'run_id': run_id, 'model': model, 'method': method, 'label': label,
                      'percentage': pct, 'source': source, 'cv_id': cv_id,
                      'parameter_source': raw_source, 'source_family': family,
                      'comparison_group': 'foundation' if model in FOUNDATION_MODELS else 'reference_baseline',
                      'evidence_note': EVIDENCE_NOTES.get((model, method), 'Separate model-specific zero-shot task adaptation'),
                      'profile': profile, 'profile_sha256': prov.get('profile_sha256', ''),
                      'parameter_signature': fingerprint(params)[:12],
                      'parameters_json': json.dumps(params, sort_keys=True),
                      'provenance_json': json.dumps(prov, sort_keys=True),
                      'integrity': seal_status, 'n_test': len(ids),
                      'artifact_fingerprint': fingerprint(files),
                      'selection_flag': selection.get('heldout_test_used_during_selection', 'not_recorded'),
                      'checkpoint_count': len(selection.get('checkpoints') or selection.get('classifier_artifacts') or selection.get('artifacts') or []),
                      'directory': str(directory), 'zero_shot': zero}
            metadata.append(record)
            arrays[run_id] = (p, pred)
            summaries[run_id] = metrics
            source_files[run_id] = files
            audit.append({'run_id': run_id, 'status': 'accepted', 'reason': seal_status})
        except (ValueError, KeyError, OSError, TypeError, IndexError, AssertionError) as exc:
            audit.append({'run_id': run_id, 'status': 'excluded', 'reason': str(exc)})
    return pd.DataFrame(metadata), arrays, pd.DataFrame(audit), summaries, source_files


def choose_runs(inventory, overrides=None):
    """No score-based ranking; ambiguous same-priority candidates remain unresolved."""
    overrides = overrides or {}
    selected, decisions = [], []
    for keys, group in inventory[~inventory.zero_shot].groupby(['model', 'method', 'percentage', 'source'], sort=False):
        key = '|'.join(map(str, keys))
        # Defensive policy enforcement even for caller-provided inventories and overrides.
        if keys[0] in FOUNDATION_MODELS:
            group = group[group.source_family == 'publication']
        preferred = group[group.source_family == 'publication']
        if preferred.empty:
            preferred = group[group.source_family == 'grid']
        if preferred.empty:
            preferred = group[group.source_family == 'default']
        if key in overrides:
            preferred = group[group.run_id == overrides[key]]
            if len(preferred) != 1:
                raise ValueError(f'Invalid run override: {key}')
            reason = 'explicit_configuration'
        else:
            reason = ('publication_only' if keys[0] in FOUNDATION_MODELS else
                      'baseline_publication_then_grid_then_default')
        if len(preferred) == 1:
            chosen = preferred.iloc[0].run_id
            selected.append(chosen)
            decisions.append({'comparison_cell': key, 'run_id': chosen, 'status': 'selected', 'rule': reason})
        else:
            decisions.append({'comparison_cell': key, 'run_id': '', 'status': 'ambiguous_or_unknown_provenance',
                              'rule': reason, 'candidates': ';'.join(group.run_id)})
    return inventory[inventory.run_id.isin(selected)].copy(), pd.DataFrame(decisions)


def coverage_table(inventory, selected, audit):
    rows = []
    for model, methods in METHODS.items():
        for method in methods:
            for pct in PERCENTAGES:
                for source in SOURCES:
                    valid = inventory[(inventory.model==model)&(inventory.method==method)&
                                      (inventory.percentage==pct)&(inventory.source==source)]
                    chosen = selected[(selected.model==model)&(selected.method==method)&
                                      (selected.percentage==pct)&(selected.source==source)]
                    prefix = f'{model}/{method}/Test/'
                    failed = audit[(audit.status=='excluded') & audit.run_id.str.startswith(prefix)
                                   & audit.run_id.str.endswith(f'/{source}/{pct}_percent')]
                    outside = audit[(audit.status=='scope_excluded') & audit.run_id.str.startswith(prefix)
                                    & audit.run_id.str.endswith(f'/{source}/{pct}_percent')]
                    status = 'available' if len(chosen)==1 else ('needs_configuration' if len(valid) else
                             ('invalid_or_incomplete' if len(failed) else 'pending'))
                    if status == 'pending' and len(outside):
                        status = 'pending_eligible_publication_run'
                    rows.append({'model':model,'method':method,'percentage':pct,'source':source,
                                 'status':status,'valid_candidates':len(valid),
                                 'selected_run':chosen.run_id.iloc[0] if len(chosen) else ''})
    return pd.DataFrame(rows)


def reliability(y, p, bins=10):
    confidence, pred = p.max(1), p.argmax(1)
    bin_id = np.minimum((confidence*bins).astype(int), bins-1)
    rows=[]
    for b in range(bins):
        mask=bin_id==b
        n=int(mask.sum())
        correct=int((pred[mask]==y[mask]).sum())
        lo,hi=wilson(correct,n)
        rows.append({'bin':b,'lower':b/bins,'upper':(b+1)/bins,'n':n,
                     'confidence':float(confidence[mask].mean()) if n else np.nan,
                     'accuracy':correct/n if n else np.nan, 'ci_low':lo,'ci_high':hi})
    return pd.DataFrame(rows)


def point_metrics(y, p, bins=10):
    pred=p.argmax(1)
    onehot=np.eye(5)[y]
    rel=reliability(y,p,bins)
    return {'f1_macro': f1_score(y,pred,labels=range(5),average='macro',zero_division=0),
            'balanced_accuracy':balanced_accuracy_score(y,pred), 'accuracy':accuracy_score(y,pred),
            'average_precision_macro':average_precision_score(onehot,p,average='macro'),
            'roc_auc_macro':roc_auc_score(onehot,p,average='macro'),
            'brier_multiclass':np.square(p-onehot).sum(1).mean(),
            'log_loss':-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean(),
            'ece':float((rel.n/len(y)*(rel.accuracy-rel.confidence).abs()).sum())}


def wilson(success, n):
    if n==0:return np.nan,np.nan
    z=norm.ppf(.975); v=success/n; d=1+z*z/n
    center=(v+z*z/(2*n))/d
    margin=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/d
    return max(0.,center-margin),min(1.,center+margin)


def class_metrics(y,p):
    cm=confusion_matrix(y,p.argmax(1),labels=range(5));rows=[]
    for k,c in enumerate(CLASSES):
        tp=int(cm[k,k]);fn=int(cm[k].sum()-tp);fp=int(cm[:,k].sum()-tp);tn=int(cm.sum()-tp-fn-fp)
        row={'class':c,'display_class':DISPLAY_CLASSES[k],'support':tp+fn,'tp':tp,'fn':fn,'fp':fp,'tn':tn,
             'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
             'average_precision':average_precision_score(y==k,p[:,k]),
             'roc_auc':roc_auc_score(y==k,p[:,k])}
        for name, a, n in [('sensitivity',tp,tp+fn),('specificity',tn,tn+fp),('precision',tp,tp+fp)]:
            lo,hi=wilson(a,n);row.update({name:a/n if n else np.nan,name+'_low':lo,name+'_high':hi})
        if c=='Perforation':
            exact=binomtest(tp,tp+fn).proportion_ci(method='exact')
            row.update(sensitivity_exact_low=exact.low,sensitivity_exact_high=exact.high)
        rows.append(row)
    return rows,cm


def stratified_weights(y, replicates, seed):
    rng=np.random.default_rng(seed)
    weights=np.zeros((replicates,len(y)),dtype=np.int16)
    for k in range(5):
        ids=np.flatnonzero(y==k)
        weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=replicates)
    return weights


def weighted_classification(y, pred, weights):
    """Vectorized confusion counts for bootstrap samples or per-replicate predictions."""
    pred=np.broadcast_to(pred,weights.shape)
    cm=np.stack([(weights*((y==i)&(pred==j))).sum(1) for i in range(5) for j in range(5)],axis=1).reshape(-1,5,5)
    tp=np.diagonal(cm,axis1=1,axis2=2)
    actual=cm.sum(2);positive=cm.sum(1)
    f1=np.divide(2*tp,actual+positive,out=np.zeros_like(tp,dtype=float),where=(actual+positive)>0).mean(1)
    return f1,(tp/actual).mean(1),tp.sum(1)/cm.sum((1,2))


def bootstrap_metrics(y,p,weights,bins=10):
    """Weighted empirical bootstrap; AP/AUROC aggregate tied score groups exactly."""
    output={m:[] for m in METRICS}
    pred=p.argmax(1);onehot=np.eye(5)[y]
    brier=np.square(p-onehot).sum(1)
    losses=-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))
    confidence=p.max(1);bid=np.minimum((confidence*bins).astype(int),bins-1)
    for start in range(0,len(weights),256):
        w=weights[start:start+256].astype(float);n=w.sum(1)
        f1,ba,acc=weighted_classification(y,pred,w)
        aps=[];aucs=[]
        for k in range(5):
            order=np.argsort(-p[:,k],kind='stable')
            ends=np.r_[np.flatnonzero(np.diff(p[order,k])!=0),len(y)-1]
            positive=w[:,order]*(y[order]==k)
            negative=w[:,order]*(y[order]!=k)
            tp=np.cumsum(positive,axis=1)[:,ends]
            fp=np.cumsum(negative,axis=1)[:,ends]
            increments=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
            precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
            aps.append((increments*precision).sum(1)/tp[:,-1])
            tpr=np.c_[np.zeros(len(w)),tp/tp[:,-1,None]]
            fpr=np.c_[np.zeros(len(w)),fp/fp[:,-1,None]]
            aucs.append((np.diff(fpr,axis=1)*(tpr[:,:-1]+tpr[:,1:])/2).sum(1))
        ece=np.zeros(len(w))
        for b in range(bins):
            mask=bid==b
            ece+=np.abs((w[:,mask]*((pred==y)[mask]-confidence[mask])).sum(1))/n
        values=[f1,ba,np.mean(aps,axis=0),acc,np.mean(aucs,axis=0),
                (w*brier).sum(1)/n,(w*losses).sum(1)/n,ece]
        for m,val in zip(METRICS,values):output[m].append(val)
    return {m:np.concatenate(v) for m,v in output.items()}


def bootstrap_cached(run, y, p, weights, cache_dir, seed, bins):
    identity=fingerprint({'artifact':run.artifact_fingerprint,'helper':digest(__file__),
                          'y':y.tolist(),'replicates':len(weights),'seed':seed,'bins':bins})
    path=Path(cache_dir)/f'{identity}_4.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            return {m:data[m].copy() for m in METRICS}
    result=bootstrap_metrics(y,p,weights,bins)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return result


def paired_test(y,pred_a,pred_b,replicates,seed):
    rng=np.random.default_rng(seed)
    observed=f1_score(y,pred_a,average='macro',zero_division=0)-f1_score(y,pred_b,average='macro',zero_division=0)
    extreme=0
    for start in range(0,replicates,256):
        size=min(256,replicates-start)
        swap=rng.integers(0,2,size=(size,len(y))).astype(bool)
        a=np.where(swap,pred_b,pred_a);b=np.where(swap,pred_a,pred_b)
        w=np.ones_like(a)
        delta=weighted_classification(y,a,w)[0]-weighted_classification(y,b,w)[0]
        extreme+=int((np.abs(delta)>=abs(observed)-1e-12).sum())
    a_correct=pred_a==y;b_correct=pred_b==y
    wins=int((a_correct&~b_correct).sum());losses=int((~a_correct&b_correct).sum())
    return {'permutation_p':(extreme+1)/(replicates+1), 'mcnemar_p':binomtest(wins,wins+losses).pvalue if wins+losses else 1.,
            'a_only_correct':wins,'b_only_correct':losses,'both_wrong':int((~a_correct&~b_correct).sum()),
            'both_correct':int((a_correct&b_correct).sum())}


def holm(values, family_size=None):
    values=np.asarray(values,dtype=float);n=family_size or len(values)
    if not len(values):return values
    order=np.argsort(values);adjusted=np.minimum(1,np.maximum.accumulate(values[order]*(n-np.arange(len(values)))))
    result=np.empty_like(adjusted);result[order]=adjusted
    return result


def formatted_table(frame, metrics):
    columns=['model','method','percentage','source','cv_id','source_family','integrity','n_test']
    out=frame[[c for c in columns if c in frame]].copy()
    for m in metrics:
        out[METRIC_LABELS[m]+' (95% CI)']=frame.apply(
            lambda r:f'{r[m]:.3f} ({r[m+"_low"]:.3f}, {r[m+"_high"]:.3f})',axis=1)
    return out
