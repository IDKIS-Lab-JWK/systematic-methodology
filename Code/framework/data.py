"""Explicit sample manifests and patient-group-safe development splits."""
import csv
import hashlib
import json
import random
from pathlib import Path


def read_records(path, *, root=None, labeled=True, check_files=True):
    path = Path(path).resolve()
    if path.suffix.lower() == '.json':
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        rows = value.get('samples', []) if isinstance(value, dict) else value
    else:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Dataset manifest must contain a nonempty sample list')
    base = Path(root).resolve() if root else path.parent
    seen, paths, result = set(), set(), []
    for row in rows:
        required = ['sample_id', 'patient_id', 'nifti_path'] + (['label'] if labeled else [])
        if not isinstance(row, dict) or any(row.get(k) is None or not str(row.get(k, '')).strip() for k in required):
            raise ValueError(f'Each sample requires nonempty {required}')
        record = {k: str(v).strip() for k, v in row.items()}
        source = Path(record['nifti_path'])
        source = (base / source).resolve() if not source.is_absolute() else source.resolve()
        if not (source.name.endswith('.nii') or source.name.endswith('.nii.gz')):
            raise ValueError(f'Expected a NIfTI image: {source}')
        if record['sample_id'] in seen or str(source).casefold() in paths:
            raise ValueError('Duplicate sample_id or image path in manifest')
        if check_files and not source.is_file():
            raise FileNotFoundError(source)
        seen.add(record['sample_id'])
        paths.add(str(source).casefold())
        record['nifti_path'] = str(source)
        result.append(record)
    return result


def validate_splits(rows, folds, test):
    patients = {r['patient_id'] for r in rows}
    test = set(test)
    if not test <= patients:
        raise ValueError('Test manifest contains unknown patients')
    if not folds:
        raise ValueError('At least one training/validation split is required')
    for fold in folds:
        train, validation = set(fold['train']), set(fold['validation'])
        if not train or not validation:
            raise ValueError('Every fold needs nonempty train and validation patients')
        if train & validation or (train | validation) & test:
            raise ValueError('Patient leakage across train, validation, or independent test')
        if train | validation | test != patients:
            raise ValueError('Every patient must occur in exactly one partition of each fold')
        train_labels = {r['label'] for r in rows if r['patient_id'] in train}
        val_labels = {r['label'] for r in rows if r['patient_id'] in validation}
        if len(train_labels) < 2 or not val_labels <= train_labels:
            raise ValueError('Each training fold requires >=2 classes and all validation classes')
    if len(folds) > 1:
        validation_patients = [p for f in folds for p in f['validation']]
        if len(validation_patients) != len(set(validation_patients)) or set(validation_patients) != patients - test:
            raise ValueError('CV validation partitions must cover development patients exactly once')


def resolve_splits(rows, *, manifest=None, seed=0, cv_folds=0, validation_fraction=.2, test_fraction=0):
    if manifest:
        obj = json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
        test = obj.get('test', [])
        folds = obj.get('folds', [{'train': obj.get('train', []), 'validation': obj.get('validation', [])}])
    else:
        if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1:
            raise ValueError('Fractions require 0 < validation < 1 and 0 <= test < 1')
        if cv_folds != 0 and cv_folds < 2:
            raise ValueError('cv_folds must be 0 or >=2')
        signatures = {}
        for row in rows:
            signatures.setdefault(row['patient_id'], set()).add(row['label'])
        strata = {}
        for patient, labels in signatures.items():
            strata.setdefault(tuple(sorted(labels)), []).append(patient)
        rng = random.Random(seed)
        test, development, buckets = [], [], [[] for _ in range(cv_folds or 2)]
        for signature in sorted(strata):
            group = sorted(strata[signature])
            rng.shuffle(group)
            ntest = round(len(group) * test_fraction)
            test.extend(group[:ntest])
            rest = group[ntest:]
            development.extend(rest)
            if cv_folds:
                for i, patient in enumerate(rest):
                    buckets[i % cv_folds].append(patient)
            else:
                nval = max(1, round(len(rest) * validation_fraction)) if len(rest) >= 2 else 0
                buckets[0].extend(rest[nval:])
                buckets[1].extend(rest[:nval])
        if test_fraction and not test:
            raise ValueError('Too few patients for requested test fraction; supply explicit splits')
        if cv_folds:
            folds = [{'train': sorted(set(development)-set(val)), 'validation': sorted(val)} for val in buckets]
        else:
            folds = [{'train': sorted(buckets[0]), 'validation': sorted(buckets[1])}]
    for name, values in [('test', test)] + [(key, f[key]) for f in folds for key in ('train', 'validation')]:
        if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values) or len(values) != len(set(values)):
            raise ValueError(f'{name} must contain unique nonempty patient IDs')
    validate_splits(rows, folds, test)
    return {'schema_version': 1, 'folds': folds, 'test': sorted(test)}


def select(rows, patients):
    patients = set(patients)
    return [row for row in rows if row['patient_id'] in patients]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

