"""Dependency-light command-line entry point for new datasets and direct testing."""
import argparse
import csv
import json
import sys
from pathlib import Path
from .data import read_records, resolve_splits, digest
from .registry import CAPABILITIES, PRESETS, validate_config, canonical_model


def parser():
    root = argparse.ArgumentParser(description='Reusable CT classification; no fixed cohort fractions.')
    sub = root.add_subparsers(dest='command',required=True)
    sub.add_parser('models',help='List supported training methods')
    for command in ('train','test','split'):
        p = sub.add_parser(command)
        p.add_argument('--dataset',required=True,help='CSV or JSON samples manifest')
        p.add_argument('--dataset-root',help='Base directory for relative image paths (default manifest directory)')
        p.add_argument('--output',required=True,help='New output directory (split: JSON file)')
        p.add_argument('--dry-run',action='store_true',help='Validate configuration/manifests without model dependencies or image reads')
        if command != 'test':
            p.add_argument('--splits',help='JSON patient partitions')
            p.add_argument('--cv-folds',type=int,default=0)
            p.add_argument('--validation-fraction',type=float,default=.2)
            p.add_argument('--test-fraction',type=float,default=0)
            p.add_argument('--seed',type=int,default=0)
        if command != 'split':
            p.add_argument('--model',required=True,type=canonical_model,choices=sorted(CAPABILITIES))
            p.add_argument('--config',help='JSON with parameters/assets/classes (train), asset overrides (test)')
            p.add_argument('--device',default='auto')
            p.add_argument('--precision',choices=['auto','fp32','fp16','bf16'],default=None,help='Training defaults auto; testing defaults saved checkpoint precision')
            p.add_argument('--batch-size',type=int,default=1)
        if command == 'train':
            p.add_argument('--method',required=True,choices=['frozen','partial','full','scratch'])
            p.add_argument('--preset',choices=sorted(PRESETS),default='default')
            p.add_argument('--set',action='append',default=[],metavar='KEY=JSON',help='Override parameter; repeatable')
        if command == 'test':
            p.add_argument('--checkpoint',action='append',required=True)
            p.add_argument('--metadata',action='append',help='One metadata path per checkpoint')
            p.add_argument('--allow-trusted-pickle',action='store_true',help='Explicit trust opt-in for legacy sklearn joblib artifacts')
    return root


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def output_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Output must be absent or an empty directory: {path}')
    path.mkdir(parents=True,exist_ok=True)
    return path


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'models':
            print(json.dumps(CAPABILITIES,indent=2))
            return 0
        rows = read_records(args.dataset,root=args.dataset_root,labeled=args.command!='test',check_files=not args.dry_run)
        if args.command != 'test':
            if args.splits and (args.cv_folds or args.test_fraction or args.validation_fraction != .2):
                raise ValueError('--splits cannot be combined with generated-split controls')
            splits = resolve_splits(rows,manifest=args.splits,seed=args.seed,cv_folds=args.cv_folds,validation_fraction=args.validation_fraction,test_fraction=args.test_fraction)
        if args.command == 'split':
            if args.dry_run:
                print(json.dumps(splits,indent=2))
            else:
                path = Path(args.output)
                if path.exists(): raise ValueError('Refusing to overwrite split manifest')
                path.parent.mkdir(parents=True,exist_ok=True)
                json_write(path,splits)
            return 0
        supplied = json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {}
        if not isinstance(supplied,dict): raise ValueError('Configuration must be a JSON object')
        if args.command == 'train':
            if not isinstance(supplied.get('parameters',{}),dict) or not isinstance(supplied.get('assets',{}),dict):
                raise ValueError('parameters and assets must be JSON objects')
            unknown = set(supplied)-{'parameters','assets','classes'}
            if unknown: raise ValueError(f'Unknown config blocks: {sorted(unknown)}')
            parameters = {**PRESETS[args.preset],**supplied.get('parameters',{})}
            if args.model == 'ThreeDINO' and args.method == 'partial' and 'weight_decay' not in supplied.get('parameters',{}):
                parameters['weight_decay'] = 0.0
            for item in args.set:
                key,sep,value = item.partition('=')
                if not sep: raise ValueError('--set must be KEY=JSON')
                parameters[key] = json.loads(value)
            config = {'model':args.model,'method':args.method,'parameters':parameters,'assets':supplied.get('assets',{}),'device':args.device,'precision':args.precision or 'auto','seed':args.seed,'batch_size':args.batch_size,'preset':args.preset}
            if 'classes' in supplied: config['classes'] = supplied['classes']
            # Asset paths in a config are relative to that config, not a private workspace.
            for key in ('checkpoint','upstream_root','rave_root'):
                if config['assets'].get(key):
                    value = Path(config['assets'][key])
                    if not value.is_absolute():
                        base = Path(args.config).resolve().parent if args.config else Path.cwd()
                        config['assets'][key] = str((base/value).resolve())
            validate_config(config)
            development_patients = set(p for fold in splits['folds'] for p in fold['train']+fold['validation'])
            development_labels = {r['label'] for r in rows if r['patient_id'] in development_patients}
            classes = config.get('classes',sorted(development_labels))
            if not isinstance(classes,list) or any(not isinstance(c,str) or not c for c in classes) or len(classes) != len(set(classes)) or set(classes) != development_labels:
                raise ValueError('classes must be unique strings matching the development label vocabulary')
            config['classes'] = classes
            for fold in splits['folds']:
                if {r['label'] for r in rows if r['patient_id'] in set(fold['train'])} != set(classes):
                    raise ValueError('Every training fold must contain every development class')
            if args.dry_run:
                print(json.dumps({'resolved_config':config,'splits':splits,'samples':len(rows)},indent=2))
                return 0
            output = output_directory(args.output)
            json_write(output/'input_manifest.json',{'samples':rows})
            from .training import train
            train(config,rows,splits,output)
            return 0
        if args.metadata and len(args.metadata) != len(args.checkpoint):
            raise ValueError('--metadata requires one path per checkpoint')
        if args.batch_size != 1:
            raise ValueError('Direct testing streams one image at a time; batch-size must be 1')
        if 'assets' in supplied:
            if set(supplied) != {'assets'}: raise ValueError('Test config accepts only the assets block')
            supplied = supplied['assets']
        if not isinstance(supplied,dict): raise ValueError('Test assets must be an object')
        for key in ('checkpoint','checkpoint_path','upstream_root','rave_root'):
            if supplied.get(key):
                path = Path(supplied[key])
                if not path.is_absolute(): supplied[key] = str((Path(args.config).resolve().parent/path).resolve())
        if args.precision is not None: supplied['precision'] = args.precision
        if args.dry_run:
            validate_test_sidecars(args)
            print(json.dumps({'model':args.model,'checkpoints':args.checkpoint,'samples':len(rows),'asset_overrides':supplied,'cache_policy':'none'},indent=2))
            return 0
        output = output_directory(args.output)
        resolved = {k:v for k,v in vars(args).items() if k != 'allow_trusted_pickle'}
        resolved['asset_overrides'] = supplied
        json_write(output/'resolved_config.json',resolved)
        from inference import predict
        result = predict(args.checkpoint,rows,model_family=args.model,device=args.device,metadata_paths=args.metadata,model_overrides=supplied,allow_trusted_pickle=args.allow_trusted_pickle)
        probabilities = result.probabilities.tolist() if hasattr(result.probabilities,'tolist') else result.probabilities
        classes = list(result.classes)
        with (output/'predictions.csv').open('w',encoding='utf-8',newline='') as handle:
            fields = ['sample_id','patient_id','label','prediction'] + [f'probability::{c}' for c in classes]
            writer = csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for row,values in zip(rows,probabilities):
                prediction = classes[max(range(len(values)),key=values.__getitem__)]
                writer.writerow({'sample_id':row['sample_id'],'patient_id':row['patient_id'],'label':row.get('label',''),'prediction':prediction,**{f'probability::{c}':v for c,v in zip(classes,values)}})
        json_write(output/'provenance.json',result.provenance)
        if all(r.get('label') for r in rows):
            if any(r['label'] not in classes for r in rows):
                raise ValueError('Test label is absent from checkpoint class order')
            from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
            truth = [classes.index(r['label']) for r in rows]
            predictions = [max(range(len(v)),key=v.__getitem__) for v in probabilities]
            json_write(output/'metrics.json',{'accuracy':accuracy_score(truth,predictions),'balanced_accuracy':balanced_accuracy_score(truth,predictions),'macro_f1':f1_score(truth,predictions,labels=list(range(len(classes))),average='macro',zero_division=0),'confusion_matrix':confusion_matrix(truth,predictions,labels=list(range(len(classes)))).tolist(),'classes':classes})
        return 0
    except (ValueError,FileNotFoundError,ImportError,RuntimeError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2


def validate_test_sidecars(args):
    """Dry-run verifies portable identity without importing torch or loading tensors."""
    if len(args.checkpoint) not in (1,5):
        raise ValueError('Testing requires one or five checkpoints')
    paths = [Path(p).resolve() for p in args.checkpoint]
    if len(set(paths)) != len(paths):
        raise ValueError('Ensemble checkpoint paths must be distinct')
    identities = []
    for i,path in enumerate(paths):
        if not path.is_file(): raise FileNotFoundError(path)
        sidecar = Path(args.metadata[i]) if args.metadata else Path(str(path)+'.metadata.json')
        if not sidecar.is_file():
            raise ValueError(f'Lightweight test dry-run requires metadata sidecar: {sidecar}')
        metadata = json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if metadata.get('model_family') != args.model:
            raise ValueError('Checkpoint metadata model_family differs from --model')
        classes,method = metadata.get('classes'),metadata.get('method')
        if not isinstance(classes,list) or len(classes)<2 or len(set(classes)) != len(classes) or not method:
            raise ValueError('Checkpoint metadata requires unique ordered classes and method')
        if not metadata.get('checkpoint_sha256') or digest(path) != metadata['checkpoint_sha256']:
            raise ValueError('Checkpoint metadata SHA-256 mismatch')
        identities.append((classes,method))
    if any(item != identities[0] for item in identities):
        raise ValueError('Ensemble metadata classes/method differ')


