import csv
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from framework.data import read_records, resolve_splits, validate_splits
from framework.cli import main


def records():
    return [{'sample_id':f's-{label}-{i}-{scan}','patient_id':f'p-{label}-{i}','label':label,'nifti_path':f'{label}-{i}-{scan}.nii.gz'} for label in ('custom-a','custom-b','custom-c') for i in range(10) for scan in range(2)]


def test_generated_cv_is_patient_safe_deterministic_and_exhaustive():
    rows = records()
    result = resolve_splits(rows,seed=3,cv_folds=3,test_fraction=.2)
    assert result == resolve_splits(rows,seed=3,cv_folds=3,test_fraction=.2)
    validate_splits(rows,result['folds'],result['test'])
    assert len(result['test']) == 6
    all_validation = [p for f in result['folds'] for p in f['validation']]
    assert len(all_validation) == len(set(all_validation)) == 24
    for fold in result['folds']:
        assert not set(fold['train']) & set(fold['validation'])
        assert not (set(fold['train']) | set(fold['validation'])) & set(result['test'])


def test_patient_leakage_rejected():
    rows = records()
    split = resolve_splits(rows)
    split['folds'][0]['validation'].append(split['folds'][0]['train'][0])
    with pytest.raises(ValueError,match='leakage'):
        validate_splits(rows,split['folds'],split['test'])


def test_explicit_splits_reject_unknown_missing_and_duplicate_patients(tmp_path):
    rows = records()
    obj = resolve_splits(rows)
    obj['folds'][0]['train'].append('unknown')
    file = tmp_path/'splits.json'
    file.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Every patient'):
        resolve_splits(rows,manifest=file)
    obj = resolve_splits(rows)
    obj['folds'][0]['train'].append(obj['folds'][0]['train'][0])
    file.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='unique'):
        resolve_splits(rows,manifest=file)


def test_manifest_resolves_relative_paths_and_rejects_duplicate_images(tmp_path):
    rows = records()[:2]
    file = tmp_path/'samples.json'
    file.write_text(json.dumps(rows))
    loaded = read_records(file,check_files=False)
    assert Path(loaded[0]['nifti_path']).parent == tmp_path
    rows[1]['nifti_path'] = rows[0]['nifti_path']
    file.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Duplicate'):
        read_records(file,check_files=False)


def test_dry_run_is_lightweight_and_supports_custom_three_classes(tmp_path,capsys):
    file = tmp_path/'samples.json'
    file.write_text(json.dumps(records()))
    result = main(['train','--model','DenseNet','--method','scratch','--dataset',str(file),'--output',str(tmp_path/'output'),'--cv-folds','3','--dry-run'])
    assert result == 0
    assert not (tmp_path/'output').exists()
    assert json.loads(capsys.readouterr().out)['samples'] == 60


def test_unsupported_combination_fails_before_importing_model(tmp_path,capsys):
    file = tmp_path/'samples.json'
    file.write_text(json.dumps(records()))
    result = main(['train','--model','FlexiCT','--method','scratch','--dataset',str(file),'--output',str(tmp_path/'output'),'--dry-run'])
    assert result == 2
    assert 'Unsupported model/method' in capsys.readouterr().err


def test_test_dry_run_checks_checkpoint_identity_without_torch(tmp_path,capsys):
    from framework.data import digest
    checkpoint=tmp_path/'example.pt'
    checkpoint.write_bytes(b'synthetic metadata-only fixture')
    sidecar=Path(str(checkpoint)+'.metadata.json')
    metadata={'model_family':'DenseNet','method':'scratch','classes':['a','b'],'checkpoint_sha256':digest(checkpoint)}
    sidecar.write_text(json.dumps(metadata))
    samples=tmp_path/'test.json'
    samples.write_text(json.dumps([{'sample_id':'new','patient_id':'new-patient','nifti_path':'new.nii.gz'}]))
    argv=['test','--model','DenseNet','--checkpoint',str(checkpoint),'--dataset',str(samples),'--output',str(tmp_path/'out'),'--dry-run']
    assert main(argv)==0
    assert not (tmp_path/'out').exists()
    metadata['model_family']='MedicalNet'
    sidecar.write_text(json.dumps(metadata))
    assert main(argv)==2
    assert 'model_family differs' in capsys.readouterr().err


def test_bad_declared_class_order_and_null_labels_rejected(tmp_path):
    rows=records()
    samples=tmp_path/'data.json'
    samples.write_text(json.dumps(rows))
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'classes':['custom-a','custom-b']}))
    assert main(['train','--model','DenseNet','--method','scratch','--dataset',str(samples),'--config',str(config),'--output',str(tmp_path/'out'),'--dry-run'])==2
    rows[0]['label']=None
    samples.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='nonempty'):
        read_records(samples,check_files=False)


@pytest.mark.parametrize('alias,canonical',[('3DINO','ThreeDINO'),('threedino','ThreeDINO'),('Pillar-0','Pillar0'),('pillar0','Pillar0'),('densenet','DenseNet'),('COLIPRI','COLIPRI')])
def test_public_model_aliases_normalize_before_cli_choice_validation(alias,canonical):
    from framework.cli import parser
    args=parser().parse_args(['test','--model',alias,'--checkpoint','model.pt','--dataset','samples.csv','--output','results'])
    assert args.model==canonical
