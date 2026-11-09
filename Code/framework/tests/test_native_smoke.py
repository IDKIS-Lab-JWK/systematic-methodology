"""Native CPU smoke: real 3D DenseNet, NIfTI, optimizer, portable checkpoint and public inference."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
torch = pytest.importorskip('torch')
nib = pytest.importorskip('nibabel')
np = pytest.importorskip('numpy')
pytest.importorskip('monai')
from framework.cli import main


def test_native_densenet_training_and_public_independent_test(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        rows = []
        for i in range(6):
            path = tmp_path/f'synthetic_{i}.nii.gz'
            image = nib.Nifti1Image(np.random.default_rng(i).normal(0,100,(24,24,24)).astype(np.float32),np.eye(4))
            image.header.set_xyzt_units('mm')
            nib.save(image,path)
            rows.append({'sample_id':f's{i}','patient_id':f'p{i}','label':['alpha','beta'][i%2],'nifti_path':str(path)})
        samples,split = tmp_path/'samples.json',tmp_path/'split.json'
        samples.write_text(json.dumps(rows[:4]))
        split.write_text(json.dumps({'train':['p0','p1'],'validation':['p2','p3'],'test':[]}))
        output = tmp_path/'training'
        assert main(['train','--model','DenseNet','--method','scratch','--dataset',str(samples),'--splits',str(split),'--preset','smoke','--device','cpu','--output',str(output)]) == 0
        checkpoint = output/'fold_1/best.pt'
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert payload['classes'] == ['alpha','beta']
        assert 'training_patients' not in payload
        assert payload['config']['preprocessing_plan']['training_patient_count'] == 2
        independent = tmp_path/'independent.json'
        independent.write_text(json.dumps(rows[4:]))
        test_output = tmp_path/'test'
        assert main(['test','--model','DenseNet','--checkpoint',str(checkpoint),'--dataset',str(independent),'--device','cpu','--output',str(test_output)]) == 0
        assert (test_output/'predictions.csv').exists()
        assert json.loads((test_output/'provenance.json').read_text())['persistent_test_cache'] is False
        assert not list(tmp_path.rglob('*cache*'))
        # Reusing development patients is rejected for the new portable format.
        assert main(['test','--model','DenseNet','--checkpoint',str(checkpoint),'--dataset',str(samples),'--device','cpu','--output',str(tmp_path/'leaky')]) == 2
    finally:
        torch.set_num_threads(previous_threads)


def test_medicalnet_custom_head_and_frozen_partial_scope(tmp_path):
    from framework.runtime import activate_models,build_runtime,set_training_mode
    from framework.registry import PRESETS
    activate_models()
    from MedicalNet.model import MedicalNetClassifier
    base = MedicalNetClassifier(classes=3)
    pretrained = tmp_path/'synthetic_backbone.pth'
    torch.save({'state_dict':{k:v for k,v in base.state_dict().items() if not k.startswith('fc.')}},pretrained)
    del base
    for method in ('frozen','partial','full'):
        config={'model':'MedicalNet','method':method,'parameters':dict(PRESETS['smoke']),'assets':{'checkpoint':str(pretrained)},'device':'cpu','batch_size':1,'preprocessing_plan':{}}
        model,runtime=build_runtime(config,['a','b','c'])
        assert model.fc.out_features==3
        trainable=[name for name,p in model.named_parameters() if p.requires_grad]
        if method=='frozen':
            assert set(trainable)=={'fc.weight','fc.bias'}
        elif method=='partial':
            assert all(name.startswith(('layer4.','fc.')) for name in trainable)
            assert any(name.startswith('layer4.') for name in trainable)
        else:
            assert all(p.requires_grad for p in model.parameters())
        set_training_mode(model,runtime,True)
        assert model.layer1.training==(method=='full')
        assert model.layer4.training==(method!='frozen')
        assert model.fc.training
        del model,runtime
