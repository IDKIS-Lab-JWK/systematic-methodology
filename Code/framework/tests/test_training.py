"""Actual optimizer/checkpoint/prediction integration on a tiny CPU model."""
import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
torch = pytest.importorskip('torch')
from framework import training
from framework.registry import PRESETS


def test_actual_train_checkpoint_roundtrip_custom_classes_and_no_test_read(tmp_path,monkeypatch):
    classes = ['first','second','third']
    rows = []
    for i in range(9):
        path = tmp_path/f'image_{i}.nii'
        path.write_bytes(bytes([i]))
        rows.append({'sample_id':f's{i}','patient_id':f'p{i}','label':classes[i%3],'nifti_path':str(path)})
    test_row = {'sample_id':'unread','patient_id':'test-patient','label':'first','nifti_path':str(tmp_path/'does-not-exist.nii')}
    rows.append(test_row)
    splits = {'folds':[{'train':[f'p{i}' for i in range(6)],'validation':['p6','p7','p8']}],'test':['test-patient']}
    initial = {}
    def factory(config,labels,train_rows=None):
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(8,len(labels)))
        initial['state'] = copy.deepcopy(model.state_dict())
        return model,{}
    seen = []
    def preprocess(family,path,config,runtime):
        seen.append(str(path))
        i = int(Path(path).stem.split('_')[-1])
        result = torch.zeros(1,2,2,2)
        result.flatten()[i%3] = 1
        return result
    monkeypatch.setattr(training,'build_runtime',factory)
    monkeypatch.setattr(training,'preprocess_one',preprocess)
    config = {'model':'DenseNet','method':'scratch','parameters':dict(PRESETS['smoke'],epochs=2,learning_rate=.05),'assets':{},'device':'cpu','precision':'fp32','seed':4,'batch_size':2}
    output = tmp_path/'run'
    output.mkdir()
    result = training.train(config,rows,splits,output)
    checkpoint = Path(result[0]['checkpoint'])
    payload = torch.load(checkpoint,weights_only=True)
    assert payload['classes'] == classes
    assert payload['state_dict']['1.weight'].shape[0] == 3
    assert not torch.equal(payload['state_dict']['1.weight'],initial['state']['1.weight'])
    assert str(tmp_path/'does-not-exist.nii') not in seen
    assert (output/'provenance.json').exists()
    assert Path(str(checkpoint)+'.metadata.json').exists()
    prediction = training.predict_checkpoint(checkpoint,[dict(rows[0],patient_id='new-patient')],device='cpu')
    assert len(prediction['probabilities'][0]) == 3
    assert abs(sum(prediction['probabilities'][0])-1) < 1e-6
    payload['model_family'] = 'MedicalNet'
    malformed = tmp_path/'wrong.pt'
    torch.save(payload,malformed)
    with pytest.raises(ValueError,match='metadata disagree'):
        training.predict_checkpoint(malformed,[rows[0]],device='cpu')
