from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference import predict, InferenceError
from inference.direct import sha256, activate_models
from inference import legacy


def checkpoint(tmp_path, number=0, classes=None):
    path = tmp_path / f'model{number}.pt'
    torch.save({'model_family':'MedicalNet', 'method':'FullFineTune',
                'classes':classes or ['positive','negative'], 'number':number},path)
    Path(str(path)+'.metadata.json').write_text(json.dumps({'model_family':'MedicalNet', 'classes':classes or ['positive','negative'], 'method':'FullFineTune', 'checkpoint_sha256':sha256(path)}))
    return path


def record(tmp_path):
    path = tmp_path/'source.nii.gz'
    path.write_bytes(b'synthetic source; adapter is mocked')
    return {'sample_id':'scan-01','patient_id':'patient-01','image_path':str(path),'label':'positive'}


def test_five_fold_mean_reads_sources_and_creates_no_cache(tmp_path,monkeypatch):
    paths=[checkpoint(tmp_path,i) for i in range(5)]
    row=record(tmp_path)
    observed=[]
    def adapter(family, method, payload, rows, classes, runtime, device):
        assert Path(rows[0]['image_path']).read_bytes().startswith(b'synthetic')
        assert rows[0]['label']=='positive'
        assert rows[0]['patient_id']=='patient-01'
        observed.append(payload['number'])
        return [[0.1*(payload['number']+1),1-0.1*(payload['number']+1)]]
    monkeypatch.setattr(legacy,'predict_payload',adapter)
    before=set(tmp_path.rglob('*'))
    result=predict(paths,[row],model_family='MedicalNet',device='cpu')
    np.testing.assert_allclose(result.probabilities,[[.3,.7]])
    assert result.classes==['positive','negative'] and result.sample_ids==['scan-01']
    assert observed==list(range(5))
    assert result.provenance['persistent_test_cache'] is False
    assert set(tmp_path.rglob('*'))==before


def test_ensemble_class_order_is_not_silently_reordered(tmp_path,monkeypatch):
    paths=[checkpoint(tmp_path,i, ['negative','positive'] if i==4 else None) for i in range(5)]
    monkeypatch.setattr(legacy,'predict_payload',lambda *a:[[.2,.8]])
    with pytest.raises(InferenceError,match='class order'):
        predict(paths,[record(tmp_path)],model_family='MedicalNet',device='cpu')


def test_checkpoint_digest_and_family_fail_before_model_loading(tmp_path,monkeypatch):
    path=checkpoint(tmp_path)
    monkeypatch.setattr(legacy,'predict_payload',lambda *a:pytest.fail('must fail before model load'))
    with pytest.raises(InferenceError,match='family'):
        predict([path],[record(tmp_path)],model_family='DenseNet')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(InferenceError,match='SHA-256'):
        predict([path],[record(tmp_path)],model_family='MedicalNet')


def test_nested_backbone_identity_cannot_be_overridden(tmp_path):
    asset=tmp_path/'asset.pt';asset.write_bytes(b'wrong frozen encoder')
    expected='a'*64
    payload={'model_identity':{'pretrained_model':{'checkpoint':{'sha256':expected}}}}
    assert legacy._find_hash(payload)==expected
    with pytest.raises(InferenceError,match='override differs'):
        legacy._asset_checkpoint(payload,{'checkpoint_path':str(asset),'checkpoint_sha256':sha256(asset)})


def test_metadata_extraction_precision_is_preserved():
    assert legacy._precision({'model_identity':{'extraction_precision':'bf16'}},{})=='bf16'
    assert legacy._precision({'model_identity':{'pretrained_model':{'runtime':{'training_precision_resolved':'bf16'}}}},{})=='bf16'
    assert legacy._precision({'parameters':{'mixed_precision':True,'autocast_dtype':'float16'}},{})=='float16'


def test_reject_partial_or_nonfinite_probability_output(tmp_path,monkeypatch):
    monkeypatch.setattr(legacy,'predict_payload',lambda *a:[[float('nan'),0]])
    with pytest.raises(InferenceError,match='non-finite'):
        predict([checkpoint(tmp_path)],[record(tmp_path)],model_family='MedicalNet',device='cpu')


def test_trainable_state_requires_exact_keys():
    activate_models()
    module=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3))
    from _supervised_adaptation.runtime import trainable_state_dict
    state=trainable_state_dict(module)
    clone=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3))
    legacy._restore(clone,{'trainable_state_dict':state})
    for name,value in module.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[name],value)
    state.pop('0.weight')
    with pytest.raises(RuntimeError,match='architecture mismatch'):
        legacy._restore(clone,{'trainable_state_dict':state})


def test_pillar_postprocessor_keeps_hu_padding_and_depth_direction():
    from inference.preprocessing import pillar_model_input
    array=np.array([[[10,11],[12,13]],[[20,21],[22,23]]],dtype=np.int16)
    got=pillar_model_input(array,target=(4,2,2))
    assert tuple(got.shape)==(1,4,2,2)
    torch.testing.assert_close(got[0,1],torch.tensor([[20.,21.],[22.,23.]]))
    torch.testing.assert_close(got[0,2],torch.tensor([[10.,11.],[12.,13.]]))
    assert torch.all(got[:,0]==-1) and torch.all(got[:,-1]==-1)


def test_cnn_direct_uses_saved_plan_and_never_volume_cache(tmp_path,monkeypatch):
    activate_models()
    nib=pytest.importorskip('nibabel')
    from _cnn.geometry import fit_plan
    from _cnn.data import VolumeCache
    from MedicalNet.model import MedicalNetClassifier
    from _supervised_adaptation.runtime import trainable_state_dict
    image=tmp_path/'image.nii.gz'
    nii=nib.Nifti1Image(np.linspace(-1000,1000,8**3,dtype=np.float32).reshape(8,8,8),np.eye(4))
    nii.header.set_xyzt_units('mm')
    nib.save(nii,image)
    plan=fit_plan([{'image_path':str(image),'patient_id':'synthetic','class_name':'a','label':0}],[32,32,32])
    model=MedicalNetClassifier(classes=2)
    payload={'parameters':{},'trainable_state_dict':trainable_state_dict(model)}
    monkeypatch.setattr(VolumeCache,'get',lambda *a:pytest.fail('persistent cache forbidden'))
    restored,preprocess,_=legacy._cnn('MedicalNet',payload,['a','b'],{'preprocessing_plan':plan},'cpu')
    before=set(tmp_path.rglob('*'))
    tensor=preprocess(image)
    assert tensor.shape==(1,32,32,32)
    torch.testing.assert_close(restored.fc.weight,model.fc.weight)
    restored.eval()
    with torch.inference_mode():
        logits=restored(tensor[None])
    assert logits.shape==(1,2) and torch.isfinite(logits).all()
    assert set(tmp_path.rglob('*'))==before
    with pytest.raises(InferenceError,match='fold-fitted'):
        legacy._cnn('MedicalNet',payload,['a','b'],{},'cpu')



def test_pillar_raw_transform_matches_rave_then_rate_loader(tmp_path,monkeypatch):
    import os
    root=os.environ.get('RAVE_TEST_ROOT')
    if not root:
        pytest.skip('Set RAVE_TEST_ROOT to the audited external RAVE checkout')
    sitk=pytest.importorskip('SimpleITK')
    from inference.preprocessing import preprocess_direct, _pillar_processor
    activate_models()
    from Pillar0 import cache
    array=np.linspace(-999.9,777.9,7*9*11,dtype=np.float32).reshape(7,9,11)
    image=sitk.GetImageFromArray(array)
    image.SetSpacing((1.1,1.7,2.3))
    image.SetDirection((-1.,0.,0.,0.,-1.,0.,0.,0.,1.))
    path=tmp_path/'source.nii.gz'
    sitk.WriteImage(image,str(path))
    before=set(tmp_path.rglob('*'))
    direct=preprocess_direct('Pillar0',path,{'assets':{'rave_root':root}})
    reread=sitk.ReadImage(str(path))
    raw=_pillar_processor(root)._process_volume(reread,np.array(reread.GetSpacing()))
    monkeypatch.setattr(cache,'load_cached_volume',lambda _:torch.from_numpy(raw).float().unsqueeze(0))
    expected=cache.load_pillar_volume(Path('unused-no-cache'))
    torch.testing.assert_close(direct,expected,rtol=0,atol=0)
    assert direct.shape==(1,384,384,384)
    assert set(tmp_path.rglob('*'))==before

def test_threedino_selected_head_representation_and_no_dataset_cache(tmp_path,monkeypatch):
    activate_models()
    import ThreeDINO.model as m
    import ThreeDINO.data as d
    asset=tmp_path/'encoder.pt';asset.write_bytes(b'pinned')
    head=torch.nn.Linear(3,2)
    head.weight.data.copy_(torch.tensor([[1.,0.,0.],[0.,1.,0.]]));head.bias.data.zero_()
    captured=[]
    monkeypatch.setattr(m,'feature_dimension',lambda n,mean:3)
    monkeypatch.setattr(m,'load_official_model',lambda **kwargs:torch.nn.Identity())
    def bank(encoder,images,representations):
        captured.append(representations)
        return {(4,True):torch.tensor([[2.,1.,0.]])}
    monkeypatch.setattr(m,'extract_representation_bank',bank)
    def dataset(rows,**kwargs):
        assert kwargs['cache_dir'] is None and kwargs['split']=='test'
        return [{'image':torch.ones(1,2,2,2)}]
    monkeypatch.setattr(d,'make_dataset',dataset)
    payload={'method':'linear_probe','candidate':{'n_last_blocks':4,'use_mean_patch_token':True},'head_state_dict':head.state_dict(), 'source_provenance':{'checkpoint_sha256':sha256(asset)}}
    model,preprocess,_=legacy._threedino(payload,['a','b'],{'checkpoint_path':str(asset),'upstream_root':str(tmp_path)},'cpu')
    torch.testing.assert_close(model(preprocess('source')[None]),torch.tensor([[2.,1.]]))
    assert captured==[[(4,True)]]


def test_colipri_restores_selected_pooling_and_processes_raw_image(tmp_path,monkeypatch):
    activate_models()
    import COLIPRI
    import COLIPRI.model as m
    from COLIPRI.pooling import COLIPRIPoolingOptimisedLinearProbe
    asset=tmp_path/'encoder.pt';asset.write_bytes(b'pinned')
    monkeypatch.setattr(COLIPRI,'MODEL_SHA256',sha256(asset))
    class Encoder(torch.nn.Module):
        def encode_image(self,images,**kwargs):
            assert kwargs=={'pool':False,'project':False,'normalize':False}
            return torch.arange(32,dtype=torch.float32).reshape(1,4,2,2,2)
    processor=object()
    monkeypatch.setattr(m,'build_image_encoder',lambda **kwargs:(Encoder(),processor,{}))
    def preprocess(processor_arg,path):
        assert processor_arg is processor and path=='source'
        return torch.ones(1,1,2,2,2),{}
    monkeypatch.setattr(m,'preprocess_nifti',preprocess)
    head=COLIPRIPoolingOptimisedLinearProbe(pooling='max',embed_dim=4,num_heads=1,num_classes=2)
    payload={'architecture':head.architecture(),'state_dict':head.state_dict()}
    model,transform,_=legacy._colipri('poolingoptimisedlinearprobe',payload,['a','b'],{'checkpoint_path':str(asset),'upstream_root':str(tmp_path)},'cpu')
    dense=Encoder().encode_image(None,pool=False,project=False,normalize=False)
    torch.testing.assert_close(model(transform('source')[None]),head(dense))


def test_flexict_pipeline_retains_scaler_and_probability_class_order(tmp_path,monkeypatch):
    activate_models()
    import FlexiCT.model as m
    import FlexiCT.data as d
    asset=tmp_path/'encoder.pt';asset.write_bytes(b'pinned')
    class Encoder(torch.nn.Module):
        def forward(self,images):
            assert images.ndim==5
            return torch.tensor([[.6,.8]])
    class Pipeline:
        classes_=np.array(['negative','positive'])
        def predict_proba(self,features):
            np.testing.assert_allclose(features,[[.6,.8]],atol=1e-6)
            return np.array([[.25,.75]])
    monkeypatch.setattr(m,'build_vision_only_encoder',lambda **kwargs:(Encoder(),{}))
    monkeypatch.setattr(d,'preprocess_nifti',lambda path:(torch.ones(1,2,2,2),{}))
    payload={'sklearn_pipeline':Pipeline(),'model_identity':{'checkpoint':{'sha256':sha256(asset)}}}
    _,transform,probability_fn=legacy._flexict('linearprobe',payload,['positive','negative'],{'checkpoint_path':str(asset),'upstream_root':str(tmp_path)},'cpu')
    np.testing.assert_allclose(probability_fn(transform('source')[None]),[[.75,.25]])


def test_legacy_test_stage_rejected_before_assets():
    activate_models()
    from types import SimpleNamespace
    import argparse
    from _supervised_adaptation.cli import validate_args
    with pytest.raises(SystemExit) as error:
        validate_args(argparse.ArgumentParser(),SimpleNamespace(stage='test',dry_run=False))
    assert error.value.code==2

def test_pillar_probe_restores_head_without_feature_store(tmp_path,monkeypatch):
    activate_models()
    import Pillar0.model as m
    import inference.pillar as p
    monkeypatch.setattr(m,'FEATURE_DIM',3)
    monkeypatch.setattr(m,'load_official_model',lambda **kwargs:torch.nn.Identity())
    monkeypatch.setattr(m,'extract_vision_features',lambda model,images:torch.tensor([[1.,2.,3.]]))
    monkeypatch.setattr(p,'preprocess_nifti',lambda path,runtime:torch.ones(1,2,2,2))
    head=torch.nn.Linear(3,2)
    payload={'head_state_dict':head.state_dict()}
    model,transform,_=legacy._pillar('linearprobe',payload,['a','b'],{},'cpu')
    torch.testing.assert_close(model(transform('source')[None]),head(torch.tensor([[1.,2.,3.]])))


def test_merlin_probe_restores_head_and_verifies_frozen_asset(tmp_path,monkeypatch):
    import types,inspect
    from framework import runtime
    asset=tmp_path/'checkpoints'/'encoder.pt';asset.parent.mkdir();asset.write_bytes(b'pinned')
    placements = []
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.classifier=torch.nn.Linear(3,2)
        def to(self, device):
            placements.append(str(device))
            # The fake encoder stays on CPU so placement is tested without a GPU.
            return super().to('cpu')
        def forward(self,images):
            return self.classifier(torch.tensor([[1.,2.,3.]]))
    head=torch.nn.Linear(3,2)
    fake_model=types.ModuleType('Merlin.model')
    fake_model.build_model=lambda *a:(Classifier(),{})
    monkeypatch.setitem(sys.modules,'Merlin.model',fake_model)
    fake_loader=types.ModuleType('merlin.models.load')
    fake_loader.Merlin=Classifier
    fake_loader.MODEL_CONFIGS={'default':{'checkpoint':'encoder.pt'}}
    monkeypatch.setitem(sys.modules,'merlin.models.load',fake_loader)
    monkeypatch.setattr(inspect,'getfile',lambda obj:str(tmp_path/'load.py'))
    monkeypatch.setattr(runtime,'preprocess_one',lambda *a:torch.ones(1,2,2,2))
    payload={'parameters':{},'classifier_state_dict':head.state_dict(),
             'model_details':{'frozen_encoder_provenance':{'checkpoint_sha256':sha256(asset)}}}
    model,transform,_=legacy._merlin('linearprobe',payload,['a','b'],{},'cuda:0')
    assert placements == ['cuda:0']
    assert model.training is False
    torch.testing.assert_close(model(transform('source')[None]),head(torch.tensor([[1.,2.,3.]])))
    asset.write_bytes(b'different')
    with pytest.raises(InferenceError,match='differs from'):
        legacy._merlin('linearprobe',payload,['a','b'],{},'cpu')

def test_legacy_colipri_method_alias_is_explicit():
    from inference.direct import canonical_method
    assert canonical_method('COLIPRI','LinearProbe')==canonical_method('COLIPRI','PoolingOptimisedLinearProbe')
    assert canonical_method('FlexiCT','LinearProbe')!=canonical_method('FlexiCT','PoolingOptimisedLinearProbe')


def test_merlin_raw_matches_official_uncached_dataset(tmp_path):
    import importlib.metadata
    try:
        importlib.metadata.version('merlin-vlm')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('Requires the separate Merlin model environment')
    nib=pytest.importorskip('nibabel')
    from inference.preprocessing import preprocess_direct
    from merlin.data.dataloaders import CTPersistentDataset
    from merlin.data.monai_transforms import ImageTransforms
    path=tmp_path/'synthetic.nii.gz'
    nii=nib.Nifti1Image(np.arange(7*9*11,dtype=np.float32).reshape(7,9,11),np.diag([1.2,1.8,3.1,1.]))
    nib.save(nii,path)
    before=set(tmp_path.rglob('*'))
    observed=preprocess_direct('Merlin',path,{})
    expected=CTPersistentDataset([{'image':str(path)}],ImageTransforms,cache_dir=None)[0]['image']
    torch.testing.assert_close(observed,expected,rtol=0,atol=0)
    assert observed.shape==(1,224,224,160)
    assert set(tmp_path.rglob('*'))==before
