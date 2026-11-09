"""Statistical and ingestion checks; uses synthetic fixtures, never mutates source results."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score

import analysis_helpers_4 as ah


class AnalysisTests(unittest.TestCase):
    def test_version_four_scope(self):
        self.assertEqual(ah.METHODS['Merlin'], ['LinearProbe','ConvLoRA_LLRD','FullFineTune'])
        self.assertEqual(ah.METHODS['3DINO'], ['LinearProbe','LoRA'])
        self.assertEqual(ah.METHODS['MedicalNet'], ['FullFineTune'])
        self.assertEqual(ah.METHODS['DenseNet'], ['FromScratch'])
        self.assertEqual(set(ah.FOUNDATION_MODELS), {'COLIPRI','FlexiCT','Merlin','Pillar0'})
        for model in ah.FOUNDATION_MODELS:
            self.assertTrue({ah.FROZEN_METHODS[model],ah.ADAPTATION_METHODS[model]}.issubset(ah.METHODS[model]))

    def test_foundation_no_nonpublication_fallback(self):
        base={'model':'Merlin','method':'ConvLoRA_LLRD','percentage':100,
              'source':'cv_ensemble','zero_shot':False,'run_id':'grid','source_family':'grid'}
        chosen,_=ah.choose_runs(pd.DataFrame([base]))
        self.assertTrue(chosen.empty)
        with self.assertRaises(ValueError):
            ah.choose_runs(pd.DataFrame([base]), {'Merlin|ConvLoRA_LLRD|100|cv_ensemble':'grid'})

    def setUp(self):
        self.rng=np.random.default_rng(41)
        self.y=np.repeat(np.arange(5),[4,5,6,3,7])
        # Deliberate ties exercise weighted empirical ranking correctly.
        raw=self.rng.integers(1,5,size=(len(self.y),5)).astype(float)
        self.p=raw/raw.sum(1,keepdims=True)
        self.ids=[f'{i:04d}' for i in range(len(self.y))]

    def test_bootstrap_matches_independent_weighted_sklearn(self):
        weights=ah.stratified_weights(self.y,17,41)
        output=ah.bootstrap_metrics(self.y,self.p,weights)
        pred=self.p.argmax(1);truth=np.eye(5)[self.y]
        for i,w in enumerate(weights):
            expected={'f1_macro':f1_score(self.y,pred,average='macro',sample_weight=w,zero_division=0),
                'accuracy':accuracy_score(self.y,pred,sample_weight=w),
                'balanced_accuracy':balanced_accuracy_score(self.y,pred,sample_weight=w),
                'average_precision_macro':average_precision_score(truth,self.p,average='macro',sample_weight=w),
                'roc_auc_macro':roc_auc_score(truth,self.p,average='macro',sample_weight=w)}
            for metric,value in expected.items():
                self.assertAlmostEqual(output[metric][i],value,places=12,msg=metric)
            ids=np.repeat(np.arange(len(self.y)),w)
            expanded=ah.point_metrics(self.y[ids],self.p[ids])
            for metric in ['brier_multiclass','log_loss','ece']:
                self.assertAlmostEqual(output[metric][i],expanded[metric],places=12)

    def test_identical_pairs_zero_delta_and_unit_p(self):
        pred=self.p.argmax(1)
        result=ah.paired_test(self.y,pred,pred,100,7)
        self.assertEqual(result['permutation_p'],1)
        self.assertEqual(result['mcnemar_p'],1)

    def test_holm_pending_family(self):
        np.testing.assert_allclose(ah.holm([.01,.03,.04],5),[.05,.12,.12])

    def write_run(self,root):
        folder=root/'Results/DenseNet/FromScratch/Test/CV_2/cv_ensemble/100_percent'
        folder.mkdir(parents=True)
        preds=self.p.argmax(1)
        records=[{'patient_id':pid,'true_class':ah.CLASSES[y], 'predicted_class':ah.CLASSES[pred],
                  'probabilities':dict(zip(ah.CLASSES,p))} for pid,y,pred,p in zip(self.ids,self.y,preds,self.p)]
        (folder/'test_patient_predictions.json').write_text(json.dumps(records[::-1]))
        rows=[{'patient_id':r['patient_id'],'true_class':r['true_class'],'predicted_class':r['predicted_class'],
               **{f'probability_{c}':r['probabilities'][c] for c in ah.CLASSES}} for r in records]
        pd.DataFrame(rows).to_csv(folder/'test_probabilities.csv',index=False)
        selection={'model':'DenseNet','method':'FromScratch','percentage':100,'cv_id':'CV_2',
                   'model_source':'cv_ensemble','parameter_provenance':{'parameter_source':'publication'},'parameters':{'seed':42},
                   'artifacts':[{'path':f'fold_{i}/model.pt'} for i in range(1,6)]}
        (folder/'model_selection.json').write_text(json.dumps(selection))
        (folder/'metrics_summary.json').write_text(json.dumps(ah.point_metrics(self.y,self.p)))
        return folder,selection

    def seal(self,folder,selection):
        context={'cv_id':'CV_2','percentage':100,'model_source':'cv_ensemble','selection_sha256':ah.fingerprint(selection)}
        records=[{'path':name,'bytes':(folder/name).stat().st_size,'sha256':ah.digest(folder/name)} for name in sorted(ah.REQUIRED)]
        seal={'context':context,'artifacts':records,'seal_sha256':ah.fingerprint({'context':context,'artifacts':records})}
        (folder/'test_completion.json').write_text(json.dumps(seal))

    def test_future_densenet_enters_only_when_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,selection=self.write_run(root)
            inventory,_,audit,_,_=ah.discover(root,self.ids,self.y)
            self.assertTrue(inventory.empty)
            self.assertIn('seal missing',audit.reason.iloc[0])
            self.seal(folder,selection)
            inventory,arrays,audit,_,_=ah.discover(root,self.ids,self.y)
            self.assertEqual(len(inventory),1)
            self.assertEqual(inventory.model.iloc[0],'DenseNet')
            chosen,_=ah.choose_runs(inventory)
            self.assertEqual(len(chosen),1)
            np.testing.assert_allclose(arrays[chosen.run_id.iloc[0]][0],self.p)

    def test_hash_tampering_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,selection=self.write_run(root);self.seal(folder,selection)
            path=folder/'metrics_summary.json'
            path.write_text(path.read_text()+' ')
            inventory,_,audit,_,_=ah.discover(root,self.ids,self.y)
            self.assertTrue(inventory.empty)
            self.assertIn('mismatch',audit.reason.iloc[0])

    def test_duplicate_patient_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,selection=self.write_run(root)
            path=folder/'test_patient_predictions.json';data=ah.read_json(path)
            data[1]['patient_id']=data[0]['patient_id'];path.write_text(json.dumps(data))
            self.seal(folder,selection)
            inventory,_,audit,_,_=ah.discover(root,self.ids,self.y)
            self.assertTrue(inventory.empty)
            self.assertIn('duplicated',audit.reason.iloc[0])

    def test_incomplete_ensemble_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,selection=self.write_run(root)
            selection['artifacts']=selection['artifacts'][:4]
            (folder/'model_selection.json').write_text(json.dumps(selection))
            self.seal(folder,selection)
            inventory,_,audit,_,_=ah.discover(root,self.ids,self.y)
            self.assertTrue(inventory.empty)
            self.assertIn('five checkpoint',audit.reason.iloc[0])

    def test_selection_never_uses_test_score(self):
        base={'model':'DenseNet','method':'FromScratch','percentage':100,'source':'cv_ensemble','zero_shot':False}
        frame=pd.DataFrame([{**base,'run_id':'publication','source_family':'publication','f1_macro':.1},
                            {**base,'run_id':'grid','source_family':'grid','f1_macro':.99}])
        chosen,_=ah.choose_runs(frame)
        self.assertEqual(chosen.run_id.tolist(),['publication'])
        frame=pd.concat([frame,pd.DataFrame([{**base,'run_id':'publication2','source_family':'publication','f1_macro':.9}])])
        chosen,audit=ah.choose_runs(frame)
        self.assertTrue(chosen.empty)
        self.assertEqual(audit.status.iloc[0],'ambiguous_or_unknown_provenance')


if __name__=='__main__':
    unittest.main(verbosity=2)
