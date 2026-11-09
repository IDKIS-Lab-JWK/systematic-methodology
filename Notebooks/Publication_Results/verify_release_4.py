"""Verify the executed publication package against its manifest and unchanged inputs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import nbformat
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'output_4'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'cache/matplotlib'))
import analysis_helpers_4 as ah

protected=[HERE/f'Publication_Results_{v}.ipynb' for v in (1,2,3)]
protected += [p for p in HERE.glob('*_2.py')]
protected += [ROOT/'Notebooks/FlexiCT/FlexiCT_100pct_Test_Error_Review.ipynb']
protection_path=OUT/'audit/protected_previous_files_4.json'
if '--snapshot' in sys.argv:
    if not protection_path.exists():
        protection_path.write_text(json.dumps({str(p.relative_to(ROOT)):ah.digest(p) for p in protected},indent=2),encoding='utf-8')
    print('Previous notebook/code protection snapshot present.')
    raise SystemExit(0)

manifest=ah.read_json(OUT/'analysis_manifest_4.json')
book=nbformat.read(HERE/'Publication_Results_4.ipynb',as_version=4)
nbformat.validate(book)
code=[c for c in book.cells if c.cell_type=='code']
assert [c.execution_count for c in code]==list(range(1,len(code)+1))
assert not any(o.output_type=='error' for c in code for o in c.outputs)
assert ah.fingerprint([(c.source,c.cell_type) for c in book.cells])==manifest['notebook_source_hash']
for name,digest in manifest['code_hashes'].items():
    assert ah.digest(HERE/name)==digest,name
artifacts=manifest['current_artifacts']
assert len({a['path'] for a in artifacts})==len(artifacts)
for artifact in artifacts:
    p=OUT/artifact['path']
    assert ah.digest(p)==artifact['sha256'],str(p)
    if p.suffix=='.png':
        with Image.open(p) as im: im.verify()
_,_,_,ids,truth=ah.load_splits(ROOT)
inventory,predictions,audit,_,hashes=ah.discover(ROOT,ids,truth)
assert hashes==manifest['input_hashes'],'Source result or run-context hashes changed'
assert len(inventory)==192 and len(ids)==278
assert not (audit.status=='excluded').any()
for relative,digest in manifest['review_input_hashes'].items():
    assert ah.digest(ROOT/relative)==digest,relative
review=pd.read_csv(OUT/'audit/local_radiologist_linkage_and_verbatim_4.csv',dtype={'patient_id':str})
assert len(review)==20 and review.review_id.is_unique
assert (review.linkage_status=='exact_id').sum()==19
assert review.loc[review.patient_id=='R20','linkage_status'].item()=='provisional_user_inference'
assert review.loc[review.patient_id=='R08','pillar_predicted_class'].item()=='Sigmoid'
assert review.pillar_matches_original_label.sum()==8 and review.same_model_prediction.sum()==10
for row in review.itertuples():
    p=ROOT/'Dataset/NIFTI'/row.true_class/(row.patient_id+'.nii.gz')
    assert ah.digest(p)==row.source_sha256
for relative,digest in ah.read_json(protection_path).items():
    assert ah.digest(ROOT/relative)==digest,relative
mainfigs=list((OUT/'main/figures').glob('*.png'))
maintables=list((OUT/'main/tables').glob('*.csv'))
assert len(mainfigs)==9 and len(maintables)==4
assert len(list((OUT/'supplementary/figures').glob('Figure_S10*.png')))==20
assert len(list((OUT/'supplementary/figures').glob('Figure_S11*.png')))==20
metrics=pd.read_csv(OUT/'audit/recomputed_metric_checks_4.csv')
assert len(metrics)==960 and metrics.passed.all()
lp=inventory[(inventory.model=='3DINO')&(inventory.method=='LinearProbe')]
assert len(lp)==15 and set(lp.percentage)=={20,40,60,80,100}
assert (inventory[(inventory.model=='Merlin')&(inventory.method=='FullFineTune')].cv_id=='CV_2').all()
assert len(manifest['configuration']['comparison_plan'])==19
assert manifest['counts']['selected_main_runs']==13
for stem in ['Figure_3a_partial_adaptation_and_linear_probes_4','Figure_3b_full_network_comparison_4']:
    for suffix in ['.png','_600dpi.png','.pdf','.svg','.jpg','.tiff']:
        assert (OUT/'main/figures'/(stem+suffix)).is_file()
report={
    'verified_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
    'executed_code_cells':len(code),'notebook_cells':len(book.cells),
    'valid_runs':len(inventory),'heldout_patients':len(ids),'summary_metric_checks':len(metrics),
    'main_numbered_figures':6,'figure_3_panels':2,'main_tables':4,'review_cases':20,'exact_comment_links':19,'provisional_comment_links':1,
    'registered_artifacts_verified':len(artifacts),'explanation_input_hashes_verified':len(manifest['review_input_hashes']),
    'source_results_unchanged':True,'previous_notebooks_and_helpers_unchanged':True,
    'notebook_sha256':ah.digest(HERE/'Publication_Results_4.ipynb'),
    'html_sha256':ah.digest(OUT/'Publication_Results_4.html'),
    'manifest_sha256':ah.digest(OUT/'analysis_manifest_4.json'),
    'remaining_author_items':['Confirm provisional original-source/cohort provisional linkage linkage','Clinical reference standard, reader protocol, recruitment, demographic/scanner and ethics details','Clinical selection/annotation and submission packaging of image panels'],
}
(OUT/'audit/release_verification_4.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
