"""Read-only integration of the existing FlexiCT error review and radiologist reread.

No diagnostic relabeling, model inference, or source-result mutation occurs here.
All new artifacts use a separate notebook-local output directory.
"""
from pathlib import Path
import json
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import analysis_helpers_4 as ah

# Editorial summaries of the supplied review, not independent diagnoses.
# Mapping provisional original-source/cohort linkage is provisional: unique unmatched comment/case, no 8701
# in the local cohort, and the user's conditional identification on 2026-10-18.
REVIEW = {
 'R01': ('Reference-label discrepancy', 'Clear appendicitis on reread; supports the model class.', 'Reconsider recorded Normal label'),
 'R02': ('Other disease / proposed exclusion', 'Colitis and jejunitis; neither normal nor appendicitis nor sigmoid diverticulitis.', 'Outside the five mutually exclusive labels'),
 'R03': ('Other disease / proposed exclusion', 'Gallbladder-adjacent enhancing lesion ultimately considered a liver tumor mimicking inflammation.', 'Reviewer recommends exclusion'),
 'R04': ('Other disease / proposed exclusion', 'Crohn disease exacerbation with surrounding inflammation and tumor infiltration.', 'Reviewer recommends exclusion'),
 'R05': ('Target diagnosis not supported', 'Incarcerated left inguinal hernia involving sigmoid; no clinical or operative proof of diverticulitis.', 'Absence of diverticulitis does not establish a normal abdomen'),
 'R06': ('Other disease / proposed exclusion', 'Malignancy-associated rectal obstruction, splenomegaly and ascites; no appendicitis.', 'Reviewer recommends exclusion'),
 'R07': ('Likely error / uncertain reference', 'Cholecystitis likely but not proven; nearby postoperative stoma may mimic inflammation.', 'Reviewer retains case; model likely wrong'),
 'R08': ('Overlapping target diagnoses', 'Acute and chronic sigmoid diverticulitis with perforation.', 'Single-label task conflates coexisting diagnoses'),
 'R09': ('Confirmed model error', 'Subtle cholecystitis clinically proven, with cholestasis.', 'Recorded disease retained'),
 'R10': ('Confirmed model error', 'Proven cholecystitis with complex anatomy, displaced kidney, hernias, sarcopenia and chronic inflammation.', 'Recorded disease retained'),
 'R12': ('Target diagnosis not supported', 'Urosepsis and gallbladder hydrops; initial noncontrast CT diagnosis of cholecystitis was not supported.', 'Absence of cholecystitis does not establish a normal abdomen'),
 'R13': ('Confirmed model error', 'No pathology on two reviews; a dense sigmoid clip is a proposed mimic.', 'Clip explanation is a reviewer hypothesis'),
 'R11': ('Confirmed model error', 'No pathology on either review.', 'Recorded Normal label retained'),
 'R17': ('Other disease / proposed exclusion', 'Extensive ulcerative colitis with secondary appendiceal enhancement rather than primary appendicitis.', 'Reviewer recommends exclusion'),
 'R18': ('Reference-label discrepancy', 'Bacterial colitis with focal sigmoid inflammation and diverticula; diverticulitis considered likely.', 'Reread supports model class provisionally'),
 'R19': ('Overlapping target diagnoses', 'Sigmoid diverticulitis with perforation.', 'Single-label task conflates coexisting diagnoses'),
 'R20': ('Likely error / uncertain reference', 'Reviewer describes pyloric perforation with minimal free air and gallbladder hydrops mimicking cholecystitis; exclusion or discussion suggested.', 'Provisional linkage: original correspondence identifier differs'),
 'R14': ('Confirmed model error', 'Operatively confirmed covered duodenal perforation with large air/fluid collection and atypical anatomy.', 'Normal prediction is a true miss on reread'),
 'R15': ('Likely error / uncertain reference', 'Gallstones, chronic wall changes and symptoms improving with antibiotics; no definitive reference diagnosis.', 'Reviewer considers both labels defensible'),
 'R16': ('Confirmed model error', 'Clear uncomplicated appendicitis; renal transplant changes anatomy.', 'Anatomical explanation is a reviewer hypothesis'),
}


def load_review(root, notebook_dir, main, predictions, ids, truth):
    base = root/'Notebooks/FlexiCT/exports/error_review_100_cv_ensemble'
    original = pd.read_csv(base/'misclassified_cases.csv', dtype={'patient_id':str})
    source = ah.read_json(notebook_dir/'inputs_4/radiologist_review_source.json')
    source_comments = {}
    chunks = list(re.finditer(r'(?m)^\s*(\d{4})\s', source['text']))
    for i, match in enumerate(chunks):
        end = chunks[i+1].start() if i+1 < len(chunks) else len(source['text'])
        source_comments[match.group(1)] = source['text'][match.start():end].strip()
    assert len(source_comments) == 20
    flex = main[(main.model=='FlexiCT') & (main.method=='FinalBlockFineTune')].iloc[0]
    pillar = main[(main.model=='Pillar0') & (main.method=='AtlasFinalBlockFineTune')].iloc[0]
    fp, fhat = predictions[flex.run_id]; pp, phat = predictions[pillar.run_id]
    expected = {ah.CLASSES[truth[i]]+'_'+ids[i] for i in np.flatnonzero(fhat!=truth)}
    assert expected == set(original.case_key) and len(expected) == 20
    assert 'R20' not in ids and not list((root/'Dataset/NIFTI').glob('*/8701.nii*'))
    assert set(source_comments)-set(original.patient_id) == {'R20'}
    assert set(original.patient_id)-set(source_comments) == {'R20'}
    hashes = {}; checks = []; rows = []; index = {pid:i for i,pid in enumerate(ids)}
    for filename in ['cases.json','maps_execution.json','occlusion_execution.json','misclassified_cases.csv','summary.json']:
        hashes[str((base/filename).relative_to(root))] = ah.digest(base/filename)
    for stage in ['maps','occlusion']:
        record = ah.read_json(base/f'{stage}_execution.json')
        assert record['cases'] == 20
        assert record['selection_sha256'] == ah.digest(Path(flex.directory)/'model_selection.json')
        assert record['implementation_sha256'] == ah.digest(root/'Notebooks/FlexiCT/error_review_inference.py')
    for number, row in enumerate(original.itertuples(),1):
        i=index[row.patient_id]; dest=base/row.case_key
        fs=[np.load(dest/f'fold_{k:02d}.npz') for k in range(1,6)]
        foldp=np.stack([f['probabilities'] for f in fs]); replay=foldp.mean(0)
        assert np.max(np.abs(replay-fp[i])) < 1e-4
        assert int(replay.argmax()) == fhat[i]
        for f in fs: f.close()
        meta=ah.read_json(dest/'preprocessing.json')
        scan=root/'Dataset/NIFTI'/row.true_class/(row.patient_id+'.nii.gz')
        assert ah.digest(scan) == meta['source_sha256']
        for p in dest.iterdir():
            if p.is_file(): hashes[str(p.relative_to(root))]=ah.digest(p)
        category,summary,interpretation=REVIEW[row.patient_id]
        source_id='R20' if row.patient_id=='R20' else row.patient_id
        rows.append({**row._asdict(), 'review_id':f'R{number:02d}',
            'review_category':category,'radiologist_summary':summary,'interpretation':interpretation,
            'source_comment_id':source_id,'linkage_status':'provisional_user_inference' if source_id=='R20' else 'exact_id',
            'verbatim_comment':source_comments[source_id],
            'pillar_predicted_class':ah.CLASSES[phat[i]],'pillar_predicted_probability':pp[i,phat[i]],
            'pillar_true_probability':pp[i,truth[i]],'pillar_probability_flexict_class':pp[i,fhat[i]],
            'pillar_matches_original_label':bool(phat[i]==truth[i]),
            'same_model_prediction':bool(phat[i]==fhat[i])})
        checks.append({'review_id':f'R{number:02d}','probability_replay_max_abs_error':float(np.max(np.abs(replay-fp[i]))),'source_volume_hash_verified':True})
    frame=pd.DataFrame(rows)
    user=pd.read_csv(notebook_dir/'inputs_4/pillar_user_table.csv')
    differences=[]
    for u in user.itertuples():
        actual=frame[frame.case_key==u.case_key].iloc[0]
        for col in ['pillar_predicted_class','pillar_predicted_probability','pillar_true_probability','pillar_probability_flexict_class']:
            value=getattr(u,col)
            if pd.isna(value): status='blank_in_user_table';delta=np.nan
            elif isinstance(value,str): status='matches' if value==actual[col] else 'mismatch';delta=np.nan
            else:
                delta=float(value)-float(actual[col]);status='within_rounded_tolerance' if abs(delta)<=0.005 else 'mismatch'
            differences.append({'review_id':actual.review_id,'case_key':u.case_key,'field':col,'user_value':value,'saved_value':actual[col],'difference':delta,'status':status})
    for p in (notebook_dir/'inputs_4').glob('*'):
        hashes[str(p.relative_to(root))]=ah.digest(p)
    return frame,pd.DataFrame(differences),pd.DataFrame(checks),hashes,base,flex,pillar


def visual_data(base, case_key):
    folder=base/case_key
    with np.load(folder/'input.npz') as x: volume=x['volume'].copy().squeeze()
    meta=ah.read_json(folder/'preprocessing.json')
    hu=volume*meta['clipped_std_correction_1']+meta['clipped_mean']
    maps=[]
    for k in range(1,6):
        with np.load(folder/f'fold_{k:02d}.npz') as x: maps.append(x['raw_maps'].copy())
    raw=np.stack(maps).mean(0)
    positives=[]
    for m in raw:
        m=np.maximum(m,0);m=m/m.max() if m.max()>0 else m
        positives.append(zoom(m,8,order=1))
    peak=np.unravel_index(positives[0].argmax(),positives[0].shape)
    contrast=zoom(raw[0]-raw[1],8,order=1);contrast/=max(np.abs(contrast).max(),1e-12)
    return hu,positives,contrast,peak


def case_panel(frame, base):
    fig,axes=plt.subplots(len(frame),4,figsize=(13,3.3*len(frame)),squeeze=False)
    for r,row in enumerate(frame.itertuples()):
        hu,maps,contrast,peak=visual_data(base,row.case_key);z=peak[0]
        for ax in axes[r]:
            ax.imshow(hu[z],cmap='gray',vmin=-160,vmax=240);ax.axis('off')
        for c in (1,2):
            a=maps[c-1][z];axes[r,c].imshow(a,cmap='inferno',vmin=0,vmax=1,alpha=.6*a)
        a=contrast[z];axes[r,3].imshow(a,cmap='coolwarm',vmin=-1,vmax=1,alpha=.65*np.abs(a))
        axes[r,0].text(-.05,.5,f'{row.review_id}\nRecorded: {row.true_class}\nPredicted: {row.predicted_class}\np={row.predicted_probability:.3f}\nAxial z={z}',transform=axes[r,0].transAxes,ha='right',va='center',fontsize=9)
        axes[r,1].set_title(textwrap.fill(row.review_category,30),fontsize=9)
    for ax,title in zip(axes[0],['CT · WL 40 / WW 400','Predicted-class CAM','Recorded-class CAM','CAM contrast']):
        ax.set_title(title+'\n'+ax.get_title(),fontsize=10)
    fig.subplots_adjust(left=.19,right=.99,top=.94,bottom=.08,wspace=.03,hspace=.25)
    fig.text(.5,.025,'Model-space LPS · 2 mm voxels · positive CAMs scaled independently 0–1\nContrast: blue = recorded class, red = predicted class. Planes at predicted-map maximum; no lesion annotation.',ha='center',fontsize=9)
    return fig


def orthogonal_panel(row, base):
    hu,maps,contrast,peak=visual_data(base,row.case_key)
    def planes(a):
        z,y,x=peak
        return [a[z],np.flipud(a[:,y,:]),np.flipud(a[:,:,x])]
    fig,axes=plt.subplots(3,4,figsize=(12,9.5))
    for r,plane in enumerate(['Axial','Coronal','Sagittal']):
        for c,ax in enumerate(axes[r]):
            ax.imshow(planes(hu)[r],cmap='gray',vmin=-160,vmax=240);ax.axis('off')
            if c in (1,2):
                a=planes(maps[c-1])[r];ax.imshow(a,cmap='inferno',vmin=0,vmax=1,alpha=.6*a)
            elif c==3:
                a=planes(contrast)[r];ax.imshow(a,cmap='coolwarm',vmin=-1,vmax=1,alpha=.65*np.abs(a))
        axes[r,0].set_title(f'{plane} · model index {peak[r]}',fontsize=9)
    for ax,title in zip(axes[0],['CT · WL 40 / WW 400','Predicted-class CAM','Recorded-class CAM','Signed CAM contrast']):
        ax.set_title(title+'\n'+ax.get_title(),fontsize=9)
    fig.suptitle(f'{row.review_id} · Recorded: {row.true_class} · Predicted: {row.predicted_class}\n'
        f'p(predicted)={row.predicted_probability:.3f} · p(recorded)={row.true_probability:.3f} · '
        f'{row.review_category}' + ('\nPROVISIONAL COMMENT–CASE LINKAGE' if row.linkage_status!='exact_id' else ''),fontsize=12)
    fig.subplots_adjust(left=.02,right=.98,bottom=.09,top=.88,wspace=.02,hspace=.15)
    fig.text(.5,.025,'LPS model space · 2 mm voxels · 16 mm native map grid · planes at predicted-map maximum\n'
        'Positive scales independent: 0 (transparent) to 1 (yellow). Contrast: blue recorded / red predicted.\n'
        'Cropped input and qualitative maps; no validated lesion localization.',ha='center',fontsize=9)
    return fig


def ct_overview(row,base):
    hu,_,_,_=visual_data(base,row.case_key)
    fig,axes=plt.subplots(2,6,figsize=(13,5))
    for ax,z in zip(axes.flat,np.linspace(0,len(hu)-1,12).round().astype(int)):
        ax.imshow(hu[z],cmap='gray',vmin=-160,vmax=240);ax.axis('off');ax.set_title(f'z={z}',fontsize=9)
    fig.suptitle(f'{row.review_id}: model-input axial overview | {row.true_class} → {row.predicted_class}',fontsize=12)
    fig.tight_layout();return fig


def manuscript_text(main, ensembles, paired, review, main_class, source_sensitivity):
    """Numbers are generated from validated predictions; interpretation stays cautious."""
    get=lambda model,method:main[(main.model==model)&(main.method==method)].iloc[0]
    flex=get('FlexiCT','FinalBlockFineTune'); pillar=get('Pillar0','AtlasFinalBlockFineTune')
    dense=get('DenseNet','FromScratch');merlin=get('Merlin','ConvLoRA_LLRD')
    f=lambda x:f'{x.f1_macro:.3f} (95% CI, {x.f1_macro_low:.3f}–{x.f1_macro_high:.3f})'
    counts=review.review_category.value_counts()
    lines=[
      'RESULTS — proposed manuscript text (exploratory retrospective reporting)',
      'Study cohort and completed analyses. The fixed held-out cohort comprised 278 patients: 47 with recorded appendicitis, 37 cholecystitis, 90 Normal, 18 perforation and 86 Sigmoid labels. The development pool contained 1,080 patients. Five nested development pools contained 248, 456, 664, 872 and 1,080 patients; the perforation development pool remained fixed at 40. Thirteen supervised pipelines were compared using the same test patients and five-fold probability ensembles. All 65 method-by-fraction ensemble cells were available (Tables 1–3; Figure 1).',
      f'Overall performance. At full development size, FlexiCT final-block adaptation had the highest observed macro-F1 among the evaluated pipelines, {f(flex)}, with balanced accuracy {flex.balanced_accuracy:.3f} and accuracy {flex.accuracy:.3f}. This is an observed ranking, not a prospectively selected winner. Macro-F1 was {f(pillar)} for Pillar-0 final-block adaptation, {f(dense)} for DenseNet from scratch and {f(merlin)} for corrected Merlin ConvLoRA+LLRD (Table 3; Figure 2).',
    ]
    for model,frozen,adapted in [('FlexiCT','LinearProbe','FinalBlockFineTune'),('Pillar0','LinearProbe','AtlasFinalBlockFineTune'),('COLIPRI','PoolingOptimisedLinearProbe','PrimusMFinalBlockFineTune'),('Merlin','LinearProbe','ConvLoRA_LLRD')]:
        a=get(model,adapted);b=get(model,frozen)
        p=paired[(paired.model_a==model)&(paired.method_a==adapted)&(paired.model_b==model)&(paired.method_b==frozen)].iloc[0]
        lines.append(f'{model} adaptation increased observed macro-F1 from {b.f1_macro:.3f} to {a.f1_macro:.3f}; paired difference {p.difference:.3f} (95% CI, {p.ci_low:.3f}–{p.ci_high:.3f}; exploratory Holm-adjusted permutation P={p.permutation_p_holm:.4g}).')
    lines.append('The Pillar-0 adaptation contrast did not meet the 0.05 threshold after multiplicity adjustment. Reported 95% effect intervals are pointwise and unadjusted for the comparison family; they need not agree with multiplicity-adjusted tests.')
    dm=paired[(paired.model_a=='Merlin')&(paired.method_a=='ConvLoRA_LLRD')&(paired.model_b=='DenseNet')].iloc[0]
    lines.append(f'DenseNet versus corrected Merlin. The observed macro-F1 advantage of DenseNet was {-dm.difference:.3f} (paired 95% CI, {-dm.ci_high:.3f}–{-dm.ci_low:.3f}; Holm-adjusted permutation P={dm.permutation_p_holm:.3f}). The interval includes zero; these data do not establish superior robustness or a reliable performance advantage over corrected Merlin.')
    full=get('Merlin','FullFineTune');lp=get('3DINO','LinearProbe');lora=get('3DINO','LoRA')
    fp=paired[(paired.model_a=='Merlin')&(paired.method_a=='FullFineTune')&(paired.model_b=='DenseNet')].iloc[0]
    dp=paired[(paired.model_a=='3DINO')&(paired.method_a=='LoRA')&(paired.model_b=='3DINO')&(paired.method_b=='LinearProbe')].iloc[0]
    lines.append(f'Retrospective full-network addition. Merlin FullFineTune CV_2 achieved macro-F1 {f(full)}, versus {f(dense)} for DenseNet trained from scratch; paired difference {fp.difference:.3f} (95% CI {fp.ci_low:.3f}–{fp.ci_high:.3f}; Holm-adjusted P={fp.permutation_p_holm:.4g}). The interval includes zero. Table S16 shows all five fractions and three matched routes. These data contradict a universal 72% CNN accuracy or macro-F1 ceiling.')
    lines.append(f'3DINO linear probing. The frozen-encoder classifier achieved macro-F1 {f(lp)}, versus {f(lora)} with LoRA. The probe-minus-LoRA difference was {-dp.difference:.3f} (95% CI {-dp.ci_high:.3f}–{-dp.ci_low:.3f}; adjusted P={dp.permutation_p_holm:.4g}). Accuracy was {lp.accuracy:.3f} and {lora.accuracy:.3f}, respectively, illustrating that accuracy and macro-F1 need not rank pipelines identically. All 15 saved probe test cells are included; the earlier omission was a notebook-scope error. Table S17 provides matched-size and matched-route comparisons.')
    lines.append('Version-4 inference recalculates all 19 planned full-development ensemble comparisons as one exploratory Holm family. Pointwise paired intervals remain unadjusted. Earlier adjusted P values must not be combined with this release.')
    perforation=main_class[main_class['class']=='Perforation'].set_index('run_id')
    values=[]
    for run,label in [(flex,'FlexiCT final block'),(pillar,'Pillar-0 final block'),(dense,'DenseNet'),(merlin,'corrected Merlin')]:
        v=perforation.loc[run.run_id]
        values.append(f'{label}: {int(v.tp)}/18, {v.sensitivity:.3f} (Wilson 95% CI, {v.sensitivity_low:.3f}–{v.sensitivity_high:.3f})')
    lines.append('Class-specific performance. Perforation sensitivity was '+ '; '.join(values)+'. The 3DINO LoRA pipeline correctly classified 0/18 perforation cases; these small denominators require cautious interpretation (Table 4; Figure 5).')
    learning=[]
    for model,method in [('FlexiCT','FinalBlockFineTune'),('Pillar0','AtlasFinalBlockFineTune'),('DenseNet','FromScratch')]:
        g=ensembles[(ensembles.model==model)&(ensembles.method==method)].set_index('percentage')
        learning.append(f'{model}, {g.loc[20,"f1_macro"]:.3f} to {g.loc[100,"f1_macro"]:.3f}')
    lines.append('Data efficiency. Macro-F1 at the smallest and largest development pools was: '+ '; '.join(learning)+'. Intermediate fractions are shown as point estimates without bands in Figures 3a and 3b; numerical intervals appear in Table S1. These nested-sample curves do not estimate robustness to external hospitals or acquisition protocols.')
    route=source_sensitivity[(source_sensitivity.percentage==100)&(source_sensitivity.alternative_source=='refit_full_development')]
    lines.append(f'Checkpoint-source sensitivity. Ensemble-minus-refit macro-F1 differences at full development size ranged from {route.difference.min():.3f} to {route.difference.max():.3f} across {len(route)} matched pipelines; {int((route.difference>0).sum())} had a higher ensemble point estimate. Source-specific results and paired intervals are supplied in Tables S1 and S9 and Figure S6. The main table consistently uses ensembles rather than choosing each pipeline\'s best-performing test route. The overall emphasis on ensembles remains a retrospective reporting change.')
    lines.extend([
      f'Exploratory error reread. FlexiCT final-block ensemble predictions disagreed with the original labels in 20 of 278 patients (7.2%). A radiologist provided narrative comments on these selected disagreements; 19 case IDs matched directly and one was linked provisionally by elimination and user confirmation of plausibility. An analyst coding of the comments grouped {counts.get("Confirmed model error",0)} as confirmed errors, {counts.get("Reference-label discrepancy",0)} as reference-label discrepancies, {counts.get("Target diagnosis not supported",0)} as absence of support for the recorded target diagnosis, {counts.get("Overlapping target diagnoses",0)} as overlapping target diagnoses, {counts.get("Other disease / proposed exclusion",0)} as other disease or proposed exclusion, and {counts.get("Likely error / uncertain reference",0)} as likely errors or uncertain references. These categories are editorial synthesis, not a second adjudicated reference standard (Figure 6; Table S12). Original labels, cohort membership and headline metrics were retained.',
      f'Within the same FlexiCT-selected 20 cases, Pillar-0 final-block ensemble predictions matched the original label in {int(review.pillar_matches_original_label.sum())}/20 and agreed with FlexiCT in {int(review.same_model_prediction.sum())}/20 (Table S13). These conditional counts do not estimate relative full-cohort performance or radiologist-confirmed accuracy. Saved predictions superseded transcription discrepancies in the supplied comparison table.',
      f'Attribution diagnostics. All 20 saved explanation runs reproduced the ensemble probabilities within 1e-4. Targeted occlusion reduced predicted-class probability more than the mean of random occlusions with a positive decrease in {int(review.positive_drop_exceeds_random.sum())}/20 cases; targeted occlusion increased predicted-class probability in {int((review.top_mask_probability_drop<0).sum())}/20. Median targeted and random probability changes were {review.top_mask_probability_drop.median():.3f} and {review.random_mask_mean_probability_drop.median():.3f}. These perturbations and coarse token Grad-CAM maps are exploratory model diagnostics, not validated lesion localization (Figures S9–S10).',
      'DISCUSSION — proposed manuscript text',
      'The comparison suggests that task-specific adaptation can materially improve the utility of pretrained CT representations for this five-class endpoint. FlexiCT had the highest observed ensemble performance, while DenseNet trained from scratch remained a competitive reference pipeline. This does not establish that transformers outperform CNNs, or that pretraining itself caused the differences: preprocessing, pretraining corpora, adaptation scope, optimization and search budgets differed across pipelines. DenseNet comparisons with both corrected Merlin adapters and full-network Merlin remained statistically uncertain; full-network Merlin had the higher ensemble point estimate. The 3DINO probe result also shows that task adaptation is not guaranteed to improve macro-F1 over a frozen representation. Architecture-family and robustness claims require controlled experiments and external validation.',
      'The reread illustrates why a model–label disagreement is not automatically a diagnostic failure. Some recorded Normal cases contained inflammatory disease; sigmoid diverticulitis and perforation coexisted in two narratives; other patients had disorders outside the mutually exclusive label set. Conversely, proven subtle cholecystitis, appendicitis and covered perforation remained clinically meaningful misses. A model can recognize inflammation yet choose an incorrect specific diagnosis. Absence of a particular target disease must not be interpreted as a disease-free Normal abdomen.',
      'The reread should be presented as an exploratory error analysis in Results, with its implications for reference-standard quality and label design developed here in Discussion. It cannot establish radiologist superiority, model superiority or corrected overall accuracy: cases were selected on FlexiCT error status after evaluation; review blinding and structured adjudication were not documented; remaining test cases were not reread. Proposed exclusions or label revisions would require a separate, consistently applied adjudication protocol for the entire cohort before a new sensitivity analysis. The provisional original-source/cohort provisional linkage linkage also needs author verification.',
      'Grad-CAM and occlusion help inspect model behavior but cannot prove that the model localized the lesion or used a radiologist-proposed mimic. The stored maps have a coarse 16-mm grid; smoothing adds no anatomical information. Display scales are independent across targets, slices are selected at the predicted-map maximum rather than a lesion annotation, and masking changes input distribution. Negative occlusion changes and limited randomization checks further caution against causal interpretation. Representative images should illustrate complementary failure mechanisms, with all 20 cases supplied transparently in the supplement.',
      'The Pillar-0 comparison belongs in the supplement as a conditional case audit. Its complete-cohort result already appears in the main table. Selection of cases solely because FlexiCT failed creates an asymmetric comparison and favors apparent rescues by the comparator; a general head-to-head claim must use all 278 paired predictions. Neither model should receive post-hoc diagnostic credit solely because a narrative mentions related inflammation.',
      'Limitations include retrospective endpoint/reporting changes after results were available, repeated inspection of the same held-out cohort, a small perforation test sample, enriched class prevalence, fixed perforation training support, missing external validation and repeated training seeds, and incomplete demographic/acquisition metadata. Patient-bootstrap intervals condition on the fitted models and do not capture training variability. Probability calibration, attribution plausibility and internal performance do not establish prospective clinical utility.',
      'AUTHOR ITEMS BEFORE SUBMISSION: verify the clinical meaning of Sigmoid and Normal; document reference-standard construction, reader experience, reread timing and blinding; confirm the provisional ID linkage; supply recruitment/exclusion, demographic/scanner and ethics information. Preserve the distinction between the original endpoint and retrospective commentary.',
    ])
    return '\n\n'.join(lines)
