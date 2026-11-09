"""Visible notebook cells for the read-only clinical error analysis and editorial draft."""


def append_review_cells(md, code):
    md('''## 21. Radiologist reread: exploratory error analysis
    The source is the latest relevant forwarded work email, **WG: 20 Cases**, forwarded October 17,
    with radiologist comments dated October 9, 2026. It concerns all 20 disagreements of the
    **FlexiCT final-block, 100%, CV ensemble** with the original test labels.

    The reader's prose is summarized below; category coding is an analyst interpretation, not
    a blinded independent adjudication. No case is relabeled or excluded from performance calculations.
    The absence of a target disease does not establish a normal abdomen. There are no lesion annotations.

    **Case linkage:** 19 exact identifiers; source comment R20 has one unique unmatched candidate,
    Perforation_R20. The user considers that linkage plausible if unique; it remains explicitly
    provisional. Public-facing figures use R01–R20. Original IDs and verbatim email excerpts stay in
    the local audit folder. Do not distribute that audit folder as the supplement.

    **Pillar-0:** the comparator is the saved final Atlas block, 100%, CV ensemble. Its full-cohort
    performance belongs in Table 3. The selected 20-case comparison belongs in the supplement and
    cannot estimate an unbiased model-versus-model error rate.''')
    code('''
import review_helpers_4 as rh
assert len(main) == 13 and len(ensembles) == 65 and pending_main.empty
review, manual_audit, replay_checks, review_input_hashes, review_base, flex_run, pillar_run = rh.load_review(
    ROOT, NOTEBOOK_DIR, main, predictions, patient_ids, y_true)
save_table(review, 'local_radiologist_linkage_and_verbatim',
    'LOCAL ONLY: original case IDs, source comments, analyst synthesis and provisional linkage.', 'audit')
save_table(manual_audit, 'pillar_manual_transcription_audit',
    'LOCAL ONLY: supplied values and blanks versus canonical Pillar-0 final-block CV-ensemble probabilities.', 'audit')
save_table(replay_checks, 'attribution_replay_checks',
    'Existing five-fold probabilities versus canonical predictions; original volume SHA-256 verified.', 'audit')
summary = review.groupby('review_category',sort=False).size().rename('cases').reset_index()
display(summary)
print('Exact ID links:', int((review.linkage_status == 'exact_id').sum()),
      '| Provisional links:', int((review.linkage_status != 'exact_id').sum()))
display(manual_audit[manual_audit.status == 'mismatch'][['review_id','field','user_value','saved_value']])
print('All original cohort labels and source prediction files are retained.')
''')
    md('''### Supplementary Tables S12–S14
    Table S12 contains the narrative reread and original labels. Table S13 uses canonical Pillar-0
    predictions, while preserving the user's transcription in the local audit. The largest discrepancy
    concerns R08: the saved Pillar-0 model predicts Sigmoid, not Normal. Do not treat rounded table
    probabilities as more authoritative than the saved prediction vector.

    Table S14 reports existing explanation diagnostics. Fold agreement is descriptive because
    the folds share training patients. Only the final classifier of one fold was randomized;
    this is a limited check, not full-network attribution validation.''')
    code('''
review_columns = ['review_id','true_class','predicted_class','predicted_probability',
    'true_probability','fold_votes_for_ensemble_prediction','review_category',
    'radiologist_summary','interpretation','linkage_status']
display(save_table(review[review_columns], 'Table_S12_radiologist_error_review',
    'All 20 FlexiCT-selected disagreements; original labels retained. Categories summarize a narrative reread with undocumented blinding and one provisional link.'))
pillar_columns = ['review_id','true_class','predicted_class','predicted_probability','true_probability',
    'pillar_predicted_class','pillar_predicted_probability','pillar_true_probability',
    'pillar_probability_flexict_class','pillar_matches_original_label','same_model_prediction']
display(save_table(review[pillar_columns], 'Table_S13_pillar_selected_case_comparison',
    'Canonical predictions on cases selected because FlexiCT disagreed with the original labels. Conditional descriptive counts only; full-cohort comparison is in Table 3.'))
diagnostic_columns = ['review_id','fold_votes_for_ensemble_prediction','mean_pairwise_fold_map_spearman',
    'randomized_head_map_spearman','top_mask_probability_drop','random_mask_mean_probability_drop',
    'air_fill_probability_drop','positive_drop_exceeds_random','masked_input_fraction',
    'attribution_fraction_below_minus900hu','crop_removed_xyz_mm']
display(save_table(review[diagnostic_columns], 'Table_S14_attribution_diagnostics',
    'Existing 3D token Grad-CAM and occlusion diagnostics. Masking changes input distribution; correlations do not establish clinical localization.'))
''')
    md('''### Main Figure 6. Four complementary disagreement mechanisms
    The four illustrative cases were chosen for distinct review categories: R01 (reference-label
    discrepancy), R13 (confirmed false positive), R08 (coexisting target diagnoses), and R14 (confirmed
    missed perforation). This is a qualitative editorial selection, not a representative sampling claim.

    Maps are reused from the executed error-review notebook. They target the class probability at
    the final transformer block's norm1 output, with a 3D token Grad-CAM adaptation. The native grid
    is 20³ at 16-mm spacing; interpolation is for display only. Raw fold maps are averaged before
    rectification; the resulting positive target maps are normalized independently. Contrast is the signed difference of raw maps.
    Slices pass through the predicted-map maximum, **not a radiologist-annotated lesion**.
    CT is the cropped 160³ model input in LPS orientation at 2-mm spacing, shown at WL 40/WW 400.
    Heatmap patterns cannot confirm the reader's proposed anatomical explanation.''')
    code('''
representative_keys = ['Normal_R01','Normal_R13','Sigmoid_R08','Perforation_R14']
representative = review.set_index('case_key').loc[representative_keys].reset_index()
save_figure(rh.case_panel(representative,review_base), 'Figure_6_clinical_error_examples',
    'Four editorially selected disagreement mechanisms: R01 reread supports appendicitis; R13 confirmed false positive with a clip proposed as a mimic; R08 diverticulitis and perforation coexist; R14 surgically confirmed covered perforation missed as Normal. CT and maps share a plane at the predicted-map maximum. Independent positive scales 0–1; signed contrast red favors predicted class, blue recorded class. No lesion localization or causal interpretation is established.', 'main')
''')
    md('''### Figure S9. Perturbation and attribution diagnostics
    Six 40³-voxel cubes (9.375% of the input) were masked using the predicted-class map. They were
    compared with the mean of three random masks and with air-filled masks. A positive change is a
    decrease in predicted-class probability. Negative values show why map intensity alone cannot
    be read as a causal explanation. The 80-mm cubes can cover multiple structures.''')
    code('''
fig, axes = plt.subplots(1,2,figsize=(13,6))
x = np.arange(len(review))
for column,label,marker in [('top_mask_probability_drop','Top regions, mean fill','o'),
                           ('random_mask_mean_probability_drop','Random regions, mean fill','s'),
                           ('air_fill_probability_drop','Top regions, air fill','^')]:
    axes[0].plot(x,review[column],marker=marker,ls='',label=label,ms=5)
axes[0].axhline(0,color='gray',lw=.8)
axes[0].set(xticks=x,xticklabels=review.review_id,ylabel='Original − masked predicted-class probability')
axes[0].tick_params(axis='x',rotation=90); axes[0].legend(fontsize=8)
axes[0].set_title('A. Occlusion sensitivity')
axes[1].scatter(review.mean_pairwise_fold_map_spearman,review.randomized_head_map_spearman,s=35)
for row in review.itertuples():
    axes[1].annotate(row.review_id,(row.mean_pairwise_fold_map_spearman,row.randomized_head_map_spearman),xytext=(3,3),textcoords='offset points',fontsize=7)
axes[1].axhline(0,color='gray',lw=.8)
axes[1].set(xlabel='Mean pairwise fold-map Spearman correlation',
            ylabel='Original versus randomized classifier map correlation',
            title='B. Limited stability / randomization checks')
fig.tight_layout()
save_figure(fig,'Figure_S9_attribution_diagnostics',
    'Existing occlusion and map diagnostics across the 20 selected disagreements. Randomization affects only fold 1 classifier; it does not validate localization. Mean-fill and air-fill are out-of-distribution interventions.')
''')
    md('''### Figures S10 and S11. Complete 20-case image atlas
    S10 provides CT, predicted-class CAM, recorded-class CAM and signed contrast in axial, coronal
    and sagittal planes for every case. S11 provides 12 axial slices through each cropped model input.
    The model crop may omit anatomy. R20 is the provisional original-source/cohort provisional linkage linkage. There are no lesion
    arrows because no verified lesion coordinates were supplied. The displayed anatomy is not used
    to infer a new diagnosis. PNG and PDF files are exported individually for editorial selection.''')
    code('''
for row in review.itertuples():
    fig = rh.orthogonal_panel(row,review_base)
    save_figure(fig,f'Figure_S10_{row.review_id}_three_plane_attribution',
        f'{row.review_id}: {row.radiologist_summary} Linkage: {row.linkage_status}. '
        'Existing model-space maps; independently normalized positive targets and signed contrast; planes through predicted-map maximum, not lesion annotations.')
''')
    code('''
for row in review.itertuples():
    save_figure(rh.ct_overview(row,review_base),f'Figure_S11_{row.review_id}_CT_overview',
        f'{row.review_id}: 12 axial slices through the cropped 2-mm model input, WL 40 / WW 400. Original full-scan anatomy may extend beyond the crop.')
''')
    md('''## 22. Proposed Results and Discussion
    This draft separates numerical findings from interpretation. It is generated from this execution,
    with explicit qualifications for retrospective endpoint changes, selected-case review and
    explanatory maps. Clinical recruitment, reference-standard, reader and ethics details require
    author input; the notebook cannot establish them from filenames or predictions.''')
    code('''
draft = rh.manuscript_text(main,ensembles,paired,review,main_class,source_sensitivity)
draft_path = EXPORT_ROOT/'Results_and_Discussion_4.txt'
draft_path.write_text(draft,encoding='utf-8')
record_artifact(draft_path,'Proposed Results and Discussion with computed estimates and editorial limitations.')
draft_html = EXPORT_ROOT/'Results_and_Discussion_4.html'
draft_html.write_text('<!doctype html><meta charset="utf-8"><title>Results and Discussion</title><style>body{max-width:900px;margin:45px auto;font:17px/1.6 Georgia;color:#182e3b}p{margin-bottom:1.1em}</style>'+''.join('<p>'+html.escape(p)+'</p>' for p in draft.split('\\n\\n')),encoding='utf-8')
record_artifact(draft_html,'Readable proposed Results and Discussion; author verification remains necessary.')
display(Markdown(draft))
''')
    md('''## 23. Main-paper versus supplementary allocation
    The proposed package respects the journal's Original Research allowance of six main figures and
    four main tables. One paired-effects plot could be moved to the supplement if the journal/editor
    prefers more clinical imagery. Complete case review, calibration, alternative checkpoint routes,
    zero-shot and sensitivity details are supplementary. The Pillar-0 case comparison stays supplementary.
    The complete notebook and local audit are reproducibility materials, not a submission-ready supplement:
    clinical authors must confirm deidentification, permissions and the journal's final packaging format.''')
    code('''
placement = pd.DataFrame([
 ('Main','Table 1','Cohort composition','Defines all denominators; add demographics when available'),
 ('Main','Table 2','Methods and initialization','Makes recipe / adaptation differences explicit'),
 ('Main','Table 3','Full-development ensemble performance','Primary comparative evidence with 95% CIs'),
 ('Main','Table 4','Class-specific sensitivity','Makes perforation and minority-class performance visible'),
 ('Main','Figure 1','Partition and development sizes','Available cohort design; not recruitment flow'),
 ('Main','Figure 2','Overall performance intervals','Macro-F1 and supporting endpoints'),
 ('Main','Figure 3','Panels 3a and 3b: data-efficiency curves','Partial adaptation/probes and full-network comparisons; five sizes'),
 ('Main','Figure 4','Paired effect sizes','Exploratory paired differences'),
 ('Main','Figure 5','Class performance','Sensitivity and class AP'),
 ('Main','Figure 6','Four CT / Grad-CAM examples','Illustrates differing error mechanisms, without localization claims'),
 ('Supplement','Tables S1–S11; S15–S17','Full numeric / statistical evidence','Other routes, calibration, zero-shot, paired sizes and uncertainty'),
 ('Supplement','Table S12','All 20 narrative rereads','Original labels and provisional linkage remain visible'),
 ('Supplement','Table S13','Pillar-0 selected-case comparison','Conditional descriptive comparison, not superiority analysis'),
 ('Supplement','Table S14','Attribution diagnostics','Occlusion, fold stability, limited randomization, crop context'),
 ('Supplement','Figures S1–S8','Performance sensitivities','Confusion, ROC/PR, overlap, routes, zero-shot and calibration'),
 ('Supplement','Figure S9','Occlusion / map diagnostics','Quantitative evidence and counterexamples'),
 ('Supplement','Figures S10–S11','All 20 image cases','Three-plane maps and cropped-input overviews'),
 ('Discussion','Narrative','Reference standard, overlapping labels, difficult anatomy','Interpret the exploratory findings; do not retrospectively improve accuracy'),
 ('Local audit','Provenance and verbatim comments','Source IDs, email excerpts, transcription discrepancies and hashes','Retain locally; do not submit as public supplementary material'),
],columns=['placement','item','content','rationale'])
display(save_table(placement,'manuscript_placement','Proposed editorial allocation for Radiology: Artificial Intelligence.','audit'))
main_figures = {__import__('re').match(r'Figure_(\\d+)',Path(a['path']).name).group(1) for a in ARTIFACTS if a['path'].startswith('main/figures/')}
main_tables = {Path(a['path']).stem for a in ARTIFACTS if a['path'].startswith('main/tables/')}
assert len(main_figures)==6 and len(main_tables)==4
assert len(review)==20 and replay_checks.source_volume_hash_verified.all()
print(f'Editorial allocation verified: {len(main_figures)} main figures, {len(main_tables)} main tables; all 20 error cases included.')
''')
