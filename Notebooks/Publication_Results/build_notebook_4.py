"""Build the version-4 notebook. Does not execute it or alter model results."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))
def code(text): cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

md('''
# Systematic publication results · version 4

**Main reporting focus:** five-fold CV ensembles evaluated on the fixed held-out cohort.
Best-validation-fold checkpoints and full-development refits are supplementary sensitivity analyses.
This notebook reads only `Systematic_Methodology` artifacts; it does not train, load checkpoints,
generate inference, tune prompts, or modify source results. Run every cell from top to bottom.
This version integrates the completed run set and the radiologist's exploratory reread.

**Version boundary:** `Publication_Results_4.ipynb` → `output_4/`; deliverable filenames use `_4`.
Future notebook versions should use a new numbered notebook and export folder.

**Version-4 scope:** 13 supervised pipelines, including both 3DINO linear probing and LoRA.
The publication-derived Merlin full-network addition (FullFineTune CV_2) is retained alongside
its frozen and corrected adapter pipelines. Figures 3a and 3b separate partial adaptation/frozen
comparators from full-network Merlin/MedicalNet fine-tuning and DenseNet training from scratch.
All five sizes and all available checkpoint routes are included. Earlier notebooks remain unchanged.
The previous omission of 3DINO linear probing was an analysis-scope omission, not a failed run.
COLIPRI native pooling is not uniformly a strict linear probe. Source provenance and method-specific
CV identities are preserved; historical Merlin full-fine-tuning grids and pre-fix adapters are excluded.
Only COLIPRI, FlexiCT, Merlin and Pillar-0 enter the separate zero-shot supplement.
DenseNet uses random initialization. No inference, retraining or reference-label changes are performed.

The earlier project documentation/statistical plan used refit as primary. Ensemble-centered reporting
was requested after existing results were available. This is a documented reporting change, not a
prospectively registered primary analysis. Inferential comparisons below are exploratory and conditional
on the fitted models. Test-sample confidence intervals do not measure optimization-seed variability.

[RSNA author instructions](https://pubs.rsna.org/page/ai/author-instructions) were checked in the previous release on 2026-10-18; the editorial package retains four main tables
and six numbered main figures, with Figure 3 split into panels a and b.
[CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300) guides the reporting checklist at the end.
''')
md('''## 1. Imports
Only analysis libraries are required. Install `requirements_4.txt` in the notebook environment if needed.
The small companion modules contain ingestion, statistics and reusable plotting functions; each
scientific calculation, table and figure is introduced and run separately below.''')
code('''
import sys
import json
import platform
import html
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
''')
md('''## 2. Locate the systematic workspace
Inputs are discovered under `Results/` and `Dataset/Splits/`. Generated results stay beside this
notebook, following the existing notebook-local export convention.''')
code('''
def locate_root(start):
    for parent in (start.resolve(), *start.resolve().parents):
        for candidate in (parent, parent / 'Systematic_Methodology'):
            if (candidate / 'Dataset' / 'Splits').is_dir() and (candidate / 'Notebooks').is_dir():
                return candidate
    raise FileNotFoundError('Cannot locate Systematic_Methodology')

ROOT = locate_root(Path.cwd())
NOTEBOOK_DIR = ROOT / 'Notebooks' / 'Publication_Results'
sys.path.insert(0, str(NOTEBOOK_DIR))
import analysis_helpers_4 as ah
import plot_helpers_4 as ph
EXPORT_ROOT = NOTEBOOK_DIR / 'output_4'
print('Inputs:', ROOT / 'Results')
print('Outputs:', EXPORT_ROOT)
''')
md('''## 3. Analysis configuration
Macro-F1 is the headline endpoint. Balanced accuracy and macro average precision are key supporting
endpoints. AP is the non-interpolated average-precision summary, **not trapezoidal PR area**.
AUROC is the unweighted mean of five one-vs-rest AUCs. Classification uses probability argmax;
undefined class F1 is zero. Precision with no positive predictions is displayed as missing.

Bootstrap: 10,000 class-stratified patient samples, percentile 95% intervals, identical resamples
for every model. This fixes observed class counts; PPV/AP/calibration remain cohort-prevalence dependent.
Permutation tests swap paired predictions within patient and test macro-F1 differences, two-sided.
Holm adjustment uses the full configured comparison-family size, including pending comparisons.
''')
code('''
ANALYSIS_VERSION = 4
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 10_000
SEED = 20260928
CALIBRATION_BINS = 10
PRIMARY_SOURCE = 'cv_ensemble'
MAIN_PERCENTAGE = 100
FIGURE_DPI = 300
RUN_OVERRIDES = {}  # 'Model|Method|100|cv_ensemble': exact run_id, never chosen by test score
EXPORT_PATIENT_LEVEL = False  # publication exports are aggregate by default
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
sns.set_palette('colorblind')
pd.set_option('display.max_colwidth', 90)
pd.set_option('display.max_rows', 30)
''')
md('''### Export helpers
Every artifact is registered with a caption and checksum in the final manifest. The manifest and index
define the current export set; old unreferenced files from earlier executions are not current results.
Bootstrap caches are keyed by source hashes, analysis-code hash, seed, and settings.''')
code('''
ARTIFACTS = []
for sub in ['main/tables', 'main/figures', 'supplementary/tables', 'supplementary/figures', 'audit', 'cache']:
    (EXPORT_ROOT / sub).mkdir(parents=True, exist_ok=True)

def record_artifact(path, caption):
    ARTIFACTS.append({'path': path.relative_to(EXPORT_ROOT).as_posix(),
                      'caption': caption, 'sha256': ah.digest(path)})

def save_table(frame, stem, caption, section='supplementary'):
    folder = EXPORT_ROOT / ('audit' if section == 'audit' else section + '/tables')
    path = folder / f'{stem}_4.csv'
    frame.to_csv(path, index=False)
    record_artifact(path, caption)
    page = path.with_suffix('.html')
    page.write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:Arial;margin:32px;color:#172b3a}table{border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f7}caption{text-align:left;font-size:16px;padding:12px 0}</style>' + '<h2>' + html.escape(stem.replace('_',' ')) + '</h2><p>' + html.escape(caption) + '</p>' + frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}'),encoding='utf-8')
    record_artifact(page, caption)
    return frame

def save_figure(fig, stem, caption, section='supplementary'):
    for extension in ['png', 'pdf']:
        path = EXPORT_ROOT / section / 'figures' / f'{stem}_4.{extension}'
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        record_artifact(path, caption)
    plt.show()
    plt.close(fig)
    display(Markdown('**Caption.** ' + caption))
''')
md('''## 4. Import split manifests
The JSON manifests are authoritative. Historical 70/30 planning rows in the text plan are not the
current five-fold protocol. Preserve string patient IDs, including leading zeros.''')
code('''
splits, cohort, split_audit, patient_ids, y_true = ah.load_splits(ROOT)
DEVELOPMENT_COUNTS = {p: splits[p]['cross_validation']['development_total'] for p in ah.PERCENTAGES}
N_TEST = len(patient_ids)
display(split_audit)
save_table(split_audit, 'split_integrity', 'Manifest hashes and patient partition checks.', 'audit');
''')
md('''### Table 1. Cohort composition
The percentages refer to the development-sampling policy. Perforation remains at 40 development
patients, so the 20% subset contains 248/1,080 development patients. Each ensemble member trains on
four development folds; the ensemble as a whole draws on the complete selected development pool.
`Sigmoid` retains the stored category name: its clinical definition requires manuscript confirmation.
Age, sex and scanner characteristics are not inferred from labels or fabricated.''')
code('''
table1 = cohort.rename(columns={'Cholezystitis': 'Cholecystitis'})
display(save_table(table1, 'Table_1_cohort',
    'Development and fixed held-out patient counts by class. Perforation is fixed across development sizes.', 'main'))
''')
md('''### Figure 1. Available cohort and evaluation design
This is a partition diagram, not a recruitment/exclusion flowchart; upstream screening counts and
reference-standard details must be supplied from the clinical study records.''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].axis('off')
boxes = [(0.5, .9, f'Included cohort: {DEVELOPMENT_COUNTS[100] + N_TEST:,} patients'),
         (.24, .58, f'Development: {DEVELOPMENT_COUNTS[100]:,}\\nNested subsets, five folds'),
         (.77, .58, f'Fixed held-out test: {N_TEST}\\nSame patients for all comparisons'),
         (.24, .2, 'Five CV checkpoints\\nMean class probabilities'),
         (.77, .2, 'Paired assessment\\nSame held-out patients')]
for x, yy, text in boxes:
    axes[0].text(x, yy, text, ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=.5', facecolor='#edf3f8', edgecolor='#47688c'))
for a, b in [((.5,.83),(.24,.68)), ((.5,.83),(.77,.68)), ((.24,.47),(.24,.3)), ((.77,.47),(.77,.3)), ((.43,.2),(.60,.2))]:
    axes[0].annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#47688c'))
development = cohort[cohort.percentage > 0].set_index('percentage')[ah.CLASSES]
development.rename(columns=dict(zip(ah.CLASSES, ah.DISPLAY_CLASSES))).plot.bar(stacked=True, ax=axes[1], width=.7)
axes[1].set(xlabel='Nominal development percentage', ylabel='Patients')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
fig.tight_layout()
save_figure(fig, 'Figure_1_cohort_design',
    f'Included cohort partition and nested development subsets. The same {N_TEST} held-out patients are evaluated throughout. Perforation development support remains 40.', 'main')
''')
md('''## 5. Discover saved test artifacts
Discovery is restricted to configured production method/Test paths. Archives, backup directories,
grid-validation outputs, caches, smoke tests, and outside-folder historical DenseNet results are excluded
by path. Zero-shot protocols are read separately for the four foundation-model families only.
Excluded methods and non-publication foundation-model configurations receive `scope_excluded`
audit entries and never enter performance tables or statistics.

Integrity validation checks all canonical files, completion hashes where the pipeline supplies them,
the exact manifest patient set and truths, CSV/JSON agreement, class mapping, finite probabilities,
normalization, and saved-prediction/argmax agreement. Older Merlin/Pillar-0 unsealed artifacts receive
the explicit status `legacy_unsealed_content_validated`; this is not equivalent to a verified seal.''')
code('''
inventory, predictions, discovery_audit, stored_metrics, input_hashes = ah.discover(ROOT, patient_ids, y_true)
if inventory.empty:
    raise RuntimeError('No valid held-out results found. Review discovery_audit before continuing.')
display(discovery_audit.groupby(['status', 'reason'], dropna=False).size().rename('runs').reset_index())
save_table(discovery_audit, 'discovery_audit', 'Accepted and excluded artifacts with reasons.', 'audit')
save_table(inventory, 'all_run_provenance', 'Exact provenance for every valid result; no score-based selection.', 'audit');
''')
md('''## 6. Recompute point estimates from patient predictions
Stored summary metrics are a cross-check rather than the analytical source of truth. A mismatch beyond
1e-6 or a different confusion matrix stops the notebook for review; no failed check is silently repaired.''')
code('''
point_rows, class_rows, matrices, metric_checks = [], [], {}, []
for run in inventory.itertuples():
    p, pred = predictions[run.run_id]
    estimates = ah.point_metrics(y_true, p, CALIBRATION_BINS)
    point_rows.append({'run_id': run.run_id, **estimates})
    classes, matrix = ah.class_metrics(y_true, p)
    class_rows.extend({'run_id': run.run_id, **row} for row in classes)
    matrices[run.run_id] = matrix
    stored = stored_metrics[run.run_id]
    for name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'average_precision_macro']:
        if name in stored:
            metric_checks.append({'run_id': run.run_id, 'metric': name,
                'recomputed': estimates[name], 'stored': stored[name],
                'passed': bool(np.isclose(estimates[name], stored[name], atol=1e-6, rtol=0))})
    if 'confusion_matrix' in stored:
        metric_checks.append({'run_id': run.run_id, 'metric': 'confusion_matrix',
                             'passed': bool(np.array_equal(matrix, stored['confusion_matrix']))})
points = pd.DataFrame(point_rows)
class_results = pd.DataFrame(class_rows).merge(inventory, on='run_id', validate='many_to_one')
checks = pd.DataFrame(metric_checks)
save_table(checks, 'recomputed_metric_checks', 'Patient-derived estimates versus source summaries.', 'audit')
assert checks.passed.all(), 'Summary mismatch: inspect recomputed_metric_checks_4.csv'
print(f'Recomputed {len(points)} runs; all {len(checks)} summary checks passed.')
''')
md('''## 7. Define comparable runs without consulting test scores
Foundation-model supervised runs require publication-derived provenance; unrelated grid/default runs
cannot replace missing eligible results. Native publication-defined grid searches remain eligible.
For reference-model pipelines, prefer publication settings, then development-selected grids,
then defaults, retaining the actual evidence label. Multiple candidates at the same preferred level remain
unresolved unless an explicit provenance-based override is configured. Unknown origins are not
automatically promoted. CV IDs may change across percentages and are never assumed interchangeable.
Only in-scope alternatives remain in supplementary tables; excluded methods are not retained there.

This is a comparison of model-specific adaptation pipelines, not a controlled architecture-only
experiment: input processing, pretrained data, readout and tuning budgets differ.''')
code('''
selected, selection_audit = ah.choose_runs(inventory, RUN_OVERRIDES)
coverage = ah.coverage_table(inventory, selected, discovery_audit)
save_table(selection_audit, 'run_selection_rules', 'Configuration-based inclusion; unresolved cells are explicit.', 'audit')
save_table(coverage, 'run_coverage', 'Expected model/method/percentage/source combinations and current status.', 'audit')
display(coverage[coverage.source == PRIMARY_SOURCE].pivot_table(
    index=['model', 'method'], columns='percentage', values='status', aggfunc='first'))
''')
md('''### Table 2. Model and adaptation methods
The table lists planned methods, including pending methods. The detailed provenance export supplies
exact parameters and profile signatures. Publication-derived settings do not imply an exact replication
of the original paper on this five-class abdominal task.''')
code('''
method_rows = []
for model, methods in ah.METHODS.items():
    for method in methods:
        chosen = selected[(selected.model == model) & (selected.method == method) & (selected.source == PRIMARY_SOURCE)]
        method_rows.append({'model': model, 'method': ah.METHOD_LABELS[method],
            'initialization': 'Random initialization' if model == 'DenseNet' else 'Pretrained encoder',
            'available_ensemble_percentages': ', '.join(map(str, sorted(chosen.percentage))) or 'Pending',
            'parameter_origins': ', '.join(sorted(chosen.source_family.unique())) or 'Pending',
            'comparison_group': 'Foundation-model pipeline' if model in ah.FOUNDATION_MODELS else 'Reference pipeline',
            'evidence_note': ah.EVIDENCE_NOTES[(model, method)]})
display(save_table(pd.DataFrame(method_rows), 'Table_2_methods',
    'Planned model/adaptation methods, initialization and available CV-ensemble coverage.', 'main'))
''')
md('''### Completed reference-pipeline coverage
The completed DenseNet and MedicalNet ensembles are verified across all five fractions. No training
or evaluation command is run by this notebook.''')
code('''
cnn_status = coverage[(coverage.model.isin(['DenseNet','MedicalNet'])) & (coverage.source == PRIMARY_SOURCE)]
display(cnn_status[['model','percentage','status']])
assert (cnn_status.status == 'available').all()
''')
md('''## 8. Define paired comparison families
The 100% exploratory comparison family is fixed below: each retained frozen and adapted foundation
method versus DenseNet, each retained adaptation versus its frozen method, and MedicalNet full fine-tuning versus DenseNet. Version 4 adds 3DINO probe contrasts and
three Merlin full-network contrasts. All 19 tests are recalculated and Holm-adjusted as one expanded
exploratory family; adjusted P values from version 3 are not reused.
Unavailable pairs are exported as pending. No test-score winner is selected. Comparisons at smaller
development sizes and alternative checkpoint sources are descriptive sensitivity analyses.''')
code('''
FROZEN_METHODS = ah.FROZEN_METHODS
PAIR_PLAN = []
for model, frozen in FROZEN_METHODS.items():
    PAIR_PLAN.append((model, frozen, 'DenseNet', 'FromScratch', 'foundation_vs_scratch'))
    adapted = ah.ADAPTATION_METHODS[model]
    PAIR_PLAN.append((model, adapted, model, frozen, 'within_model_adaptation'))
    PAIR_PLAN.append((model, adapted, 'DenseNet', 'FromScratch', 'adapted_foundation_vs_scratch'))
for model in ['MedicalNet']:
    PAIR_PLAN.append((model, 'LoRA' if model == '3DINO' else 'FullFineTune', 'DenseNet', 'FromScratch', 'reference_baselines'))
PAIR_PLAN.extend([
    ('Merlin','FullFineTune','DenseNet','FromScratch','full_network_addition'),
    ('Merlin','FullFineTune','Merlin','ConvLoRA_LLRD','full_network_addition'),
    ('Merlin','FullFineTune','Merlin','LinearProbe','full_network_addition')])
assert len(PAIR_PLAN)==19 and len({p[:4] for p in PAIR_PLAN})==19
pair_plan = pd.DataFrame(PAIR_PLAN, columns=['model_a','method_a','model_b','method_b','family'])
save_table(pair_plan, 'planned_comparisons', 'All configured 100% ensemble comparisons, including pending pairs.', 'audit')
display(pair_plan)
''')
md('''## 9. Patient-level bootstrap uncertainty
Use identical stratified multinomial weights for all runs, equivalent to resampling patients with
replacement within each reference class. Vectorized AP calculations account for tied scores.
The first execution computes intervals for every valid run; subsequent executions reuse only
hash-matching caches. Changing predictions or statistical settings invalidates the relevant cache.''')
code('''
bootstrap_weights = ah.stratified_weights(y_true, BOOTSTRAP_REPLICATES, SEED)
assert np.all(bootstrap_weights.sum(axis=1) == N_TEST)
bootstrap = {}
for number, run in enumerate(inventory.itertuples(), 1):
    bootstrap[run.run_id] = ah.bootstrap_cached(run, y_true, predictions[run.run_id][0],
        bootstrap_weights, EXPORT_ROOT / 'cache', SEED, CALIBRATION_BINS)
    if number % 20 == 0 or number == len(inventory):
        print(f'Bootstrap complete: {number}/{len(inventory)} runs', flush=True)
''')
md('''### Assemble interval tables
Point estimates and intervals are kept in numeric columns for reproducibility; rounded publication
tables are derived from these columns. Display ordering follows the configured model/method order,
not held-out performance.''')
code('''
interval_rows = []
for run_id, samples in bootstrap.items():
    row = {'run_id': run_id}
    for metric, values in samples.items():
        row[metric + '_low'], row[metric + '_high'] = np.quantile(values, [.025, .975])
    interval_rows.append(row)
results = inventory.merge(points, on='run_id', validate='one_to_one').merge(pd.DataFrame(interval_rows), on='run_id', validate='one_to_one')
chosen_results = results[results.run_id.isin(selected.run_id)].copy()
ensembles = chosen_results[chosen_results.source == PRIMARY_SOURCE].copy()
main = ensembles[ensembles.percentage == MAIN_PERCENTAGE].copy()
order = {(m, method): i for i, (m, method) in enumerate((m, t) for m, ts in ah.METHODS.items() for t in ts)}
main = main.assign(display_order=[order[(r.model,r.method)] for r in main.itertuples()]).sort_values('display_order')
save_table(results, 'Table_S1_all_numeric_results', 'All valid runs, point estimates and patient-bootstrap 95% CIs.')
''')
md('''## 10. Main CV-ensemble performance
### Table 3. Full development pool
The table uses the same held-out patients for every row. Full provenance is in Table S1 and the audit;
refit results never fill a missing ensemble cell.''')
code('''
table3 = ah.formatted_table(main, ['f1_macro','balanced_accuracy','average_precision_macro','roc_auc_macro','accuracy'])
table3['status'] = 'Available'
pending_main = coverage[(coverage.source == PRIMARY_SOURCE) & (coverage.percentage == MAIN_PERCENTAGE) & (coverage.status != 'available')]
for row in pending_main.itertuples():
    table3 = pd.concat([table3, pd.DataFrame([{'model':row.model,'method':row.method,
        'percentage':MAIN_PERCENTAGE,'source':PRIMARY_SOURCE,'status':row.status}])], ignore_index=True)
table3 = table3[['model','method','n_test'] + [c for c in table3 if '(95% CI)' in c]].copy()
table3['method'] = table3.method.map(ah.METHOD_LABELS)
table3['model'] = table3.model.replace({'Pillar0':'Pillar-0'})
table3 = table3.rename(columns={'model':'Model','method':'Method','n_test':'Test patients'})
display(save_table(table3, 'Table_3_main_performance',
    f'CV ensembles at the full development pool on {N_TEST} held-out patients; estimates with 95% stratified patient-bootstrap CIs. Merlin ConvLoRA+LLRD uses the corrected CV_2 implementation; 3DINO includes linear probing and LoRA; Merlin full-network CV_2 is separately identified. Full provenance: Table S1.', 'main'))
''')
md('''### Figure 2. Performance with uncertainty
Overlapping individual confidence intervals do not test a paired difference; use the paired analysis
below for comparisons. Rows are not ranked by test score.''')
code('''
fig = ph.intervals(main, ['f1_macro','balanced_accuracy','average_precision_macro'])
save_figure(fig, 'Figure_2_main_performance',
    f'Full-development CV-ensemble performance on {N_TEST} matched held-out patients. Points and lines show estimates and 95% class-stratified patient-bootstrap intervals; rows follow configuration order.', 'main')
''')
md("""## 11. Data efficiency: Figures 3a and 3b
Figure 3a includes partial adaptation and frozen-encoder comparators, including the 3DINO linear probe.
Figure 3b compares full-network Merlin and MedicalNet fine-tuning with DenseNet trained from scratch.
The horizontal axis uses actual development counts. Points are observed ensemble estimates;
lines only connect observations. No smoothing, monotonic fit, extrapolation or confidence bands are
shown. Source-specific hyperparameters and identities remain in the provenance audit.
""")
code('''
figure3_data=ensembles.copy()
figure3_data['development_n']=figure3_data.percentage.map(DEVELOPMENT_COUNTS)
figure3_data['macro_f1_percent']=100*figure3_data.f1_macro
assert len(figure3_data)==65
save_table(figure3_data[['model','method','percentage','development_n','f1_macro','cv_id','run_id']],
    'Figure_3_source_data','Exact Figure 3 values and provenance.','audit')
def save_figure3_variants(fig,stem,caption):
    variants=[('.png',300,{}),('_600dpi.png',600,{}),('.pdf',300,{}),('.svg',300,{}),
        ('.jpg',300,{'pil_kwargs':{'quality':95,'subsampling':0}}),
        ('.tiff',600,{'pil_kwargs':{'compression':'tiff_lzw'}})]
    for suffix,dpi,kwargs in variants:
        path=EXPORT_ROOT/'main/figures'/(stem+'_4'+suffix)
        fig.savefig(path,dpi=dpi,facecolor='white',**kwargs)
        record_artifact(path,caption)
    plt.show();plt.close(fig)
    display(Markdown('**Caption.** '+caption))
save_figure3_variants(ph.draw_a(figure3_data),'Figure_3a_partial_adaptation_and_linear_probes',ph.CAPTIONS['Figure_3a'])
save_figure3_variants(ph.draw_b(figure3_data),'Figure_3b_full_network_comparison',ph.CAPTIONS['Figure_3b'])
caption_path=EXPORT_ROOT/'Figure_3_captions_4.txt'
caption_path.write_text('\\n\\n'.join(ph.CAPTIONS.values()),encoding='utf-8')
record_artifact(caption_path,'Updated individual and combined captions for Figure 3a/b.')
''')
md('''### Supplementary learning endpoints
Balanced accuracy emphasizes equal class recall; average precision summarizes ranking at the observed
test prevalence. These complement the macro-F1 learning curves.''')
code('''
for metric, number in [('balanced_accuracy','S1'), ('average_precision_macro','S2')]:
    fig = ph.learning_panels(ensembles, metric, DEVELOPMENT_COUNTS)
    save_figure(fig, f'Figure_{number}_learning_{metric}',
        f'{ah.METRIC_LABELS[metric]} across nested development pools; same fixed test cohort, pointwise 95% patient-bootstrap intervals.')
''')
md('''### Reduced-data differences and area under the learning curve
Within-method differences use paired resamples against that method's 100% result. AULC is the trapezoidal
integral of macro-F1 over **actual development counts**, divided by the count span. It is calculated only
when all five sizes exist; missing curves are not interpolated. These are descriptive data-efficiency
summaries, not noninferiority tests or estimates of the minimum clinically sufficient sample size.''')
code('''
retention_rows, aulc_rows = [], []
for (model, method), group in ensembles.groupby(['model','method']):
    group = group.set_index('percentage')
    if 100 not in group.index:
        continue
    reference = group.loc[100]
    for pct, run in group.iterrows():
        if pct == 100:
            continue
        delta = bootstrap[run.run_id]['f1_macro'] - bootstrap[reference.run_id]['f1_macro']
        lo, hi = np.quantile(delta, [.025,.975])
        retention_rows.append({'model':model,'method':method,'percentage':pct,
            'development_n':DEVELOPMENT_COUNTS[pct], 'difference_vs_100':run.f1_macro-reference.f1_macro,
            'ci_low':lo,'ci_high':hi,'fraction_of_100_f1':run.f1_macro/reference.f1_macro if reference.f1_macro else np.nan})
    if set(group.index) == set(ah.PERCENTAGES) and group.source_family.nunique() == 1:
        ordered = group.loc[ah.PERCENTAGES]
        x = np.array(list(DEVELOPMENT_COUNTS.values()), dtype=float)
        dx = np.diff(x) / (x[-1]-x[0])
        samples = np.stack([bootstrap[r]['f1_macro'] for r in ordered.run_id])
        values = np.sum((samples[:-1]+samples[1:]) / 2 * dx[:,None], axis=0)
        estimate = np.sum((ordered.f1_macro.to_numpy()[:-1]+ordered.f1_macro.to_numpy()[1:])/2*dx)
        lo, hi = np.quantile(values,[.025,.975])
        aulc_rows.append({'model':model,'method':method,'normalized_macro_f1_aulc':estimate,'ci_low':lo,'ci_high':hi})
save_table(pd.DataFrame(retention_rows), 'Table_S2_reduced_data_differences', 'Paired macro-F1 differences versus each method at 100% development.')
display(save_table(pd.DataFrame(aulc_rows), 'Table_S3_learning_curve_area', 'Normalized macro-F1 AULC over actual development counts; complete five-size curves only.'))
''')
md('''## 12. Paired method comparisons
### Calculate differences before hypothesis tests
Each pair uses the exact same patients and bootstrap draws. Positive differences favor method A.
The direction and membership of each pair come from the configuration, never from the observed scores.''')
code('''
paired_rows, paired_samples = [], {}
for pair_index, (ma, ta, mb, tb, family) in enumerate(PAIR_PLAN):
    a = main[(main.model==ma)&(main.method==ta)]
    b = main[(main.model==mb)&(main.method==tb)]
    label = f'{ma} {ah.METHOD_LABELS[ta]} − {mb} {ah.METHOD_LABELS[tb]}'
    row = {'pair_index':pair_index,'comparison':label,'family':family,'status':'pending',
           'model_a':ma,'method_a':ta,'model_b':mb,'method_b':tb}
    if len(a)==1 and len(b)==1:
        a,b = a.iloc[0],b.iloc[0]
        delta = bootstrap[a.run_id]['f1_macro']-bootstrap[b.run_id]['f1_macro']
        lo,hi = np.quantile(delta,[.025,.975])
        row.update(status='available',run_a=a.run_id,run_b=b.run_id,
                   difference=a.f1_macro-b.f1_macro,ci_low=lo,ci_high=hi)
        paired_samples[pair_index] = delta
    paired_rows.append(row)
paired = pd.DataFrame(paired_rows)
display(paired[['comparison','status']])
''')
md('''### Permutation tests and multiplicity
Macro-F1 uses 10,000 paired permutations. Exact McNemar tests describe discordant correctness as a
supporting analysis; they do not test macro-F1. Both sets of P values receive Holm adjustment over the
full planned family. Unadjusted confidence intervals remain effect-size summaries. No bootstrap
sign proportion is mislabeled as a P value or a posterior probability of superiority.''')
code('''
for idx, row in paired[paired.status=='available'].iterrows():
    tests = ah.paired_test(y_true, predictions[row.run_a][1], predictions[row.run_b][1],
                           PERMUTATION_REPLICATES, SEED + int(row.pair_index))
    for name,value in tests.items():
        paired.loc[idx,name] = value
available_mask = paired.status=='available'
for name in ['permutation_p','mcnemar_p']:
    if name in paired:
        paired.loc[available_mask,name+'_holm'] = ah.holm(paired.loc[available_mask,name], len(PAIR_PLAN))
save_table(paired, 'Table_S4_paired_comparisons',
    'Exploratory paired differences and two-sided tests at 100% development. Holm family includes every planned comparison, including pending pairs.')
display(paired[available_mask])
''')
md('''### Figure 4. Adaptation effects
The figure includes available configured exploratory comparisons. All completed DenseNet contrasts are included.''')
code('''
fig = ph.difference_plot(paired[paired.status=='available'])
save_figure(fig, 'Figure_4_paired_effects',
    'Macro-F1 differences (A minus B) for configured 100% CV-ensemble comparisons, with paired 95% patient-bootstrap intervals. Comparisons are exploratory; adjusted P values and pending pairs appear in Table S4.', 'main')
''')
md('''### Supplementary paired effects at smaller development sizes
These matched-size comparisons extend the adaptation and CNN contrasts across the learning curves.
They report paired effect estimates and intervals without adding a large family of hypothesis tests.
Unavailable combinations remain pending; no interpolation or substitution of refit predictions occurs.''')
code('''
size_pair_rows = []
for pct in [20,40,60,80]:
    subset = ensembles[ensembles.percentage==pct]
    for ma,ta,mb,tb,family in PAIR_PLAN:
        a = subset[(subset.model==ma)&(subset.method==ta)]
        b = subset[(subset.model==mb)&(subset.method==tb)]
        row = {'percentage':pct,'development_n':DEVELOPMENT_COUNTS[pct],
               'model_a':ma,'method_a':ta,'model_b':mb,'method_b':tb,'status':'pending'}
        if len(a)==1 and len(b)==1:
            a,b = a.iloc[0],b.iloc[0]
            delta = bootstrap[a.run_id]['f1_macro']-bootstrap[b.run_id]['f1_macro']
            lo,hi = np.quantile(delta,[.025,.975])
            row.update(status='available',run_a=a.run_id,run_b=b.run_id,
                       difference=a.f1_macro-b.f1_macro,ci_low=lo,ci_high=hi)
        size_pair_rows.append(row)
save_table(pd.DataFrame(size_pair_rows),'Table_S15_paired_effects_by_size',
    'Exploratory paired macro-F1 differences at matched reduced development sizes; 95% intervals and explicit pending combinations.');
''')
md("""### Tables S16–S17. Full-network Merlin and 3DINO probe comparisons
Matched fractions and checkpoint routes are compared using the same patient resamples. These are
comparisons of complete selected pipelines, whose native hyperparameters can differ. Pointwise
intervals are descriptive; the 100% ensemble macro-F1 tests are part of Table S4's 19-test family.
""")
code('''
def all_route_contrast(ma,ta,mb,tb):
    rows=[]
    for pct in ah.PERCENTAGES:
        for route in ah.SOURCES:
            q=chosen_results[(chosen_results.percentage==pct)&(chosen_results.source==route)]
            a=q[(q.model==ma)&(q.method==ta)];b=q[(q.model==mb)&(q.method==tb)]
            assert len(a)==1 and len(b)==1
            a,b=a.iloc[0],b.iloc[0]
            for metric in ['f1_macro','accuracy','balanced_accuracy']:
                lo,hi=np.quantile(bootstrap[a.run_id][metric]-bootstrap[b.run_id][metric],[.025,.975])
                rows.append(dict(percentage=pct,development_n=DEVELOPMENT_COUNTS[pct],source=route,
                    model_a=ma,method_a=ta,model_b=mb,method_b=tb,metric=metric,
                    estimate_a=a[metric],estimate_b=b[metric],difference=a[metric]-b[metric],
                    ci_low=lo,ci_high=hi,run_a=a.run_id,run_b=b.run_id))
    return pd.DataFrame(rows)
full_network_comparison=all_route_contrast('Merlin','FullFineTune','DenseNet','FromScratch')
probe_comparison=all_route_contrast('3DINO','LinearProbe','3DINO','LoRA')
display(save_table(full_network_comparison,'Table_S16_Merlin_full_network_vs_DenseNet',
    'Merlin full fine-tuning versus DenseNet from scratch at every matched size and route; paired 95% pointwise intervals.'))
display(save_table(probe_comparison,'Table_S17_3DINO_linear_probe_vs_LoRA',
    '3DINO linear probing versus LoRA at every matched size and route; paired 95% pointwise intervals.'))
''')
md('''## 13. Class-specific performance
### Table 4. Class sensitivity with uncertainty
The compact main table gives one row per method and one sensitivity column per class; full diagnostic
metrics and denominators are retained in Table S5.
Sensitivity, specificity and precision receive Wilson 95% intervals. Perforation sensitivity also has
an exact binomial interval in the numeric supplement. Every denominator is exported. F1, class AP and
class AUROC are point estimates here and are not presented with binomial intervals.''')
code('''
main_class = class_results[class_results.run_id.isin(main.run_id)].copy()
main_class['Sensitivity (95% CI)'] = main_class.apply(lambda r:
    f'{r.sensitivity:.3f} ({r.sensitivity_low:.3f}, {r.sensitivity_high:.3f})', axis=1)
table4 = main_class.pivot(index=['model','method'],columns='display_class',values='Sensitivity (95% CI)').reindex(columns=ah.DISPLAY_CLASSES).reset_index()
table4 = table4.set_index(['model','method']).reindex(pd.MultiIndex.from_frame(main[['model','method']])).reset_index()
table4['method'] = table4.method.map(ah.METHOD_LABELS)
table4['model'] = table4.model.replace({'Pillar0':'Pillar-0'})
table4 = table4.rename(columns={'model':'Model','method':'Method'})
display(save_table(table4, 'Table_4_class_performance',
    'Class sensitivities with Wilson 95% intervals for 100% CV ensembles. Supports: Appendicitis 47, Cholecystitis 37, Normal 90, Perforation 18, Sigmoid 86. Table S5 supplies specificity, precision, F1, AP, AUROC and all confusion-count denominators.', 'main'))
save_table(class_results, 'Table_S5_all_class_metrics', 'All valid runs: class metrics, counts, Wilson intervals and exact Perforation sensitivity intervals.');
''')
md('''### Figure 5. Class sensitivity and average precision
The two panels distinguish correct classification from probability ranking. Perforation estimates
are based on 18 test patients; see Table 4 for uncertainty rather than interpreting color alone.''')
code('''
fig, axes = plt.subplots(1,2,figsize=(14,max(4,len(main)*.43+1.5)))
for ax, metric, title in zip(axes, ['sensitivity','average_precision'], ['Sensitivity','Average precision']):
    heat = main_class.pivot(index='label',columns='display_class',values=metric).reindex(index=main.label,columns=ah.DISPLAY_CLASSES)
    if heat.empty:
        ph.empty_panel(ax)
    else:
        sns.heatmap(heat,annot=True,fmt='.2f',vmin=0,vmax=1,cmap='Blues',ax=ax,cbar=True)
    ax.set_title(title);ax.set_xlabel('');ax.set_ylabel('');ax.tick_params(axis='x',rotation=30)
axes[1].set_yticklabels([])
fig.tight_layout()
save_figure(fig, 'Figure_5_class_performance',
    'Sensitivity and class average precision for full-development CV ensembles. Class supports: '+
    ', '.join(f'{c} n={int((y_true==i).sum())}' for i,c in enumerate(ah.DISPLAY_CLASSES))+'. See Table 4 for sensitivity intervals.', 'main')
''')
md('''### Supplementary confusion matrices
Counts preserve denominators. Row normalization answers which predicted class each reference class
was assigned to. All available main-analysis methods are shown, not only the best-performing method.''')
code('''
for normalized, suffix in [(False,'counts'), (True,'row_normalized')]:
    fig = ph.confusion_panels(main, matrices, normalize=normalized)
    save_figure(fig, f'Figure_S3_confusion_{suffix}',
        f'Full-development CV-ensemble confusion matrices ({suffix.replace("_"," ")}); rows are reference classes, columns predicted classes.')
''')
md('''### Supplementary precision–recall and ROC curves
These curves use one-versus-rest truth for the mutually exclusive five-class task. PR baselines equal
the class prevalence in this fixed cohort. Curve values and thresholds are exported for reproducibility.''')
code('''
curve_rows = []
for kind in ['PR','ROC']:
    fig, axes = plt.subplots(2,3,figsize=(15,9))
    for k, ax in enumerate(list(axes.flat)[:5]):
        for run in main.itertuples():
            scores = predictions[run.run_id][0][:,k]
            if kind == 'PR':
                precision, recall, thresholds = precision_recall_curve(y_true==k, scores)
                x, yy = recall, precision
                thresholds = np.r_[thresholds, np.nan]
            else:
                x, yy, thresholds = roc_curve(y_true==k,scores)
            method_index = ah.METHODS[run.model].index(run.method)
            ax.plot(x, yy, label=run.label, lw=1.1, alpha=.85, color=ph.COLORS[run.model],
                    linestyle=['-','--',':','-.'][method_index % 4])
            curve_rows.extend({'run_id':run.run_id,'class':ah.CLASSES[k],'curve':kind,
                               'x':xx,'y':vv,'threshold':tt} for xx,vv,tt in zip(x,yy,thresholds))
        if kind == 'PR':
            ax.axhline((y_true==k).mean(),ls='--',color='#888',lw=1)
            ax.set(xlabel='Recall',ylabel='Precision')
        else:
            ax.plot([0,1],[0,1],ls='--',color='#888',lw=1)
            ax.set(xlabel='False-positive rate',ylabel='Sensitivity')
        ax.set_title(ah.DISPLAY_CLASSES[k]);ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    axes.flat[-1].axis('off')
    handles,labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',fontsize=8)
    fig.tight_layout()
    save_figure(fig,f'Figure_S4_{kind}_curves',f'One-versus-rest {kind} curves for 100% CV ensembles; fixed patient cohort. Dashed lines show the relevant no-skill baseline.')
save_table(pd.DataFrame(curve_rows), 'curve_coordinates', 'Patient-probability-derived PR/ROC coordinates and thresholds.');
''')
md('''## 14. Calibration
Assess raw probabilities without fitting calibration on the held-out test set. Multiclass Brier score
is the mean **sum** of five squared errors (range 0–2); log loss uses natural logarithms and clips true-class
probabilities at 1e-15. Top-label ECE uses ten fixed equal-width confidence bins.
Calibration metrics depend on cohort prevalence and are not evidence of deployment readiness.''')
code('''
calibration_table = ah.formatted_table(main, ['brier_multiclass','log_loss','ece'])
display(save_table(calibration_table, 'Table_S6_calibration', 'Raw-probability calibration metrics with patient-bootstrap intervals; no test-set recalibration.'))
reliability_rows = []
for run in main.itertuples():
    bins = ah.reliability(y_true,predictions[run.run_id][0],CALIBRATION_BINS)
    bins['run_id'] = run.run_id
    reliability_rows.append(bins)
reliability_data = pd.concat(reliability_rows,ignore_index=True) if reliability_rows else pd.DataFrame()
save_table(reliability_data,'reliability_bins','Fixed-bin confidence, accuracy, support and Wilson accuracy intervals.');
''')
md('''### Supplementary Figure S8. Reliability by model family
Bins with no observations are omitted. Small-bin estimates are uncertain; point size indicates bin
support, and exported bin tables include Wilson intervals. The diagonal denotes perfect top-label
calibration. Zero-shot relative scores are kept outside this main calibration analysis.''')
code('''
fig, axes = plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
for ax, model in zip(axes.flat, ah.METHODS):
    group = main[main.model==model]
    ax.plot([0,1],[0,1],ls='--',color='#999',lw=1)
    if group.empty:
        ph.empty_panel(ax)
    for j, run in enumerate(group.itertuples()):
        data = reliability_data[(reliability_data.run_id==run.run_id)&(reliability_data.n>0)]
        color = ph.METHOD_COLORS[(model,run.method)]
        ax.plot(data.confidence,data.accuracy,label=ah.METHOD_LABELS[run.method],color=color,lw=1)
        ax.scatter(data.confidence,data.accuracy,s=10+data.n*.4,color=color)
    ax.set_title(model);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.15)
    if not group.empty:
        ax.legend(fontsize=7,loc='upper left')
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.8,'Ten fixed confidence bins.\\nPoint size indicates support.\\nDiagonal: perfect calibration.\\nNo test-set calibration fitting.',va='top')
fig.supxlabel('Mean maximum predicted probability');fig.supylabel('Observed top-label accuracy')
fig.tight_layout()
save_figure(fig,'Figure_S8_reliability',
    'Top-label reliability diagrams for raw 100% CV-ensemble probabilities. Ten equal-width bins; marker area indicates support. Empty bins are omitted; bin uncertainty is supplied in the exported table.')
''')
md('''## 15. Error overlap and ranking uncertainty (exploratory)
Error overlap describes whether methods fail on the same patients. No cross-model ensemble is tuned
on test labels. Bootstrap rank frequencies describe sensitivity to resampling this cohort; they are not
posterior probabilities that a model is truly best.''')
code('''
overlap = np.zeros((len(main),len(main)),dtype=int)
for i,a in enumerate(main.itertuples()):
    for j,b in enumerate(main.itertuples()):
        overlap[i,j] = np.sum((predictions[a.run_id][1]!=y_true)&(predictions[b.run_id][1]!=y_true))
overlap_table = pd.DataFrame(overlap,index=main.label,columns=main.label)
save_table(overlap_table.rename_axis('model_method').reset_index(),'Table_S7_error_overlap',
           'Number of held-out patients misclassified by both methods; diagonal is each method error count.')
fig,ax = plt.subplots(figsize=(12,10))
if len(main):
    sns.heatmap(overlap_table,annot=True,fmt='d',cmap='Blues',ax=ax)
else:
    ph.empty_panel(ax)
fig.tight_layout()
save_figure(fig,'Figure_S5_error_overlap','Pairwise overlap in misclassified patients at 100% development; diagonal entries are total errors per method.')
''')
code('''
rank_rows = []
if len(main):
    scores = np.stack([bootstrap[r]['f1_macro'] for r in main.run_id],axis=1)
    ranks = pd.DataFrame(scores).rank(axis=1,ascending=False,method='average').to_numpy()
    tied_best = np.isclose(scores,scores.max(1,keepdims=True),atol=1e-12,rtol=0)
    best_share = tied_best / tied_best.sum(1,keepdims=True)
    for j,run in enumerate(main.itertuples()):
        rank_rows.append({'run_id':run.run_id,'label':run.label,'median_bootstrap_rank':np.median(ranks[:,j]),
            'fractional_best_frequency':best_share[:,j].mean(),
            'within_0_02_of_best_frequency':(scores[:,j]>=scores.max(1)-.02).mean(),
            'comparison_set_n':len(main)})
display(save_table(pd.DataFrame(rank_rows),'Table_S8_rank_stability',
    'Descriptive cohort-bootstrap rank stability among the 13 included 100% ensemble methods; ties share best frequency.'))
''')
md('''## 16. Supplementary checkpoint-source sensitivity
Compare ensemble predictions with best-validation-fold and refit predictions only when model, method,
percentage, CV ID and effective parameter signature match. This isolates checkpoint source rather than
mixing different trained configurations. Best-fold means chosen by development validation, not test.
All source-specific performance values also appear in Table S1.''')
code('''
source_rows = []
for ensemble in ensembles.itertuples():
    alternatives = chosen_results[(chosen_results.model==ensemble.model)&(chosen_results.method==ensemble.method)&
        (chosen_results.percentage==ensemble.percentage)&(chosen_results.cv_id==ensemble.cv_id)&
        (chosen_results.parameter_signature==ensemble.parameter_signature)&
        (chosen_results.source.isin(['best_cv_fold','refit_full_development']))]
    for alternative in alternatives.itertuples():
        delta = bootstrap[ensemble.run_id]['f1_macro']-bootstrap[alternative.run_id]['f1_macro']
        lo,hi = np.quantile(delta,[.025,.975])
        source_rows.append({'model':ensemble.model,'method':ensemble.method,'percentage':ensemble.percentage,
            'comparison':f'{ensemble.label}: ensemble − {alternative.source}',
            'run_a':ensemble.run_id,'run_b':alternative.run_id,'alternative_source':alternative.source,
            'difference':ensemble.f1_macro-alternative.f1_macro,'ci_low':lo,'ci_high':hi})
source_sensitivity = pd.DataFrame(source_rows)
display(save_table(source_sensitivity,'Table_S9_checkpoint_source_sensitivity',
    'Paired ensemble-minus-alternative macro-F1 differences for identical CV identity and effective parameters.'))
''')
code('''
source_100 = source_sensitivity[source_sensitivity.percentage==100] if len(source_sensitivity) else source_sensitivity
fig = ph.difference_plot(source_100)
save_figure(fig,'Figure_S6_checkpoint_sources',
    'Paired macro-F1 differences between CV ensemble and matched best-fold/refit sources at 100% development, with 95% patient-bootstrap intervals.')
''')
md('''### Legacy-artifact sensitivity
Legacy unsealed predictions pass content/summary checks but lack cryptographic completion evidence.
This table identifies the sealed subset of the main comparison without implying that removing an
entire model family is a fair model-performance test.''')
code('''
sealed_main = main[main.integrity=='hash_seal_verified']
save_table(ah.formatted_table(sealed_main,['f1_macro','balanced_accuracy','average_precision_macro']),
    'Table_S10_sealed_artifacts_only','Descriptive main-analysis subset with verified completion seals; legacy runs remain explicitly identified elsewhere.');
''')
md('''## 17. Zero-shot results (separate supplementary analysis)
Only COLIPRI, FlexiCT, Merlin and Pillar-0 enter this supplement. DenseNet, MedicalNet and 3DINO
are excluded from zero-shot discovery. No supervised checkpoint-source label is assigned to zero-shot runs. The five-class
task adaptations differ across models. Pillar-0's prompt-similarity task remains experimental and was
not demonstrated by its source study. Softmax/normalized relative scores are not assumed calibrated.
Zero-shot scores are not pooled with supervised results or used to choose prompts.''')
code('''
zero_shot = results[results.zero_shot].copy()
zero_table = ah.formatted_table(zero_shot,['f1_macro','balanced_accuracy','average_precision_macro','roc_auc_macro'])
zero_table['protocol'] = zero_shot.run_id.str.split('/').str[-1]
zero_table['interpretation'] = ['Experimental prompt similarity' if m=='Pillar0' else 'Model-specific five-class task adaptation' for m in zero_table.model]
display(save_table(zero_table,'Table_S11_zero_shot','Separate five-class zero-shot results; relative score semantics and task adaptations differ.'))
fig = ph.intervals(zero_shot,['f1_macro','balanced_accuracy','average_precision_macro'])
save_figure(fig,'Figure_S7_zero_shot','Separate zero-shot comparison with patient-bootstrap intervals. Pillar-0 is experimental; all methods use model-specific five-class adaptations.')
''')
md('''## 18. Resource and clinical-reporting availability
Parameter counts, memory and timings require standardized measurements and hardware context before
an efficiency frontier can be interpreted. The notebook reports availability instead of synthesizing
missing values. Calibration fitting, metadata subgroups, external testing and reader comparisons are
not performed unless appropriate additional data and a specified analysis become available.''')
code('''
resource_rows = []
for run in main.itertuples():
    path = Path(run.directory) / 'resource_summary.json'
    resource_rows.append({'run_id':run.run_id,'label':run.label,
        'status':'available_for_schema_review' if path.exists() else 'standardized_resource_summary_missing',
        'resource_json':json.dumps(ah.read_json(path),sort_keys=True) if path.exists() else ''})
save_table(pd.DataFrame(resource_rows),'resource_availability','Availability of per-test standardized resource summaries; no incomparable timings are pooled.','audit')
reporting_gaps = pd.DataFrame([
    ('Recruitment and exclusions','Clinical records required; Figure 1 shows partition only'),
    ('Reference standard and label definitions','Document adjudication, readers, blinding; confirm Sigmoid category meaning'),
    ('Demographic/scanner subgroups','Linked patient metadata required; not inferred from filenames'),
    ('Pretraining overlap','Requires pretrained-cohort documentation; local disjointness alone cannot exclude overlap'),
    ('External testing','No external-cohort predictions loaded by this notebook'),
    ('Reader comparison / clinical utility','No reader or prospective workflow data loaded'),
    ('Training-seed robustness','Patient-bootstrap intervals do not substitute for repeated training seeds'),
    ('Calibration fitting','Not performed; requires leakage-safe development predictions and a specified fit procedure'),
    ('Historical test inspection','Document ensemble reporting change and prior result inspection in manuscript'),
],columns=['reporting_item','status'])
display(save_table(reporting_gaps,'reporting_gaps','Manuscript evidence requirements not supplied by saved predictions.','audit'))
''')
md('''## 19. CLAIM-oriented reporting map and limitations
This map locates notebook evidence; it does not certify that the clinical study satisfies every CLAIM
item. Cohort enrichment/class imbalance, the fixed Perforation pool, absence of external testing,
historical held-out inspection, differing preprocessing/tuning budgets and unmeasured seed variability
must be reflected in the manuscript. Neither aggregate accuracy nor calibration alone demonstrates
clinical benefit. The overall endpoint choice differs from the earlier plan's co-primary framing and
is recorded as part of version 4's exploratory analysis specification.''')
code('''
claim_map = pd.DataFrame([
    ('Data partitions / sample sizes','Table 1; Figure 1; split_integrity_4.csv'),
    ('Model methods / reproducibility','Table 2; all_run_provenance_4.csv; input hashes'),
    ('Missing or excluded results','run_coverage_4.csv; discovery_audit_4.csv'),
    ('Statistical uncertainty','Sections 3, 8–12; paired comparison plan and Table S4'),
    ('Diagnostic performance','Tables 3–4; Figures 2–5; Table S5'),
    ('Calibration / failures','Figure S8; Tables S6–S7; confusion matrices'),
    ('Sensitivity and reproducibility','Tables S9–S11; analysis manifest and source hashes'),
    ('Unavailable clinical evidence','reporting_gaps_4.csv; clinical records needed'),
], columns=['reporting_domain','evidence'])
save_table(claim_map,'claim_reporting_map','Notebook evidence map aligned to CLAIM reporting domains; not a completed submission checklist.','audit')
display(claim_map)
''')
md('''## 20. Optional local patient-level export
Disabled by default. Aggregate tables and figures do not expose patient IDs. If enabled for local
analysis, this file remains separate from the manuscript main/supplementary tables.''')
code('''
if EXPORT_PATIENT_LEVEL:
    patient_rows = []
    for run in inventory.itertuples():
        p,pred = predictions[run.run_id]
        for index,pid in enumerate(patient_ids):
            patient_rows.append({'run_id':run.run_id,'patient_id':pid,'true_class':ah.CLASSES[y_true[index]],
                'predicted_class':ah.CLASSES[pred[index]], **dict(zip(ah.CLASSES,p[index]))})
    save_table(pd.DataFrame(patient_rows),'local_patient_predictions','Local analytical export; not a manuscript table.','audit')
else:
    print('Patient-level export disabled; manuscript exports are aggregate.')
''')
from review_cells_4 import append_review_cells
append_review_cells(md, code)

md("""## 24. Detailed interpretation for Results and Discussion
The following companion updates all 13 pipelines, all checkpoint routes, the expanded paired
family and the architectural claims. It is generated from this execution's validated results.
""")
code('''
import detailed_notes_4 as dn
detailed=dn.generate(main,ensembles,chosen_results,paired,main_class,review,probe_comparison,full_network_comparison)
for suffix,content in [('txt',detailed),('html',dn.as_html(detailed))]:
    path=EXPORT_ROOT/f'Detailed_Results_and_Interpretation_4.{suffix}'
    path.write_text(content,encoding='utf-8')
    record_artifact(path,'Detailed version-4 results, pipeline interpretation and discussion guidance.')
display(Markdown(detailed))
''')

md('''## 25. Verify and freeze this execution's export manifest
The manifest stores exact inputs, settings, selected run IDs, software versions, companion-code hashes,
and every current output checksum. New results are discovered on rerun; cached computations are never
reused across changed source hashes. Raw results and other notebooks remain untouched.''')
code('''
assert not discovery_audit[discovery_audit.status=='accepted'].run_id.duplicated().any()
assert all(len(predictions[r][0]) == N_TEST for r in inventory.run_id)
assert set(main.source) <= {PRIMARY_SOURCE}
assert not main.duplicated(['model','method','percentage']).any()
assert set(zero_shot.model) <= set(ah.FOUNDATION_MODELS)
assert all(r.method in ah.METHODS[r.model] for r in inventory[~inventory.zero_shot].itertuples())
assert (inventory[(~inventory.zero_shot)&inventory.model.isin(ah.FOUNDATION_MODELS)].source_family == 'publication').all()
assert all((EXPORT_ROOT / artifact['path']).is_file() for artifact in ARTIFACTS)
configuration = {'version':ANALYSIS_VERSION,'bootstrap_replicates':BOOTSTRAP_REPLICATES,
    'permutation_replicates':PERMUTATION_REPLICATES,'seed':SEED,'calibration_bins':CALIBRATION_BINS,
    'main_source':PRIMARY_SOURCE,'main_percentage':MAIN_PERCENTAGE,'run_overrides':RUN_OVERRIDES,
    'class_order':ah.CLASSES,'selection_policy':'foundation publication-only; baselines publication then grid then default; unresolved ties excluded',
    'allowed_supervised_methods':ah.METHODS,'zero_shot_models':list(ah.FOUNDATION_MODELS),
    'comparison_plan':pair_plan.to_dict('records'),'inferential_status':'exploratory, retrospective reporting amendment'}
manifest = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),'configuration':configuration,
    'python':platform.python_version(),'packages':{p:version(p) for p in ['numpy','pandas','matplotlib','scipy','scikit-learn','nbformat']},
    'code_hashes':{p.name:ah.digest(p) for p in [NOTEBOOK_DIR/'analysis_helpers_4.py',NOTEBOOK_DIR/'plot_helpers_4.py',NOTEBOOK_DIR/'build_notebook_4.py',NOTEBOOK_DIR/'execute_notebook_4.py',NOTEBOOK_DIR/'review_helpers_4.py',NOTEBOOK_DIR/'review_cells_4.py',NOTEBOOK_DIR/'detailed_notes_4.py']},
    'notebook_source_hash':ah.fingerprint([(''.join(c.get('source',[])),c['cell_type']) for c in ah.read_json(NOTEBOOK_DIR/'Publication_Results_4.ipynb')['cells']]),
    'split_hashes':split_audit.to_dict('records'),'input_hashes':input_hashes,'review_input_hashes':review_input_hashes,
    'selected_runs':selected.run_id.tolist(),'current_artifacts':ARTIFACTS,
    'counts':{'valid_runs':len(inventory),'excluded_runs':int((discovery_audit.status=='excluded').sum()),
              'scope_excluded_runs':int((discovery_audit.status=='scope_excluded').sum()),
              'selected_main_runs':len(main),'heldout_patients':N_TEST}}
manifest_path = EXPORT_ROOT / 'analysis_manifest_4.json'
manifest_path.write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
''')
md('''### Export index and manuscript-ready observations
The numerical observations below are generated from available results, without selecting or declaring
a test-score winner. Clinical reporting gaps and the provisional case link require author review.''')
code('''
index_lines = ['# Publication results · version 4', '',
    f'Generated: {manifest["generated_at_utc"]}', '',
    'Current output set is defined by analysis_manifest_4.json. Prior unlisted files are not current results.', '',
    '## Analysis status', '',
    f'- {len(inventory)} valid stored runs; {len(main)} full-development ensemble results; {N_TEST} held-out patients.',
    '- Retrospective ensemble-centered reporting; paired inference is exploratory.', '', '## Deliverables', '']
for item in ARTIFACTS:
    index_lines.append(f'- [{item["path"]}]({item["path"]}): {item["caption"]}')
(EXPORT_ROOT/'README_4.md').write_text('\\n'.join(index_lines),encoding='utf-8')
observations = [f'Of {DEVELOPMENT_COUNTS[100]+N_TEST:,} included patients, {DEVELOPMENT_COUNTS[100]:,} were assigned to development and {N_TEST} to fixed held-out testing.']
for run in main.itertuples():
    observations.append(f'{run.label}: macro F1 {run.f1_macro:.3f} (95% CI {run.f1_macro_low:.3f}, {run.f1_macro_high:.3f}); balanced accuracy {run.balanced_accuracy:.3f} (95% CI {run.balanced_accuracy_low:.3f}, {run.balanced_accuracy_high:.3f}).')
observations.append('These are retrospective comparative estimates; they do not establish external generalizability or clinical utility.')
(EXPORT_ROOT/'results_observations_4.md').write_text('\\n\\n'.join(observations),encoding='utf-8')
record_artifact(EXPORT_ROOT/'results_observations_4.md','Numerical manuscript observations without test-score winner selection.')
record_artifact(EXPORT_ROOT/'README_4.md','Captioned index of this execution.')
manifest['current_artifacts'] = ARTIFACTS
manifest_path.write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
display(Markdown(f'**Complete:** {len(inventory)} validated runs, {len(main)} main ensemble rows, '
                 f'{len(ARTIFACTS)} registered artifacts. Outputs: `{EXPORT_ROOT}`.'))
display(pending_main[['model','method','status']])
''')

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.11'},
    'publication_analysis':{'version':4,'output_folder':'output_4','reporting_focus':'cv_ensemble'}})
path=Path(__file__).with_name('Publication_Results_4.ipynb')
nbf.write(notebook,path)
print(f'Created {path} with {len(cells)} cells')
