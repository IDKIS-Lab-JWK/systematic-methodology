# MedicalNet and 3D DenseNet in the systematic methodology

These runners use the existing patient manifests, five-fold selection, refitting,
and held-out reporting contract. MedicalNet is a pretrained 3D ResNet18 trained
end to end; DenseNet is **MONAI DenseNet121 with `spatial_dims=3`, trained from
scratch**. The older DenseNet outside this folder also used 3D convolutions.
The original DenseNet paper describes a 2D architecture; that does not make our
implementation 2D.

Implementation and bounded checks were added on 2026-09-22/23. No full scientific
CV experiment or evaluation of the real held-out test set was run during setup.
The earlier model implementations and saved scientific results remain intact.

## Run commands

Use PowerShell from `Systematic_Methodology`. The isolated `.venv_cnn` has been
created on this machine. To reproduce it elsewhere with Python 3.11:

```powershell
.\CNN_Environment_Setup.ps1
# If the Python launcher is unavailable:
.\CNN_Environment_Setup.ps1 -BootstrapPython C:\Path\To\Python311\python.exe
```

The setup pins PyTorch 2.8.0 CUDA 12.8, MONAI 1.5.1 and the requirements in
`Code/Models/_cnn/requirements.txt`. MedicalNet expects the existing
`../resnet_18_23dataset.pth`; `-Checkpoint` can point to an identical copy.
Its SHA-256 is checked before training. DenseNet requires no pretrained file.

Inspect the planned commands without training or result creation:

```powershell
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -DryRun
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -DryRun
```

Train all five folds, refit, and evaluate the refit on the fixed test set:

```powershell
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -ParameterSource publication -Stage all
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -ParameterSource publication -Stage all
```

Run the learning curves sequentially (run one model command at a time):

```powershell
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20,40,60,80,100 -Stage all
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20,40,60,80,100 -Stage all
```

The default percentage is **100**, so specify `-Percentages 20` for the first
small-pool experiment. `-Stage cv` includes refitting but does not evaluate the
test set. Repeating the same command resumes compatible partial checkpoints or
verifies/reuses completed results. Changes to parameters, microbatch, precision,
code, runtime, checkpoint, or patient identities cannot silently reuse a run.

Optional secondary reports after CV, without new optimization:

```powershell
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -Stage test -ModelSource cv_ensemble
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -Stage test -ModelSource best_cv_fold
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -Stage test -ModelSource cv_ensemble
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -Stage test -ModelSource best_cv_fold
```

Optional development-only searches:

```powershell
# Three learning rates, five folds each; selected fold models are reused by CV.
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -ParameterSource grid -Stage grid
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -ParameterSource grid -Stage cv
# Alternatively, -Stage all runs grid, CV/refit and test in one invocation.

# Predeclared 192^3 versus 256^3 sensitivity, separate result root to avoid
# colliding with the learning-rate grid for the same percentage.
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -ParameterSource grid `
  -GridConfig .\Code\Models\DenseNet\resolution_grid.json `
  -ResultsRoot .\Results\DenseNet_Resolution -Stage grid
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -ParameterSource grid `
  -GridConfig .\Code\Models\DenseNet\resolution_grid.json `
  -ResultsRoot .\Results\DenseNet_Resolution -Stage cv
```

For a fresh learning-rate search without test evaluation, use `-Stage grid`,
then `-Stage cv`. The `cv` stage expects a completed grid when the parameter
source is `grid`. Decide the analysis/model source before viewing test results;
do not choose preprocessing by comparing test scores. The CV score used to
choose a candidate is a development score, not an unbiased nested-CV estimate.

## Dataset distribution and leakage control

Inputs come from `Dataset/NIFTI/<class>/<patient>.nii.gz` and the existing
`Dataset/Splits/{20,40,60,80,100}_percent.json`. No new split is made and none of
the old percentage-folder Train/Validation/Test directories are used.

| Development pool | Appendicitis | Cholezystitis | Normal | Perforation | Sigmoid | Total |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 38 | 30 | 72 | 40 | 68 | 248 |
| 40% | 76 | 59 | 144 | 40 | 137 | 456 |
| 60% | 114 | 89 | 216 | 40 | 205 | 664 |
| 80% | 152 | 118 | 288 | 40 | 274 | 872 |
| 100% | 190 | 148 | 360 | 40 | 342 | 1,080 |
| Fixed held-out test | 47 | 37 | 90 | 18 | 86 | 278 |

The development pools are nested and use the existing seed-42 patient folds.
Perforation stays at 40 development patients: 32 train and 8 validate in each
fold. Training labels alone determine effective-number class weights. There
is no oversampling or change to validation/test prevalence. Refitting computes
weights and geometry again from the selected entire development pool.

Fitting preprocessing exclusively on training observations follows leakage
prevention guidance, rather than a spacing prescription from an architecture
paper. See [Kapoor and Narayanan, Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804)
and [scikit-learn's preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Preprocessing: practical rationale and evidence

The same policy is applied to both CNNs. Its numeric plan is fitted separately
for every training fold; validation and test never alter that plan. The final
refit artifact carries its own development-only plan into inference.

| Step | Executed policy and practical reason | Evidence and scope |
|---|---|---|
| Input validation | Require a finite 3D NIfTI, invertible affine and explicit millimetre units. Read scaled intensities from the existing CT conversion. | Engineering checks prevent treating array indices as physical distances. HU correctness relies on the existing CT conversion; it cannot be inferred from an arbitrary NIfTI header. |
| Orientation | Resample into axis-aligned RAS using the complete affine, including obliquity; model layout is `[N,1,S,A,R]`. | Consistent anatomical axes. This is an implementation choice, not a new learned transform. No left/right random flips. |
| Reference spacing | Take the per-axis median native spacing of the fold's training patients. | Med3D §3.1 uses domain median spacing. Restricting the fit to training patients prevents leakage. |
| Working spacing | Coarsen the reference only as much as needed to contain every training scan in the chosen canvas. Fit from world-coordinate acquisition boundaries, not array sizes alone. | Full-volume classification lacks lesion boxes. Joint geometry/memory planning is motivated by nnU-Net; our maximum-extent formula is a classification adaptation, not nnU-Net's published planner. |
| Anti-aliasing | Gaussian prefilter only along downsampled native axes, followed by one physical-space linear interpolation. The Gaussian sigma is `max((scale-1)/2,0)`; oblique native directions use projected target resolution. | Reduce aliasing without repeated resizing. Sigma follows the scikit-image resize convention; its oblique projection and nearest-edge filtering are our implementation choices. |
| Coverage and padding | Center the acquired world bounding box. Fill outside acquisition with −1000 HU. No organ crop, random crop, lesion mask, or second resize. | Preserve location and acquired anatomy without requiring localization annotations. Air padding maps to zero after intensity scaling. |
| Larger validation/test scans | Keep **exactly the fitted working spacing**. Expand the canvas to multiples of 32 as needed, with batch-1 inference and adaptive global pooling. Log actual dimensions. | Resolves full coverage without fitting on validation/test or silently changing resolution. A memory error stops execution; it never triggers hidden coarsening or cropping. |
| Intensity | Clip to [−200,250] HU, then linearly map to [0,1]; no cohort intensity fitting or per-volume z-score. | Abdominal precedent from Med3D's LiTS experiment (§4.3). The [0,1] mapping and transfer to five-class classification/DenseNet are adaptations. Absolute attenuation remains comparable across patients. |
| Augmentation | None in the baseline. | A controlled starting point preserving laterality and full field of view without lesion boxes. This is not either paper's augmentation recipe and is not a claim that augmentation is ineffective. |

For each axis `a`, training set `T`, and canvas length `C[a]`:

```text
reference[a] = median(native_spacing[i,a] for i in T)
working[a]   = max(reference[a], max(world_span[i,a] for i in T)/(C[a]-1))
```

An unseen scan uses that same `working` value. Its required voxel count is
`ceil(world_span / working) + 1`, rounded up to a multiple of 32 and bounded
below by the training canvas. This calculation positions the individual scan;
it estimates no population statistic and uses no labels.

The private development-only geometry audit (not distributed)
records 25 training-fold plans plus 5 refit plans. At 100%, native median spacing
is approximately 0.758 × 0.758 × 3.000 mm; the refit working spacing is
2.030 × 1.961 × 3.000 mm. All training scans fit the 256³ canvas. One validation
scan in fold 1 of each percentage expands to 288 × 256 × 256; the other audited
validation tensors remain 256³. No real test headers were used for this audit.

**256³ is a feasible resource choice, not a clinically validated optimum.**
Geometric coverage does not establish that small free-air foci, thin walls or
small inflammatory changes remain detectable after filtering/downsampling.
The HU window also suppresses differences outside its limits. Resolution and
window sensitivity should be evaluated on development data before final test
analysis. The supplied resolution grid changes only the canvas; it does not
reuse test information. The whole-volume maximum can be conservative when a
training scan is unusually extensive; it is deliberately preferred to dropping
acquired anatomy. These limitations belong in any eventual publication.

Sources: [Med3D](https://arxiv.org/pdf/1904.00625),
[nnU-Net, Nature Methods 2021](https://doi.org/10.1038/s41592-020-01008-z),
[nnU-Net author manuscript](https://arxiv.org/pdf/1904.08128), and
[scikit-image resize documentation](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize).
nnU-Net is a segmentation precedent; neither it nor the two backbone papers
establishes the correct tensor dimensions for this classification task.

## Parameters: original publications versus this task

`publication` means **publication-backed anchors with documented adaptations**,
not reproduction of an original paper's complete experiment. Each JSON contains
inert `# provenance` and `# references` fields, which are excluded from executable
candidate parameters but retained in profile hashes and documentation.

| Setting | MedicalNet `publication` | DenseNet `publication` | Basis |
|---|---|---|---|
| Model | Released 3D ResNet18, all layers trainable, new GAP + 5-class head | MONAI 3D DenseNet121, random initialization, 1 input channel, 5 classes | MedicalNet official architecture/weights; DenseNet 3D extension is MONAI-backed. |
| Optimizer | Adam, LR 0.001, decay 0 | SGD, LR 0.01, momentum 0.9, Nesterov, decay 0.0001 | Med3D transfer uses Adam 0.001, including its classification comparison; transfer decay is unspecified, so zero is an explicit default. DenseNet paper supplies SGD/momentum/decay; LR 0.01 is adapted from ImageNet's 0.1 for this small-batch task, not a proven scaling law. |
| Schedule/budget | Constant LR, at most 200 epoch equivalents, patience 10 | 90 epoch equivalents; LR ×0.1 at 30 and 60, no early termination | MedicalNet 200 comes from upstream demo `setting.py`, not the paper; patience is local. DenseNet milestones/budget follow the paper's ImageNet schedule. Select the best validation checkpoint in either case. |
| Batch/input | Effective and default microbatch 4, 256³, evaluation batch 1 | Same | GPU/resource choices, not original paper settings. |
| Dropout | 0 | 0 | Original DenseNet dropout 0.2 was for nonaugmented CIFAR/SVHN experiments, not a universal medical setting. |
| Loss | Effective-number weighted cross entropy, beta 0.999, computed on training labels | Same | [Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555) supplies the weighting form; beta is the existing project's choice. Gradient clipping at norm 1 is a local stability choice. |

Original DenseNet ImageNet training used **2D images**, batch 256, LR 0.1 and
90 epochs. MedicalNet pretraining used a different SGD recipe; it is not the
transfer optimizer. Its paper's general intensity preprocessing uses percentile
clipping followed by z-scoring, whereas the abdominal LiTS experiment supplies
the HU window used here. References: [DenseNet §4.2](https://arxiv.org/pdf/1608.06993),
[Med3D](https://arxiv.org/pdf/1904.00625),
[pinned MedicalNet demo settings](https://github.com/Tencent/MedicalNet/blob/20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd/setting.py),
[MONAI 1.5.1 DenseNet source](https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/densenet.py).

`default` is a separate practical profile: AdamW 0.0001, decay 0.0001, constant
LR, cap 200, patience 10; DenseNet dropout 0.2. These preserve older optimizer
anchors, not exact legacy experiments. MedicalNet's old head-only warmup is
omitted. `grid` tests MedicalNet LRs 0.0001/0.0003/0.001 or DenseNet
0.003/0.01/0.03 while retaining each publication profile's other settings.

The existing selection score is `0.7*macro_F1 + 0.3*balanced_accuracy` per fold.
Grid selection uses `0.7*(mean_F1-SD_F1) + 0.3*(mean_BA-SD_BA)`; ties use
Perforation recall, then macro-F1, then candidate ID. The patient stream preserves
the inherited full effective batches and deterministic reshuffling. An epoch
equivalent is `ceil(training_count/effective_batch)` optimizer steps.

Refit uses the **median selected optimizer step** across five folds, matching
the existing systematic workflow. Since refit has more patients, this gives
fewer epoch equivalents than the corresponding CV epoch count. Its epoch-based
milestones use refit steps per epoch and may not all be reached if its selected
budget is short. The refit spacing is fitted on the selected development pool.

BF16 is chosen by `auto` on the 4090; otherwise execution resolves to FP32.
Changing `-MicroBatchSize` to 1 or 2 can reduce training memory, but changes
BatchNorm statistics despite equal effective batch size. It is a new run
identity, not an equivalent resume. FP32 memory has not been benchmarked at
the full batch-4 resolution. Run one model on this GPU at a time.

## Outputs, logs, storage and resume

```text
Results/MedicalNet/FullFineTune/       # DenseNet/FromScratch has the same layout
  Grid/<percentage>_percent/          # only explicitly requested grids
  CV/cv_runs.json
  CV/CV_<n>/<percentage>_percent/
    fold_01/ ... fold_05/
    refit_full_development/
    cv_fold_summary.csv, refit_selection.json, best_fold_model.json
  Test/CV_<n>/<model_source>/<percentage>_percent/
    metrics_summary.json, model_selection.json
    test_patient_predictions.json, test_probabilities.csv
Results/<model>/Logs/
  invocations.jsonl, inference_timing.jsonl
Results/<model>/Cache/Volumes/
```

Each trained fold/refit saves `model.pt`, exact-resume `last_checkpoint.pt`,
`parameters.json`, `training_context.json`, `training_state.json`,
`preprocessing_plan.json`, `preprocessing_observed.json`, `training_history.csv`
and a hashed `completion.json`. Fold validation also saves metrics and patient
probabilities. The plan includes formulas, numeric spacing, training-patient and
geometry fingerprints, rationale and sources. The observation audit includes
actual grids, padding expansion and anti-alias sigmas, including observations
restored from checkpoints. Inference logs record each artifact's applied plan
and patient grids. Refit has no validation history, so its history CSV is empty
apart from headers; its completed-step budget and timing are in training state.

Model states include BatchNorm buffers; checkpoints include optimizer,
scheduler, sampler and RNG state. CPU interruption/resume was tested for exact
tensor equality. GPU runtime and math flags are recorded; bitwise equivalence
across different hardware/software is not promised. Source paths, sizes and
modification times bind run reuse; source content and cache artifacts are
SHA-256 checked on first use per process. Do not alter NIfTIs in place during
experiments. Completion seals detect changed output files.

The float32 256³ input cache costs about 64 MiB per patient: 15.5 GiB at 20%,
67.5 GiB at 100%, plus minor expansion/metadata. Preflight asks for 20% headroom
and 2 GiB extra (about 83 GiB free at 100%). Model/checkpoint storage is additional.
`-CacheRoot D:\CNNCache` can use another disk. Default `-CachePolicy temporary`
removes only marked, owned preprocessing namespaces after each sealed fold/refit
or completed inference. Interrupted work retains its cache. `retain` keeps
all namespaces and can consume hundreds of GiB across folds/configurations.
Exclusive result/cache leases prevent conflicting writers. No permanent
foundation-model feature cache is touched.

Active training time accumulates across checkpoints and excludes stopped gaps;
work lost since the last checkpoint is not counted. Invocation logs include
failed/restarted invocations and end-to-end wall time. Inference has separate
timings. The historical timing and verification record is private and is not distributed;
these estimates retain the limits described above.

## Reproduce bounded verification

```powershell
.\.venv_cnn\Scripts\python.exe Code\Models\_cnn\audit.py --action geometry
.\.venv_cnn\Scripts\python.exe Code\Models\_cnn\audit.py --action benchmark --model MedicalNet --real_patient
.\.venv_cnn\Scripts\python.exe Code\Models\_cnn\audit.py --action benchmark --model DenseNet --real_patient
.\.venv_cnn\Scripts\python.exe -m pytest Code\Models\_cnn\tests -q
```

The benchmark uses five development scans for preprocessing timing and only
three optimizer steps on a repeated development tensor. It is a memory/setup
check, not an accuracy experiment or an end-to-end runtime forecast. Its JSON
reports are stored under `Logs/CNN_Baselines/`.
