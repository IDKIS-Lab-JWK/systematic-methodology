> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# Merlin Five-Fold CV Classification

Four Merlin supervised adaptation variants share one patient-level data, metric, grid,
five-fold CV, model-selection, and held-out-test pipeline:

- `train_linear_probe.py`: the Merlin image encoder is frozen in evaluation
  mode and a strict Kaiming-initialized `Linear(2048, 5)` head is trained.
- `train_full_finetune.py`: all Merlin image-encoder parameters are trainable.
- `train_conv_lora.py`: the backbone is frozen and ConvLoRA is injected into
  3D convolutional layers in `layer1` through `layer4`.
- `train_conv_lora_llrd.py`: an independent, publication-inspired image-only
  supervised variant that retains ConvLoRA while adding explicit layer-wise
  learning-rate decay (LLRD), selective weight decay, and update-wise cosine
  scheduling.

`run_zero_shot.py` is a separate frozen, inference-only strategy. It uses the
joint 512-dimensional image-text projections from `Merlin()` and never enters
the supervised grid/CV/refit pipeline.

Run commands from the `Systematic_Methodology` directory so the installed
`merlin-vlm` package cannot be shadowed by another local folder.

## Installation

```powershell
python -m pip install -r Code\Models\Merlin\requirements.txt
```

## Zero-shot classification

The versioned prompt profile is `zero_shot_prompts.json`. It records exact
task-specific prompts and their provenance. These are not claimed to be the
publication's exact prompts because the authors did not release a prompt bank
for this dataset's five classes.

Two complementary modes are available:

- `five_class_adapted`: mean the cosine similarities to each class's positive
  prompts and select the largest of the five scores. This produces one label
  per CT and is the primary result for comparison with the supervised models.
- `author_binary`: independently compare the mean finding-present and
  finding-absent cosine similarities for Appendicitis, Cholecystitis,
  Perforation, and sigmoid diverticulitis. Its scoring rule is closer to the
  publication's method, but it can return zero or multiple findings and does
  not force a Normal label. This dataset lacks independently reviewed
  present/absent annotations: its reported binary metrics use one-vs-rest proxy
  truth and must not be described as a reproduction of the publication's
  finding-presence evaluation. Another primary label does not necessarily
  prove that a clinically coexisting finding is absent.

Both modes use raw cosine scores. Any exported softmax or sigmoid values are
explicitly uncalibrated derived confidences. Prompt selection, temperature
selection, training, CV, and refitting are not performed. The fixed test set is
read once from the canonical 100% manifest, while the run is recorded as 0%
training data. The code verifies that held-out membership is identical in all
available percentage manifests before inference.

Run these commands from `Systematic_Methodology`:

```powershell
# Validate manifests, NIfTI paths, prompts, and output paths; no model load.
python Code\Models\Merlin\run_zero_shot.py --dry_run

# Download/load Merlin, score one CT in reference FP32, and write no metrics.
python Code\Models\Merlin\run_zero_shot.py `
  --validate_setup `
  --skip_checkpoint_hash

# Recommended run: encode the test cohort once and export both modes.
python Code\Models\Merlin\run_zero_shot.py `
  --mode both `
  --batch_size 1 `
  --num_workers 0 `
  --amp

# Run only the mutually exclusive five-class adaptation.
python Code\Models\Merlin\run_zero_shot.py --mode five_class_adapted --amp

# Run only the publication-scoring-inspired one-vs-rest proxy analysis.
python Code\Models\Merlin\run_zero_shot.py --mode author_binary --amp
```

The equivalent launcher defaults to both modes and CUDA AMP:

```powershell
.\Merlin_Zero_Shot_Run_Script.ps1
```

Use `--cache_root D:\MerlinCache` (or `-CacheRoot D:\MerlinCache` with the
launcher) to put persistent preprocessing on another drive. Use `--force` or
`-Force` only when intentionally replacing artifacts for the same prompt-set
ID. `--skip_checkpoint_hash` avoids reading the complete checkpoint for its
SHA-256. The code enforces the pinned `merlin-vlm==0.0.7` API. The first real
run can require network access for torchvision's ResNet-152 initialization,
Clinical-Longformer, and Merlin's joint checkpoint. Merlin writes its
checkpoint below the active environment's installed package, so use a writable
virtual/Conda environment.

Zero-shot outputs are isolated from trained-model results:

```text
Results/Merlin/ZeroShot/Test/
|-- five_class_adapted/<prompt_set_id>/
|   |-- metrics_summary.json
|   |-- model_selection.json
|   |-- prompt_provenance.json
|   |-- test_patient_predictions.json
|   |-- test_probabilities.csv
|   `-- test_similarity_scores.csv
`-- author_binary/<prompt_set_id>/
    |-- metrics_summary.json
    |-- model_selection.json
    |-- prompt_provenance.json
    |-- test_binary_patient_predictions.json
    `-- test_binary_scores.csv
```

The prompt profile currently preserves the dataset label `Cholezystitis` while
using clinically spelled cholecystitis text, interprets `Sigmoid` as acute
sigmoid diverticulitis, and interprets `Perforation` as gastrointestinal or
hollow-viscus perforation. These last two meanings are implementation
assumptions rather than independently verified dataset provenance; treat the
results as exploratory until they are confirmed. Change those semantics only
by creating a new prompt set ID so results remain distinguishable.

## Linear probing

Linear probing is an independent supervised strategy under
`Results/Merlin/LinearProbe/`. It freezes every
`Merlin(ImageEmbedding=True)` encoder parameter, keeps the encoder in
evaluation mode, materializes each raw 2048-dimensional CT feature once, and
trains only one five-logit linear layer. The strict head intentionally contains
no LayerNorm, dropout, hidden layer, or ConvLoRA parameter.

The publication-adapted profile follows the final Merlin Nature paper where
reported: Kaiming initialization, AdamW at `1e-5`, 10 training epochs, and a
cosine scheduler configured with a 300-epoch horizon to zero. The paper's
finding task used masked binary cross-entropy for 30 incompletely labelled
findings. This project instead uses class-weighted multiclass cross-entropy and
argmax because every CT has one of five mutually exclusive labels. Unreported
batch size, weight decay, warmup, and checkpoint selection are explicitly
identified as implementation choices in `publication_linear_probe.json`.

Recommended publication-adapted run at 100%:

```powershell
python Code\Models\Merlin\train_linear_probe.py `
  --stage all `
  --percentage 100 `
  --parameter_source publication `
  --model_source refit_full_development `
  --feature_extraction_batch_size 1 `
  --amp `
  --resume
```

Run the publication profile over all five training percentages:

```powershell
.\Merlin_Linear_Probe_Run_Script.ps1
```

Use the project-grounded defaults without a grid:

```powershell
python Code\Models\Merlin\train_linear_probe.py `
  --stage all `
  --percentage 100 `
  --parameter_source default `
  --model_source refit_full_development `
  --amp `
  --resume
```

Run the nine-candidate linear-probe grid at 40%:

```powershell
python Code\Models\Merlin\train_linear_probe.py `
  --stage grid `
  --percentage 40 `
  --parameter_source grid `
  --percentage_specific_grid true `
  --amp `
  --resume
```

The feature-extraction batch size is independent of the small probe head's
`batch_size`. Frozen features are stored under
`Results/Merlin/Cache/Features/` and are tied to the preprocessing manifest,
Merlin checkpoint SHA-256, and package version. Feature extraction is always
FP32 so runs cannot accidentally mix FP16- and FP32-derived representations;
`--amp` applies only to the lightweight head training.

## Shared persistent preprocessing cache

All grid, CV, full-development refit, and held-out-test loaders now use one
deterministic persistent cache shared across dataset percentages, folds,
train/validation roles, evaluation strategies, Linear Probe, Full Fine-Tuning,
ConvLoRA, and ConvLoRA + LLRD:

```text
Results/Merlin/Cache/Shared/
|-- merlin_ct_ras_1p5x1p5x3_224x224x160_v1/
|   |-- cache_manifest.json
|   `-- <one hashed tensor per encountered NIfTI path>.pt
`-- _tmp/
```

The cached Merlin transform contains no random augmentation. It performs RAS
orientation, 1.5 x 1.5 x 3 mm spacing, HU scaling from `[-1000, 1000]` to
`[0, 1]`, padding, and a 224 x 224 x 160 center crop. Since Merlin keys each
entry by source image path, a patient is preprocessed once and reused by every
split that references that path. Split JSON files still control which patients
are loaded; sharing deterministic image tensors does not merge labels or
patient membership between splits.

Use another drive for both persistent and temporary cache writes:

```powershell
python Code\Models\Merlin\train_conv_lora.py `
  --stage all `
  --percentage 100 `
  --cache_root D:\MerlinCache `
  --num_workers 0 `
  --amp `
  --resume
```

This produces
`D:\MerlinCache\Shared\merlin_ct_ras_1p5x1p5x3_224x224x160_v1`.
`--cache_temp_dir` can override the temporary-write directory separately.

Use a new `--cache_namespace` whenever deterministic preprocessing or source
image contents change. Existing legacy percentage/fold caches are not read or
deleted automatically; after stopping all Merlin processes and confirming the
shared cache works, they can be removed separately to reclaim space.

## Grid selection behavior

`--parameter_source auto` is the default and preserves the existing `--grid`
interface:

| `--grid` | `--percentage_specific_grid` | Behavior |
|---|---|---|
| `false` | either | Use project defaults, plus any explicitly supplied parameter values |
| `true` | `true` | Reuse or run the requested percentage's grid |
| `true` | `false` | Prefer the shared 40% grid, then any completed grid; run 40% if none exists |

An explicit `--parameter_source` can select `grid`, `default`, `publication`,
`publication_derived`, or `manual` independently of the legacy `--grid`
default:

| `--parameter_source` | Behavior |
|---|---|
| `auto` | Preserve the existing `--grid` behavior |
| `grid` | Select parameters from the shared or percentage-specific grid |
| `default` | Use the original project defaults in `cli.py` |
| `publication` | Load the variant's immutable audited publication profile: final Merlin Nature linear probing or HCC-Merlin adaptation profiles as applicable |
| `publication_derived` | Load the immutable publication profile and apply explicitly supplied parameter overrides |
| `manual` | Use project defaults plus explicitly supplied parameter options |

`default`, `publication`, and `grid` reject explicit parameter overrides so
their provenance stays unambiguous. `manual` and `publication_derived` require
at least one explicit parameter option. A publication-derived option must
actually differ from its publication value.

An incomplete grid resumes at the missing fold runs when `--resume` is
used. Completed candidate and fold results are reused unless `--force_grid` or
`--force_cv` is supplied.

`grid_summary.json`, `grid_summary_by_selection_score.csv`, and
`best_hyperparameters.json` are rewritten after every fully evaluated
hyperparameter candidate. Therefore, results from completed candidates remain
ranked and usable for inspection even when a later candidate is interrupted.
An incomplete summary is marked with `"grid_complete": false` and is not
treated as a completed grid by parameter resolution.

Each candidate is evaluated on five train/validation pairs: one fold is
validation and the other four folds are training. The grid score is:

```text
0.7 * (mean F1 - std F1) + 0.3 * (mean balanced accuracy - std balanced accuracy)
```

The best candidate is selected from its five-fold aggregate. Its five existing
fold runs are reused for CV reporting and best-fold selection, so they are not
trained a second time. The best fold checkpoint is identified with:

```text
0.7 * F1 + 0.3 * balanced accuracy
```

The held-out test patients are never loaded during grid search or CV.

## Numbered CV runs

Each exact hyperparameter dictionary is assigned a stable run folder beneath
the variant's `CV/` directory:

```text
CV/
|-- cv_runs.json
|-- CV_1/
|   |-- cv_parameters.json
|   |-- 20_percent/
|   |-- 40_percent/
|   `-- ...
`-- CV_2/
    |-- cv_parameters.json
    `-- ...
```

Before CV starts, the script compares the resolved parameters with every
`CV_x/cv_parameters.json`:

- An exact match resumes missing folds or reuses a completed percentage run.
- `--force_cv` reruns the matching `CV_x`; it does not allocate a new number.
- A previously unseen parameter dictionary creates the next `CV_x`.
- The same parameters map to the same `CV_x` across dataset percentages.

`cv_parameters.json` records the parameters, their fingerprint, whether their
origin was `grid_search`, `default`, `publication_backed`, or
`publication_adapted`, `publication_derived`, or `provided_manually`,
provenance history, and
per-percentage completion.
`cv_runs.json` is the general registry of all numbered CV folders.

When a matching completed result still exists in the legacy
`CV/<percentage>_percent/` layout, it is imported into the matching
`CV_x/<percentage>_percent/` using its existing metrics and checkpoints. This
upgrade path does not retrain the model.

Each CV fold contains its own `best_model.pt`. When a selected grid fold is
reused on the same volume, the CV checkpoint is an NTFS hard link to avoid
duplicating a multi-gigabyte file; copying is the automatic fallback.

## Variant defaults

With `--grid false` and no explicit parameter arguments, the following
variant-specific defaults are used. Linear Probe uses the legacy project's
frozen-classifier optimization defaults with a corrected strict linear head.
The Full Fine-Tuning and ConvLoRA columns preserve their original project
defaults; the independent ConvLoRA + LLRD column uses the
publication-inspired image-only supervised recipe.

| Parameter | Linear Probe | Full Fine-Tuning | ConvLoRA | ConvLoRA + LLRD |
|---|---:|---:|---:|---:|
| Encoder/adapter learning rate | Frozen | `1e-5` | `1e-4` | `1e-4` at `layer4`, decayed toward `layer1` |
| Classifier-head learning rate | `1e-4` | `1e-4` | `1e-4` | `1e-4` |
| Image layer-decay factor | Not applicable | Not applicable | Not applicable | `0.7` |
| Batch size | `1` | `1` | `1` | `2` |
| Gradient accumulation | `1` | `4` | `4` | `8` |
| Epochs | `100` | `50` | `100` | `300` |
| Early-stopping patience | `10` | `8` | `10` | `5` |
| Warmup epochs | `0` | `2` | `2` | `0` |
| Minimum LR factor | `1.0` (constant LR) | `0.1` | `0.1` | `0.0` |
| Weight decay | `1e-4` on head | `1e-4` | `1e-4` | `0.01`, excluding biases and normalization parameters |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Scheduler step | Per epoch | Per epoch | Per epoch | Per optimizer update |
| Mixed precision | Frozen features stay FP32; head AMP only with `--amp` | Disabled unless `--amp` is supplied | Disabled unless `--amp` is supplied | Disabled unless `--amp` is supplied |
| Gradient checkpointing | Disabled | Enabled | Enabled | Enabled |
| ConvLoRA rank / alpha | Not applicable | Not applicable | `2 / 2` | `2 / 2` |
| Train image projection | Frozen | Not applicable | Enabled when requested | Disabled |

Explicit parameter arguments still classify the run as `provided_manually`;
using these defaults without parameter overrides classifies it as `default`.

## Publication-backed and publication-adapted profiles

Audited publication-sourced parameter profiles are separate from the original
project defaults:

- `publication_full_finetune.json`
- `publication_conv_lora.json`
- `publication_conv_lora_llrd.json`
- `publication_linear_probe.json`

The three fine-tuning profiles were audited against the accepted MIDL 2026 paper, especially
pages 6, 17, and 18, and the authors' official repository at commit
`0e5fc47829b74033a6d72a8778c0c16c8644d6ad`. The accepted paper takes
precedence where the repository launcher conflicts with the reported
experiment. In particular, the profiles use the paper's `1e-5` full
fine-tuning learning rate and `1e-4` PEFT learning rate.

The separate linear-probe profile was audited against the final 2026 Merlin
Nature paper and official StanfordMIMI inference/model code. It uses the raw
2048-dimensional image representation, a frozen encoder, a single
Kaiming-initialized linear layer, AdamW at `1e-5`, 10 epochs, and the reported
300-epoch cosine horizon. Every unreported value and five-class task adaptation
is labelled separately in the JSON profile.

Each profile contains a complete parameter dictionary and records parameters
as paper-exact, official-code-backed, or adaptations required by this
supervised image classifier. The ConvLoRA + LLRD profile incorporates the
authors' image-side rank/alpha and layer-decay settings while deliberately
using a five-class supervised head and scheduler steps aligned with actual
optimizer updates.

None of these image-only classifier variants is an exact reproduction of the
publication's complete method. The publication trains a joint image-text
contrastive VLM, learns temperature, uses text LoRA in its full PEFT method,
and trains a 512-dimensional contrastive image projection. In this project,
`Merlin(ImageEmbedding=True)` returns the pooled 2048-dimensional feature
before that projection. The ConvLoRA + LLRD variant therefore trains only the
image adapters and five-class classifier; it does not add dormant text or
contrastive components merely to resemble the source implementation.

Publication-adapted Merlin linear probing:

```powershell
python Code\Models\Merlin\train_linear_probe.py `
  --stage all `
  --percentage 100 `
  --parameter_source publication `
  --model_source refit_full_development `
  --feature_extraction_batch_size 1 `
  --amp `
  --resume
```

Publication-backed full fine-tuning:

```powershell
python Code\Models\Merlin\train_full_finetune.py `
  --stage all `
  --percentage 40 `
  --parameter_source publication `
  --model_source refit_full_development `
  --amp `
  --resume
```

Publication-backed ConvLoRA:

```powershell
python Code\Models\Merlin\train_conv_lora.py `
  --stage all `
  --percentage 40 `
  --parameter_source publication `
  --model_source refit_full_development `
  --amp `
  --resume
```

Publication-inspired image-only ConvLoRA + LLRD:

```powershell
python Code\Models\Merlin\train_conv_lora_llrd.py `
  --stage all `
  --percentage 40 `
  --parameter_source publication `
  --model_source refit_full_development `
  --amp `
  --resume
```

The Full Fine-Tuning and existing ConvLoRA profiles are recorded with
`parameter_origin: publication_backed`. The deliberately adapted image-only
ConvLoRA + LLRD and five-class Linear Probe profiles are recorded with
`parameter_origin: publication_adapted`. All four retain the profile, paper,
source references, evidence categories, and known methodological differences
in `cv_parameters.json` and test provenance.

### Publication-derived overrides

Use `publication_derived` when retaining the audited publication profile but
changing one or more reported/adapted parameters. The original profile JSON is
never edited. Only parameter options explicitly present in the command replace
publication values:

```powershell
python Code\Models\Merlin\train_conv_lora.py `
  --stage all `
  --percentage 40 `
  --parameter_source publication_derived `
  --batch_size 2 `
  --grad_accum_steps 16 `
  --publication_override_reason "GPU memory constraint" `
  --model_source best_cv_fold `
  --amp `
  --resume
```

The effective parameters determine `CV_x` identity. Every derived run records
`parameter_origin: publication_derived` and a prominent
`publication_parameter_overrides` object containing:

- `highlight: "DEVIATION FROM PUBLICATION"`;
- publication and effective values for every changed parameter;
- a readable `old -> new` representation and the supplied reason;
- publication and derived effective batch sizes;
- whether the effective batch size changed.

This block is written into the CV registry and parameter file, fold
training-state and metrics JSON, newly written checkpoints, the CV best-fold
summary, held-out `model_selection.json`, held-out metrics, and dry-run output.
CV summary CSV files also show the origin and override count.

If the effective parameters exactly match a completed earlier `CV_x`, that CV
is reused. Its original training provenance is retained, while the current
publication-derived request and its highlighted differences are recorded in
the best-fold JSON under `latest_parameter_provenance` and
`latest_publication_parameter_overrides`.

## Full fine-tuning

Use/reuse the shared 40% grid, run 40% CV, refit on the complete development
pool, and evaluate the held-out test:

```powershell
python Code\Models\Merlin\train_full_finetune.py `
  --stage all `
  --percentage 40 `
  --grid true `
  --percentage_specific_grid false `
  --model_source refit_full_development `
  --amp `
  --resume
```

Run or reuse a percentage-specific grid:

```powershell
python Code\Models\Merlin\train_full_finetune.py `
  --stage all `
  --percentage 60 `
  --grid true `
  --percentage_specific_grid true `
  --model_source refit_full_development `
  --amp `
  --resume
```

Use explicitly supplied parameters without a grid:

```powershell
python Code\Models\Merlin\train_full_finetune.py `
  --stage all `
  --percentage 40 `
  --grid false `
  --encoder_lr 1e-5 `
  --head_lr 1e-5 `
  --dropout_prob 0.1 `
  --epochs 50 `
  --batch_size 1 `
  --grad_accum_steps 4 `
  --amp `
  --resume
```

## ConvLoRA

Run ConvLoRA using its percentage-specific five-fold grid:

```powershell
python Code\Models\Merlin\train_conv_lora.py `
  --stage all `
  --percentage 40 `
  --grid true `
  --percentage_specific_grid true `
  --model_source refit_full_development `
  --amp `
  --resume
```

Run ConvLoRA with the original project defaults and no grid:

```powershell
python Code\Models\Merlin\train_conv_lora.py `
  --stage all `
  --percentage 40 `
  --grid false `
  --resume
```

## ConvLoRA + LLRD

ConvLoRA + LLRD is a separate training strategy rather than a modification of
the existing ConvLoRA pipeline. It freezes the Merlin backbone, injects
rank-2/alpha-2 ConvLoRA adapters into `layer1` through `layer4`, and trains
those adapters with the five-class classifier head. With an adapter base rate
of `1e-4` and decay factor `0.7`, the image-stage learning rates are:

| Trainable component | Learning rate |
|---|---:|
| Classifier head | `1.00e-4` |
| `layer4` ConvLoRA | `1.00e-4` |
| `layer3` ConvLoRA | `7.00e-5` |
| `layer2` ConvLoRA | `4.90e-5` |
| `layer1` ConvLoRA | `3.43e-5` |

AdamW weight decay is applied to eligible weights but not biases or
normalization parameters. The cosine schedule is planned to reach zero at the
final optimizer update and advances after each real optimizer update, including
when gradient accumulation is active. The loss,
validation, early-stopping, CV, refit, and held-out-test procedures remain the
same supervised five-class procedures used by the other variants.

This is intentionally a valid image-only adaptation inspired by the authors'
image-side PEFT implementation. It is not their complete multimodal method:
there is no text LoRA, image-report contrastive loss, learned temperature, or
prompt-similarity classifier.

Run the publication-derived profile across all data percentages with the
repository launcher:

```powershell
.\Conv_LoRA_LLRD_Run_Script.ps1
```

Run the independent shared-grid series instead:

```powershell
.\Conv_LoRA_LLRD_Run_Script_CV1.ps1
```

For a single percentage, invoke the entry point directly:

```powershell
python Code\Models\Merlin\train_conv_lora_llrd.py `
  --stage all `
  --percentage 40 `
  --parameter_source publication_derived `
  --batch_size 2 `
  --publication_override_reason "Batch size adjusted for available GPU memory" `
  --model_source refit_full_development `
  --amp `
  --resume
```

## Run individual stages

```powershell
# Grid only
python Code\Models\Merlin\train_conv_lora.py --stage grid --percentage 40 --grid true --percentage_specific_grid true --amp --resume

# Five-fold CV using a completed/shared grid
python Code\Models\Merlin\train_conv_lora.py --stage cv --percentage 20 --grid true --percentage_specific_grid false --amp --resume

# Held-out test using the best fold checkpoint
python Code\Models\Merlin\train_conv_lora.py --stage test --percentage 20 --grid true --percentage_specific_grid false --model_source best_cv_fold --amp

# Held-out test using all five fold checkpoints as an ensemble
python Code\Models\Merlin\train_conv_lora.py --stage test --percentage 20 --grid true --percentage_specific_grid false --model_source cv_ensemble --amp

# Held-out test using an explicitly supplied checkpoint
python Code\Models\Merlin\train_conv_lora.py --stage test --percentage 20 --grid false --model_source provided_checkpoint --checkpoint C:\path\best_model.pt --amp
```

Test-only execution requires a matching numbered CV parameter record. Use the
same grid options or explicit parameters that created the intended `CV_x`.

## Preview workload

The Full Fine-Tuning and existing ConvLoRA grids each contain 12 candidates
(60 fold-training runs). The focused ConvLoRA + LLRD grid varies only the base
adapter rate across three values, for 15 fold-training runs. The Linear Probe
grid contains nine learning-rate/weight-decay candidates, for 45 lightweight
fold runs after frozen features have been cached. Preview without training:

```powershell
python Code\Models\Merlin\train_full_finetune.py --stage grid --percentage 40 --grid true --percentage_specific_grid true --dry_run
```

For a one-candidate pipeline check, add `--max_grid_candidates 1`.

## Metrics and artifacts

Outputs retain the previous Merlin names:

- `best_model.pt`
- `last_checkpoint.pt`
- `training_state.json`
- `training_history.csv`
- `metrics_summary.json`
- `validation_patient_predictions.json`
- `validation_probabilities.csv`
- `test_patient_predictions.json`
- `test_probabilities.csv`
- timestamped execution logs

Metrics include macro F1, balanced accuracy, accuracy, macro Average Precision,
class-wise Average Precision, class-wise sensitivity, class-wise specificity,
class-wise precision, and a confusion matrix. AUROC is intentionally omitted.

## Results

```text
Results/Merlin/
|-- LinearProbe/
|   |-- Grid/<percentage>_percent/
|   |-- CV/
|   |   |-- cv_runs.json
|   |   `-- CV_x/<percentage>_percent/fold_01/...fold_05/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
|-- FullFineTune/
|   |-- Grid/<percentage>_percent/
|   |-- CV/
|   |   |-- cv_runs.json
|   |   `-- CV_x/<percentage>_percent/fold_01/...fold_05/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
|-- ConvLoRA/
|   |-- Grid/<percentage>_percent/
|   |-- CV/
|   |   |-- cv_runs.json
|   |   `-- CV_x/<percentage>_percent/fold_01/...fold_05/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
|-- ConvLoRA_LLRD/
|   |-- Grid/<percentage>_percent/
|   |-- CV/
|   |   |-- cv_runs.json
|   |   `-- CV_x/<percentage>_percent/fold_01/...fold_05/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
`-- Cache/
    |-- Shared/<preprocessing_namespace>/
    `-- Features/merlin_image_embedding_2048_<provenance_signature>/
```

Each `<model_source>` folder is named `refit_full_development`, `cv_ensemble`,
`best_cv_fold`, or `provided_checkpoint`. Every percentage-specific held-out
test folder contains `model_selection.json`, recording its `CV_x`, evaluation
strategy, output directory, fold-local checkpoint paths, and hyperparameter
provenance.

Grid folders also contain `grid_summary_by_selection_score.csv` and
`best_hyperparameters.json`. Percentage-specific CV folders contain
`cv_fold_summary.csv` and `best_fold_model.json`.
