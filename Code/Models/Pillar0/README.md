> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# Pillar-0 five-class abdomen CT pipeline

This directory is a new, isolated Pillar-0 implementation for the same
single-label five-class task and reporting workflow used by the repository's
Merlin experiments. It does **not** import, overwrite, or reuse Merlin model
code or Merlin cache artifacts.

The canonical class order is fixed everywhere:

1. `Appendicitis`
2. `Cholezystitis`
3. `Normal`
4. `Perforation`
5. `Sigmoid`

`Cholezystitis` is retained as the dataset label, while clinical prompt text
uses the standard spelling *cholecystitis*. `Sigmoid` is interpreted as acute
sigmoid diverticulitis and `Perforation` as gastrointestinal/hollow-viscus
perforation. These semantic assumptions must be clinically verified against
the original dataset provenance before zero-shot values are treated as more
than exploratory.

## What is implemented

- A frozen official Pillar-0 image encoder plus a five-output linear probe.
- End-to-end supervised fine-tuning of the official image encoder plus a
  five-output classifier.
- An explicitly experimental, capability-gated zero-shot prompt-similarity
  evaluation.
- A new NIfTI-to-RAVE cache at Pillar-0's required 3-D geometry, independent
  of Merlin's cache.
- Publication-adapted, practical-default, and curated-search parameter files.
- Merlin-compatible five-fold development selection, held-out testing,
  metrics, prediction tables, checkpoints, and run metadata.

The complete production cache and a default 100%-development five-fold linear
probe/refit/held-out run have completed. The earlier positive-only five-class
and one-vs-rest prompt runs are immutable legacy v1 evidence and are not
canonical zero-shot results. The replacement v2 five-class margin protocol is
implemented but has not been run. Full fine-tuning has passed a real
encoder-gradient smoke only; it has no reported classification result. Setup
evidence must not be reported as model performance. See
`Results/Pillar0/README.md` for the measured outputs.

## Model access: a separate Pillar-0 acceptance is required

The official checkpoint is the gated
[`YalaLab/Pillar0-AbdomenCT`](https://huggingface.co/YalaLab/Pillar0-AbdomenCT)
repository. Approval or terms accepted for MedSigLIP do not transfer to this
model. Use the same Hugging Face account as the local token, open the model
page, accept its displayed license/access terms and contact-sharing request,
then authenticate that account locally:

```powershell
hf auth login
```

A new token is not necessary if the existing token belongs to the account that
accepted the Pillar-0 gate. The first model run downloads the pinned revision
through `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`.
The 2026-08-10 preflight confirmed that the official checkpoint is accessible
in the current Pillar environment. Separate Pillar acceptance is still required
in principle for a different account or clean environment; MedSigLIP access
does not grant it automatically.

The pinned image checkpoint revision is:

```text
d73d111bb278b3f6411f3c8e72c7e6f8a5fb1bb8
```

Zero-shot additionally uses
[`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
That text model is separate and large; see the zero-shot limitation below.

## Why NIfTI, not PNG/2.5-D or DICOM

Pillar-0 is a full-volume 3-D CT encoder. PNG slices and 2.5-D stacks do not
meet its spatial or eleven-channel input contract and are not accepted by this
pipeline. Official RAVE can ingest either a DICOM series or a NIfTI volume, but
this repository already has one NIfTI per classified patient. NIfTI is selected
here because it preserves the complete 3-D study while avoiding DICOM series
selection ambiguity and duplicate-series risk.

Sources are read only from:

```text
Dataset/NIFTI/<class>/<patient_id>.nii.gz
```

The source NIfTI is converted into a new shared Pillar-0 cache with:

- canonical SimpleITK conversion;
- `1.25 x 1.25 x 1.25 mm` target spacing;
- in-plane crop/pad to `384 x 384`;
- raw RAVE archives with shape `D x 384 x 384`, where `1 <= D <= 384`;
- raw Hounsfield units stored once without forcing short archives to 384 slices;
- RATE-style center pad/crop in the model loader to `384 x 384 x 384`, using
  `-1 HU` for depth padding, followed by a depth-axis flip;
- eleven CT windows applied on the fly: lung, mediastinum, abdomen, liver,
  bone, brain, subdural, stroke, temporal bone, soft tissue, and min/max;
- no additional input normalization, matching the official RATE-Evals
  Pillar adapter.

The default namespace is:

```text
Results/Pillar0/Cache/Shared/pillar0_abdct_rave_1p25mm_384cube_rawhu_lz4_v1
```

This is never Merlin's cache. A cache manifest records source fingerprints,
split/config hashes, exact model and RAVE revisions, output mapping, geometry,
timestamps, and validation status.

### LZ4 versus HEVC

[`rave_abdomen_384_lz4.yaml`](rave_abdomen_384_lz4.yaml) is the executable
default. It preserves raw HU values and does not need an external video codec.

[`rave_abdomen_384_hevc.yaml`](rave_abdomen_384_hevc.yaml) records the official
Pillar-pretraining storage choice for provenance comparison. The pinned RAVE
HEVC exporter maps HU into 10-bit video and requires `ffmpeg` with `libx265`.
The current cache implementation deliberately enforces the raw-HU LZ4
contract, so the HEVC file is not silently substituted into an LZ4 namespace.

## Environment

Use a Pillar-specific environment so installing RAVE cannot disturb Merlin's
environment. On this workstation, a lightweight virtual environment can reuse
the existing CUDA-enabled packages from `Medgemma_gpu`:

```powershell
conda run -n Medgemma_gpu python -m venv .venv_pillar0 --system-site-packages
& .\.venv_pillar0\Scripts\python.exe -m pip install -r Code\Models\Pillar0\requirements.txt
```

If the parent environment does not already contain a CUDA build of PyTorch,
install the build matching the local driver first. The audited workstation has
one RTX 4090. `transformers==4.53.2` is pinned exactly because the official
remote model code was runtime-tested with that release.

The validated end-to-end backward smoke kept FP32 master/model weights and used
BF16 autocast. The released wrapper did not activate gradient checkpointing.
The final post-fix 384-cube smoke required 65.4327 seconds for
forward/backward, peaked at
35,089,342,464 allocated CUDA bytes, and produced 375 finite encoder-gradient
tensors. This is a setup test, not a training metric. Allowing for optimizer
states, allocator fragmentation, validation, and data loading, a GPU with at
least 48 GB VRAM is recommended for full fine-tuning. Experimental all-BF16
model weights timed out locally, and `save_on_cpu` activation offload failed in
the local Windows/CUDA runtime; both remain disabled. The linear probe is the
appropriate baseline for the available 24 GB GPU.

The RAVE dependency is pinned in `requirements.txt` to official commit:

```text
20adeb873021c864e9410fae8898b0c8da309769
```

## Preflight and cache workflow

Run commands from `Systematic_Methodology`. Replace `python` with the intended
Pillar environment's executable when necessary.

Split membership is defined by the dataset folder/manifest patient ID. Every
manifest is validated when loaded: duplicate IDs within a class or partition,
IDs shared by classes, IDs shared by CV folds, and development/held-out overlap
all fail before model work begins. The current manifests contain 1,358 unique
IDs and zero patient-ID overlap. DICOM header demographics are not used to infer
identity or override the declared split.

Record a non-identifying summary of all percentage and CV checks when needed:

```powershell
python Code\Models\Pillar0\audit_split_patient_ids.py `
  --output Results\Pillar0\Setup\patient_id_split_audit.json
```

Create a read-only deterministic cache plan:

```powershell
python Code\Models\Pillar0\prepare_cache.py `
  --action plan `
  --percentage 100 `
  --config Code\Models\Pillar0\rave_abdomen_384_lz4.yaml `
  --expected_shape_dhw 384 384 384 `
  --expected_spacing_xyz_mm 1.25 1.25 1.25
```

Exercise the RAVE invocation without creating cache artifacts:

```powershell
python Code\Models\Pillar0\prepare_cache.py `
  --action prepare `
  --dry_run `
  --percentage 100 `
  --config Code\Models\Pillar0\rave_abdomen_384_lz4.yaml
```

Build and then validate the shared cache:

```powershell
python Code\Models\Pillar0\prepare_cache.py `
  --action prepare `
  --percentage 100 `
  --config Code\Models\Pillar0\rave_abdomen_384_lz4.yaml `
  --workers 4

python Code\Models\Pillar0\prepare_cache.py --action validate
```

Use `--source_fingerprint sha256` for the strongest, slower source audit. Cache
creation is idempotent only when the plan fingerprint matches; an incompatible
existing namespace fails closed instead of overwriting data.

## Training and evaluation workflow

The scripts preserve the existing repository sequence:

1. search on development folds only;
2. five-fold cross-validation for the selected configuration;
3. deterministic best-fold/ensemble/refit handling recorded in metadata;
4. one evaluation on the fixed held-out test set.

`grid` is a Merlin-compatible alias for `search`. Search definitions use
`--grid_config` (alias `--search_config`); a single manual profile uses
`--parameter_source manual --parameter_config <file>`. The exact trained
workflow is:

```powershell
# Frozen encoder + linear probe: curated search
python Code\Models\Pillar0\train_linear_probe.py `
  --stage grid `
  --percentage 40 `
  --grid_config Code\Models\Pillar0\search_linear_probe.json

# CV and held-out test using the selected search candidate
python Code\Models\Pillar0\train_linear_probe.py `
  --stage cv --percentage 40 --parameter_source search --resume
python Code\Models\Pillar0\train_linear_probe.py `
  --stage test --percentage 40 --parameter_source search `
  --model_source refit_full_development

# End-to-end supervised fine-tuning: curated search
python Code\Models\Pillar0\train_full_finetune.py `
  --stage search `
  --percentage 40 `
  --grid_config Code\Models\Pillar0\search_full_finetune.json
python Code\Models\Pillar0\train_full_finetune.py `
  --stage cv --percentage 40 --parameter_source search --resume
python Code\Models\Pillar0\train_full_finetune.py `
  --stage test --percentage 40 --parameter_source search `
  --model_source refit_full_development
```

For a literal manually supplied profile rather than an installed profile name:

```powershell
python Code\Models\Pillar0\train_linear_probe.py `
  --stage all --percentage 40 --parameter_source manual `
  --parameter_config Code\Models\Pillar0\default_linear_probe.json `
  --resume
```

`--parameter_source default` and `--parameter_source publication` load the
corresponding built-in files directly. Use the same percentage for search, CV,
and test. The root PowerShell launchers wrap these arguments.

```powershell
.\Pillar0_Linear_Probe_Run_Script.ps1 `
  -Stage grid -Percentage 40

.\Pillar0_Full_FineTune_Run_Script.ps1 `
  -Stage grid -Percentage 40

.\Pillar0_Zero_Shot_Run_Script.ps1 `
  -Action plan
```

Run these from `Systematic_Methodology`. The cache launcher defaults to a
read-only plan and does not fit or score a model.

Completed supervised held-out results are immutable by default. Use `--resume`
to reuse a matching completed artifact; an intentional replacement additionally
requires `--force_test` and should be documented in the timeline.

### Parameter profiles

| File | Status | Intended use |
|---|---|---|
| `publication_linear_probe.json` | Publication-adapted | RATE-Evals Adam, `1e-3`, batch `8192`, 1000 epochs, no weight decay; five-class CE is an explicit task adaptation. |
| `default_linear_probe.json` | Practical default | Cached embeddings, AdamW at `3e-4`, 500 epochs maximum, batch 256, patience 50, and effective-number class weights. |
| `search_linear_probe.json` | Curated search | Six coupled candidates pruned at 50/150/500 epochs; no Cartesian explosion. |
| `publication_full_finetune.json` | Publication-adapted | RSNA core schedule plus official finetune settings; native abdomen `384^3` replaces the RSNA head crop. |
| `default_full_finetune.json` | Practical default | Batch 1, accumulation 1, at most 500 optimizer steps, FP32 weights with BF16 autocast, and no unsupported checkpointing/offload. Requires substantially more than 24 GB for training. |
| `search_full_finetune.json` | Curated search | Five coupled candidates, with 10-step/one-fold and 50-step/two-fold pruning rounds before full CV. |

"Publication-adapted" is deliberate. The study demonstrates both frozen
linear evaluation and supervised end-to-end adaptation, but not this exact
five-class task. Each publication profile states which values are sourced and
which choices resolve an unreported or task-specific detail.

Both search files use the robust Merlin-compatible selection score:

```text
0.7 * (mean macro-F1 - std macro-F1)
+ 0.3 * (mean balanced accuracy - std balanced accuracy)
```

Perforation recall is the first tie-break because it is the smallest class.
Successive-halving rounds increase budget and fold coverage while discarding
weak candidates. The held-out test never participates.

## Experimental zero-shot

Inspect access, then run the one canonical fixed profile without prompt,
threshold, or temperature tuning:

```powershell
python Code\Models\Pillar0\run_zero_shot.py `
  --action access

python Code\Models\Pillar0\run_zero_shot.py `
  --action run `
  --prompt_profile Code\Models\Pillar0\zero_shot_prompts_five_class_margin_v2.json
```

Pillar-0 did **not** demonstrate zero-shot classification in the cited study.
Its image encoder was aligned asymmetrically to frozen Qwen3-Embedding-8B
report representations, but the released 93M-parameter vision checkpoint does
not itself contain the 8B text tower. A valid run therefore requires all of:

1. Qwen embeddings produced with the pinned official preprocessing contract;
2. the trained Pillar text projection compatible with the image embedding;
3. the finite positive logit scale stored in the Pillar checkpoint;
4. L2 normalization and fixed mean-per-prompt cosine scoring;
5. no development/test-driven prompt, threshold, or temperature selection.

For every class `c`, the v2 score is fixed as:

```text
class_score[c] =
    mean cosine(image, positive prompts[c])
  - mean cosine(image, negative prompts[c])

scaled_logit[c] = checkpoint_logit_scale * class_score[c]
prediction = argmax(class_score)
relative_score = softmax(scaled_logit)
```

Each projected prompt is normalized separately. The positive and negative
prompt means are deliberately not renormalized, so each dot product remains
the exact mean of the individual prompt cosine similarities. The softmax
values are uncalibrated relative scores, not clinical probabilities.

The released checkpoint previously passed both the trained 4096-to-1152 text
projection probe and the learned logit-scale check. That establishes technical
compatibility only. The runner fails closed if either component is unavailable;
raw Qwen output, a random projection, or an arbitrary temperature fallback is
forbidden. Every output remains `experimental` and
`publication_demonstrated: false`.

The canonical profile is
`zero_shot_prompts_five_class_margin_v2.json`, with output under
`ZeroShot/Test/exploratory_five_class_prompt_similarity/<prompt_set_id>/`.
The hash-bound v1 `zero_shot_prompts.json` is retained unchanged solely to
explain archived results; the current scorer rejects it. The old positive-only
and one-vs-rest results must never be relabeled as v2.

Prompt text and its SHA-256 are copied into result provenance. Changing even
one phrase requires a new `prompt_set_id`; never replace an existing prompt set
and compare it as if unchanged. Existing prompt-set result directories fail
closed; use CLI `--force` or launcher `-Force` only for an intentional,
documented replacement.

## Results and reporting

### Canonical completed results

The complete lossless production cache contains all 1,358 patients, occupies
61.64 GB, and passed exhaustive archive validation. Raw depths range from 158
to 384 before the RATE loader produces `384^3` model input.

| Method | Development protocol | Held-out N | Accuracy | Balanced accuracy | Macro-F1 | Macro AP | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Frozen encoder + default linear probe | 5-fold CV, median best epoch 75, full-development refit | 278 | 0.6978 | 0.6698 | 0.6520 | 0.7175 | Supervised baseline; `CV_1`, best fold 3 |

The authoritative development and held-out patient IDs are disjoint. The v2
experimental five-class margin method has not been run, so no zero-shot metric
is included in the canonical table. The archived v1 positive-only run reported
accuracy 0.5899 and macro-F1 0.5487, but its scoring contract is superseded and
its values must not be reused or compared as v2. End-to-end full fine-tuning is
implemented and backward-validated, but no full-fine-tune classification metric
has been generated because the validated step requires 35.09 GB peak allocated
memory on this workstation.

Generated artifacts follow this layout:

```text
Results/Pillar0/
|-- Cache/Shared/<cache_namespace>/
|   |-- cache_plan.json
|   |-- cache_manifest.json
|   |-- source_series.csv
|   `-- artifacts/
|-- Setup/
|-- LinearProbe/
|   |-- Grid/<percentage>_percent/
|   |-- CV/CV_x/<percentage>_percent/{fold_01...fold_05,refit_full_development}/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
|-- FullFineTune/
|   |-- Grid/<percentage>_percent/
|   |-- CV/CV_x/<percentage>_percent/{fold_01...fold_05,refit_full_development}/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
`-- ZeroShot/Test/exploratory_five_class_prompt_similarity/<prompt_set_id>/
```

Supervised run directories retain the established artifact names where the
stage makes them applicable:

- `best_model.pt` and `last_checkpoint.pt`;
- `training_state.json` and `training_history.csv`;
- `metrics_summary.json`;
- `validation_patient_predictions.json` / `test_patient_predictions.json`;
- `validation_probabilities.csv` / `test_probabilities.csv`;
- timestamped run logs and exact configuration/provenance snapshots.

When the v2 zero-shot run is executed, its directory contains
`metrics_summary.json`, `test_patient_predictions.json`,
`test_probabilities.csv`, `test_similarity_scores.csv`,
`prompt_profile_snapshot.json`, and `model_selection.json`. The similarity
table records each class's positive mean cosine, negative mean cosine, margin,
and scaled logit for every patient.

The metric schema remains: accuracy, balanced accuracy, macro F1, macro average
precision, per-class precision/recall/sensitivity/specificity/AP, confusion
matrix, and canonical class order. Predictions and probabilities remain
patient-level and mutually exclusive through softmax.

The retained files under `Results/Pillar0/Setup` and the three
`*_smoke_*` cache namespaces are setup/smoke evidence. The validated five-case
cache contains one source per class, and the model smoke exercised one volume.
They are not classification datasets and must never be discovered as Grid, CV,
Test, or ZeroShot metric results.

## Split and leakage controls

- Existing percentage manifests and their five deterministic folds are loaded,
  not regenerated by the model code.
- Dataset folder/manifest patient IDs are authoritative. All 1,358 are unique,
  development and held-out IDs are disjoint at every percentage, and the five
  CV folds are mutually disjoint. Manifest loading and runtime partition checks
  fail if this contract is violated. DICOM header demographics do not redefine
  patient identity or split membership.
- RAVE caching is deterministic, label-independent preprocessing. A shared raw
  cache may contain held-out volumes, but held-out labels/features are never
  read by parameter fitting, early stopping, prompt selection, or search.
- Linear embeddings are generated by the frozen, pinned encoder. Classifier
  fitting and feature standardization, if enabled later, must use training-fold
  data only.
- Full-fine-tune augmentation is applied only to the training partition.
- Search rankings use development folds only. The held-out test is evaluated
  once after all choices are fixed.
- Zero-shot prompts, positive-minus-negative aggregation, and checkpoint
  temperature are fixed before held-out evaluation and are not calibrated on
  this dataset.
- Every result records split hashes, cache manifest hash, source/config hashes,
  code/model revisions, seed, class order, and timestamps.

## Isolation, audit trail, and rollback

Pillar-0 code and static configuration live only under
`Code/Models/Pillar0`; generated model/cache artifacts live only under
`Results/Pillar0`. No Merlin file or Merlin cache is a Pillar write target.

Date-level changes are summarized in the methodology changelog. Time-ordered
commands, files, decisions, validation evidence, and blockers are recorded in
`Logs/Pillar0/IMPLEMENTATION_TIMELINE.md`. Because a valid Git worktree is not
assumed, that timeline and file hashes are the rollback source of truth.

To roll back generated work, first preserve any desired result metadata, then
remove only the explicitly named Pillar-0 result namespace/run directory. The
source NIfTI/DICOM datasets and Merlin subtree are never deletion targets. To
roll back the implementation itself, use the timeline's created-file inventory
and reverse only its Pillar-0 additions/shared-document entries.

## Audited official sources

- [Pillar-0 paper](https://arxiv.org/html/2511.17803v1)
- [Pillar-0 project site](https://yalalab.github.io/pillar-0/)
- [Official Pillar0-AbdomenCT checkpoint](https://huggingface.co/YalaLab/Pillar0-AbdomenCT)
- [Official pillar-pretrain repository](https://github.com/YalaLab/pillar-pretrain), audited at `ab06083ac0d3706017780fb3e3facd23c4fbd0cd`
- [Official pillar-finetune repository](https://github.com/YalaLab/pillar-finetune), audited at `47506d5267cb448867fe415d416910ce6aad9422`
- [Official RATE-Evals repository](https://github.com/YalaLab/rate-evals), audited at `508afc03691dca8dcad23918f94540cb927d0614`
- [Official RAVE repository](https://github.com/YalaLab/rave), audited at `20adeb873021c864e9410fae8898b0c8da309769`

The pinned commits and model revisions are recorded in cache/run metadata so a
later upstream change cannot silently alter an experiment.

## Atlas partial fine-tuning

The additive `AtlasFinalBlockFineTune` and `AtlasFinalStageFineTune` workflows
leave the existing linear probe, full fine-tune, and exploratory zero-shot code
untouched. They have independent method IDs, profiles, CV registries,
checkpoints, and result roots. Both load the pinned
`YalaLab/Pillar0-AbdomenCT` revision and validate the released Atlas stage
depths as `[2,2,8]`:

- `final_block`: `model.visual.atlas_models[2].blocks[7]` plus a new
  `LayerNorm(1152) -> Linear(5)` head;
- `final_stage`: all of `model.visual.atlas_models[2]` plus the same head.

All other image and text/checkpoint parameters stay frozen. Native
eleven-channel windowing and released multiscale max pooling remain unchanged.
Input volumes come only from the validated normal raw-HU cache namespace. The
publication profile retains the reported AdamW `1e-5`, effective batch 32
(the publication used batch 8 times accumulation 4), 10,000-step ceiling, 1,000-step
no-improvement rule, and random rotations. The new five-class head, head LR,
scheduler, weight decay, validation interval, and selected block/stage scope
are explicitly marked adapted inside the corresponding
`parameter_source: publication` profile.

Fixed publication, project-grid, and practical-default profiles are provided
for each strategy. Publication mode bypasses selection and proceeds directly
to CV/refit/test; the three-candidate manual grid runs only with
`parameter_source: grid`. Their
`# source (not executable)` fields are valid JSON provenance and are stripped
before execution/candidate hashing. The launcher defaults to publication,
`InputSource=frozen`, and one CPU microbatch of prefetch. The microbatch default
remains **1**: the publication effective batch of **32** therefore accumulates
32 microbatches before an optimizer update. An explicit microbatch must be
positive, at most the effective batch, and divide it evenly.

### Temporary representations and unchanged random augmentation

The default `frozen` route builds a temporary representation cache at launch,
using the existing permanent RAVE cache. It stores all ten distinct orientations
that the current augmentation can produce. Training still draws a plane from
three choices and a turn count from four choices, in the same order with the
same generator. The three zero-turn combinations all select identity: its
probability remains 3/12, while each other orientation remains 1/12. The cache
does not replace these draws with uniform selection over ten views. Validation
and held-out inference always select identity. No new augmentation is introduced.

Each orientation is encoded once per cache build. For `final_block`, extraction
stops before `atlas_models[2].blocks[7]`; for `final_stage`, it stops before
`atlas_models[2]`. It saves 64 x 384 final-scale tokens and two earlier 384-D
pooled vectors. Training runs only the selected block/stage and classifier,
with the released multiscale readout and native L2 normalization preserved.
With `-AdaptationScope both`, both boundaries are saved during the same traversal
of each orientation, then the two independent strategies run in sequence.
Single-scope launches save only the selected boundary. Full fine-tuning and
linear probing continue to use their existing implementations.

The cache is lossless and stores the native activation dtype. FP32 tensor data
for all ten views costs about **0.97 MiB per patient per boundary**, or **1.28 GiB
for 1,358 patients**, before file/manifest overhead; both boundaries double that
budget. BF16 can use less. A disk preflight includes file overhead, 8% margin,
and 2 GiB of free headroom. Initial extraction still costs ten frozen-prefix
passes per patient; subsequent training reuses those outputs. No cohort speedup
or accuracy comparison has been measured.

`-FrozenCacheRoot` overrides the default `Results/Pillar0/Cache/FrozenRuns/`.
An exclusive lease prevents concurrent writers. The cache binds raw-archive
hashes, loaded vision weights, installed encoder source, precision/runtime,
augmentation and boundary contracts. Sealed, atomic per-patient progress allows
an interrupted build to reuse completed patients. Before training, all ten views
of the first newly cached patient must pass the original full encoder's forward
check at `rtol=1e-5`, `atol=1e-6`, including the CPU storage/stacking layout.
Checkpoint identity also binds ordered training/validation tensor-value hashes.

After every requested scope and stage in a launch completes, its owned temporary
cache is deleted. On interruption or error it is retained for resume. The
permanent RAVE and linear-probe feature caches remain available. A later launch
after cleanup recreates the required representations. The existing
`Pillar0_Cache_Run_Script.ps1` continues to prepare permanent RAVE inputs;
temporary rotation representations are created by the training launcher.

Prefetch uses one worker and at most one waiting microbatch in **CPU RAM**. It
loads the next selected cached view while the GPU handles the current batch.
The worker uses a private copy of the augmentation generator; consuming a batch
commits its state, while discarding a waiting batch does not. Patient order,
effective batch, accumulation and checkpoint resume semantics are preserved.
`-NoPrefetch` disables it. `-InputSource cache` retains the original raw-HU
training path and also supports prefetch. This mode uses substantially larger
CPU input tensors. `frozen` is the default for new runs.

### Fresh 100% run and preserved backup

On 2026-09-14, all existing final-block `CV_1/100_percent` artifacts were copied
to `Results/Pillar0/AtlasFinalBlockFineTune/Backup_CV_1/`, together with the CV
registry, immutable parameter document and original implementation/launcher
snapshots. Its `BACKUP_MANIFEST.json` records sizes and SHA-256 hashes for all
37 copied files (238,209,675 bytes). The old run contains two completed folds
and an interrupted third; it is not a completed five-fold/refit/test result.
Original artifacts also remain unchanged under `CV_1`.

The new frozen identity was verified to allocate **CV_2** using a temporary
copy of the current registry. No production `CV_2` or representation cache was
created during implementation. Start the requested fresh run from the
`Systematic_Methodology` directory:

```powershell
.\Pillar0_Atlas_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_block `
  -Percentage 100 `
  -InputSource frozen `
  -NoResume
```

`-NoResume` starts the new configuration and rejects pre-existing outputs for
that same configuration; it does not overwrite them. After this new run is
interrupted, omit `-NoResume` to continue it. The old RAVE-input checkpoint is
kept as backup, rather than converted into a frozen-input checkpoint.

Other supported launcher plans:

```powershell
.\Pillar0_Atlas_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_block -DryRun
.\Pillar0_Atlas_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope both -Percentage 100 -DryRun
.\Pillar0_Atlas_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_stage -Stage all -Percentage 40 `
  -ParameterSource publication
.\Pillar0_Atlas_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_block -Stage all -Percentage 40 `
  -ParameterSource grid
```

Outputs are isolated under
`Results/Pillar0/AtlasFinalBlockFineTune/{Grid|CV|Test}` and
`Results/Pillar0/AtlasFinalStageFineTune/{Grid|CV|Test}`. Fixed publication and
default runs have no grid directory and
cannot overwrite `LinearProbe`, `FullFineTune`, or `ZeroShot`. Exact
optimizer-step resume binds the raw cache manifest, immutable model revision,
split/profile/code/runtime identity, ordered concrete input-file identity,
current and best trainable states,
optimizer/scheduler, shuffled stream, augmentation generator, and CPU/CUDA
RNG. Frozen mode adds the selected boundary/augmentation contract and concrete
tensor-value identity. The original numerical implementation files retain
their recorded hashes.

The 2026-09-14 Pillar/shared validation passed **74 tests plus 6 subtests**.
The pinned full-size model matched original features exactly for all ten views
at both boundaries on one synthetic 384-cube input in BF16 and FP32, with actual
temporary shard I/O and cleanup. Small released-model component tests matched
logits, gradients and optimizer updates; pipeline tests matched training and
validation and checked exact interruption/resume behavior. Corruption, partial
cache recovery, exclusive access and bounded-prefetch cases also passed.
This is numerical/workflow validation, not a measured cohort accuracy result;
different GPU microbatch shapes can still introduce floating-point differences.
No new production training or cohort cache extraction was started. Evidence is
`Logs/Pillar0/VALIDATION_20260914_FROZEN_AUGMENTATIONS.json` from the workspace
root. The existing exploratory zero-shot implementation remains unchanged.
