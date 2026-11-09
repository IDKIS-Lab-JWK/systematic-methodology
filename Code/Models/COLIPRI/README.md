> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# COLIPRI pooling-optimised frozen-encoder probe

This package implements COLIPRI's model-native classification-probe family for
the systematic five-class CT methodology. The complete image encoder is frozen,
its raw pre-pooling dense feature map is cached once per patient, and the
publication-described pooling heads and five-class classifier are trained on
development data only.

The project reports this complete searched family as
`PoolingOptimisedLinearProbe` (`pooling_optimised_linear_probe`) because only
average and maximum pooling are mathematically strict linear probes. The three
attention-pooling variants contain trainable pooling parameters; their
artifacts record that distinction explicitly.

The namespace migration did not run patient caching, classifier training,
CV/refit, inference, or held-out evaluation. Existing results were moved from
the legacy `LinearProbe` folder without rewriting their sealed payloads or
binary checkpoints.

## Scientific basis and five-class adaptation

The implementation is based on Appendix C.2 (classification probing) of the
[COLIPRI paper](https://arxiv.org/abs/2510.15042) and the
[official Microsoft model/code release](https://huggingface.co/microsoft/colipri/tree/be872ebec5c62894cf183b09835ec6fd7be537f5).
It preserves the complete ordered Cartesian search:

| Search dimension | Publication-backed values |
| --- | --- |
| Pooling | `average`, `max`, `learned_attention`, `average_attention`, `multi_learned_attention` |
| Learning rate | `0.003`, `0.01`, `0.03`, `0.1` |
| Candidate count | `5 x 4 = 20` |

Every candidate shares the following fixed recipe:

| Parameter | Value |
| --- | ---: |
| Frozen image encoder | Yes |
| Classifier input dimension | 864 |
| Attention heads, when applicable | 12 |
| Optimizer | SGD |
| Momentum | 0.95 |
| Weight decay | 0 |
| Schedule | Cosine annealing to zero |
| Optimizer steps | 12,500 |
| Effective batch size | 16 |
| Output classes | 5 |

The detailed Appendix C.2 value of 12,500 steps is used as the reproducible
source when less-specific publication text differs. `publication` mode rejects
profile replacement, parameter injection, and candidate truncation. The
packaged four-candidate `grid` profile is a project search and cannot be
reported as publication-backed. Fixed `default` and `manual` profiles are
available only through `cv` or `test`, not through `grid` or `all`.

The original probe predicts 18 CT-RATE findings. This repository instead uses
one mutually exclusive five-class head in the canonical order `Appendicitis`,
`Cholezystitis`, `Normal`, `Perforation`, `Sigmoid`, the existing patient-grouped
five folds, and the sealed held-out cohort. Candidate selection uses mean
five-fold macro one-vs-rest AUPRC, with deterministic tie-breakers recorded in
`publication_pooling_optimised_linear_probe.json`.

The paper does not specify a probe loss for this adapted endpoint. The project
therefore uses the class-balanced effective-number method of
[Cui et al. (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html):
training-partition-only weighted multiclass cross-entropy with the project
choice `beta=0.999`; no weighted sampler is used. This loss and beta, the five
labels, folds, and selection rule are task adaptations. The preserved
pooling/LR search and fixed optimizer schedule remain publication-backed, but
results are not described as an exact reproduction of the original endpoint.

## What each pooling candidate trains

- `average` and `max` pool the frozen 864-channel tokens without trainable
  pooling parameters, then train only `Linear(864, 5)`. These are strict linear
  probes.
- `learned_attention` trains one learned query and a 12-head attention pool
  before the classifier.
- `average_attention` uses the mean frozen token as the query but trains the
  12-head attention pool and classifier.
- `multi_learned_attention` trains four queries and the attention pool, averages
  their four outputs, and applies the classifier.

The encoder stays in evaluation mode with every encoder parameter frozen for
all 20 candidates. The complete family is named
`PoolingOptimisedLinearProbe` for the systematic experiment table, while
`strict_linear` in each saved architecture prevents attention-pooling
candidates from being mistaken for classifier-only probes.

The released package supplies the mean-query `AttentionPool1D` and four-query
`MultiLearnedQueryAttentionPool1D`, but it does not release the original
classification-probe training script or the single-learned-query head. The
latter is therefore implemented as the direct one-query analogue using
PyTorch `MultiheadAttention`, 12 heads from the pinned model configuration, and
query initialization `N(0,1) / sqrt(864)`. Average and maximum pooling are
parameter-free. The grid and schedule are publication-backed; unspecified
head initialization and the adapted loss prevent a bit-exact reproduction
claim.

## Pinned source, checkpoint, and released processor

Source and weights are pinned to official revision
`be872ebec5c62894cf183b09835ec6fd7be537f5`. The required
`model.safetensors` is exactly 1,033,541,380 bytes with SHA-256
`97d66586507d1f48577b5149777d02e12d25787435a2a295cac5563df9291c14`.
Provisioning rejects a dirty or differently revised source tree and rejects a
checkpoint with a different size or digest. It can use an explicit local
checkpoint, the pinned checkout, or the Hugging Face cache/download; asset
auto-provisioning events are appended under `Results/COLIPRI/Setup/`.

This implementation calls the released official image-only processor. It
produces RAS-oriented, 2 mm isotropic volumes, linearly resamples with
antialiasing disabled, clamps HU to `[-1000, 1000]`, maps intensities to
`[-1, 1]`, and centre-crops or pads with the minimum value to
`1 x 192 x 192 x 192`. Processor configuration files and their hashes are part
of the cache identity.

The paper used an unshipped precomputed multi-resolution pyramid. The released
processor is the reproducible public inference path, but it is not claimed to
be bit-identical to that unpublished preprocessing. This boundary is retained
in every cache contract.

## Environment and setup

Use the isolated CPython 3.11 environment from the methodology root:

```powershell
cd Systematic_Methodology

# Uses py -3.11 when the Windows launcher is installed.
.\COLIPRI_Environment_Setup.ps1

# Or supply a CPython 3.11 executable directly.
.\COLIPRI_Environment_Setup.ps1 -BootstrapPython C:\Python311\python.exe

# Side-effect-free setup plan, then pinned asset/runtime verification.
.\COLIPRI_Setup_Run_Script.ps1 -DryRun
.\COLIPRI_Setup_Run_Script.ps1

# Optional real model-construction preflight after setup.
.\COLIPRI_Setup_Run_Script.ps1 -LoadModel
```

`METHOD_IDENTITY_MIGRATION.json` at the result root maps the historical
`LinearProbe` label to this canonical method. Historical sealed JSON, CSV, and
checkpoint payloads retain their original embedded label and hashes; analysis
must use the canonical folder/manifest identity. New executions write only to
`PoolingOptimisedLinearProbe`, leaving `Results/COLIPRI/LinearProbe` available
for a future strict classifier-only probe.

Normal resume is compatible with those untouched historical identities. The
runner verifies the migration manifest's exact legacy profile, probe
implementation, and cache implementation hashes, then requires all remaining
candidate, split, feature/cache, runtime, seed, precision, and checkpoint
fields to match. Legacy absolute paths remain serialized to preserve existing
seals but are resolved read-only through the canonical folder. This exception
is migration-specific; unrelated implementation or experiment changes still
fail closed, and `-NoResume` disables it.

The environment pins PyTorch 2.8/CUDA 12.8 and the direct dependencies in
`requirements.txt`, including the upstream-lock-backed SciPy `1.16.2`.
Environment installation ends with the model's
side-effect-free `check_setup.py --dry_run`; asset provisioning and optional
model construction remain explicit setup steps.

For feature extraction, `auto` uses BF16 when CUDA reports support and
otherwise FP16; CPU extraction is FP32. Cached dense tensors are always stored
in FP32. Probe training supports `auto`, `bf16`, or `fp32`: `auto` selects BF16
when supported and otherwise FP32. Training does not silently use FP16.

## Cache targets

The cache PowerShell launcher and `prepare_cache.py` accept
`-CacheTarget probe|finetune|both` (Python: `--cache_target`), defaulting to
**both**. `Action` still defaults to `plan`; use `prepare` to build.

| Target | Stored data | Used by |
| --- | --- | --- |
| `probe` | Existing dense encoder features in `Cache/Features/` | Pooling probe |
| `finetune` | Preprocessed FP32 CTs in `Cache/Volumes/` | Both Primus-M fine-tuning scopes |
| `both` | Both caches, processed sequentially | All three training methods |

Each target covers the sealed 100% inventory once. The final-block and
final-stage methods share identical CT preprocessing. Probe extraction keeps
its existing direct NIfTI path and feature identity. One prepared CT is
`[1,192,192,192]` FP32 (27 MiB): about 35.81 GiB for 1,358 patients, or
96.23 GiB with probe features, before overhead/free-space margins.
`-CacheRoot` is the parent of both cache directories. `Device` and
`FeaturePrecision` apply to probe extraction; CT preprocessing always runs on
CPU, preserves FP32 values, and does not load encoder weights.

These are the two permanent cache types. Temporary frozen encoder outputs are
created automatically by the fine-tuning launcher, not by this cache script;
see the frozen-output section below.

```powershell
.\COLIPRI_Cache_Run_Script.ps1 -Action plan
.\COLIPRI_Cache_Run_Script.ps1 -Action prepare
.\COLIPRI_Cache_Run_Script.ps1 -Action prepare -CacheTarget finetune
.\COLIPRI_Cache_Run_Script.ps1 -Action validate -CacheTarget finetune
```

CT shards and progress are written atomically. The sealed manifest binds source
identities, the pinned processor/configuration, preprocessing code, CPU runtime,
shape and dtype. Interrupted preparation rehashes sources and verifies reusable
shards. SHA-256 preparation or exhaustive validation issues
`volume_cache_execution_ready.json`, binding the manifest and artifact stats.
Training requires that receipt and checks source stats. `stat` and structural
validation are diagnostic and cannot establish readiness. Completed `prepare`
reports readiness and any required validation. Changed inputs/configuration
cannot silently reuse old CTs; use a new cache root when rebuilding changed inputs.

## Direct streamed probe feature cache

The cache is direct:

```text
NIfTI -> official 192-cube processor -> frozen image encoder
      -> raw unprojected/unpooled 864 x 24 x 24 x 24 FP32 tensor
```

One tensor is about 45.56 MiB. The sealed 1,358-patient inventory therefore
requires about 60.42 GiB before filesystem/PyTorch overhead. Preparation
preflights remaining space with an 8% margin plus 2 GiB before model creation;
use `-CacheRoot` on a larger drive when needed. Extraction processes one patient
at a time. Probe training assembles patients in order and, by default, prepares
one next microbatch in a CPU thread while the current one runs on the GPU.
`NumWorkers` remains `0` (no multiprocessing loader). `-NoPrefetch` disables
the thread; microbatch size controls training memory use.
Cache `Percentage` is fixed at
`100`; plan, prepare, and validate form this full inventory once. Training and
test launchers vary the experiment percentage.

```powershell
# Read-only inventory, asset, shape, and storage plan.
.\COLIPRI_Cache_Run_Script.ps1 -Action plan -Percentage 100 -CacheTarget probe

# Resumable extraction; choose a larger drive if required.
.\COLIPRI_Cache_Run_Script.ps1 -Action prepare -Percentage 100 -CacheTarget probe `
  -CacheRoot E:\COLIPRI_Cache

# Reportable validation: fresh source/artifact SHA-256 plus payload loading.
.\COLIPRI_Cache_Run_Script.ps1 -Action validate -Percentage 100 -CacheTarget probe `
  -CacheRoot E:\COLIPRI_Cache
```

Source-file SHA-256 is the scientific default. Initial preparation reads the
approximately 92 GB NIfTI cohort once and records each hash; an interrupted
preparation reuses a stored hash only when path, byte size, and nanosecond mtime
remain unchanged. Explicit validation deliberately rereads every source and
feature artifact and loads each payload. `-SourceFingerprint stat` and
`-SkipArtifactValidation` are faster diagnostic modes, but neither establishes
the reportable source/content identity or execution readiness. Non-dry-run
COLIPRI probe entrypoints reject `stat`; it is available only through cache
diagnostics or probe dry-run planning.

An uninterrupted fresh SHA-256 preparation validates every source and payload
as it is created and atomically issues
`feature_cache_execution_ready.json`. A resumed preparation must be followed by
explicit exhaustive SHA-256 validation, which issues or refreshes that receipt.
The receipt binds the completion file and seal, manifest file, entry
fingerprint, and a full inventory fingerprint over every artifact's relative
path, byte size, and nanosecond mtime. Real `FeatureStore` construction and
training fail closed if the receipt is missing, modified, or stale. Fast
startup checks the receipt, source statistics for requested rows, and artifact
statistics; any path/size/mtime change requires rerunning exhaustive validation.
The receipt records its validation source and SHA-256 source mode, hashes both
its binding and its own document, and records artifact count and total bytes in
the binding alongside the artifact-stat inventory hash.

Artifacts are written atomically. The progress file, final manifest, completion
seal, exact 100% patient/source inventory, pinned source/checkpoint,
preprocessing configuration, feature shape, extraction precision, and source
fingerprint mode all participate in cache identity. Cache schema 4 also binds
the resolved Python implementation/version/build/machine, PyTorch, requested
and resolved precision, device type/index/name/capability, CUDA/cuDNN, and
pinned direct-package versions, plus SHA-256s of local `__init__.py`, `cache.py`,
and `model.py`. A dry plan explicitly marks this runtime as
`resolved_at_execution`. The same `CacheRoot`,
precision, and fingerprint settings must be passed to cache and training
launchers.

Calling `prepare` on a completed cache returns `execution_ready`, the receipt
path when present, and a concrete `next_action` when exhaustive SHA-256
validation is still required. A reused completion by itself is not represented
as permission to train.

## Grid, CV, refit, and held-out runs

Plan a single complete publication experiment without training:

```powershell
.\COLIPRI_Pooling_Optimised_Linear_Probe_Run_Script.ps1 -Stage all -Percentage 100 `
  -ParameterSource publication -CacheRoot E:\COLIPRI_Cache -DryRun
```

Run every requested percentage and held-out model source after exhaustive cache
validation:

```powershell
.\COLIPRI_Pooling_Optimised_Linear_Probe_All_Run_Script.ps1 `
  -ParameterSource publication -CacheRoot E:\COLIPRI_Cache
```

The wrapper checks every child process's native exit code, stops on the first
nonzero result, and processes `80, 60, 40, 20` as the outer order (the
existing 100% run is not repeated by this resume wrapper). Within
each percentage it runs `Stage=all` for
`refit_full_development`, then `Stage=test` for `cv_ensemble` and
`best_cv_fold`. It accepts only `publication` or `grid`, because `all` includes
a search. Use the single-run launcher with `Stage=cv` or `Stage=test` for a
fixed `default` or `manual` profile.

The publication effective batch remains 16. Both PowerShell probe launchers
and the Python entrypoint default `MicroBatchSize` to the full batch, **16**.
It may be `1, 2, 4, 8, 16`; gradient accumulation is derived as
`16 / MicroBatchSize`. Larger values fail before training with an error naming
the maximum batch size; smaller values must divide the effective batch evenly.
To resume a historical probe run made with microbatch 1, explicitly pass
`-MicroBatchSize 1` (Python: `--micro_batch_size 1`). Microbatch size is part
of checkpoint identity and cannot be changed during exact resume.
`CheckpointInterval` defaults to 250 optimizer steps. Training uses a
deterministic serializable shuffled stream, saves the state of every candidate,
optimizer, scheduler, and stream, and resumes only when the complete training
identity matches. `-NoResume` disables that normal behavior.

For the current interrupted publication run, the guarded preflight identifies
the 40% fold-2 checkpoint at optimizer step 2,750 of 12,500 with all 20
candidates pending. Resume continues from that checkpoint; it does not replace
the completed fold 1 or the completed 60%/80% outputs.

Checkpoint model tensors are restored on the requested accelerator, while the
serialized batch order and `torch.Generator` byte state are explicitly restored
on CPU because the deterministic stream uses a CPU generator. This permits an
already-completed shared-training checkpoint to finish writing any interrupted
candidate artifacts without retraining or changing the scientific identity.

The complete publication grid is attempted even when an extreme learning-rate
candidate diverges on this adapted five-class weighted-loss task. A non-finite
loss or gradient marks only that candidate/fold as
`failed_optimizer_divergence`; its last finite diagnostic artifact and sealed
failure provenance are retained, but it is ineligible for ranking, CV, refit,
or held-out evaluation. The other candidates continue unchanged. The runner
does not silently clip gradients, lower the declared learning rate, or replace
the failed trial. If that failed model also produces non-finite validation
outputs, its fold is sealed with `metrics_status` set to
`not_computed_failed_optimizer_divergence`, null metrics, and empty prediction
files; no synthetic probabilities or performance values are created. Any
candidate with one or more such failed folds has null aggregate metrics and is
ranked after all eligible candidates. If implementation/profile identity changes after a failure
that occurred before the first checkpoint, the launcher may refresh only the
context/index pre-training scaffold; the presence of any checkpoint or
candidate artifact still makes a context mismatch fail closed.

Outputs remain model-local:

```text
Results/COLIPRI/
|-- Setup/asset_provisioning.jsonl
|-- Setup/frozen_cache_events.jsonl
|-- Cache/FrozenRuns/<owner-and-contract>/  # temporary; retained on failure
|   |-- artifacts/<patient-key>_<scope>.pt
|   `-- frozen_cache_manifest.json
|-- Cache/Features/<contract-signature>/
|   |-- artifacts/<patient-key>.pt
|   |-- feature_cache_progress.json
|   |-- feature_cache_manifest.json
|   |-- feature_cache_completion.json
|   `-- feature_cache_execution_ready.json
|-- Cache/Volumes/<contract-signature>/
|   |-- artifacts/<patient-key>.pt
|   |-- volume_cache_progress.json
|   |-- volume_cache_manifest.json
|   `-- volume_cache_execution_ready.json
`-- PoolingOptimisedLinearProbe/
    |-- Publication_Grid/<percentage>_percent/...
    |-- Grid/<percentage>_percent/...
    |-- CV/cv_runs.json
    |-- CV/CV_x/<percentage>_percent/
    |   |-- fold_01/...fold_05/
    |   `-- refit_full_development/
    `-- Test/CV_x/<model_source>/<percentage>_percent/...
```

Each exact selected parameter dictionary receives a stable `CV_x` registration.
Grid reuse must match the complete grid context; CV seals bind split and
development-row fingerprints, seed, parameters, and model/cache identity.
The exact `cv_run_context.json` is sealed before fold work, and completion
recursively verifies all five nested fold seals plus the full-development refit
seal. Held-out lookup never writes the registry or `cv_parameters.json` and
requires the stable source/profile/candidate provenance identity. This keeps
one `CV_x` for the same configuration across percentages, while each percentage
context and completion seal binds its full actual provenance. New model/CV
identity hashes `backend.py`, `pooling.py`, the isolated
`_pooling_optimised_linear_probe/workflow.py`, and shared
`_linear_probe/common.py` and `metrics.py`, plus Python, NumPy, SciPy,
scikit-learn, PyTorch, precision, and accelerator runtime details. Resumed folds/checkpoints
fail closed after a relevant code/runtime change. Validation checks exact fold
and refit contexts, patient partitions, parameters, feature identity, and grid
origin. Held-out reuse binds exact artifact hashes/model source,
parameters/model identity, completion context, and ordered patient/truth rows.
A custom `--results_root` derives its cache and provisioning-log paths unless
either is explicitly overridden. Held-out testing resolves that exact sealed CV
before using one of:

Completed-grid validation recomputes every candidate summary from its five
sealed fold metrics and compares the complete ranking. Nonempty partial grid or
CV roots without their root context fail closed. A supplied `candidate_id` must
be exactly the canonical 12-character fingerprint of the remaining parameter
dictionary.

- `refit_full_development`: one selected head refitted on all development
  patients;
- `cv_ensemble`: mean of the class-aligned probabilities from all five fold
  heads;
- `best_cv_fold`: the one fold head selected only from development validation
  metrics.

The primary held-out report is predeclared as `refit_full_development`.
`cv_ensemble` and `best_cv_fold` remain secondary sensitivity analyses; they
must not be promoted after inspecting held-out metrics.

Held-out labels never participate in grid selection, CV selection, or refit.
`-ForceTest` is the explicit override for an already sealed test directory.

## Deferred strict common probe

The independent
[Frozen-LExpert benchmark paper](https://arxiv.org/html/2608.05960v1) and its
[pinned reference implementation](https://github.com/maulikchevli/frozen-lexpert/tree/4c9734c9d839e03523bc339c2d5934fd53e1942c)
define a different common frozen-feature evaluator. For COLIPRI it freezes the
released model end to end, uses its pretrained projected/attention-pooled,
L2-normalized 768-dimensional image embedding, and fits per-label one-vs-rest
L2 logistic regression with `C=1`, balanced class weights, LBFGS, and
`max_iter=2000`, without the native trainable pooling grid.

That strict classifier-only common probe is intentionally deferred. If added,
it must use a separate cache and result identity and explicitly adapt the
benchmark's multi-label OvR output to this mutually exclusive five-class task.
It must not be mixed with the current native 20-candidate results.

No COLIPRI-specific LoRA or ConvLoRA recipe or full-encoder classification
fine-tuning recipe is introduced. The separately named Primus-M partial
fine-tuning and zero-shot methods below are additive and do not alter this
pooling-optimised probe.

## Primus-M partial fine-tuning and zero-shot

`PrimusMFinalBlockFineTune` and `PrimusMFinalStageFineTune` use the same pinned
source revision and exact `model.safetensors` SHA-256 as the existing workflow,
but have independent method IDs, profiles, CV registries, checkpoints, and
result roots. The official Primus-M hierarchy is validated as 16 flat EVA
blocks. The block strategy updates block 15 plus the final norm; because
Primus-M publishes no native stage grouping, the stage strategy is explicitly
defined as the final quarter, blocks 12--15 plus the final norm. The released
768-D projector and multi-head attention pooler stay frozen, followed by a new
`LayerNorm -> Linear(5)` classifier.

Each fixed publication profile keeps the reviewed `1e-4` encoder-adaptation
anchor and the short 37,500-step downstream protocol. Scope restriction,
five-class head, class weighting, validation, and early stopping are marked
publication-adapted inside the same `parameter_source: publication` profile.
Publication mode bypasses grid selection and goes directly to CV/refit/test.
Each strategy also has an independent three-candidate manual project grid,
used only with `parameter_source: grid`, and a fixed practical default; all
`#` source keys are inert.
Both scopes default to `-InputSource frozen` (Python: `--input_source frozen`).
They create temporary pre-block tokens from the prepared CTs, then train from
those tokens. `-InputSource cache` directly loads prepared CTs instead.
Pass the same `-CacheRoot` used for preparation.
A missing/stale cache fails with a prepare/validate instruction. Run identity
binds the cache manifest/contract and ordered shard paths, sizes, mtimes and
SHA-256s alongside the patient split and model/runtime identity.

Use `-InputSource nifti` for an existing original-NIfTI checkpoint. Switching
input mode changes training identity and cannot resume that checkpoint. Start
cached-input experiments in a fresh `-ResultsRoot` if an active legacy-resume
record blocks that configuration, passing the original `-CacheRoot` explicitly
to share CTs. Retiring a backed-up legacy run is another option, described below.
The original NIfTI
adapter, shared trainer, and probe cache/model implementations remain unchanged.

On 2026-09-11 the user chose to restart the final-block 100% experiment and had
already moved the old run into
`Results/COLIPRI/PrimusMFinalBlockFineTune/Backup_CV_1/`. Its remaining method-level
`FIXED_PUBLICATION_MIGRATION.json` was moved unchanged into that backup. The
active registry already contained only `CV_2`; neither it nor any training code
was changed. The current frozen configuration resolves to `CV_2`, shared by the
80%, 60% and 40% runs. A fresh 100% run can therefore use the normal results root:

```powershell
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_block -Percentage 100 -InputSource frozen -NoResume
```

This starts from the pinned pretrained weights in `CV/CV_2/100_percent`, with
separate 100% folds/refit/test outputs. `-NoResume` refuses existing output for
that percentage; it does not delete it or force a restart. Omit `-NoResume` when
later resuming this new run. `cache` is a different input configuration and does
not identify this frozen `CV_2`. The backup is outside active CV discovery;
simply selecting `nifti` no longer resumes that archived run. Existing runtime
logs and checkpoints were preserved. Validation is recorded in
`Logs/COLIPRI/VALIDATION_20260911_FRESH_100_PERCENT.json`.

### Temporary frozen outputs and prefetching

`frozen_forward.py` splits the pinned released forward immediately before
block 12 (final stage) or block 15 (final block). The selected blocks, final
normalization, released projector/pooler and five-class head still run at each
training step. The preceding layers stay in evaluation mode. There is currently
no random image augmentation; selected trainable blocks retain their existing
stochastic-depth behaviour and saved CUDA RNG state.

`-AdaptationScope both` (Python: `--adaptation_scope both`) prepares both
boundaries in one traversal per required patient, then runs final-stage and
final-block workflows with their separate profiles/results. Only the selected
scope's tensor is read during each method's training. A single-scope launch
stores just its own boundary. Cache construction includes one extra full
forward for the first patient to verify the split against released pooling.
Verification uses the contiguous CPU-shard/stacked-input layout and keeps strict
`rtol=1e-5, atol=1e-6` checks. Failures log the measured error and output dtype.
The split preserves the released `einops.rearrange` operations and singleton
batch strides. A 2026-09-10 fix corrected a value-equivalent reshape that changed
the projector/pooler memory layout and failed full-size BF16 verification.
This failure occurred before writing shards or starting training; retry the same
launch after the fix. Permanent caches and original-NIfTI resumes are unaffected.

Temporary shards live under `Cache/FrozenRuns/<owner-and-contract>/`. The
contract binds permanent CT identity, pretrained weights, installed encoder
code, precision, runtime/math settings and extraction batch size 1. Tensor
values are stored without conversion; mixed-precision residual outputs may
still be FP32. An FP32 boundary is 45.56 MiB per patient, about 60.42 GiB for
1,358 patients; both boundaries would be about 120.85 GiB. Only patients needed
by the requested stages/percentage are generated. Disk preflight uses the
FP32 upper estimate plus 8% and 2 GiB. This is additional temporary storage,
on top of the two permanent caches.

Progress is atomic and sealed. A resumed process verifies existing tensor
hashes, source bindings, shapes and dtypes. Training checkpoints also bind the
ordered training/validation tensor-value digests. If deleted inputs are
regenerated with different values, checkpoint resume fails rather than silently
continuing with different features. An exclusive OS-backed file lock
covers the entire launch; another launch needing the same cache waits.
The lock dependency is declared explicitly as `filelock==3.32.3` in the
COLIPRI environment requirements.
After every requested fold/candidate/refit/test/scope succeeds, cleanup removes
only the owned manifest and declared temporary shards. Interrupted/failed
launches retain them for resume; unexpected files or links prevent deletion.
The permanent `Features/` and `Volumes/` caches and all scientific outputs stay.
Small lease files and `Setup/frozen_cache_events.jsonl` remain as bookkeeping.
The event log records preparation, reuse, verification, retention and deletion.

Each model identity binds only its selected boundary, so later single-scope
evaluation can regenerate inputs after a combined launch's cleanup. Original
NIfTI checkpoints require `-InputSource nifti`; prepared-CT checkpoints require
`-InputSource cache`. Keep the saved precision and microbatch. An active legacy
compatibility record can require a fresh result root with the same explicit
permanent `-CacheRoot`; the retired 100% final-block record described above is
now stored with its backup instead.

One-microbatch CPU prefetch is enabled by default for the 20-candidate probe,
both fine-tuning scopes, refits and evaluation, including when microbatch equals
effective batch. `-NoPrefetch` / `--no-prefetch` disables it independently of
input mode. The waiting batch stays in system RAM, without allocating an extra
GPU batch. Memory includes one upcoming microbatch plus temporary loading
buffers; no full-dataset RAM cache is used. Prefetch neither skips encoder work
in CT/NIfTI mode nor changes candidates, batch sizes or validation schedules.

The encoder split is verified on small synthetic CTs using the actual installed
Primus/EVA, rotary embeddings, projector and attention pooler: logits, trainable
gradients and an optimizer update agree at matching batch shapes in CPU FP32,
CUDA FP32 and CUDA BF16. Per-patient GPU extraction can select different kernels
from batched original-input training and introduce rounding differences,
especially in BF16. Frozen mode therefore does not claim bit-identical
retraining or unchanged cohort metrics. Prefetch tests preserve identical
trained tensors across uninterrupted and resumed runs, including all 20 probe
candidates. No production cohort speedup or accuracy comparison has been run.

For the full-size local-checkpoint regression (one synthetic 192-cube in each
of CUDA BF16 and FP32, both boundaries, real temporary shard I/O and cleanup),
set `COLIPRI_TEST_RELEASED_FROZEN=1` when running pytest. It is opt-in because it
requires CUDA and the pinned checkpoint already available locally. The layout
fix passed 152 COLIPRI/shared tests with this check enabled; bounded diagnostics
also matched the released BF16 pooled outputs exactly for three existing CTs
from the 80% development split. These checks did not start training or measure
cohort performance.

For both partial-fine-tuning scopes, omitting `-MicroBatchSize` (Python:
`--micro_batch_size`) uses the selected profile's effective batch size: **8**
for the packaged publication/grid profiles and **4** for the packaged practical
default profiles. Custom profiles determine their own default. An explicit
microbatch must be positive, no larger than the effective batch, and divide it
evenly; errors name the actual batch limit. Custom grids with different batch
sizes require an explicit microbatch that is valid for every candidate.

```powershell
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_block -DryRun
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_stage -Stage all -Percentage 40 `
  -ParameterSource publication
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope final_block -Stage all -Percentage 40 `
  -ParameterSource grid
# Both methods share the temporary extraction; cleanup follows both successes.
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 `
  -AdaptationScope both -CacheRoot E:\COLIPRI_Cache -DryRun
# Keep prepared-CT input but use prefetching.
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 -InputSource cache -DryRun
# Disable prefetch independently; frozen outputs remain the default.
.\COLIPRI_PrimusM_Partial_FineTune_Run_Script.ps1 -NoPrefetch -DryRun
```

`ZeroShot` loads the complete released image-text model, not the image-only
cache path. It uses the paper's short positive/negative prompt pairs, cosine
similarity divided by `0.07`, and per-class pairwise softmax. Normalizing the
five positive evidences into one mutually exclusive distribution is recorded
as a task adaptation and is uncalibrated. Prompts, checkpoint/source identity,
ordered 278-patient held-out identity, and per-patient progress are sealed;
the concrete held-out NIfTI inventory is bound before scoring, and held-out
labels never tune prompts or temperature.

```powershell
.\COLIPRI_Zero_Shot_Run_Script.ps1 -DryRun
.\COLIPRI_Zero_Shot_Run_Script.ps1 -Checkpoint D:\Models\model.safetensors
```

Partial-FT outputs use
`Results/COLIPRI/PrimusMFinalBlockFineTune/{Grid|CV|Test}` and
`Results/COLIPRI/PrimusMFinalStageFineTune/{Grid|CV|Test}`. Fixed publication
and default runs have no grid directory;
zero-shot outputs use
`Results/COLIPRI/ZeroShot/Test/colipri_publication_short_pairwise_five_class_v1`.
Neither path overlaps `PoolingOptimisedLinearProbe` or the reserved strict
`LinearProbe` namespace.

## Reproducibility and validation boundary

The public source/checkpoint audit materialized the pinned
`model.safetensors` and verified its exact size and SHA-256. That was asset
provisioning for integration; it was not model construction or inference.
Synthetic tests and dry-run plans do not establish production performance.

No patient volume was preprocessed or encoded, no feature cache was built, and
no classifier, grid, CV, refit, inference, or held-out evaluation was run while
adding this package. A real `-LoadModel` setup preflight, complete cache build,
exhaustive validation, and then the documented experiment sequence remain
required before performance results can be claimed.

Run the focused synthetic suite after installing the isolated environment:

```powershell
.\.venv_colipri\Scripts\python.exe -m pytest Code\Models\COLIPRI\tests
```

See the [checkpoint catalogue](../../../provenance/checkpoints.json) and
[reproduction boundary](../../../REPRODUCING.md) for the published artifacts.
Internal implementation timelines and audit records are not distributed.

