> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# 3DINO five-class CT classification

This is an isolated integration of the official 3DINO-ViT encoder into the
repository's patient-level five-class methodology. Python source lives under
`Code/Models/ThreeDINO`; generated artifacts are confined to `Results/3DINO`.
No Merlin or Pillar-0 module, cache, checkpoint, registry, or result is used as
a 3DINO write target.

## Method boundary

The current selected methods are linear probing and LoRA. See the
[LoRA implementation and run guide](LORA.md) for the related study's exact
defaults and our five-class adaptations. Full fine-tuning remains an older,
disabled experimental path outside the current selected protocol.

| Method | Encoder | Evidence status | Default availability |
|---|---|---|---|
| Linear probe | Frozen and kept in evaluation mode | Publication-backed classification method, adapted to this repository's five labels and folds | Primary |
| LoRA | Original weights frozen; rank-4 adapters in all 24 attention blocks plus MLP classifier trainable | Related-study code adapted to five-class CT | `train_lora.py`; default/manual profiles |
| Full fine-tune | All encoder weights trainable | Conditional project experiment; not demonstrated for classification in the 3DINO paper | Disabled until explicit authorization |
| Zero-shot | Not applicable | The released 3DINO model has no aligned text encoder | Not implemented |

The publication-backed claim applies to the adaptation family and reported
linear-probe search, not to this dataset. The paper did not evaluate
`Appendicitis`, `Cholezystitis`, `Normal`, `Perforation`, and `Sigmoid`, nor did
it use these patient manifests or the repository's selection score.

## Upstream access and licence preflight

The official source and gated `AICONSlab/3DINO-ViT` checkpoint are released
under CC BY-NC-ND 4.0 terms with additional upstream explanations of
non-commercial use. This repository does not claim that access has been
granted or that a planned use is permitted. Confirm the current terms and the
project's eligibility before using either asset.

Standard model-bearing commands resolve two independently pinned assets:

- official source `https://github.com/AICONSlab/3DINO` at commit
  `85bd4435c1b2ada41cd34cd15cad17c4d3c88d89`;
- `3dino_vit_weights.pth` from `AICONSlab/3DINO-ViT` at Hugging Face commit
  `8a00a2bb14becb3fbe955064837322a3217f7ac0`.

The expected checkpoint is `1,417,113,974` bytes with Hub ETag/LFS content
SHA-256
`b68dde2e435648bdf72edae70f970b07e4a903a8122c5be9e360bdb076cf695c`.
The source and weight revisions are separate identities and must not be
substituted for one another.

Before the first model-bearing run, open the
[`AICONSlab/3DINO-ViT`](https://huggingface.co/AICONSlab/3DINO-ViT) page using
the intended Hugging Face account, accept its access conditions, and authenticate
the machine if necessary:

```powershell
hf auth login
```

The normal resolver is local-first and missing-only. It reuses a valid pinned
source checkout and a matching local/Hugging Face cache entry without network
work. If the default source directory is absent, it clones the official
repository and checks out the pinned source commit. If no local checkpoint
override is supplied, `huggingface_hub` resolves the pinned file, downloading it
only when it is absent from the local Hub cache. An existing partial, modified,
wrong-revision, empty, wrong-size, or wrong-hash asset fails closed; automation
does not overwrite or silently repair an ambiguous existing directory/file.

`--upstream_root` and `--pretrained_checkpoint` remain optional local overrides.
An explicitly supplied checkpoint path is strict: if it is missing or invalid,
the resolver fails instead of falling back to a different Hub-cached file.
`--local_files_only` (launcher `-LocalFilesOnly`) disables cloning and network
access and therefore requires both assets to be available locally. Repository,
revision, filename, acquisition action, path, size, and SHA-256 are recorded in
the token-safe append-only
`Results/3DINO/Setup/asset_provisioning.jsonl` by default. The normal Hugging
Face credential store is used; access-token values and credential environment
variables are never written to that log, setup reports, run logs, or model
provenance.

After resolution, the loader verifies source commit/layout/import origin and
the runtime ViT-L architecture. It requires every encoder tensor and shape
except an explicitly permitted inference-unused `mask_token`. Missing
transformer weights are fatal rather than silently left random.

### Supported Windows compatibility runtime

The [official 3DINO repository](https://github.com/AICONSlab/3DINO) states
Python 3.9. This Windows integration deliberately supports one different,
explicit **compatibility** target: **64-bit CPython 3.10.x on Windows**. It is
not described as an exact reproduction of the upstream Python environment.
CPython 3.10 was selected because the official-era binary set supports it,
including the published `xformers==0.0.18` `cp310-win_amd64` wheel; that wheel
also declares `torch==2.0.0` ([PyPI release metadata](https://pypi.org/pypi/xformers/0.0.18/json)).

From the `Systematic_Methodology` directory, create the dedicated interpreter
explicitly with the Windows Python launcher:

```powershell
py -3.10 -m venv .venv_3dino
.\.venv_3dino\Scripts\python.exe -m pip install -r Code\Models\ThreeDINO\requirements.txt
```

`check_setup.py` enforces this fail-closed policy before either readiness flag
can be true:

| Runtime component | Required policy |
|---|---|
| OS/interpreter | Windows `win32`, 64-bit CPython 3.10.x |
| Binary-era core | `torch==2.0.0`, `torchvision==0.15.0`, `xformers==0.0.18`, `torchmetrics==0.10.3`, `monai==1.3.0`, `nibabel==5.1.0` |
| Native loading | `torch._C` and the xFormers C++/CUDA extension must load successfully; finding a package without importing its binaries is insufficient |
| Required ranges | The remaining source, preprocessing, and evaluation dependencies must import and satisfy the ranges in `requirements.txt` |
| CUDA request | When `--device` begins with `cuda`, `torch.cuda.is_available()` must also be true |

Local wheel suffixes such as `torch==2.0.0+cu117` are accepted when the public
version matches the exact pin. A CPU preflight does not claim CUDA readiness.
Select a CUDA/driver configuration compatible with the intended workstation
before requesting a CUDA device. Dependency installation itself does not grant
checkpoint access or licence permission.

## Setup validation and asset provisioning

Inspect the interpreter, package versions, native imports, all NIfTI paths,
split safety, source compatibility, checkpoint identity, and required sealed
cache manifest without training:

```powershell
.\.venv_3dino\Scripts\python.exe Code\Models\ThreeDINO\check_setup.py
```

This command may perform the same missing-only source clone/checkpoint download
as a model-bearing run. Add `--local_files_only` for a strictly offline check,
or supply `--upstream_root` and `--checkpoint_path` to validate explicit local
assets. `--load_model` is optional and performs strict model construction; it
is not a training or performance run. The separate `smoke_inference.py` is also
setup evidence only and must not be reported as a classification result. If the
runtime policy fails, source/checkpoint model adapter imports are skipped and
both `ready_for_cache_build` and `ready_for_model_execution` remain false. A
missing or invalid requested cache also keeps model execution false.
`--no_auto_provision` deliberately skips Git/SCM attestation and is therefore a
structural diagnostic only; it cannot declare the setup ready for model
execution or load source code.
`--skip_artifact_validation` is a faster structural diagnostic only: it does
not read, hash, or deserialize cache artifacts, explicitly leaves them
unverified, and can never set `ready_for_model_execution` to true.

## NIfTI preprocessing and cache

The model accepts only one full 3-D NIfTI per patient. DICOM series and PNG or
2.5-D images are not substituted. The transform is split deliberately:

1. deterministic, label-independent prefix: load as one channel, orient RAS,
   map the 0.05th and 99.95th intensity percentiles to `[-1, 1]`, crop
   foreground, and resize to `144 x 144 x 112`;
2. training post-cache transform: random `112^3` crop, one randomly selected
   contrast/sharpen/smooth/noise transform, independent axis flips, intensity
   scale, and intensity shift;
3. validation/test post-cache transform: center `112^3` crop only.

Only step 1 is persisted. Random crops and augmentations vary on every training
draw, so the cache cannot silently turn publication-style training into one
fixed feature per patient. Each draw is nevertheless reproducible. Its
augmentation seed is a keyed permutation of the absolute draw position; the key
binds the sealed run seed, complete training-row fingerprint, and the
transform-contract fingerprint. Separately, the deterministic sampler maps
that draw to a sample through its seeded epoch permutation. Neither mapping
depends on worker assignment or prefetch scheduling.

Plan the cache without writing tensors:

```powershell
.\3DINO_Cache_Run_Script.ps1 -Action plan -Percentage 100
```

Exercise the prepare route without writes:

```powershell
.\3DINO_Cache_Run_Script.ps1 -Action prepare -Percentage 100 -DryRun
```

An intentional production cache build and later exhaustive validation are:

```powershell
.\3DINO_Cache_Run_Script.ps1 -Action prepare -Percentage 100
.\3DINO_Cache_Run_Script.ps1 -Action validate
```

Those last two commands are documented for later use; they were not run while
adding this implementation. A namespace is sealed to the split-manifest hash,
source paths, source SHA-256 values and metadata, and preprocessing contract.
The completed v2 manifest additionally seals every cached `.pt` file's byte
size and SHA-256. Exhaustive validation rehashes and deserializes every file,
checks its finite floating-point `[1, 144, 144, 112]` tensor and patient/class
identity, and rejects undeclared cache artifacts. Incompatible, incomplete, or
orphaned reuse fails closed. Use `-CacheRoot` and `-CacheNamespace` to create an
explicitly separate cache location/version.

## Publication-backed linear probing

The encoder is frozen, run under `no_grad`, and kept in evaluation mode. The
official ViT-L representation width is 1024. One encoder pass yields all four
publication recipes:

| Recipe | Components | Feature width |
|---|---|---:|
| last 1, CLS only | final CLS token | 1024 |
| last 1 + mean patch | final CLS plus final-layer mean patch token | 2048 |
| last 4, CLS only | concatenated CLS tokens from four final blocks | 4096 |
| last 4 + mean patch | four CLS tokens plus final-layer mean patch token | 5120 |

`publication_linear_probe.json` forms the Cartesian product of these recipes
and the paper's 13 reported learning rates:

```text
0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01,
0.02, 0.05, 0.1, 0.2, 0.3, 0.5
```

All 52 independent heads train from the same stochastic image batch and shared
encoder pass. The fixed recipe follows the paper: SGD, momentum 0.9, zero
weight decay, cosine annealing to zero, cross-entropy, batch size 32, and
12,500 optimizer iterations. Publication mode also follows the pinned evaluator
with an infinite shuffled sampler, full batches only (`drop_last`), and enabled
CUDA autocast/fp16. The infinite stream can cross permutation boundaries to
form every complete batch, matching the fixed-iteration sampler contract.

The grid therefore optimizes 52 different linear heads and selects one
hyperparameter configuration; it does not train 52 encoders. Each candidate is
the combination of one learning rate, one layer-count choice, and one pooling
choice. The candidate-facing results use the same five-fold
`candidate_<id>/fold_01...fold_05` organization as the Merlin grids, while the
implementation retains the official efficiency of one frozen-encoder forward
pass per image batch. The publication grid is a single search stage, so it has
no visible `round_01` directory.

Every head initialization is derived from the run seed and that candidate's
stable ID through the recorded
`threedino-candidate-sha256-seed-v1` contract. Consequently, the same candidate
starts from the same weights whether it appears alone, in a differently ordered
bank, or beside more candidates; initializing one head also leaves the caller's
global RNG stream unchanged. `grid_context.json` binds the ordered candidate
IDs, complete candidate-set fingerprint, candidate count, initialization schema,
grid-profile hash, split/cache/model execution context, and seed-derived
execution identity. A changed candidate set therefore cannot silently resume a
shared fold or reuse a completed grid.

There is a documented paper/code learning-rate ambiguity. The paper presents
the 13 values as the searched learning rates; official `linear3d.py` multiplies
each supplied value by `(batch_size * world_size) / 256` before creating its
optimizer group. The publication profile records the printed value as
`nominal_lr`, follows this official scaling rule, and records the resulting
effective `lr`. At batch 32 on one process, the effective value is one eighth
of nominal. The pinned code's parser defaults use a different, narrower LR
list; this profile intentionally prioritizes the 13 values printed in the
paper, while retaining the code's scaling behavior. Results must retain both
values, their source, and the rule.

Validate path, parameter, and output planning without loading a model:

```powershell
.\3DINO_Linear_Probe_Run_Script.ps1 `
  -Stage all -Percentage 100 -ParameterSource publication -DryRun
```

The corresponding future model-bearing command is the same without `-DryRun`.
That first real command provisions missing pinned assets automatically and
subsequent commands reuse them. `-LocalFilesOnly` refuses any network or clone
operation. `-ModelRepoId`, `-ModelRevision`, `-UpstreamRevision`,
`-UpstreamRoot`, `-PretrainedCheckpoint`, and
`-PretrainedCheckpointSha256` expose explicit provenance/local overrides;
`-ProvisionLog` overrides the default token-safe JSONL audit destination.
`-DryRun` remains side-effect-free: it does not clone, download, authenticate,
or write the provisioning log.
`-NoResume` disables the launcher's default resumption behavior. Search/CV
selection uses development folds only; held-out patients are not loaded for
head selection or early stopping.

The primary parameter-source interface matches Merlin and Pillar-0:

| `-ParameterSource` | Meaning | Linear-probe output |
|---|---|---|
| `publication` | Complete, untruncated 52-candidate publication-backed grid | `LinearProbe/Publication_Grid/<percentage>_percent/` |
| `grid` | Project-designed grid, optionally supplied with `-GridConfig` | `LinearProbe/Grid/<percentage>_percent/` |
| `default` | One installed practical configuration; no grid | Matching stable `LinearProbe/CV/CV_x/` |
| `manual` | One configuration supplied with `-ParameterConfig`; no grid | Matching stable `LinearProbe/CV/CV_x/` |

`-GridConfig` and `-MaxGridCandidates` are the primary grid controls. Candidate
truncation and a replacement grid file are rejected for `publication`, because
either would invalidate the claim that the complete publication search space
was preserved. They are available for the project `grid` source. The older
`search`, `-SearchConfig`, and `-MaxSearchCandidates` spellings are accepted
only as compatibility aliases; new commands and records use `grid` terminology.
Generated evidence fields such as `search_type`, `profile_type`, and the
publication-preservation flags are not user-overridable arguments.
`-Stage grid -ParameterSource manual` is rejected: `manual` supplies one fixed
configuration for CV/test, whereas a grid stage requires `grid` or
`publication`.

A publication run records the following methodological classification:

```json
{
  "parameter_source": "publication",
  "search_type": "publication_grid",
  "profile_type": "publication_adapted",
  "publication_search_space_preserved": true,
  "publication_fixed_recipe_preserved": true,
  "publication_exact": false
}
```

`publication_exact` is false because the publication search and fixed training
recipe are applied to this repository's five-class dataset, patient folds, and
imbalance-aware fold-aggregation score. The user selects `publication`; the
remaining fields are derived and validated by the workflow.

Across the five development folds, candidates use the same systematic ranking
formula as the Merlin grids:

```text
0.7 * (mean macro-F1 - standard deviation of macro-F1)
+ 0.3 * (mean balanced accuracy - standard deviation of balanced accuracy)
```

This replaces the official evaluator's single validation-split top-1 selection
for the repository's imbalanced five-class, five-fold protocol; it does not
change the publication-defined 52-candidate space or fixed head-training recipe.

`-NumWorkers` controls CPU workers for the training loader and remains `0` by
default. A user may opt into persistent workers, for example:

```powershell
.\3DINO_Linear_Probe_Run_Script.ps1 -NumWorkers 10
```

Validation and held-out-test loaders are intentionally fixed at zero workers,
so frequent validation does not repeatedly create Windows worker processes.

The launchers also accept `-ResultsRoot`. Leaving it unset preserves the normal
methodology root, `Results/3DINO`. The draw-indexed checkpoint format has an
internal schema number for compatibility checking; it does not create or
require an `ExactResumeV2` result directory. After archiving or removing a
legacy `LinearProbe` result subtree, start a fresh run in the ordinary location
while retaining the reusable sealed `Cache` and provisioning `Setup` records:

```powershell
.\3DINO_Linear_Probe_Run_Script.ps1 `
  -Stage all -Percentage 100 -ParameterSource publication -NumWorkers 10
```

The same `-ResultsRoot` and `-CacheRoot` options are available on the full
fine-tune launcher for deliberate alternate output/cache locations. A
worker-count change is made only by stopping at a committed checkpoint and
restarting the command with the new `-NumWorkers` value.

## Conditional project-adapted full fine-tuning

The 3DINO paper freezes its encoder for classification. Therefore
`default_full_finetune.json` and `search_full_finetune.json` are explicitly
labelled `project_adapted_conditional`; there is no
`publication_full_finetune.json`.

Updating the released checkpoint may constitute an adaptation or derivative
under the upstream no-derivatives terms. The code gate is a workflow safeguard,
not legal permission. It may be enabled only after documented permission or
another confirmed legal basis, non-commercial eligibility, and a successful
full forward/backward memory preflight on the intended hardware.

Planning remains available without enabling weight updates:

```powershell
.\3DINO_Full_FineTune_Run_Script.ps1 `
  -Stage all -Percentage 100 -ParameterSource default -DryRun
```

The bundled full-fine-tune profiles deliberately retain
`license_approval_claimed: false` and `memory_preflight_passed: false`; the CLI
does not convert those declarations into approval. If and only if the external
requirements above are satisfied, make a reviewed manual copy of the profile,
set both authorization fields to `true`, and pass that file together with the
independent CLI acknowledgement:

```powershell
.\3DINO_Full_FineTune_Run_Script.ps1 `
  -Stage all -Percentage 100 -ParameterSource manual `
  -ParameterConfig path\to\reviewed_full_finetune.json `
  -AllowExperimentalFullFineTune
```

The reviewed JSON must retain a top-level `authorization` object with both
claims set to literal JSON `true`. The switch alone is insufficient. Neither
command makes the method publication-backed. No full-fine-tuning model or
metric was generated while adding this implementation.

## Profiles

| File | Status | Purpose |
|---|---|---|
| `publication_linear_probe.json` | Publication-adapted search | Complete 52-head search with paper values and explicit official-code LR scaling reconciliation |
| `default_linear_probe.json` | Practical project default | One resource-conscious frozen-head configuration |
| `search_linear_probe.json` | Practical project grid | Six coupled candidates rather than the full 52-head publication grid |
| `default_full_finetune.json` | Conditional project adaptation | Disabled-by-default end-to-end engineering profile |
| `search_full_finetune.json` | Conditional project search | Small successive-halving feasibility search, also disabled by default |

`--parameter_source publication` is valid only for linear probing.
`--parameter_source manual` requires `--parameter_config`. `grid` is the
primary project-search name; `search` remains a legacy alias. Exact effective
parameters, source/profile type, candidate ID, checkpoint/source provenance,
split hash, and class order are retained with the resulting CV identity.

## Systematic workflow and outputs

Grid-backed linear probing preserves this repository sequence:

```text
development-only five-fold candidate grid
    -> select one parameter dictionary
    -> stable CV_x registration
    -> verify and reuse the selected candidate's five grid folds in CV_x
    -> full-development refit
    -> one fixed held-out test
```

The selected grid folds are not trained a second time. Their parameters,
metrics, checkpoints, source paths, hashes, and model identity are verified,
then published into the stable `CV_x` view using a hard link where supported or
a verified copy otherwise. This reuse is allowed only from a complete,
hash-consistent `grid_completion.json`; partial or altered grid output fails
closed. Each reused CV fold writes `grid_reuse.json` with its source/destination,
candidate ID, artifact hashes, materialization mode, result origin, and
`reused_without_training: true`. Publication selections use
`selected_publication_grid_run` as their result origin; project-grid selections
use `selected_grid_run`. `default` and `manual` have no preceding grid and
therefore train their five CV folds normally. A different exact parameter
dictionary receives another stable `CV_x` identity.

Selected-fold publication is crash-repairable. If an interruption leaves a
complete or partial hash-matching artifact set without its final
`grid_reuse.json`, resume idempotently materializes only the missing pieces and
writes the marker without training. Hard links remain preferred. The copy
fallback writes a same-directory temporary file, verifies its SHA-256, and uses
atomic replacement; it never exposes or overwrites different destination data.
Unexpected files in a markerless destination fail closed.

With the launcher's default resume enabled, rerunning a matching completed grid
or completed CV is a validated no-op: the grid/CV completion seal and bound
artifacts are rechecked, the existing selection/result is returned, and no
training, refit, or artifact rewrite occurs. Completed-CV validation rehashes
and validates the actual bytes of every fold and full-development refit
checkpoint, not only their metadata. `-NoResume` instead rejects completed or
partial grid and CV destinations; it does not overwrite them.

Freshly completed and resumed grids expose the same selection provenance,
including `selection_file_sha256` for the exact `selected_parameters.json`.
CV registration therefore binds the same selected file identity regardless of
whether the grid finished in the current process or was reused from its seal.

Training checkpoints use exact-or-fail-closed draw-indexed resumption. The
launcher default remains `--num_workers 0`; when a positive value is requested,
only the training loader uses persistent workers. Validation and test loaders
remain fixed at zero workers. The committed absolute draw position—not the
prefetch/request cursor—is saved atomically with model/head,
optimizer/scheduler/scaler, sampler, transform-contract, and runtime state.
Prefetched but unconsumed draws are discarded after interruption and recreated
from their stable draw identities, including the same patient order, crop, and
augmentation.

Worker count and prefetch scheduling are operational provenance, not model,
candidate, search, or CV identity. The training-worker count may therefore be
changed when restarting from a compatible draw-indexed checkpoint; it cannot
be changed inside a running process. Batch size, data/cache/split identity, run
seed, sampler and transform contracts, parameters, scheduler horizon, and model
identity remain immutable and fail closed if changed. Injected DataLoaders are
refused for exact resumption; callers must supply a dataset so the internal
loader owns and validates the draw-identity contract.

Legacy checkpoints were created from evolving process/worker RNG and a cursor
that becomes prefetch-sensitive with multiple workers. They cannot be proven
equivalent to the draw-indexed sequence and are deliberately rejected. The
compatibility number remains internal checkpoint metadata, not a user-facing
experiment name or folder. Archive/remove the legacy active method subtree
before a fresh normal-root run; do not mix incompatible schemas in one search
directory. CUDA state and device identity remain bound, while numerical bitwise
equality also depends on deterministic kernels in the pinned PyTorch/CUDA
stack.

Cross-validation becomes eligible for internal held-out evaluation only after
a final `cv_completion.json` is written. That seal binds all five fold models
and the full-development refit by checkpoint SHA-256, training-state SHA-256,
fold/refit identity, effective parameters, training-row fingerprint, runtime
plan, class order, and base-model identity. When folds came from a selected
grid, the seal additionally binds all five `grid_reuse.json` files by SHA-256,
one selected candidate, one allowed result origin, and one source grid root.
Reuse markers in a CV that does not claim grid reuse are rejected rather than
ignored. Internal testing revalidates the seal, reuse markers, and all fold/refit
checkpoint bytes before it reads held-out rows; a missing, changed, or duplicate
ensemble checkpoint fails closed. Explicitly supplied external checkpoints
remain allowed but are labelled unverified.

The canonical output boundary is:

```text
Results/3DINO/
|-- Cache/Shared/<cache_namespace>/
|-- Setup/asset_provisioning.jsonl
|-- LinearProbe/
|   |-- Publication_Grid/<percentage>_percent/
|   |   |-- shared_training/{fold_01...fold_05}/
|   |   |-- candidate_<id>/{fold_01...fold_05,candidate_summary.json}/
|   |   |-- grid_context.json
|   |   |-- grid_index.json
|   |   |-- grid_summary.json
|   |   |-- grid_summary_by_selection_score.csv
|   |   |-- best_hyperparameters.json
|   |   |-- selected_parameters.json
|   |   `-- grid_completion.json
|   |-- Grid/<percentage>_percent/
|   |   |-- shared_training/{fold_01...fold_05}/
|   |   |-- candidate_<id>/{fold_01...fold_05,candidate_summary.json}/
|   |   |-- grid_context.json
|   |   |-- grid_index.json
|   |   |-- grid_summary.json
|   |   |-- grid_summary_by_selection_score.csv
|   |   |-- best_hyperparameters.json
|   |   |-- selected_parameters.json
|   |   `-- grid_completion.json
|   |-- CV/CV_x/<percentage>_percent/{fold_01...fold_05,refit_full_development}/
|   `-- Test/CV_x/<model_source>/<percentage>_percent/
`-- FullFineTune/
    |-- Grid/<percentage>_percent/
    |-- CV/CV_x/<percentage>_percent/{fold_01...fold_05,refit_full_development}/
    `-- Test/CV_x/<model_source>/<percentage>_percent/
```

`Publication_Grid` and `Grid` are sibling search families. Neither adds a
visible `round_01` for a single-round grid. The frozen 1.42 GB encoder is bound
by provenance and is not copied into every candidate directory; candidate
checkpoints contain the independently optimized linear-head state.
`shared_training/fold_x` holds the resumable all-head optimizer/scheduler/data
stream state for the one joint fold execution. Each candidate's `fold_x`
contains `parameters.json`, `best_model.pt`, `metrics_summary.json`, validation
predictions/probabilities, candidate-bound `training_state.json`, and
`shared_training_reference.json`, keeping the scientific result candidate-first
without duplicating the joint optimizer/data-stream state.

Completed held-out destinations fail closed. A matching result may be reused
with resume; intentional replacement additionally requires `--force_test` (or
launcher `-ForceTest`). The expected patient-level outputs use the established
names: `metrics_summary.json`, `model_selection.json`,
`test_patient_predictions.json`, and `test_probabilities.csv`. A result is
reusable only when a final `test_completion.json` seals hashes for all four
artifacts and matches the current split/cache/seed/model/checkpoint provenance.
Internal refit, best-fold, and ensemble checkpoints must also match the
registered effective parameters, class order, method, base-checkpoint SHA, and
derived refit budget before held-out rows are touched. A
`provided_checkpoint` remains explicitly external: its SHA is recorded, but
its parameters, selection history, and publication evidence are labelled
unverified rather than inferred from the requested CV context.

Metrics follow the repository schema: accuracy, balanced accuracy, macro F1,
macro average precision, class-wise precision/recall/sensitivity/specificity/F1/AP,
and the 5 x 5 confusion matrix in the exact class order. The companion notebook
is `Notebooks/3DINO/3DINO_Test_Results_Summary.ipynb`.

## Validation scope

Synthetic tests exercise split leakage rejection, NIfTI-only rows, deterministic
cache-plan identity, cache-byte and payload tamper rejection, namespace
isolation, the four representation dimensions, single-pass feature extraction,
frozen-gradient behavior, full-fine-tune authorization, metric schemas,
52-head expansion, LR scaling, stable candidate ranking, canonical
publication/custom grid locations, candidate-first artifact materialization,
single-round `round_*` rejection, candidate-set/context binding, candidate-ID
and seed-based head initialization independent of bank order/size, completed
grid/CV no-op resume, markerless full/partial publication repair, atomic
copy-fallback materialization, selected-fold CV reuse/tamper rejection, final-CV
binding of all reuse markers/origin/source root and checkpoint bytes, unclaimed
reuse-marker rejection, fresh/resumed selection-file hash parity, invalid
grid/manual routing, mocked local/Hub asset resolution, cross-worker
draw-sequence/resumption equivalence, legacy-schema fail-closed compatibility,
fixed zero-worker validation/test loading, and CV-checkpoint seal validation
without external weights or source code:

```powershell
.\.venv_3dino\Scripts\python.exe -m pytest Code\Models\ThreeDINO\tests
```

The final crash-hardened non-production validation passed `136` tests with
`144` dependency-deprecation warnings in `56.40s`. All `27` ThreeDINO Python
files compiled, all three 3DINO PowerShell launchers parsed, and all five
top-level JSON profiles/configurations parsed. No production model,
patient-data, cache, training, CV, refit, inference, or held-out command was run
for that validation.

Static parser/JSON/bytecode checks may be run independently. These validations
do not create a production cache, train a classifier, perform CV/refit, or read
the fixed held-out test for inference. `-DryRun` and mocked provisioning tests
also do not clone source or download the checkpoint.

## Official sources

- [3DINO paper](https://www.nature.com/articles/s41746-025-02035-w)
- [Official 3DINO source](https://github.com/AICONSlab/3DINO)
- [Official gated 3DINO-ViT weights](https://huggingface.co/AICONSlab/3DINO-ViT)
