# Supervised adaptation workflow

Fixed publication/default recipes run CV, a median-best-step full-development
refit, and held-out evaluation. Manual `parameter_source=grid` runs parameter
selection first. A publication search requires an explicit
`parameter_selection=publication_grid` and at least two candidates. A missing
declaration defaults to fixed parameters and rejects multiple candidates.

FlexiCT final-quarter adaptation adds blocks 12--15 plus final norm to the
trainable head; its classifier, checkpoint loading, preprocessing, and trainer
reuse the existing final-block implementation. Publication-backed/adapted
evidence stays descriptive; the executable source is still `publication`.

## Microbatch defaults

The shared Python parser's generic microbatch default remains 1; model-specific
entrypoints can override it. COLIPRI's Primus-M entrypoint resolves an omitted
microbatch from the selected profile before starting this workflow, and its
PowerShell launcher uses the same resolution. For both scopes, the packaged
publication/grid batch is 8 and the practical-default batch is 4. Explicit
microbatches must be positive, no larger than the effective batch, and divide
it evenly. Oversized values produce an error stating the actual limit.
Custom grids with different batch sizes require an explicit common microbatch.
The [COLIPRI README](../COLIPRI/README.md) documents the model-specific rules.

Resume requires the microbatch recorded in the checkpoint, even when a
launcher's current default has changed.

## Migration and resume

COLIPRI now defaults to temporary frozen encoder outputs for both Primus-M scopes. Prepare
their shared inputs with `COLIPRI_Cache_Run_Script.ps1 -Action prepare
-CacheTarget finetune`, or omit the target to prepare CTs and probe features.
Pass the same `-CacheRoot` to training. The adapter supplies `cache_path` and
`cache_sha256` so this trainer binds actual shards; model identity additionally
binds the permanent CT cache contract/manifest and the selected frozen boundary's
weights, code, extraction precision and runtime. Disposable tensor timestamps
are excluded from experiment identity so a later launch can regenerate them.
Training checkpoints additionally bind ordered frozen tensor-value digests;
regeneration with changed values cannot silently resume a checkpoint.
Single-scope evaluation and a combined `-AdaptationScope both` launch retain
the same model identity for that scope. The COLIPRI CLI owns cache leases until
all requested scopes/stages complete, removes temporary tensors after success,
and retains them on interruption. `-InputSource cache` uses prepared CTs directly.
`-InputSource nifti` retains the original
adapter for legacy resume. Switching input modes requires a new run; an active
legacy compatibility record can require a fresh result root. The shared trainer does not
waive any input/runtime/checkpoint checks.

COLIPRI's `PrefetchedAdaptationBackend` wraps deterministic loading with one
CPU microbatch of lookahead, including across optimizer-step boundaries and
during evaluation. It uses a separate sampler and verifies requested indices;
the sampler, RNG and optimizer states recorded by this trainer are unchanged.
The wrapper is model-specific: other adapters are not opted into prefetching.
`-NoPrefetch` disables it. Workers are joined before cache cleanup. The original
trainer, runtime, workflow, compatibility records and parameter profiles were
not modified by this optimisation.
The subsequent COLIPRI full-size BF16 verification fix only changes its frozen
forward/cache modules. Original-NIfTI resume hashes remain unchanged. Frozen
execution keeps the released tensor strides; verification includes CPU-shard
layout and logs numerical failures without relaxing its tolerance.

Pillar-0 Atlas fine-tuning now uses its own `Pillar0/prefetch.py` backend wrapper
and augmentation-aware frozen adapter. The shared numerical trainer, original
Atlas adapter and publication parameters are unchanged. One CPU microbatch is
prepared with a cloned augmentation generator; only a consumed batch advances
the checkpointed RNG. Cached inputs cover all ten existing orientations, with
the original two plane/turn draws and their probabilities preserved. Checkpoints
add ordered training/validation tensor-value identities in frozen mode. Default
microbatch remains 1 and publication effective batch remains 32. Original RAVE
mode retains its numerical identity. The old 100% final-block `CV_1` remains
intact and has an independent `Backup_CV_1` copy; the new default frozen identity
was verified to allocate `CV_2` using a temporary registry copy. No live registry
was changed by that check. Both boundary modes, RNG resume, logits, gradients,
optimizer updates, cache interruption/corruption and BF16/FP32 full-size forwards
passed the Pillar/shared suite (74 tests plus 6 subtests). Validation evidence is
in `Logs/Pillar0/VALIDATION_20260914_FROZEN_AUGMENTATIONS.json` from the workspace root.

The 2026-09-06 migration retained the existing CV IDs and all trained tensors,
fold seals, predictions, and metrics for the two fixed FlexiCT methods.
Only CV registry/context and test provenance metadata were rebound and sealed.
The original grid folders were moved intact to
`Archives/FixedPublication_20260906/Results/FlexiCT/` after validation.
Pre-migration CV/Test metadata and tensor snapshots are archived there too.
Tensor snapshots use hard links and must be treated as immutable; JSON
snapshots are independent copies.

Each affected result method has a fingerprinted
`FIXED_PUBLICATION_MIGRATION.json`. `compatibility.py` checks the exact
reviewed implementation, unchanged checkpoint/representation, profile,
percentage, seed, and parameter source before returning the original model
identity to the trainer. All original runtime, microbatch, optimizer, stream,
and RNG checkpoint checks remain enabled. COLIPRI's original partial final
block run initially had a compatibility record without any artifact migration.
On 2026-09-11 the user had backed up that run and chose a fresh 100% experiment.
The remaining method-level record was moved unchanged into `Backup_CV_1`, so
it no longer invokes legacy compatibility for the active frozen configuration.
Read-only registry lookup resolves that configuration to `CV_2` for 100%.
This retirement changes no shared or model training code, CV registration,
checkpoint, profile, or numerical implementation hash.

Historical paths inside immutable fold/checkpoint metadata can be resolved
using `resolve_historical_artifact(method_root, historical_path)`; canonical
CV files are preferred, followed by the archive for grid-only provenance.

The migration utility provides `prepare`, `apply`, and `verify` stages. It
requires complete matching CV/refit materializations before retiring a grid;
it refuses genuine searches, missing computation, changed artifacts, and
unreviewed model/trainer changes. It is idempotent against its saved plan.
It does not execute a pretrained model or train any parameter.

```powershell
.\.venv_flexict\Scripts\python.exe `
  Code\Models\_supervised_adaptation\migrate_fixed_publication.py verify
```

Rollback evidence is in `Archives/FixedPublication_20260906/README.md`.
