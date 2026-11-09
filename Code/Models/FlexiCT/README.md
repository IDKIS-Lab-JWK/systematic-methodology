> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# FlexiCT systematic workflows

This package implements the native FlexiCT-3D-VLM projected probe, final-block
and final-quarter fine-tuning, frozen single-head attention readout, and frozen image-text
zero-shot evaluation for the systematic five-class CT methodology. The existing
linear probe freezes the complete vision encoder and released VLM vision
projection, caches one 1024-dimensional feature per patient, and fits only a
training-partition `StandardScaler` plus multinomial logistic regression.

No production checkpoint, scan, cache, CV model, or held-out prediction is
created by the test suite.

## Scientific basis and task adaptation

The implementation is pinned to the official FlexiCT repository revision
`fd74ac06a31813fe02805674e96f811e9b2a5c43` and the
[FlexiCT paper](https://arxiv.org/html/2605.21906). The native frozen-feature
recipe is reproduced from Supplement S0.17 and the pinned
[`ours_C4KC_linear.py`](https://github.com/ricklisz/FlexiCT/blob/fd74ac06a31813fe02805674e96f811e9b2a5c43/downstream/retrieval/ours_C4KC_linear.py):

1. run the frozen FlexiCT-3D-VLM vision backbone in evaluation mode;
2. concatenate normalized CLS and the mean of normalized patch tokens
   (`864 + 864 = 1728` dimensions);
3. apply the released, frozen bias-free VLM vision projection (`1728 -> 1024`);
4. L2-normalize the projected feature;
5. fit `StandardScaler` on the current training partition only;
6. fit L2 multinomial logistic regression with LBFGS,
   `class_weight="balanced"`, `max_iter=20000`, and `random_state=0`;
7. search the complete ordered publication/code-backed grid
   `C = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`.

The scaler is a feature normalization step and the logistic classifier is
linear. The 1728-to-1024 projection is a released pretrained VLM component; it
is never fitted here. Results are labelled `LinearProbe` in the project, while
the provenance records set `strict_raw_backbone_probe=false` / `strict_linear=false`
and make this frozen pretrained projection explicit.

The original phenotype experiment uses lesion-centred crops and repeated
stratified CV. This project instead uses mutually exclusive whole-volume
abdominal classes, the fixed patient folds, and the common robust
macro-F1/balanced-accuracy selection rule. These are declared task adaptations;
the result is not described as an exact reproduction of the original endpoint.

The implemented whole-volume preprocessing is based on the official CT-RATE
VLM evaluation path: explicit LPS orientation, 2 mm isotropic linear
resampling, HU clipping to `[-1000, 1000]`, whole-volume z-scoring, symmetric
minimum-value padding, and a centre crop to `1 x 160 x 160 x 160`. The paper's
1.5 mm pretraining spacing is provenance context, not the evaluation recipe
used here.

`class_weight="balanced"` is the native loss-side imbalance correction. There
is no weighted sampler because sklearn logistic regression consumes the cached
feature matrix directly.

## Environment and assets

Use a separate Python 3.11 environment. The pinned packages are in
`requirements.txt`, including SciPy `1.16.1`, Transformers `4.56.1`, and
Hugging Face Hub `0.34.4`; setup validates those exact runtime pins. The linear
probe, caches, and supervised adapters still use the vision-only loader and do
not import or materialize Qwen. Transformers is installed solely so the
separate zero-shot workflow can construct the released text architecture.

```powershell
py -3.11 -m venv .venv_flexict
.\.venv_flexict\Scripts\python.exe -m pip install -r Code\Models\FlexiCT\requirements.txt
```

The source checkout is cloned automatically at its pinned commit when cache
preparation is first run, unless `--local_files_only` is supplied. The released
`FlexiCT-3D-VLM` checkpoint must be obtained through the
[official checkpoint instructions](https://github.com/ricklisz/FlexiCT#model-checkpoints).
Supply it explicitly or set `FLEXICT_CHECKPOINT` / `FLEXICT_VLM_CHECKPOINT`:

```powershell
$env:FLEXICT_VLM_CHECKPOINT = 'D:\models\FlexiCT-3D-VLM.pth'
.\.venv_flexict\Scripts\python.exe Code\Models\FlexiCT\check_setup.py --load_model
```

The loader first uses PyTorch weights-only deserialization, requires an exact
`vision_model.*` state-dict match, requires the released
`vlm_vision_projection.weight` with shape `1024 x 1728`, freezes every
parameter, and sets evaluation mode. It does not instantiate `Flexi_CT_VLM` or
its text model. The audited workspace checkpoint now passes weights-only
state-dictionary loading and has the recorded SHA-256 above. If and only if a
verified official file fails safe loading, legacy pickle loading can
be enabled with both `--allow_trusted_legacy_checkpoint` and an independently
verified `--checkpoint_sha256 <digest>`. The fallback rehashes the file before
deserialization; the opt-in flag without an expected digest is rejected.

On CUDA, `--feature_precision auto` chooses BF16 when the installed PyTorch/GPU
reports BF16 support (including an RTX 4090 with a compatible CUDA build) and
otherwise falls back to FP32. It never silently substitutes FP16. Explicit
`bf16`, `fp16`, and `fp32` remain available, and each resolved precision has a
different cache contract. CPU extraction is FP32 only.

## Direct resumable feature cache

The cache is deliberately direct: NIfTI -> preprocessed in-memory volume ->
final 1024-D embedding. It does not retain a roughly 22 GB intermediate volume
cache. Extraction handles one `160^3` volume at a time, so there is no
classifier batch-size setting and the minimum GPU extraction batch is already
one. `--num_workers` is retained for launcher-shape compatibility but must be
`0`; nonzero values are rejected rather than silently ignored.

The cache inventory is always formed once from the union of development and
fixed held-out patients in `Dataset/Splits/100_percent.json`, then reused by all
20/40/60/80/100% experiments. Feature construction is label-independent.
Held-out labels and features are not read by grid/CV selection; held-out cached
features become accessible only in the test stage. Cache `--percentage` is
therefore constrained to `100`; experiment percentages are accepted only by
the probe launcher.

```powershell
# Side-effect-free inventory/asset plan
.\.venv_flexict\Scripts\python.exe Code\Models\FlexiCT\prepare_cache.py --action plan --dry_run

# Resumable direct extraction
.\.venv_flexict\Scripts\python.exe Code\Models\FlexiCT\prepare_cache.py --action prepare

# Exhaustive validation: completion tree, every artifact, and fresh source SHA-256
.\.venv_flexict\Scripts\python.exe Code\Models\FlexiCT\prepare_cache.py --action validate
```

The scientific default is `--source_fingerprint sha256`. First preparation
hashes each source. Normal training-stage reuse may trust the stored SHA only
when absolute path, byte size, and nanosecond mtime still match; explicit
`--action validate` rehashes every source. `--source_fingerprint stat` is an
explicitly weaker diagnostic option and receives a separate cache identity.
Non-dry-run `LinearProbe` execution rejects it; reportable training requires
`sha256`.

Each patient artifact is written by atomic replacement. The final tree seal
binds the exact 100% artifact set and the cache manifest. The experiment model
identity additionally binds the manifest SHA-256 and canonical entries
fingerprint. Checkpoint SHA-256, pinned source commit, preprocessing order,
input/patch geometry, representation, precision, source-fingerprint mode, split
manifest, checkpoint-deserialization policy, patient/source inventory, and the
exact Python/PyTorch/CUDA/cuDNN/GPU/dependency extraction runtime all
participate in the schema-4 cache contract. SHA-256s of local `__init__.py`,
`cache.py`, `data.py`, and `model.py` are also bound, so a partial cache cannot
silently mix features produced before and after an adapter edit.
Partial caches resume only when that complete contract matches; stale,
truncated, undeclared, path-escaping, or hash-mismatched artifacts fail closed.
An artifact file without a matching atomic progress entry is an uncommitted
orphan and is regenerated rather than adopted.

## Training and outputs

Plan a complete publication-grid run without loading the model or cache:

```powershell
.\.venv_flexict\Scripts\python.exe Code\Models\FlexiCT\train_linear_probe.py `
  --stage all --percentage 100 --parameter_source publication --dry_run
```

Remove `--dry_run` after the cache passes validation. Publication mode rejects
a replacement grid and candidate truncation. The project `grid` source may use
`--grid_config` and a positive `--max_grid_candidates`.

The project exposes three held-out model sources after one selected CV run:

- `refit_full_development`: one scaler/classifier refitted on all development
  patients;
- `cv_ensemble`: average the class-aligned probabilities from the five
  fold-specific scaler/classifier pipelines;
- `best_cv_fold`: use the one fold pipeline selected only from development
  validation metrics.

The primary held-out report is predeclared as `refit_full_development`.
`cv_ensemble` and `best_cv_fold` are secondary sensitivity analyses. Running
all three is intentional; choosing the best one after observing held-out
metrics would be test-set selection and is not permitted.

Every classifier must expose classes exactly `[0, 1, 2, 3, 4]` in the canonical
order `Appendicitis, Cholezystitis, Normal, Perforation, Sigmoid`.
Convergence warnings are retained in each training-state document rather than
silently discarded.

The exact `cv_run_context.json` is written before fold execution. The outer CV
seal is not sufficient by itself: resume and test also validate each of the five
fold completion seals and the full-development refit seal. Held-out lookup is
read-only, so it cannot create or repair a registry or `cv_parameters.json`.
The registry uses stable source/profile/candidate provenance so one configuration
keeps one `CV_x` across percentages; every percentage context and completion
seal binds its full actual provenance. The model/CV identity hashes local
`backend.py` and shared `_linear_probe/common.py`, `metrics.py`, and
`workflow.py`, plus Python, NumPy, SciPy, scikit-learn, and joblib versions;
resumed folds fail closed after changes to those probe paths or runtime.
Validation also checks each grid/CV fold's context, patient partition,
parameters, feature identity, and reuse origin, and the refit's exact
development partition/context. Held-out reuse must match the selected artifact
hashes, model source, parameters/model identity, completion context, and ordered
patient/truth rows. If `--results_root` changes, omitted `--cache_root` and
`--provision_log` values derive from that new root.

Completed-grid validation recalculates every candidate summary from the five
sealed fold metrics and compares the full ranking. A partial grid or CV root
that already contains artifacts but lacks its root context fails closed.

## Deferred linear-probe variants

These are documented for later reproduction but are intentionally not mixed
into the native implementation:

1. **Common frozen-lexpert probe.** The external
   [frozen-feature benchmark](https://arxiv.org/html/2608.05960v1) and its
   [code](https://github.com/maulikchevli/frozen-lexpert/tree/4c9734c9d839e03523bc339c2d5934fd53e1942c)
   use the raw, unprojected 1728-D CLS-plus-mean-patch feature, per-label OvR L2
   logistic regression (`C=1`, balanced LBFGS, `max_iter=2000`), and no
   `StandardScaler`. Reproduce it in a separate cache/experiment namespace; it
   is not interchangeable with the native projected multinomial probe.
2. **Official FlexiCT-2D learned head (3D attention adaptation now added
   separately).** The pinned
   [`downstream/2d_classify`](https://github.com/ricklisz/FlexiCT/tree/fd74ac06a31813fe02805674e96f811e9b2a5c43/downstream/2d_classify)
   path caches 2D features and trains an `nn.Linear` head with SGD momentum
   `0.9`, cosine decay, batch 64, and an eight-value LR sweep. It is a
   slice/case aggregation protocol, not the projected logistic endpoint. A
   separately named `FrozenSingleHeadAttentionReadout` now adapts its published
   LUNA16 3D one-head aggregation family to 20 cached depth-wise tokens; it is
   not relabeled as the existing `LinearProbe`.
3. **RADIO hierarchical control.** The pinned
   [`ours_radio_linear.py`](https://github.com/ricklisz/FlexiCT/blob/fd74ac06a31813fe02805674e96f811e9b2a5c43/downstream/retrieval/ours_radio_linear.py)
   defaults to intermediate blocks `[3, 7, 11, 15]` and lesion ROI scales 32
   and 64. The paper's projected-feature description and this code default are
   not identical, and the local dataset has no corresponding lesion-coordinate
   protocol, so this remains a separately named deferred control.

## Validation

The focused tests use synthetic manifests, embeddings, checkpoints, and a tiny
fake upstream module. They do not extract any production volume:

```powershell
.\.venv_flexict\Scripts\python.exe -m pytest Code\Models\FlexiCT\tests
```

The released code is MIT licensed. The checkpoint is distributed under the
upstream non-commercial/no-derivatives terms documented by FlexiCT; verify the
current terms before redistribution or weight adaptation.

## Final-block FT, final-quarter FT, and frozen single-head attention

`FinalBlockFineTune` reloads the exact official checkpoint for every trainable
model, validates the 16-block hierarchy, and updates only `vision.blocks[15]`,
`vision.norm`, and a new five-class `LayerNorm(1728) -> Linear(5)` head. Logits
use normalized CLS concatenated with mean normalized patch token. The released
VLM projection is loaded and verified as part of checkpoint integrity but is
frozen and bypassed. The publication profile keeps the paper's AdamW
`(0.9,0.98)`, `5e-5`, weight decay `3e-5`, and power-1 polynomial anchors;
scope, five-class head, step ceiling, and validation stopping are clearly
marked adapted.

`FinalQuarterFineTune` additionally trains blocks 12--14, so its complete
encoder scope is `vision.blocks[12:16] + vision.norm`. It inherits the same
checkpoint loader, preprocessing, classifier representation, and optimizer
recipe. A final quarter is a local scope choice for this flat 16-block encoder,
not a native architecture stage. Its fixed publication/default profiles and
three-candidate manual grid write to `Results/FlexiCT/FinalQuarterFineTune/`.
The publication profile labels the scope as adapted and the inherited optimizer
anchors as backed. Effective batch remains 4; the new launcher uses microbatch 1.

```powershell
.\FlexiCT_Final_Quarter_FineTune_Run_Script.ps1 -DryRun
.\FlexiCT_Final_Quarter_FineTune_Run_Script.ps1 -Percentage 40 `
  -Checkpoint .\Results\FlexiCT\Models\3D_VLM_final_model.pth
```

All three methods use direct CV/refit/test for fixed `publication` and `default`
sources. `-ParameterSource grid` runs the manual search. Repeating a command
resumes matching checkpoints; changing microbatch/precision during a partial
run is still incompatible with exact optimizer-step resume.

`FrozenSingleHeadAttentionReadout` never updates or reloads the encoder during
head training. First, the dedicated resumable cache extracts final normalized
`20 x 20 x 20 x 864` patch tokens and averages each depth plane over height and
width, storing finite FP16 `[20,864]` tensors with per-shard SHA-256. The head
contains one learned query, `MultiheadAttention(num_heads=1, dropout=0)`, a
residual LayerNorm, and `Linear(864,5)`. Its publication profile uses frozen
encoder, SGD momentum `0.9`, cosine decay, 50 epochs, effective batch 64, and
the published batch-scaled LR `0.02 * 64 / 256 = 0.005`. The exact referenced
Curia implementation is gated, so the transparent local query-attention code
and depth-token adaptation are recorded as publication-adapted.

The token-cache contract binds the exact checkpoint, pinned source revision,
preprocessing and implementation hashes, Python/PyTorch/CUDA/cuDNN/device
runtime, and each source NIfTI's SHA-256. A saved source digest is reused during
resume only when its absolute path, size, and nanosecond mtime are unchanged;
every completed shard is then verified against its manifest hash before use.

```powershell
.\FlexiCT_Final_Block_FineTune_Run_Script.ps1 -DryRun
.\FlexiCT_Attention_Cache_Run_Script.ps1 -DryRun
.\FlexiCT_Attention_Cache_Run_Script.ps1 -Checkpoint D:\Models\FlexiCT-3D-VLM.pth
.\FlexiCT_Frozen_Attention_Readout_Run_Script.ps1 -DryRun
```

The methods write to `Results/FlexiCT/FinalBlockFineTune/`,
`Results/FlexiCT/FinalQuarterFineTune/`, and
`Results/FlexiCT/FrozenSingleHeadAttentionReadout/`; the token cache is
under `Results/FlexiCT/Cache/AttentionTokens/`. They cannot overwrite the
existing `LinearProbe` or feature cache. Real runs resume only from an exact
checkpoint/cache/profile/split/code/runtime/input identity.

The completed final-block and attention outputs were migrated on 2026-09-06
to fixed-publication provenance. Their original grids are recoverable outside
Results in `Archives/FixedPublication_20260906/`. A fingerprinted migration
manifest in each method records exact code compatibility and historical path
resolution; it does not relax checkpoint, runtime, or batch validation.

## Frozen image-text zero-shot (implemented, not run)

`ZeroShot` uses the pinned official `FlexiCT-3D-VLM` checkpoint SHA-256
`bb4e0a697380f3636723b76b175cd1d6ca2e72659a4c269be3b3e5d41c4b7403`.
It reconstructs the released 1024-dimensional VLM image projection and the
Qwen3 text architecture, then strictly loads the checkpoint's complete
`text_model.backbone.*` state and scalar `logit_scale`. Every parameter remains
frozen. The Qwen base weights are not downloaded: only `config.json` and the
tokenizer files are resolved from the immutable
`Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` because the FlexiCT checkpoint
itself supplies the aligned text weights.

The publication/code-backed method uses L2-normalized projected image and text
embeddings, last-non-padding-token text pooling, the checkpoint-learned logit
multiplier, and a positive-versus-negative softmax for each concept. The
released evaluator targets 18 independent CT-RATE abnormalities. Therefore the
local five abdominal prompt pairs—including the explicit interpretation of
`Normal`, `Perforation`, and `Sigmoid`—and normalization of five positive
probabilities into one mutually exclusive distribution are marked
publication-adapted. There is no fitting, CV, calibration, or prompt selection
on held-out labels.

The run validates that held-out membership is identical across all available
percentage manifests, binds the ordered held-out files, source commit,
checkpoint, prompt profile, text snapshot inventory, implementation, and full
runtime, and saves atomic patient-level progress after every scan. An exact
resume continues at the next patient. Completed output is seal-validated and
reused; unknown partial files or any context change fail closed.

```powershell
# Side-effect-free plan; no model construction, download, or inference
.\FlexiCT_Zero_Shot_Run_Script.ps1 -DryRun

# First run; downloads only pinned Qwen configuration/tokenizer files if absent
.\FlexiCT_Zero_Shot_Run_Script.ps1 `
  -Checkpoint Results\FlexiCT\Models\3D_VLM_final_model.pth

# Fully offline reuse after the pinned snapshot is cached
.\FlexiCT_Zero_Shot_Run_Script.ps1 -LocalFilesOnly `
  -Checkpoint Results\FlexiCT\Models\3D_VLM_final_model.pth
```

Use `-HFCacheRoot D:\HFCache` to choose the Hugging Face cache location. Hub
authentication, whenever required by the local environment, is read through
standard Hugging Face configuration or `HF_TOKEN`; tokens are never accepted as
launcher arguments or written to provenance. Outputs are isolated under
`Results/FlexiCT/ZeroShot/Test/flexict_upstream_pairwise_five_class_v1/` and
cannot overwrite linear-probe, cache, or supervised-adaptation artifacts.
