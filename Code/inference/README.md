# Direct inference

The public entry point is `python run.py test` from the repository root. The
Python API is `inference.predict` after placing `Code` on `sys.path`:

```python
from inference import predict
result = predict(
    ['path/to/model.pt'],
    [{'sample_id': 'case-001', 'patient_id': 'patient-001',
      'nifti_path': 'path/to/volume.nii.gz'}],
    model_family='MedicalNet', device='cpu',
)
```

Select one checkpoint or exactly five distinct fold checkpoints. Five-fold
inference averages probabilities in the checkpoint's stored class order. All
members must agree on family, method and class order. Labels are optional and
are not passed to the model. The wrapper computes metrics when labels exist.

`PredictionResult` contains `probabilities`, `classes`, `sample_ids`, and
`provenance`. It creates no output files; the public wrapper writes predictions,
metrics and provenance. Images are read and transformed one at a time in memory.
There is no fallback to cached CT tensors, embeddings or token shards. Training
may continue to use its existing caching workflows.

## Metadata and model assets

The default sidecar is `<checkpoint>.metadata.json`. It contains
`model_family`, `method`, ordered `classes`, `checkpoint_sha256`, and optional
`preprocessing_plan`, `parameters`, `model_identity`, `runtime_assets`. The
fold-fitted CNN preprocessing plan is required and is never fitted on test data.
Sidecar family, method, classes and artifact digest must agree with the artifact.

Frozen or partial checkpoints require the original pretrained model assets.
Supply relocatable paths through sidecar `runtime_assets` or `model_overrides`:

| Family | Required assets |
| --- | --- |
| MedicalNet, DenseNet study checkpoints | No pretrained file; learned full model state and saved preprocessing plan |
| 3DINO | `upstream_root`, `checkpoint_path`, recorded `checkpoint_sha256`; upstream revision is pinned |
| FlexiCT | `upstream_root`, `checkpoint_path`, recorded `checkpoint_sha256` |
| COLIPRI | `upstream_root`, `checkpoint_path`; published backbone SHA-256 is enforced |
| Pillar0 | Installed/cached pinned Hugging Face model plus `rave_root` for the audited RAVE checkout |
| Merlin | `merlin-vlm==0.0.7` with its official model installed in the package's checkpoint directory |

`checkpoint` is accepted as an alias for `checkpoint_path`. Asset digests in
saved model identities take precedence over overrides: a conflicting override
fails. Merlin frozen probes verify the installed encoder against saved provenance.
Pillar defaults to local model files; explicitly set `local_files_only=false` to
permit the model loader to resolve an authorized model download. No test images
are uploaded by this inference implementation.

RAVE must be the audited revision
`20adeb873021c864e9410fae8898b0c8da309769`; the numerical processor files are
verified before use. Inference calls its CT numerical processor directly and
never calls the pipeline or any exporter. This retains int16 HU conversion,
native orientation, 1.25 mm resampling, in-plane minimum-value padding, final
-1 HU depth padding and the RATE-Evals depth flip. Merlin uses its official
`ImageTransforms` without constructing `CTPersistentDataset`. 3DINO uses the
original test transform with `cache_dir=None`. FlexiCT/COLIPRI use their original
source-image processors, and CNNs use the saved training-derived geometry.

Legacy sklearn `.joblib` files are pickle-based. Loading them requires explicit
`allow_trusted_pickle=True` and a matching sidecar SHA-256. PyTorch artifacts use
`weights_only=True`; the same explicit opt-in plus digest is required for an
otherwise incompatible trusted legacy artifact. Never opt in for an untrusted
file. Published inference-only PyTorch derivatives do not need this opt-in.

The original recorded extraction/evaluation precision is used when available.
CPU prediction uses FP32; CUDA uses the stored precision. `model_overrides` may
explicitly set `precision` to `fp32`, `fp16`, `bf16` or `auto`; this is recorded in
provenance and can change numerical output. GPU memory still has to fit the
selected model and one input volume. In-memory execution removes disk cache
requirements, not the model's hardware requirements.

## Historical scripts

Historical supervised `--stage test` and `--stage all` execution is disabled
before cache preparation. Use `run.py test` for held-out inference and explicit
`--stage cv`, `--stage grid` or `--stage search` for historical training. Their
read-only `--dry_run` remains available. Merlin/Pillar historical zero-shot
launchers are disabled because they require test caches. FlexiCT/COLIPRI
zero-shot scripts already read source images and save scores rather than image
features, so those scripts remain available. The general classifier wrapper does
not currently provide zero-shot prompt inference.

## Verification

Run `python -B -m pytest Code/inference/tests -q -p no:cacheprovider` in a model
environment with PyTorch and the relevant numerical dependencies. The tests cover
streamed five-fold averaging without file creation, class order and identity
validation, pretrained hash disagreement, strict learned-state restoration,
model-specific probe reconstruction, source preprocessing and a synthetic
MedicalNet forward. Set `RAVE_TEST_ROOT` to an external audited checkout to run
the exact raw RAVE/legacy model-input numerical equivalence test. These tests do
not constitute a rerun of all pretrained models on the clinical held-out cohort.
