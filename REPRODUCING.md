# Reproducing and using the experiment

## View the study

Open `Notebooks/Publication_Results/Publication_Results_4.ipynb`, or its saved
`output_4/Publication_Results_4.html` edition. No data, model environment or
checkpoint loading is needed to view saved outputs. Version 4 is the only
publication notebook included. Open `Case_Study/FlexiCT_20_Case_Study.pdf` for
all 20 interpretations and paired CT/Grad-CAM figures.

## Use a trained checkpoint

Download the selected binaries using [CHECKPOINTS.md](CHECKPOINTS.md).
The selected fold and development-refit files are restored under
`Results/<model>/<method>/CV/<CV_id>/<size>_percent/`. Read its adjacent metadata
before use. `provenance/checkpoints.json` lists all 390 checkpoint files, their
new distribution hashes, original-source hashes and model-tensor fingerprints.

The checkpoints are inference-only exports: optimizer/scheduler/scaler/RNG
states, internal training context and local paths were removed. Every retained
model tensor was checked against the original and after serialization. These
files are not exact training-resume snapshots. Do not compare their new file
hash against an original historical completion seal; source and distributed
identities are deliberately recorded separately.

Use the public runner for testing your own source NIfTI images. Specify the
model family and selected checkpoint(s); a five-fold ensemble averages fold
probabilities. Partial/head-only checkpoints additionally need their original
pretrained backbone and pinned upstream source, supplied locally. Those
external pretrained assets are not bundled. CNN inference uses the fold-fitted
preprocessing plan saved with each checkpoint; it never fits on test images.

Testing performs source preprocessing and inference without a persistent test
cache. Predictions, metrics when labels exist, and provenance are normal
outputs. Training cache options are separate from this inference contract.

## Recompute the historical study

`protocol/study.json` and `configs/study/` retain the study and resolved parameter
choices. Historical per-model PowerShell launchers and model adapters remain
available. Recomputing requires the authorized original data and exact private
split manifests, plus the external model assets and model-specific environments.
The original split hashes are recorded; assignments and clinical source data
are not distributed.

New generic wrapper training uses an explicitly separate configurable recipe;
it is not a claim to reproduce every historical optimizer, augmentation,
stopping rule or parameter-search schedule. Historical percentages and fixed
five-class definitions are retained here for study documentation only; the
reusable public data interface has no percentage experiment ladder.

Paired statistical tests, patient bootstrap intervals and attribution replay
also require the private patient-level source predictions/review inputs. Public
aggregate tables cannot recreate those analyses. The saved notebook can be
viewed completely, but running all its cells without private inputs is not a
supported reproduction route.

## Repository branches and public checkpoint hosting

This study is the `experiment` branch of `https://github.com/IDKIS-Lab-JWK/systematic-methodology`. The reusable framework
is on `main`. The selected checkpoint binaries are hosted in the public Google
Drive folder linked in [CHECKPOINTS.md](CHECKPOINTS.md); no LFS setup is needed.
Cloning retrieves source, metadata and publication outputs. Download only the
checkpoint selection needed for your experiment, verify its SHA-256, and use the
expected Results path. A source-code clone does not contain clinical data or
silently download all model weights.
