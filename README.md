# Systematic Methodology

A configurable framework for training and testing abdominal CT classifiers
across seven model families. Bring your own development data, select the model
and tuning scope, or test a specific compatible checkpoint on source images.

## Start here

1. Read [installation and model environments](docs/INSTALLATION.md).
2. Use the [data contract](docs/DATA.md) for new data or supplied patient splits.
3. Follow the [runner guide](docs/USAGE.md) for training and direct inference.

```sh
python run.py --help
```

The public runner exposes model, tuning method, parameters, data, checkpoint,
device, precision, seed and output options. It validates supported combinations
before execution and saves the resolved configuration. No fixed cohort size,
class-count distribution or development-percentage ladder is imposed on new data.

## Models and training

The adapters cover Merlin, Pillar0, 3DINO, FlexiCT, COLIPRI, MedicalNet and
DenseNet. Frozen-encoder training, partial adaptation, full-network fine-tuning
and training from scratch are offered where the architecture supports them.
The runner's capability listing is authoritative; unsupported combinations fail
early. Model environments remain separate because their numerical dependencies
and external source/checkpoint requirements differ.

New-data training uses a configurable framework recipe. It does not silently
claim to reproduce the performed study's publication-derived schedules,
augmentations or model-selection procedures. Model profiles and adapter source
retain the evidence and architectural provenance for those methods.

## Testing a checkpoint

Specify the model family and checkpoint, together with required local upstream
or pretrained-backbone assets. Ordered class labels, architecture and
preprocessing come from validated metadata. Head-only or partial weights require
their exact original backbone. Fold-fitted CNN preprocessing is loaded from the
checkpoint metadata and is never estimated from test inputs.

Inference reads source images, preprocesses and predicts in memory, and writes
predictions, optional labeled metrics and provenance. It does not create or
require a persistent test cache. Multiple compatible checkpoints can be used
for a probability ensemble.

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | Public command-line entry point |
| `Code/Models/` | Native model components and shared numerical code |
| `configs/` | Synthetic/example user configurations |
| `docs/` | Installation, data and usage documentation |
| `Code/framework/tests/`, `Code/inference/tests/` | Focused public-runner and inference tests |
| `scripts/` | Environment provisioning helper |

Data, pretrained assets, trained models, caches and runtime outputs are local
inputs/products and are not bundled here. The [experiment branch](https://github.com/IDKIS-Lab-JWK/systematic-methodology/tree/experiment) contains the performed study,
checkpoint catalogue and public Google Drive download instructions.

See [contribution guidance](CONTRIBUTING.md) and
[third-party notices and licensing status](THIRD_PARTY_NOTICES.md).

## Preview without clinical data

```sh
python run.py train --model DenseNet --method scratch --dataset examples/development.csv --splits examples/splits.json --preset smoke --dry-run --output runs/demo
```

This validates the fictional example configuration and patient partitions; it
does not read images or run training. Replace the example manifest for a real run.

## Verification

The core regression suite passed 76 tests, with two optional environment-specific
tests skipped in the core environment. Model-specific preprocessing checks were
also exercised separately. Exported checkpoints from all seven families produced
predictions on a synthetic NIfTI through the public runner. A native DenseNet121
training/checkpoint/public-inference round trip and MedicalNet custom-class and
training-scope checks passed. This is bounded software validation, not a new
clinical performance evaluation or a full retraining of every foundation model.

## Branches and cloning

| Branch | Purpose |
|---|---|
| `main` | Reusable framework and new-dataset experiments |
| [`experiment`](https://github.com/IDKIS-Lab-JWK/systematic-methodology/tree/experiment) | Performed study, latest notebook, 20-case report and checkpoint catalogue |

To download only the reusable branch:

```sh
git clone --single-branch --branch main https://github.com/IDKIS-Lab-JWK/systematic-methodology.git
cd systematic-methodology
```

Study checkpoints are hosted on [public Google Drive](https://drive.google.com/drive/folders/1Rtn1sMIduWCzlX5z-3r1SI2v6B7FfKIT), not Git LFS.
Use the experiment branch's [download guide](https://github.com/IDKIS-Lab-JWK/systematic-methodology/blob/experiment/CHECKPOINTS.md)
to restore a selected checkpoint and its adjacent metadata to the expected path.

## Licence and citation

Original code: [MIT](LICENSE). Model/checkpoint and clinical-material terms remain
separate in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Software citation
metadata is provided in [CITATION.cff](CITATION.cff); upstream references remain
in each model's profiles and documentation.
