# Reusable commands

Run from this repository root, using the Python executable in the selected model environment. `python run.py --help` and `python run.py models` do not import PyTorch. Install the per-model environment first; the model assets are separate downloads governed by their upstream access and licence conditions. The wrapper never imports code from the original private workspace.

## Train on new development data

```
python run.py train --model DenseNet --method scratch --dataset my-data/development.csv --splits my-data/splits.json --config configs/densenet.json --device cuda --precision auto --batch-size 1 --output runs/densenet
```

The dataset defines arbitrary string labels. Set `classes` in the JSON configuration to declare their ordered probability columns, or omit it to use the sorted development labels. Every training fold must contain every development class. The independent test partition never contributes to class discovery, preprocessing fitting, checkpoint selection, or training. See [DATA.md](DATA.md).

`--splits` accepts patient partitions you already prepared. Alternatively generate a patient-safe holdout or cross-validation layout:

```
python run.py split --dataset my-data/development.csv --cv-folds 5 --test-fraction 0.2 --seed 42 --output my-data/splits.json
python run.py train --model DenseNet --method scratch --dataset my-data/development.csv --cv-folds 5 --test-fraction 0.2 --seed 42 --preset default --output runs/cv
```

The generated partitions group all scans of the same patient and approximately stratify by the patient's complete label signature. Rare strata can make the requested partition impossible; validation fails rather than dropping a class or permitting leakage. Inspect the resulting counts before a large run. No fixed study development-size ladder is used.

`--preset default` supplies a generic 20-epoch recipe. `--preset smoke` uses one epoch and a 64-cubed CNN canvas. JSON `parameters` overrides the preset; repeated `--set KEY=JSON` overrides individual parameters last. For example `--set epochs=3 --set learning_rate=0.0003`. These are new generic AdamW/cross-entropy recipes, not publication hyperparameter searches. Frozen FlexiCT/COLIPRI use a trainable PyTorch linear head; COLIPRI uses the official pooled projected representation, without the study's pooling search. The generic path does not apply stochastic augmentation, scheduler search, early stopping, or full-development refitting. It saves the epoch with minimum validation cross entropy.

Add `--dry-run` to validate manifests, patient partitions, model/method support, and configuration without importing model libraries or reading image voxels. It does not certify upstream asset availability or GPU memory. The example CSV references fictional images and is only usable with `--dry-run` until replaced with real local image paths.

## Supported new-training combinations

| Model | frozen | partial | full | scratch |
| --- | --- | --- | --- | --- |
| DenseNet | — | — | — | MONAI 3D DenseNet121 |
| MedicalNet | linear head | layer4 + head | complete encoder + head | — |
| Merlin | linear head | ConvLoRA | image encoder + head | — |
| ThreeDINO | linear head | attention LoRA | experimental opt-in | — |
| FlexiCT | projected embedding linear head | final block / final quarter | — | — |
| COLIPRI | pooled projected embedding linear head | final block / final stage | — | — |
| Pillar0 | frozen image encoder + head | Atlas final block / final stage | vision tower + head | — |

A dash means that combination is not exposed by this reusable adapter, not that the architecture could never support it. Use `parameters.partial_scope` for the listed block/stage variants. ThreeDINO full requires `parameters.allow_experimental_full_finetune=true`. ThreeDINO partial requires `--batch-size 2` or larger, `weight_decay=0`, and enough training samples for a full batch; its BatchNorm head drops the final incomplete training batch. CNNs currently require `--batch-size 1` because physical-space coverage can produce variable volume shapes; their canvas dimensions must be multiples of 32 and at least 64.

Configuration examples live in `configs/`. Replace illustrative asset paths. Relative asset paths resolve against the configuration file. MedicalNet, FlexiCT, COLIPRI and ThreeDINO need their base checkpoint; the latter three also need the supported upstream source checkout. ThreeDINO additionally needs its upstream revision. Pillar0 needs the pinned RAVE source (`assets.rave_root`) and the official model in the local Hugging Face cache by default. Merlin uses its installed official package/model assets. Upstream validation and model loading remain fail-closed. Optional `assets.checkpoint_sha256` pins the base file; actual training records its observed hash.

`--device` accepts `auto`, `cpu`, or a PyTorch device such as `cuda:0`. Training `--precision auto` selects CUDA bf16 when supported and fp32 otherwise; fp16/bf16 require compatible CUDA. `--seed` controls splits, initialization, and batch order. Hardware kernels may still differ across environments.

Actual runs save resolved configuration, the exact split manifest, source-image SHA-256 fingerprints for development samples, environment provenance, trainable parameter shapes, epoch histories, validation-selected `fold_N/best.pt`, and `<checkpoint>.metadata.json`. The CNN preprocessing plan is fitted separately on each fold's training images and embedded in its checkpoint. Raw patient IDs are kept in the local run manifests; portable new checkpoints retain salted patient fingerprints for overlap rejection. Treat run directories as private analysis outputs. Image tensors and feature caches are never persisted by this generic path. Training therefore repeats preprocessing each epoch and may be slower than the historical experiment workflows.

## Direct independent testing

```
python run.py test --model DenseNet --checkpoint runs/densenet/fold_1/best.pt --dataset my-data/independent-test.csv --device cpu --output runs/independent-test
```

Select exactly one checkpoint, or repeat `--checkpoint` five times for the supported five-fold probability ensemble. Arbitrary CV fold counts can still be tested one checkpoint at a time. Supply model identity explicitly. The loader verifies sidecar hashes and class order, rejects incompatible model families or ensemble methods, and loads tensor state strictly. New framework checkpoints additionally reject known development patient fingerprints in the independent test manifest.

Testing streams one image at a time directly from NIfTI; `--batch-size` must be 1. No persistent test volume or feature cache is created. Omit `--precision` to use the saved checkpoint convention; supply `--precision fp32`, `fp16`, `bf16`, or `auto` explicitly to override it where the model adapter supports that precision. Use `--config` with an `assets` block to relocate base checkpoints/source directories. The recorded base checkpoint identity remains enforced. Test configuration may also be a flat asset map for compatibility with the inference API. Model-specific legacy assets may use `checkpoint_path`; the new format accepts that spelling as an alias of `checkpoint`.

Labels are optional for independent testing. The output contains `predictions.csv`, resolved settings, and provenance; fully labelled test data also produces `metrics.json`. A sidecar can be explicitly selected with `--metadata` once per checkpoint. Legacy sklearn/joblib artifacts require the explicit `--allow-trusted-pickle` option and a matching SHA-256 sidecar. Lightweight test `--dry-run` requires sidecars and checks their identity and hashes, but does not deserialize model tensors.

## Verification scope

Automated tests exercise patient leakage prevention, custom class handling, configuration failures, actual optimizer updates, checkpoint identity rejection, and a real CPU DenseNet121 train/save/public-test round trip on synthetic NIfTI volumes. They do not constitute full training validation of every large pretrained model, mixed-precision GPU behavior, or model performance on clinical data. Each model's existing tests and setup checks remain relevant to its separate environment.
