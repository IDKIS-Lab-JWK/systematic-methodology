# Data contract

The wrapper accepts a UTF-8 CSV or JSON sample manifest. CSV columns are:

| Field | Meaning |
| --- | --- |
| sample_id | Unique scan/sample identifier |
| patient_id | Stable patient grouping identifier shared by every scan of that patient |
| nifti_path | Path to a scalar 3D `.nii` or `.nii.gz` image |
| label | Arbitrary class string; required for development, optional for independent testing |

JSON uses a list of these objects, or `{"samples": [...]}`. Relative image paths resolve from the manifest directory, or from `--dataset-root` when provided. Duplicate sample IDs and image paths are rejected. Patient IDs may repeat across scans. The framework cannot recognize the same person under inconsistent patient IDs; supply one stable identity across visits and datasets.

Use de-identified local inputs. `examples/development.csv` and `examples/splits.json` contain synthetic names only; no images or real patient records are included.

## Supplied patient partitions

For one holdout split:

```json
{
  "train": ["synthetic-patient-a1", "synthetic-patient-b1"],
  "validation": ["synthetic-patient-a2", "synthetic-patient-b2"],
  "test": ["synthetic-patient-a3", "synthetic-patient-b3"]
}
```

All values are patient IDs, never sample IDs. Every patient in the accompanying development manifest must appear in exactly one partition. The test list may be empty when the independent test dataset is supplied later as a separate manifest. Training never opens image voxels belonging to the declared test partition, although manifest existence checks verify all declared paths.

For cross-validation, use `{"folds": [{"train": [...], "validation": [...]}, ...], "test": [...]}`. Every fold must cover all development patients without overlap; each development patient must appear in validation exactly once across folds. Test patients cannot appear in any fold. Each training fold must include all configured development classes, with at least two classes. Multiple scans of one patient remain together even when their sample labels differ.

Without `--splits`, the wrapper shuffles patient groups deterministically with `--seed`. `--validation-fraction` controls a single holdout split; `--cv-folds N` requests N-fold CV instead. `--test-fraction` reserves independent patients before either split. Stratification uses complete patient label signatures and rounds per stratum, so overall fractions may differ from the requested fractions. Insufficient examples are a validation error. Save the generated manifest using `run.py split` or retain the training run's `splits.json`.

## Image preparation

Provide CT values in the units expected by the selected model, with valid orientation, voxel spacing and affine metadata. CNN baseline NIfTI files must explicitly declare millimetres in their header. CNN geometry statistics are learned from each training fold only; validation and test volumes retain that spacing and expand their canvas when needed to preserve acquired coverage. Images larger than the trained canvas can therefore need more memory.

The pretrained adapters retain their model-specific numerical preprocessing: official Merlin transforms; pinned RAVE processing and RATE geometry for Pillar0; ThreeDINO deterministic prefix and center crop; official FlexiCT preprocessing; and the official COLIPRI processor. These transforms execute directly in memory. No generic resizing is substituted across model families.

The new-development trainer uses single-label cross entropy. Multilabel targets, segmentation masks, DICOM-series selection, missing-label development samples and regression are not supported by this interface. The same label vocabulary and class order must be used when comparing compatible checkpoints. Independent test labels outside the saved vocabulary are rejected for metric calculation.
