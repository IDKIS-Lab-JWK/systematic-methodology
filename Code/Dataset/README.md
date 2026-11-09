# Dataset utilities

The public `python run.py split` command generates patient-group-safe splits
for new labeled NIfTI manifests. It supports a training/validation split or
cross-validation and an independent held-out test partition. Supplied splits
are validated by the same contract. Read `docs/DATA.md` from the repository root.

`prepare_dataset.py` is an optional image conversion utility. Its original
merge-existing helper assumes the historical folder layout; new users should
supply source locations explicitly. No clinical data is bundled. The fixed
cohort-count split generator belongs only to the experimental repository.
