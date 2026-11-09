# DenseNet adapter

Use `python run.py models` from the repository root for supported training
methods and `python run.py train --help` or `python run.py test --help` for the
public interface. New datasets use arbitrary ordered class labels and patient
manifests; the public runner does not use the original study's development-size
protocol. See [usage](../../../docs/USAGE.md),
[data](../../../docs/DATA.md) and [installation](../../../docs/INSTALLATION.md).

This directory retains the native architecture, preprocessing, parameter
profiles, upstream identities and numerical tests. Source/provenance fields
inside profiles distinguish publication evidence from project adaptations.
The generic framework training recipe is separately identified in its saved
configuration; choosing an architecture does not imply reproducing a published
training schedule.

Historical model entry points keep study-specific internal contracts for
training compatibility. Cached legacy test dispatch is disabled; testing uses
the public runner's direct source-image inference and explicit checkpoints.
Pretrained weights and upstream sources are supplied externally as documented
by setup/provisioning modules and the model's requirements file. Never replace
a required frozen backbone with an unverified checkpoint.
