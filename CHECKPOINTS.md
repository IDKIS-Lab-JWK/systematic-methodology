# Download the trained study checkpoints

All **390** selected inference-only checkpoints are hosted in the project's
[public Google Drive folder](https://drive.google.com/drive/folders/1Rtn1sMIduWCzlX5z-3r1SI2v6B7FfKIT), owned by sarmadahmed2015@gmail.com.
No Google account is required for public downloads. The binaries are not in
Git or Git LFS. Metadata sidecars remain beside their expected `Results/...`
locations in the `experiment` branch.

## Choose a small, explicit selection

```sh
python -m pip install -r scripts/requirements-download.txt
python scripts/download_checkpoints.py --list
python scripts/download_checkpoints.py --model DenseNet --method FromScratch --percentage 100 --partition fold_01 --dry-run
python scripts/download_checkpoints.py --model DenseNet --method FromScratch --percentage 100 --partition fold_01
```

A filter selects downloads unless `--list` or `--dry-run` is supplied. No arguments
only list the catalogue. `--all` downloads the full **73.7 GB** explicitly. Here,
`--percentage` selects an existing historical study checkpoint; it is not a
percentage-based training option in the reusable framework.

For example, the command above restores:

```text
Results/DenseNet/FromScratch/CV/CV_1/100_percent/fold_01/model.pt
Results/DenseNet/FromScratch/CV/CV_1/100_percent/fold_01/model.pt.metadata.json
```

The `.metadata.json` file is already in Git. Each catalogue entry contains its
public Drive file link, exact repository-relative destination, byte size and
SHA-256. The downloader checks the bytes before publishing the completed file,
resumes interrupted transfers, skips only verified existing files and refuses
to replace a differing final file. No model imports or clinical data are needed.
Temporary `*.download/` transfer state is ignored by Git. Atomic file publication
requires a normal local filesystem with hard links, such as NTFS or ext4.

```sh
python scripts/download_checkpoints.py --model DenseNet --method FromScratch --percentage 100 --partition fold_01 --verify-only
```

For a CV ensemble, select `fold_01` through `fold_05` for one matching model,
method and size, then pass those five files with repeated `--checkpoint` flags
to `run.py test`. The `refit_full_development` partition is a separate model.
Use `--path` for one exact catalogue path or `--output-root` to choose the
parent directory under which the same `Results/...` tree is reconstructed.

## Test your source images

```sh
python run.py test --model DenseNet --checkpoint Results/DenseNet/FromScratch/CV/CV_1/100_percent/fold_01/model.pt --dataset my-test.csv --device cuda --output runs/independent-test
```

Read [the data contract](docs/DATA.md) for the NIfTI manifest. Labels are optional.
Testing performs preprocessing and inference in memory and creates no persistent
image/feature cache. Frozen heads or partial adaptation checkpoints additionally
require the original compatible pretrained backbone and upstream source.
Legacy FlexiCT sklearn probe artifacts require `--allow-trusted-pickle` plus
their verified sidecar; tensor checkpoints use safe weights-only loading.

## Manual downloads and availability

You can also open an entry's `google_drive_url` in
[provenance/checkpoints.json](provenance/checkpoints.json), download it and place
it at its recorded `path`. Flat filenames in Drive encode the original hierarchy;
the catalogue is authoritative for where each file belongs. Google may display
a virus-scan confirmation for large files or temporarily limit download traffic.
The helper handles the normal confirmation, but cannot bypass a quota or access
restriction. Keep the recorded checksums when mirroring files.

These public files are model exports, not optimizer/RNG resume snapshots.
Clinical source data, caches and private audit inputs are not distributed.
Checkpoint provenance and upstream terms remain separate from the project's
software MIT licence; see [third-party notices](THIRD_PARTY_NOTICES.md).
