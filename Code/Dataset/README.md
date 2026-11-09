# Dataset Code

Dataset-related executable code lives in this directory:

- `prepare_dataset.py` copies or converts imaging data into
  `Systematic_Methodology/Dataset/`.
- `generate_split_manifests.py` reads the class-wise dataset and writes JSON
  manifests into `Systematic_Methodology/Dataset/Splits/`.

From the `Systematic_Methodology` directory:

```powershell
python Code\Dataset\prepare_dataset.py merge-existing
python Code\Dataset\generate_split_manifests.py
```

Both scripts derive their default locations from the repository structure, so
no path arguments are required for the standard layout.

